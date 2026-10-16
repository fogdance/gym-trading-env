"""Streaming three-way join and bounded reader over the original Replay.

SQLite stores only join/projection metadata. Observations always come from the
original native NPZ chunks; teacher facts never enter an observation builder.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import (digest, read_manifest, signed, write_json, trace_manifest,
    iter_replay_rows, iter_sidecar, replay_fields, update_array_hash, E_QUALITY_SCHEMA)
from .fg_s3_e_observation import validate_e_snapshot
from .fg_s3_e_quality_supervision import (
    Admission, POLICIES, member, inspect_source, project_label, PROTOCOL)

INDEX_SCHEMA = 'FG_S3_E_QUALITY_JOIN_INDEX_V1'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',',':'), allow_nan=False)


def _join_row(row, step, ordinal, trace, corpus):
    if (row['ordinal'] != ordinal or row['stepid'] != step['stepid'].tobytes().hex()
            or step['stepid'].shape != (20,) or step['stepid'].dtype != np.uint8
            or row['action'] != int(step['action'])
            or not np.isclose(row['official_reward'], float(step['reward']), rtol=0, atol=1e-7)
            or any(bool(step[k]) != row[k] for k in ('is_first','is_last','is_terminal'))
            or bool(step['market_advance']) != row['valid_bar']
            or row['legal_add'] != bool(step['action_mask'][1])):
        raise ValueError('SUPERVISION_PHYSICAL_ROW_JOIN_MISMATCH')
    if (not np.isin(step['action_mask'],[0.,1.]).all() or not step['action_mask'][0]
            or not 0 <= int(step['action']) < 2
            or not step['action_mask'][int(step['action'])]):
        raise ValueError('SUPERVISION_ACTION_MASK_MISMATCH')
    for name, value in step.items():
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError('SUPERVISION_NONFINITE_REPLAY: ' + name)
    label = row.get('teacher')
    # Original native corpus rows include labels, including invalid terminal
    # facts. Missing labels are not silently converted to a fake WAIT.
    if not label:
        raise ValueError('SUPERVISION_MISSING_TEACHER')
    from .fg_s3_quality_catalog_v2 import validate_corpus_label
    validate_corpus_label(label)
    facts, strict = label['L1'], label['L2']
    if (label['account_snapshot_hash'] != row['account_snapshot_hash']
            or facts['teacher_identity'] != trace['identity']['teacher_identity']
            or facts['state_proof']['state_id'] != label['state_id']
            or facts['state_proof']['source_hash'] != fingerprint(trace['identity']['source'])
            or facts['actual_legal_wait'] != bool(step['action_mask'][0])
            or facts['actual_legal_add'] != bool(step['action_mask'][1])
            or strict.get('protocol') != 'FG_S3_E_QUALITY_STRICT_SUFFIX_V2'
            or strict.get('teacher_identity') != corpus['teacher_identity']
            or strict.get('account_snapshot_hash') != row['account_snapshot_hash']
            or read_signed(strict) is None):
        raise ValueError('SUPERVISION_TEACHER_ACCOUNT_JOIN_MISMATCH')
    if facts.get('decision_cursor',row['cursor']) != row['cursor']:
        raise ValueError('SUPERVISION_POST_STEP_CURSOR_MISMATCH')
    env = SimpleNamespace(contract=trace['contract'], current_step=row['cursor'],
                          _raw=SimpleNamespace(index={row['cursor']:pd.Timestamp(row['time'])}))
    validate_e_snapshot(row['e_snapshot'], env, trace['data_identity'], arrays=step)


def read_signed(value):
    if value.get('content_hash') != fingerprint({k:v for k,v in value.items() if k!='content_hash'}):
        raise ValueError('SUPERVISION_TEACHER_PROOF_HASH_MISMATCH')
    return value


def _transition(previous, current):
    advanced, minute = (bool(current and current['advanced']), bool(current and current['valid_bar']))
    if previous['decision_executed'] != advanced or previous['decision_valid_minute'] != minute:
        raise ValueError('SUPERVISION_NEXT_EXECUTION_MISMATCH')
    if current is None:
        return
    if previous['is_terminal']:
        raise ValueError('SUPERVISION_TRUE_TERMINAL_CROSSED')
    if current['reset']:
        if (previous['status'] != 'CHUNK_END' or previous['is_terminal']
                or current['cursor'] != previous['cursor'] or current['advanced']
                or current['lots_after'] != previous['lots_after']):
            raise ValueError('SUPERVISION_ILLEGAL_ACCOUNT_RESET')
    elif (current['incoming_action'] != previous['action']
            or current['signal_time'] != previous['time']
            or current['lots_before'] != previous['lots_after']
            or current['cursor'] < previous['cursor']
            or current['advanced'] != (current['cursor'] > previous['cursor'])):
        raise ValueError('SUPERVISION_ACTION_FEEDBACK_SHIFT')


def _strata(label, projected):
    facts=label['L1']; tags=[]
    if projected['teacher_valid']:
        tags.append('STRICT_ADD' if projected['teacher_action'] else 'STRICT_WAIT')
    if projected['quality_outcome_valid']:
        kind=facts.get('exit_type')
        if kind=='TP5':tags.append('TP_FAST' if facts['held_market_minutes']<=5 else 'TP_SLOW')
        elif kind=='SL30':tags.append('SL30')
        else:tags.append('OTHER')
        tags.append('CROSS_DAY' if facts.get('crossed_trading_days',0)>0 else 'SAME_DAY')
    for window in (5,15):
        if facts.get(f'early10_{window}_valid') and facts.get(f'early10_{window}'):
            tags.append(f'EARLY10_{window}')
    return tags


def build_join_index(admission, path):
    if not isinstance(admission, Admission):
        raise ValueError('SUPERVISION_PREFLIGHT_OBJECT_REQUIRED; JSON claims are not admission')
    corpus = inspect_source(admission.source)
    root = Path(admission.source.corpus_root)
    path = Path(path).absolute()
    if path.is_relative_to(root) or path.exists():
        raise ValueError('SUPERVISION_INDEX_OUTPUT_MUST_BE_NEW_AND_OUTSIDE_CORPUS')
    # All output ancestors receive the same path validation as input members.
    from .fg_s3_e_quality_course_preflight import _local_path
    path = _local_path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.touch(exist_ok=False)
    db=sqlite3.connect(path)
    db.execute('CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    db.execute('''CREATE TABLE rows (policy TEXT, ordinal INTEGER, stepid BLOB PRIMARY KEY,
        segment INTEGER, cursor INTEGER, day TEXT, chunk TEXT, offset INTEGER,
        labels TEXT, audit TEXT, strata TEXT, first INTEGER, last INTEGER,
        terminal INTEGER, advanced INTEGER, UNIQUE(policy,ordinal))''')
    db.execute('CREATE INDEX by_policy ON rows(policy,segment,ordinal)')
    reports={}; join_hash=hashlib.sha256()
    try:
        for policy in POLICIES:
            original=member(root,f'policies/{policy}')
            student=member(root,f'student_replay/{policy}')
            sidecar=member(root,f'teacher_sidecar/{policy}/steps.jsonl.gz')
            trace=trace_manifest(original,schema=E_QUALITY_SCHEMA)
            published=trace_manifest(student,schema=E_QUALITY_SCHEMA)
            if (trace['policy']['id'] != policy or trace['identity']['source'] != corpus['source_binding']
                    or trace['chunks'] != published['chunks'] or trace['array_hash'] != published['array_hash']
                    or trace['records'] != published['records']
                    or any(trace[k] != published[k] for k in
                           ('e_observation_identity','data_identity','obs_version'))
                    or trace['data_identity'] != corpus['e_profile_identity']
                    or trace['replay_trace_hash'] != fingerprint({k:trace[k] for k in
                        ('identity','policy','env_config_hash','reward_version','array_hash','event_hash')})
                    or digest(sidecar) != digest(original/'steps.jsonl.gz')
                    or any(trace['files'][c['path']] != published['files'][c['path']] for c in trace['chunks'])):
                raise ValueError('SUPERVISION_PUBLICATION_TRACE_OWNER_MISMATCH')
            allowed={'schema','reward_version','obs_version','records','chunks','array_hash',
                     'e_observation_identity','data_identity','files','content_hash'}
            if set(published) != allowed:
                raise ValueError('SUPERVISION_POSTERIOR_IN_STUDENT_MANIFEST')
            previous=None; segment=-1; offset=0; chunk_index=0
            count=Counter(); array_hash=hashlib.sha256(); event_hash=hashlib.sha256()
            for native, row in zip_longest(iter_replay_rows(student,published),iter_sidecar(original)):
                if native is None or row is None:
                    raise ValueError('SUPERVISION_MISSING_OR_EXTRA_TEACHER')
                ordinal,step=native
                _join_row(row,step,ordinal,trace,corpus)
                if previous is None:
                    if not row['reset'] or not row['is_first']:
                        raise ValueError('SUPERVISION_TRACE_MISSING_INITIAL_RESET')
                else:_transition(previous,row)
                if row['reset'] != row['is_first']:
                    raise ValueError('SUPERVISION_RESET_FLAG_MISMATCH')
                if row['is_first']:segment+=1
                while offset>=published['chunks'][chunk_index]['length']:
                    offset-=published['chunks'][chunk_index]['length'];chunk_index+=1
                chunk=published['chunks'][chunk_index]['path']
                label=row['teacher']
                owner=fingerprint(dict(protocol='FG_S3_E_QUALITY_PAIR_OWNER_V1',
                    corpus=admission.source.corpus_content_hash, policy=policy,
                    trace=trace['replay_trace_hash'], segment=segment,
                    teacher=corpus['teacher_identity'], suffix=label['L2']['suffix_boundary_identity']))
                projected=project_label(label,row,step,owner)
                values={k:v.tolist() for k,v in projected.items()}
                audit=dict(state_id=label['state_id'],account_snapshot_hash=row['account_snapshot_hash'],
                    teacher_hash=label['content_hash'], owner=owner, time=row['time'],
                    decision_executed=row['decision_executed'], legal_mask=step['action_mask'].tolist(),
                    strict_reason=label['L2']['reason'], objective='discounted_native_float32_Quality_TP_feedback',
                    action_value_add=label['L2']['action_value_add'],action_value_wait=label['L2']['action_value_wait'],
                    witness_hash=fingerprint(label['L2'].get('value_rational_witness')))
                tags=_strata(label,projected)
                db.execute('INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (policy,ordinal,step['stepid'].tobytes(),segment,row['cursor'],str(row['trading_date']),
                     f'student_replay/{policy}/{chunk}',offset,_json(values),_json(audit),_json(tags),
                     int(row['is_first']),int(row['is_last']),int(row['is_terminal']),int(row['advanced'])))
                join_hash.update(_json(dict(policy=policy,ordinal=ordinal,stepid=row['stepid'],
                    audit=audit,labels=values)).encode())
                update_array_hash(array_hash,step)
                event_hash.update(_json({k:v for k,v in row.items() if k not in ('stepid','ordinal')}).encode())
                count['rows']+=1;count['advanced']+=int(row['advanced'])
                count['L1_raw_outcomes']+=int(label['L1']['immediate_outcome_verified'])
                count['L2_raw_valid']+=int(label['L2']['teacher_valid'])
                count['raw_strict_ADD' if label['L2']['strict_action']==1 else 'raw_strict_WAIT']+=int(label['L2']['teacher_valid'])
                count['funds_outside_certificate']+=int(label['L2']['reason']=='FUNDS_OUTSIDE_ORIGINAL_CERTIFICATE')
                for key in ('quality_score_valid','quality_outcome_valid','teacher_valid'):
                    count[key]+=int(projected[key])
                for key in ('quality_event_valid','quality_touch_valid','quality_adverse_valid'):
                    count[key]+=int(projected[key].sum())
                for tag in tags:count[tag]+=1
                previous=row;offset+=1
            if previous is None:raise ValueError('SUPERVISION_EMPTY_ACCOUNT_TRACE')
            _transition(previous,None)
            if array_hash.hexdigest()!=trace['array_hash'] or event_hash.hexdigest()!=trace['event_hash']:
                raise ValueError('SUPERVISION_NATIVE_ARRAY_OR_TRACE_HASH_MISMATCH')
            reports[policy]=dict(count)
        report=dict(protocol=INDEX_SCHEMA,source_identity=admission.receipt['content_identity'],
            policies=reports,join_hash=join_hash.hexdigest(),actual_optimizer_exposures=0,
            index_kind='JOIN_METADATA_ONLY; original native Replay arrays remain external')
        db.execute('INSERT INTO meta VALUES (?,?)',('report',_json(report)))
        db.commit()
    finally:db.close()
    receipt=signed(dict(**report,index_sha256=digest(path),index_path=str(path),
                        source_receipt_hash=admission.receipt['content_hash']))
    write_json(path.with_suffix('.json'),receipt)
    return receipt
