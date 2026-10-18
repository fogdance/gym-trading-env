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

INDEX_SCHEMA = 'FG_S3_E_QUALITY_JOIN_INDEX_V3'
READER_SCHEMA = 'FG_S3_E_QUALITY_READER_STATE_V1'
SAMPLING_STAGES = ('WM_NATURAL','HEADS_VALID','ACTOR_STRICT','ACTOR_QUALITY_PAIR')


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
    db.execute('''CREATE TABLE eligibility (policy TEXT, ordinal INTEGER, segment INTEGER,
        owner TEXT, heads INTEGER, strict INTEGER, action INTEGER, quality INTEGER, score REAL,
        PRIMARY KEY(policy,ordinal))''')
    db.execute('CREATE TABLE tags (policy TEXT, ordinal INTEGER, tag TEXT, PRIMARY KEY(policy,ordinal,tag))')
    db.execute('CREATE INDEX tag_lookup ON tags(policy,tag,ordinal)')
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
            previous=None; segment=-1; offset=0; chunk_index=0; suffix_domain=None
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
                boundary=fingerprint(label['L2']['suffix_boundary_identity'])
                if suffix_domain is not None and suffix_domain!=boundary:
                    raise ValueError('SUPERVISION_SUFFIX_DOMAIN_CHANGED_WITHIN_ACCOUNT')
                suffix_domain=boundary
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
                audit.update(incoming_action=row['incoming_action'],status=row['status'],
                    reward_events=row['reward_events'],native_exit_reasons=[e['reason'] for e in row['exits']])
                tags=_strata(label,projected)
                tags.append('POLICY:'+policy)
                db.execute('INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (policy,ordinal,step['stepid'].tobytes(),segment,row['cursor'],str(row['trading_date']),
                     f'student_replay/{policy}/{chunk}',offset,_json(values),_json(audit),_json(tags),
                     int(row['is_first']),int(row['is_last']),int(row['is_terminal']),int(row['advanced'])))
                heads=bool(projected['quality_score_valid'] or any(projected['quality_'+k+'_valid'].any()
                           for k in ('event','touch','adverse')))
                db.execute('INSERT INTO eligibility VALUES (?,?,?,?,?,?,?,?,?)',
                    (policy,ordinal,segment,owner,int(heads),int(projected['teacher_valid']),
                     int(projected['teacher_action']),int(projected['quality_score_valid'] and
                     projected['quality_outcome_valid']),float(projected['quality_score'])))
                db.executemany('INSERT INTO tags VALUES (?,?,?)',[(policy,ordinal,tag) for tag in tags])
                join_hash.update(_json(dict(policy=policy,ordinal=ordinal,stepid=row['stepid'],
                    audit=audit,labels=values)).encode())
                update_array_hash(array_hash,step)
                event_hash.update(_json({k:v for k,v in row.items() if k not in ('stepid','ordinal')}).encode())
                count['rows']+=1;count['advanced']+=int(row['advanced'])
                count['decision_executed']+=int(row['decision_executed'])
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


def batch_pair_counts(batch):
    """Use the real model's owner/reset contract on the effective time axis."""
    from dreamerv3.fg_s3_quality_v2 import pair_identity
    labels={k:v[:,1:] for k,v in batch.items() if k.startswith('quality_')}
    identity=np.asarray(pair_identity(labels,batch['is_first'][:,1:]))
    y,valid=batch['teacher_action'][:,1:],batch['teacher_valid'][:,1:]
    strict=identity & (valid & (y==1))[:,:,None] & (valid & (y==0))[:,None,:]
    known=labels['quality_score_valid'] & labels['quality_outcome_valid']
    score=labels['quality_score']
    quality=identity & known[:,:,None] & known[:,None,:] & (score[:,:,None]-score[:,None,:]>=15)
    return dict(strict_pair=int(strict.sum()),quality_pair=int(quality.sum()))


def _stat(path):
    st=path.stat()
    return st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns


class EQualityCourseReader:
    @classmethod
    def open(cls, admission, *, stage, sampling_spec, seed, index_path, cache_bytes=64*1024*1024):
        return cls(admission,stage,sampling_spec,seed,index_path,cache_bytes)

    def __init__(self,admission,stage,spec,seed,index_path,cache_bytes):
        if not isinstance(admission,Admission):raise ValueError('SUPERVISION_PREFLIGHT_OBJECT_REQUIRED')
        if stage not in SAMPLING_STAGES or type(seed) is not int or seed<0:
            raise ValueError('SUPERVISION_SAMPLING_STAGE_OR_SEED')
        if type(cache_bytes) is not int or cache_bytes<0:raise ValueError('SUPERVISION_CACHE_BUDGET')
        if (not isinstance(spec,dict) or set(spec)-{'mode','natural_weight','strata_weights','policies'}
                or spec.get('mode') not in ('NATURAL','STRATIFIED','MIXED')):
            raise ValueError('SUPERVISION_SAMPLING_SPEC')
        mode=spec['mode'];natural=spec.get('natural_weight')
        if mode=='MIXED' and (type(natural) not in (int,float) or not np.isfinite(natural) or not 0<natural<1):
            raise ValueError('SUPERVISION_EXPLICIT_MIX_WEIGHT_REQUIRED')
        if mode!='MIXED' and natural is not None:raise ValueError('SUPERVISION_UNUSED_MIX_WEIGHT')
        weights=spec.get('strata_weights',{})
        if (not isinstance(weights,dict) or mode=='NATURAL' and weights or
                mode!='NATURAL' and not weights or any(not isinstance(k,str) or
                type(v) not in (int,float) or not np.isfinite(v) or v<=0 for k,v in weights.items())):
            raise ValueError('SUPERVISION_EXPLICIT_STRATA_WEIGHTS_REQUIRED')
        policies=spec.get('policies',list(POLICIES))
        if (not isinstance(policies,list) or not policies or len(set(policies))!=len(policies)
                or not set(policies)<=set(POLICIES)):
            raise ValueError('SUPERVISION_POLICY_SELECTION')
        self.corpus=inspect_source(admission.source)
        self.root=Path(admission.source.corpus_root)
        self.admission=admission;self.stage=stage;self.spec=json.loads(_json(spec))
        self.mode=mode;self.natural_weight=natural;self.weights=dict(weights);self.policies=list(policies)
        self.contract=read_manifest(self.root/'source_lock.json')['contract']
        self.rng=np.random.default_rng(seed)
        from .fg_s3_e_quality_course_preflight import _local_path
        self.index_path=_local_path(index_path)
        receipt=read_manifest(self.index_path.with_suffix('.json'))
        if receipt.get('protocol')!=INDEX_SCHEMA:
            raise ValueError('SUPERVISION_INDEX_SCHEMA_MISMATCH: V3 receipt required; rebuild with current build_join_index')
        self.index_hash=digest(self.index_path)
        if (self.index_hash!=receipt['index_sha256']
                or receipt['source_identity']!=admission.receipt['content_identity']):
            raise ValueError('SUPERVISION_INDEX_IDENTITY_MISMATCH')
        self.index_stat=_stat(self.index_path)
        # Prefetch owns sampling; checkpoint/diagnostics access it only while
        # that worker is paused. SQLite remains read-only and query_only.
        self.db=sqlite3.connect(self.index_path.as_uri()+'?mode=ro',uri=True,check_same_thread=False)
        self.db.execute('PRAGMA query_only=ON');self.db.execute('PRAGMA cache_size=-4096')
        self.report=json.loads(self.db.execute("SELECT value FROM meta WHERE key='report'").fetchone()[0])
        if self.report.get('protocol')!=INDEX_SCHEMA:
            self.db.close()
            raise ValueError('SUPERVISION_INDEX_SCHEMA_MISMATCH: V3 database required; rebuild with current build_join_index')
        if any(self.report[k]!=receipt[k] for k in self.report):
            self.db.close();raise ValueError('SUPERVISION_INDEX_REPORT_MISMATCH')
        self.files={rel:(sha,_stat(member(self.root,rel))) for rel,sha in self.corpus['files'].items()}
        self.files['corpus_manifest.json']=(digest(self.root/'corpus_manifest.json'),_stat(self.root/'corpus_manifest.json'))
        self.cache=OrderedDict();self.cache_bytes=cache_bytes;self.cached_bytes=0;self.peak_cached_bytes=0
        self.counts=Counter(batches=0,delivered_candidate_exposures=0,strict_pair=0,quality_pair=0)
        self.seen_market=set();self.seen_account=set();self._choices={}

    def _guard(self):
        if _stat(self.index_path)!=self.index_stat:raise ValueError('SUPERVISION_INDEX_CHANGED')
        for rel,(expected,stat) in self.files.items():
            path=member(self.root,rel)
            current=_stat(path)
            if current!=stat:
                if digest(path)!=expected:raise ValueError('SUPERVISION_CONTENT_CHANGED: '+rel)
                self.files[rel]=(expected,current)
                self.cache.pop(rel,None)
                self.cached_bytes=sum(sum(x.nbytes for x in arrays.values()) for arrays in self.cache.values())

    def _condition(self,length,tag):
        clause='r.policy=? AND r.segment=? AND r.ordinal BETWEEN ? AND ?'
        base='e.policy=r.policy AND e.ordinal BETWEEN r.ordinal+1 AND r.ordinal+?'
        extra=[];params=[]
        if self.stage in ('HEADS_VALID','ACTOR_STRICT'):
            key='heads' if self.stage=='HEADS_VALID' else 'strict'
            extra.append(f'EXISTS (SELECT 1 FROM eligibility e WHERE {base} AND e.{key}=1)')
            params.append(length)
        elif self.stage=='ACTOR_QUALITY_PAIR':
            extra.append(f'(SELECT MAX(e.score)-MIN(e.score) FROM eligibility e WHERE {base} AND e.quality=1)>=15')
            params.append(length)
        if tag is not None:
            extra.append('EXISTS (SELECT 1 FROM tags t WHERE t.policy=r.policy AND t.ordinal BETWEEN r.ordinal+1 AND r.ordinal+? AND t.tag=?)')
            params.extend([length,tag])
        return clause+''.join(' AND '+s for s in extra),params

    def _groups(self,length,tag=None):
        key=(length,tag)
        if key in self._choices:return self._choices[key]
        groups=[];condition,extra=self._condition(length,tag)
        for policy,segment,lo,hi in self.db.execute('SELECT policy,segment,MIN(ordinal),MAX(ordinal) FROM rows GROUP BY policy,segment ORDER BY policy,segment'):
            if policy not in self.policies or hi-lo<length:continue
            n=self.db.execute('SELECT COUNT(*) FROM rows r WHERE '+condition,
                             [policy,segment,lo,hi-length,*extra]).fetchone()[0]
            if n:groups.append((policy,segment,lo,hi-length,n))
        self._choices[key]=groups
        return groups

    def _choose(self,length,tag):
        groups=self._groups(length,tag)
        total=sum(g[-1] for g in groups)
        if not total:raise ValueError('INSUFFICIENT_CURRICULUM_SUPPORT: '+str(tag or self.stage))
        pick=int(self.rng.integers(total))
        for policy,segment,lo,hi,n in groups:
            if pick<n:
                condition,extra=self._condition(length,tag)
                ordinal=self.db.execute('SELECT r.ordinal FROM rows r WHERE '+condition+' ORDER BY r.ordinal LIMIT 1 OFFSET ?',
                    [policy,segment,lo,hi,*extra,pick]).fetchone()[0]
                return policy,ordinal
            pick-=n
        raise AssertionError('Unreachable window selection')

    def _chunk(self,rel):
        if rel in self.cache:
            self.cache.move_to_end(rel);return self.cache[rel]
        path=member(self.root,rel);before=_stat(path)
        if digest(path)!=self.corpus['files'][rel]:raise ValueError('SUPERVISION_CONTENT_CHANGED: '+rel)
        with np.load(path,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
        if _stat(path)!=before:raise ValueError('SUPERVISION_CONTENT_CHANGED_DURING_READ')
        size=sum(v.nbytes for v in arrays.values())
        if size<=self.cache_bytes:
            while self.cache and self.cached_bytes+size>self.cache_bytes:
                _,old=self.cache.popitem(last=False);self.cached_bytes-=sum(v.nbytes for v in old.values())
            self.cache[rel]=arrays;self.cached_bytes+=size
            self.peak_cached_bytes=max(self.peak_cached_bytes,self.cached_bytes)
        return arrays

    def sample(self,batch_size,effective_length):
        if type(batch_size) is not int or type(effective_length) is not int or min(batch_size,effective_length)<1:
            raise ValueError('SUPERVISION_POSITIVE_BATCH_AND_LENGTH_REQUIRED')
        self._guard()
        for tag in self.weights:
            if not self._groups(effective_length,tag):
                raise ValueError('INSUFFICIENT_CURRICULUM_SUPPORT: '+tag)
        if not self._groups(effective_length):raise ValueError('INSUFFICIENT_CURRICULUM_SUPPORT: '+self.stage)
        from .fg_s3_quality_catalog_v2 import FIELDS
        fields={**replay_fields(E_QUALITY_SCHEMA),'stepid':((20,),np.dtype('uint8'))}
        shape=(batch_size,effective_length+1)
        batch={k:np.zeros((*shape,*dims),dtype) for k,(dims,dtype) in fields.items()}
        batch.update({k:np.zeros((*shape,*dims),dtype) for k,(dtype,dims) in FIELDS.items()})
        batch.update(teacher_action=np.zeros(shape,np.int32),teacher_valid=np.zeros(shape,bool),
                     consec=np.zeros(shape,np.int32))
        delivered=Counter();market=set();accounts=set()
        for b in range(batch_size):
            stratified=self.mode=='STRATIFIED' or self.mode=='MIXED' and self.rng.random()>=self.natural_weight
            tag=None
            if stratified:
                tags=sorted(self.weights);weights=np.array([self.weights[k] for k in tags],float)
                tag=tags[int(self.rng.choice(len(tags),p=weights/weights.sum()))]
            policy,start=self._choose(effective_length,tag)
            rows=self.db.execute('SELECT ordinal,stepid,cursor,chunk,offset,labels,audit FROM rows WHERE policy=? AND ordinal BETWEEN ? AND ? ORDER BY ordinal',
                                 (policy,start,start+effective_length))
            for t,(ordinal,stepid,cursor,chunk,offset,labels,audit) in enumerate(rows):
                arrays=self._chunk(chunk)
                if arrays['stepid'][offset].tobytes()!=stepid:raise ValueError('SUPERVISION_INDEX_STEPID_MISMATCH')
                for key in fields:batch[key][b,t]=arrays[key][offset]
                if t:
                    labels=json.loads(labels);audit=json.loads(audit)
                    for key in (*FIELDS,'teacher_action','teacher_valid'):batch[key][b,t]=labels[key]
                    market.add((self.contract,cursor))
                    if audit['decision_executed']:accounts.add((policy,ordinal))
            delivered['policy:'+policy]+=effective_length
            delivered['stratum:'+(tag or 'NATURAL')]+=1
        delivered.update(batch_pair_counts(batch))
        self.counts.update(delivered);self.seen_market.update(market);self.seen_account.update(accounts)
        self.counts['batches']+=1;self.counts['delivered_candidate_exposures']+=batch_size*effective_length
        return batch

    def coverage(self):
        self._guard()
        # Market denominator excludes account multiplicity.
        raw_market=self.db.execute('SELECT COUNT(DISTINCT cursor) FROM rows').fetchone()[0]
        raw_decisions=sum(p['decision_executed'] for p in self.report['policies'].values())
        strict_pairs=self.db.execute('''SELECT COUNT(*) FROM eligibility a JOIN eligibility b
            ON a.policy=b.policy AND a.segment=b.segment AND a.owner=b.owner
            WHERE a.strict=1 AND b.strict=1 AND a.action=1 AND b.action=0''').fetchone()[0]
        quality_pairs=self.db.execute('''SELECT COUNT(*) FROM eligibility a JOIN eligibility b
            ON a.policy=b.policy AND a.segment=b.segment AND a.owner=b.owner
            WHERE a.quality=1 AND b.quality=1 AND a.score-b.score>=15''').fetchone()[0]
        return dict(protocol=READER_SCHEMA,stage=self.stage,sampling_spec=self.spec,
            source_identity=self.report['source_identity'],index_sha256=self.index_hash,
            policies=self.report['policies'],unique_market_cursors=raw_market,
            distinct_account_decisions=raw_decisions,actual_optimizer_exposures=0,
            delivered_unique_market_cursors=len(self.seen_market),delivered_unique_account_decisions=len(self.seen_account),
            delivered_counts=dict(self.counts),all_episode_strict_pairs=strict_pairs,
            all_episode_quality_pairs=quality_pairs,
            support={str(length)+':'+str(tag or self.stage):sum(g[-1] for g in groups)
                     for (length,tag),groups in self._choices.items()},
            peak_cached_bytes=self.peak_cached_bytes,cache_budget_bytes=self.cache_bytes)

    def audit_windows(self, batch, *, supervision_fields=()):
        """Bind native rows and, when requested, canonical post-step labels.

        Only metadata is returned. This does not construct or sample a Replay.
        It also supplies past native reward events for F1 diagnostics.
        """
        self._guard()
        if supervision_fields:
            from .fg_s3_quality_catalog_v2 import FIELDS
            if not set(supervision_fields)<=set(FIELDS)|{'teacher_action','teacher_valid'}:
                raise ValueError('SUPERVISION_AUDIT_UNKNOWN_FIELD')
        ids=batch['stepid']
        if ids.dtype!=np.uint8 or ids.ndim!=3 or ids.shape[-1]!=20:
            raise ValueError('F1_PHYSICAL_STEPID_SHAPE')
        fields={**replay_fields(E_QUALITY_SCHEMA),'stepid':((20,),np.dtype('uint8'))}
        for key,(shape,dtype) in fields.items():
            if key not in batch or batch[key].dtype!=dtype or batch[key].shape!=(*ids.shape[:2],*shape):
                raise ValueError('F1_NATIVE_FIELD_SHAPE_OR_DTYPE: '+key)
        metadata=[]
        for b,window in enumerate(ids):
            rows=[];previous=None;episode=None
            for t,stepid in enumerate(window):
                item=self.db.execute('SELECT policy,ordinal,segment,cursor,chunk,offset,audit,labels FROM rows WHERE stepid=?',
                                     (stepid.tobytes(),)).fetchone()
                if item is None:raise ValueError('F1_UNKNOWN_STEPID')
                policy,ordinal,segment,cursor,chunk,offset,audit,labels=item
                if episode is None:episode=(policy,segment)
                if (policy,segment)!=episode or previous is not None and ordinal!=previous+1:
                    raise ValueError('F1_OWNER_EPISODE_OR_ORDINAL_MISMATCH')
                arrays=self._chunk(chunk)
                for key in fields:
                    if batch[key][b,t].tobytes()!=arrays[key][offset].tobytes():
                        raise ValueError('F1_NATIVE_WINDOW_BYTES_MISMATCH: '+key)
                if supervision_fields:
                    canonical=json.loads(labels)
                    for key in supervision_fields:
                        value=batch[key][b,t]
                        expected=np.asarray(canonical[key],dtype=value.dtype) if t else np.zeros_like(value)
                        if value.tobytes()!=expected.tobytes():
                            raise ValueError('SUPERVISION_POST_STEP_LABEL_MISMATCH: '+key)
                past=json.loads(audit)
                rows.append(dict(policy=policy,segment=segment,ordinal=ordinal,cursor=cursor,
                    stepid=stepid.tobytes().hex(),incoming_action=past['incoming_action'],
                    status=past['status'],reward_events=past['reward_events'],
                    native_exit_reasons=past['native_exit_reasons']))
                previous=ordinal
            reset=self.db.execute('SELECT MIN(ordinal),MIN(cursor) FROM rows WHERE policy=? AND segment=?',episode).fetchone()
            metadata.append(dict(contract=self.contract,policy=episode[0],segment=episode[1],
                reset_ordinal=reset[0],distance_from_reset_market_minutes=rows[0]['cursor']-reset[1],rows=rows))
        return metadata

    def state_dict(self):
        return json.loads(_json(dict(protocol=READER_SCHEMA,source=self.report['source_identity'],index=self.index_hash,
            stage=self.stage,sampling_spec=self.spec,rng=self.rng.bit_generator.state,
            counts=dict(self.counts),seen_market=sorted(self.seen_market),seen_account=sorted(self.seen_account))))

    def load_state_dict(self,state):
        self._guard()
        if any(state.get(k)!=v for k,v in dict(protocol=READER_SCHEMA,source=self.report['source_identity'],
                index=self.index_hash,stage=self.stage,sampling_spec=self.spec).items()):
            raise ValueError('SUPERVISION_READER_RESTORE_IDENTITY_MISMATCH')
        rng=np.random.default_rng();rng.bit_generator.state=state['rng']
        self.rng=rng;self.counts=Counter(state['counts'])
        self.seen_market={tuple(x) for x in state['seen_market']}
        self.seen_account={tuple(x) for x in state['seen_account']}

    def close(self):
        self.db.close();self.cache.clear();self.cached_bytes=0
