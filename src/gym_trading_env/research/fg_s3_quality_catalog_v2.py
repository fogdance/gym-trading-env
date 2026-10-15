"""Authenticated quality sidecars and independently masked dense TRAIN join."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import signed, write_json

FIELDS = {
    'quality_score': (np.float32, ()), 'quality_score_valid': (bool, ()),
    'quality_outcome_valid': (bool, ()), 'quality_sl': (bool, ()),
    'quality_early5': (bool, ()), 'quality_tp': (bool, ()),
    'quality_event': (np.int32, (7,)), 'quality_event_valid': (bool, (7,)),
    'quality_touch': (np.float32, (4,2)), 'quality_touch_valid': (bool, (4,2)),
    'quality_adverse': (np.float32, (4,)), 'quality_adverse_valid': (bool, (4,)),
    'quality_q_c1': (np.float32, (2,)), 'quality_q_c1_valid': (bool, ()),
    'quality_q_v2': (np.float32, (2,)), 'quality_q_v2_valid': (bool, ()),
    'quality_owner': (np.uint32, (8,)),
    'quality_soft_target':(np.float32,()),'quality_soft_valid':(bool,()),
    'quality_soft_source':(np.int32,()),
}


def soft_teacher(label,strict):
    """Reviewed section 5.5: explicit source, never relabel strict actions."""
    import math
    q=label.get('q',{})
    if q.get('q_v2_pair_valid'):
        delta=q['q_v2'][1]-q['q_v2'][0]
        value=1/(1+math.exp(-delta)) if delta>=0 else math.exp(delta)/(1+math.exp(delta))
        return dict(soft_target_source='Q_V2_VERIFIED',soft_target_pADD=value)
    if not strict.get('valid') or not label.get('immediate_outcome_verified'):
        return dict(soft_target_source='INVALID',soft_target_pADD=None)
    early=label.get('first_adverse',{}).get('10')
    risk=label.get('exit_type')=='SL30' or early is not None and early<=5
    high=label.get('exit_type')=='TP5' and label.get('score_valid') and label['score']>=80
    value=.90 if strict['action'] else .01 if risk else .25 if high else .05
    return dict(soft_target_source='HEURISTIC_V2',soft_target_pADD=value)


@dataclass(frozen=True)
class QualityCatalogIdentity:
    protocol: str
    teacher_identity: str
    fold: str = 'TRAIN'
    clock_protocol: str = 'S3_MARKET_MINUTE'


def validate_label(label):
    payload = {k:v for k,v in label.items() if k != 'content_hash'}
    if fingerprint(payload) != label.get('content_hash'):
        raise ValueError('Quality teacher proof hash mismatch')
    if label.get('protocol')!='FG_S3_QUALITY_TEACHER_V1': raise ValueError('Unknown quality label protocol')
    if label.get('score_policy')!='FG_S3_QUALITY_TP_MARKET_DAY_V1':
        raise ValueError('Catalog requires the Quality-first B(t,d) score protocol')
    if label.get('fold')!='TRAIN' or label.get('training_action_label_allowed') is not False:
        raise ValueError('Quality facts are TRAIN auxiliary labels, not action teachers')
    layers=label.get('layers',{})
    if set(layers)!= {'L0','L1','L2','L3'} or any(
            not {'valid','reason','protocol','source'}<=set(layer) for layer in layers.values()):
        raise ValueError('Catalog requires four independent supervision layers')
    if (bool(layers['L1']['valid'])!=bool(label.get('immediate_outcome_verified')) or
            bool(layers['L2']['valid'])!=bool(label['score_valid']) or
            bool(layers['L3']['valid'])!=bool(label.get('decision_oracle_certified'))):
        raise ValueError('Catalog layer validity differs from its certified target')
    if label.get('score_valid') and (not label.get('immediate_outcome_verified')
            or label.get('exit_type') not in ('TP5','SL30') or label.get('score') is None
            or not np.isfinite(label['score']) or not 0<=label['score']<=100):
        raise ValueError('Unverified quality score target')
    if label.get('score_valid'):
        from .fg_s3_quality_score import quality_for_exit
        expected,_=quality_for_exit('TAKE_PROFIT' if label['exit_type']=='TP5' else 'STOP_LOSS',
            label['held_market_minutes'],label['crossed_trading_days'])
        if label['score']!=expected: raise ValueError('Catalog score differs from the frozen Quality-first B formula')
        event=label.get('native_event')
        if (not event or fingerprint(event)!=label.get('native_event_hash') or
                event['net_trade_pnl']!=label.get('net_pnl') or
                event['reason']!=('TAKE_PROFIT' if label['exit_type']=='TP5' else 'STOP_LOSS') or
                event['held_market_bars']!=label['held_market_minutes']):
            raise ValueError('Catalog score lacks its exact native settlement proof')
    for key,shape in (('event',(7,)),('event_valid',(7,)),('touch',(4,2)),('touch_valid',(4,2)),
                      ('adverse',(4,)),('adverse_valid',(4,))):
        if np.asarray(label[key]).shape!=shape: raise ValueError('Quality horizon shape differs: '+key)
    if not np.isin(label['event'],[0,1,2,3]).all(): raise ValueError('Invalid quality event class')
    q = label.get('q', {})
    source=label.get('soft_target_source','INVALID')
    if (source not in ('INVALID','HEURISTIC_V2','Q_V2_VERIFIED') or
            source=='Q_V2_VERIFIED' and not q.get('q_v2_pair_valid') or
            source=='HEURISTIC_V2' and not label.get('decision_oracle_certified')):
        raise ValueError('Soft teacher source is not certified for its declared role')
    if q:
        if fingerprint({k:v for k,v in q.items() if k != 'content_hash'}) != q.get('content_hash'):
            raise ValueError('Quality Q proof hash mismatch')
        for key in ('q_c1','q_v2'):
            if q.get(key+'_pair_valid') and (len(q[key]) != 2 or any(v is None or not np.isfinite(v) for v in q[key])):
                raise ValueError('Uncertified paired Q values')
    return label


def attach_quality(batch, records, *, teacher_identities, fold='TRAIN'):
    if fold != 'TRAIN' or any(k in batch for k in FIELDS):
        raise ValueError('Quality join requires an unjoined TRAIN batch')
    if 'teacher_valid' not in batch:
        raise ValueError('Quality join requires the validated native teacher join first')
    index = {r['stepid']:r for r in records}
    if len(index) != len(records): raise ValueError('Duplicate quality Replay owner')
    shape = batch['stepid'].shape[:2]
    out = {k:np.zeros((*shape,*dims),dtype) for k,(dtype,dims) in FIELDS.items()}
    for pos in np.ndindex(shape):
        row = index.get(batch['stepid'][pos].tobytes().hex())
        if row is None: raise ValueError('Unknown quality owner')
        label = validate_label(row['quality'])
        if (label['teacher_identity'] not in teacher_identities or label['state_id'] != row['state_id']
                or label.get('strict_identity') != row['oracle_identity']
                or bool(label.get('decision_oracle_certified')) != bool(row['oracle_valid'])):
            raise ValueError('Quality owner/source/snapshot alignment mismatch')
        # Prefix or terminal rows may be physically present without a completed
        # feedback transaction. They never acquire supervision by joining.
        legal_completed=bool(row['execution_result']['advanced'] and all(row['action_mask']))
        if legal_completed:
            q=label.get('q',{})
            for kind in ('c1','v2'):
                valid=bool(q.get('q_'+kind+'_pair_valid'))
                out['quality_q_'+kind+'_valid'][pos]=valid
                if valid: out['quality_q_'+kind][pos]=q['q_'+kind]
            owner=fingerprint(dict(episode=row['episode_id'],contract=row['contract'],
                teacher=label['teacher_identity'],strict=row['oracle_identity']))
            out['quality_owner'][pos]=np.frombuffer(bytes.fromhex(owner),'>u4').astype(np.uint32)
            soft_source=label['soft_target_source']
            out['quality_soft_source'][pos]={'INVALID':0,'HEURISTIC_V2':1,'Q_V2_VERIFIED':2}[soft_source]
            out['quality_soft_valid'][pos]=soft_source!='INVALID'
            if soft_source!='INVALID': out['quality_soft_target'][pos]=label['soft_target_pADD']
        eligible=bool(legal_completed and label.get('immediate_outcome_verified'))
        if not eligible: continue
        out['quality_outcome_valid'][pos] = True
        out['quality_score_valid'][pos] = label['score_valid']
        out['quality_score'][pos] = label['score'] if label['score_valid'] else 0.
        out['quality_tp'][pos] = label.get('exit_type') == 'TP5'
        out['quality_sl'][pos] = label.get('exit_type') == 'SL30'
        first = label.get('first_adverse',{}).get('10')
        out['quality_early5'][pos] = first is not None and first <= 5
        for key in ('event','event_valid','touch','touch_valid','adverse','adverse_valid'):
            out['quality_'+key][pos] = label[key]
    return {**batch,**out}


def coverage(records):
    counts = Counter(total_rows=len(records))
    scores = []
    for row in records:
        label = validate_label(row['quality'])
        counts['strict_valid'] += int(row['teacher_valid'])
        flat=bool(row['execution_result']['advanced'] and all(row['action_mask']))
        counts['completed_dual_legal_flat_decisions']+=int(flat)
        low=float(row['equity_before'])<300000
        if flat:
            counts['flat_funds/'+('below_300000' if low else 'at_least_300000')]+=1
            counts['flat_outcome/'+label.get('exit_type','UNKNOWN')]+=1
            counts['flat_entry_filled']+=int(label.get('immediate_outcome_verified',False))
            counts['flat_quality_valid']+=int(label['score_valid'])
            counts['flat_strict_valid']+=int(row['teacher_valid'])
            if low:
                counts['low_funds_flat_quality_valid']+=int(label['score_valid'])
                counts['low_funds_flat_strict_valid']+=int(row['teacher_valid'])
        counts['immediate_outcome_verified'] += int(label.get('immediate_outcome_verified',False))
        counts['score_valid'] += int(label['score_valid'])
        counts['funds/'+('below_300000' if float(row['equity_before'])<300000 else 'at_least_300000')]+=1
        counts['soft/'+label.get('soft_target_source','INVALID')]+=1
        if label['score_valid']: scores.append(label['score'])
        for k in ('event_valid','touch_valid','adverse_valid'):
            counts[k] += int(np.asarray(label[k]).sum())
        for k in ('q_c1_pair_valid','q_v2_pair_valid'):
            counts[k] += int(label.get('q',{}).get(k,False))
        for h in range(7):
            if label['event_valid'][h]:
                counts[f'event_h{h}/valid']+=1
                counts[f'event_h{h}/'+('TP5','SL30','OTHER','OPEN')[label['event'][h]]]+=1
        counts['reason/'+label.get('primary',label.get('invalid_reason') or 'UNKNOWN')] += 1
    return signed(dict(protocol='FG_S3_QUALITY_COVERAGE_V1',counts=dict(counts),
        score_mean=float(np.mean(scores)) if scores else None,
        score_quantiles=np.quantile(scores,[0,.25,.5,.75,1]).tolist() if scores else None,
        denominator='committed native rows; each task uses its own valid count'))


def write_catalog_receipt(path, identity, records):
    from pathlib import Path
    from .fg_replay import read_manifest
    for row in records: validate_label(row['quality'])
    result = signed(dict(protocol='FG_S3_QUALITY_CATALOG_V1',identity=identity,
        record_count=len(records),records_hash=fingerprint(records),coverage=coverage(records)))
    if Path(path).exists():
        if read_manifest(path)!=result: raise ValueError('Existing quality Catalog receipt differs')
        return result
    write_json(path,result)
    return result


def validate_catalog(path):
    """Authenticate the committed journal prefix and prepared TRAIN source."""
    from pathlib import Path
    import hashlib,json,re
    from .fg_replay import read_manifest,check_files,digest
    manifest=read_manifest(path)
    identity=manifest['identity']
    if (manifest.get('protocol')!='FG_S3_QUALITY_CATALOG_V1' or identity.get('fold')!='TRAIN' or
            identity.get('score_policy')!='FG_S3_QUALITY_TP_MARKET_DAY_V1'):
        raise ValueError('Catalog requires independently authenticated TRAIN quality facts')
    prepared=Path(identity['prepared_dir'])
    if digest(prepared/'manifest.json')!=identity['prepared_manifest_sha256']:
        raise ValueError('Catalog preparation binding changed')
    source=read_manifest(prepared/'manifest.json')
    if (source.get('protocol')!='FG_S3_V2_PREPARED_V1' or
            source.get('clock_protocol')!='S3_MARKET_MINUTE' or not source.get('code') or
            'quality_binding.json' not in source['files'] or 'quality_pool.json' not in source['files']):
        raise ValueError('Catalog preparation is not a complete quality source')
    check_files({str(prepared/p):h for p,h in source['files'].items()})
    check_files(source['code'])
    binding=read_manifest(prepared/'quality_binding.json')
    pool=read_manifest(prepared/'quality_pool.json')
    if binding.get('fold')!='TRAIN' or binding.get('score_policy')!='FG_S3_QUALITY_TP_MARKET_DAY_V1':
        raise ValueError('Catalog TRAIN/score binding differs')
    if source['code']!=identity['source_code'] or source['versions']!=identity['source_git']:
        raise ValueError('Catalog source code/Git identity differs')
    if identity.get('checkpoint'):
        from embodied.run.fg_checkpoint import validate_bundle
        node=Path(identity['checkpoint']);validate_bundle(node)
        if digest(node/'manifest.json')!=identity['checkpoint_manifest_sha256']:
            raise ValueError('Catalog FullCheckpoint binding changed')
    journal=Path(identity['journal']).read_bytes()
    size=identity['journal_prefix_bytes']
    if type(size) is not int or size<1 or len(journal)<size:
        raise ValueError('Catalog committed journal prefix is missing')
    prefix=journal[:size]
    if hashlib.sha256(prefix).hexdigest()!=identity['journal_prefix_sha256']:
        raise ValueError('Catalog committed journal bytes changed')
    records=[json.loads(s) for s in prefix.splitlines()]
    if len(records)!=manifest['record_count'] or fingerprint(records)!=manifest['records_hash']:
        raise ValueError('Catalog committed record ownership changed')
    teachers=set(identity['teacher_identities']);owners=set()
    if teachers!={query['teacher_identity'] for query in pool['queries'].values()}:
        raise ValueError('Catalog teacher whitelist differs from its authenticated pool')
    for row in records:
        label=validate_label(row['quality'])
        contract=row['contract']
        if (contract not in pool['queries'] or label['teacher_identity']!=pool['queries'][contract]['teacher_identity'] or
                row['oracle_identity']!=fingerprint(pool['queries'][contract]) or
                label['state_proof']['source_hash']!=fingerprint(pool['sources'][contract])):
            raise ValueError('Catalog contract/teacher source proof differs')
        if (not re.fullmatch('[0-9a-f]{40}',row['stepid']) or row['stepid'] in owners or
                label['teacher_identity'] not in teachers or label['state_id']!=row['state_id'] or
                label['strict_identity']!=row['oracle_identity'] or
                bool(label['decision_oracle_certified'])!=bool(row['oracle_valid'])):
            raise ValueError('Catalog native Replay owner/source/state alignment differs')
        owners.add(row['stepid'])
        if (label['decision_oracle_certified'] and label['layers']['L3']['protocol']=='STRICT_SUFFIX_AUDITOR' and
                float(row['equity_before'])<300000):
            raise ValueError('Catalog illegally extended the original 300000 funds certificate')
    if coverage(records)!=manifest['coverage']:
        raise ValueError('Catalog independent supervision coverage differs')
    return signed(dict(status='PASS_AUTHENTICATED_TRAIN_QUALITY_CATALOG',
        catalog_sha256=digest(path),record_count=len(records),coverage=manifest['coverage'],
        current_journal_extra_bytes=len(journal)-size,actor_action_labels_from_quality=False))


def train_priors(records):
    event=np.zeros((7,4),np.int64);touch_n=np.zeros((4,2),np.int64);touch_y=touch_n.copy()
    adverse_n=np.zeros(4,np.int64);adverse_y=adverse_n.copy()
    scores=[]
    for row in records:
        label=validate_label(row['quality'])
        if not row['execution_result']['advanced'] or not label.get('immediate_outcome_verified'): continue
        for h in range(7):
            if label['event_valid'][h]: event[h,label['event'][h]]+=1
        tm=np.asarray(label['touch_valid'],int);am=np.asarray(label['adverse_valid'],int)
        touch_n+=tm;touch_y+=tm*np.asarray(label['touch'],int)
        adverse_n+=am;adverse_y+=am*np.asarray(label['adverse'],int)
        if label['score_valid']: scores.append(label['score'])
    return signed(dict(protocol='FG_S3_QUALITY_TRAIN_PRIORS_V1',fold='TRAIN',
        source_records_hash=fingerprint(records),event_counts=event.tolist(),
        touch_counts=touch_n.tolist(),touch_positive=touch_y.tolist(),
        adverse_counts=adverse_n.tolist(),adverse_positive=adverse_y.tolist(),
        score_count=len(scores),score_mean=float(np.mean(scores)) if scores else None,
        score_median=float(np.median(scores)) if scores else None,
        calibration_status='NOT_CALIBRATED',scope='committed training state distribution'))
