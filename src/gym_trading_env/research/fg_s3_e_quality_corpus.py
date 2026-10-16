"""E-native multi-account teaching corpus; no model or training lifecycle.

Native Driver/Replay performs execution; this module supplies policy specs and
separate posterior facts. It cannot admit a training run.
"""
from __future__ import annotations
import copy
from pathlib import Path

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import (collect_trace, reconcile_trace, signed, write_json,
                        E_QUALITY_PROTOCOL,E_QUALITY_SCHEMA)
from .fg_single3_replay import Single3BehaviorPolicy
from .fg_s3_quality_teacher_v2 import NativeQualityTeacher

SCHEMA='FG_S3_E_QUALITY_CORPUS_V2'
TEACHER_SCHEMA='FG_S3_E_QUALITY_TEACHER_V2'


def behavior_specs(seed=202):
    if type(seed) is not int or seed<0:raise ValueError('Explicit nonnegative behavior seed required')
    return [dict(id=name,kind=kind,seed_components=[seed,0,i],**params) for i,(name,kind,params) in enumerate([
        ('P0','always_wait',{}),('P1','always_add',{}),
        ('P2_low','probabilistic',dict(probability=.1)),
        ('P2_mid','probabilistic',dict(probability=.5)),
        ('P3','single_delay',dict(wait_minutes=15))])]


def l1_labeler(teacher, strict=None):
    def label(env, step, row):
        facts=teacher.query_state(env,env._published_mask)
        l2=(strict.query_values(env,env._published_mask) if strict is not None else
            dict(strict_action=None,teacher_valid=False,reason='STAGE_D_NOT_BUILT',
                 action_value_add=None,action_value_wait=None,margin=None,tie=None))
        return signed(dict(protocol=TEACHER_SCHEMA,fold='TRAIN',
            state_id=facts['state_id'],account_snapshot_hash=facts['account_snapshot_hash'],
            L1=facts,L2=l2,quality_q_v2_valid=False,prediction_admitted=False,decision_admitted=False))
    return label


def generate_behavior(factory, spec, output, dreamer_root, *, source, initial_state=None,
                      end_cursor=None, strict=None, policy_factory=Single3BehaviorPolicy):
    if source.get('fold')!='TRAIN':raise ValueError('Corpus requires explicit TRAIN source')
    env=factory()
    if env.entry_config.get('state_path'):
        raise ValueError('Corpus refuses a persistent state_path')
    teacher=None
    try:
        if initial_state is not None:env.import_state(initial_state)
        else:env.reset(seed=0)
        teacher=NativeQualityTeacher(env,factory,fold='TRAIN',source=source)
        trace=collect_trace('',spec,output,dict(protocol=SCHEMA,source=source,
            teacher_identity=teacher.identity_hash,training_admitted=False),dreamer_root,
            env_factory=lambda:env,protocol=E_QUALITY_PROTOCOL,policy_factory=policy_factory,
            initial_state=initial_state,end_cursor=end_cursor,labeler=l1_labeler(teacher,strict))
        receipt=reconcile_trace(output,dreamer_root,env_factory=factory,schema=E_QUALITY_SCHEMA)
        write_json(Path(output)/'reconciliation.json',signed(receipt))
        return trace,receipt
    finally:
        if teacher is not None:teacher.close()


def coverage(paths):
    """Recompute observed counts; counterfactual labels never become trades."""
    from collections import Counter
    import numpy as np
    from .fg_replay import trace_manifest,iter_sidecar,iter_replay_rows
    market=set();advanced=set();days=set();states=set();reports={}
    def distribution(values):
        return dict(n=len(values),p50=float(np.quantile(values,.5)) if values else None,
                    p90=float(np.quantile(values,.9)) if values else None,
                    minimum=min(values) if values else None,maximum=max(values) if values else None)
    for path in paths:
        manifest=trace_manifest(path,schema=E_QUALITY_SCHEMA)
        name=manifest['policy']['id']
        if name in reports:raise ValueError('Duplicate behavior in Corpus coverage')
        count=Counter();numeric={k:[] for k in ['quality_score','tp_market_minutes',
            'mae_5','mae_15','mae_60','mae_240','mae_full','underwater_minutes','underwater_longest_minutes']}
        for row,(_,step) in zip(iter_sidecar(path),
                               iter_replay_rows(path,manifest),strict=True):
            count['replay_rows']+=1;count['advanced']+=int(row['advanced'])
            count['valid_market_minutes']+=int(row['valid_bar'])
            count['macd_valid_slots']+=int(step['market_macd_5m'][3]==1)
            market.add((manifest['contract'],row['cursor']));days.add((manifest['contract'],row['trading_date']))
            if row['valid_bar']:advanced.add((manifest['contract'],row['cursor']))
            executed=row['decision_executed']
            if executed:
                states.add((manifest['replay_trace_hash'],row['account_snapshot_hash']))
                count['legal_add_opportunities']+=int(row['legal_add'])
                count['voluntary_wait']+=int(row['legal_add'] and row['action']==0)
                count['forced_wait']+=int(not row['legal_add'] and row['action']==0)
            else:count['nonexecuted_proposals']+=1
            count['filled_add']+=sum(f['reason']=='ENTRY' for f in row['fills'])
            for event in row['exits']:count['executed_exit/'+event['reason']]+=1
            label=row.get('teacher',{})
            facts=label.get('L1',{})
            l2=label.get('L2',{})
            count['strict_valid']+=int(l2.get('teacher_valid',False))
            if l2.get('teacher_valid'):count['strict_action/'+str(l2['strict_action'])]+=1
            if facts.get('immediate_outcome_verified'):
                count['l1_candidates']+=1;count['l1_outcome/'+facts['exit_type']]+=1
                for key in numeric:
                    value=facts.get('score') if key=='quality_score' else facts.get('held_market_minutes') if key=='tp_market_minutes' and facts['exit_type']=='TP5' else facts.get(key) if key!='tp_market_minutes' else None
                    if value is not None:numeric[key].append(value)
                count['l1_same_day_tp']+=int(facts['exit_type']=='TP5' and facts['crossed_trading_days']==0)
                count['l1_cross_day_tp']+=int(facts['exit_type']=='TP5' and facts['crossed_trading_days']>0)
                for horizon in (5,15):
                    count[f'early10_{horizon}_valid']+=int(facts[f'early10_{horizon}_valid'])
                    count[f'early10_{horizon}_positive']+=int(facts[f'early10_{horizon}'])
        reports[name]=dict(counts=dict(count),distributions={k:distribution(v) for k,v in numeric.items()},
            units=dict(quality_score='0..100',tp_market_minutes='valid_market_minutes',mae='quoted_price_points',underwater='valid_market_minutes'),
            trace_hash=manifest['replay_trace_hash'])
    return signed(dict(protocol='FG_S3_E_QUALITY_COVERAGE_V2',behaviors=reports,
        unique_market_cursors=len(market),unique_advanced_market_minutes=len(advanced),
        unique_contract_official_days=len(days),distinct_account_decision_states=len(states),
        optimizer_exposures=0,teacher_labels_are_independent_market_samples=False,
        distribution_denominator='observed L1 candidates, not executed trades; incomplete prefix MAE is observed exposure'))
