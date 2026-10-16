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
            dict(strict_action=None,teacher_valid=False,reason='NO_STRICT_TEACHER_BOUND',
                 action_value_add=None,action_value_wait=None,margin=None,tie=None))
        value=signed(dict(protocol=TEACHER_SCHEMA,fold='TRAIN',
            state_id=facts['state_id'],account_snapshot_hash=facts['account_snapshot_hash'],
            L1=facts,L2=l2,quality_q_v2_valid=False,prediction_admitted=False,decision_admitted=False))
        from .fg_s3_quality_catalog_v2 import validate_corpus_label
        return validate_corpus_label(value)
    return label


def generate_behavior(factory, spec, output, dreamer_root, *, source, initial_state=None,
                      end_cursor=None, strict=None, policy_factory=Single3BehaviorPolicy,
                      labeler_factory=None):
    if source.get('fold')!='TRAIN':raise ValueError('Corpus requires explicit TRAIN source')
    env=factory()
    if env.entry_config.get('state_path'):
        raise ValueError('Corpus refuses a persistent state_path')
    teacher=None;labels=None
    try:
        if initial_state is not None:env.import_state(initial_state)
        else:env.reset(seed=0)
        teacher=NativeQualityTeacher(env,factory,fold='TRAIN',source=source)
        if spec['kind']=='quality_oracle_schedule':
            from .fg_s3_e_quality_strict import QualityP4Policy
            if strict is None:raise ValueError('P4 requires a complete Quality strict suffix')
            policy_factory=lambda value:QualityP4Policy(value,env=env,teacher=strict)
        labels=l1_labeler(teacher,strict)
        if labeler_factory is not None:
            labels=labeler_factory(teacher.identity_hash)
            teacher.close()
        trace=collect_trace('',spec,output,dict(protocol=SCHEMA,source=source,
            teacher_identity=teacher.identity_hash,training_admitted=False),dreamer_root,
            env_factory=lambda:env,protocol=E_QUALITY_PROTOCOL,policy_factory=policy_factory,
            initial_state=initial_state,end_cursor=end_cursor,labeler=labels)
        receipt=reconcile_trace(output,dreamer_root,env_factory=factory,schema=E_QUALITY_SCHEMA,label_verifier=labels)
        write_json(Path(output)/'reconciliation.json',signed(receipt))
        return trace,receipt
    finally:
        if labeler_factory is not None and labels is not None and hasattr(labels,'close'):labels.close()
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
        count=Counter({k:0 for k in ['replay_rows','advanced','valid_market_minutes','macd_valid_slots',
            'legal_add_opportunities','filled_add','voluntary_wait','forced_wait','nonexecuted_proposals',
            'strict_valid','l1_candidates','l1_outcome/TP5','l1_outcome/SL30','l1_outcome/OTHER_EXIT','l1_outcome/RIGHT_CENSORED',
            'executed_exit/TAKE_PROFIT','executed_exit/STOP_LOSS','executed_exit/OTHER']})
        closed_pnl=closed_fees=__import__('decimal').Decimal(0)
        segments={}
        numeric={k:[] for k in ['quality_score','tp_market_minutes',
            'mae_5','mae_15','mae_60','mae_240','mae_full','first_adverse10_market_minutes','underwater_minutes','underwater_longest_minutes','macd_dif','macd_signal','macd_hist',
            'raw_macd_dif','raw_macd_signal','raw_macd_hist']}
        for row,(_,step) in zip(iter_sidecar(path),
                               iter_replay_rows(path,manifest),strict=True):
            count['replay_rows']+=1;count['advanced']+=int(row['advanced'])
            count['valid_market_minutes']+=int(row['valid_bar'])
            count['macd_valid_slots']+=int(step['market_macd_5m'][3]==1)
            if step['market_macd_5m'][3]==1:
                for key,value in zip(('macd_dif','macd_signal','macd_hist'),step['market_macd_5m'][:3]):numeric[key].append(float(value))
                for key,value in zip(('raw_macd_dif','raw_macd_signal','raw_macd_hist'),row['e_snapshot']['macd_5m']['raw']):numeric[key].append(float(value))
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
            for event in row['exits']:
                count['executed_exit/'+event['reason']]+=1
                if event['reason'] not in ('TAKE_PROFIT','STOP_LOSS'):count['executed_exit/OTHER']+=1
                closed_pnl+=__import__('decimal').Decimal(event['net_trade_pnl'])
                closed_fees+=__import__('decimal').Decimal(event['open_fee'])+__import__('decimal').Decimal(event['close_fee'])
            label=row.get('teacher',{})
            from .fg_s3_quality_catalog_v2 import validate_corpus_label
            if label:validate_corpus_label(label)
            facts=label.get('L1',{})
            l2=label.get('L2',{})
            count['strict_valid']+=int(l2.get('teacher_valid',False))
            if l2.get('teacher_valid'):count['strict_action/'+str(l2['strict_action'])]+=1
            if facts.get('immediate_outcome_verified'):
                count['l1_candidates']+=1;count['l1_outcome/'+facts['exit_type']]+=1
                for key in numeric:
                    value=(facts.get('first_adverse',{}).get('10') if key=='first_adverse10_market_minutes' else
                        facts.get('score') if key=='quality_score' else facts.get('held_market_minutes') if key=='tp_market_minutes' and facts['exit_type']=='TP5' else facts.get(key) if key!='tp_market_minutes' else None)
                    if value is not None:numeric[key].append(value)
                count['underwater_full_valid']+=int(facts['underwater_full_valid'])
                for horizon,valid in zip((5,60,240),facts['mae_window_complete']):count[f'mae_{horizon}_complete']+=int(valid)
                count['mae_15_complete']+=int(facts['mae_15_complete'])
                count['l1_same_day_tp']+=int(facts['exit_type']=='TP5' and facts['crossed_trading_days']==0)
                count['l1_cross_day_tp']+=int(facts['exit_type']=='TP5' and facts['crossed_trading_days']>0)
                for horizon in (5,15):
                    count[f'early10_{horizon}_valid']+=int(facts[f'early10_{horizon}_valid'])
                    count[f'early10_{horizon}_positive']+=int(facts[f'early10_{horizon}'])
            # Key market segments use the already authenticated dated-session
            # identifier, not guessed wall-clock hours or outcome-driven bins.
            bucket=row['e_snapshot']['macd_5m'].get('bucket_id')
            session=bucket.rsplit(':',1)[0] if bucket else 'NO_COMPLETE_1M_YET'
            segment=segments.setdefault(str(row['trading_date'])+'|'+session,Counter())
            segment['replay_rows']+=1
            segment['advanced_market_minutes']+=int(row['valid_bar'])
            segment['legal_add_opportunities']+=int(executed and row['legal_add'])
            segment['filled_add']+=sum(f['reason']=='ENTRY' for f in row['fills'])
            segment['strict_valid']+=int(l2.get('teacher_valid',False))
            segment['l1_candidates']+=int(facts.get('immediate_outcome_verified',False))
            if facts.get('immediate_outcome_verified'):segment['l1_outcome/'+facts['exit_type']]+=1
            for event in row['exits']:segment['executed_exit/'+event['reason']]+=1
        count['executed_right_censored']=manifest['open_entries']
        reports[name]=dict(counts=dict(count),closed_net_pnl_CNY=str(closed_pnl),closed_fees_CNY=str(closed_fees),
            complete_exit_rate=(count['filled_add']-manifest['open_entries'])/count['filled_add'] if count['filled_add'] else None,distributions={k:distribution(v) for k,v in numeric.items()},
            units=dict(quality_score='0..100',tp_market_minutes='valid_market_minutes',mae='quoted_price_points',underwater='valid_market_minutes',macd='1000*raw/current_visible_close',raw_macd='quoted_price_points'),
            by_official_day_session={k:dict(v) for k,v in sorted(segments.items())},
            trace_hash=manifest['replay_trace_hash'])
    return signed(dict(protocol='FG_S3_E_QUALITY_COVERAGE_V2',behaviors=reports,
        unique_market_cursors=len(market),unique_advanced_market_minutes=len(advanced),
        unique_contract_official_days=len(days),distinct_account_decision_states=len(states),
        optimizer_exposures=0,teacher_labels_are_independent_market_samples=False,
        distribution_denominator='observed L1 candidates, not executed trades; incomplete prefix MAE is observed exposure'))


def coverage_csv(report):
    """A reviewable projection of the same recomputable JSON counts."""
    import csv
    import io
    with io.StringIO(newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['behavior','official_day_session','metric','value'])
        for behavior,body in sorted(report['behaviors'].items()):
            for metric,value in sorted(body['counts'].items()):writer.writerow([behavior,'ALL',metric,value])
            for segment,counts in body['by_official_day_session'].items():
                for metric,value in sorted(counts.items()):writer.writerow([behavior,segment,metric,value])
        return stream.getvalue()


def write_coverage_csv(path,report):
    with Path(path).open('w',newline='') as stream:stream.write(coverage_csv(report))
