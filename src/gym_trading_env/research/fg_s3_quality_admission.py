"""Evidence gates frozen before training; no empirical thresholds are inferred.

Admission remeasures sealed records. A PASS string, a count, or an accurate
single-trade forecast cannot substitute for replicated support or action safety.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import pickle

import numpy as np

from embodied.run.fg_checkpoint import file_hash,validate_bundle
from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import read_manifest,check_files

POLICY='FG_S3_QUALITY_ADMISSION_POLICY_V2'
PREDICTION='FG_S3_QUALITY_PREDICTION_ADMISSION_V2'
DECISION='FG_S3_QUALITY_DECISION_ADMISSION_V1'
Q_ADMISSION='FG_S3_QUALITY_Q_ADMISSION_V2'


def _required(record,fields,gate):
    if any(record.get(k) is None for k in fields):
        raise ValueError(gate+'_PENDING: missing authenticated evidence fields')


def _number(value,name, *,minimum=None,maximum=None,integer=False):
    if (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
            or integer and type(value) is not int or minimum is not None and value<minimum
            or maximum is not None and value>maximum):
        raise ValueError('Invalid frozen admission threshold: '+name)
    return value


def validate_policy(policy):
    """Require explicit, versioned research choices; deliberately no defaults."""
    from .fg_reward_candidates import _official_day_key
    if policy.get('protocol')!=POLICY:
        raise ValueError('ADMISSION_POLICY_PENDING: an explicit V2 policy must precede training')
    for key in ('trained_contracts','heldout_contracts','replay_seeds','time_folds'):
        if not isinstance(policy.get(key),list) or not policy[key]:
            raise ValueError('Missing frozen admission plan: '+key)
    train,held=policy['trained_contracts'],policy['heldout_contracts']
    if (len(set(train))!=len(train) or len(set(held))!=len(held) or set(train)&set(held)
            or len(held)<2 or any(not isinstance(c,str) for c in train+held)):
        raise ValueError('Prediction admission needs multiple disjoint held-out contracts')
    seeds=policy['replay_seeds']
    if len(set(seeds))!=len(seeds) or len(seeds)<2 or any(type(s) is not int or s<0 for s in seeds):
        raise ValueError('Prediction admission needs independent training replay seeds')
    if policy.get('target_replay_seed') not in seeds:
        raise ValueError('The predictor selected for Actor influence must be chosen before training')
    folds=policy['time_folds'];names=[];windows={c:[] for c in [*held,'CZCE.FG409']}
    for fold in folds:
        names.append(fold['id'])
        if not isinstance(fold['id'],str) or not fold['id']:
            raise ValueError('Frozen time fold needs a stable name')
        for contract in windows:
            bounds=fold.get('windows',{}).get(contract,{})
            start,end=(_official_day_key(bounds.get(k)) for k in ('start_day','end_day'))
            if start>end:raise ValueError('Frozen time fold is reversed')
            windows[contract].append((start,end))
    if (len(folds)<2 or len(set(names))!=len(names) or
            any(a[1]>=b[0] for values in windows.values() for a,b in zip(sorted(values),sorted(values)[1:]))):
        raise ValueError('Admission needs multiple nonoverlapping official-day folds')
    prediction=policy.get('prediction',{})
    for key in ('minimum_samples_per_horizon','minimum_score_samples','minimum_binary_samples','minimum_official_days'):
        _number(prediction.get(key),key,minimum=2,integer=True)
    for key in ('minimum_tp_per_horizon','minimum_sl_per_horizon','minimum_low_funds_per_horizon'):
        _number(prediction.get(key),key,minimum=1,integer=True)
    for key in ('maximum_event_ece','maximum_binary_ece','minimum_sl_recall'):
        _number(prediction.get(key),key,minimum=0,maximum=1)
    decision=policy.get('decision',{})
    for key in ('minimum_changed_decisions','minimum_strict_add','minimum_strict_wait','minimum_closed_entries'):
        _number(decision.get(key),key,minimum=1,integer=True)
    # Non-increase limits can be stricter than zero; no positive risk budget
    # is silently introduced by an implementation default.
    for key in ('maximum_false_add_rate_delta','maximum_sl_rate_delta','maximum_drawdown_delta'):
        _number(decision.get(key),key,maximum=0)
    if decision.get('require_native_relative_value') is not True:
        raise ValueError('Decision admission requires native relative WAIT/ADD evidence')
    recipes=decision.get('component_sets',[])
    if not recipes or any(not r or r!=[k for k in ('outcome','score','q') if k in r] for r in recipes):
        raise ValueError('Freeze the exact diagnostic fusion recipes before training')
    q=policy.get('q',{})
    for key in ('minimum_verified_states','minimum_low_funds_states','minimum_official_days'):
        _number(q.get(key),key,minimum=2 if key!='minimum_low_funds_states' else 1,integer=True)
    _number(q.get('maximum_value_mse'),'maximum_value_mse',minimum=0)
    _number(q.get('minimum_advantage_accuracy'),'minimum_advantage_accuracy',minimum=0,maximum=1)
    return policy


def checkpoint_policy(node):
    """Policy is owned by the initial FullCheckpoint, not attached post hoc."""
    node=Path(node).resolve();meta=validate_bundle(node);identity=meta['identity']
    if meta['step']!=identity['env_steps']:
        raise ValueError('Admission requires the final predeclared Env budget, not a selected intermediate Checkpoint')
    if identity.get('mode')!='quality_v2' or identity.get('quality_score_policy')!='FG_S3_QUALITY_TP_MARKET_DAY_V1':
        raise ValueError('Admission Checkpoint uses a different quality protocol')
    binding=identity.get('quality_admission_policy')
    if not binding:raise ValueError('ADMISSION_POLICY_PENDING: Checkpoint predates a frozen policy')
    path=Path(binding['path'])
    if (path!=node.parent.parent/'quality_admission_policy.json' or
            identity['immutable_files'].get(str(path))!=binding['sha256'] or
            file_hash(path)!=binding['sha256']):
        raise ValueError('Admission policy is not an immutable initial Run input')
    initial=node.parent/'step_000000000000'
    validate_bundle(initial,identity)
    initial_counts=pickle.loads((initial/'v2_counters.pkl').read_bytes())
    if initial_counts['all_updates']!=0:raise ValueError('Admission policy was not frozen before learning')
    policy=validate_policy(read_manifest(path))
    if set(policy['trained_contracts'])!=set(identity['quality_training_contracts']):
        raise ValueError('Frozen admission TRAIN split differs from Checkpoint')
    return policy,binding['sha256'],meta


def _scored(path):
    from .fg_s3_quality_prediction_evaluation import require_sealed_inference
    root=Path(path).resolve();receipt=read_manifest(root/'receipt.json')
    if (receipt.get('training_labels_created') is not False or
            receipt.get('future_labels_generated_after_prediction_seal') is not True):
        raise ValueError('Admission requires sealed evaluation-only labels')
    check_files({str(root/p):h for p,h in receipt['files'].items()})
    sealed,context,node,identity=require_sealed_inference(receipt['inference_dir'])
    if file_hash(Path(receipt['inference_dir'])/'receipt.json')!=receipt['source_receipt_sha256']:
        raise ValueError('Scoring/inference binding changed')
    rows=[json.loads(s) for s in (root/'labels_evaluation_only.jsonl').read_text().splitlines()]
    if any(receipt.get(k)!=sealed.get(k) for k in ('fold','contract','checkpoint','checkpoint_manifest_sha256')):
        raise ValueError('Scoring receipt changed its sealed model or evaluation source')
    if len({r['state_id'] for r in rows})!=len(rows):
        raise ValueError('Evaluation duplicated native states to inflate support')
    tape=[json.loads(s) for s in (Path(receipt['inference_dir'])/'actions.jsonl').read_text().splitlines()]
    eligible=[r for r in tape[:-1] if all(r['mask']) and not r['is_last']]
    if len(rows)!=len(eligible) or any(
            r['state_id']!=t['decision_state_id'] or r['action']!=t['action'] or
            fingerprint(r['prediction'])!=fingerprint(t['quality']) or
            float(r['equity'])!=float(t['equity']) for r,t in zip(rows,eligible)):
        raise ValueError('Scoring rows differ from the sealed causal predictions/accounts')
    prior=read_manifest(Path(receipt['inference_dir'])/'train_prediction_prior.json')
    return receipt,sealed,context,node,identity,rows,prior


def matched_prediction_rows(primary,reference, *,policy=None,time_fold=None):
    """Authenticate both independent carries, then select a predeclared fold."""
    left,right=_scored(primary),_scored(reference)
    a,b=left[0],right[0]
    if (a['fold']!=b['fold'] or a['contract']!=b['contract'] or
            right[1].get('same_state_source_receipt_sha256')!=a['source_receipt_sha256'] or
            left[6]['source_records_hash']!=right[6]['source_records_hash'] or len(left[5])!=len(right[5])):
        raise ValueError('Prediction comparison requires the same real tape and TRAIN prior')
    for x,y in zip(left[5],right[5]):
        if x['state_id']!=y['state_id'] or x['action']!=y['action'] or fingerprint(x['label'])!=fingerprint(y['label']):
            raise ValueError('Matched native states/actions/labels differ')
    if time_fold is not None:
        if policy is None:raise ValueError('A time fold must be frozen before training')
        selected=[f for f in policy['time_folds'] if f['id']==time_fold]
        if len(selected)!=1:raise ValueError('Unknown frozen time fold')
        from .fg_reward_candidates import _official_day_key
        bounds=selected[0]['windows'][a['contract']]
        start,end=(_official_day_key(bounds[k]) for k in ('start_day','end_day'))
        keep=lambda r:start<=r['label']['entry_official_trading_date']<=end
        left=(*left[:5],[r for r in left[5] if keep(r)],left[6])
        right=(*right[:5],[r for r in right[5] if keep(r)],right[6])
    return left,right


def _forecast_support(rows,metrics,policy):
    p=policy['prediction']
    if (metrics['score_samples']<p['minimum_score_samples'] or
            len({r['label']['entry_official_trading_date'] for r in rows})<p['minimum_official_days']):
        raise ValueError('Insufficient score/official-day admission support')
    for h in range(7):
        m=metrics['horizons'][str(h)]
        if (m['samples']<p['minimum_samples_per_horizon'] or
                m.get('class_counts',[0]*4)[0]<p['minimum_tp_per_horizon'] or
                m.get('class_counts',[0]*4)[1]<p['minimum_sl_per_horizon'] or
                m.get('by_funds',{}).get('below_300000',{}).get('samples',0)<p['minimum_low_funds_per_horizon']):
            raise ValueError('Insufficient TP/SL/funds coverage for horizon '+str(h))
        if (m['calibration']['ece']>p['maximum_event_ece'] or m['sl_recall']<p['minimum_sl_recall']):
            raise ValueError('Outcome calibration/SL recall failed the frozen policy')
    for m in metrics['binary_windows'].values():
        if m['samples']<p['minimum_binary_samples'] or m['calibration']['ece']>p['maximum_binary_ece']:
            raise ValueError('Binary prediction support/calibration failed the frozen policy')


def _forecast_improved(a,b,name):
    values=[a.get(k) for k in ('nll','brier','train_prior_nll','train_prior_brier')]+[b.get('nll'),b.get('brier')]
    if (any(v is None or not math.isfinite(v) for v in values) or
            not (a['nll']<min(a['train_prior_nll'],b['nll']) and a['brier']<min(a['train_prior_brier'],b['brier']))):
        raise ValueError(name+' did not beat TRAIN prior and same-state reference in NLL/Brier')


def _improvements(left,right):
    for h in range(7):
        _forecast_improved(left['horizons'][str(h)],right['horizons'][str(h)],'Outcome horizon '+str(h))
    for name in ([f'touch_{h}_{k}' for h in range(4) for k in range(2)]+[f'adverse_{h}' for h in range(4)]):
        _forecast_improved(left['binary_windows'].get(name,{}),right['binary_windows'].get(name,{}),name)
    values=[left.get(k) for k in ('score_mae','score_train_prior_mae')]+[right.get('score_mae')]
    if any(v is None or not math.isfinite(v) for v in values) or not values[0]<min(values[1:]):
        raise ValueError('Score did not beat TRAIN prior and reference')


def _head_parameter_changes(node,names):
    from dreamerv3.fg_s3a_actor import parameter_digests
    node=Path(node);digests=[]
    prefixes=tuple(name+'/' for name in names)
    for checkpoint in (node.parent/'step_000000000000',node):
        params=pickle.loads((checkpoint/'agent.pkl').read_bytes())['params']
        digests.append(parameter_digests({k:v for k,v in params.items() if k.startswith(prefixes)}))
        del params
    for prefix in prefixes:
        keys={k for k in digests[0] if k.startswith(prefix)}
        if not keys or keys!={k for k in digests[1] if k.startswith(prefix)} or not any(
                digests[0][k]!=digests[1][k] for k in keys):
            raise ValueError('Admission head has no actual parameter change: '+prefix)


def _head_updates(node,identity):
    counts=pickle.loads((Path(node)/'v2_counters.pkl').read_bytes())
    ext=identity['agent_extension']
    if (ext['quality_stage']!='HEADS_ONLY' or ext['quality_outcome_lambda']<=0 or ext['quality_score_lambda']<=0
            or any(counts.get(k+'_updates',0)<1 for k in ('quality_event','quality_touch','quality_adverse','quality_score'))):
        raise ValueError('Prediction admission lacks actual auxiliary head updates')
    _head_parameter_changes(node,('quality_event','quality_touch','quality_adverse','quality_score'))


def require_prediction_admission(path):
    from .fg_s3_quality_prediction_evaluation import quality_scores
    admission=read_manifest(path)
    if admission.get('protocol')!=PREDICTION or admission.get('status')!='PASS_QUALITY_PREDICTION_ADMISSION':
        raise ValueError('Actor influence is PREDICTION_GATE_PENDING: V2 evidence required')
    _required(admission,('checkpoint','checkpoint_manifest_sha256','policy_sha256','evaluation_files','comparisons'),'PREDICTION_GATE')
    policy,policy_hash,meta=checkpoint_policy(admission['checkpoint'])
    if (admission.get('policy_sha256')!=policy_hash or
            file_hash(Path(admission['checkpoint'])/'manifest.json')!=admission['checkpoint_manifest_sha256'] or
            meta['identity']['mix_replay_seed']!=policy['target_replay_seed']):
        raise ValueError('Prediction admission changed its frozen policy/model selection')
    _head_updates(admission['checkpoint'],meta['identity'])
    check_files(admission['evaluation_files'])
    grid=set();dev_grid=set();replicas={}
    origin=read_manifest(Path(admission['checkpoint']).parent.parent/'source_admission.json')['origin']['identity_hash']
    for entry in admission['comparisons']:
        primary,reference=Path(entry['primary_scoring_dir']),Path(entry['reference_scoring_dir'])
        for root in (primary,reference):
            if admission['evaluation_files'].get(str(root/'receipt.json'))!=file_hash(root/'receipt.json'):
                raise ValueError('Unbound admission comparison')
        left,right=matched_prediction_rows(primary,reference,policy=policy,time_fold=entry['time_fold'])
        receipt,_,_,node,identity,rows,prior=left
        if (left[1]['policy_mode']!='eval_argmax' or receipt.get('autonomous_actor_performance') is not True
                or left[1].get('q_parameter_overlay')):
            raise ValueError('Forecast admission uses the predeclared unmodified Actor history')
        initial=Path(node).parent/'step_000000000000'
        if right[3]!=initial or right[4]!=identity:
            raise ValueError('Prediction reference must be this replica\'s actual initial Checkpoint')
        rp,rhash,_=checkpoint_policy(node)
        _head_updates(node,identity)
        seed=identity['mix_replay_seed'];contract=receipt['contract']
        if rhash!=policy_hash or rp!=policy or seed not in policy['replay_seeds']:
            raise ValueError('Replication was not in the frozen admission plan')
        model_origin=read_manifest(node.parent.parent/'source_admission.json')['origin']['identity_hash']
        target=meta['identity']
        if (model_origin!=origin or identity['agent_extension']!=target['agent_extension'] or
                identity['env_steps']!=target['env_steps'] or identity['model_config_sha256']!=target['model_config_sha256'] or
                any(identity['training_identity'][k]!=target['training_identity'][k] for k in
                    ('training_hash','reward_params_hash','observation_identity'))):
            raise ValueError('Replications changed the fixed model, reward or TRAIN source')
        previous=replicas.setdefault(seed,receipt['checkpoint_manifest_sha256'])
        if previous!=receipt['checkpoint_manifest_sha256']:raise ValueError('Post-hoc Checkpoint selection within one replication')
        lm=quality_scores(rows,prior,role=receipt['role']);rm=quality_scores(right[5],right[6],role=receipt['role'])
        _forecast_support(rows,lm,policy);_improvements(lm,rm)
        if receipt['fold']=='DEV':dev_grid.add((entry['time_fold'],seed))
        elif receipt['fold']=='TRAIN_HOLDOUT' and contract in policy['heldout_contracts']:
            for h in range(7):grid.add((contract,entry['time_fold'],seed,h))
        else:raise ValueError('Admission evidence escaped its held-out evaluation roles')
    expected={(c,f['id'],s,h) for c in policy['heldout_contracts'] for f in policy['time_folds']
              for s in policy['replay_seeds'] for h in range(7)}
    if grid!=expected or dev_grid!={(f['id'],s) for f in policy['time_folds'] for s in policy['replay_seeds']}:
        raise ValueError('Missing contract/time-fold/replication/horizon or independent DEV evidence')
    if len(set(replicas.values()))!=len(policy['replay_seeds']):
        raise ValueError('Replications reused one trained Checkpoint')
    if replicas[policy['target_replay_seed']]!=admission['checkpoint_manifest_sha256']:
        raise ValueError('The admitted predictor was not the model actually evaluated')
    return admission


def _window(policy,fold,contract):
    from .fg_reward_candidates import _official_day_key
    selected=[f for f in policy['time_folds'] if f['id']==fold]
    if len(selected)!=1:raise ValueError('Unknown predeclared action-safety time fold')
    bounds=selected[0]['windows'][contract]
    return tuple(_official_day_key(bounds[k]) for k in ('start_day','end_day'))


def relative_action_safety(tape,audits,components,window,limits,unimix):
    """Measure interventions on the same causal history and native L3 states."""
    from .fg_s3_quality_prediction_evaluation import diagnostic_action
    from .fg_reward_candidates import _official_day_key
    changed=strict_add=strict_wait=baseline_fp=candidate_fp=positive=0
    for r in tape[:-1]:
        if not window[0]<=_official_day_key(r['official_trading_date'])<=window[1]:continue
        if not all(r['mask']) or r['is_last']:continue
        proposed,_,_=diagnostic_action(r['quality'],r['mask'],components,unimix)
        proof=audits.get(r['decision_state_id'],{})
        if proof.get('valid'):
            strict_add+=proof['action']==1;strict_wait+=proof['action']==0
            baseline_fp+=r['action']==1 and proof['action']==0
            candidate_fp+=proposed==1 and proof['action']==0
        if proposed!=r['action']:
            changed+=1
            if not proof.get('valid') or proposed!=proof['action']:
                raise ValueError('Predicted quality intervention lacks native relative WAIT/ADD support')
            positive+=proposed==1
    if (changed<limits['minimum_changed_decisions'] or not positive or
            strict_add<limits['minimum_strict_add'] or strict_wait<limits['minimum_strict_wait']):
        raise ValueError('Insufficient independently verified action-direction support')
    if (candidate_fp-baseline_fp)/strict_wait>limits['maximum_false_add_rate_delta']:
        raise ValueError('Intervention increased false ADD on certified WAIT states')
    return dict(changed=changed,strict_add=strict_add,strict_wait=strict_wait,
                baseline_false_add=baseline_fp,candidate_false_add=candidate_fp)


def _account_risk(tape,state,context,window,minimum_entries):
    """Attribute BOTH SL and drawdown to complete trades entered in this fold.

    A trade closed after the fold uses its complete observed follow-up. An
    unresolved trade has no complete-cohort SL/drawdown estimate. The native
    single-position rule makes the first-entry through last-exit span contain
    only this entry cohort; any interleaving foreign entry is rejected.
    An unobserved fold cutoff is reported as unknown, separately from the
    right censoring of trades at the end of the sealed tape.
    """
    from .fg_single3 import load_env
    from .fg_s3_quality_teacher_v2 import official_day_index
    from gym_trading_env.envs.entry_contract import timestamp
    env=load_env(context['rows'][0],context['frozen_inputs'])
    days=official_day_index(env)['days']
    entries={f['entry_id']:f for f in state['fills'] if f['reason']=='ENTRY'}
    closed={e['entry_id']:e for e in state['attributions']}
    records={**state['entries'],**closed}
    # Native entry fills use bob timestamps; equity and exits use eob. The
    # authenticated account retains the physical entry cursor in BOTH open
    # positions and closed attributions, including the first bar of a session.
    cursors={}
    for key,f in entries.items():
        record=records.get(key,{})
        i=record.get('entry_cursor')
        if (type(i) is not int or not 0<=i<len(days) or env.bobs[i]!=timestamp(f['time'])):
            raise ValueError('Native entry lacks its authenticated fill-bar cursor')
        cursors[key]=i
    cohort={key:f for key,f in entries.items() if window[0]<=days[cursors[key]]<=window[1]}
    exits={key:closed[key] for key in cohort if key in closed}
    for e in exits.values():
        i=e['exit_cursor']
        if not 0<=i<len(days) or env._raw.index[i]!=timestamp(e['exit_time']):
            raise ValueError('Native exit differs from its authenticated market cursor')
    unresolved=sorted(set(cohort)-set(exits))
    postfold=sorted(key for key,e in exits.items() if days[e['exit_cursor']]>window[1])
    fold_rows=[i for i,day in enumerate(days) if window[0]<=day<=window[1]]
    cutoff_observed=bool(fold_rows and state['current_step']>=fold_rows[-1] and days[-1]>window[1])
    result=dict(risk_basis='ENTRY_FOLD_COMPLETE_TRADE_COHORT_V1',entry_fold=list(window),
        status='RIGHT_CENSORED' if unresolved else 'INSUFFICIENT_CLOSED_SUPPORT',
        entry_ids=sorted(cohort),closed_entries=len(exits),minimum_closed_entries=minimum_entries,
        fold_cutoff_observed=cutoff_observed,
        open_at_fold_end_entry_ids=sorted([*postfold,*unresolved]) if cutoff_observed else None,
        post_fold_complete_entry_ids=postfold,right_censored_entry_ids=unresolved,
        sl_rate=None,drawdown=None,equity_span=None)
    if not cohort or unresolved:return result
    first=min(cursors[key] for key in cohort)
    last=max(e['exit_cursor'] for e in exits.values())
    if any(first<=i<=last for key,i in cursors.items() if key not in cohort):
        raise ValueError('Non-cohort trade interleaves the complete entry-fold risk span')
    before=[r for r in tape if r['cursor']<first]
    span=[r for r in tape if first<=r['cursor']<=last]
    if not before or before[-1]['cursor']!=first-1:
        raise ValueError('Missing native equity anchor immediately before cohort entry')
    if not span or span[0]['cursor']!=first or span[-1]['cursor']!=last:
        raise ValueError('Complete trade follow-up lacks native equity through exit')
    equity=np.asarray([before[-1]['equity'],*[r['equity'] for r in span]],float)
    if not np.isfinite(equity).all():raise ValueError('Nonfinite native cohort equity')
    result.update(status='COMPLETE_COHORT' if len(exits)>=minimum_entries else 'INSUFFICIENT_CLOSED_SUPPORT',
        sl_rate=sum(e['reason']=='STOP_LOSS' for e in exits.values())/len(cohort),
        drawdown=float((np.maximum.accumulate(equity)-equity).max()),
        equity_span=dict(anchor_cursor=first-1,first_entry_cursor=first,last_exit_cursor=last,
            last_exit_official_day=days[last],samples=len(equity)))
    return result


def require_decision_admission(path):
    """A second, native action-safety gate; forecast accuracy alone is insufficient."""
    from .fg_s3_quality_prediction_evaluation import native_decision_audits,diagnostic_action
    from .fg_reward_candidates import _official_day_key
    admission=read_manifest(path)
    if admission.get('protocol')!=DECISION or admission.get('status')!='PASS_QUALITY_DECISION_ADMISSION':
        raise ValueError('DECISION_GATE_PENDING: measured relative-action evidence required')
    _required(admission,('prediction_admission','prediction_admission_sha256','components','policy_sha256',
        'checkpoint_manifest_sha256','evidence_files','comparisons'),'DECISION_GATE')
    prediction=require_prediction_admission(admission['prediction_admission'])
    if file_hash(admission['prediction_admission'])!=admission['prediction_admission_sha256']:
        raise ValueError('Decision admission changed its admitted prediction model')
    policy,policy_hash,_=checkpoint_policy(prediction['checkpoint'])
    components=admission['components']
    if (admission.get('policy_sha256')!=policy_hash or components not in policy['decision']['component_sets'] or
            admission['checkpoint_manifest_sha256']!=prediction['checkpoint_manifest_sha256']):
        raise ValueError('Action intervention differs from its frozen model/policy/recipe')
    if 'q' in components:
        _required(admission,('q_admission','q_admission_sha256'),'FULL_Q')
        q=require_q_admission(admission['q_admission'])
        if file_hash(admission['q_admission'])!=admission['q_admission_sha256']:
            raise ValueError('Action safety changed its admitted Q source')
    check_files(admission['evidence_files']);grid=set();dev_grid=set();cache={}
    evaluated_models={}
    for item in prediction['comparisons']:
        result=_scored(item['primary_scoring_dir'])
        evaluated_models[result[4]['mix_replay_seed']]=result[0]['checkpoint_manifest_sha256']
    d=policy['decision']
    for pair in admission['comparisons']:
        left,right=_scored(pair['baseline_scoring_dir']),_scored(pair['candidate_scoring_dir'])
        a,b=left[0],right[0]
        for root in (Path(pair['baseline_scoring_dir']),Path(pair['candidate_scoring_dir'])):
            if admission['evidence_files'].get(str(root/'receipt.json'))!=file_hash(root/'receipt.json'):
                raise ValueError('Unbound native action-safety evaluation')
        if (a['checkpoint_manifest_sha256']!=b['checkpoint_manifest_sha256'] or a['fold']!=b['fold'] or
                a['contract']!=b['contract'] or left[1]['policy_mode']!='eval_argmax' or
                right[1].get('diagnostic_components')!=components or
                right[1]['policy_mode']!='eval_diagnostic_quality_argmax'):
            raise ValueError('Action-safety baseline and candidate do not share the fixed model/recipe')
        _,phash,_=checkpoint_policy(left[3]);seed=left[4]['mix_replay_seed']
        if phash!=policy_hash or seed not in policy['replay_seeds']:
            raise ValueError('Unplanned action-safety replication')
        if evaluated_models.get(seed)!=a['checkpoint_manifest_sha256']:
            raise ValueError('Action safety did not evaluate the admitted prediction weights')
        for sealed in (left[1],right[1]):
            overlay=sealed.get('q_parameter_overlay')
            if 'q' in components:
                if not overlay or overlay['admission_sha256']!=admission['q_admission_sha256']:
                    raise ValueError('Action safety did not evaluate the admitted numerical Q weights')
            elif overlay:
                raise ValueError('Unplanned Q parameter intervention')
        window=_window(policy,pair['time_fold'],a['contract'])
        for inference in (a['inference_dir'],b['inference_dir']):
            if inference not in cache:cache[inference]=native_decision_audits(inference)
        audits,tape,state,context,_=cache[a['inference_dir']]
        other=cache[b['inference_dir']]
        cfg=read_manifest(left[3].parent.parent/'source_admission.json')['target_config']
        unimix=cfg['agent']['policy']['unimix']
        relative_action_safety(tape,audits,components,window,d,unimix)
        risk0=_account_risk(tape,state,context,window,d['minimum_closed_entries'])
        risk1=_account_risk(other[1],other[2],other[3],window,d['minimum_closed_entries'])
        if any(r['status']!='COMPLETE_COHORT' for r in (risk0,risk1)):
            raise ValueError('DECISION_GATE_PENDING: incomplete entry-fold trade support ('+
                             risk0['status']+', '+risk1['status']+')')
        if (risk1['sl_rate']-risk0['sl_rate']>d['maximum_sl_rate_delta'] or
                risk1['drawdown']-risk0['drawdown']>d['maximum_drawdown_delta']):
            raise ValueError('Native closed-loop SL/drawdown safety failed the frozen policy')
        if a['fold']=='DEV':dev_grid.add((pair['time_fold'],seed))
        elif a['fold']=='TRAIN_HOLDOUT' and a['contract'] in policy['heldout_contracts']:
            grid.add((a['contract'],pair['time_fold'],seed))
        else:raise ValueError('Action-safety evidence escaped its frozen held-out roles')
    if (grid!={(c,f['id'],s) for c in policy['heldout_contracts'] for f in policy['time_folds'] for s in policy['replay_seeds']} or
            dev_grid!={(f['id'],s) for f in policy['time_folds'] for s in policy['replay_seeds']}):
        raise ValueError('Incomplete native action-safety contract/time-fold/replication evidence')
    return admission


def admitted_q_overlay(path,target_node,target_params):
    """Verified Q weights in a private evaluation model, before fusion admission.

    HEADS_ONLY keeps Q frozen. This read-only transfer makes Q decision
    diagnostics possible without requiring a decision PASS before measuring it.
    """
    from dreamerv3.fg_s3a_actor import parameter_digests
    from embodied.run.fg_s3_source_transfer import admitted_auxiliary_parameters
    admission=require_q_admission(path)
    _,policy_hash,_=checkpoint_policy(target_node)
    if admission['policy_sha256']!=policy_hash:
        raise ValueError('Diagnostic Q source changed the pretraining policy')
    node=Path(admission['checkpoint'])
    source=pickle.loads((node/'agent.pkl').read_bytes())['params']
    base=('enc/','dyn/','dec/','rew/','con/','avail/','pol/','val/')
    select=lambda params:parameter_digests({k:v for k,v in params.items() if k.startswith(base)})
    if select(source)!=select(target_params):
        raise ValueError('Diagnostic Q features differ from the frozen prediction model')
    prefixes=('quality_q_c1/','quality_q_v2/')
    digests=parameter_digests({k:v for k,v in source.items() if k.startswith(prefixes)})
    copied=admitted_auxiliary_parameters(target_params,source,digests,prefixes=prefixes)
    binding=dict(admission_receipt=str(Path(path).resolve()),admission_sha256=file_hash(path),
        checkpoint_manifest_sha256=admission['checkpoint_manifest_sha256'],head_param_digests=digests,
        frozen_checkpoint_modified=False,scope='private_evaluation_model_only')
    return binding,copied


def verify_native_q(proof,factory, *,state_id,snapshot_hash=None):
    """Repeat the complete bounded financial DP in an independent native Env."""
    from .fg_s3_quality_q_v2 import CertifiedQEngine,PROTOCOL
    from .fg_s3_oracle_state_query import decision_state
    if (proof.get('protocol')!=PROTOCOL or proof.get('status')!='Q_VALID' or
            not proof.get('q_c1_pair_valid') or not proof.get('q_v2_pair_valid') or
            not proof.get('root_snapshot') or not proof.get('engine')):
        raise ValueError('Q admission requires complete native state and paired proof')
    root=proof['root_snapshot'];engine_id=proof['engine']
    if (fingerprint(root)!=proof['root_snapshot_hash'] or
            snapshot_hash is not None and proof['root_snapshot_hash']!=snapshot_hash):
        raise ValueError('Q proof account snapshot differs from its native owner')
    if fingerprint(proof.get('transition_witness'))!=proof.get('transition_witness_hash'):
        raise ValueError('Q proof transition witness bytes changed')
    engine=CertifiedQEngine(factory,max_states=engine_id['max_states'],
        max_seconds=engine_id['max_seconds'],max_suffix_bars=engine_id['max_suffix_bars'])
    actual=factory()
    try:
        if engine.identity!=engine_id or fingerprint(engine_id)!=proof['engine_identity']:
            raise ValueError('Q reward/utility/rules/code proof identity changed')
        actual.import_state(root)
        if fingerprint(decision_state(actual))!=state_id or proof['state_id']!=state_id:
            raise ValueError('Q proof does not own the exact account decision')
        repeated=engine.solve(actual)
        keys=('status','q_c1','q_v2','q_c1_pair_valid','q_v2_pair_valid',
              'q_c1_rational_witness','q_v2_rational_witness','strict_action','strict_tie',
              'transition_witness_hash')
        if any(repeated.get(k)!=proof.get(k) for k in keys):
            raise ValueError('Independent native Q replay did not reproduce its exact witness')
    finally:
        engine.close()  # No Env.close(): a configured state_path must not be written.
    return repeated


def require_q_admission(path):
    """Authenticate training coverage, NN updates, native proofs and holdout error."""
    from .fg_s3_quality_catalog_v2 import validate_catalog
    from .fg_single3 import load_env
    admission=read_manifest(path)
    if admission.get('protocol')!=Q_ADMISSION or admission.get('status')!='PASS_TRAIN_Q_FUSION_ADMISSION':
        raise ValueError('FULL_Q_PENDING: V2 native proof evidence required')
    _required(admission,('checkpoint','checkpoint_manifest_sha256','policy_sha256','evidence_files','catalogs','holdout_evaluations'),'FULL_Q')
    policy,policy_hash,meta=checkpoint_policy(admission['checkpoint'])
    node=Path(admission['checkpoint']);identity=meta['identity']
    if (policy_hash!=admission.get('policy_sha256') or
            file_hash(node/'manifest.json')!=admission['checkpoint_manifest_sha256']):
        raise ValueError('Q admission Checkpoint/policy binding differs')
    counts=pickle.loads((node/'v2_counters.pkl').read_bytes())
    ext=identity['agent_extension']
    if any(counts.get(k+'_updates',0)<1 or ext.get(k+'_lambda',0)<=0 for k in ('quality_q_c1','quality_q_v2')):
        raise ValueError('Q admission has no actual supported Q-head parameter updates')
    _head_parameter_changes(node,('quality_q_c1','quality_q_v2'))
    check_files(admission['evidence_files'])
    seen=set();contracts=set();days=set();low=0
    for path in admission['catalogs']:
        if admission['evidence_files'].get(str(Path(path)))!=file_hash(path):
            raise ValueError('Unbound Q training Catalog')
        validated=validate_catalog(path);catalog=read_manifest(path);owner=catalog['identity']
        if owner.get('checkpoint_manifest_sha256')!=admission['checkpoint_manifest_sha256']:
            raise ValueError('Q training support is not bound to the admitted model')
        pool=read_manifest(Path(owner['prepared_dir'])/'quality_pool.json')
        source_rows=read_manifest(Path(owner['prepared_dir'])/'train_pool.json')['files']
        mapping={r['contract']:r for r in source_rows}
        raw=Path(owner['journal']).read_bytes()[:owner['journal_prefix_bytes']]
        for row in map(json.loads,raw.splitlines()):
            label=row['quality'];q=label.get('q',{})
            if not q.get('q_v2_pair_valid'):continue
            key=(row['contract'],row['state_id'])
            if key in seen:raise ValueError('Duplicated Q states cannot inflate proof coverage')
            seen.add(key);contract=row['contract']
            factory=lambda row=mapping[contract],files=pool['frozen']:load_env(row,files)
            verify_native_q(q,factory,state_id=row['state_id'],snapshot_hash=label['account_snapshot_hash'])
            contracts.add(contract);days.add(label['entry_official_trading_date'])
            low+=float(row['equity_before'])<300000
    qpolicy=policy['q']
    if (len(seen)<qpolicy['minimum_verified_states'] or len(contracts)<2 or
            len(days)<qpolicy['minimum_official_days'] or low<qpolicy['minimum_low_funds_states']):
        raise ValueError('FULL_Q_PENDING: insufficient independently verified TRAIN Q coverage')
    evaluated=set()
    for item in admission['holdout_evaluations']:
        root=Path(item['scoring_dir'])
        if admission['evidence_files'].get(str(root/'receipt.json'))!=file_hash(root/'receipt.json'):
            raise ValueError('Unbound held-out Q evaluation')
        receipt,_,context,_,_,rows,_=_scored(root)
        if receipt['fold']!='TRAIN_HOLDOUT' or receipt['checkpoint_manifest_sha256']!=admission['checkpoint_manifest_sha256']:
            raise ValueError('Q error was not measured on the admitted held-out model')
        window=_window(policy,item['time_fold'],receipt['contract'])
        good=[r for r in rows if r.get('q_proof',{}).get('q_v2_pair_valid') and
              window[0]<=r['label']['entry_official_trading_date']<=window[1]]
        if (len(good)<qpolicy['minimum_verified_states'] or
                len({r['state_id'] for r in good})!=len(good) or
                len({r['label']['entry_official_trading_date'] for r in good})<qpolicy['minimum_official_days'] or
                sum(float(r['equity'])<300000 for r in good)<qpolicy['minimum_low_funds_states']):
            raise ValueError('FULL_Q_PENDING: insufficient held-out numerical Q support')
        factory=lambda:load_env(context['rows'][0],context['frozen_inputs'])
        for row in good:verify_native_q(row['q_proof'],factory,state_id=row['state_id'],snapshot_hash=row['label']['account_snapshot_hash'])
        for name in ('q_c1','q_v2'):
            target=np.asarray([r['q_proof'][name] for r in good])
            predicted=np.asarray([r['prediction']['quality/'+name+'_prediction'] for r in good])
            if predicted.shape!=target.shape or not np.isfinite(predicted).all():
                raise ValueError('Invalid sealed Q predictions')
            error=float(((predicted-target)**2).mean())
            accuracy=float(((predicted[:,1]>predicted[:,0])==(target[:,1]>target[:,0])).mean())
            if error>qpolicy['maximum_value_mse'] or accuracy<qpolicy['minimum_advantage_accuracy']:
                raise ValueError('Held-out '+name+' errors failed the frozen policy')
        evaluated.add((receipt['contract'],item['time_fold']))
    if evaluated!={(c,f['id']) for c in policy['heldout_contracts'] for f in policy['time_folds']}:
        raise ValueError('Q admission lacks every frozen held-out contract/time fold')
    return admission
