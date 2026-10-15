"""Posthoc DEV quality scoring. Never constructs a TRAIN catalog or teacher."""
from __future__ import annotations

import json
from pathlib import Path

import embodied
from embodied.envs.from_gymnasium import FromGymnasium
import numpy as np

from embodied.run.fg_checkpoint import validate_bundle,file_hash
from .fg_replay import read_manifest,signed,write_json,check_files
from .fg_single3 import load_env
from .fg_single3_evaluation import require_dev
from .fg_s3_oracle_state_query import decision_state
from .fg_s3_quality_teacher_v2 import trace_native_trade
from gym_trading_env.envs.entry_contract import fingerprint

ROLE='DEV_EVALUATION_ONLY'


def quality_scores(rows,prior=None, *,role=ROLE):
    from .fg_s3_prediction_evaluation import calibration
    from scipy.stats import spearmanr,kendalltau
    out=dict(role=role,horizons={},score_samples=0,score_mae=None,p80_precision=None,p80_recall=None,
             score_spearman=None,score_kendall=None,high_score_missed=None,
             calibration_status='NOT_CALIBRATED',prior_scope='TRAIN_ONLY' if prior else 'TRAIN_PRIOR_UNAVAILABLE',
             q_c1_preference_accuracy=None,q_v2_regression_error=None,q_proof_samples=0)
    for h in range(7):
        matched=[r for r in rows if r['label'].get('event_valid',[False]*7)[h]]
        result=dict(samples=len(matched),nll=None,brier=None,train_prior_nll=None,train_prior_brier=None,sl_recall=None)
        if matched:
            p=np.asarray([r['prediction']['quality/outcome_probs'][h] for r in matched])
            if p.shape!=(len(matched),4) or not np.isfinite(p).all() or not np.allclose(p.sum(-1),1,atol=2e-5) or (p<0).any():
                raise ValueError('Sealed OutcomeHead probabilities are invalid')
            y=np.asarray([r['label']['event'][h] for r in matched],int)
            target=np.eye(4)[y];pred=p.argmax(-1)
            result.update(nll=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),
                brier=float(((p-target)**2).sum(-1).mean()),
                sl_recall=float((pred[y==1]==1).mean()) if (y==1).any() else None,
                calibration=calibration(p.max(-1),pred==y),
                class_counts=np.bincount(y,minlength=4).tolist())
            result['by_funds']={}
            for name,below in (('below_300000',True),('at_least_300000',False)):
                ix=np.asarray([(float(r['equity'])<300000)==below for r in matched])
                cy=y[ix];cp=p[ix]
                result['by_funds'][name]=dict(samples=int(ix.sum()),
                    nll=float(-np.log(np.clip(cp[np.arange(len(cy)),cy],1e-12,1)).mean()) if ix.any() else None,
                    brier=float(((cp-np.eye(4)[cy])**2).sum(-1).mean()) if ix.any() else None,
                    sl_recall=float((cp.argmax(-1)[cy==1]==1).mean()) if (cy==1).any() else None,
                    class_counts=np.bincount(cy,minlength=4).tolist())
            if prior:
                counts=np.asarray(prior['event_counts'][h],float)
                if counts.sum():
                    base=counts/counts.sum()
                    result.update(train_prior_nll=float(-np.log(np.clip(base[y],1e-12,1)).mean()),
                                  train_prior_brier=float(((base-target)**2).sum(-1).mean()))
        out['horizons'][str(h)]=result
    out['binary_windows']={}
    for kind,dimensions in (('touch',[(h,k) for h in range(4) for k in range(2)]),
                            ('adverse',[(h,) for h in range(4)])):
        for index in dimensions:
            def at(value):
                for i in index: value=value[i]
                return value
            matched=[r for r in rows if at(r['label'].get(kind+'_valid',[[False]*2]*4 if kind=='touch' else [False]*4))]
            result=dict(samples=len(matched),nll=None,brier=None,train_prior_nll=None,train_prior_brier=None)
            if matched:
                p=np.asarray([at(r['prediction']['quality/'+kind+'_probability']) for r in matched])
                y=np.asarray([at(r['label'][kind]) for r in matched])
                if not np.isfinite(p).all() or (p<0).any() or (p>1).any(): raise ValueError('Invalid binary probability')
                nll=lambda prob:float(-np.log(np.clip(np.where(y,prob,1-prob),1e-12,1)).mean())
                result.update(nll=nll(p),brier=float(((p-y)**2).mean()),calibration=calibration(p,y))
                if prior:
                    count=at(prior[kind+'_counts']);positive=at(prior[kind+'_positive'])
                    if count:
                        base=positive/count
                        result.update(train_prior_nll=nll(base),train_prior_brier=float(((base-y)**2).mean()))
            out['binary_windows'][kind+'_'+('_'.join(map(str,index)))]=result
    scored=[r for r in rows if r['label'].get('score_valid')]
    if scored:
        p=np.asarray([r['prediction']['quality/score_prediction'] for r in scored])
        y=np.asarray([r['label']['score'] for r in scored])
        if not np.isfinite(p).all() or (p<0).any() or (p>100).any():
            raise ValueError('Sealed quality scores must be finite and within 0..100')
        predicted,positive=p>=80,y>=80
        out.update(score_samples=len(y),score_mae=float(np.abs(p-y).mean()),
            p80_precision=float(positive[predicted].mean()) if predicted.any() else None,
            p80_recall=float(predicted[positive].mean()) if positive.any() else None,
            high_score_missed=float(np.mean([r.get('proposed_action',r['action'])==0
                for r in scored if r['label']['score']>=80])) if positive.any() else None)
        if prior and prior.get('score_median') is not None:
            out['score_train_prior_mae']=float(np.abs(y-prior['score_median']).mean())
        if len(y)>1 and np.ptp(y)>0 and np.ptp(p)>0:
            out.update(score_spearman=float(spearmanr(p,y).statistic),score_kendall=float(kendalltau(p,y).statistic))
        out['score_by_funds']={}
        for name,selection in (('below_300000',lambda r:float(r['equity'])<300000),
                               ('at_least_300000',lambda r:float(r['equity'])>=300000)):
            group=[r for r in scored if selection(r)]
            out['score_by_funds'][name]=dict(samples=len(group),mae=float(np.mean([
                abs(r['prediction']['quality/score_prediction']-r['label']['score']) for r in group])) if group else None)
        out['score_by_exit_type']={}
        for kind in ('TP5','SL30'):
            group=[r for r in scored if r['label'].get('exit_type')==kind]
            out['score_by_exit_type'][kind]=dict(samples=len(group),mae=float(np.mean([
                abs(r['prediction']['quality/score_prediction']-r['label']['score']) for r in group])) if group else None)
    out['decision_components']={}
    for key in ('delta_O','delta_S','delta_Q'):
        values=[r['prediction']['quality/'+key] for r in rows]
        changed=sum((r['prediction']['quality/z_final']>0)!=
                    (r['prediction']['quality/z_final']-r['prediction']['quality/'+key]>0) for r in rows)
        out['decision_components'][key]=dict(samples=len(values),mean=float(np.mean(values)) if values else None,
                                            leave_one_out_argmax_changes=changed)
    c1=[r for r in rows if r.get('q_proof',{}).get('q_c1_pair_valid')]
    v2=[r for r in rows if r.get('q_proof',{}).get('q_v2_pair_valid')]
    if c1:
        out['q_c1_preference_accuracy']=float(np.mean([
            (r['prediction']['quality/q_c1_prediction'][1]>r['prediction']['quality/q_c1_prediction'][0])
            ==bool(r['q_proof']['strict_action']) for r in c1]))
    if v2:
        out['q_v2_regression_error']=float(np.mean([np.mean((np.asarray(r['prediction']['quality/q_v2_prediction'])-
                                                       r['q_proof']['q_v2'])**2) for r in v2]))
    out.update(q_c1_proof_samples=len(c1),q_proof_samples=len(v2))
    return out


def same_state_inference(models,observations,executed_actions):
    """Reconstruct EACH model's causal carry from the same actual history."""
    from embodied.run.fg_s3_dagger_policy import replace_prevact
    if len(observations)!=len(executed_actions): raise ValueError('Common history lengths differ')
    result={}
    for name,model in models.items():
        carry=model.init_policy(1);rows=[]
        for obs,action in zip(observations,executed_actions):
            carry,proposal,diagnostics=model.policy(carry,obs,mode='eval')
            rows.append(dict(action=int(proposal['action'][0]),diagnostics={k:np.asarray(v).tolist()
                for k,v in diagnostics.items() if k.startswith(('quality/','tutor/','b2/')) or k=='p_add'}))
            carry=replace_prevact(carry,np.asarray([action],np.int32))
        result[name]=rows
    return result


def require_sealed_inference(eval_dir):
    source=Path(eval_dir).resolve()
    receipt=read_manifest(source/'receipt.json')
    fold=receipt.get('fold')
    actor=fold in ('DEV','TRAIN_HOLDOUT') and receipt.get('status')=='ACTOR_ONLY_'+fold+'_DIAGNOSTIC_COMPLETE'
    reference=(fold in ('DEV','TRAIN_HOLDOUT') and
               receipt.get('status')=='SEALED_REFERENCE_PREDICTIONS_COMPLETE' and
               receipt.get('autonomous_actor_performance') is False)
    if (not (actor or reference)
            or receipt['teacher_guidance'] or receipt['oracle_files_read'] or receipt['beta']!=0
            or receipt['policy_mode']!=('reference_on_sealed_history' if reference else 'eval_argmax')):
        raise ValueError('Quality scoring requires sealed evaluation-only inference')
    check_files({str(source/n):h for n,h in receipt['causal_prediction_files'].items()})
    node=Path(receipt['checkpoint'])
    identity=read_manifest(node.parent.parent/'run_identity.json')['identity']
    validate_bundle(node,identity)
    if file_hash(node/'manifest.json')!=receipt['checkpoint_manifest_sha256']:
        raise ValueError('Sealed quality Checkpoint changed')
    from .fg_s3_v2_evaluation import require_quality_evaluation_snapshot
    manifest_name=receipt.get('evaluation_manifest','dev_manifest.json')
    if Path(manifest_name).name!=manifest_name: raise ValueError('Evaluation manifest escaped output')
    dev=require_quality_evaluation_snapshot(source/manifest_name,identity=identity)
    if dev['role']!=fold: raise ValueError('Sealed inference fold differs from source snapshot')
    if reference:
        parent=Path(receipt['same_state_source'])
        if file_hash(parent/'receipt.json')!=receipt['same_state_source_receipt_sha256']:
            raise ValueError('Reference source inference changed')
        parent_receipt=read_manifest(parent/'receipt.json')
        check_files({str(parent/p):h for p,h in parent_receipt['causal_prediction_files'].items()})
        if parent_receipt['fold']!=fold: raise ValueError('Reference inference role changed')
    return receipt,dev,node,identity


def score_quality_predictions(eval_dir,output_dir):
    source,output=Path(eval_dir).resolve(),Path(output_dir).resolve()
    receipt,dev,node,identity=require_sealed_inference(source)
    role='TRAIN_HOLDOUT_EVALUATION_ONLY' if receipt['fold']=='TRAIN_HOLDOUT' else ROLE
    if output==source or output.is_relative_to(source): raise ValueError('Use a separate new posthoc output')
    records=[json.loads(s) for s in (source/'actions.jsonl').read_text().splitlines()]
    full_research=receipt['agent_extension']['quality_stage']=='CONTROLLED_FULL'
    if not records or any('quality' not in r for r in records): raise ValueError('Missing sealed quality predictions')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'prediction_seal.json',signed(dict(role=role,future_labels_generated=False,
        source_receipt_sha256=file_hash(source/'receipt.json'),source_files=receipt['causal_prediction_files'])))
    def replay(generate=False):
        env=load_env(dev['rows'][0],dev['frozen_inputs'])
        clone=load_env(dev['rows'][0],dev['frozen_inputs']) if generate else None
        engine=None
        if generate and full_research:
            from .fg_s3_quality_q_v2 import CertifiedQEngine
            engine=CertifiedQEngine(lambda:load_env(dev['rows'][0],dev['frozen_inputs']))
        wrapped=embodied.wrappers.CheckSpaces(embodied.wrappers.UnifyDtypes(FromGymnasium(env,reset_seed=0)))
        driver=embodied.Driver([lambda:wrapped],parallel=False)
        seen=[];scored=[]
        def policy(carry,obs): return carry,dict(action=np.asarray([records[len(seen)]['action']],np.int32)),{}
        def capture(step,worker):
            i=len(seen);r=records[i]
            if (env.current_step!=r['cursor'] or fingerprint(decision_state(env))!=r['decision_state_id']
                    or not np.array_equal(step['action_mask'],r['mask']) or int(step['action'])!=r['action']):
                raise ValueError('Sealed quality trajectory/account replay differs')
            seen.append(i)
            if generate and i+1<len(records) and all(r['mask']) and not r['is_last']:
                label=trace_native_trade(env,clone,day_complete=bool(dev['rows'][0]['calendar_complete']))
                label.update(role=role,train_join_allowed=False)
                scored.append(dict(action=r['action'],prediction=r['quality'],label=label,
                                   proposed_action=r.get('reference_proposal',r['action']),
                                   equity=str(env._calculate_equity()),state_id=r['decision_state_id'],
                                   q_proof=engine.solve(env) if engine is not None else {}))
        driver.on_step(capture)
        try:
            driver(policy,steps=len(records))
            if fingerprint(env.export_state())!=fingerprint(json.loads((source/'account_state.json').read_text())):
                raise ValueError('Sealed final quality account differs')
        finally:
            driver.close()
            if clone is not None: clone.close()
            if engine is not None: engine.close()
        return scored
    replay(False)  # Reconcile every real action BEFORE generating any future label.
    rows=replay(True)
    prior_path=source/'train_prediction_prior.json'
    prior=read_manifest(prior_path) if prior_path.is_file() else None
    if prior is not None and (prior.get('fold')!='TRAIN' or prior.get('checkpoint_manifest_sha256')!=receipt['checkpoint_manifest_sha256']):
        raise ValueError('Quality prior must come from the matched TRAIN Checkpoint')
    with (output/'labels_evaluation_only.jsonl').open('x') as f:
        for row in rows: f.write(json.dumps(row,allow_nan=False)+'\n')
    result=signed(dict(status='QUALITY_POSTHOC_'+receipt['fold']+'_COMPLETE',role=role,
        fold=receipt['fold'],contract=receipt['contract'],training_labels_created=False,
        metrics=quality_scores(rows,prior,role=role),source_receipt_sha256=file_hash(source/'receipt.json'),
        inference_dir=str(source),checkpoint=str(node),
        autonomous_actor_performance=receipt['status'].startswith('ACTOR_ONLY_'),
        inference_kind='same_state_reference' if receipt['status'].startswith('SEALED_REFERENCE_') else 'actor_only',
        checkpoint_manifest_sha256=receipt['checkpoint_manifest_sha256'],
        train_prior_sha256=file_hash(prior_path) if prior is not None else None,
        future_labels_generated_after_prediction_seal=True,rows=len(rows),
        files={name:file_hash(output/name) for name in ('labels_evaluation_only.jsonl','prediction_seal.json')}))
    write_json(output/'receipt.json',result)
    return result


def reference_predictions(eval_dir,checkpoint,output_dir, *,device='cpu'):
    """Own model/carry on one sealed causal tape; never a free-running return."""
    import pickle
    import yaml
    from embodied.run.fg_s3_source_transfer import make_agent
    from embodied.run.fg_s3_dagger_policy import replace_prevact
    from embodied.run.fg_s3_v2_train import execution_code
    from dreamerv3.fg_s3_v2_agent import S3V2Agent
    from .fg_single3_evaluation import ActorPolicy
    source,output,node=Path(eval_dir).resolve(),Path(output_dir).resolve(),Path(checkpoint).resolve()
    sealed,context,_,primary_identity=require_sealed_inference(source)
    if sealed['status'].startswith('SEALED_REFERENCE'): raise ValueError('Reference must anchor an autonomous sealed Actor tape')
    if sealed.get('policy_rng')!='frozen_config_seed_counter_zero':
        raise ValueError('Matched reference requires the declared common evaluation RNG protocol')
    if output==source or output.is_relative_to(source) or output.is_relative_to(node.parent.parent):
        raise ValueError('Reference output must be a separate new directory')
    identity=read_manifest(node.parent.parent/'run_identity.json')['identity']
    validate_bundle(node,identity)
    check_files(identity['immutable_files'])
    if identity['code']!=execution_code() or identity['mode']!='quality_v2':
        raise ValueError('Reference model code/architecture differs from audited quality inference')
    for member in (primary_identity,identity):
        if member['agent_extension']['quality_stage']!='HEADS_ONLY':
            raise ValueError('Initial matched reference requires auxiliary-only source models')
    origin=lambda root:read_manifest(root/'source_admission.json')['origin']['identity_hash']
    if origin(node.parent.parent)!=origin(Path(sealed['checkpoint']).parent.parent):
        raise ValueError('Matched prediction reference does not share the fixed source')
    if set(identity['quality_training_contracts'])!=set(primary_identity['quality_training_contracts']):
        raise ValueError('Matched reference quality TRAIN split differs')
    from .fg_s3_v2_evaluation import require_quality_evaluation_snapshot
    manifest_name=sealed.get('evaluation_manifest','dev_manifest.json')
    require_quality_evaluation_snapshot(source/manifest_name,identity=identity)
    cfg=yaml.safe_load((node.parent.parent/'config.yaml').read_text())
    if fingerprint(cfg)!=identity['model_config_sha256']: raise ValueError('Reference resolved model config changed')
    output.mkdir(parents=True,exist_ok=False)
    model=make_agent(cfg,output/'model',agent_cls=S3V2Agent,device=device,
        bc_lambda=identity['bc_lambda'],extra_agent_config=identity['agent_extension'],
        composition=identity.get('resolved_composition'))
    model.load_exact(pickle.loads((node/'agent.pkl').read_bytes()))
    model.sync_policy_from_train(reset_counter=True)
    policy_actor=ActorPolicy(model)
    tape=[json.loads(s) for s in (source/'actions.jsonl').read_text().splitlines()]
    env=load_env(context['rows'][0],context['frozen_inputs'])
    wrapped=embodied.wrappers.CheckSpaces(embodied.wrappers.UnifyDtypes(FromGymnasium(env,reset_seed=0)))
    driver=embodied.Driver([lambda:wrapped],parallel=False);driver.reset(model.init_policy)
    rows=[];last_proposal=None
    def policy(carry,obs):
        nonlocal last_proposal
        carry,proposal,_=policy_actor(carry,obs)
        last_proposal=int(proposal['action'][0])
        actual=np.asarray([tape[len(rows)]['action']],np.int32)
        return replace_prevact(carry,actual),dict(action=actual),{}
    def capture(step,worker):
        old=tape[len(rows)]
        if (env.current_step!=old['cursor'] or fingerprint(decision_state(env))!=old['decision_state_id'] or
                not np.array_equal(step['action_mask'],old['mask']) or int(step['action'])!=old['action']):
            raise ValueError('Reference causal history differs from the sealed real tape')
        rows.append({**old,'quality':policy_actor.last_diagnostics['quality'],
            'p_add':policy_actor.last_diagnostics['p_add'],'reference_proposal':last_proposal,
            'prediction_phase':'CAUSAL_REFERENCE_ON_SEALED_HISTORY'})
    driver.on_step(capture)
    try:
        driver(policy,steps=len(tape))
        if fingerprint(env.export_state())!=fingerprint(json.loads((source/'account_state.json').read_text())):
            raise ValueError('Reference final native account differs from the sealed tape')
    finally: driver.close()
    with (output/'actions.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    for name in ('account_state.json',manifest_name):
        (output/name).write_bytes((source/name).read_bytes())
    prior=read_manifest(source/'train_prediction_prior.json')
    prior={k:v for k,v in prior.items() if k!='content_hash'}
    prior.update(common_train_prior_sha256=file_hash(source/'train_prediction_prior.json'),
                 checkpoint_manifest_sha256=file_hash(node/'manifest.json'))
    write_json(output/'train_prediction_prior.json',signed(prior))
    names=('actions.jsonl','account_state.json',manifest_name,'train_prediction_prior.json')
    result=signed(dict(status='SEALED_REFERENCE_PREDICTIONS_COMPLETE',fold=sealed['fold'],
        contract=sealed['contract'],evaluation_manifest=manifest_name,output=str(output),
        checkpoint=str(node),checkpoint_manifest_sha256=file_hash(node/'manifest.json'),
        agent_extension=identity['agent_extension'],teacher_guidance=False,oracle_files_read=False,beta=0.,
        policy_mode='reference_on_sealed_history',autonomous_actor_performance=False,
        policy_rng='frozen_config_seed_counter_zero',
        actual_actions_source='sealed_primary_actor_tape; each model owns its causal carry',
        same_state_source=str(source),same_state_source_receipt_sha256=file_hash(source/'receipt.json'),
        causal_prediction_files={name:file_hash(output/name) for name in names},
        rows=len(rows),test='NOT_RUN'))
    write_json(output/'receipt.json',result)
    return result


def compare_prediction_receipts(primary_dir,reference_dir,output_dir):
    """Compare predictions against exactly the same sealed labels/states."""
    paths=[Path(p).resolve() for p in (primary_dir,reference_dir)]
    receipts=[];all_rows=[];priors=[];seals=[]
    for root in paths:
        receipt=read_manifest(root/'receipt.json')
        if receipt.get('training_labels_created') is not False or not receipt.get('future_labels_generated_after_prediction_seal'):
            raise ValueError('Comparison requires sealed posthoc evaluation-only labels')
        check_files({str(root/p):h for p,h in receipt['files'].items()})
        seal,_,_,_=require_sealed_inference(receipt['inference_dir'])
        if file_hash(Path(receipt['inference_dir'])/'receipt.json')!=receipt['source_receipt_sha256']:
            raise ValueError('Comparison inference binding changed')
        receipts.append(receipt);seals.append(seal)
        all_rows.append([json.loads(s) for s in (root/'labels_evaluation_only.jsonl').read_text().splitlines()])
        priors.append(read_manifest(Path(receipt['inference_dir'])/'train_prediction_prior.json'))
    primary,reference=receipts
    if (primary['fold']!=reference['fold'] or primary['contract']!=reference['contract'] or
            seals[1].get('same_state_source_receipt_sha256')!=primary['source_receipt_sha256'] or
            priors[0]['source_records_hash']!=priors[1]['source_records_hash']):
        raise ValueError('Reference is not matched to the same real tape and TRAIN prior')
    left,right=all_rows
    if len(left)!=len(right): raise ValueError('Matched prediction coverage differs')
    for a,b in zip(left,right):
        if a['state_id']!=b['state_id'] or a['action']!=b['action'] or fingerprint(a['label'])!=fingerprint(b['label']):
            raise ValueError('Matched prediction account, action, or native label differs')
    lm=quality_scores(left,priors[0],role=primary['role'])
    rm=quality_scores(right,priors[1],role=primary['role'])
    for key,value in lm['horizons'].items():
        value.update(reference_nll=rm['horizons'][key]['nll'],reference_brier=rm['horizons'][key]['brier'])
    for key,value in lm['binary_windows'].items():
        value.update(reference_nll=rm['binary_windows'][key]['nll'],reference_brier=rm['binary_windows'][key]['brier'])
    lm['score_reference_mae']=rm['score_mae']
    output=Path(output_dir).resolve()
    if any(output==p or output.is_relative_to(p) for p in paths):
        raise ValueError('Comparison output must be separate and new')
    output.mkdir(parents=True,exist_ok=False)
    comparison_files=[]
    for horizon,value in lm['horizons'].items():
        metrics={k:value.get(k) for k in ('nll','brier','train_prior_nll','train_prior_brier',
                                         'reference_nll','reference_brier')}
        metrics.update({k:lm.get(k) for k in ('score_mae','score_train_prior_mae','score_reference_mae')})
        if value['samples'] and lm['score_samples'] and all(v is not None for v in metrics.values()):
            path=output/f'horizon_{horizon}.json'
            write_json(path,signed(dict(protocol='FG_S3_QUALITY_MATCHED_COMPARISON_V1',
                fold=primary['fold'],contract=primary['contract'],samples=value['samples'],
                checkpoint_manifest_sha256=primary['checkpoint_manifest_sha256'],metrics=metrics,
                horizon=int(horizon),primary_scoring_dir=str(paths[0]),reference_scoring_dir=str(paths[1]),
                comparison_code_sha256=file_hash(__file__),
                source_receipts={str(p/'receipt.json'):file_hash(p/'receipt.json') for p in paths},
                same_state_labels=True,reference_carry='independent_model_on_real_executed_actions')))
            comparison_files.append(dict(evaluation_receipt=str(path),samples=value['samples'],metrics=metrics))
    result=signed(dict(status='QUALITY_MATCHED_PREDICTION_COMPARISON_COMPLETE',fold=primary['fold'],
        role=primary['role'],training_labels_created=False,matched_states=len(left),metrics=lm,
        checkpoint=primary['checkpoint'],checkpoint_manifest_sha256=primary['checkpoint_manifest_sha256'],
        reference_checkpoint_manifest_sha256=reference['checkpoint_manifest_sha256'],
        comparisons=comparison_files,prediction_admission='REVIEW_PENDING; no automatic Actor activation',
        files={p.name:file_hash(p) for p in output.iterdir() if p.is_file()},
        source_receipts={str(p/'receipt.json'):file_hash(p/'receipt.json') for p in paths}))
    write_json(output/'receipt.json',result)
    return result
