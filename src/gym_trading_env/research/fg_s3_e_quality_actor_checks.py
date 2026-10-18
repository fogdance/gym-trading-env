"""F3 engineering diagnostics; never grant formal training/research authority."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def score_batch(data,result,start):
    from .fg_s3_prediction_evaluation import calibration
    y=data['teacher_action'][:,start:];mask=data['action_mask'][:,start:]
    valid=(data['teacher_valid'][:,start:] & np.isin(y,[0,1]) & (mask==1).all(-1)
        & ~data['is_first'][:,start:] & ~data['is_last'][:,start:] & ~data['is_terminal'][:,start:])
    p=result['p_add'][valid];target=y[valid];pred=p>.5
    metrics={k:np.asarray(v).tolist() for k,v in result['metrics'].items()}
    return dict(loss=float(result['loss']),metrics=metrics,
        p_add_quantiles=None if not len(p) else np.quantile(p,[0,.1,.25,.5,.75,.9,1]).tolist(),
        p_add_histogram=np.histogram(p,bins=np.linspace(0,1,11))[0].tolist(),
        calibration=None if not len(p) else calibration(p,target),
        confusion=dict(tp=int(((target==1)&pred).sum()),fn=int(((target==1)&~pred).sum()),
            fp=int(((target==0)&pred).sum()),tn=int(((target==0)&~pred).sum())),
        constant_wait_accuracy=None if not len(p) else float((target==0).mean()),
        valid_l2_count=int(valid.sum()),valid_add=int((target==1).sum()),valid_wait=int((target==0).sum()),
        evaluation_scope='NATURAL_ELIGIBLE_ACTOR_STRICT_TARGET_STATES',research_admitted=False)


def evaluate_native(agent,factory,output,*,steps=32,initial_state=None,scope='DEV_ONLY',source=None):
    """Original ActorPolicy→Driver→Env→Replay and complete native reconciler."""
    from .fg_single3_evaluation import ActorPolicy,embodied_runtime
    from .fg_replay import collect_trace,reconcile_trace,E_QUALITY_PROTOCOL,E_QUALITY_SCHEMA,iter_sidecar,write_json,signed
    from .fg_s3_e_quality_foundation import PROJECT
    from dreamerv3.fg_s3_e_quality_actor import PROTOCOL
    from dreamerv3.fg_s3a_actor import parameter_digests
    if type(steps) is not int or not 1<=steps<=512:raise ValueError('F3_BOUNDED_EVALUATION_STEPS_REQUIRED')
    if agent.config.get('quality_f3_protocol')!=PROTOCOL:raise ValueError('F3_ACTOR_CHECKPOINT_REQUIRED')
    if any(agent.model.quality[k] for k in ('quality_decision_admitted','quality_prediction_admitted','quality_q_admitted','quality_fuse_score','quality_fuse_outcome')):
        raise ValueError('F3_TEACHER_FREE_POLICY_REQUIRED')
    out=Path(output)
    if out.exists():raise ValueError('F3_EVALUATION_NEW_OUTPUT_REQUIRED')
    env=factory()
    try:
        start=env.current_step if initial_state is None else initial_state['current_step']
        end=min(start+steps,len(env._raw)-1)
    finally:env.close()
    before=parameter_digests(agent.params);agent.sync_policy_from_train(reset_counter=True)
    spec=dict(id='F3_ACTOR_ARGMAX',kind='STUDENT_ONLY',deterministic=True)
    trace=collect_trace(None,spec,out,dict(protocol=PROTOCOL,source=source,scope=scope),PROJECT.parent/'dreamerv3',
        env_factory=factory,protocol=E_QUALITY_PROTOCOL,policy_factory=lambda _:ActorPolicy(agent),
        runtime_factory=embodied_runtime,initial_state=initial_state,end_cursor=end,labeler=None)
    replay=reconcile_trace(out,PROJECT.parent/'dreamerv3',env_factory=factory,schema=E_QUALITY_SCHEMA,
        runtime_factory=embodied_runtime,label_verifier=None)
    rows=list(iter_sidecar(out));state=json.loads((out/'account_state.json').read_text())
    if any('teacher' in r for r in rows):raise ValueError('F3_EVALUATION_TEACHER_FOUND')
    decisions=[r for r in rows if r['decision_executed'] and r['legal_add'] and not r['is_last']]
    padd=[r['policy_diagnostics']['p_add'] for r in decisions]
    for row in rows:
        d=row['policy_diagnostics']['quality']
        if any(d[k]!=0 for k in ('quality/delta_O','quality/delta_S','quality/delta_Q')):raise ValueError('F3_EVALUATION_FUSION_FOUND')
        if not row['is_last'] and row['action']!=d['quality/argmax_action']:raise ValueError('F3_EVALUATION_NOT_ARGMAX')
    if before!=parameter_digests(agent.params):raise ValueError('F3_EVALUATION_MODIFIED_PARAMETERS')
    entries=[f for f in state['fills'] if f['reason']=='ENTRY' and f.get('committed',True)]
    result=dict(status='F3_ACTOR_ONLY_ENGINEERING_DIAGNOSTIC',scope=scope,real_TRAIN_updates=0,
        training_admitted=False,research_admitted=False,teacher_used=False,prediction_fusion=False,
        original_driver=True,complete_account_replay=replay,trace_identity=trace['replay_trace_hash'],
        native_reward_schema=trace['reward_version'],market_advance=trace['valid_minutes'],
        executed_legal_decisions=len(decisions),add_proposals=sum(r['action']==1 for r in decisions),
        wait_proposals=sum(r['action']==0 for r in decisions),unexecuted_tail_proposal=int(rows[-1]['action']),
        actual_entry_fills=len(entries),native_tp_exits=trace['tp_events'],native_sl_exits=trace['sl_events'],
        native_total_reward=trace['total_reward'],p_add_quantiles=None if not padd else np.quantile(padd,[0,.25,.5,.75,1]).tolist(),
        all_parameter_bytes_equal=True,before=before,after=parameter_digests(agent.params))
    write_json(out/'actor_evaluation.json',signed(result));return result


def _output(path):
    from .fg_s3_e_quality_course_preflight import _local_path
    from .fg_s3_e_quality_foundation import PROJECT
    out=_local_path(path)
    if out.exists() or not any(out.is_relative_to(p) for p in (PROJECT/'docs/receipts/wp06',PROJECT/'artifacts/fg_s3_v2/e_quality_wp06')):
        raise ValueError('F3_DIAGNOSTIC_NEW_ISOLATED_OUTPUT_REQUIRED')
    return out


def probe(output,*,device='cpu'):
    """Fixed bounded synthetic source; no editable real-source bypass."""
    import sys
    from .fg_s3_e_quality_foundation import PROJECT
    from .fg_s3_e_quality_actor import validate_settings,preflight
    from .fg_s3_e_quality_reader import build_join_index
    from .fg_replay import write_json,signed
    from embodied.run.fg_s3_e_quality_actor import ActorRun,engineering_permit
    from dreamerv3.fg_s3_e_quality_actor import LOSS_DEFAULTS
    from dreamerv3.fg_s3a_actor import parameter_digests
    out=_output(output)
    if device not in ('cpu','gpu'):raise ValueError('F3_PROBE_DEVICE_INVALID')
    sys.path.insert(0,str(PROJECT/'tests/entry_v1'))
    from wp04_fixture import synthetic_foundation
    # The existing Corpus builder is CPU-only and deliberately hides CUDA.
    # Build its F1 evidence on CPU, then explicitly select the target device.
    s=synthetic_foundation(out/'source',multiscale=True,device='cpu');origin=Path(s['run_dir'])
    if device=='gpu':
        import os
        os.environ['JAX_PLATFORMS']='cuda';os.environ.pop('CUDA_VISIBLE_DEVICES',None)
    s.update(schema='FG_S3_E_QUALITY_F3_CONFIG_V1',stage='ACTOR_SUPERVISED',sampling_stage='ACTOR_STRICT',
        source_stage='F1',source_run=str(origin),source_checkpoint=str(origin/'ckpt/step_000000000004'),
        actor_loss=dict(LOSS_DEFAULTS),aux_sampling_spec=None,max_consumed_batches=40,
        prepared_dir=str(out/'prepared'),run_dir=str(out/'run'),target_steps=8,burnin_steps=4,device=device)
    p=preflight(validate_settings(s));index=out/'join.sqlite';build_join_index(p.source,index)
    run=ActorRun(p,index,out/'run',_probe=engineering_permit(p));before=parameter_digests(run.agent.params)
    result=run.run();after=parameter_digests(run.agent.params)
    frozen=[k for k,v in run.agent.model.f3_state_owner.items() if v is None]
    if not all(before[k]==after[k] for k in frozen):raise AssertionError('F3_PROBE_FROZEN_STATE_CHANGED')
    import jax
    result.update(scope='BUILTIN_SYNTHETIC_ENGINEERING_ONLY',device=device,execution_devices=[str(d) for d in jax.devices()],
        real_TRAIN_updates=0,before=before,after=after,frozen_keys=frozen,frozen_byte_equal=True,
        actor_parameter_changes=sum(before[k]!=after[k] for k in before if k.startswith('quality_actor/')))
    write_json(out/'probe.json',signed(result));return result


def read_check(config,output,*,purpose='development',producer_gym=None):
    """Fresh CPU inference; does not consume source F1/F2 parameters or train."""
    from .fg_s3_e_quality_actor import read_settings,create_agent,F3WindowAdapter,model_config
    from .fg_s3_e_quality_supervision import SupervisionSource,QualitySupervisionAdmission
    from .fg_s3_e_quality_reader import build_join_index,EQualityCourseReader
    from .fg_replay import write_json,signed
    from dreamerv3.fg_s3_e_quality_actor import STAGE,read_only_forward
    from dreamerv3.fg_s3a_actor import parameter_digests
    from gym_trading_env.envs.entry_contract import fingerprint
    s=read_settings(config);out=_output(output)
    if s['device']!='cpu':raise ValueError('F3_REAL_READ_CHECK_CPU_ONLY')
    source=QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(s['corpus_root'],stage=STAGE),
        purpose=purpose,producer_gym=producer_gym,log_path=out/'native.log' if purpose=='native-verification' else None)
    index=out/'join.sqlite';joined=build_join_index(source,index)
    # This diagnostic has one Natural Reader at a time, even with an auxiliary train spec.
    diagnostic={**s,'aux_sampling_spec':None}
    agent=create_agent(diagnostic,out/'model',controlled_small=True);before=parameter_digests(agent.params)
    reports={};market=set();accounts=set();valid=0
    for policy in sorted(joined['policies']):
        spec=dict(mode='NATURAL',policies=[policy])
        reader=EQualityCourseReader.open(source,stage='ACTOR_STRICT',sampling_spec=spec,seed=s['model_init']['seed'],index_path=index)
        try:
            adapter=F3WindowAdapter(reader,{**diagnostic,'sampling_spec':spec})
            try:adapter.validate_support()
            except ValueError as error:
                if not str(error).startswith('INSUFFICIENT_CURRICULUM_SUPPORT'):raise
                reports[policy]=dict(status='NOT_RUN',reason=str(error),coverage=reader.coverage());continue
            windows=[]
            for _ in range(2):
                data,metadata=adapter.sample();result=read_only_forward(agent,data)
                if not np.isfinite(result['loss']) or not np.isfinite(result['p_add']).all():raise FloatingPointError('F3_READ_CHECK_NONFINITE')
                score=score_batch(data,result,1+s['burnin_steps']);windows.append(score);valid+=score['valid_l2_count']
                for window in metadata:
                    for row in window['rows'][1+s['burnin_steps']:]:
                        market.add((window['contract'],row['cursor']));accounts.add((row['policy'],row['stepid']))
            reports[policy]=dict(status='READ_ONLY_LIMITED',windows=windows,coverage=reader.coverage())
        finally:reader.close()
    if before!=parameter_digests(agent.params) or int(agent.n_updates)!=0:raise AssertionError('F3_READ_CHECK_MODIFIED_STATE')
    result=dict(status='F3_TRAIN_CORPUS_READ_ONLY_LIMITED' if valid else 'INSUFFICIENT_CURRICULUM_SUPPORT',
        source=source.receipt,course_identity=source.source.course_identity_hash,corpus_identity=source.source.corpus_content_hash,
        model_hash=fingerprint(model_config(diagnostic,controlled_small=True)),source_model_consumed=False,
        model_origin='FRESH_CONTROLLED_DIAGNOSTIC',policies=reports,valid_l2_exposures=valid,
        unique_target_market_cursors=len(market),target_account_rows=len(accounts),all_parameter_bytes_equal=True,
        before=before,after=parameter_digests(agent.params),real_TRAIN_updates=0,actual_optimizer_updates=0,
        training_admitted=False,research_admitted=False,foundation_status='FOUNDATION_NOT_TRAINED',
        blockers=['REAL_TRAINED_F1_OR_F2_NOT_PROVIDED','INDEPENDENT_RESEARCH_EXAM_NOT_RUN'],
        input_geometry=dict(prefix=1,burnin_steps=s['burnin_steps'],target_steps=s['target_steps']),
        data_reads='IMMUTABLE_TRAIN_CORPUS_ONLY')
    write_json(out/'read_check.json',signed(result))
    if not valid:raise ValueError('INSUFFICIENT_CURRICULUM_SUPPORT: F3_READ_CHECK_NO_L2')
    return result


def evaluate_checkpoint(run_dir,checkpoint,output,*,steps=32):
    import pickle,copy
    from .fg_replay import read_manifest,digest
    from .fg_s3_e_quality_actor import validate_settings,preflight,create_agent,execution_identity
    from .fg_s3_e_quality_course_preflight import _local_path
    from embodied.run.fg_checkpoint import validate_bundle
    from .fg_s3_e_quality_corpus_cli import source_inputs,_factory
    from e_data_v1.provider import ETrainAdmission
    root=_local_path(run_dir);node=_local_path(checkpoint);out=_output(output)
    if not node.is_relative_to(root/'ckpt'):raise ValueError('F3_EVALUATION_NODE_OUTSIDE_RUN')
    identity=read_manifest(root/'run_identity.json')['identity'];meta=validate_bundle(node,identity)
    if identity.get('protocol')!='FG_S3_E_QUALITY_F3_RUN_V1' or identity.get('code')!=execution_identity():raise ValueError('F3_EVALUATION_CODE_OR_STAGE_MISMATCH')
    s=validate_settings(identity['preparation']['settings']);c=pickle.loads((node/'counters.pkl').read_bytes())
    if c['optimizer_updates']!=s['optimizer_updates'] or not meta['protected'] or c['numerical_rejections']:
        raise ValueError('F3_EVALUATION_COMPLETED_CHECKPOINT_REQUIRED')
    if preflight(s).receipt['identity']!=identity['preparation']:raise ValueError('F3_EVALUATION_SOURCE_MISMATCH')
    corpus=Path(s['corpus_root']);saved=read_manifest(corpus/'source_lock.json')
    current,_,_=source_inputs(saved['source_pool']['path'],saved['env_receipt']['path'],saved['contract'],**saved.get('selection',{}))
    if current!=saved:raise ValueError('F3_EVALUATION_TRAIN_SOURCE_MISMATCH')
    profile=ETrainAdmission.from_train_pool(saved['source_pool']['path'],corpus/'e_source',corpus/'selected_train_pool.json',contracts=[saved['contract']])
    row=copy.deepcopy(saved['row']);row['env_config']=str(corpus/'inputs'/saved['contract']/'env.yaml')
    agent=create_agent({**s,'device':'cpu'},out/'model',controlled_small=identity['controlled_small'])
    agent.load_exact(pickle.loads((node/'agent.pkl').read_bytes()))
    from gym_trading_env.envs.entry_contract import fingerprint
    return evaluate_native(agent,_factory(row,profile),out/'trace',steps=steps,
        initial_state=json.loads((corpus/'initial_account.json').read_text()),scope='FINITE_TRAIN_READ_ONLY_NOT_RESEARCH',
        source=dict(corpus=identity['preparation']['source'],checkpoint_manifest_sha256=digest(node/'manifest.json'),
                    f3_run_identity=fingerprint(identity)))
