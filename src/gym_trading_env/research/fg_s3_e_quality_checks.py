"""CPU data/loss diagnostics with raw receipts. No optimizer or lifecycle."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import resource
import time

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import write_json, signed
from .fg_s3_e_quality_course_preflight import _local_path
from .fg_s3_e_quality_supervision import SupervisionSource,QualitySupervisionAdmission

PROJECT=Path(__file__).resolve().parents[3]


def _output(value,corpus):
    from e_data_v1.io import independent_output
    path=_local_path(value)
    if (set(path.parts)&{'prepared','runs','run','ckpt','checkpoint','checkpoints'} or path.exists()
            or path.is_relative_to(_local_path(corpus))):
        raise ValueError('WP03_DIAGNOSTIC_OUTPUT_MUST_BE_NEW_AND_ISOLATED')
    independent_output(path,inputs=[corpus,PROJECT,PROJECT.parent/'dreamerv3'])
    return path


def cpu_model(length,output):
    """Fresh real backend, waiting public composition; no admitted stage."""
    from embodied.run.fg_s3_source_transfer import _make_agent
    from dreamerv3.fg_s3_v2_agent import _S3V2Model
    from dreamerv3.fg_s3_composition import resolve_composition
    from dreamerv3.fg_s3_e_config import e_target_config
    from dreamerv3.fg_s3_quality_v2 import quality_extensions
    from .fg_s3_v2_gpu_check import small_config
    from .fg_s3_e_quality_supervision import supervision_settings
    cfg=e_target_config(small_config());cfg['batch_length']=length;cfg['report_length']=length
    return _make_agent(cfg,output,device='cpu',agent_cls=_S3V2Model,bc_lambda=1.,
        extra_agent_config=quality_extensions(supervision_settings()),composition=resolve_composition(
            dict(composition=dict(encoder='e_multiscale_1m15m',teaching='quality_v2'))))


def cpu_forward(agent,batch,*,effective_ids=None):
    """Call the actual context, training_loss_fn and quality_loss without opt."""
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import numpy as np
    from dreamerv3.fg_s3_quality_v2 import predict
    for key,value in batch.items():
        if np.issubdtype(value.dtype,np.floating) and not np.isfinite(value).all():
            raise ValueError('WP03_NONFINITE_BATCH_BEFORE_JAX: '+key)
    def forward(data):
        carry,obs,prev,ids=agent.model._apply_replay_context(agent.model.init_train(len(data['is_first'])),data)
        loss,(_,_,_,metrics)=agent.model.training_loss_fn(data,ids if effective_ids is None else effective_ids)(
            carry,obs,prev,False)
        wm,(_,_,outs,_)=agent.model.wm_only(carry,obs,prev,False)
        features=agent.model.feat2tensor(outs['repfeat'])
        policy=agent.model._masked_policy(agent.model.quality_actor(jax.lax.stop_gradient(features),2),
                                           obs['action_mask'],False)[agent.model.action_key]
        predictions=predict(agent.model,features,2)
        return loss,metrics,ids,features,policy.logits,predictions,wm
    # Installed Ninjax.scan requires a modifiable private functional context.
    # Discard its returned state; no optimizer is called and agent.params is
    # never assigned. Evidence below compares every parameter/optimizer byte.
    _,out=nj.pure(forward)(agent.params,{k:jnp.asarray(v) for k,v in batch.items()},
                           seed=202,create=False,modify=True)
    return jax.tree.map(lambda v:np.asarray(v),out)


def cpu_actor_terms(agent,features,batch):
    """Existing Actor objective on actual model features; math only.

    CONTROLLED_FULL selects the existing pure objective's numeric terms here.
    This dict is NOT a stage configuration (Prediction/Decision remain false),
    is never given to an Agent factory, and cannot pass production admission.
    """
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import numpy as np
    from dreamerv3.fg_s3_quality_v2 import teaching_objective,DEFAULTS
    from dreamerv3.fg_s3_tutor_loss import eligible_teacher
    from .fg_s3_quality_catalog_v2 import FIELDS
    cfg={**agent.model.quality,'quality_stage':'CONTROLLED_FULL',
         'quality_strict_pair_lambda':DEFAULTS['quality_strict_pair_lambda'],
         'quality_pair_lambda':DEFAULTS['quality_pair_lambda']}
    labels={k:jnp.asarray(batch[k][:,1:]) for k in FIELDS}
    y=jnp.asarray(batch['teacher_action'][:,1:]);mask=jnp.asarray(batch['action_mask'][:,1:])
    valid=eligible_teacher(mask,y,jnp.asarray(batch['teacher_valid'][:,1:]))
    def actor():
        dist=agent.model._masked_policy(agent.model.quality_actor(jax.lax.stop_gradient(jnp.asarray(features)),2),
                                       mask,False)[agent.model.action_key]
        return teaching_objective(dist,labels,y,valid,jnp.asarray(batch['is_first'][:,1:]),cfg)
    params={k:v for k,v in agent.params.items() if k.startswith('quality_actor/')}
    pure=nj.pure(actor)
    _,(value,metrics)=pure(params,seed=202,create=False,modify=False)
    grad=jax.grad(lambda p:pure(p,seed=202,create=False,modify=False)[1][0])(params)
    return float(value),jax.tree.map(np.asarray,metrics),jax.tree.map(np.asarray,grad)


def cpu_prediction_gradients(agent,features,batch):
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import numpy as np
    from dreamerv3.fg_s3_quality_v2 import predict,prediction_objective
    from .fg_s3_quality_catalog_v2 import FIELDS
    labels={k:jnp.asarray(batch[k][:,1:]) for k in FIELDS}
    def heads():
        prediction=predict(agent.model,jax.lax.stop_gradient(jnp.asarray(features)),2)
        return prediction_objective(prediction,labels,jnp.asarray(batch['teacher_action'][:,1:]),
            jnp.asarray(batch['teacher_valid'][:,1:]),agent.model.quality)[0]
    params={k:v for k,v in agent.params.items() if k.startswith('quality_') and not k.startswith('quality_actor/')}
    pure=nj.pure(heads)
    grad=jax.grad(lambda p:pure(p,seed=202,create=False,modify=False)[1])(params)
    return jax.tree.map(np.asarray,grad)


def cpu_loss_evidence(batch,output,*,policy_batches=None):
    import jax
    import numpy as np
    from dreamerv3.fg_s3a_actor import parameter_digests
    from .fg_s3_quality_catalog_v2 import FIELDS
    began=time.monotonic()
    if any(d.platform!='cpu' for d in jax.devices()):raise ValueError('WP03_CPU_ONLY')
    agent=cpu_model(batch['stepid'].shape[1]-1,output/'model')
    before=parameter_digests(agent.params)
    baseline=cpu_forward(agent,batch)
    if not all(np.isfinite(v).all() for v in (baseline[0],baseline[3],baseline[4],baseline[6])):
        raise ValueError('WP03_NONFINITE_CPU_LOSS')
    np.testing.assert_array_equal(baseline[2],batch['stepid'][:,1:])
    changed=copy.deepcopy(batch)
    changed['teacher_action'][changed['teacher_valid']]=1-changed['teacher_action'][changed['teacher_valid']]
    changed['quality_score'][changed['quality_score_valid']]=100-changed['quality_score'][changed['quality_score_valid']]
    other=cpu_forward(agent,changed)
    np.testing.assert_array_equal(baseline[3],other[3]);np.testing.assert_array_equal(baseline[4],other[4])
    assert float(baseline[6])==float(other[6])  # Original WM target/Reward independent of teacher.
    actor,am,ag=cpu_actor_terms(agent,baseline[3],batch)
    _,_,changed_ag=cpu_actor_terms(agent,baseline[3],changed)
    assert any(np.any(v) for v in ag.values())
    assert any(not np.array_equal(v,changed_ag[k]) for k,v in ag.items())
    hg=cpu_prediction_gradients(agent,baseline[3],batch)
    assert all(not np.any(v) for k,v in hg.items() if k.startswith(('quality_q_c1/','quality_q_v2/')))
    empty=copy.deepcopy(batch);empty['teacher_valid'][...]=False
    for key in FIELDS:empty[key][...]=0
    assert float(cpu_forward(agent,empty)[0])==0
    assert not any(np.any(v) for v in cpu_actor_terms(agent,baseline[3],empty)[2].values())
    assert not any(np.any(v) for v in cpu_prediction_gradients(agent,baseline[3],empty).values())
    negatives={}
    for name in ('stepid_shift','old_q','soft'):
        bad=copy.deepcopy(batch)
        try:
            if name=='stepid_shift':cpu_forward(agent,bad,effective_ids=np.roll(bad['stepid'][:,1:],1,axis=1))
            else:
                bad['quality_q_v2_valid' if name=='old_q' else 'quality_soft_valid'][:,1]=True
                cpu_forward(agent,bad)
        except Exception as exc:
            expected='S3_V2_TEACHER_REPLAY_CONTEXT_SHIFT' if name=='stepid_shift' else 'Q_SOFT_FORBIDDEN'
            if expected not in str(exc):raise
            negatives[name]=dict(rejected=True,error=str(exc))
        else:raise ValueError('WP03_REQUIRED_NEGATIVE_WAS_ACCEPTED: '+name)
    per_policy={}
    for policy,raw in (policy_batches or {}).items():
        result=cpu_forward(agent,raw)
        assert np.isfinite(result[0])
        per_policy[policy]=dict(loss=float(result[0]),strict_valid=int(raw['teacher_valid'][:,1:].sum()),
            outcome_valid=int(raw['quality_outcome_valid'][:,1:].sum()),
            effective_stepid_hash=fingerprint(raw['stepid'][:,1:].tolist()))
    agent.sync_policy_from_train();sealed=agent.save_exact()
    obs={k:batch[k][:,1] for k in agent.obs_space}
    _,_,first=agent.policy(agent.init_policy(1),obs,mode='eval')
    agent.load_exact(sealed)
    _,_,same=agent.policy(agent.init_policy(1),{k:changed[k][:,1] for k in agent.obs_space},mode='eval')
    for key in first:np.testing.assert_array_equal(first[key],same[key])
    cold=cpu_model(batch['stepid'].shape[1]-1,output/'cold_model')
    cold.load_exact(sealed)
    _,_,restored=cold.policy(cold.init_policy(1),obs,mode='eval')
    for key in first:np.testing.assert_array_equal(first[key],restored[key])
    assert before==parameter_digests(agent.params)==parameter_digests(cold.params)
    assert int(agent.n_updates)==int(cold.n_updates)==0
    return signed(dict(status='PASS_CPU_LOSS_ONLY',scope='NO_OPTIMIZER_UPDATE_OR_STAGE_ADMISSION',
        source='FRESH_PARAMETERS_ONLY',batch_shape=list(batch['stepid'].shape),
        loss=float(baseline[0]),wm_loss=float(baseline[6]),actor_numeric_loss=actor,
        strict_pair_count=int(am['quality/strict_pair_count']),quality_pair_count=int(am['quality/quality_pair_count']),
        teacher_changes_actor_gradient=True,teacher_does_not_change_actor_logits=True,
        actor_only_policy_future_label_isolation=True,all_invalid_loss_and_gradients_zero=True,
        old_q_soft_contribution_and_gradients_zero=True,
        exact_cold_agent_restore=True,parameters_and_optimizer_unchanged=True,actual_optimizer_updates=0,
        prediction_admitted=False,decision_admitted=False,training_admitted=False,
        head_gradient_norms={name:float(np.sqrt(sum(float(np.square(v.astype(np.float64)).sum())
            for k,v in hg.items() if k.startswith('quality_'+name+'/'))))
            for name in ('event','touch','adverse','score','q_c1','q_v2')},
        per_policy=per_policy,negative_cases=negatives,seconds=time.monotonic()-began))


def reader_evidence(admission,out,*,length=64,seed=202,with_model=True):
    import numpy as np
    from .fg_replay import read_manifest
    from .fg_s3_e_quality_reader import EQualityCourseReader,build_join_index
    began=time.monotonic()
    index=out/'join.sqlite'
    joined=build_join_index(admission,index)
    write_json(out/'WP03_TEACHER_JOIN_AUDIT.json',joined)
    index_seconds=time.monotonic()-began
    views={};support={};model_batch=None;scope=None
    all_tags=('TP_FAST','TP_SLOW','SL30','EARLY10_5','EARLY10_15','SAME_DAY','CROSS_DAY','STRICT_ADD','STRICT_WAIT')
    start=time.monotonic();delivered=0
    for stage in ('WM_NATURAL','HEADS_VALID','ACTOR_STRICT','ACTOR_QUALITY_PAIR'):
        for mode in (('NATURAL',) if stage=='WM_NATURAL' else ('NATURAL','STRATIFIED','MIXED')):
            spec=dict(mode=mode)
            if mode!='NATURAL':
                tags=('STRICT_ADD','STRICT_WAIT') if stage=='ACTOR_STRICT' else all_tags
                spec['strata_weights']={tag:1. for tag in tags}
            if mode=='MIXED':spec['natural_weight']=.5
            reader=EQualityCourseReader.open(admission,stage=stage,sampling_spec=spec,seed=seed,index_path=index)
            try:
                if scope is None:scope=reader.coverage()
                # Query support separately; never silently replace a missing class.
                available={tag:sum(g[-1] for g in reader._groups(length,tag)) for tag in all_tags}
                support[stage+':'+mode]=available
                if not reader._groups(length) or any(available[tag]==0 for tag in spec.get('strata_weights',{})):
                    views[stage+':'+mode]=dict(status='INSUFFICIENT_CURRICULUM_SUPPORT',support=available)
                    continue
                for _ in range(4):
                    raw=reader.sample(1,length);delivered+=length
                    if stage=='ACTOR_QUALITY_PAIR' and model_batch is None:model_batch=raw
                state=reader.state_dict();expected=reader.sample(1,length)
                reader.load_state_dict(state);actual=reader.sample(1,length)
                delivered+=2*length
                for key in expected:np.testing.assert_array_equal(expected[key],actual[key])
                coverage=reader.coverage()
                coverage['exact_next_batch_restore']=True
                coverage['effective_stepid_hash']=fingerprint(raw['stepid'][:,1:].tolist())
                views[stage+':'+mode]=coverage
            finally:reader.close()
    policy_batches={};policy_support={}
    for policy in joined['policies']:
        reader=EQualityCourseReader.open(admission,stage='HEADS_VALID',
            sampling_spec=dict(mode='NATURAL',policies=[policy]),seed=seed,index_path=index)
        try:
            windows=sum(g[-1] for g in reader._groups(length))
            policy_support[policy]=dict(available_windows=windows,
                status='SUPPORTED' if windows else 'INSUFFICIENT_CURRICULUM_SUPPORT')
            if windows:policy_batches[policy]=reader.sample(1,length)
        finally:reader.close()
    reference=read_manifest(Path(admission.source.corpus_root)/'coverage.json')
    if (scope['unique_market_cursors']!=reference['unique_market_cursors'] or
            scope['distinct_account_decisions']!=reference['distinct_account_decision_states']):
        raise ValueError('WP03_READER_WP02_COVERAGE_MISMATCH')
    sampling_seconds=time.monotonic()-start
    reader_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if with_model:
        if model_batch is None:raise ValueError('INSUFFICIENT_CURRICULUM_SUPPORT: no legal quality-pair model batch')
        loss=cpu_loss_evidence(model_batch,out,policy_batches=policy_batches)
        write_json(out/'WP03_CPU_FORWARD_LOSS_EVIDENCE.json',loss)
    report=signed(dict(status='PASS_READER_CPU' if with_model else 'PASS_READER_ONLY',
        corpus_identity=admission.source.corpus_content_hash,source_identity=joined['source_identity'],
        index_sha256=joined['index_sha256'],stage_views=views,course_support=support,policy_support=policy_support,
        curriculum_support='LIMITED' if (any('status' in v for v in views.values()) or
            any(v['available_windows']==0 for v in policy_support.values())) else 'SUPPORTED_FOR_REQUESTED_WINDOWS',
        actual_optimizer_exposures=0,training_admitted=False,
        independent_wp02_count_comparison='PASS',index_seconds=index_seconds,
        reader_sampling_seconds=sampling_seconds,measured_sample_rows=delivered,
        measured_sample_rows_per_second=delivered/sampling_seconds,
        reader_process_peak_rss_bytes=reader_peak_rss,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        index_bytes=index.stat().st_size,
        corpus_published_bytes=sum((Path(admission.source.corpus_root)/p).stat().st_size
                                   for p in admission.receipt['identity']['files'])))
    write_json(out/'WP03_COURSE_READER_COVERAGE.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description='WP03 CPU supervision checks, never formal training')
    sub=parser.add_subparsers(dest='command',required=True)
    source=sub.add_parser('source')
    source.add_argument('--corpus',required=True);source.add_argument('--out',required=True)
    source.add_argument('--purpose',choices=['development','native-verification'],default='development')
    source.add_argument('--producer-gym')
    source.add_argument('--producer-lock',help='Operator-reviewed frozen producer pin; requires its separately supplied SHA256')
    source.add_argument('--producer-lock-sha256')
    source.add_argument('--workers',type=int,help='CPU process budget forwarded to native G3')
    reader=sub.add_parser('read')
    reader.add_argument('--corpus',required=True);reader.add_argument('--out',required=True)
    reader.add_argument('--length',type=int,default=64);reader.add_argument('--seed',type=int,default=202)
    reader.add_argument('--without-model',action='store_true')
    multi=sub.add_parser('multi-read',help='Read-only multi-Corpus scheduling and cold Checkpoint probe')
    multi.add_argument('--config',required=True);multi.add_argument('--out',required=True)
    multi.add_argument('--batch-size',type=int,default=4);multi.add_argument('--length',type=int,default=8)
    multi.add_argument('--adapter',choices=('native','f3'),default='native')
    multi.add_argument('--burnin-steps',type=int)
    multi.add_argument('--require-published',action='store_true',help='Require 2+ actual published TRAIN batch receipts')
    args=parser.parse_args(argv)
    try:
        if args.command=='multi-read':
            from .fg_s3_e_quality_multi_checks import multi_evidence
            result=multi_evidence(args.config,args.out,batch_size=args.batch_size,
                length=args.length,require_published=args.require_published,
                adapter=args.adapter,burnin_steps=args.burnin_steps)
            print(json.dumps(result,ensure_ascii=False),flush=True)
            return 0
        out=_output(args.out,args.corpus)
        source=SupervisionSource.from_corpus(args.corpus)
        admission=QualitySupervisionAdmission.preflight(source,purpose=getattr(args,'purpose','development'),
            producer_gym=getattr(args,'producer_gym',None),log_path=out/'native_verify.log',
            producer_lock=getattr(args,'producer_lock',None),producer_lock_sha256=getattr(args,'producer_lock_sha256',None),
            workers=getattr(args,'workers',None))
        write_json(out/'WP03_SOURCE_ADMISSION_RECEIPT.json',admission.receipt)
        if admission.native_result is not None:
            write_json(out/'native_verification.json',admission.native_result)
        result=admission.receipt
        if args.command=='read':
            result=reader_evidence(admission,out,length=args.length,seed=args.seed,with_model=not args.without_model)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        return 0
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:
        print(json.dumps(dict(status='NO_GO',error=str(exc),training_admitted=False)),flush=True)
        return 2
