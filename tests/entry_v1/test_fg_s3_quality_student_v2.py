"""Small CPU network acceptance, distinct from real GPU/policy performance."""
import copy
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embodied.jax.outs import MaskedCategorical
from embodied.run.fg_s3_source_transfer import make_agent
from dreamerv3.fg_s3_v2_agent import S3V2Agent
from dreamerv3.fg_s3a_actor import parameter_digests
from dreamerv3.fg_s3_quality_v2 import (quality_extensions, fuse_logits,
    teaching_objective, prediction_objective, active_branches)
from gym_trading_env.research.fg_s3_quality_catalog_v2 import FIELDS
from gym_trading_env.research.fg_s3_v2_gpu_check import small_config, gpu_batch
from gym_trading_env.research.fg_replay import signed, write_json


def controlled_extensions(settings):
    # Backend-only synthetic exercise of retained Full modules, not research
    # admission or an authorized production training configuration.
    return quality_extensions(dict(quality_stage='CONTROLLED_FULL',
        quality_prediction_admitted=True,quality_decision_admitted=True,**settings))


def labels(shape=(1,4)):
    d={key:np.zeros((*shape,*dims),dtype) for key,(dtype,dims) in FIELDS.items()}
    for key in ('quality_score_valid','quality_outcome_valid','quality_event_valid','quality_touch_valid',
                'quality_adverse_valid','quality_q_c1_valid','quality_q_v2_valid'):
        d[key][...]=True
    d['quality_score'][...]=np.resize([100.,0.,85.,30.],shape)
    d['quality_tp'][...]=d['quality_score']>0
    d['quality_sl'][...]=d['quality_score']==0
    d['quality_event'][...]=np.resize([0,1,2,3],(*shape,7))
    d['quality_touch'][...]=1
    d['quality_adverse'][...]=1
    d['quality_q_c1'][...]=[0.,1.]
    d['quality_q_v2'][...]=[-.2,.8]
    d['quality_soft_valid'][...]=True;d['quality_soft_source'][...]=2
    d['quality_soft_target'][...]=1/(1+np.exp(-1))
    # Legal shared synthetic lineage; the all-zero owner is invalid.
    d['quality_owner'][...,0]=1
    return {k:jnp.asarray(v) for k,v in d.items()}


def predictions(shape=(1,4)):
    return {n:jnp.zeros((*shape,*s)) for n,s in
        (('event',(7,4)),('touch',(4,2)),('adverse',(4,)),('score',()),('q_c1',(2,)),('q_v2',(2,)))}


def synthetic_batch():
    from gym_trading_env.research.fg_replay import FIELDS as NATIVE
    shape=(1,65);rng=np.random.default_rng(202)
    raw={k:np.zeros((*shape,*dims),dtype) for k,(dims,dtype) in NATIVE.items()}
    raw['market_seq']=rng.normal(0,.1,raw['market_seq'].shape).astype(np.float32)
    raw['action_mask'][...]=1;raw['is_first'][0,0]=True
    raw['stepid']=rng.integers(0,255,(*shape,20),np.uint8)
    raw['market_advance']=np.ones(shape,np.int32);raw['consec']=np.zeros(shape,np.int32)
    raw['teacher_action']=np.resize([1,0,1,0],shape).astype(np.int32)
    raw['teacher_valid']=np.ones(shape,bool)
    raw.update({k:np.asarray(v) for k,v in labels(shape).items()})
    return raw


def test_actor_same_forward_bounds_and_prediction_stop_gradient():
    cfg=controlled_extensions({'quality_q_admitted':True})
    pred=predictions();raw=jnp.zeros((1,4,2));mask=jnp.ones_like(raw)
    final,m=fuse_logits(raw,pred,mask,cfg)
    np.testing.assert_allclose(final[...,1]-final[...,0],sum(m['quality/'+k] for k in ('delta_O','delta_S','delta_Q')))
    assert all(np.max(np.abs(m['quality/'+k]))<=.5 for k in ('delta_O','delta_S','delta_Q'))
    grad=jax.grad(lambda p:fuse_logits(raw,p,mask,cfg)[0].sum())(pred)
    assert all(not np.asarray(g).any() for g in grad.values())
    closed=mask.at[...,1].set(0)
    np.testing.assert_array_equal(fuse_logits(raw,pred,closed,cfg)[0],raw)


def test_pair_identity_soft_targets_and_empty_gradients():
    cfg=controlled_extensions({});lab=labels();y=jnp.array([[1,0,1,0]]);valid=jnp.ones_like(y,bool)
    first=jnp.array([[1,0,0,0]],bool);raw=jnp.zeros((1,4,2))
    def obj(x,l=lab,v=valid):
        dist=MaskedCategorical(x,jnp.ones_like(x,bool),unimix=.05)
        return teaching_objective(dist,l,y,v,first,cfg)
    loss,m=obj(raw)
    assert m['quality/strict_pair_count']==4 and m['quality/quality_pair_count']>0
    other={**lab,'quality_owner':lab['quality_owner'].at[:,1::2,0].set(42)}
    assert obj(raw,other)[1]['quality/strict_pair_count']==0
    empty={k:jnp.zeros_like(v) for k,v in lab.items()}
    value,metrics=obj(raw,empty,jnp.zeros_like(valid))
    assert value==0 and all(np.isfinite(np.asarray(v)).all() for v in metrics.values())
    assert not np.asarray(jax.grad(lambda x:obj(x,empty,jnp.zeros_like(valid))[0])(raw)).any()


def test_q_weak_wait_and_independent_prediction_masks():
    cfg=controlled_extensions({});lab=labels();pred=predictions()
    no_values={**lab,'quality_q_c1_valid':jnp.zeros((1,4),bool),'quality_q_v2_valid':jnp.zeros((1,4),bool)}
    y=jnp.zeros((1,4),jnp.int32);valid=jnp.ones_like(y,bool)
    loss,m=prediction_objective(pred,no_values,y,valid,cfg)
    assert m['quality/q_c1_rank_loss']==0 and m['quality/q_v2_loss']==0
    grads=jax.grad(lambda p:prediction_objective(p,no_values,y,valid,cfg)[0])(pred)
    assert not np.asarray(grads['q_c1']).any() and not np.asarray(grads['q_v2']).any()
    # OPEN at H60 and later TP are separate valid windows, not a binary bad trade.
    early={**pred,'event':pred['event'].at[...,2,3].set(2).at[...,3,0].set(2)}
    assert np.isfinite(float(prediction_objective(early,lab,y,valid,cfg)[0]))


def test_subunit_quality_weights_cancel_inside_single_member_class():
    cfg=controlled_extensions({'quality_kappa':0.})
    lab=labels((1,2));lab['quality_score']=jnp.array([[100.,90.]])
    lab['quality_sl']=jnp.zeros((1,2),bool)
    y=jnp.array([[1,0]]);valid=jnp.ones_like(y,bool)
    raw=jnp.array([[[0.,1.],[.2,-.4]]])
    dist=MaskedCategorical(raw,jnp.ones_like(raw,bool),unimix=.05)
    _,m=teaching_objective(dist,lab,y,valid,jnp.array([[1,0]],bool),cfg)
    expected=float(-dist.logp(y).mean())
    assert float(m['quality/balanced_loss'])==np.float32(expected)


def test_all_heads_gradient_and_exact_model_restore_cpu(tmp_path):
    assert all(d.platform=='cpu' for d in jax.devices())
    cfg=small_config()
    agent=make_agent(cfg,tmp_path/'model',device='cpu',agent_cls=S3V2Agent,
        bc_lambda=1.,extra_agent_config=controlled_extensions({'quality_gradient_every_updates':1}))
    raw=synthetic_batch()
    # The inherited warmup schedule starts at LR=0. Preserve that protocol;
    # distinguish the first gradient from the first nonzero parameter update.
    agent.train(agent.init_train(1),gpu_batch(agent,raw,0))
    before=agent.save_exact()
    digests=parameter_digests(before['params'])
    carry=agent.init_train(1)
    _,_,metrics=agent.train(carry,gpu_batch(agent,raw,1))
    after=agent.save_exact()
    after_digests=parameter_digests(after['params'])
    groups=('quality_actor','quality_event','quality_touch','quality_adverse','quality_score','quality_q_c1','quality_q_v2')
    evidence={}
    for prefix in groups:
        grad=float(metrics['opt/'+prefix+'_grad_norm'])
        delta=float(metrics['opt/'+prefix+'_delta_norm'])
        assert grad>0 and delta>0,(prefix,grad,delta)
        evidence[prefix]=dict(grad_norm=grad,delta_norm=delta)
    frozen=[k for k in digests if k.startswith(('pol/','val/','slowval/','slowval_count/','retnorm/','valnorm/','advnorm/'))]
    assert all(digests[k]==after_digests[k] for k in frozen)
    assert float(metrics['opt/wm_delta_norm'])>0
    assert bool(metrics['quality/gradient_sampled'])
    assert all(float(metrics['quality/grad/'+name+'_norm'])>0 for name in
               ('natural','balanced','strict_pair','quality_pair','soft','total_actor'))
    # Restoring the entire real Agent reproduces the next update exactly.
    agent.load_exact(before)
    _,_,again=agent.train(agent.init_train(1),gpu_batch(agent,raw,1))
    assert parameter_digests(agent.save_exact()['params'])==after_digests
    # Previously trained head momentum must not move an unsupported branch.
    unsupported=copy.deepcopy(raw)
    unsupported['teacher_valid'][...]=False
    unsupported.update({k:np.zeros_like(v) for k,v in raw.items() if k in FIELDS})
    _,_,inactive=agent.train(agent.init_train(1),gpu_batch(agent,unsupported,2))
    inactive_digests=parameter_digests(agent.save_exact()['params'])
    protected=[k for k in after_digests if k.startswith('quality_') or
               (k.startswith('opt/state/') and 'quality' in k)]
    assert all(inactive_digests[k]==after_digests[k] for k in protected)
    # Future-label changes without a training update cannot change policy.
    agent.sync_policy_from_train()
    sealed=agent.save_exact()
    obs={k:raw[k][:,0] for k in agent.obs_space}
    _,_,p1=agent.policy(agent.init_policy(1),obs,mode='eval')
    unsupported['quality_score'][...]=99
    agent.load_exact(sealed)
    _,_,p2=agent.policy(agent.init_policy(1),obs,mode='eval')
    assert all(np.array_equal(p1[k],p2[k]) for k in p1 if k.startswith('quality/'))
    output=tmp_path/'joint_cpu_receipt.json'
    write_json(output,signed(dict(status='PASS_SYNTHETIC_CPU_JOINT_UPDATE',heads=evidence,
        actor_term_gradients={k:float(v) for k,v in metrics.items() if k.startswith('quality/grad/')},
        wm_delta_norm=float(metrics['opt/wm_delta_norm']),frozen_parameter_count=len(frozen),
        unsupported_head_and_optimizer_freeze=True,policy_future_label_isolation=True,
        exact_update_restore=True, real_TRAIN_supervision='NOT_USED_IN_THIS_SYNTHETIC_TEST',gpu='NOT_RUN')))
    print('\nQUALITY_CPU_RECEIPT='+str(output))


def test_quality_actor_parameter_warm_start_and_fresh_joint_optimizer(tmp_path):
    from embodied.run.fg_s3_source_transfer import _apply_parameter_warm_start
    from dreamerv3.fg_s3_tutor_loss import agent_extensions
    cfg=small_config()
    source=make_agent(cfg,tmp_path/'source',device='cpu',agent_cls=S3V2Agent,bc_lambda=1.,
        extra_agent_config=agent_extensions(dict(mode='tutor',tutor_hard_kappa=1.)))
    batch=synthetic_batch();batch={k:v for k,v in batch.items() if not k.startswith('quality_')}
    for seed in (0,1): source.train(source.init_train(1),gpu_batch(source,batch,seed))
    original=source.save_exact()
    target=make_agent(cfg,tmp_path/'target',device='cpu',agent_cls=S3V2Agent,bc_lambda=1.,
        extra_agent_config=quality_extensions({}))
    fresh=parameter_digests(target.params)
    spec=signed(dict(kind='warm_start',origin=dict(step=2,environment_steps=0),
        optimizer_transfer_policy='fresh_joint_opt_all_slots',clock_protocol='S3_MARKET_MINUTE',
        agent_extension=quality_extensions({})))
    receipt=_apply_parameter_warm_start(target,spec,original)
    after=parameter_digests(target.params)
    source_digests=parameter_digests(original['params'])
    for k in source_digests:
        if k.startswith('pol/'):
            assert after['quality_actor/'+k[4:]]==after[k]==source_digests[k]
        if k.startswith('opt/'):
            assert after[k]==fresh[k]
    assert target.n_updates==0 and receipt['quality_actor_copied_keys']
    assert receipt['slow_critic_inherited_from_source']
    assert not receipt['slow_critic_initialized_from_critic']
    write_json(tmp_path/'warm_start_receipt.json',receipt)


def test_controlled_prediction_transfer_requires_evaluated_weights():
    from embodied.run.fg_s3_source_transfer import admitted_auxiliary_parameters
    source={name+'/out/kernel':np.full((2,3),.25,np.float32) for name in
            ('quality_event','quality_touch','quality_adverse','quality_score')}
    source['quality_q_v2/out/kernel']=np.ones((2,2),np.float32)
    target={k:np.zeros_like(v) for k,v in source.items()}
    proof=parameter_digests({k:v for k,v in source.items() if not k.startswith('quality_q_')})
    copied=admitted_auxiliary_parameters(target,source,proof)
    assert parameter_digests(copied)==proof
    assert len(copied)==4 and not any(k.startswith('quality_q_') for k in copied)
    assert all(not v.any() for v in target.values())
    changed={**source,'quality_score/out/kernel':source['quality_score/out/kernel']+1}
    with pytest.raises(ValueError,match='evidence differs'):
        admitted_auxiliary_parameters(target,changed,proof)
    with pytest.raises(ValueError,match='schema/finite'):
        admitted_auxiliary_parameters({**target,'quality_score/out/kernel':np.zeros((1,),np.float32)},source,proof)
    source['quality_q_c1/out/kernel']=np.full((2,2),.5,np.float32)
    target['quality_q_c1/out/kernel']=np.zeros((2,2),np.float32)
    prefixes=('quality_q_c1/','quality_q_v2/')
    qproof=parameter_digests({k:v for k,v in source.items() if k.startswith(prefixes)})
    copied=admitted_auxiliary_parameters(target,source,qproof,prefixes=prefixes)
    assert parameter_digests(copied)==qproof and all(k.startswith(prefixes) for k in copied)


def test_quality_first_heads_only_updates_and_actor_isolation(tmp_path):
    extension=quality_extensions({})
    assert extension['quality_stage']=='HEADS_ONLY'
    assert not any(extension[k] for k in ('quality_fuse_outcome','quality_fuse_score','quality_q_admitted'))
    with pytest.raises(ValueError,match='prediction admission'):
        quality_extensions({'quality_stage':'CONTROLLED_FULL'})
    with pytest.raises(ValueError,match='forbids'):
        quality_extensions({'quality_fuse_score':True})
    agent=make_agent(small_config(),tmp_path/'head_only',device='cpu',agent_cls=S3V2Agent,
                    bc_lambda=1.,extra_agent_config=extension)
    raw=synthetic_batch()
    raw['teacher_valid'][...]=False  # L3 absent; L1/L2 still train the heads.
    agent.train(agent.init_train(1),gpu_batch(agent,raw,0))
    before=agent.save_exact();digests=parameter_digests(before['params'])
    _,_,metrics=agent.train(agent.init_train(1),gpu_batch(agent,raw,1))
    after=agent.save_exact();after_digests=parameter_digests(after['params'])
    updated={}
    for name in ('quality_event','quality_touch','quality_adverse','quality_score'):
        grad=float(metrics['opt/'+name+'_grad_norm']);delta=float(metrics['opt/'+name+'_delta_norm'])
        assert grad>0 and delta>0
        updated[name]=dict(grad_norm=grad,delta_norm=delta)
    for name in ('quality_actor','quality_q_c1','quality_q_v2','actor','critic','wm','encoder','rssm'):
        assert float(metrics['opt/'+name+'_grad_norm'])==0,name
        assert float(metrics['opt/'+name+'_delta_norm'])==0,name
    frozen=[k for k in digests if not k.startswith(('opt/','quality_event/','quality_touch/','quality_adverse/','quality_score/'))]
    assert all(digests[k]==after_digests[k] for k in frozen)
    assert metrics['quality/actor_loss']==0
    obs={k:raw[k][:,0] for k in agent.obs_space}
    agent.sync_policy_from_train()
    _,action1,p1=agent.policy(agent.init_policy(1),obs,mode='eval')
    changed=copy.deepcopy(after)
    for key in changed['params']:
        if key.startswith(('quality_actor/','quality_event/','quality_score/','quality_q_v2/')):
            changed['params'][key]=changed['params'][key]+np.float32(2)
    agent.load_exact(changed);agent.sync_policy_from_train()
    _,action2,p2=agent.policy(agent.init_policy(1),obs,mode='eval')
    np.testing.assert_array_equal(action1['action'],action2['action'])
    for key in ('quality/z_final','quality/pADD_after_mask_unimix'):
        np.testing.assert_array_equal(p1[key],p2[key])
    write_json(tmp_path/'heads_only_receipt.json',signed(dict(
        status='PASS_QUALITY_FIRST_HEADS_ONLY_SYNTHETIC_CPU',heads=updated,
        frozen_model_parameter_count=len(frozen),actor_influence=False,base_model_gradients_zero=True,
        q_heads_frozen=True,quality_predictions_cannot_change_policy=True,
        supervision='synthetic backend test; not predictive research acceptance',gpu='NOT_RUN')))
