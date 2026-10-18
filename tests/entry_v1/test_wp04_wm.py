import copy
import json

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from gym_trading_env.research.fg_s3_e_quality_foundation import create_agent,F1WindowAdapter
from dreamerv3.fg_s3_e_quality_foundation import context,check_state_schema,loss as f1_loss
from dreamerv3.fg_s3a_actor import parameter_digests
from wp04_fixture import native_reader


@pytest.fixture(scope='module')
def native_case(tmp_path_factory):
    root=tmp_path_factory.mktemp('wp04_native')
    settings,reader=native_reader(root)
    try:yield settings,reader,root
    finally:reader.close()


@pytest.fixture(scope='module')
def wm_case(native_case):
    settings,reader,root=native_case
    agent=create_agent(settings,root/'model',controlled_small=True)
    return settings,reader,agent,agent.save_exact()


def train_batch(agent,raw,seed=202):
    from gym_trading_env.research.fg_s3_v2_gpu_check import gpu_batch
    return agent.train(agent.init_train(len(raw['is_first'])),gpu_batch(agent,raw,seed))


def functional(agent,data,*,grad=False):
    def fn(raw):
        carry,obs,act,ids=context(agent.model,raw)
        loss,(_,_,outs,mets)=f1_loss(agent.model,carry,obs,act,False)
        return loss,(ids,carry,outs,mets)
    def call(raw):
        if grad:
            loss,_,derivative,aux=nj.grad(fn,agent.model.modules,has_aux=True)(raw)
            return loss,derivative,aux
        return fn(raw)
    _,out=nj.pure(call)(agent.params,{k:jnp.asarray(v) for k,v in data.items()},
                            seed=202,create=False,modify=True)
    return jax.tree.map(np.asarray,out)


def test_B01_reader_native_window_and_future_labels_are_separate(native_case):
    s,r,_=native_case;adapter=F1WindowAdapter(r,s)
    raw=r.sample(1,12);data,meta=adapter.adapt(raw)
    assert data['stepid'].shape==(1,13,20) and len(meta[0]['rows'])==13
    assert not any(k.startswith(('teacher_','quality_')) for k in data)
    changed=copy.deepcopy(raw)
    for key in changed:
        if key.startswith(('quality_','teacher_')):changed[key][...]=0
    other,_=adapter.adapt(changed)
    for key in data:np.testing.assert_array_equal(data[key],other[key])


@pytest.mark.parametrize('attack',['shuffle','owner','shift','action','reward','carry','length'])
def test_B02_corrupt_native_window_rejected_before_optimizer(native_case,attack):
    s,r,_=native_case;adapter=F1WindowAdapter(r,s);raw=r.sample(1,12)
    if attack=='shuffle':
        for key in raw:raw[key][:,[2,3]]=raw[key][:,[3,2]]
    elif attack=='owner':
        other=r.sample(1,12)
        while r.audit_windows(other)[0]['policy']==r.audit_windows(raw)[0]['policy']:other=r.sample(1,12)
        for key in raw:raw[key][:,2]=other[key][:,2]
    elif attack=='shift':raw['stepid']=np.roll(raw['stepid'],1,axis=1)
    elif attack=='action':raw['action'][:,3]=1-raw['action'][:,3]
    elif attack=='reward':raw['reward'][:,3]+=1
    elif attack=='carry':raw['consec'][...]=1
    else:raw={k:v[:,:-1] for k,v in raw.items()}
    with pytest.raises(ValueError):adapter.adapt(raw)


def test_B03_actual_original_optimizer_step_and_frozen_slots(wm_case,tmp_path):
    s,r,agent,initial=wm_case;agent.load_exact(initial)
    raw,_=F1WindowAdapter(r,s).sample()
    before=parameter_digests(agent.params)
    _,outs,metrics=train_batch(agent,raw)
    # The unchanged canonical optimizer has warmup=1000: its first update
    # advances clocks/moments with lr=0. Prove learning after the next update.
    first=parameter_digests(agent.params)
    _,outs,metrics=train_batch(agent,raw,203)
    after=parameter_digests(agent.params)
    assert int(agent.n_updates)==2 and bool(metrics['f1/finite'])
    np.testing.assert_array_equal(outs['effective_stepid'],raw['stepid'][:,5:])
    assert metrics['f1/target_positions']==8 and metrics['f1/burnin_positions']==4
    frozen=[k for k,v in agent.model.f1_mutable_keys.items() if not v]
    assert all(before[k]==after[k] for k in frozen)
    assert all(any(before[k]!=after[k] for k in before if k.startswith(p))
               for p in ('enc/','dyn/','dec/','rew/','con/','avail/'))
    (tmp_path/'parameter_delta.json').write_text(json.dumps(dict(before=before,first=first,after=after,
        mutable=agent.model.f1_mutable_keys,shared_clocks=agent.model.f1_shared_clocks),indent=2))


def test_B04_nonzero_inherited_momentum_and_negative_zero_stay_frozen(wm_case):
    s,r,agent,initial=wm_case;agent.load_exact(initial)
    for k in agent.params:
        if not agent.model.f1_mutable_keys[k]:
            agent.params[k]=jnp.full_like(agent.params[k],.125 if k.startswith('opt/') else -0.)
    before=parameter_digests(agent.params);data,_=F1WindowAdapter(r,s).sample()
    train_batch(agent,data)
    after=parameter_digests(agent.params)
    assert all(before[k]==after[k] for k,v in agent.model.f1_mutable_keys.items() if not v)


def test_B05_only_T_targets_and_detached_K_history(wm_case):
    s,r,agent,initial=wm_case;agent.load_exact(initial)
    data,_=F1WindowAdapter(r,s).sample();loss,grads,(ids,carry,outs,mets)=functional(agent,data,grad=True)
    assert np.isfinite(loss) and ids.shape==(1,8,20)
    assert all(v.shape==(1,8) for v in outs['losses'].values())
    assert set(outs['losses'])==set(agent.model.scales)-{'policy','value','repval'}
    assert not {'market_context_15m','market_macd_5m'} & outs['losses'].keys()
    def by_input(seq):
        changed={**{k:jnp.asarray(v) for k,v in data.items()},'market_seq_e_shared':seq}
        return nj.pure(lambda raw:agent.model.wm_only(*context(agent.model,raw)[:3],False)[0])(
            agent.params,changed,seed=202,create=False,modify=True)[1]
    derivative=np.asarray(jax.grad(by_input)(jnp.asarray(data['market_seq_e_shared'])))
    assert not np.any(derivative[:,:5]) and np.any(derivative[:,5:])


def test_B06_each_window_ignores_previous_random_carry(wm_case):
    s,r,agent,initial=wm_case;agent.load_exact(initial);raw,_=F1WindowAdapter(r,s).sample()
    _,_,metrics=train_batch(agent,raw,202);first=parameter_digests(agent.params)
    agent.load_exact(initial)
    carry=jax.tree.map(lambda x:jnp.full_like(x,99),agent.init_train(1))
    from gym_trading_env.research.fg_s3_v2_gpu_check import gpu_batch
    agent.train(carry,gpu_batch(agent,raw,202))
    assert first==parameter_digests(agent.params)


def test_B07_unknown_parameter_and_optimizer_leaf_schema_rejected(wm_case):
    _,_,agent,initial=wm_case;agent.load_exact(initial)
    for key in ('alien/kernel','enc/new_uninitialized/kernel','opt/state/alien'):
        bad={**agent.params,key:jnp.ones((2,))}
        with pytest.raises(ValueError,match='STATE_SCHEMA'):check_state_schema(agent.model,bad)


def test_B08_insufficient_K_T_support_does_not_reduce_length(native_case):
    s,r,_=native_case;s=copy.deepcopy(s);s.update(target_steps=256,burnin_steps=32)
    adapter=F1WindowAdapter(r,s);before=r.state_dict()
    with pytest.raises(ValueError,match='INSUFFICIENT_CURRICULUM_SUPPORT'):adapter.sample()
    assert before==r.state_dict()


def test_B09_K32_T64_four_native_WM_updates_without_actor_forward(native_case,monkeypatch):
    from pathlib import Path
    from dreamerv3.agent import Agent as OriginalModel
    from dreamerv3.fg_s3_v2_agent import _S3V2Model
    s,r,root=native_case;s=copy.deepcopy(s);s.update(target_steps=64,burnin_steps=32)
    original=OriginalModel.loss
    initializing=[True]
    def init_only(model,*args,**kwargs):
        # Original nj.grad replays init_loss under create=False, even while
        # the outer Agent is still constructing its full optimizer layout.
        if not initializing[0]:raise AssertionError('F1 evaluated original Actor/Critic imagination loss')
        return original(model,*args,**kwargs)
    monkeypatch.setattr(OriginalModel,'loss',init_only)
    def forbidden(*args,**kwargs):raise AssertionError('F1 evaluated teacher Quality loss')
    monkeypatch.setattr(_S3V2Model,'quality_loss',forbidden)
    agent=create_agent(s,root/'long_window_model',controlled_small=True)
    initializing[0]=False
    before=parameter_digests(agent.params);adapter=F1WindowAdapter(r,s);metrics=[]
    for step in range(4):
        raw,_=adapter.sample();_,out,m=train_batch(agent,raw,step+202)
        np.testing.assert_array_equal(out['effective_stepid'],raw['stepid'][:,33:])
        assert m['f1/target_positions']==64 and m['f1/burnin_positions']==32
        metrics.append({k:np.asarray(v).tolist() for k,v in m.items()})
    after=parameter_digests(agent.params)
    assert int(agent.n_updates)==4
    assert all(before[k]==after[k] for k,v in agent.model.f1_mutable_keys.items() if not v)
    assert all(any(before[k]!=after[k] for k in before if k.startswith(p))
               for p in ('enc/','dyn/','dec/','rew/','con/','avail/'))
    evidence=dict(scope='CONTROLLED_SYNTHETIC_ONLY',batch_size=1,target_steps=64,burnin_steps=32,
        updates=4,before=before,after=after,mutable=agent.model.f1_mutable_keys,
        shared_clocks=agent.model.f1_shared_clocks,metrics=metrics,real_TRAIN_updates=0)
    dest=Path(__file__).resolve().parents[2]/'docs/receipts/wp04/B_WM_update_evidence.json'
    if dest.exists():dest=dest.with_name(dest.stem+'_'+__import__('uuid').uuid4().hex+'.json')
    dest.write_text(json.dumps(evidence,indent=2)+'\n')


def test_B10_nonfinite_candidate_rejected_without_parameter_optimizer_or_counter_commit(wm_case):
    s,r,agent,initial=wm_case;agent.load_exact(initial)
    raw,_=F1WindowAdapter(r,s).sample()
    key='enc/minute/mlp0/kernel'
    agent.params[key]=jnp.full_like(agent.params[key],np.finfo(np.float32).max/2)
    before=parameter_digests(agent.params)
    # Native inputs stay legal/finite. Deliberately overflowing finite network
    # weights exercise candidate rejection, rather than Encoder's input guard.
    with pytest.raises(FloatingPointError,match='NONFINITE_UPDATE_REJECTED'):
        train_batch(agent,raw)
    assert before==parameter_digests(agent.params) and int(agent.n_updates)==0


def test_B11_original_weight_decay_and_momentum_cannot_move_frozen_leaf():
    from types import SimpleNamespace
    from dreamerv3.agent import Agent
    from dreamerv3.fg_s3_v2_agent import masked_joint_optimizer
    from dreamerv3.fg_s3_e_quality_course_contract import WM_GROUPS
    opt=Agent._make_opt(None,lr=.01,wd=.1,warmup=0)
    params={'enc/probe/kernel':jnp.ones((2,2)),'pol/probe/kernel':jnp.ones((2,2))}
    state=opt.init(params);grads={k:jnp.ones_like(v) for k,v in params.items()}
    _,state=opt.update(grads,state,params)  # Genuine nonzero inherited moments.
    before=jax.tree.map(np.asarray,state)
    holder=SimpleNamespace(opt=opt)
    with masked_joint_optimizer(holder,False,False,allowed_prefixes=WM_GROUPS):
        updates,after=holder.opt.update(grads,state,params)
    assert not np.any(updates['pol/probe/kernel']) and np.any(updates['enc/probe/kernel'])
    np.testing.assert_array_equal(before[1][1]['pol/probe/kernel'],after[1][1]['pol/probe/kernel'])
    np.testing.assert_array_equal(before[2][1]['pol/probe/kernel'],after[2][1]['pol/probe/kernel'])


def test_B12_signed_zero_is_a_native_byte_change_even_when_values_equal(tmp_path):
    from wp03_fixture import native_corpus
    from wp04_fixture import settings
    from gym_trading_env.research.fg_s3_e_quality_supervision import SupervisionSource,QualitySupervisionAdmission
    from gym_trading_env.research.fg_s3_e_quality_reader import build_join_index,EQualityCourseReader
    root=native_corpus(tmp_path/'corpus',minutes=16,chunk=20)
    a=QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(root));index=tmp_path/'join.sqlite'
    build_join_index(a,index)
    r=EQualityCourseReader.open(a,stage='WM_NATURAL',sampling_spec=dict(mode='NATURAL'),seed=202,index_path=index)
    try:
        adapter=F1WindowAdapter(r,settings(root));raw=r.sample(1,12);adapter.adapt(raw)
        x=raw['market_seq_e_shared'];where=tuple(np.argwhere(x==0)[0]);old=x.copy()
        x[where]=-np.copysign(np.float32(0.),x[where])
        assert np.array_equal(old,x) and old.tobytes()!=x.tobytes()
        with pytest.raises(ValueError,match='NATIVE_WINDOW_BYTES_MISMATCH'):adapter.adapt(raw)
    finally:r.close()
