import copy,json
from pathlib import Path
from types import SimpleNamespace
import jax,jax.numpy as jnp
import numpy as np
import pytest
from wp06_fixture import engineering_settings
from gym_trading_env.research.fg_s3_e_quality_actor import create_agent,preflight,F3WindowAdapter
from gym_trading_env.research.fg_s3_e_quality_reader import EQualityCourseReader,build_join_index
from gym_trading_env.research.fg_s3_v2_gpu_check import gpu_batch
from dreamerv3.fg_s3_e_quality_actor import GROUPS,STAGE,LABELS,register_state_schema,read_only_forward
from dreamerv3.fg_s3a_actor import parameter_digests

@pytest.fixture(scope='module')
def case(tmp_path_factory):
    root=tmp_path_factory.mktemp('wp06_update');s=engineering_settings(root);p=preflight(s)
    index=root/'join.sqlite';build_join_index(p.source,index)
    r=EQualityCourseReader.open(p.source,stage='ACTOR_STRICT',sampling_spec=s['sampling_spec'],seed=202,index_path=index)
    adapter=F3WindowAdapter(r,s);raw=r.sample(s['batch_size'],12);data,metadata=adapter.adapt(raw)
    a=create_agent(s,root/'model',controlled_small=True)
    yield a,a.save_exact(),data,r,adapter,raw,metadata
    r.close()

def train(a,data,seed=202):
    return a.train(a.init_train(len(data['is_first'])),gpu_batch(a,copy.deepcopy(data),seed))

def supervised(data,action=None):
    """Explicit synthetic post-adapter loss probes, never authenticated labels."""
    d=copy.deepcopy(data);d['teacher_valid'][...]=True
    d['teacher_action'][...]=np.arange(d['teacher_action'].shape[1])%2 if action is None else action
    return d

def test_C01_actual_actor_update_complete_parameter_and_slot_freeze(case):
    a,initial,data,*_=case;a.load_exact(initial);d=copy.deepcopy(data);before=parameter_digests(a.params)
    train(a,d);_,out,m=train(a,d,203);after=parameter_digests(a.params)
    assert m['f3/finite'] and m['f3/updated'] and int(a.n_updates)==2
    assert any(before[k]!=after[k] for k in before if k.startswith(GROUPS))
    assert any(before[k]!=after[k] for k,o in a.model.f3_state_owner.items() if o==GROUPS[0] and k.startswith('opt/'))
    assert all(before[k]==after[k] for k,o in a.model.f3_state_owner.items() if o is None)
    np.testing.assert_array_equal(out['effective_stepid'],d['stepid'][:,5:])
    Path('docs/receipts/wp06/C_actor_delta.json').write_text(json.dumps(dict(before=before,after=after,
        state_owner=a.model.f3_state_owner,metrics={k:np.asarray(v).tolist() for k,v in m.items()},
        engineering_only=True,labels='ORIGINAL_READER_L2_UNMODIFIED'),indent=2))

def test_C02_no_valid_labels_freezes_all_state_including_inherited_slots(case):
    a,initial,data,*_=case;a.load_exact(initial);train(a,supervised(data));d=supervised(data)
    d['teacher_valid'][...]=False;d['teacher_action'][...]=999
    d['quality_score_valid'][...]=False;d['quality_score'][...]=np.nan
    before=parameter_digests(a.params);count=int(a.n_updates)
    _,_,m=train(a,d)
    assert m['f3/skip_no_valid'] and not m['f3/updated'] and m['opt/updates']==count
    assert parameter_digests(a.params)==before and int(a.n_updates)==count

def test_C03_teacher_never_changes_features_and_burnin_never_supervised(case):
    a,initial,data,*_=case;a.load_exact(initial);d=supervised(data,1);one=read_only_forward(a,d)
    changed=supervised(d,0);changed['quality_score'][...]=99;two=read_only_forward(a,changed)
    assert one['features'].tobytes()==two['features'].tobytes() and one['logits'].tobytes()==two['logits'].tobytes()
    assert one['loss']!=two['loss']
    changed=copy.deepcopy(d)
    for k in LABELS:
        if k!='f3_natural':changed[k][:,:5]=0
    assert read_only_forward(a,changed)['loss'].tobytes()==one['loss'].tobytes()
    history=copy.deepcopy(d);history['market_seq_e_shared'][:,1:5]*=-1
    assert read_only_forward(a,history)['features'].tobytes()!=one['features'].tobytes()

def test_C04_nonzero_frozen_momentum_and_negative_zero_survive(case):
    a,initial,data,*_=case;a.load_exact(initial)
    for k,o in a.model.f3_state_owner.items():
        if o is None:a.params[k]=jnp.full_like(a.params[k],.125 if k.startswith('opt/') else -0.)
    before=parameter_digests(a.params);train(a,supervised(data));after=parameter_digests(a.params)
    assert all(before[k]==after[k] for k,o in a.model.f3_state_owner.items() if o is None)

def test_C05_nonfinite_candidate_entire_state_rollback(case):
    a,initial,data,*_=case;a.load_exact(initial)
    for k in a.params:
        if k.startswith(GROUPS):a.params[k]=jnp.full_like(a.params[k],1e38)
    before=parameter_digests(a.params)
    with pytest.raises(FloatingPointError,match='F3_NONFINITE'):train(a,supervised(data))
    assert parameter_digests(a.params)==before and int(a.n_updates)==0

def test_C06_unknown_slots_or_dynamic_wm_rule_rejected(case):
    import optax
    from dreamerv3.fg_s3_v2_agent import masked_joint_optimizer
    a,initial,*_=case
    for key in ('alien/kernel','opt/state/1/1/alien/kernel'):
        with pytest.raises(ValueError):register_state_schema(SimpleNamespace(config=a.model.config),{**initial['params'],key:np.zeros(2,np.float32)})
    opt=SimpleNamespace(opt=optax.sgd(.1));params={'enc/kernel':jnp.ones(2),'quality_actor/kernel':jnp.ones(2)}
    state=opt.opt.init(params)
    with masked_joint_optimizer(opt,False,False,{GROUPS[0]:True,'enc/':True},allowed_prefixes=GROUPS,stage=STAGE):
        with pytest.raises(ValueError,match='OUTSIDE_ALLOWLIST'):opt.opt.update(params,state,params)

@pytest.mark.parametrize('field',['teacher_action','teacher_valid','quality_owner','quality_score'])
def test_C07_canonical_post_step_labels_reject_shift_or_owner_forgery(case,field):
    *_,adapter,raw,metadata=case
    d=copy.deepcopy(raw);d[field][:,5]+=1 if field!='teacher_valid' else False
    if field=='teacher_valid':d[field][:,5]=~d[field][:,5]
    before=adapter.reader.state_dict()
    with pytest.raises(ValueError,match='POST_STEP_LABEL_MISMATCH'):adapter.adapt(d)
    assert adapter.reader.state_dict()==before
    assert all(len(m['rows'])==13 for m in metadata)

def test_C08_q_soft_and_bad_native_window_reject_before_model(case):
    *_,adapter,raw,_=case
    for field in ('quality_q_v2_valid','quality_soft_valid'):
        d=copy.deepcopy(raw);d[field][...]=True
        with pytest.raises(ValueError,match='Q_SOFT_TARGETS_FORBIDDEN'):adapter.adapt(d)
    d=copy.deepcopy(raw);d['stepid'][:,5]=d['stepid'][:,4]
    with pytest.raises(ValueError,match='ORDINAL_MISMATCH'):adapter.adapt(d)

def test_C09_reset_terminal_single_action_masks_never_become_wait_truth(case):
    a,initial,data,*_=case;a.load_exact(initial)
    d=supervised(data,0);d['action_mask'][...,1]=0
    before=parameter_digests(a.params);_,_,m=train(a,d)
    assert not m['f3/updated'] and m['f3/valid_count']==0 and before==parameter_digests(a.params)
    d=supervised(data,1);d['is_terminal'][...]=True
    _,_,m=train(a,d);assert not m['f3/updated'] and m['f3/valid_count']==0
