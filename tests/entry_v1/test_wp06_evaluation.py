import copy,json,os
from pathlib import Path
import jax,jax.numpy as jnp,ninjax as nj
import numpy as np
import pytest
from wp06_fixture import engineering_settings
from gym_trading_env.research.fg_s3_e_quality_actor import create_agent,preflight,F3WindowAdapter
from gym_trading_env.research.fg_s3_e_quality_reader import EQualityCourseReader,build_join_index
from gym_trading_env.research.fg_s3_e_quality_actor_checks import score_batch,evaluate_native
from dreamerv3.fg_s3_e_quality_actor import read_only_forward,distribution
from dreamerv3.fg_s3a_actor import parameter_digests
from test_wp06_updates import train

@pytest.fixture(scope='module')
def case(tmp_path_factory):
    root=Path(os.environ['FG_WP06_EVAL_ROOT']) if 'FG_WP06_EVAL_ROOT' in os.environ else tmp_path_factory.mktemp('wp06_eval')
    root.mkdir(parents=True,exist_ok=True);s=engineering_settings(root,target=64,burnin=32)
    p=preflight(s);index=root/'join.sqlite';build_join_index(p.source,index)
    r=EQualityCourseReader.open(p.source,stage='ACTOR_STRICT',sampling_spec=s['sampling_spec'],seed=202,index_path=index)
    try:data,_=F3WindowAdapter(r,s).sample()
    finally:r.close()
    a=create_agent(s,root/'model',controlled_small=True)
    return root,a,data,s

def test_E01_training_inference_same_distribution_mask_and_no_predictions(case,monkeypatch):
    _,a,data,_=case;out=read_only_forward(a,data);mask=data['action_mask'][:,33:]
    def forbidden(*args,**kwargs):raise AssertionError('F3 called frozen prediction heads')
    monkeypatch.setattr('dreamerv3.fg_s3_quality_v2.predict',forbidden)
    def fn(feat,mask):
        from dreamerv3.fg_s3_quality_v2 import policy_forward
        p,raw=distribution(a.model,feat,mask,2);q,d,pred=policy_forward(a.model,feat,mask,2)
        assert not pred
        return p[a.model.action_key].logits,q[a.model.action_key].logits,d
    _,(p,q,d)=nj.pure(fn)(a.params,jnp.asarray(out['features']),jnp.asarray(mask),seed=202,create=False,modify=True)
    np.testing.assert_array_equal(p,q)
    assert all(np.all(np.asarray(d[k])==0) for k in ('quality/delta_O','quality/delta_S','quality/delta_Q'))
    expected=np.argmax(np.asarray(q),-1)
    np.testing.assert_array_equal(expected,d['quality/argmax_action'])

def test_E02_original_model_learns_both_synthetic_classes_with_frozen_wm(case):
    root,a,data,s=case;d=copy.deepcopy(data);start=33
    before=parameter_digests(a.params);features=read_only_forward(a,d)['features']
    legal=((d['action_mask'][:,start:]==1).all(-1) & ~d['is_first'][:,start:] & ~d['is_last'][:,start:] & ~d['is_terminal'][:,start:])
    # A declared synthetic, linearly learnable task on native frozen features.
    # These are NOT certified L2 labels and are never published to the Corpus.
    coordinate=int(np.argmax(features[...,:32][legal].var(0)));x=features[...,coordinate]
    y=(x>np.median(x[legal])).astype(np.int32)
    d['teacher_action'][:,start:]=y;d['teacher_valid'][:,start:]=legal
    fresh=score_batch(d,read_only_forward(a,d),start)
    assert fresh['valid_add'] and fresh['valid_wait']
    _,_,first=train(a,d,202)
    np.testing.assert_allclose(first['opt/loss'],fresh['loss'],rtol=2e-6,atol=2e-7)
    history=[float(first['opt/loss'])]
    for update in range(1,512):
        _,_,metrics=train(a,d,202)
        if update%64==0:history.append(float(metrics['opt/loss']))
    out=read_only_forward(a,d);trained=score_batch(d,out,start);after=parameter_digests(a.params)
    report=dict(status='SYNTHETIC_FROZEN_FEATURE_CLASSIFICATION_ONLY',updates=512,coordinate=coordinate,
        fresh=fresh,trained=trained,train_loss_history=history,before=before,after=after,
        frozen_byte_equal=all(before[k]==after[k] for k,v in a.model.f3_state_owner.items() if v is None),real_TRAIN_updates=0,
        source_labels='SYNTHETIC_FEATURE_MEDIAN_NOT_CERTIFIED_L2',research_admitted=False)
    Path('docs/receipts/wp06/E_learnability.json').write_text(json.dumps(report,indent=2))
    assert trained['loss']<fresh['loss']
    assert trained['confusion']['tp']>0 and trained['confusion']['tn']>0
    assert report['frozen_byte_equal']

def test_E03_original_driver_native_quality_reward_and_complete_account_replay(case,monkeypatch):
    root,a,_,_=case
    from wp02_fixture import native_factory
    from gym_trading_env.research.fg_replay import iter_sidecar
    def forbidden(*args,**kwargs):raise AssertionError('Teacher/Oracle must not be constructed')
    monkeypatch.setattr('gym_trading_env.research.fg_s3_e_quality_strict.QualityStrictTeacher',forbidden)
    monkeypatch.setattr('gym_trading_env.research.fg_s3_quality_teacher_v2.NativeQualityTeacher',forbidden)
    factory,_,_=native_factory(root/'actor_market',minutes=40,chunk=9)
    result=evaluate_native(a,factory,root/'actor_trace',steps=32,scope='SYNTHETIC_DEV_ONLY')
    assert result['complete_account_replay']['status']=='PASS' and not result['teacher_used']
    assert result['native_reward_schema']=='FG_SINGLE3_REWARD_MARKET_MINUTE_TP_QUALITY_V1'
    assert result['real_TRAIN_updates']==0 and result['all_parameter_bytes_equal']
    assert result['complete_account_replay']['records']>32
    assert not any('teacher' in r for r in iter_sidecar(root/'actor_trace'))
    Path('docs/receipts/wp06/E_native_actor.json').write_text(json.dumps(result,indent=2))

def test_E04_evaluation_refuses_wrong_stage_and_invalid_budget(case,tmp_path):
    _,a,_,_=case
    with pytest.raises(ValueError,match='BOUNDED_EVALUATION_STEPS_REQUIRED'):
        evaluate_native(a,lambda:None,tmp_path/'bad',steps=100000)
    assert not (tmp_path/'bad').exists()
    from types import SimpleNamespace
    wrong=SimpleNamespace(config={'quality_f3_protocol':'FG_S3_E_QUALITY_HEADS_V1'})
    with pytest.raises(ValueError,match='ACTOR_CHECKPOINT_REQUIRED'):
        evaluate_native(wrong,lambda:None,tmp_path/'wrong_stage',steps=1)
    assert not (tmp_path/'wrong_stage').exists()
