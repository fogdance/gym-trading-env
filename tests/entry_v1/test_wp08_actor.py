import copy
import json
from pathlib import Path

import numpy as np
import pytest

from wp08_fixture import f3_reference,controlled_actor,ARTIFACTS


def test_actual_f3_readonly_loading_and_teacher_free_argmax(tmp_path,monkeypatch):
    from gym_trading_env.research.fg_s3_e_quality_exam_actor import ExamFrozenActor
    from wp02_fixture import native_factory
    from embodied.run.fg_s3_e_quality_exam import inspect_exam_actor
    s=f3_reference();node=Path(s['run_dir'])/'ckpt/step_000000000004'
    actor=ExamFrozenActor.from_f3_checkpoint(s['run_dir'],node,tmp_path/'model',engineering=True)
    def forbidden(*a,**k):raise AssertionError('TEACHER_OR_TRAINING_USED')
    monkeypatch.setattr(actor.agent,'train',forbidden)
    monkeypatch.setattr('gym_trading_env.research.fg_s3_e_quality_strict.QualityStrictTeacher',forbidden)
    monkeypatch.setattr('gym_trading_env.research.fg_s3_quality_teacher_v2.NativeQualityTeacher',forbidden)
    factory,_,_=native_factory(tmp_path/'market',minutes=60,chunk=0)
    env=factory()
    try:
        obs=env.reset(seed=202)[0]
        obs=dict(obs,reward=np.float32(0),is_first=True,is_last=False,is_terminal=False)
        batched={k:np.asarray(v)[None] for k,v in obs.items()}
        state=actor.save();carry,action,_=actor(None,batched)
        actor.load(state);carry2,action2,_=actor(None,batched)
        np.testing.assert_array_equal(action['action'],action2['action'])
        from wp07_fixture import exact
        import jax
        exact(jax.device_get(carry),jax.device_get(carry2))
        # Original pol is not a policy source. Mutate it in this one negative
        # probe and restore before asserting the frozen model invariant.
        pol={k:v for k,v in actor.agent.policy_params.items() if k.startswith('pol/')}
        import jax.numpy as jnp
        for k,v in pol.items():actor.agent.policy_params[k]=jnp.full_like(v,999.)
        actor.agent.load_policy_state(state['inference'])
        _,changed,_=actor(None,batched)
        np.testing.assert_array_equal(action['action'],changed['action'])
        actor.agent.policy_params.update(pol)
        assert actor.verify_frozen()['all_parameter_bytes_equal']
        Path('docs/receipts/wp08/B_f3_readonly.json').write_text(json.dumps(dict(
            receipt=actor.receipt,frozen=actor.verify_frozen(),action=int(action['action'][0]),
            teacher_calls=0,optimizer_calls=0,source_run_unchanged=True),indent=2)+'\n')
        with pytest.raises(ValueError):inspect_exam_actor('F3',s['run_dir'],node,engineering=False)
        with pytest.raises(ValueError):inspect_exam_actor('F4',s['run_dir'],node,engineering=True)
    finally:env.close()


def test_policy_rng_state_and_frozen_mutation_rejection(tmp_path):
    actor=controlled_actor(tmp_path/'model')
    with pytest.raises(RuntimeError,match='TRAINING_FORBIDDEN'):actor.agent.train(None,None)
    state=actor.save();actor.load(state)
    with pytest.raises(ValueError,match='NOT_JSON'):actor.require_mode(dict(exam_mode='DEV_EXAM'))
    with pytest.raises(ValueError):actor.agent.load_policy_state({**state['inference'],'seed':-1})
    key=next(k for k in actor.agent.params if k.startswith('val/'))
    actor.agent.params[key]=actor.agent.params[key]+1
    with pytest.raises(ValueError,match='STATE_CHANGED'):actor.verify_frozen()


def test_corrupt_actual_source_checkpoint_rejected_before_model_construction(tmp_path,monkeypatch):
    import shutil
    from gym_trading_env.research import fg_s3_e_quality_actor as construction
    from gym_trading_env.research.fg_s3_e_quality_exam_actor import ExamFrozenActor
    s=f3_reference();clone=tmp_path/'clone';shutil.copytree(s['run_dir'],clone)
    node=clone/'ckpt/step_000000000004';p=node/'agent.pkl'
    p.write_bytes(p.read_bytes()+b'CORRUPT')
    def forbidden(*a,**k):raise AssertionError('MODEL_CREATED_BEFORE_SOURCE_AUTHENTICATION')
    monkeypatch.setattr(construction,'create_agent',forbidden)
    with pytest.raises(ValueError,match='corrupt file'):ExamFrozenActor.from_f3_checkpoint(clone,node,tmp_path/'model',engineering=True)
    assert not (tmp_path/'model').exists()


@pytest.mark.parametrize('mode',['SHADOW','ONLINE_SUPERVISED','ONLINE_RL_CONTROLLED'])
def test_actual_f4_complete_schema_and_value_preserved(tmp_path,mode):
    from wp08_fixture import RELEASE
    from gym_trading_env.research.fg_s3_e_quality_exam_actor import ExamFrozenActor
    from gym_trading_env.research.fg_replay import read_manifest
    import pickle
    root=RELEASE/'Release_G_native_01'/mode/'continuous'
    node=root/'ckpt/step_000000000003'
    actor=ExamFrozenActor.from_f4_checkpoint(root,node,tmp_path/'model',engineering=True)
    state=pickle.loads((node/'agent.pkl').read_bytes())
    from dreamerv3.fg_s3a_actor import parameter_digests
    assert actor.before==parameter_digests(state['params'])
    assert any(k.startswith('val/') for k in actor.before)
    assert any(k.startswith('slowval/') for k in actor.before)
    assert actor.source['execution_mode']==mode
    assert actor.verify_frozen()['source_files_unchanged']
    Path(f'docs/receipts/wp08/B_f4_{mode}.json').write_text(json.dumps(dict(
        source=actor.source,frozen=actor.verify_frozen(),all_state_leaves_preserved=True),indent=2)+'\n')
