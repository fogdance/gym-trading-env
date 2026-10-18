import copy

import numpy as np
import pytest

from test_wp03_reader import indexed,reader
from gym_trading_env.research.fg_s3_e_quality_checks import (
    cpu_model,cpu_forward,cpu_actor_terms,cpu_prediction_gradients,cpu_loss_evidence)


@pytest.fixture(scope='module')
def cpu_case(indexed,tmp_path_factory):
    r=reader(indexed,'ACTOR_QUALITY_PAIR')
    try:batch=r.sample(1,8)
    finally:r.close()
    agent=cpu_model(8,tmp_path_factory.mktemp('wp03_cpu_model'))
    return agent,batch


def test_D01_real_E_macd_quality_agent_loss_ready_without_update(cpu_case):
    from dreamerv3.fg_s3a_actor import parameter_digests
    agent,batch=cpu_case;before=parameter_digests(agent.params)
    result=cpu_forward(agent,batch)
    np.testing.assert_array_equal(result[2],batch['stepid'][:,1:])
    assert np.isfinite(result[0]) and np.isfinite(result[6]) and np.isfinite(result[4]).all()
    assert int(agent.n_updates)==0 and before==parameter_digests(agent.params)


def test_D02_effective_stepid_shift_is_rejected(cpu_case):
    agent,batch=cpu_case
    with pytest.raises(Exception,match='S3_V2_TEACHER_REPLAY_CONTEXT_SHIFT'):
        cpu_forward(agent,batch,effective_ids=np.roll(batch['stepid'][:,1:],1,axis=1))


@pytest.mark.parametrize('field',['quality_q_c1_valid','quality_q_v2_valid','quality_soft_valid','quality_soft_source'])
def test_D03_old_q_soft_cannot_enter_real_agent_loss(cpu_case,field):
    agent,batch=cpu_case;bad=copy.deepcopy(batch);bad[field][:,1]=1
    with pytest.raises(Exception,match='Q_SOFT_FORBIDDEN'):cpu_forward(agent,bad)


def test_D04_teacher_changes_only_supervision_not_observation_logits_or_reward(cpu_case):
    agent,batch=cpu_case;original=cpu_forward(agent,batch)
    changed=copy.deepcopy(batch);changed['teacher_action'][changed['teacher_valid']]=1-changed['teacher_action'][changed['teacher_valid']]
    changed['quality_score'][changed['quality_score_valid']]=100-changed['quality_score'][changed['quality_score_valid']]
    result=cpu_forward(agent,changed)
    np.testing.assert_array_equal(original[3],result[3]);np.testing.assert_array_equal(original[4],result[4])
    np.testing.assert_array_equal(batch['reward'],changed['reward'])
    for key in agent.obs_space:np.testing.assert_array_equal(batch[key],changed[key])
    grads=cpu_actor_terms(agent,original[3],batch)[2];other=cpu_actor_terms(agent,original[3],changed)[2]
    assert any(not np.array_equal(v,other[k]) for k,v in grads.items())


def test_D05_invalid_labels_have_zero_loss_and_parameter_gradients(cpu_case):
    from gym_trading_env.research.fg_s3_quality_catalog_v2 import FIELDS
    agent,batch=cpu_case;features=cpu_forward(agent,batch)[3]
    empty=copy.deepcopy(batch);empty['teacher_valid'][...]=False
    for k in FIELDS:empty[k][...]=0
    assert float(cpu_forward(agent,empty)[0])==0
    assert not any(np.any(v) for v in cpu_actor_terms(agent,features,empty)[2].values())
    assert not any(np.any(v) for v in cpu_prediction_gradients(agent,features,empty).values())


def test_D06_old_q_parameter_gradients_are_zero_even_with_strict_targets(cpu_case):
    agent,batch=cpu_case;result=cpu_forward(agent,batch)
    assert batch['teacher_valid'].any()
    grads=cpu_prediction_gradients(agent,result[3],batch)
    assert any(np.any(v) for k,v in grads.items() if k.startswith('quality_event/'))
    assert all(not np.any(v) for k,v in grads.items() if k.startswith(('quality_q_c1/','quality_q_v2/')))


def test_D07_nonfinite_invalid_target_rejected_before_jax(cpu_case):
    agent,batch=cpu_case;bad=copy.deepcopy(batch)
    bad['quality_q_v2'][...]=np.nan
    with pytest.raises(ValueError,match='NONFINITE_BATCH_BEFORE_JAX'):cpu_forward(agent,bad)


def test_D08_complete_read_only_evidence_and_cold_agent_restore(cpu_case,tmp_path):
    _,batch=cpu_case
    evidence=cpu_loss_evidence(batch,tmp_path)
    assert evidence['status']=='PASS_CPU_LOSS_ONLY'
    assert evidence['exact_cold_agent_restore'] and evidence['parameters_and_optimizer_unchanged']
    assert evidence['actual_optimizer_updates']==0 and not evidence['training_admitted']
    assert all(v['rejected'] for v in evidence['negative_cases'].values())
