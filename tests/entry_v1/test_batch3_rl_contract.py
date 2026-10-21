"""Explicit F4 admission policy and recoverable state contract."""
import copy
import pytest
from embodied.run.fg_s3_e_quality_online_rl import RLAdmissionState, validate_rl_policy


POLICY=dict(warmup_env_steps=32,calibration_batches=20,calibration_horizon=3,calibration_seed=202)

def test_explicit_finite_prefix_and_no_policy_for_bc_or_shadow():
    spec=dict(execution_mode='ONLINE_RL_CONTROLLED',env_steps=64,collect_every=16)
    assert validate_rl_policy(POLICY,spec)==POLICY
    for bad in (None,{}, {**POLICY,'warmup_env_steps':64},{**POLICY,'warmup_env_steps':17},
        {**POLICY,'calibration_batches':0},{**POLICY,'calibration_seed':True}):
        with pytest.raises(ValueError,match='EXPLICIT_BOUNDED'):validate_rl_policy(bad,spec)
    for mode in ('SHADOW','ONLINE_SUPERVISED'):
        assert validate_rl_policy(None,{**spec,'execution_mode':mode}) is None
        with pytest.raises(ValueError,match='ONLY_FOR_CONTROLLED'):validate_rl_policy(POLICY,{**spec,'execution_mode':mode})

def test_state_machine_rejects_skipping_checks_and_keeps_precise_snapshot():
    state=RLAdmissionState('RUN',POLICY)
    with pytest.raises(ValueError,match='ILLEGAL'):state.transition('RL_ACTIVE')
    state.transition('WM_CALIBRATING');state.state['attempt']=1
    state.transition('RL_PROBING');state.state['technical']=dict(evidence_hash='BOUND_SYNTHETIC_STATE_CONTRACT')
    state.transition('AWAIT_OPERATOR_REVIEW')
    snapshot=state.save();restored=RLAdmissionState('RUN',POLICY);restored.load(snapshot)
    assert restored.save()==snapshot
    bad=copy.deepcopy(snapshot);bad['technical']=None
    with pytest.raises(ValueError,match='TECHNICAL_EVIDENCE'):restored.load(bad)
    restored.transition('RL_ACTIVE');restored.transition('FINAL_VERIFY');restored.transition('COMPLETE_VERIFIED')
    with pytest.raises(ValueError,match='ILLEGAL'):restored.transition('WARMUP')
