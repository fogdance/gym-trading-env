"""Real isolated Actor/critic update and rejection, on native synthetic windows."""
import copy
import json
from pathlib import Path
import random

import numpy as np
import pytest

from test_wp07_data import live
from test_wp07_rl import rl
from wp07_fixture import f3_source


def test_actual_candidate_updates_actor_and_critic_live_agent_and_rng_unchanged(rl):
    from embodied.run.fg_s3_e_quality_online_probe import isolated_rl_probe,require_numeric_probe,state_digest,NumericProbe
    from embodied.run.fg_s3_e_quality_online_gate import OnlineGates
    from gym_trading_env.research.fg_s3_e_quality_online_checks import formal_mask_policy
    root,agent,data=rl
    settings=copy.deepcopy(f3_source());settings['batch_size']=len(data['is_first'])
    before=agent.save_exact();py=random.getstate();nr=np.random.get_state()
    probe=isolated_rl_probe(before,settings,data,directory=root/'isolated_candidate',seed=202,
        controlled_small=True,context=dict(scope='NATIVE_SYNTHETIC_CANDIDATE_NOT_FORMAL'))
    report=require_numeric_probe(probe)
    assert 1<=report['actual_candidate_updates']<=2 and report['actor_changed'] and report['value_changed']
    assert report['frozen_equal'] and report['metrics']['f4/imag_invalid']==0
    assert state_digest(before)==state_digest(agent.save_exact())
    assert state_digest(py)==state_digest(random.getstate()) and state_digest(nr)==state_digest(np.random.get_state())
    gate=OnlineGates('SYNTHETIC',formal_mask_policy(settings))
    assert gate.numeric(probe) and gate.ready['RL_NUMERIC_READY']
    with pytest.raises(ValueError,match='GATE_CLOSED'):gate.require_rl(None,authorized=True)
    with pytest.raises(ValueError,match='ACTUAL_ISOLATED'):gate.numeric(NumericProbe(copy.deepcopy(probe.report)))
    Path('docs/receipts/wp07/BATCH3_ISOLATED_CANDIDATE.json').write_text(json.dumps(dict(
        scope='SYNTHETIC_REAL_NATIVE_NUMERIC_NOT_FORMAL',report=report,live_agent_unchanged=True,
        live_rng_unchanged=True,actual_live_rl_updates=0),indent=2)+'\n')


def test_nan_child_rejected_without_live_model_or_rng_pollution(rl):
    from embodied.run.fg_s3_e_quality_online_probe import isolated_rl_probe,state_digest
    root,agent,data=rl
    settings=copy.deepcopy(f3_source());settings['batch_size']=len(data['is_first'])
    bad=copy.deepcopy(data);bad['market_seq_e_shared'][...]=np.nan
    before=state_digest(agent.save_exact());py=state_digest(random.getstate());nr=state_digest(np.random.get_state())
    with pytest.raises(ValueError,match='ISOLATED_RL_PROBE_FAILED'):
        isolated_rl_probe(agent.save_exact(),settings,bad,directory=root/'nan_candidate',seed=202,
            controlled_small=True,context=dict(scope='INJECTED_NAN_REJECTION_ONLY'))
    assert before==state_digest(agent.save_exact())
    assert py==state_digest(random.getstate()) and nr==state_digest(np.random.get_state())


def test_actual_calibration_math_pass_and_fail_without_lowering_formal_thresholds(rl):
    """Declared synthetic causal tensors/IDs, not Reader or formal source authority.

    Constant heads make the native forward calculation independently predictable.
    This verifies actual calibration execution and the unchanged 25k/20-report
    gate. The real Reader-backed tiny WM is separately required to BLOCK a Run.
    """
    import jax.numpy as jnp
    from embodied.run.fg_s3_e_quality_online_gate import OnlineGates
    from embodied.run.fg_s3_e_quality_online_probe import state_digest
    from gym_trading_env.research.fg_s3_e_quality_online_checks import calibration,formal_mask_policy
    root,agent,data=rl;healthy=agent.save_exact()
    try:
        for key in agent.params:
            if key.startswith(('avail/','rew/','con/')):
                value=jnp.zeros_like(agent.params[key])
                if key.endswith('/bias') and key.startswith(('avail/','con/')):value=jnp.full_like(value,12.)
                agent.params[key]=value
        before=state_digest(agent.save_exact())
        batch={k:np.repeat(v[:1],256,axis=0) for k,v in data.items()}
        batch['action_mask'][...]=1;batch['action'][...]=0;batch['reward'][...]=0
        batch['is_terminal'][...]=False;batch['market_advance'][...]=1
        gate=OnlineGates('CALIBRATION_MATH_ONLY',formal_mask_policy(f3_source()),
            wm_policy=dict(max_reward_mae=.01,max_continue_mae=.01))
        reports=[]
        for number in range(20):
            ids=np.arange(number*batch['stepid'].shape[0]*batch['stepid'].shape[1],
                (number+1)*batch['stepid'].shape[0]*batch['stepid'].shape[1],dtype='>u8')
            batch['stepid'][...]=0
            batch['stepid'][...,-8:]=ids.view(np.uint8).reshape(*batch['stepid'].shape[:2],8)
            record=calibration(agent,batch,source_identity='SYNTHETIC_CAUSAL_MATH_NOT_A_CORPUS',horizon=3,seed=202+number)
            gate.calibration(record);reports.append(record.report)
        assert gate.ready['WM_VALID_FOR_RL'] and len(gate.samples)>=25000
        assert gate.mask.required_passes==20 and gate.mask.min_train_samples>=25000
        assert before==state_digest(agent.save_exact()) and not gate.formal_admitted
        batch['reward'][...]=3
        bad=calibration(agent,batch,source_identity='SYNTHETIC_CAUSAL_MATH_NOT_A_CORPUS',horizon=3)
        assert not gate.calibration(bad) and not gate.ready['WM_VALID_FOR_RL']
        assert before==state_digest(agent.save_exact())
        Path('docs/receipts/wp07/BATCH3_CALIBRATION_MATH.json').write_text(json.dumps(dict(
            scope='SYNTHETIC_TENSOR_MATH_NOT_READER_OR_SOURCE_ADMISSION',actual_calibration_calls=21,
            unique_native_shaped_stepids=len(gate.samples),thresholds_unchanged=True,
            native_forward_read_only=True,pass_report=reports[-1],failure_report=bad.report,
            formal_training_admitted=False),indent=2)+'\n')
    finally:agent.load_exact(healthy)
