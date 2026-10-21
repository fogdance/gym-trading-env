import copy
import json
from pathlib import Path

import numpy as np
import pytest

from test_wp07_data import live
from wp07_fixture import f3_source,exact


def test_actual_multistep_calibration_gates_and_formal_no_go(live):
    from gym_trading_env.research.fg_s3_e_quality_online_checks import calibration,Calibration,formal_mask_policy
    from embodied.run.fg_s3_e_quality_online_gate import OnlineGates
    from gym_trading_env.research.fg_replay import reconcile_trace,E_QUALITY_SCHEMA
    from gym_trading_env.research.fg_single3_evaluation import embodied_runtime
    root,c,p,r=live;p.publish();data,_=r.sample()
    record=calibration(c.agent,data,source_identity=r.identity,horizon=3)
    report=record.report
    assert report['horizon']==3 and report['all_model_and_optimizer_bytes_equal']
    assert report['finite'] and report['arrays']['true_reward']
    np.testing.assert_allclose(report['arrays']['true_continue'],332/333,rtol=1e-6)
    policy=formal_mask_policy(f3_source());g=OnlineGates('synthetic',policy,collection_root=root)
    assert not g.calibration(record)
    assert not g.ready['WM_VALID_FOR_RL'] and not g.mask.actor_enabled
    factory=lambda:type(c.env)(config_path=root/'market/env.yaml',e_profile=c.env._e_profile)
    from gym_trading_env.research.fg_s3_e_quality_online_evidence import verify_account
    evidence=verify_account(c,factory,root/'gate_trace',run_identity='synthetic',
        source_identity=r.identity,verifier_identity=dict(scope='SYNTHETIC_READ_ONLY'),market_step=48,final=False)
    assert g.collection(c,evidence)
    with pytest.raises(ValueError,match='ACTUAL_CALIBRATION'):g.calibration(Calibration(copy.deepcopy(report)))
    with pytest.raises(ValueError,match='GATE_CLOSED'):g.require_rl(None,authorized=True)
    state=g.save();restored=OnlineGates('synthetic',policy,collection_root=root);restored.load(state);exact(state,restored.save())
    forged=copy.deepcopy(state);forged['ready']['RL_NUMERIC_READY']=True
    with pytest.raises(ValueError,match='MISSING_EVIDENCE'):restored.load(forged)
    weak=copy.deepcopy(policy);weak['config']['prior_fpr']=.1
    with pytest.raises(ValueError,match='weakened'):OnlineGates('synthetic',weak)
    Path('docs/receipts/wp07/F_calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    Path('docs/receipts/wp07/F_gates.json').write_text(json.dumps(g.report(),indent=2)+'\n')


def test_exploration_inside_real_mask_and_restore_rng(live):
    from embodied.run.fg_s3_e_quality_online import OnlinePolicy
    _,c,p,r=live;data,_=r.sample();agent=c.agent
    obs={key:data[key][0:1,5].copy() for key in agent.obs_space}
    obs['is_last'][...]=False;obs['is_terminal'][...]=False;obs['is_first'][...]=False
    exploratory=OnlinePolicy(agent,exploration_rate=1.,seed=303)
    state=exploratory.save();actions=[]
    obs['action_mask'][...]=[1,1]
    for _ in range(12):
        carry,act,_=exploratory(agent.init_policy(1),obs);actions.append(int(act['action'][0]))
        assert int(np.asarray(carry[-1]['action']).reshape(-1)[0])==actions[-1]
    assert set(actions)=={0,1}
    exploratory.load(state)
    for expected in actions:
        _,act,_=exploratory(agent.init_policy(1),obs);assert act['action'][0]==expected
    obs['action_mask'][...]=[1,0]
    for _ in range(3):
        _,act,_=exploratory(agent.init_policy(1),obs);assert act['action'][0]==0
    deterministic=OnlinePolicy(agent,exploration_rate=0.,seed=303)
    rng=copy.deepcopy(deterministic.rng.bit_generator.state)
    _,act,_=deterministic(agent.init_policy(1),obs)
    assert act['action'][0]==0 and rng==deterministic.rng.bit_generator.state
