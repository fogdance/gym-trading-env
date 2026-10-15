"""Current evidence contracts; fixture thresholds are not research approvals."""
import copy

import pytest

from gym_trading_env.research.fg_replay import signed,write_json
from gym_trading_env.research.fg_s3_quality_admission import (
    POLICY,PREDICTION,DECISION,Q_ADMISSION,validate_policy,_forecast_support,
    require_prediction_admission,require_decision_admission,require_q_admission,
    verify_native_q,relative_action_safety)
from gym_trading_env.research.fg_s3_quality_prediction_evaluation import quality_scores


def fixture_policy():
    """Small synthetic test thresholds. No production/default policy exists."""
    contracts=['CZCE.FG401','CZCE.FG405','CZCE.FG409']
    return signed(dict(protocol=POLICY,trained_contracts=['CZCE.FG201'],
        heldout_contracts=contracts[:2],replay_seeds=[202,203],target_replay_seed=202,
        time_folds=[dict(id='early',windows={c:dict(start_day='20260105',end_day='20260106') for c in contracts}),
                    dict(id='late',windows={c:dict(start_day='20260107',end_day='20260108') for c in contracts})],
        prediction=dict(minimum_samples_per_horizon=2,minimum_score_samples=2,minimum_binary_samples=2,
            minimum_official_days=2,minimum_tp_per_horizon=1,minimum_sl_per_horizon=1,
            minimum_low_funds_per_horizon=1,maximum_event_ece=.2,maximum_binary_ece=.2,minimum_sl_recall=.5),
        decision=dict(minimum_changed_decisions=1,minimum_strict_add=1,minimum_strict_wait=1,
            minimum_closed_entries=2,maximum_false_add_rate_delta=0.,maximum_sl_rate_delta=0.,
            maximum_drawdown_delta=0.,require_native_relative_value=True,
            component_sets=[['score'],['outcome','score']]),
        q=dict(minimum_verified_states=2,minimum_low_funds_states=1,minimum_official_days=2,
               maximum_value_mse=.1,minimum_advantage_accuracy=.8)))


def test_policy_requires_explicit_multifold_replicated_risk_thresholds():
    policy=fixture_policy();assert validate_policy(policy)==policy
    for field,value in (('heldout_contracts',['CZCE.FG401']),('replay_seeds',[202]),('time_folds',policy['time_folds'][:1])):
        bad=copy.deepcopy(policy);bad[field]=value
        with pytest.raises(ValueError):validate_policy(bad)
    bad=copy.deepcopy(policy);bad['prediction']['minimum_samples_per_horizon']=1
    with pytest.raises(ValueError,match='threshold'):validate_policy(bad)
    bad=copy.deepcopy(policy);del bad['prediction']['maximum_event_ece']
    with pytest.raises(ValueError,match='threshold'):validate_policy(bad)


@pytest.mark.parametrize('validate,protocol,status',[
    (require_prediction_admission,PREDICTION,'PASS_QUALITY_PREDICTION_ADMISSION'),
    (require_decision_admission,DECISION,'PASS_QUALITY_DECISION_ADMISSION'),
    (require_q_admission,Q_ADMISSION,'PASS_TRAIN_Q_FUSION_ADMISSION')])
def test_status_and_positive_counts_alone_cannot_admit(validate,protocol,status,tmp_path):
    path=tmp_path/'flags_only.json'
    write_json(path,signed(dict(protocol=protocol,status=status,fold='TRAIN_HOLDOUT',samples=100000,
                               q_v2_pair_valid_count=100000,short_run_verified=True)))
    with pytest.raises(ValueError,match='PENDING'):validate(path)


def prediction_rows():
    rows=[]
    for i in (0,1):
        label=dict(event=[i]*7,event_valid=[True]*7,score_valid=True,score=100 if not i else 0,
            entry_official_trading_date='2026010'+str(5+i),
            touch=[[1,0]]*4,touch_valid=[[True,True]]*4,adverse=[i]*4,adverse_valid=[True]*4)
        p=[.9,.03,.03,.04] if not i else [.03,.9,.03,.04]
        prediction={'quality/outcome_probs':[p]*7,'quality/score_prediction':90 if not i else 10,
            'quality/touch_probability':[[.9,.1]]*4,'quality/adverse_probability':[.1 if not i else .9]*4,
            'quality/delta_O':0.,'quality/delta_S':0.,'quality/delta_Q':0.,'quality/z_final':-.1}
        rows.append(dict(prediction=prediction,label=label,action=0,equity=298182))
    return rows


def test_every_horizon_needs_sl_funds_days_and_calibration_support():
    policy=fixture_policy();rows=prediction_rows();metrics=quality_scores(rows)
    _forecast_support(rows,metrics,policy)
    for horizon in ('0','6'):
        bad=copy.deepcopy(metrics);bad['horizons'][horizon]['class_counts'][1]=0
        with pytest.raises(ValueError,match='TP/SL/funds'):_forecast_support(rows,bad,policy)
    bad=copy.deepcopy(metrics);bad['horizons']['2']['calibration']['ece']=.9
    with pytest.raises(ValueError,match='calibration'):_forecast_support(rows,bad,policy)
    bad=copy.deepcopy(metrics);bad['horizons']['4']['by_funds']['below_300000']['samples']=0
    with pytest.raises(ValueError,match='TP/SL/funds'):_forecast_support(rows,bad,policy)


def test_native_q_proof_repeats_exactly_and_rejects_changed_value_or_account(tmp_path):
    from test_fg_s3_quality_teacher_v2 import quality_env
    from gym_trading_env.research.fg_s3_quality_q_v2 import CertifiedQEngine
    from gym_trading_env.envs.entry_contract import fingerprint
    from gym_trading_env.research.fg_s3_oracle_state_query import decision_state
    env,factory=quality_env(tmp_path,[(1000,1000,1000,1000)]*2+[(1000,1007,1000,1007)])
    engine=CertifiedQEngine(factory,max_seconds=20)
    q=engine.solve(env);state_id=fingerprint(decision_state(env))
    assert q['q_v2_pair_valid']
    verify_native_q(q,factory,state_id=state_id,snapshot_hash=fingerprint(env.export_state()))
    bad=copy.deepcopy(q);bad['q_v2'][1]+=1
    with pytest.raises(ValueError,match='exact witness'):verify_native_q(bad,factory,state_id=state_id)
    bad=copy.deepcopy(q);bad['root_snapshot_hash']='0'*64
    with pytest.raises(ValueError,match='snapshot'):verify_native_q(bad,factory,state_id=state_id)
    bad=copy.deepcopy(q);bad['transition_witness'][0]['configured_reward']='999'
    with pytest.raises(ValueError,match='witness bytes'):verify_native_q(bad,factory,state_id=state_id)
    engine.close();env.close()


def test_high_predicted_score_does_not_override_native_strict_wait_or_tie():
    from dreamerv3.fg_s3_quality_v2 import quality_extensions
    with pytest.raises(ValueError,match='DECISION_GATE_PENDING'):
        quality_extensions(dict(quality_stage='CONTROLLED_FULL',quality_prediction_admitted=True))
    unfused=quality_extensions(dict(quality_stage='CONTROLLED_FULL',quality_prediction_admitted=True,
                                   quality_fuse_outcome=False,quality_fuse_score=False))
    assert not unfused['quality_decision_admitted'] and not unfused['quality_q_admitted']
    row=dict(official_trading_date='20260105',mask=[1,1],is_last=False,action=0,decision_state_id='known',
        quality={'quality/base_policy_logit':-.1,'quality/diagnostic_delta_S':.5})
    # B=100 can still be a certified weak WAIT (including a native Q tie).
    audits={'known':dict(valid=True,action=0)}
    with pytest.raises(ValueError,match='relative WAIT/ADD'):
        relative_action_safety([row,{}],audits,['score'],('20260105','20260106'),
                               fixture_policy()['decision'],.05)
