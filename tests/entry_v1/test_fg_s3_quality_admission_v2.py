"""Current evidence contracts; fixture thresholds are not research approvals."""
import copy

import pytest

from gym_trading_env.research.fg_replay import signed,write_json
from gym_trading_env.research.fg_s3_quality_admission import (
    POLICY,PREDICTION,DECISION,Q_ADMISSION,validate_policy,_forecast_support,_improvements,_account_risk,
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


def test_controlled_prepare_reads_train_split_from_bound_policy(tmp_path,monkeypatch):
    """Isolate the consumer contract; upstream research acceptance is stubbed."""
    from gym_trading_env.research import fg_s3_quality_admission as gates
    from gym_trading_env.research import fg_s3_quality_experiment_v2 as preparation
    from gym_trading_env.research.fg_replay import digest
    policy=fixture_policy();policy_path=tmp_path/'policy.json';write_json(policy_path,policy)
    admission=signed(dict(protocol=PREDICTION,status='PASS_QUALITY_PREDICTION_ADMISSION',
        checkpoint=str(tmp_path/'checkpoint'),checkpoint_manifest_sha256='1'*64,
        policy_sha256=digest(policy_path),evaluation_files={},comparisons=[]))
    assert 'trained_contracts' not in admission
    receipt_path=tmp_path/'admission.json';write_json(receipt_path,admission)
    monkeypatch.setattr(gates,'require_prediction_admission',lambda path:admission)
    monkeypatch.setattr(gates,'checkpoint_policy',lambda node:(policy,digest(policy_path),{}))
    rows=[dict(contract='CZCE.FG201',calendar_complete=False)]
    monkeypatch.setattr(preparation,'_derive_rows',lambda *args:(rows,{},'fixture',{'CZCE.FG201':{}}))
    class ReachedNativeBuilder(Exception):pass
    def stop_before_native_build(jobs):
        assert jobs[0][0]['contract']=='CZCE.FG201'
        raise ReachedNativeBuilder
    monkeypatch.setattr(preparation,'_build_members',stop_before_native_build)
    settings=dict(prepared_dir=str(tmp_path/'prepared'),quality_admission_policy=None,
        quality_stage='CONTROLLED_FULL',quality_prediction_admission_receipt=str(receipt_path),
        quality_prediction_admission_sha256=digest(receipt_path),env_slice_bars=100,
        quality_max_market_steps=None,quality_state_cache_entries=0)
    with pytest.raises(ReachedNativeBuilder):preparation.prepare_quality(settings)
    rows[0]['contract']='CZCE.FG205'
    with pytest.raises(ValueError,match='TRAIN split'):
        preparation.prepare_quality({**settings,'prepared_dir':str(tmp_path/'mismatched')})
    assert 'trained_contracts' not in admission


@pytest.mark.parametrize('window',['touch_3_1','adverse_3'])
def test_each_binary_window_must_beat_prior_and_same_state_reference(window):
    rows=prediction_rows();reference=copy.deepcopy(rows)
    prior=dict(event_counts=[[5,5,0,0]]*7,touch_counts=[[10,10]]*4,touch_positive=[[5,5]]*4,
               adverse_counts=[10]*4,adverse_positive=[5]*4,score_median=50)
    for i,row in enumerate(reference):
        p=row['prediction']
        p['quality/outcome_probs']=[[(v+.25)/2 for v in h] for h in p['quality/outcome_probs']]
        p['quality/score_prediction']=80 if not i else 20
        p['quality/touch_probability']=[[.85,.15]]*4
        p['quality/adverse_probability']=[.15 if not i else .85]*4
    _improvements(quality_scores(rows,prior),quality_scores(reference,prior))
    for i,row in enumerate(rows):
        if window.startswith('touch'):row['prediction']['quality/touch_probability'][3]=[.9,.16]
        else:row['prediction']['quality/adverse_probability'][3]=.16 if not i else .84
    metrics=quality_scores(rows,prior)
    _forecast_support(rows,metrics,fixture_policy())  # Still passes absolute calibration/support.
    with pytest.raises(ValueError,match=window):
        _improvements(metrics,quality_scores(reference,prior))


@pytest.fixture
def native_fold_risk(tmp_path):
    """A real SL crosses official days within the same natural date."""
    import pandas as pd
    import yaml
    from test_fg_s3_quality_teacher_v2 import quality_env
    from gym_trading_env.research.fg_single3 import load_env
    from gym_trading_env.research.fg_replay import digest
    flat=(1000,1000,1000,1000);sl=(1000,1000,965,980);tp=(1000,1007,1000,1007)
    times=pd.to_datetime(['2026-01-05 14:59','2026-01-05 15:00',
        *['2026-01-05 21:0'+str(i) for i in range(1,7)]])
    original,_=quality_env(tmp_path,[flat,flat,sl,flat,tp,flat,sl,flat],times=times,
        day=['20260105']*2+['20260106']*6,liquidation='2026-01-07 15:00',
        calendar=[dict(start='2026-01-05 14:58',end='2026-01-05 15:00'),
                  dict(start='2026-01-05 21:00',end='2026-01-05 21:06')])
    path=tmp_path/'market.csv';original._raw.to_csv(path,index_label='Date');original.close()
    config_path=tmp_path/'env.yaml';config=yaml.safe_load(config_path.read_text())
    config['trading']['data_path']=str(path);config_path.write_text(yaml.safe_dump(config))
    row=dict(contract=config['entry']['contract'],env_config=str(config_path),path=str(path),
             sha256=digest(path),contract_config=str(tmp_path/'contract.yaml'))
    context=dict(rows=[row],frozen_inputs={str(p):digest(p) for p in
        (path,config_path,tmp_path/'contract.yaml',tmp_path/'directions.yaml')})
    env=load_env(row,context['frozen_inputs']);tape=[];states=[]
    def capture():
        tape.append(dict(cursor=env.current_step,official_trading_date=env._trading_day(env._raw.index[env.current_step]),
                         equity=float(env._calculate_equity()),is_first=not tape))
        states.append(env.export_state())
    capture()
    for action in (1,0,1,0,1,0):env.step(action);capture()
    assert [e['reason'] for e in env.attributions]==['STOP_LOSS','TAKE_PROFIT','STOP_LOSS']
    assert env.attributions[0]['net_trade_pnl']=='-1818'
    try:yield tape,states,context,tmp_path
    finally:env.close()


def test_entry_fold_risk_follows_complete_trades_and_excludes_other_cohorts(native_fold_risk):
    tape,states,context,output=native_fold_risk
    a=_account_risk(tape,states[-1],context,('20260105','20260105'),1)
    b=_account_risk(tape,states[-1],context,('20260106','20260106'),2)
    assert a['sl_rate']==1 and a['drawdown']==1818  # The SL occurs after the entry fold.
    assert b['sl_rate']==.5 and b['drawdown']==1818  # Carry-in SL belongs exclusively to A.
    assert a['status']==b['status']=='COMPLETE_COHORT'
    assert a['post_fold_complete_entry_ids'] and not a['right_censored_entry_ids']
    assert a['open_at_fold_end_entry_ids']==a['post_fold_complete_entry_ids']
    assert b['post_fold_complete_entry_ids']==b['right_censored_entry_ids']==[]
    assert not b['fold_cutoff_observed'] and b['open_at_fold_end_entry_ids'] is None
    assert a['risk_basis']==b['risk_basis']=='ENTRY_FOLD_COMPLETE_TRADE_COHORT_V1'
    write_json(output/'entry_fold_risk_receipt.json',signed(dict(fold_a=a,fold_b=b)))


def test_fold_cutoff_open_trade_is_distinct_from_postfold_complete_outcome(native_fold_risk):
    tape,states,context,output=native_fold_risk
    risk=_account_risk(tape[:2],states[1],context,('20260105','20260105'),1)
    assert risk['status']=='RIGHT_CENSORED' and risk['sl_rate'] is None and risk['drawdown'] is None
    assert risk['right_censored_entry_ids']==risk['open_at_fold_end_entry_ids']
    assert risk['post_fold_complete_entry_ids']==[]
    partial=_account_risk(tape[:2],states[1],context,('20260105','20260106'),1)
    assert not partial['fold_cutoff_observed'] and partial['open_at_fold_end_entry_ids'] is None
    assert partial['right_censored_entry_ids']==risk['right_censored_entry_ids']
    write_json(output/'censored_fold_risk_receipt.json',signed(risk))


def test_complete_cohort_requires_native_equity_before_entry(native_fold_risk):
    tape,states,context,_=native_fold_risk
    with pytest.raises(ValueError,match='equity anchor'):
        _account_risk(tape[1:],states[-1],context,('20260105','20260105'),1)
