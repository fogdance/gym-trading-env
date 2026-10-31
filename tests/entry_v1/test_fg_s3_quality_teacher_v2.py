"""Reviewed mathematical anchors and actual native financial proof contracts."""
import copy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from test_fg_single3_env import single_env
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from gym_trading_env.rewards.single3_reward import Single3RewardConfig, QUALITY_TP_SCHEMA
from gym_trading_env.research.fg_s3_quality_teacher_v2 import NativeQualityTeacher, wait_reasons
from gym_trading_env.research.fg_s3_quality_q_v2 import CertifiedQEngine, QualityOracle, build_quality_suffix
from gym_trading_env.research.fg_s3_oracle_state_query import decision_state
from gym_trading_env.research.fg_s3_quality_catalog_v2 import attach_quality,coverage,validate_label
from gym_trading_env.research.fg_replay import digest,signed


def quality_env(tmp_path, prices=None, *, day=None, **kw):
    old = single_env(tmp_path,prices,candidate='C1',fees=3,**kw)
    raw = old._raw.copy()
    raw['trading_day'] = day or ['20260105']*len(raw)
    config_path = tmp_path/'env.yaml'
    cfg = yaml.safe_load(config_path.read_text())
    cfg['entry']['reward'] = asdict(Single3RewardConfig('C1',.5,.4,3.,QUALITY_TP_SCHEMA))
    config_path.write_text(yaml.safe_dump(cfg))
    def factory(): return FixedDirectionEntryEnv(df=raw.copy(),config_path=str(config_path))
    return factory(), factory


def test_censor_unknown_and_independent_horizons(tmp_path):
    env,factory=quality_env(tmp_path,[(1000,1000,1000,1000)]*8)
    teacher=NativeQualityTeacher(env,factory,fold='TRAIN',source={})
    before=copy.deepcopy(env.export_state())
    label=teacher.query_state(env,env._published_mask,snapshot=decision_state(env))
    assert label['immediate_outcome_verified'] and label['exit_type']=='RIGHT_CENSORED'
    assert label['score'] is None and not label['score_valid']
    assert label['event'][0]==3 and label['event_valid']==[True,False,False,False,False,False,False]
    assert label['touch_valid'][0]==[True,True] and not any(label['touch_valid'][1])
    assert env.export_state()==before
    with pytest.raises(ValueError,match='TRAIN'): NativeQualityTeacher(env,factory,fold='DEV',source={})
    stale=decision_state(env); stale['equity']='1'
    with pytest.raises(ValueError,match='identity'): teacher.query_state(env,env._published_mask,snapshot=stale)
    teacher.close();env.close()


def test_original_env_parity_funds_boundary_and_forced_wait(tmp_path):
    prices=[(1000,1000,1000,1000),(1000,1000,965,980),
            (1000,1000,1000,1000),(1000,1008,1000,1007)]
    env,factory=quality_env(tmp_path,prices)
    teacher=NativeQualityTeacher(env,factory,fold='TRAIN',source={})
    sl=teacher.query_state(env,env._published_mask)
    assert sl['score']==0 and sl['net_pnl']=='-1818'
    env.step(1)
    assert float(env._calculate_equity())==298182 and list(env._published_mask)==[1,1]
    below=teacher.query_state(env,env._published_mask)
    assert below['immediate_outcome_verified'] and below['exit_type']=='TP' and below['score_valid']
    assert below['funds_after']==str(env._calculate_equity()+282)
    env.step(1)
    held=teacher.query_state(env,env._published_mask)
    assert held['invalid_reason']=='FORCED_WAIT_POSITION' and not held['score_valid']
    assert wait_reasons(held,dict(valid=False))['primary']=='FORCED_WAIT_POSITION'
    teacher.close();env.close()


def test_q_native_suffix_and_weak_wait_tie(tmp_path):
    env,factory=quality_env(tmp_path,[(1000,1000,1000,1000),
        (1000,1000,1000,1000),(1000,1007,1000,1007)])
    engine=CertifiedQEngine(factory,max_seconds=20)
    q=engine.solve(env)
    assert q['q_c1_pair_valid'] and q['q_v2_pair_valid']
    # Immediate ADD and WAIT->ADD both close on the same bar. Zero MAE,
    # one native feedback delay in both branches, so this is a genuine tie.
    assert q['q_c1'][0]==q['q_c1'][1] and q['strict_tie'] and q['strict_action']==0
    assert q['q_v2'][0]==pytest.approx((332/333)*1.25)
    assert q['q_v2'][1]==q['q_v2'][0]
    engine.close();env.close()


def test_cross_official_day_precedes_market_time_curve(tmp_path):
    times=pd.to_datetime(['2026-01-05 14:59','2026-01-05 15:00',
                          '2026-01-05 21:01','2026-01-05 21:02'])
    env,factory=quality_env(tmp_path,[(1000,1000,1000,1000)]*3+[(1000,1007,1000,1007)],
        times=times,day=['20260105','20260105','20260106','20260106'],
        calendar=[dict(start='2026-01-05 14:58',end='2026-01-05 15:00'),
                  dict(start='2026-01-05 21:00',end='2026-01-05 21:02')],liquidation='2026-01-05 21:03')
    teacher=NativeQualityTeacher(env,factory,fold='TRAIN',source={})
    label=teacher.query_state(env,env._published_mask)
    assert label['held_market_minutes']==3 and label['crossed_trading_days']==1
    assert label['score']==45 and label['exit_natural_minutes']>300
    assert label['original_c1_reward']!=label['configured_reward']
    teacher.close();env.close()


def test_q_censor_and_budget_are_not_numeric_targets(tmp_path):
    env,factory=quality_env(tmp_path,[(1000,1000,1000,1000)]*4)
    engine=CertifiedQEngine(factory,max_seconds=20)
    q=engine.solve(env)
    assert q['status']=='CENSORED' and not q['q_v2_pair_valid'] and q['q_v2'][1] is None
    engine.close()
    engine=CertifiedQEngine(factory,max_suffix_bars=1)
    assert engine.solve(env)['status']=='NO_PROOF'
    engine.close();env.close()


def test_rebuilt_new_reward_strict_suffix_matches_native(tmp_path):
    env,factory=quality_env(tmp_path,[(1000,1000,1000,1000),(1000,1000,1000,1000),
        (1000,1007,1000,1007),(1000,1000,1000,1000)])
    builder=factory()
    suffix=build_quality_suffix(builder,source={'kind':'synthetic_native'})
    assert suffix.identity['target_reward_schema']==QUALITY_TP_SCHEMA
    assert suffix.query(env,env._published_mask)['valid']
    teacher=NativeQualityTeacher(env,factory,fold='TRAIN',source={})
    oracle=QualityOracle(teacher,CertifiedQEngine(factory,max_seconds=20),suffix=suffix)
    answer=oracle.query(env,env._published_mask)
    assert oracle.last_label['strict_identity']==oracle.identity_hash
    assert answer['q_add'] is None  # D0 wire contract is preserved, numeric Q owns its own proof.
    oracle.close();builder.close();env.close()


def test_quality_first_layers_low_funds_and_native_replay_join(tmp_path):
    env,factory=quality_env(tmp_path,[(1000,1000,1000,1000),(1000,1000,965,980),
        (1000,1000,996,1000),(1000,1008,1000,1007)])
    builder=factory();suffix=build_quality_suffix(builder,source={})
    oracle=QualityOracle(NativeQualityTeacher(env,factory,fold='TRAIN',source={}),None,suffix=suffix)
    env.step(1)
    answer=oracle.query(env,env._published_mask)
    label=oracle.last_label
    assert float(label['equity'])==298182 and not answer['valid']
    assert label['score']==100  # MAE is independently observed, with no deduction.
    assert label['mae_full']==4 and label['strict_action'] is None
    assert label['event_valid'][5:]==[True,True]
    assert label['market_minutes_to_entry_day_end'] is None  # Exit is known; full day clock is not.
    assert label['tp_event_reward_new']==1 and label['new_reward_undiscounted']==pytest.approx(1)
    assert label['strict_certificate_floor']=='300000'
    assert [label['layers'][k]['valid'] for k in ('L0','L1','L2','L3')]==[True,True,True,False]
    assert not label['training_action_label_allowed'] and label['soft_target_source']=='INVALID'
    stepid=np.zeros((1,1,20),np.uint8)
    row=dict(stepid=stepid[0,0].tobytes().hex(),quality=label,state_id=label['state_id'],
        oracle_identity=oracle.identity_hash,oracle_valid=False,teacher_valid=False,
        execution_result=dict(advanced=True),action_mask=[1,1],episode_id='real-loss',
        contract=env.contract,equity_before=label['equity'])
    joined=attach_quality(dict(stepid=stepid,teacher_valid=np.zeros((1,1),bool)),[row],
                          teacher_identities={oracle.teacher.identity_hash})
    assert joined['quality_score_valid'][0,0] and joined['quality_event_valid'][0,0].any()
    assert not joined['teacher_valid'][0,0] and not joined['quality_soft_valid'][0,0]
    counts=coverage([row])['counts']
    assert counts['low_funds_flat_quality_valid']==1 and counts['low_funds_flat_strict_valid']==0
    tampered={k:v for k,v in label.items() if k!='content_hash'}
    tampered['score']=94.  # Re-signing an old MAE-adjusted target does not admit it as B.
    with pytest.raises(ValueError,match='frozen Quality-first B formula'):
        validate_label(signed(tampered))
    with pytest.raises(ValueError,match='TRAIN'):
        attach_quality(dict(stepid=stepid),[],teacher_identities=set(),fold='DEV')
    oracle.close();env.close();builder.close()


def test_quality_first_source_guard_detects_input_mutation(tmp_path):
    env,factory=quality_env(tmp_path,[(1000,1000,1000,1000)]*3)
    source=tmp_path/'frozen_input.txt';source.write_text('frozen')
    teacher=NativeQualityTeacher(env,factory,fold='TRAIN',source=dict(files={str(source):digest(source)}))
    source.write_text('changed')
    with pytest.raises(ValueError,match='Frozen quality source changed'):
        teacher.query_state(env,env._published_mask)
    # The frozen source guard also runs on close (including the unmodified
    # f0c3bb3 baseline); changing a source never makes cleanup accept it.
    with pytest.raises(ValueError,match='Frozen quality source changed'):
        teacher.close()
    env.close()


def test_quality_first_explicitly_refuses_unadmitted_e_profile(tmp_path):
    env,factory=quality_env(tmp_path,[(1000,1000,1000,1000)]*3)
    env._e_profile=object()
    with pytest.raises(ValueError,match='E-profile'):
        NativeQualityTeacher(env,factory,fold='TRAIN',source={})


def test_bounded_native_exposure_is_censored_and_calendar_is_private(tmp_path):
    prices=[(1000,1002,999,1000)]*8+[(1000,1008,1000,1007)]
    env,factory=quality_env(tmp_path,prices)
    teacher=NativeQualityTeacher(env,factory,fold='TRAIN',source=dict(max_market_steps=3))
    before=env.export_state();calendar=teacher.calendar_index
    label=teacher.query_state(env,env._published_mask)
    assert label['exit_type']=='RIGHT_CENSORED' and label['censor_reason']=='EXPOSURE_BUDGET'
    assert label['held_market_minutes']==3 and not label['score_valid']
    assert not any(label['event_valid'])
    assert label['touch_valid'][0][0] and label['touch'][0][0]==1
    assert env.export_state()==before and teacher.calendar_index is calendar
    assert not hasattr(env,'calendar_index')
    teacher.close();env.close()


def test_full_state_cache_is_private_and_never_aliases_returned_labels(tmp_path):
    env,factory=quality_env(tmp_path,[(1000,1000,1000,1000),(1000,1007,1000,1007)])
    teacher=NativeQualityTeacher(env,factory,fold='TRAIN',source={'state_cache_entries':1})
    state=copy.deepcopy(env.export_state())
    first=teacher.query_state(env,env._published_mask)
    expected=copy.deepcopy(first)
    first['layers']['L2']['valid']=False
    assert teacher.query_state(env,env._published_mask)==expected
    assert teacher.statistics['native_traces']==teacher.statistics['cache_hits']==1
    assert env.export_state()==state
    assert not any('cache' in k for k in env._get_obs())
    teacher.close();env.close()


def test_strict_parity_rejects_a_candidate_outside_old_uniform_sample(tmp_path,monkeypatch):
    from gym_trading_env.research.fg_single3_oracle import CandidateIndex
    original=CandidateIndex.candidate
    def corrupt(index,cursor):
        row=original(index,cursor)
        if cursor==1 and row.get('exit'):
            from decimal import Decimal
            row['exit']['net_trade_pnl']=str(Decimal(row['exit']['net_trade_pnl'])+1)
        return row
    env,_=quality_env(tmp_path,[(1000,1007,1000,1007)]*24)
    monkeypatch.setattr(CandidateIndex,'candidate',corrupt)
    with pytest.raises(RuntimeError,match='Indexed/reference exit mismatch'):
        build_quality_suffix(env,source={})
    env.close()
