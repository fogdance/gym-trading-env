from pathlib import Path
from fractions import Fraction
import numpy as np
import pytest
from wp02_fixture import native_factory
from gym_trading_env.research.fg_s3_e_quality_strict import QualityStrictTeacher,QualityP4Policy
from gym_trading_env.research.fg_s3_quality_q_v2 import CertifiedQEngine
from gym_trading_env.research.fg_s3_e_quality_corpus import generate_behavior
from gym_trading_env.research.fg_replay import iter_sidecar


def test_D01_quality_paired_values_match_independent_native_exhaustion(tmp_path):
    factory,_,_=native_factory(tmp_path/'market',minutes=20)
    env=factory();env.reset(seed=0)
    teacher=QualityStrictTeacher(env,source=dict(fold='TRAIN'))
    try:
        for _ in range(10):env.step(0)
        before=env.export_state()
        pair=teacher.query_values(env,env._published_mask)
        engine=CertifiedQEngine(factory,max_suffix_bars=12,max_states=10000,max_seconds=30)
        expected=engine.solve(env)
        assert pair['teacher_valid'] and expected['q_c1_pair_valid']
        assert pair['action_value_add']==pytest.approx(expected['q_c1'][1],abs=1e-12)
        assert pair['action_value_wait']==pytest.approx(expected['q_c1'][0],abs=1e-12)
        assert pair['strict_action']==expected['strict_action']==1
        assert env.export_state()==before
        engine.close()
    finally:env.close()


def test_D02_p4_complete_real_account_and_l2_own_validity(tmp_path):
    factory,_,_=native_factory(tmp_path/'market',minutes=30,chunk=9)
    builder=factory();builder.reset(seed=0)
    teacher=QualityStrictTeacher(builder,source=dict(fold='TRAIN'))
    spec=dict(id='P4',kind='quality_oracle_schedule',seed_components=[202,0,5],
              teacher_identity=teacher.identity_hash,hindsight_behavior=True)
    out=tmp_path/'p4'
    trace,receipt=generate_behavior(factory,spec,out,Path(__file__).resolve().parents[3]/'dreamerv3',
        source=dict(fold='TRAIN',state_cache_entries=0),strict=teacher)
    assert receipt['status']=='PASS' and trace['tp_events']>0
    assert trace['sl_events']==0
    actual=[r['cursor']-1 for r in iter_sidecar(out) for fill in r['fills'] if fill['reason']=='ENTRY']
    assert actual==teacher.plan()['add_cursors']
    labels=[r['teacher'] for r in iter_sidecar(out)]
    assert any(x['L2']['teacher_valid'] and x['L2']['strict_action']==1 for x in labels)
    assert any(x['L2']['teacher_valid'] and x['L2']['strict_action']==0 for x in labels)
    assert all(not x['L2']['teacher_valid'] for x in labels if x['L1'].get('invalid_reason'))
    assert teacher.plan()['native_validation']['selected_plan']['every_feedback_matched']
    with pytest.raises(ValueError,match='old C1'):
        QualityP4Policy({**spec,'kind':'oracle_schedule'},env=builder,teacher=teacher)
    builder.close()


def test_D03_forced_wait_and_unknown_boundary_never_action_labels(tmp_path):
    factory,path,_=native_factory(tmp_path/'market',minutes=20)
    env=factory();teacher=QualityStrictTeacher(env,source=dict(fold='TRAIN'))
    env.step(1)
    if env.net_lots:
        x=teacher.query_values(env,env._published_mask)
        assert not x['teacher_valid'] and x['strict_action'] is None
    import yaml
    config=yaml.safe_load(path.read_text());config['entry'].pop('main_exit_at',None)
    path.write_text(yaml.safe_dump(config))
    terms=path.parent/'contract.yaml';value=yaml.safe_load(terms.read_text())
    value['liquidate_at']='2026-01-07 15:00:00+08:00';terms.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError,match='right-censored'):
        QualityStrictTeacher(factory(),source=dict(fold='TRAIN'))
    env.close()


def test_D04_multiple_good_entries_are_not_all_strict_adds(tmp_path):
    from gym_trading_env.research.fg_s3_quality_teacher_v2 import NativeQualityTeacher
    factory,_,_=native_factory(tmp_path/'market',minutes=20)
    env=factory();env.reset(seed=0)
    for _ in range(6):env.step(0)
    strict=QualityStrictTeacher(env,source=dict(fold='TRAIN'),start_cursor=6)
    facts=NativeQualityTeacher(env,factory,fold='TRAIN',source={})
    l1=facts.query_state(env,env._published_mask)
    l2=strict.query_values(env,env._published_mask)
    assert l1['score']==100 and l1['exit_type']=='TP5'
    assert l2['teacher_valid'] and l2['strict_action']==0 and l2['tie']
    for _ in range(4):env.step(0)
    assert facts.query_state(env,env._published_mask)['score']==100
    assert strict.query_values(env,env._published_mask)['strict_action']==1
    assert all(t>=6 for t in strict.plan()['add_cursors'])
    facts.close();env.close()


def test_D05_low_funds_are_invalid_not_wait_and_old_reward_is_rejected(tmp_path):
    factory,path,_=native_factory(tmp_path/'market',minutes=20)
    env=factory();strict=QualityStrictTeacher(env,source=dict(fold='TRAIN'))
    env.step(1)
    while env.entries:env.step(0)
    value=strict.query_values(env,env._published_mask)
    assert not value['teacher_valid'] and value['strict_action'] is None
    assert 'FUNDS' in value['reason']
    import yaml
    config=yaml.safe_load(path.read_text());config['entry']['reward']['schema']='FG_SINGLE3_REWARD_MARKET_MINUTE_V2'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match='Quality TP'):
        QualityStrictTeacher(factory(),source=dict(fold='TRAIN'))
    env.close()
