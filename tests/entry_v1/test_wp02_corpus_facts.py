from pathlib import Path
import copy
import numpy as np
import pytest
from wp02_fixture import native_factory
from gym_trading_env.research.fg_s3_e_quality_corpus import behavior_specs,generate_behavior
from gym_trading_env.research.fg_s3_quality_teacher_v2 import NativeQualityTeacher
from gym_trading_env.research.fg_replay import iter_sidecar


def test_C01_native_e_fork_isolated_complete_l1_and_underwater(tmp_path):
    factory,_,_=native_factory(tmp_path/'market',minutes=30)
    env=factory();before=copy.deepcopy(env.export_state())
    teacher=NativeQualityTeacher(env,factory,fold='TRAIN',source=dict(fold='TRAIN'))
    try:
        label=teacher.query_state(env,env._published_mask)
        assert label['exit_type']=='SL30' and label['score']==0
        assert label['mae_15']>=0 and label['underwater_minutes']==0
        assert label['underwater_observed_minutes']==2 # settled exit minute excluded
        assert env.export_state()==before
        assert env._e_market is not teacher.clone._e_market
        assert not np.shares_memory(env._features,teacher.clone._features)
        label['score']=-9
        assert teacher.query_state(env,env._published_mask)['score']==0
        env._e_profile.identity='f'*64
        with pytest.raises(ValueError,match='identity changed'):teacher.query_state(env,env._published_mask)
    finally:teacher.close();env.close()


@pytest.mark.parametrize('index',range(5))
def test_C02_p0_p3_original_account_replay_and_separate_labels(tmp_path,index):
    factory,_,_=native_factory(tmp_path/'market',minutes=30,chunk=9)
    spec=behavior_specs(17)[index]
    out=tmp_path/spec['id']
    trace,receipt=generate_behavior(factory,spec,out,Path(__file__).resolve().parents[3]/'dreamerv3',
        source=dict(fold='TRAIN',state_cache_entries=0))
    assert receipt['status']=='PASS'
    rows=list(iter_sidecar(out))
    assert all(r['teacher']['account_snapshot_hash']==r['account_snapshot_hash'] for r in rows)
    if index==0:
        assert trace['tp_events']==trace['sl_events']==0
        assert all(r['action']==0 for r in rows)
    if index==1:assert trace['tp_events']>0 and trace['sl_events']>0
    for row in rows:
        if row['lots_after'] or row['is_last']:
            assert not row['teacher']['L1']['score_valid']
        assert row['teacher']['L2']['strict_action'] is None
    if index==4:
        assert all(r['p3_waited']>=15 for r in rows if r['action']==1)


def test_C03_reject_nontrain_before_constructing_env(tmp_path):
    def forbidden():raise AssertionError('opened forbidden market')
    with pytest.raises(ValueError,match='TRAIN'):
        generate_behavior(forbidden,behavior_specs()[0],tmp_path/'bad','.',source=dict(fold='TEST'))


def test_C04_p2_seed_reproducibility_and_variation(tmp_path):
    factory,_,_=native_factory(tmp_path/'market',minutes=30)
    actions=[];hashes=[]
    for number,seed in enumerate([17,17,18]):
        out=tmp_path/str(number)
        trace,_=generate_behavior(factory,behavior_specs(seed)[3],out,
            Path(__file__).resolve().parents[3]/'dreamerv3',source=dict(fold='TRAIN',state_cache_entries=0))
        actions.append([r['action'] for r in iter_sidecar(out)])
        hashes.append(trace['event_hash'])
    assert actions[0]==actions[1] and hashes[0]==hashes[1]
    assert actions[0]!=actions[2]


def test_C05_aliasing_fork_never_admitted(tmp_path):
    factory,_,_=native_factory(tmp_path/'market',minutes=10)
    env=factory()
    try:
        with pytest.raises(ValueError,match='aliases'):
            NativeQualityTeacher(env,lambda:env,fold='TRAIN',source={})
    finally:env.close()


def test_C06_persistent_old_state_path_is_never_written(tmp_path):
    factory,_,_=native_factory(tmp_path/'market',minutes=10)
    target=tmp_path/'old_state.json'
    def unsafe_factory():
        env=factory();env.entry_config['state_path']=str(target);return env
    with pytest.raises(ValueError,match='state_path'):
        generate_behavior(unsafe_factory,behavior_specs()[0],tmp_path/'output','.',source=dict(fold='TRAIN'))
    assert not target.exists() and not (tmp_path/'output').exists()


def test_C07_coverage_separates_market_accounts_and_counterfactuals(tmp_path):
    from gym_trading_env.research.fg_s3_e_quality_corpus import coverage
    factory,_,_=native_factory(tmp_path/'market',minutes=12)
    paths=[]
    for spec in behavior_specs()[:2]:
        out=tmp_path/spec['id'];paths.append(out)
        generate_behavior(factory,spec,out,Path(__file__).resolve().parents[3]/'dreamerv3',
                          source=dict(fold='TRAIN',state_cache_entries=0))
    report=coverage(paths)
    assert report['unique_market_cursors']==12 and report['unique_advanced_market_minutes']==11
    assert report['distinct_account_decision_states']>12 and report['optimizer_exposures']==0
    assert report['behaviors']['P0']['counts']['filled_add']==0
    assert report['behaviors']['P1']['counts']['filled_add']>0
    assert report==coverage(paths)
    for behavior in report['behaviors'].values():
        assert sum(s['replay_rows'] for s in behavior['by_official_day_session'].values())==behavior['counts']['replay_rows']
    from gym_trading_env.research.fg_s3_e_quality_corpus import write_coverage_csv
    a,b=tmp_path/'a.csv',tmp_path/'b.csv'
    write_coverage_csv(a,report);write_coverage_csv(b,coverage(paths))
    assert a.read_bytes()==b.read_bytes() and b'official_day_session' in a.read_bytes()


def test_generation_can_defer_labels_but_independent_replay_still_recomputes_them(tmp_path):
    from gym_trading_env.research.fg_s3_e_quality_corpus import generate_behavior, behavior_specs, l1_labeler
    from gym_trading_env.research.fg_s3_quality_teacher_v2 import NativeQualityTeacher
    from gym_trading_env.research.fg_replay import reconcile_trace, E_QUALITY_SCHEMA
    factory, _, _ = native_factory(tmp_path/'market', minutes=20)
    env = factory(); env.reset(seed=0)
    source = dict(fold='TRAIN', state_cache_entries=0)
    root = Path(__file__).resolve().parents[3]/'dreamerv3'
    trace, financial = generate_behavior(factory, behavior_specs()[0], tmp_path/'trace', root,
        source=source, verify_labels=False)
    assert financial['status'] == 'PASS' and financial['teacher_labels_verified'] == 0
    assert financial['label_verification'] == 'DEFERRED_TO_INDEPENDENT_NATIVE_G3'
    teacher = NativeQualityTeacher(env, factory, fold='TRAIN', source=source)
    try:
        checked = reconcile_trace(tmp_path/'trace', root, env_factory=factory,
            schema=E_QUALITY_SCHEMA, label_verifier=l1_labeler(teacher))
        assert checked['teacher_labels_verified'] == checked['records'] == trace['records']
    finally:
        teacher.close(); env.close()
