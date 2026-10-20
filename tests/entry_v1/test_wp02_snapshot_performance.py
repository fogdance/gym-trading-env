"""Exact snapshot identities and mutation rejection across the faster path."""
import copy

import pytest

from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.utils.market_prefix import MarketPrefix
from wp02_fixture import native_factory


def test_account_snapshot_copy_retains_json_normalization_and_full_independent_history():
    import json
    from decimal import Decimal
    from gym_trading_env.utils.snapshot_json import copy_account_snapshot
    row=dict(time='2021-12-09 09:01:00+08:00',equity=300000.,
             unrealized_pnl=-0.,net_lots=0.,intrabar=False)
    state=dict(equity_history=[row,dict(extra=[Decimal('1.25'),('a','b')])],
               amount=Decimal('2.5'),rng={'state':2**127})
    expected=json.loads(json.dumps(state,default=str))
    actual=copy_account_snapshot(state)
    assert actual==expected
    actual['equity_history'][0]['equity']=0
    actual['equity_history'][1]['extra'].clear()
    assert state['equity_history'][0]['equity']==300000.
    assert len(state['equity_history'][1]['extra'])==2


def test_cached_history_fingerprint_is_original_sha_for_append_rewind_and_mutation():
    from gym_trading_env.utils.account_fingerprint import AccountFingerprint
    from decimal import Decimal
    history=[dict(time='中文',equity=300000.,unrealized_pnl=-0.,net_lots=0.,intrabar=False)]
    cache=AccountFingerprint(history)
    state=dict(a=Decimal('1.25'),equity_history=copy.deepcopy(history),z={'nested':['x',1]})
    assert cache.digest(state)==fingerprint(state)
    state['equity_history'][0]['unrealized_pnl']=0.
    assert cache.digest(state)==fingerprint(state)
    state['equity_history'][0]['intrabar']=0
    assert cache.digest(state)==fingerprint(state)
    state['equity_history']=copy.deepcopy(history)
    state['equity_history'].append({**history[0],'equity':299999.})
    assert cache.digest(state)==fingerprint(state)
    state['equity_history'][0]['equity']=123.
    assert cache.digest(state)==fingerprint(state)
    assert cache.history[0]['equity']==300000.
    state['equity_history'].clear()
    assert cache.digest(state)==fingerprint(state)
    assert AccountFingerprint([]).digest(state)==fingerprint(state)


def test_worker_budget_uses_real_memory_cpu_and_independent_job_count():
    from gym_trading_env.research.fg_s3_e_quality_corpus_cli import _worker_limits
    gib = 1024**3
    limits = _worker_limits(6, available=10*gib, peak_rss=800*1024**2, process_memory=256*1024**2, cpus=24)
    assert limits['workers'] == 6 and limits['native_threads_per_worker'] == 1
    assert limits['total_worker_processes'] == 24 and limits['label_workers'] == [3]*6
    assert _worker_limits(6, available=4*gib, peak_rss=800*1024**2, process_memory=gib, cpus=24)['workers'] == 1
    assert _worker_limits(6, available=10*gib, peak_rss=800*1024**2, process_memory=256*1024**2, cpus=2)['total_worker_processes'] == 2
    with pytest.raises(ValueError, match='Insufficient'):
        _worker_limits(6, available=gib, peak_rss=800*1024**2, process_memory=gib, cpus=24)


def test_parallel_teacher_transport_matches_serial_native_accounts_and_labels(tmp_path):
    from pathlib import Path
    from gym_trading_env.research.fg_s3_e_quality_corpus import generate_behavior, behavior_specs
    from gym_trading_env.research.fg_s3_quality_label_pool import NativeLabelPool
    from gym_trading_env.research.fg_replay import iter_sidecar
    from gym_trading_env.research.fg_s3_e_quality_strict import QualityStrictTeacher
    factory, config, profile = native_factory(tmp_path/'market', minutes=20, chunk=7)
    env = factory();env.reset(seed=0)
    initial = env.export_state()
    source = dict(fold='TRAIN', state_cache_entries=0)
    strict = QualityStrictTeacher(env,source=source)
    strict.compile_value_store(tmp_path/'shared_exact')
    dreamer = Path(__file__).resolve().parents[3]/'dreamerv3'
    labeler_factory = lambda identity:NativeLabelPool(dict(env_config=str(config)), profile,
        initial, source, strict, identity, workers=2)
    serial, a = generate_behavior(factory, behavior_specs()[0], tmp_path/'serial', dreamer,
        source=source, initial_state=initial,strict=strict)
    parallel, b = generate_behavior(factory, behavior_specs()[0], tmp_path/'parallel', dreamer,
        source=source, initial_state=initial, strict=strict,labeler_factory=labeler_factory)
    assert a['status'] == b['status'] == 'PASS'
    assert a['teacher_labels_verified'] == b['teacher_labels_verified'] == parallel['records']
    assert serial['event_hash'] == parallel['event_hash']
    assert serial['array_hash'] == parallel['array_hash']
    assert [r['teacher'] for r in iter_sidecar(tmp_path/'serial')] == [r['teacher'] for r in iter_sidecar(tmp_path/'parallel')]
    workers=b['label_transport']['native_workers']
    assert sum(w['queries'] for w in workers)==b['label_transport']['queries']
    assert sum(w['e_observation_cache']['hits'] for w in workers)>0
    assert sum(w['strict_pair_cache']['hits']+w['strict_pair_cache']['misses'] for w in workers)>0
    strict._value_store.close()
    env.close()


def test_ordered_history_transport_rejects_changed_old_account_samples(tmp_path):
    from gym_trading_env.research.fg_s3_quality_label_pool import NativeLabelPool
    from gym_trading_env.research.fg_s3_quality_teacher_v2 import NativeQualityTeacher
    factory,config,profile=native_factory(tmp_path/'market',minutes=20)
    env=factory();env.reset(seed=0)
    source=dict(fold='TRAIN',state_cache_entries=0)
    teacher=NativeQualityTeacher(env,factory,fold='TRAIN',source=source)
    pool=NativeLabelPool(dict(env_config=str(config)),profile,env.export_state(),source,
                         None,teacher.identity_hash,workers=1)
    try:
        first=pool(env,None,None).result()
        env.step(0)
        second=pool(env,None,None).result()
        assert second['account_snapshot_hash']!=first['account_snapshot_hash']
        assert pool.statistics['delta_history_items']<pool.statistics['full_history_items']
        env.equity_history[0]['equity']+=1
        with pytest.raises(ValueError,match='complete native account snapshot'):
            pool(env,None,None).result()
    finally:pool.close();teacher.close();env.close()


def test_block_source_guard_rechecks_changes_before_teacher_closes(tmp_path):
    from gym_trading_env.research.fg_s3_quality_teacher_v2 import NativeQualityTeacher
    from gym_trading_env.research.fg_replay import digest
    factory,_,_=native_factory(tmp_path/'market',minutes=20)
    marker=tmp_path/'source.txt';marker.write_text('frozen')
    env=factory();env.reset(seed=0)
    teacher=NativeQualityTeacher(env,factory,fold='TRAIN',source=dict(
        fold='TRAIN',files={str(marker):digest(marker)},source_guard_interval=1024))
    teacher.query_state(env,env._published_mask)
    marker.write_text('changed')
    try:
        with pytest.raises(ValueError,match='Frozen quality source changed'):
            teacher.close()
    finally:env.close()


def test_prefix_json_matches_original_hash_and_does_not_include_suffix(tmp_path):
    factory, _, _ = native_factory(tmp_path/'market', minutes=20)
    env = factory()
    try:
        for i in (0, 1, 7, 19, 3):
            assert env._prefix_hash(i) == fingerprint(env._raw.iloc[:i+1].to_dict(orient='split'))
        before = env._prefix_hash(3)
        env._raw.loc[env._raw.index[-1], 'Volume'] += 1
        assert env._prefix_hash(3) == before
        assert env._prefix_hash(19) == fingerprint(env._raw.to_dict(orient='split'))
        frame = env._raw.copy()
        frame['audit_unicode'] = '中文 "\\\n'
        frame['audit_float'] = [-0.0, 1e-12] * 10
        cache = MarketPrefix(frame)
        for count in (0, 1, 20):
            assert cache.digest(count) == fingerprint(frame.iloc[:count].to_dict(orient='split'))
    finally:
        env.close()


def test_warm_prefix_cache_still_rejects_processed_history_tampering(tmp_path):
    factory, _, _ = native_factory(tmp_path/'market', minutes=20)
    env = factory()
    try:
        for _ in range(4):
            env.step(0)
        state = env.export_state()
        env._raw.loc[env._raw.index[0], 'Volume'] += 1
        with pytest.raises(ValueError, match='Processed market history changed'):
            env.import_state(state)
    finally:
        env.close()


def test_json_restore_keeps_full_history_and_independent_snapshot_ownership(tmp_path):
    factory, _, _ = native_factory(tmp_path/'market', minutes=20)
    env, restored = factory(), factory()
    try:
        for _ in range(5):
            env.step(int(bool(env._published_mask[1])))
        state = env.export_state()
        saved = copy.deepcopy(state)
        restored.import_state(state)
        assert state == saved == restored.export_state()
        state['equity_history'].clear()
        state['fills'].clear()
        assert restored.export_state() == saved
        for _ in range(3):
            a, b = env.step(0), restored.step(0)
            assert a[1:4] == b[1:4]
            assert env.export_state() == restored.export_state()
    finally:
        env.close()
        restored.close()


def test_private_e_cache_matches_uncached_native_steps_warmup_restore_and_ownership(tmp_path):
    import numpy as np
    from gym_trading_env.research.fg_s3_quality_teacher_v2 import NativeQualityTeacher
    factory,_,_=native_factory(tmp_path/'market',minutes=190,chunk=64)
    env=factory()
    teacher=NativeQualityTeacher(env,factory,fold='TRAIN',source=dict(fold='TRAIN'))
    clone=teacher.clone
    cache=clone._e_observation_cache
    cache.capacity=4
    checkpoint=None
    try:
        for cursor in range(181):
            if cursor in (0,90,174,175,180):
                expected=env._get_obs();actual=clone._get_obs()
                for key in expected:np.testing.assert_array_equal(actual[key],expected[key])
                assert clone.export_state()==env.export_state()
                if cursor==90:checkpoint=env.export_state()
            if cursor<180:
                if env.truncated:
                    env.reset();clone.reset()
                action=int(cursor%7==0 and bool(env._published_mask[1]))
                a,b=env.step(action),clone.step(action)
                assert a[1:4]==b[1:4]
        assert env._get_obs()['market_macd_5m'][3]==1
        actual=clone._get_obs()
        actual['market_context_15m'][:]=99
        clone.e_snapshot['source_contract']='mutated output'
        for key,value in env._get_obs().items():
            np.testing.assert_array_equal(clone._get_obs()[key],value)
        assert clone.e_snapshot==env.e_snapshot
        assert cache.statistics['hits']>0 and cache.statistics['evictions']>0
        assert len(cache.entries)<=4
        clone.import_state(checkpoint);env.import_state(checkpoint)
        assert clone.export_state()==env.export_state()==checkpoint
        clone._features[clone.current_step,0]+=1
        with pytest.raises(ValueError,match='original 18 channels'):
            clone._get_obs()
    finally:
        teacher.close();env.close()


def test_native_source_guard_rejects_in_memory_market_before_next_label(tmp_path):
    from gym_trading_env.research.fg_s3_quality_teacher_v2 import NativeQualityTeacher
    factory,_,_=native_factory(tmp_path/'market',minutes=20)
    env=factory()
    teacher=NativeQualityTeacher(env,factory,fold='TRAIN',source=dict(
        fold='TRAIN',source_guard_interval=1024,state_cache_entries=0))
    teacher.query_state(env,env._published_mask)
    env._raw=env._raw.copy()
    env._raw.iloc[0,env._raw.columns.get_loc('Volume')]+=1
    try:
        with pytest.raises(ValueError,match='Quality source/state identity mismatch'):
            teacher.query_state(env,env._published_mask)
    finally:teacher.close();env.close()


def test_l1_market_reuse_keeps_funded_and_losing_native_accounts_exact(tmp_path):
    from gym_trading_env.research.fg_s3_quality_teacher_v2 import NativeQualityTeacher
    factory,_,_=native_factory(tmp_path/'market',minutes=20,chunk=7)
    funded,losing=factory(),factory()
    losing.step(1)
    while losing.entries:losing.step(0)
    while funded.current_step<losing.current_step:funded.step(0)
    assert funded._calculate_equity()>losing._calculate_equity()
    source=dict(fold='TRAIN',state_cache_entries=0)
    cached=NativeQualityTeacher(funded,factory,fold='TRAIN',source=source)
    plain=NativeQualityTeacher(funded,factory,fold='TRAIN',source=source)
    cache=cached.clone._native_execution_cache
    cache.row_capacity=3
    del plain.clone._native_execution_cache
    before_funded,before_losing=funded.export_state(),losing.export_state()
    try:
        answers=[]
        for env in (funded,losing,funded):
            actual=cached.query_state(env,env._published_mask)
            assert actual==plain.query_state(env,env._published_mask)
            answers.append(actual)
        assert all(x['immediate_outcome_verified'] for x in answers)
        assert answers[0]==answers[2]
        assert answers[0]['account_snapshot_hash']!=answers[1]['account_snapshot_hash']
        assert answers[0]['funds_after']!=answers[1]['funds_after']
        assert cache.statistics['row_hits']>0 and cache.statistics['quote_hits']>0
        assert cache.statistics['row_evictions']>0
        assert len(cache.rows)<=3 and len(cache.quotes)<=cache.quote_capacity
        assert funded.export_state()==before_funded and losing.export_state()==before_losing
        assert not hasattr(funded,'_native_execution_cache')
    finally:cached.close();plain.close();funded.close();losing.close()


def test_native_quote_reuse_keeps_effective_terms_side_tick_and_decimal_context(tmp_path):
    from decimal import Decimal, localcontext
    from gym_trading_env.envs.entry_execution import NativeMarketExecutionCache
    factory,_,_=native_factory(tmp_path/'market',minutes=20)
    cached,plain=factory(),factory()
    cache=cached._native_execution_cache=NativeMarketExecutionCache(row_capacity=2,quote_capacity=2,term_capacity=2)
    try:
        for spread,slippage,tick in [('0','0','1'),('0.25','0.5','0.5'),('1','0','1.0')]:
            for env in (cached,plain):
                env.terms.tick=Decimal(tick)
                for _,_,terms in env.terms.periods:
                    terms.update(spread=Decimal(spread),slippage=Decimal(slippage))
            for side,opening in [('LONG',False),('LONG',True),('SHORT',False),('SHORT',True)]:
                for precision in (10,6):
                    with localcontext() as context:
                        context.prec=precision
                        expected=plain._price(1000.125,side,plain._raw.index[1],opening=opening)
                        for _ in range(2):
                            actual=cached._price(1000.125,side,cached._raw.index[1],opening=opening)
                            assert actual.as_tuple()==expected.as_tuple()
        assert cache.statistics['quote_hits']>0 and cache.statistics['quote_evictions']>0
        assert len(cache.quotes)<=2
        assert cache.statistics['term_hits']>0
        start,end,terms=cached.terms.periods[0]
        split=cached._raw.index[10]
        first=dict(terms,spread=Decimal('0'));second=dict(terms,spread=Decimal('2'))
        cached.terms.periods=[(start,split,first),(split,end,second)]
        for i in (1,10,19,1):
            time=cached._raw.index[i]
            assert cached._terms_at(time) is cached.terms.at(time)
        assert cache.statistics['term_evictions']>0 and len(cache.effective_terms)<=2
        first['spread']=Decimal('3')
        assert cached._terms_at(cached._raw.index[1])['spread']==Decimal('3')
        # A previously cached timestamp now belongs to the replaced period.
        cached.terms.periods=[(start,end,second)]
        assert cached._terms_at(cached._raw.index[1]) is second
        assert cache.row(cached._raw,1).equals(cached._raw.iloc[1])
        frame=cached._raw.copy();frame.iloc[1,frame.columns.get_loc('Close')]+=1
        assert cache.row(frame,1).equals(frame.iloc[1]) and len(cache.rows)==1
    finally:cached.close();plain.close()
