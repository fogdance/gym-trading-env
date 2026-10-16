"""Exact snapshot identities and mutation rejection across the faster path."""
import copy

import pytest

from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.utils.market_prefix import MarketPrefix
from wp02_fixture import native_factory


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
    factory, config, profile = native_factory(tmp_path/'market', minutes=20, chunk=7)
    env = factory();env.reset(seed=0)
    initial = env.export_state()
    source = dict(fold='TRAIN', state_cache_entries=0)
    dreamer = Path(__file__).resolve().parents[3]/'dreamerv3'
    labeler_factory = lambda identity:NativeLabelPool(dict(env_config=str(config)), profile,
        initial, source, None, identity, workers=2)
    serial, a = generate_behavior(factory, behavior_specs()[0], tmp_path/'serial', dreamer,
        source=source, initial_state=initial)
    parallel, b = generate_behavior(factory, behavior_specs()[0], tmp_path/'parallel', dreamer,
        source=source, initial_state=initial, labeler_factory=labeler_factory)
    assert a['status'] == b['status'] == 'PASS'
    assert a['teacher_labels_verified'] == b['teacher_labels_verified'] == parallel['records']
    assert serial['event_hash'] == parallel['event_hash']
    assert serial['array_hash'] == parallel['array_hash']
    assert [r['teacher'] for r in iter_sidecar(tmp_path/'serial')] == [r['teacher'] for r in iter_sidecar(tmp_path/'parallel')]
    env.close()


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
