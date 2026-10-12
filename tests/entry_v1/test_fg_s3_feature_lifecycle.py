"""Launcher resource ownership using real NumPy mappings, without model training."""
from types import SimpleNamespace

import numpy as np
import pytest

from gym_trading_env.research import fg_s3_experiment as manager


def test_actor_launcher_closes_cache_before_worker_and_after_audit(tmp_path, monkeypatch):
    features = tmp_path / 'features'
    features.mkdir()
    np.save(features / 'features.npy', np.ones((2, 3), np.float32))
    np.save(features / 'index.npy', np.arange(2, dtype=np.int32))
    prepared = tmp_path / 'prepared'
    logdir = tmp_path / 'run'
    cfg = dict(experiment='cache_lifecycle', trainer='s3a', diagnostic=True,
               paths=dict(features=str(features)),
               train=dict(mode='S3A_FULL', logdir=str(logdir)))
    receipt = dict(status='READY_FOR_BC')
    mappings = []

    def admission(*args, **kwargs):
        assert all(mapping.closed for group in mappings for mapping in group)
        x = np.load(features / 'features.npy', mmap_mode='r')
        index = np.load(features / 'index.npy', mmap_mode='r')
        mappings.append((x._mmap, index._mmap))
        return None, {}, None, SimpleNamespace(features=[x], index=[index])

    def identity(*args):
        cache = args[-1]
        np.testing.assert_array_equal(cache.features[0], np.ones((2, 3), np.float32))
        assert not cache.index[0]._mmap.closed
        return 'cache_lifecycle', {}

    def worker(*args, **kwargs):
        assert all(mapping.closed for group in mappings for mapping in group)
        logdir.mkdir()

    monkeypatch.setattr(manager, 'require_prepared', lambda *a, **k: (receipt, cfg))
    monkeypatch.setattr(manager, 'admission', admission)
    monkeypatch.setattr(manager, '_bind_features', lambda *a, **k: None)
    monkeypatch.setattr(manager, '_training_identity', identity)
    monkeypatch.setattr(manager, '_run_phase', worker)
    monkeypatch.setattr(manager, '_save_run_snapshot', lambda *a, **k: None)
    monkeypatch.setattr(manager, '_audit_checkpoint', lambda *a, **k: dict(
        bc_updates=1000, freeze_audit=dict(frozen_keys=281)))

    result = manager.start(prepared)
    assert result['status'] == 'TRAINING_CHECKPOINT_VERIFIED'
    assert len(mappings) == 2
    assert all(mapping.closed for group in mappings for mapping in group)

    def bad_checkpoint(*args):
        raise ValueError('bad checkpoint')

    monkeypatch.setattr(manager, '_audit_checkpoint', bad_checkpoint)
    with pytest.raises(ValueError, match='bad checkpoint'):
        manager.audit(prepared)
    assert len(mappings) == 3
    assert all(mapping.closed for group in mappings for mapping in group)
