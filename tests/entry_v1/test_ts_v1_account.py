"""Thinking Student V1 Native account snapshot replayed along a recorded trace."""
from pathlib import Path

import numpy as np
from wp02_fixture import native_factory

from gym_trading_env.research.fg_replay import (
    collect_trace, iter_replay_rows, trace_manifest, E_QUALITY_PROTOCOL, E_QUALITY_SCHEMA)
from gym_trading_env.research.fg_single3_replay import Single3BehaviorPolicy
from gym_trading_env.research.ts_v1_account import replay_account_native


def test_replayed_account_native_agrees_with_every_recorded_row(tmp_path):
    factory, path, _ = native_factory(tmp_path / 'market')
    root, dreamer = tmp_path / 'trace', Path(__file__).resolve().parents[3] / 'dreamerv3'
    spec = dict(id='P2_mid', kind='probabilistic', probability=.5, seed_components=[7, 0, 3])
    collect_trace(path, spec, root, dict(fold='CPU_SYNTHETIC'), dreamer, env_factory=factory,
                  protocol=E_QUALITY_PROTOCOL, policy_factory=Single3BehaviorPolicy)
    stepids, values = replay_account_native(root, factory)
    steps = [step for _, step in iter_replay_rows(root, trace_manifest(root, schema=E_QUALITY_SCHEMA))]
    assert values.shape == (len(steps), 10)
    np.testing.assert_array_equal(stepids, np.stack([step['stepid'] for step in steps]))
    ps = np.stack([step['position_state'] for step in steps])
    flat = ps[:, 1] == 0
    assert flat.any() and (~flat).any(), 'the trace must hold positions at some rows'
    # Facts that position_state also carries must agree row for row.
    np.testing.assert_array_equal(values[:, 0], flat)
    np.testing.assert_array_equal(values[:, 1], ps[:, 0])
    np.testing.assert_allclose(values[:, 2], ps[:, 1])
    np.testing.assert_allclose(values[~flat, 3], ps[~flat, 4], rtol=1e-6)  # TP distance / 5
    np.testing.assert_allclose(values[~flat, 6], ps[~flat, 5], rtol=1e-6)  # held market minutes
    assert not values[flat, 2:].any(), 'a flat account has no position facts'


def test_replay_rejects_a_recording_the_native_env_does_not_reproduce(tmp_path):
    import pytest
    factory, path, _ = native_factory(tmp_path / 'market')
    root, dreamer = tmp_path / 'trace', Path(__file__).resolve().parents[3] / 'dreamerv3'
    spec = dict(id='P1', kind='always_add', seed_components=[7, 0, 1])
    collect_trace(path, spec, root, dict(fold='CPU_SYNTHETIC'), dreamer, env_factory=factory,
                  protocol=E_QUALITY_PROTOCOL, policy_factory=Single3BehaviorPolicy)
    chunk = sorted((root / 'replay').glob('*.npz'))[0]
    with np.load(chunk) as data:
        arrays = {k: data[k].copy() for k in data.files}
    arrays['action'][arrays['action'] == 1] = 0  # pretend the recorded ADDs were WAITs
    np.savez(chunk, **arrays)
    with pytest.raises(ValueError, match='differs at row'):
        replay_account_native(root, factory)
