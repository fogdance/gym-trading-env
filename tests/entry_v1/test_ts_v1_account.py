"""Thinking Student V1 Native account snapshot along a real Native account."""
import numpy as np
import pytest
from wp02_fixture import native_factory

from gym_trading_env.research import ts_v1_corpus as C
from gym_trading_env.research.ts_v1_account import ACCOUNT_NATIVE_FIELDS, StudentEnv

F = {name: i for i, name in enumerate(ACCOUNT_NATIVE_FIELDS)}


@pytest.fixture(scope='module')
def account(tmp_path_factory):
    root = tmp_path_factory.mktemp('ts_account')
    _, config, _ = native_factory(root / 'market')
    row = dict(contract='CZCE.FG2701', path=str(config.parent / 'market.csv'),
               env_config=str(config))
    config = C.write_inputs(row, root / 'inputs')
    env = C.native_env(config, market=False)
    arrays, _, _ = C.run_behavior(env, C.Behavior(C.BEHAVIORS['P2_mid']))
    return config, arrays


def test_account_native_agrees_with_the_position_facts(account):
    _, arrays = account
    values, ps = arrays['account_native'], arrays['position_state']
    assert values.shape == (len(ps), len(ACCOUNT_NATIVE_FIELDS)) == (len(ps), 14)
    flat = ps[:, 1] == 0
    assert flat.any() and (~flat).any(), 'the account must hold positions at some rows'
    np.testing.assert_array_equal(values[:, F['flat']], flat)
    np.testing.assert_array_equal(values[:, F['direction']], ps[:, 0])
    np.testing.assert_allclose(values[:, F['held_ratio']], ps[:, 1])
    np.testing.assert_allclose(values[~flat, F['tp_distance']], ps[~flat, 4], rtol=1e-6)
    np.testing.assert_allclose(values[~flat, F['held_minutes']], ps[~flat, 5], rtol=1e-6)
    position = [F[k] for k in ACCOUNT_NATIVE_FIELDS[2:10]]
    assert not values[flat][:, position].any(), 'a flat account has no position facts'


def test_deadlines_and_funds_are_causal_account_facts(account):
    _, arrays = account
    values = arrays['account_native']
    for name in ('to_add_deadline', 'to_forced_exit'):
        assert (np.diff(values[:, F[name]]) <= 0).all(), f'{name} only counts down'
    assert values[-1, F['to_forced_exit']] == 0, 'the final minute is the forced exit'
    # A well-funded account saturates both cover ratios; flat margin cover is 1.
    np.testing.assert_array_equal(values[:, F['funds_cover']], 1.)
    np.testing.assert_array_equal(values[:, F['margin_cover']], 1.)
    # The deadline field is zero exactly when the deadline has closed ADD.
    closed = values[:, F['to_add_deadline']] == 0
    assert not (arrays['action_mask'][closed, 1] > 0.5).any()


def test_student_env_adds_the_account_to_every_observation(account):
    config, _ = account
    env = StudentEnv(C.native_env(config, market=False))
    obs, _ = env.reset(seed=0)
    flats = []
    for _ in range(20):
        assert obs['account_native'].shape == (14,)
        flats.append(obs['account_native'][F['flat']])
        assert flats[-1] == (obs['position_state'][1] == 0)
        obs, *_ = env.step(int(obs['action_mask'][1] > 0.5))
    assert 0 in flats and 1 in flats, 'the account must open a position'
