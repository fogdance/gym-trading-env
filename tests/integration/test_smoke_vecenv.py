# tests/integration/test_smoke_vecenv.py

import numpy as np
import pandas as pd
import pytest

from gym_trading_env.envs.trading_env import CustomTradingEnv

pytestmark = pytest.mark.integration

sb3 = pytest.importorskip("stable_baselines3")
from stable_baselines3.common.vec_env import DummyVecEnv


def make_one_day_df(start="2020-01-01 21:01:00", periods=1440):
    idx = pd.date_range(start=start, periods=periods, freq="min")
    base = 1.1000
    drift = np.linspace(0, 0.0010, periods)
    c = base + drift
    return pd.DataFrame(
        {"Date": idx, "Open": c, "High": c, "Low": c, "Close": c, "Volume": np.ones(periods, dtype=float)}
    )


def make_env(seed):
    def _thunk():
        df = make_one_day_df()
        env = CustomTradingEnv(df=df, config_path="tests/test.yaml")
        env.config.training.randomize_start = False
        env.reset(seed=seed)
        return env
    return _thunk


def test_dummy_vecenv_smoke():
    n_envs = 2
    venv = DummyVecEnv([make_env(0), make_env(1)])

    obs = venv.reset()
    # VecEnv 下 obs 仍然是 dict of np arrays
    assert isinstance(obs, dict)
    for k, arr in obs.items():
        assert np.isfinite(arr).all()

    for _ in range(20):
        actions = np.array([venv.action_space.sample() for _ in range(n_envs)], dtype=np.int64)
        obs, rewards, dones, infos = venv.step(actions)
        for k, arr in obs.items():
            assert np.isfinite(arr).all()
        assert np.isfinite(rewards).all()

    venv.close()
