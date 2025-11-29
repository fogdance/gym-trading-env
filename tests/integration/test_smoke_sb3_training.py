# tests/integration/test_smoke_sb3_training.py

import numpy as np
import pandas as pd
import pytest

from gym_trading_env.envs.trading_env import CustomTradingEnv

pytestmark = pytest.mark.integration

sb3 = pytest.importorskip("stable_baselines3")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def make_one_day_df(start="2020-01-01 21:01:00", periods=1440):
    idx = pd.date_range(start=start, periods=periods, freq="min")
    base = 1.1000
    drift = np.linspace(0, 0.0015, periods)
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


def test_sb3_ppo_short_training_smoke():
    venv = DummyVecEnv([make_env(0)])  # 先从 1 个 env 开始，最稳

    model = PPO(
        policy="MultiInputPolicy",   # Dict obs 必须用它
        env=venv,
        n_steps=64,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=0,
        seed=0,
    )

    # 极短训练：目标是“不崩溃 + 不 NaN”
    model.learn(total_timesteps=256)

    obs = venv.reset()
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv.step(action)
        assert np.isfinite(rewards).all()
        # obs 是 dict of arrays
        for _, arr in obs.items():
            assert np.isfinite(arr).all()

    venv.close()
