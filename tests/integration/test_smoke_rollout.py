# tests/integration/test_smoke_rollout.py

import numpy as np
import pandas as pd
import pytest

from gym_trading_env.envs.trading_env import CustomTradingEnv
from gym_trading_env.utils.market_features import FEATURES_MARKET, build_market_features
from gym_trading_env.utils.agent_features import FEATURES_AGENT

pytestmark = pytest.mark.integration


def make_one_day_df(start="2020-01-01 21:01:00", periods=1440):
    idx = pd.date_range(start=start, periods=periods, freq="min")
    base = 1.1000
    drift = np.linspace(0, 0.0020, periods)
    c = base + drift
    return pd.DataFrame(
        {"Date": idx, "Open": c, "High": c, "Low": c, "Close": c, "Volume": np.ones(periods, dtype=float)}
    )


def assert_obs_ok(obs):
    assert isinstance(obs, dict)
    assert "market_seq" in obs and "agent_state" in obs
    m = obs["market_seq"]
    a = obs["agent_state"]
    assert m.shape == (60, len(FEATURES_MARKET))
    assert a.shape == (len(FEATURES_AGENT),)
    assert m.dtype == np.float32
    assert a.dtype == np.float32
    assert np.isfinite(m).all()
    assert np.isfinite(a).all()


def assert_info_ok(info):
    # 只做“契约字段存在性 + 可数值化 + finite”，不绑定具体实现细节
    keys = ["balance", "equity", "used_margin", "free_margin", "fees_collected"]
    for k in keys:
        assert k in info, f"info missing key: {k}"
        v = float(info[k])  # Decimal / str / float 都接受
        assert np.isfinite(v), f"info[{k}] not finite: {info[k]}"


def test_rollout_smoke_until_episode_end():
    df = make_one_day_df()
    env = CustomTradingEnv(df=df, config_path="tests/test.yaml")
    env.config.training.randomize_start = False  # episode_length=1440 时必须从 0 起

    obs, info = env.reset(seed=0)
    assert_obs_ok(obs)
    assert_info_ok(info)

    # 跑到 episode 结束（最多 1500 步兜底）
    terminated = truncated = False
    for _ in range(1500):
        a = int(env.action_space.sample())  # 离散 index
        obs, reward, terminated, truncated, info = env.step(a)

        assert_obs_ok(obs)
        assert np.isfinite(float(reward))
        assert_info_ok(info)

        if terminated or truncated:
            break

    assert terminated or truncated, "episode did not finish within expected steps"

    # 能再 reset 一次
    obs2, info2 = env.reset(seed=1)
    assert_obs_ok(obs2)
    assert_info_ok(info2)

    env.close()
