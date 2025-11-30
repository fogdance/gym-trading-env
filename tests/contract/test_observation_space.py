# tests/contract/test_observation_space.py
import numpy as np
import pandas as pd
import pytest

from gym_trading_env.envs.trading_env import CustomTradingEnv, Action
from gym_trading_env.utils.market_features import FEATURES_MARKET, build_market_features
from gym_trading_env.utils.agent_features import FEATURES_AGENT

pytestmark = pytest.mark.unit

DAY_MINUTES = 345


def make_one_day_df(start="2020-01-01 21:01:00", periods=DAY_MINUTES):
    idx = pd.date_range(start=start, periods=periods, freq="min")
    base = 1.1000
    drift = np.linspace(0, 0.0020, periods)
    c = base + drift
    return pd.DataFrame(
        {"Date": idx, "Open": c, "High": c, "Low": c, "Close": c, "Volume": 1.0}
    )


def aidx(env: CustomTradingEnv, act: Action) -> int:
    """DreamerV3 离散动作：喂给 env.step 的是 valid_actions 的 index（0..N-1）。"""
    return env.valid_actions.index(act)


def nonzero_rows(mat: np.ndarray):
    return np.where(np.any(mat != 0.0, axis=1))[0]


def assert_future_zeroed(market_seq: np.ndarray, frontier: int):
    if frontier + 1 < market_seq.shape[0]:
        assert np.all(market_seq[frontier + 1 :, :] == 0.0), "future rows must be all zeros"


def test_obs_shapes_dtypes_and_episode_len_is_one_day():
    df = make_one_day_df()
    env = CustomTradingEnv(df=df, config_path="tests/test.yaml")
    env.config.training.randomize_start = False


    obs, info = env.reset()
    market = obs["market_seq"]
    agent = obs["agent_state"]

    assert market.shape == (env.window_size, len(FEATURES_MARKET))
    assert agent.shape == (len(FEATURES_AGENT),)
    assert market.dtype == np.float32
    assert agent.dtype == np.float32
    assert np.isfinite(market).all()
    assert np.isfinite(agent).all()

    env.close()


def test_reset_must_start_at_minute0_and_only_reveal_row0():
    """
    episode_length=1440 时不允许从非 0 minute 起步：
    reset 必须从 day 的 minute=0 开始，并且只 reveal 第 0 行。
    """
    df = make_one_day_df()
    env = CustomTradingEnv(df=df, config_path="tests/test.yaml")
    env.config.training.randomize_start = False

    obs, _ = env.reset()
    market = obs["market_seq"]

    nz = nonzero_rows(market)
    assert nz.tolist() == [0], f"reset must reveal only row0, got nonzero rows: {nz.tolist()}"
    assert_future_zeroed(market, frontier=0)

    # frontier 行的 mask_t 应该是 1（如果你们 mask_t 在 FEATURES_MARKET 中）
    if "mask_t" in FEATURES_MARKET:
        mask_col = FEATURES_MARKET.index("mask_t")
        assert float(market[0, mask_col]) == 1.0

    env.close()


def test_hold_reveals_exactly_one_more_minute():
    df = make_one_day_df()
    env = CustomTradingEnv(df=df, config_path="tests/test.yaml")
    env.config.training.randomize_start = False

    obs0, _ = env.reset()
    f0 = int(nonzero_rows(obs0["market_seq"])[-1])
    assert f0 == 0

    obs1, r, term, trunc, info = env.step(aidx(env, Action.HOLD))
    market1 = obs1["market_seq"]

    nz1 = nonzero_rows(market1).tolist()
    assert nz1 == [0, 1], f"after HOLD from reset, visible rows must be [0,1], got {nz1}"
    assert_future_zeroed(market1, frontier=1)

    if "mask_t" in FEATURES_MARKET:
        mask_col = FEATURES_MARKET.index("mask_t")
        assert float(market1[1, mask_col]) == 1.0

    env.close()
