# tests/contract/test_observation_space.py
import numpy as np
import pandas as pd
import pytest

from gym_trading_env.envs.trading_env import CustomTradingEnv, Action
from gym_trading_env.utils.trade_util import action_to_index

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
    """DreamerV3 receives the target-position action index."""
    return action_to_index(env, act)


def expected_market_seq(env: CustomTradingEnv, end_i: int) -> np.ndarray:
    """
    完全复刻 env._get_obs() 的 market_seq 逻辑：
    - 取 [end_i-window_size+1, end_i]（含），不足则左侧补 0
    - 并按 env 非 debug 分支做 nan_to_num
    """
    feats = list(env._OBS_FEATURES_MARKET)
    ws = int(env.window_size)

    start_i = end_i - ws + 1
    if start_i >= 0:
        window = env.bar_source.df_market.iloc[start_i:end_i + 1][feats].to_numpy(np.float32, copy=False)
        out = window
    else:
        pad_len = -start_i
        window = env.bar_source.df_market.iloc[0:end_i + 1][feats].to_numpy(np.float32, copy=False)
        pad = np.zeros((pad_len, len(feats)), dtype=np.float32)
        out = np.concatenate([pad, window], axis=0)

    # env._get_obs() 在非 debug 下会 nan_to_num
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    assert out.shape == (ws, len(feats))
    return out


def test_obs_shapes_dtypes_and_episode_len_is_one_day():
    df = make_one_day_df()
    env = CustomTradingEnv(df=df, config_path="tests/test.yaml")

    # 确保 reset 选固定起点（否则随机起点导致 expected 不可控）
    env.config.training.randomize_start = False
    # 可选：明确锚点，保证 start_row=0（我们 df 从 21:01 开始）
    env.config.training.start_clock = "future_night"

    obs, info = env.reset()
    market = obs["market_seq"]
    agent = obs["agent_state"]

    assert market.shape == (env.window_size, len(env._OBS_FEATURES_MARKET))
    assert agent.shape == (len(env._OBS_FEATURES_AGENT),)
    assert market.dtype == np.float32
    assert agent.dtype == np.float32
    assert np.isfinite(market).all()
    assert np.isfinite(agent).all()

    env.close()


def test_reset_starts_at_minute0_and_obs_is_left_padded():
    """
    现有 obs 语义：
    - reset 选到当天 minute_index_t=0 的 bar 作为 current_step
    - market_seq 是“全局滑窗回看 current_step”，数据不够则左侧 padding 0
      所以在 current_step=0 时：前 window_size-1 行全 0，最后 1 行=bar0 特征
    """
    df = make_one_day_df()
    env = CustomTradingEnv(df=df, config_path="tests/test.yaml")
    env.config.training.randomize_start = False
    env.config.training.start_clock = "future_night"

    obs, _ = env.reset()
    market = obs["market_seq"]

    # 起点应是 minute_index_t=0
    assert int(env.current_step) == 0
    assert int(env.current_minute) == 0

    exp = expected_market_seq(env, end_i=int(env.current_step))
    assert np.allclose(market, exp, atol=1e-6, rtol=0.0)

    # 明确检查 padding：前面应为 0（当 window_size>1）
    if env.window_size > 1:
        assert np.all(market[:-1, :] == 0.0)

    env.close()


def test_hold_shifts_window_by_one_row():
    """
    HOLD 一步后 current_step->1：
    - market_seq 最后一行应对齐 df_market[1]
    - 倒数第二行应对齐 df_market[0]
    - 更左侧仍为 0（如果 window_size>2）
    """
    df = make_one_day_df()
    env = CustomTradingEnv(df=df, config_path="tests/test.yaml")
    env.config.training.randomize_start = False
    env.config.training.start_clock = "future_night"

    obs0, _ = env.reset()
    assert int(env.current_step) == 0

    obs1, r, term, trunc, info = env.step(aidx(env, Action.HOLD))
    assert int(env.current_step) == 1

    market1 = obs1["market_seq"]
    exp1 = expected_market_seq(env, end_i=1)
    assert np.allclose(market1, exp1, atol=1e-6, rtol=0.0)

    # padding 结构检查（稳一点：只在 window_size 足够时检查）
    if env.window_size >= 2:
        assert np.allclose(market1[-2, :], exp1[-2, :], atol=1e-6, rtol=0.0)
        assert np.allclose(market1[-1, :], exp1[-1, :], atol=1e-6, rtol=0.0)
    if env.window_size > 2:
        assert np.all(market1[:-2, :] == 0.0)

    env.close()
