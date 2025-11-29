# tests/contract/test_agent_state_contract.py
import numpy as np
import pandas as pd
import pytest

from gym_trading_env.envs.trading_env import CustomTradingEnv, Action
from gym_trading_env.utils.build_xt import FEATURES_AGENT, FEATURES_MARKET
from gym_trading_env.utils.trade_util import step_wrapper

pytestmark = pytest.mark.unit


def make_one_session_minute_df(start="2020-01-01 21:01:00", periods=120):
    idx = pd.date_range(start=start, periods=periods, freq="min")
    base = 1.1000
    drift = np.linspace(0, 0.0010, periods)
    c = base + drift
    return pd.DataFrame(
        {
            "Date": idx,
            "Open": c,
            "High": c,
            "Low":  c,
            "Close": c,
            "Volume": np.ones(periods, dtype=float),
        }
    )


def _idx():
    return {k: FEATURES_AGENT.index(k) for k in FEATURES_AGENT}


def _assert_finite(x: np.ndarray):
    assert np.isfinite(x).all(), "agent_state contains NaN/Inf"


def _assert_close(a, b, tol=1e-5):
    assert abs(float(a) - float(b)) <= tol, f"{a} != {b}"



def test_agent_state_flat_defaults_and_shapes():
    df = make_one_session_minute_df()
    env = CustomTradingEnv(df=df, config_path="tests/test.yaml")
    env.config.training.randomize_start = False

    obs, info = env.reset()
    agent = obs["agent_state"]
    market = obs["market_seq"]

    assert market.shape[1] == len(FEATURES_MARKET)
    assert agent.shape == (len(FEATURES_AGENT),)
    assert market.dtype == np.float32
    assert agent.dtype == np.float32

    _assert_finite(agent)

    idx = _idx()
    # flat 状态下的强约束（不依赖你内部实现的具体细节）
    _assert_close(agent[idx["pos_t"]], 0.0)
    _assert_close(agent[idx["have_long_t"]], 0.0)
    _assert_close(agent[idx["have_short_t"]], 0.0)
    _assert_close(agent[idx["entry_price_t"]], 0.0)
    _assert_close(agent[idx["holding_minutes_t"]], 0.0)
    _assert_close(agent[idx["upnl_t"]], 0.0)
    _assert_close(agent[idx["realized_pnl_step_t"]], 0.0)
    _assert_close(agent[idx["realized_pnl_cum_t"]], 0.0)
    _assert_close(agent[idx["fee_step_t"]], 0.0)
    _assert_close(agent[idx["fee_cum_t"]], 0.0)

    # equity / peak / drawdown 的一致性
    _assert_close(agent[idx["equity_t"]], float(info["equity"]), tol=1e-2)
    _assert_close(agent[idx["max_equity_t"]], float(info["equity"]), tol=1e-2)
    _assert_close(agent[idx["drawdown_t"]], 0.0)

    # 风控相关字段：flat 下应为 0 或非负（避免训练污染）
    for k in ["sigma_entry_t", "sl_ticks_t", "tp_ticks_t", "sl_price_t", "tp_price_t"]:
        _assert_close(agent[idx[k]], 0.0)

    # minutes_to_timeout：不同实现可能是 0 或剩余分钟；但必须 finite 且 >=0
    assert agent[idx["minutes_to_timeout_t"]] >= 0.0

    env.close()


def test_agent_state_equity_and_fee_consistency_over_steps():
    df = make_one_session_minute_df()
    env = CustomTradingEnv(df=df, config_path="tests/test.yaml")
    env.config.training.randomize_start = False

    obs, info = env.reset()
    idx = _idx()

    def assert_consistency(obs, info):
        agent = obs["agent_state"].astype(float)
        _assert_finite(agent)
        _assert_close(agent[idx["equity_t"]], float(info["equity"]), tol=1e-2)
        _assert_close(agent[idx["fee_cum_t"]], float(info["fees_collected"]), tol=1e-8)

    # 1) HOLD
    obs, r, term, trunc, info = step_wrapper(env, Action.HOLD)
    assert_consistency(obs, info)
    agent = obs["agent_state"].astype(float)
    _assert_close(agent[idx["pos_t"]], 0.0)
    _assert_close(agent[idx["fee_step_t"]], 0.0)
    _assert_close(agent[idx["realized_pnl_step_t"]], 0.0)

    # 2) LONG_OPEN0（如果你项目里没有 LONG_OPEN0，就把 0 去掉）
    open_action = Action.LONG_OPEN0 if hasattr(Action, "LONG_OPEN0") else Action.LONG_OPEN
    obs, r, term, trunc, info = step_wrapper(env, open_action)
    assert_consistency(obs, info)
    agent = obs["agent_state"].astype(float)

    assert agent[idx["have_long_t"]] in (0.0, 1.0)
    assert agent[idx["have_short_t"]] == 0.0
    assert agent[idx["pos_t"]] >= 0.0
    assert agent[idx["entry_price_t"]] >= 0.0
    assert agent[idx["holding_minutes_t"]] >= 0.0
    assert agent[idx["fee_cum_t"]] >= 0.0

    # risk_reward 关闭时，sl/tp 应保持为 0（按你测试配置默认）
    rr_enabled = bool(getattr(env.config.risk, "risk_reward_ratio_enable", False))
    if not rr_enabled:
        for k in ["sl_ticks_t", "tp_ticks_t", "sl_price_t", "tp_price_t", "sigma_entry_t"]:
            _assert_close(agent[idx[k]], 0.0)

    # 3) LONG_CLOSE0
    close_action = Action.LONG_CLOSE0 if hasattr(Action, "LONG_CLOSE0") else Action.LONG_CLOSE
    obs, r, term, trunc, info = step_wrapper(env, close_action)
    assert_consistency(obs, info)
    agent = obs["agent_state"].astype(float)

    _assert_close(agent[idx["pos_t"]], 0.0)
    _assert_close(agent[idx["have_long_t"]], 0.0)
    _assert_close(agent[idx["have_short_t"]], 0.0)
    _assert_close(agent[idx["entry_price_t"]], 0.0)
    _assert_close(agent[idx["holding_minutes_t"]], 0.0)
    _assert_close(agent[idx["upnl_t"]], 0.0)
    # realized_pnl_step_t 在 close 这一步应该是 finite（可正可负）
    assert np.isfinite(agent[idx["realized_pnl_step_t"]])

    env.close()
