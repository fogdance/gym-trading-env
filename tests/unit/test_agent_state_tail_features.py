# tests/unit/test_agent_state_tail_features.py
import pytest
import numpy as np
import pandas as pd

from gym_trading_env.envs.trading_env import CustomTradingEnv, Action
from gym_trading_env.utils.market_features import FEATURES_MARKET, build_market_features
from gym_trading_env.utils.agent_features import FEATURES_AGENT
from gym_trading_env.utils.trade_util import step_wrapper

pytestmark = pytest.mark.unit

def make_df_linear(start="2020-01-01 21:01:00", minutes=150, p0=1.1000, dp=0.0001):
    idx = pd.date_range(start=start, periods=minutes, freq="min")
    close = p0 + dp * np.arange(minutes, dtype=float)
    df = pd.DataFrame(
        {"Date": idx, "Open": close, "High": close, "Low": close, "Close": close, "Volume": 1.0}
    )
    return df

def _idx():
    return {k: FEATURES_AGENT.index(k) for k in FEATURES_AGENT}

def _assert_finite(vec):
    assert np.isfinite(vec).all(), f"agent_state has NaN/Inf: {vec}"

def test_agent_state_has_all_fields_and_tail_fields_reasonable():
    env = CustomTradingEnv(df=make_df_linear(), config_path="tests/test.yaml")
    env.config.training.randomize_start = False
    obs, info = env.reset()

    vec0 = obs["agent_state"].astype(float)
    assert vec0.shape[0] == len(FEATURES_AGENT)
    _assert_finite(vec0)

    ix = _idx()
    # reset: 无持仓时，风控类字段通常为 0（或至少非负、有限）
    for k in ["sigma_entry_t", "sl_ticks_t", "tp_ticks_t", "sl_price_t", "tp_price_t", "minutes_to_timeout_t"]:
        v = float(vec0[ix[k]])
        assert v >= 0.0

    # 开仓后：仍需满足“合理范围/关系”
    obs1, r1, term1, trunc1, info1 = step_wrapper(env, Action.LONG_OPEN0)
    vec1 = obs1["agent_state"].astype(float)
    _assert_finite(vec1)

    sigma = float(vec1[ix["sigma_entry_t"]])
    sl_ticks = float(vec1[ix["sl_ticks_t"]])
    tp_ticks = float(vec1[ix["tp_ticks_t"]])
    sl_price = float(vec1[ix["sl_price_t"]])
    tp_price = float(vec1[ix["tp_price_t"]])
    tmo = float(vec1[ix["minutes_to_timeout_t"]])
    entry = float(vec1[ix["entry_price_t"]])

    assert sigma >= 0.0
    assert sl_ticks >= 0.0 and tp_ticks >= 0.0
    assert sl_price >= 0.0 and tp_price >= 0.0
    assert tmo >= 0.0

    rr_enabled = bool(getattr(env.config.risk, "risk_reward_ratio_enable", False))
    if rr_enabled:
        # long：止损 < entry < 止盈（若策略给出）
        if sl_ticks > 0 or tp_ticks > 0:
            assert sl_price < entry < tp_price
    else:
        # 关闭 RR 时，推荐契约：这些字段为 0（更可控）
        assert sl_ticks == 0.0
        assert tp_ticks == 0.0
        assert sl_price == 0.0
        assert tp_price == 0.0

    # 若 timeout 启用（tmo > 0），再走一步 HOLD 应不增加（最好递减）
    obs2, r2, term2, trunc2, info2 = step_wrapper(env, Action.HOLD)
    vec2 = obs2["agent_state"].astype(float)
    _assert_finite(vec2)

    tmo2 = float(vec2[ix["minutes_to_timeout_t"]])
    if tmo > 0:
        assert tmo2 <= tmo

    env.close()
