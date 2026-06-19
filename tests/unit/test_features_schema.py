# tests/unit/test_features_schema.py
import pytest
from gym_trading_env.utils.market_features import (
    FEATURES_MARKET,
    FEATURES_MARKET_OBS,
    FEATURES_RISK_CONTEXT,
    FEATURES_HTF_CONTEXT,
    build_market_features,
)
from gym_trading_env.utils.agent_features import FEATURES_AGENT

pytestmark = pytest.mark.unit

def _assert_unique(seq, name):
    assert len(seq) == len(set(seq)), f"{name} contains duplicates: {[x for x in seq if seq.count(x) > 1]}"

def test_features_market_schema():
    assert isinstance(FEATURES_MARKET, list)
    assert all(isinstance(x, str) for x in FEATURES_MARKET)
    _assert_unique(FEATURES_MARKET, "FEATURES_MARKET")
    # 强约束：顺序即契约（改动需要显式更新测试）
    assert FEATURES_MARKET[0] == "C_t"
    assert FEATURES_MARKET[-1] == "weekday_cos_t"
    assert "mask_t" in FEATURES_MARKET


def test_features_market_obs_schema():
    expected = [
        "volume_surprise_logratio_floor",
        "volume_surprise_rolling_percentile",
        "volume_impulse_recent",
        "volume_impulse_slope_3",
        "volume_impulse_slope_5",
        "volume_impulse_ready_flag",
        "oi_rel_yclose_log",
        "oi_rel_session_open_log",
        "oi_delta_rolling_z",
        "oi_rolling_percentile",
        "oi_delta_slope_3",
        "oi_delta_slope_5",
        "oi_price_confirm_short_continuous",
        "oi_price_confirm_long_continuous",
        "oi_impulse_ready_flag",
        "obs_cumVWAP_t",
        "obs_cumVWAP_t_rolling_percentile",
        "obs_dC_minus_cumVWAP_t",
        "obs_cmp_C_vs_cumVWAP_t",
        "obs_session_high_t",
        "obs_session_high_t_rolling_percentile",
        "obs_session_low_t",
        "obs_range_t",
        "obs_range_t_rolling_percentile",
        "obs_open_drift_t",
        "obs_bar_dir_t",
        "obs_minute_index_t",
        "obs_session_phase_t",
        "vol_rolling_percentile",
        "obs_dI_from_yclose_t",
        "obs_pct_chg_from_ref_t",
        "obs_pct_chg_from_ref_t_rolling_percentile",
        "obs_mask_t",
        "dyn5m_macd_line_norm",
        "dyn5m_macd_signal_norm",
        "dyn5m_macd_hist_norm",
        "dyn5m_macd_hist_delta",
        "dyn5m_macd_distance_norm",
        "dyn5m_macd_hist_slope_3",
        "dyn5m_macd_hist_slope_5",
        "dyn5m_macd_cross_age_frac",
        "dyn5m_macd_cross_dir",
        "dyn5m_macd_ready_flag",
    ]
    assert FEATURES_MARKET_OBS == expected
    _assert_unique(FEATURES_MARKET_OBS, "FEATURES_MARKET_OBS")
    removed = {
        "obs_V_t",
        "obs_I_t",
        "obs_volatility_t",
        "obs_weekday_sin_t",
        "obs_weekday_cos_t",
    }
    assert removed.isdisjoint(set(FEATURES_MARKET_OBS))


def test_features_risk_context_schema():
    expected = [
        "atr_1m_30_price_frac",
        "atr_1m_60_price_frac",
        "atr_1m_30_rolling_percentile",
        "atr_1m_60_rolling_percentile",
        "current_bar_range_atr_30",
        "intraday_volatility_percentile",
        "atr_ready_flag",
    ]
    assert FEATURES_RISK_CONTEXT == expected
    _assert_unique(FEATURES_RISK_CONTEXT, "FEATURES_RISK_CONTEXT")
    assert all("train_percentile" not in name for name in FEATURES_RISK_CONTEXT)


def test_features_htf_context_schema():
    expected = [
        "daily_ret_1",
        "daily_ret_3",
        "daily_ret_5",
        "daily_slope_5",
        "daily_slope_10",
        "daily_close_pos_in_range_5",
        "daily_close_pos_in_range_10",
        "daily_range_rolling_percentile",
        "daily_ready_flag",
        "h1_ret_1",
        "h1_ret_3",
        "h1_ret_6",
        "h1_ret_12",
        "h1_slope_6",
        "h1_slope_12",
        "h1_close_pos_in_range_6",
        "h1_close_pos_in_range_12",
        "h1_range_rolling_percentile",
        "last_completed_h1_age_frac",
        "h1_ready_flag",
    ]
    assert FEATURES_HTF_CONTEXT == expected
    _assert_unique(FEATURES_HTF_CONTEXT, "FEATURES_HTF_CONTEXT")
    assert all("train_percentile" not in name for name in FEATURES_HTF_CONTEXT)


def test_features_agent_schema():
    assert isinstance(FEATURES_AGENT, list)
    assert all(isinstance(x, str) for x in FEATURES_AGENT)
    _assert_unique(FEATURES_AGENT, "FEATURES_AGENT")
    assert FEATURES_AGENT[0] == "pos_t"
    assert FEATURES_AGENT[-1] == "minutes_to_timeout_t"
    # 关键字段必须存在
    must = {"equity_t", "max_equity_t", "drawdown_t", "fee_cum_t", "realized_pnl_cum_t"}
    assert must.issubset(set(FEATURES_AGENT))
