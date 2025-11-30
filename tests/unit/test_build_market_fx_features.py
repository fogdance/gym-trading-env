# tests/unit/test_build_market_fx_features.py
import pytest
import numpy as np
import pandas as pd
from gym_trading_env.utils.market_features import FEATURES_MARKET, REQUIRED_MARKET_COLS, build_market_features
from gym_trading_env.utils.agent_features import FEATURES_AGENT

pytestmark = pytest.mark.unit

def _mk_fx_df(start="2020-01-01 21:01:00", closes=None, vols=None):
    if closes is None:
        closes = [1.0, 2.0, 3.0, 4.0, 5.0]
    n = len(closes)
    idx = pd.date_range(start=start, periods=n, freq="1min")
    if vols is None:
        vols = [1.0] * n
    df = pd.DataFrame(
        {
            "Date": idx,
            "Open": closes,
            "High": closes,
            "Low":  closes,
            "Close": closes,
            "Volume": vols,
        }
    )
    return df

def test_build_market_fx_columns_types_and_no_nan():
    df = _mk_fx_df()
    out = build_market_features(df, tz="Asia/Singapore", rollover_hour_local=5, is_future=False)

    # 列完整 + 顺序契约
    assert list(out.columns) == REQUIRED_MARKET_COLS

    # dtype: 特征应为 float，day_id 为 int32
    for c in FEATURES_MARKET:
        assert np.issubdtype(out[c].dtype, np.floating), f"{c} dtype={out[c].dtype}"
    assert out["day_id"].dtype == np.int32

    # 无 NaN/Inf
    X = out[FEATURES_MARKET].to_numpy(dtype=float)
    assert np.isfinite(X).all()

def test_build_market_fx_key_features_values():
    closes = [1.0, 2.0, 3.0, 4.0, 5.0]
    df = _mk_fx_df(closes=closes, vols=[1,1,1,1,1])
    out = build_market_features(df, tz="Asia/Singapore", rollover_hour_local=5, is_future=False)

    # minute_index: 从 session start(05:00) 起应为 0..n-1
    assert out["minute_index_t"].astype(int).tolist() == [0,1,2,3,4]

    # mask：无 gap/无缺失 -> 全 1
    assert np.all(out["mask_t"].to_numpy() == 1.0)

    # cumVWAP：volume==1 -> 累计均值
    exp_cum = np.cumsum(closes) / np.arange(1, len(closes)+1)
    np.testing.assert_allclose(out["cumVWAP_t"].to_numpy(), exp_cum, rtol=0, atol=1e-12)

    # dC_minus_cumVWAP & cmp
    d = np.array(closes) - exp_cum
    np.testing.assert_allclose(out["dC_minus_cumVWAP_t"].to_numpy(), d, rtol=0, atol=1e-12)
    cmp = np.sign(d)
    np.testing.assert_allclose(out["cmp_C_vs_cumVWAP_t"].to_numpy(), cmp, rtol=0, atol=0)

    # session high/low
    np.testing.assert_allclose(out["session_high_t"].to_numpy(), np.maximum.accumulate(closes), rtol=0, atol=0)
    np.testing.assert_allclose(out["session_low_t"].to_numpy(),  np.minimum.accumulate(closes), rtol=0, atol=0)

    # bar_dir：首根 0，之后递增为 1
    assert out["bar_dir_t"].astype(int).tolist() == [0,1,1,1,1]

    
    # turnover：累计成交额 = cumsum(C*V)
    exp_turnover = np.cumsum(np.array(closes, dtype=float) * 1.0)
    np.testing.assert_allclose(out["turnover_t"].to_numpy(dtype=float), exp_turnover, rtol=0, atol=1e-12)

def test_build_market_fx_rollover_ref_close_uses_prev_session_close():
    # 构造跨 rollover(05:00) 的 3 根：04:59 属于上一 session，05:00/05:01 属于新 session
    idx = pd.to_datetime(["2020-01-01 04:59:00", "2020-01-01 21:01:00", "2020-01-01 05:01:00"])
    closes = [100.0, 200.0, 201.0]
    df = pd.DataFrame(
        {"Date": idx, "Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume":[1,1,1]}
    )
    out = build_market_features(df, tz="Asia/Singapore", rollover_hour_local=5, is_future=False)

    # day_id 应在 rollover 处切换（不关心具体数值，只关心变化）
    assert out["day_id"].iloc[0] != out["day_id"].iloc[1]

    # 新 session 的 ref_close_t 应等于上一 session 的最后 close（这里只有 04:59=100）
    assert float(out["ref_close_t"].iloc[1]) == 100.0
    assert float(out["ref_close_t"].iloc[2]) == 100.0
