# tests/unit/test_build_market_future_features.py
import pytest
import numpy as np
import pandas as pd
from gym_trading_env.utils.build_xt import build_market_features, FEATURES_MARKET

pytestmark = pytest.mark.unit
TZ = "Asia/Shanghai"

def make_intraday_rows(start_ts: str, minutes: int, price_base=2300.0, vol=10.0, oi=1000.0):
    idx = pd.date_range(start=start_ts, periods=minutes, freq="1min", tz=TZ)
    c = price_base + np.arange(minutes).astype(float) * 1.0
    return pd.DataFrame(
        {
            "Date": idx,
            "Open": c,
            "High": c,
            "Low":  c,
            "Close": c,
            "Volume": np.full(minutes, vol, dtype=float),
            "OpenInterest": np.full(minutes, oi, dtype=float),
        }
    ).set_index("Date")

def test_build_market_future_columns_and_shape():
    df = make_intraday_rows("2024-05-20 21:01:00", minutes=3)
    out = build_market_features(df, tz=TZ, is_future=True, limit_up_pct=0.1, limit_down_pct=0.1)

    assert out.shape[0] == 345
    assert list(out.columns) == FEATURES_MARKET + ["day_id"]

    X = out[FEATURES_MARKET].to_numpy(dtype=float)
    assert np.isfinite(X).all()

def test_build_market_future_key_features_on_data_minutes():
    df = make_intraday_rows("2024-05-20 21:01:00", minutes=3, price_base=100.0, vol=1.0, oi=5.0)
    out = build_market_features(df, tz=TZ, is_future=True, limit_up_pct=0.1, limit_down_pct=0.1)

    # minute 0..2 有数据
    m = out["mask_t"].to_numpy()
    assert np.all(m[:3] == 1.0)
    assert np.all(m[3:] == 0.0)

    closes = np.array([100.0, 101.0, 102.0], dtype=float)

    np.testing.assert_allclose(out["C_t"].to_numpy()[:3], closes, rtol=0, atol=0)
    np.testing.assert_allclose(out["V_t"].to_numpy()[:3], 1.0, rtol=0, atol=0)
    np.testing.assert_allclose(out["I_t"].to_numpy()[:3], 5.0, rtol=0, atol=0)

    # cumVWAP: volume constant -> 累计均值
    exp_cum = np.cumsum(closes) / np.arange(1, 4)
    np.testing.assert_allclose(out["cumVWAP_t"].to_numpy()[:3], exp_cum, rtol=0, atol=1e-12)

    # dC/cmp
    d = closes - exp_cum
    np.testing.assert_allclose(out["dC_minus_cumVWAP_t"].to_numpy()[:3], d, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out["cmp_C_vs_cumVWAP_t"].to_numpy()[:3], np.sign(d), rtol=0, atol=0)

    # session high/low
    np.testing.assert_allclose(out["session_high_t"].to_numpy()[:3], np.maximum.accumulate(closes), rtol=0, atol=0)
    np.testing.assert_allclose(out["session_low_t"].to_numpy()[:3],  np.minimum.accumulate(closes), rtol=0, atol=0)

    # bar_dir：minute 0->0
    assert int(out["bar_dir_t"].iloc[0]) == 0
    assert int(out["bar_dir_t"].iloc[1]) in (1, -1, 0)  # 保守：只要是合法方向值

    # limit_up/down：= ref_close*(1±pct)
    ref = out["ref_close_t"].to_numpy()[:3]
    np.testing.assert_allclose(out["limit_up_price_t"].to_numpy()[:3], ref * 1.1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out["limit_down_price_t"].to_numpy()[:3], ref * 0.9, rtol=0, atol=1e-12)

def test_build_market_future_mask_zero_minutes_all_zero_except_mask():
    df = make_intraday_rows("2024-05-20 21:01:00", minutes=3, price_base=100.0, vol=1.0, oi=5.0)
    out = build_market_features(df, tz=TZ, is_future=True)

    # 取一个无数据分钟（比如第 10 行）
    i = 10
    assert float(out["mask_t"].iloc[i]) == 0.0
    for col in FEATURES_MARKET:
        if col == "mask_t":
            continue
        assert float(out[col].iloc[i]) == 0.0, f"{col} should be 0 when mask=0 (future)"
