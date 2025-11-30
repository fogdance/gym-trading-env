# tests/unit/test_build_market_future_features.py
import pytest
import numpy as np
import pandas as pd

from gym_trading_env.utils.market_features import FEATURES_MARKET, REQUIRED_MARKET_COLS, build_market_features
from gym_trading_env.utils.agent_features import FEATURES_AGENT

pytestmark = pytest.mark.unit
TZ = "Asia/Shanghai"


def make_intraday_rows(
    start_ts: str,
    minutes: int,
    price_base=2300.0,
    vol=10.0,
    oi_base=1000.0,
    oi_step=0.0,
):
    """
    Make consecutive minute rows starting from start_ts in TZ.
    This intentionally only covers the first few canonical minutes; the rest should be mask=0 after strict reindex.
    """
    idx = pd.date_range(start=start_ts, periods=minutes, freq="1min", tz=TZ)
    c = price_base + np.arange(minutes, dtype=float) * 1.0
    oi = oi_base + np.arange(minutes, dtype=float) * float(oi_step)
    return (
        pd.DataFrame(
            {
                "Date": idx,
                "Open": c,
                "High": c,
                "Low": c,
                "Close": c,
                "Volume": np.full(minutes, float(vol), dtype=float),
                "OpenInterest": oi.astype(float),
            }
        )
        .set_index("Date")
    )


def _arr(out: pd.DataFrame, col: str, n=3) -> np.ndarray:
    return out[col].to_numpy(dtype=float)[:n]


def _assert_all_invalid_minutes_zeroed(out: pd.DataFrame, start_invalid_row: int = 3):
    m = out["mask_t"].to_numpy(dtype=float)
    assert np.all(m[start_invalid_row:] == 0.0), "Expected all minutes after provided data to be mask=0"
    for col in FEATURES_MARKET:
        if col == "mask_t":
            continue
        a = out[col].to_numpy(dtype=float)
        assert np.all(a[start_invalid_row:] == 0.0), f"{col} should be 0 when mask=0 (future branch contract)"


def test_build_market_future_columns_and_shape_and_finite():
    df = make_intraday_rows("2024-05-20 21:01:00", minutes=3)
    out = build_market_features(df, tz=TZ, is_future=True, limit_up_pct=0.1, limit_down_pct=0.1)

    assert out.shape[0] == 345
    assert list(out.columns) == REQUIRED_MARKET_COLS

    X = out[FEATURES_MARKET].to_numpy(dtype=float)
    assert np.isfinite(X).all(), "Features contain NaN/Inf"


def test_build_market_future_oracles_on_first_3_valid_minutes():
    """
    L1 oracle test: verify EACH market feature column on valid minutes against hand/identity computations,
    and verify mask=0 minutes are fully zeroed (except mask_t).
    """
    # 3 valid minutes, monotonic close, constant volume, non-constant OI to catch dI_from_yclose rule.
    df = make_intraday_rows(
        "2024-05-20 21:01:00",
        minutes=3,
        price_base=100.0,
        vol=1.0,
        oi_base=5.0,
        oi_step=1.0,   # OI: 5,6,7 => dI_from_yclose should become 0,1,2 (first session yclose uses first_I0)
    )
    out = build_market_features(df, tz=TZ, is_future=True, limit_up_pct=0.1, limit_down_pct=0.1)

    # mask: first 3 valid, rest invalid
    m = out["mask_t"].to_numpy(dtype=float)
    assert np.all(m[:3] == 1.0)
    assert np.all(m[3:] == 0.0)

    closes = np.array([100.0, 101.0, 102.0], dtype=float)
    vols = np.array([1.0, 1.0, 1.0], dtype=float)
    ois = np.array([5.0, 6.0, 7.0], dtype=float)

    # C/V/I
    np.testing.assert_allclose(_arr(out, "C_t"), closes, rtol=0, atol=0)
    np.testing.assert_allclose(_arr(out, "V_t"), vols, rtol=0, atol=0)
    np.testing.assert_allclose(_arr(out, "I_t"), ois, rtol=0, atol=0)

    # ref_close_t: with no prev_session, first session ref_close should be first close (for valid minutes)
    ref = np.array([closes[0], closes[0], closes[0]], dtype=float)
    np.testing.assert_allclose(_arr(out, "ref_close_t"), ref, rtol=0, atol=0)

    # cumVWAP_t: volume constant => cumulative mean of closes
    exp_cumvwap = np.cumsum(closes * vols) / np.cumsum(vols)
    np.testing.assert_allclose(_arr(out, "cumVWAP_t"), exp_cumvwap, rtol=0, atol=1e-12)

    # dC_minus_cumVWAP_t and cmp_C_vs_cumVWAP_t
    d = closes - exp_cumvwap
    np.testing.assert_allclose(_arr(out, "dC_minus_cumVWAP_t"), d, rtol=0, atol=1e-12)
    np.testing.assert_allclose(_arr(out, "cmp_C_vs_cumVWAP_t"), np.sign(d), rtol=0, atol=0)

    # session_high / session_low (valid minutes)
    np.testing.assert_allclose(_arr(out, "session_high_t"), np.maximum.accumulate(closes), rtol=0, atol=0)
    np.testing.assert_allclose(_arr(out, "session_low_t"), np.minimum.accumulate(closes), rtol=0, atol=0)

    # bar_dir_t: minute0 forced 0; subsequent = sign(delta close) when prev_mask=1
    np.testing.assert_allclose(_arr(out, "bar_dir_t"), np.array([0.0, 1.0, 1.0]), rtol=0, atol=0)

    # turnover_t: cumulative amount = cumsum(C_t * V_t) within session
    exp_turnover = np.cumsum(closes * vols)
    np.testing.assert_allclose(_arr(out, "turnover_t"), exp_turnover, rtol=0, atol=1e-12)


    # minute_index_t: first three should map to 0,1,2
    np.testing.assert_allclose(_arr(out, "minute_index_t"), np.array([0.0, 1.0, 2.0]), rtol=0, atol=0)

    # limit up/down: ref_close*(1±pct)
    np.testing.assert_allclose(_arr(out, "limit_up_price_t"), ref * 1.1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(_arr(out, "limit_down_price_t"), ref * 0.9, rtol=0, atol=1e-12)

    # dI_from_yclose_t: first session yclose = first I, so delta = [0,1,2]
    exp_dI = ois - ois[0]
    np.testing.assert_allclose(_arr(out, "dI_from_yclose_t"), exp_dI, rtol=0, atol=1e-12)

    # dP_from_ref_t and pct_chg_from_ref_t
    exp_dP = closes - ref
    exp_pct = closes / ref - 1.0
    np.testing.assert_allclose(_arr(out, "dP_from_ref_t"), exp_dP, rtol=0, atol=1e-12)
    np.testing.assert_allclose(_arr(out, "pct_chg_from_ref_t"), exp_pct, rtol=0, atol=1e-12)

    # weekday sin/cos: strong contracts (do not assume exact trading-day mapping here)
    ws = _arr(out, "weekday_sin_t")
    wc = _arr(out, "weekday_cos_t")
    assert np.all(np.isfinite(ws)) and np.all(np.isfinite(wc))
    assert np.all(np.abs(ws) <= 1.0 + 1e-12)
    assert np.all(np.abs(wc) <= 1.0 + 1e-12)
    # within a session they should be constant on valid minutes
    np.testing.assert_allclose(ws, np.full(3, ws[0]), rtol=0, atol=0)
    np.testing.assert_allclose(wc, np.full(3, wc[0]), rtol=0, atol=0)
    # sin^2 + cos^2 == 1
    np.testing.assert_allclose(ws * ws + wc * wc, np.ones(3), rtol=0, atol=1e-12)

    # mask=0 minutes: everything zeroed except mask itself
    _assert_all_invalid_minutes_zeroed(out, start_invalid_row=3)


def test_build_market_future_bar_dir_requires_prev_valid_minute():
    """
    If minute 1 is missing, then for minute 2 prev_mask==0 => bar_dir_t must be 0 (even if close increased).
    """
    df = make_intraday_rows("2024-05-20 21:01:00", minutes=3, price_base=100.0, vol=1.0, oi_base=5.0)
    # drop the middle minute (21:02)
    df = df.iloc[[0, 2]].copy()

    out = build_market_features(df, tz=TZ, is_future=True)

    m = out["mask_t"].to_numpy(dtype=float)
    assert m[0] == 1.0
    assert m[1] == 0.0
    assert m[2] == 1.0

    # minute 2 should NOT look back across an invalid minute
    assert float(out["bar_dir_t"].iloc[2]) == 0.0
    # and invalid minute is fully zeroed
    for col in FEATURES_MARKET:
        if col == "mask_t":
            continue
        assert float(out[col].iloc[1]) == 0.0
