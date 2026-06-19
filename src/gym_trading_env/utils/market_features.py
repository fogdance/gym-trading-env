# utils/market_features.py

import numpy as np
import pandas as pd
from typing import List, Optional

from gym_trading_env.utils.session_fx import compute_session_meta
from gym_trading_env.utils.session_futures_strict import strict_reindex_futures_345
from gym_trading_env.utils.timebase import FEATURE_TZ as DEFAULT_TZ, ensure_index_tz_strict


# -----------------------------
# Market-side features (sequence) - OBS (normalized)
# -----------------------------
FEATURES_MARKET_OBS: List[str] = [
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

# -----------------------------
# Market-side features (sequence) - RAW
# -----------------------------
FEATURES_MARKET: List[str] = [
    "C_t",                    # Closing price at time t
    "V_t",                    # Volume at time t
    "I_t",                    # Open Interest at time t
    "cumVWAP_t",              # Cumulative VWAP up to time t
    "dC_minus_cumVWAP_t",     # C_t - cumVWAP_t
    "cmp_C_vs_cumVWAP_t",     # sign(C_t - cumVWAP_t) -> {-1,0,1}
    "ref_close_t",            # Reference closing price (e.g., previous day close)
    "session_open_t",         # NEW: session first valid open (fallback first valid close)
    "session_high_t",         # Session high price up to time t
    "session_low_t",          # Session low price up to time t
    "range_frac_t",           # NEW: (session_high-session_low)/ref_close (>=0)
    "bar_dir_t",              # Direction of the current bar (1 = up, -1 = down, 0 = flat)
    "turnover_t",             # cumulative sum of (price * volume) within session
    "minute_index_t",         # Minute index within the trading session
    "limit_up_price_t",       # Upper price limit at time t
    "limit_down_price_t",     # Lower price limit at time t
    "dI_from_yclose_t",       # Change in imbalance from yesterday's close
    "dP_from_ref_t",          # Price change from reference price
    "pct_chg_from_ref_t",     # Percentage change from reference price
    "open_drift_t",           # NEW: log(C/session_open) (direction kept)
    "mask_t",                 # Mask flag (e.g., valid data or trading halt)
    "weekday_sin_t",          # Sine-encoded weekday (for cyclical time feature)
    "weekday_cos_t",          # Cosine-encoded weekday (for cyclical time feature)
]

# 仅用于环境内部逻辑（止损/滑点/撮合等），不进 obs
AUX_MARKET_COLS = ["H_t", "L_t"]

# env 里会强依赖的列
REQUIRED_MARKET_COLS = ["day_id", "session_id", "trading_day"] + AUX_MARKET_COLS + FEATURES_MARKET + FEATURES_MARKET_OBS


# -----------------------------
# Helpers
# -----------------------------
def _weekday_cyc_from_sid(session_id: pd.Series) -> pd.DataFrame:
    d = pd.to_datetime(session_id.astype(str), format="%Y%m%d", errors="coerce")
    wd = d.dt.weekday
    return pd.DataFrame(
        {
            "weekday_sin_t": np.sin(2 * np.pi * (wd / 7.0)),
            "weekday_cos_t": np.cos(2 * np.pi * (wd / 7.0)),
        },
        index=session_id.index,
    )


def _rolling_percentile_causal(values: np.ndarray, window: int = 240, min_periods: int = 30) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    out = np.zeros_like(vals, dtype=float)
    for i in range(len(vals)):
        start = max(0, i - int(window) + 1)
        hist = vals[start:i + 1]
        hist = hist[np.isfinite(hist)]
        if hist.size < int(min_periods) or not np.isfinite(vals[i]):
            continue
        out[i] = 2.0 * (np.sum(hist <= vals[i]) / float(hist.size)) - 1.0
    return np.clip(out, -1.0, 1.0)


def _rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    out = np.zeros_like(vals, dtype=float)
    if int(window) <= 1:
        return out
    x = np.arange(int(window), dtype=float)
    x -= x.mean()
    denom = float(np.sum(x * x))
    for i in range(len(vals)):
        if i + 1 < int(window):
            continue
        y = vals[i - int(window) + 1:i + 1]
        if np.all(np.isfinite(y)):
            out[i] = float(np.sum(x * (y - y.mean())) / denom) if denom > 0 else 0.0
    return out


def _ema_next(prev: Optional[float], value: float, span: int) -> float:
    if prev is None:
        return float(value)
    alpha = 2.0 / (float(span) + 1.0)
    return float(prev + alpha * (float(value) - prev))


def _dynamic_5m_macd(close: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    macd = np.zeros(n, dtype=float)
    signal = np.zeros(n, dtype=float)
    hist = np.zeros(n, dtype=float)
    ready = np.zeros(n, dtype=float)

    fast_completed: Optional[float] = None
    slow_completed: Optional[float] = None
    signal_completed: Optional[float] = None
    completed_5m = 0
    valid_ordinal = -1
    last_valid_close = 0.0

    for i in range(n):
        if not bool(valid[i]):
            if i > 0:
                macd[i] = macd[i - 1]
                signal[i] = signal[i - 1]
                hist[i] = hist[i - 1]
                ready[i] = ready[i - 1]
            continue

        valid_ordinal += 1
        if valid_ordinal > 0 and valid_ordinal % 5 == 0:
            fast_completed = _ema_next(fast_completed, last_valid_close, 12)
            slow_completed = _ema_next(slow_completed, last_valid_close, 26)
            completed_macd = float(fast_completed - slow_completed)
            signal_completed = _ema_next(signal_completed, completed_macd, 9)
            completed_5m += 1

        c = float(close[i])
        fast_dyn = _ema_next(fast_completed, c, 12)
        slow_dyn = _ema_next(slow_completed, c, 26)
        macd_dyn = float(fast_dyn - slow_dyn)
        signal_dyn = _ema_next(signal_completed, macd_dyn, 9)
        macd[i] = macd_dyn
        signal[i] = signal_dyn
        hist[i] = macd_dyn - signal_dyn
        ready[i] = 1.0 if completed_5m >= 35 else 0.0
        last_valid_close = c

    return macd, signal, hist, ready


def _add_obs_features_inplace(df: pd.DataFrame) -> None:
    """Generate the formal v2 market_seq fields from raw market features."""
    eps = 1e-12
    m = df.get("mask_t", pd.Series(1.0, index=df.index)).astype(float)
    valid = (m.to_numpy(dtype=float, copy=False) > 0.0)
    mask_np = m.to_numpy(dtype=float, copy=False)
    sid = df.get("session_id", pd.Series(0, index=df.index)).astype(str)

    C_s = pd.to_numeric(df.get("C_t", 0.0), errors="coerce").fillna(0.0).astype(float)
    C = C_s.to_numpy(dtype=float, copy=False)
    ref = pd.to_numeric(df.get("ref_close_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    ref_safe = np.where(ref > eps, ref, np.where(C > eps, C, eps))
    H = pd.to_numeric(df.get("H_t", C_s), errors="coerce").fillna(0.0).astype(float).to_numpy()
    L = pd.to_numeric(df.get("L_t", C_s), errors="coerce").fillna(0.0).astype(float).to_numpy()
    V_s = pd.to_numeric(df.get("V_t", 0.0), errors="coerce").fillna(0.0).astype(float)
    I_s = pd.to_numeric(df.get("I_t", 0.0), errors="coerce").fillna(0.0).astype(float)
    I = I_s.to_numpy(dtype=float, copy=False)

    def _log_ratio(x: np.ndarray) -> np.ndarray:
        x_safe = np.where(x > eps, x, eps)
        out = np.zeros_like(x_safe, dtype=float)
        out[valid] = np.log(x_safe[valid] / ref_safe[valid])
        return np.clip(out, -1.0, 1.0)

    def _signed_log1p_ratio(z: np.ndarray) -> np.ndarray:
        out = np.zeros_like(z, dtype=float)
        ratio = np.abs(z) / np.maximum(ref_safe, eps)
        out[valid] = np.sign(z[valid]) * np.log1p(ratio[valid])
        return np.clip(out, -1.0, 1.0)

    cumVWAP = pd.to_numeric(df.get("cumVWAP_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    df["obs_cumVWAP_t"] = _log_ratio(cumVWAP)

    sh = pd.to_numeric(df.get("session_high_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    sl = pd.to_numeric(df.get("session_low_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    df["obs_session_high_t"] = _log_ratio(sh)
    df["obs_session_low_t"] = _log_ratio(sl)

    df["obs_pct_chg_from_ref_t"] = _log_ratio(C)

    if "range_frac_t" in df.columns:
        rf = pd.to_numeric(df["range_frac_t"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    else:
        rf = np.maximum(sh - sl, 0.0) / np.maximum(ref_safe, eps)

    obs_range = np.zeros_like(rf, dtype=float)
    rf_pos = np.clip(rf, 0.0, None)
    obs_range[valid] = np.log1p(rf_pos[valid])
    df["obs_range_t"] = np.clip(obs_range, 0.0, 1.0)

    dC = pd.to_numeric(df.get("dC_minus_cumVWAP_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    df["obs_dC_minus_cumVWAP_t"] = _signed_log1p_ratio(dC)

    dI = pd.to_numeric(df.get("dI_from_yclose_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    I_yclose = I - dI
    denom = np.maximum(np.abs(I_yclose), eps)

    out_dI = np.zeros_like(dI, dtype=float)
    ratio_I = np.abs(dI) / denom
    out_dI[valid] = np.sign(dI[valid]) * np.log1p(ratio_I[valid])
    df["obs_dI_from_yclose_t"] = np.clip(out_dI, -1.0, 1.0)

    if "open_drift_t" in df.columns:
        od = pd.to_numeric(df["open_drift_t"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    else:
        so = pd.to_numeric(df.get("session_open_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
        so_safe = np.where(so > eps, so, np.where(C > eps, C, eps))
        od = np.zeros_like(C, dtype=float)
        od[valid] = np.log(np.where(C > eps, C, eps)[valid] / so_safe[valid])

    df["obs_open_drift_t"] = np.clip(od, -1.0, 1.0)

    df["obs_cmp_C_vs_cumVWAP_t"] = (
        pd.to_numeric(df.get("cmp_C_vs_cumVWAP_t", 0.0), errors="coerce").fillna(0.0).astype(float)
    )
    df["obs_bar_dir_t"] = pd.to_numeric(df.get("bar_dir_t", 0.0), errors="coerce").fillna(0.0).astype(float)

    V_valid = V_s.where(m > 0.0, np.nan)
    ema = V_valid.groupby(sid).transform(
        lambda x: x.ewm(span=30, adjust=False, min_periods=5, ignore_na=True).mean()
    )
    ema_prev = ema.groupby(sid).shift(1)
    exp_mean = V_valid.groupby(sid).transform(lambda x: x.expanding(min_periods=1).mean())
    base = ema_prev.fillna(exp_mean.groupby(sid).shift(1)).fillna(0.0)
    volume_ratio = (V_s.to_numpy(dtype=float, copy=False) + 1.0) / (np.maximum(base.to_numpy(dtype=float, copy=False), 1.0) + 1.0)
    volume_surprise = np.clip(np.log(np.maximum(volume_ratio, eps)), -5.0, 5.0) * mask_np
    df["volume_surprise_logratio_floor"] = volume_surprise
    df["volume_surprise_rolling_percentile"] = _rolling_percentile_causal(volume_surprise, window=240, min_periods=30)
    df["volume_impulse_recent"] = pd.Series(volume_surprise, index=df.index).rolling(5, min_periods=1).mean().fillna(0.0).to_numpy(dtype=float)
    df["volume_impulse_slope_3"] = _rolling_slope(volume_surprise, 3)
    df["volume_impulse_slope_5"] = _rolling_slope(volume_surprise, 5)
    ready30 = pd.Series(valid.astype(float), index=df.index).rolling(30, min_periods=1).sum().to_numpy(dtype=float)
    df["volume_impulse_ready_flag"] = (ready30 >= 30.0).astype(float)

    rel_yclose = np.zeros_like(I, dtype=float)
    rel_yclose[valid] = np.sign(I[valid] - I_yclose[valid]) * np.log1p(
        np.abs(I[valid] - I_yclose[valid]) / np.maximum(np.abs(I_yclose[valid]), eps)
    )
    df["oi_rel_yclose_log"] = np.clip(rel_yclose, -5.0, 5.0)

    first_oi = I_s.where(m > 0.0, np.nan).groupby(sid).transform("first").fillna(I_s).to_numpy(dtype=float, copy=False)
    rel_open = np.zeros_like(I, dtype=float)
    rel_open[valid] = np.sign(I[valid] - first_oi[valid]) * np.log1p(
        np.abs(I[valid] - first_oi[valid]) / np.maximum(np.abs(first_oi[valid]), eps)
    )
    df["oi_rel_session_open_log"] = np.clip(rel_open, -5.0, 5.0)

    oi_delta = I_s.diff().fillna(0.0).where(m > 0.0, 0.0)
    oi_mean = oi_delta.rolling(240, min_periods=30).mean().shift(1)
    oi_std = oi_delta.rolling(240, min_periods=30).std(ddof=0).shift(1).replace(0.0, np.nan)
    oi_z = ((oi_delta - oi_mean) / oi_std).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    oi_z = np.clip(oi_z, -5.0, 5.0) * mask_np
    df["oi_delta_rolling_z"] = oi_z
    df["oi_rolling_percentile"] = _rolling_percentile_causal(oi_z, window=240, min_periods=30)
    df["oi_delta_slope_3"] = _rolling_slope(oi_z, 3)
    df["oi_delta_slope_5"] = _rolling_slope(oi_z, 5)

    prev5 = pd.Series(C, index=df.index).shift(5).to_numpy(dtype=float)
    logret5 = np.zeros_like(C, dtype=float)
    ok5 = valid & np.isfinite(prev5) & (prev5 > eps) & (C > eps)
    logret5[ok5] = np.log(C[ok5] / prev5[ok5])
    df["oi_price_confirm_short_continuous"] = np.clip(np.maximum(-logret5, 0.0) * np.maximum(oi_z, 0.0), 0.0, 5.0)
    df["oi_price_confirm_long_continuous"] = np.clip(np.maximum(logret5, 0.0) * np.maximum(oi_z, 0.0), 0.0, 5.0)
    df["oi_impulse_ready_flag"] = (ready30 >= 30.0).astype(float)

    for src, dst in [
        ("obs_cumVWAP_t", "obs_cumVWAP_t_rolling_percentile"),
        ("obs_session_high_t", "obs_session_high_t_rolling_percentile"),
        ("obs_range_t", "obs_range_t_rolling_percentile"),
        ("obs_pct_chg_from_ref_t", "obs_pct_chg_from_ref_t_rolling_percentile"),
    ]:
        df[dst] = _rolling_percentile_causal(
            pd.to_numeric(df[src], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            window=240,
            min_periods=30,
        )

    df["obs_mask_t"] = m.astype(float)

    mi = pd.to_numeric(df.get("minute_index_t", 0.0), errors="coerce").fillna(0.0).astype(float)
    if "day_id" in df.columns:
        denom_mi = pd.to_numeric(df["minute_index_t"], errors="coerce").groupby(df["day_id"]).transform("max")
    else:
        denom_mi = pd.Series(float(np.nanmax(mi.to_numpy())), index=df.index)
    denom_mi = pd.to_numeric(denom_mi, errors="coerce").fillna(1.0).astype(float).clip(lower=1.0)
    df["obs_minute_index_t"] = (mi / denom_mi).clip(0.0, 1.0).astype(float)

    mi_np = mi.to_numpy(dtype=float, copy=False)
    mi_max = float(np.nanmax(mi_np)) if len(mi_np) else 0.0
    if mi_max <= 400.0:
        df["obs_session_phase_t"] = (mi >= 120.0).astype(float)  # 0 night, 1 day
    else:
        df["obs_session_phase_t"] = 0.0

    C_valid = C_s.where(m > 0.0, np.nan)
    prev = C_valid.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.log(C_valid / prev)
    logret = logret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raw_vol = logret.rolling(window=30, min_periods=5).std().fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    df["vol_rolling_percentile"] = _rolling_percentile_causal(raw_vol, window=240, min_periods=30)

    prev_close = np.roll(C, 1)
    prev_close[0] = C[0] if len(C) else 0.0
    tr = np.maximum.reduce([
        np.maximum(H - L, 0.0),
        np.abs(H - prev_close),
        np.abs(L - prev_close),
    ])
    atr30 = pd.Series(np.where(valid, tr, np.nan), index=df.index).rolling(30, min_periods=30).mean().fillna(0.0).to_numpy(dtype=float)
    denom_macd = np.maximum(atr30, np.maximum(ref_safe * 1e-6, eps))
    macd, signal, hist, macd_ready = _dynamic_5m_macd(C, valid)
    hist_delta = np.zeros_like(hist)
    hist_delta[1:] = hist[1:] - hist[:-1]
    hist_slope3 = _rolling_slope(hist, 3)
    hist_slope5 = _rolling_slope(hist, 5)
    dist = macd - signal
    sign = np.sign(dist)
    cross_age = np.zeros_like(sign, dtype=float)
    last_cross_valid_ordinal = 0
    valid_age_ordinal = -1
    prev_sign = 0.0
    for i, sgn in enumerate(sign):
        if not bool(valid[i]):
            if i > 0:
                cross_age[i] = cross_age[i - 1]
            continue
        valid_age_ordinal += 1
        if sgn != 0 and prev_sign != 0 and sgn != prev_sign:
            last_cross_valid_ordinal = valid_age_ordinal
        if sgn != 0:
            prev_sign = sgn
        cross_age[i] = min(max(valid_age_ordinal - last_cross_valid_ordinal, 0) / 120.0, 1.0)

    df["dyn5m_macd_line_norm"] = np.clip(macd / denom_macd, -5.0, 5.0)
    df["dyn5m_macd_signal_norm"] = np.clip(signal / denom_macd, -5.0, 5.0)
    df["dyn5m_macd_hist_norm"] = np.clip(hist / denom_macd, -5.0, 5.0)
    df["dyn5m_macd_hist_delta"] = np.clip(hist_delta / denom_macd, -5.0, 5.0)
    df["dyn5m_macd_distance_norm"] = np.clip(dist / denom_macd, -5.0, 5.0)
    df["dyn5m_macd_hist_slope_3"] = np.clip(hist_slope3 / denom_macd, -5.0, 5.0)
    df["dyn5m_macd_hist_slope_5"] = np.clip(hist_slope5 / denom_macd, -5.0, 5.0)
    df["dyn5m_macd_cross_age_frac"] = cross_age
    df["dyn5m_macd_cross_dir"] = sign
    df["dyn5m_macd_ready_flag"] = macd_ready

    no_mask = {
        "volume_impulse_ready_flag",
        "oi_impulse_ready_flag",
        "obs_mask_t",
        "obs_minute_index_t",
        "obs_session_phase_t",
        "dyn5m_macd_ready_flag",
    }
    for col in FEATURES_MARKET_OBS:
        if col not in df.columns:
            df[col] = 0.0
        if col not in no_mask:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float) * m
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )


# -----------------------------
# Main API
# -----------------------------
def build_market_features(
    df_1m: pd.DataFrame,
    tz: str = DEFAULT_TZ,
    rollover_hour_local: int = 5,
    df_prev_session: Optional[pd.DataFrame] = None,
    is_future: bool = False,
    limit_up_pct: Optional[float] = None,
    limit_down_pct: Optional[float] = None,
):
    need_cols = {"Open", "High", "Low", "Close"}
    if not need_cols.issubset(df_1m.columns):
        raise ValueError(f"df_1m must contain {need_cols}")

    _ = ensure_index_tz_strict(df_1m.index, target_tz=tz)  # only validate

    if is_future:
        return _build_market_future(df_1m, tz, df_prev_session, limit_up_pct, limit_down_pct)
    else:
        return _build_market_fx(df_1m, tz, rollover_hour_local)


# -----------------------------
# FX
# -----------------------------
def _build_market_fx(df_1m: pd.DataFrame, tz: str = "Asia/Singapore", rollover_hour_local: int = 5):
    df = df_1m.copy()

    # legacy compatibility: Date column -> index
    if not isinstance(df.index, pd.DatetimeIndex) and "Date" in df.columns:
        df.index = pd.to_datetime(df["Date"])
        df.drop(columns=["Date"], inplace=True)

    meta = compute_session_meta(df, tz=tz, rollover_hour_local=rollover_hour_local)
    sid = meta["session_id"].astype(str)
    df["session_id"] = sid
    df["trading_day"] = sid.astype(int)

    # base OHLC
    df["L_t"] = pd.to_numeric(df["Low"], errors="coerce").astype(float)
    df["H_t"] = pd.to_numeric(df["High"], errors="coerce").astype(float)
    df["C_t"] = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    df["V_t"] = pd.to_numeric(df.get("Volume", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(float)
    df["I_t"] = 0.0

    # mask: hard gap rows invalid for learning / VWAP / hi-lo update
    has_ohlc = df[["Open", "High", "Low", "Close"]].notna().all(axis=1).astype(int)
    no_gap = (1 - meta["is_hard_gap"]).astype(int)
    df["mask_t"] = (has_ohlc * no_gap).astype(float)
    df["day_id"] = meta["day_id"].astype(np.int32)
    m_int = df["mask_t"].astype(int)

    # ref_close
    df["ref_close_t"] = pd.to_numeric(meta["prev_session_close"], errors="coerce").fillna(0.0).astype(float)

    # minute index / weekday
    df["minute_index_t"] = pd.to_numeric(meta["minute_index"], errors="coerce").fillna(0).astype(int)
    df["weekday_sin_t"] = pd.to_numeric(meta["weekday_sin"], errors="coerce").fillna(0.0).astype(float)
    df["weekday_cos_t"] = pd.to_numeric(meta["weekday_cos"], errors="coerce").fillna(0.0).astype(float)

    # cumVWAP (mask-filtered)
    valid_V = (df["V_t"] * m_int).astype(float)
    csum_cv = (df["C_t"] * valid_V).groupby(sid).cumsum()
    csum_v = valid_V.groupby(sid).cumsum()
    with np.errstate(divide="ignore", invalid="ignore"):
        cum_vwap = csum_cv / csum_v.replace(0, np.nan)
    c_valid = df["C_t"].where(m_int == 1, np.nan)
    cum_mean_c = c_valid.groupby(sid).expanding().mean().reset_index(level=0, drop=True)
    df["cumVWAP_t"] = pd.Series(cum_vwap).fillna(cum_mean_c).fillna(0.0).astype(float)

    df["dC_minus_cumVWAP_t"] = (df["C_t"] - df["cumVWAP_t"]).astype(float)
    df["cmp_C_vs_cumVWAP_t"] = np.sign(df["dC_minus_cumVWAP_t"]).astype(int)

    # session high/low (valid-only update, then ffill)
    H_valid = pd.to_numeric(df["High"], errors="coerce").astype(float).where(m_int == 1, np.nan)
    L_valid = pd.to_numeric(df["Low"], errors="coerce").astype(float).where(m_int == 1, np.nan)
    df["session_high_t"] = H_valid.groupby(sid).cummax().groupby(sid).ffill().fillna(0.0).astype(float)
    df["session_low_t"] = L_valid.groupby(sid).cummin().groupby(sid).ffill().fillna(0.0).astype(float)

    # NEW: session_open_t (first valid open; fallback first valid close)
    open_valid = pd.to_numeric(df["Open"], errors="coerce").astype(float).where(m_int == 1, np.nan)
    first_open = open_valid.groupby(sid).first()
    first_close = df["C_t"].where(m_int == 1, np.nan).groupby(sid).first()
    session_open = sid.map(first_open).astype(float)
    session_open = session_open.fillna(sid.map(first_close).astype(float)).fillna(0.0).astype(float)
    df["session_open_t"] = session_open

    # NEW: range_frac_t = (H-L)/ref_close
    eps = 1e-12
    ref_s = pd.to_numeric(df["ref_close_t"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    C_np = df["C_t"].astype(float).to_numpy()
    ref_safe = np.where(ref_s > eps, ref_s, np.where(C_np > eps, C_np, eps))
    rng = np.maximum(df["session_high_t"].astype(float).to_numpy() - df["session_low_t"].astype(float).to_numpy(), 0.0)
    range_frac = np.zeros_like(rng, dtype=float)
    vmask = (m_int.to_numpy(dtype=int, copy=False) == 1)
    range_frac[vmask] = rng[vmask] / ref_safe[vmask]
    df["range_frac_t"] = range_frac.astype(float)

    # NEW: open_drift_t = log(C/session_open)
    so = pd.to_numeric(df["session_open_t"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    so_safe = np.where(so > eps, so, np.where(C_np > eps, C_np, eps))
    open_drift = np.zeros_like(C_np, dtype=float)
    open_drift[vmask] = np.log(np.where(C_np > eps, C_np, eps)[vmask] / so_safe[vmask])
    df["open_drift_t"] = open_drift.astype(float)

    # bar_dir: only between consecutive valid minutes; minute_index==0 -> 0
    prev_C = df["C_t"].shift(1)
    prev_mask = m_int.shift(1).fillna(0)
    raw_dir = np.sign(df["C_t"] - prev_C).astype(float)
    bar_dir = np.where((m_int == 1) & (prev_mask == 1), raw_dir, 0.0)
    bar_dir = np.where(df["minute_index_t"].astype(int) == 0, 0.0, bar_dir)
    df["bar_dir_t"] = bar_dir.astype(int)

    # turnover: invalid minutes contribute 0
    cv = (df["C_t"] * valid_V).astype(float)
    df["turnover_t"] = cv.groupby(sid).cumsum().astype(float)

    # limits / dI
    df["limit_up_price_t"] = 0.0
    df["limit_down_price_t"] = 0.0
    df["dI_from_yclose_t"] = 0.0

    # dP / pct
    df["dP_from_ref_t"] = (df["C_t"] - df["ref_close_t"]).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = df["C_t"] / df["ref_close_t"] - 1.0
    df["pct_chg_from_ref_t"] = pd.Series(pct).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    # === RAW mask-zeroing (match futures behavior) ===
    m = df["mask_t"].astype(float)
    for col in FEATURES_MARKET + AUX_MARKET_COLS:
        if col == "mask_t":
            continue
        if col in ("minute_index_t", "weekday_sin_t", "weekday_cos_t"):
            continue
        if col in df.columns:
            df[col] = (df[col].astype(float) * m).astype(float)

    # clean raw
    for col in FEATURES_MARKET:
        df[col] = pd.to_numeric(df.get(col, 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    # obs
    _add_obs_features_inplace(df)
    for col in FEATURES_MARKET_OBS:
        df[col] = pd.to_numeric(df.get(col, 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    return df[REQUIRED_MARKET_COLS]


# -----------------------------
# Futures
# -----------------------------
def _build_market_future(
    df_1m: pd.DataFrame,
    tz: str = "Asia/Singapore",
    df_prev_session: Optional[pd.DataFrame] = None,
    limit_up_pct: Optional[float] = None,
    limit_down_pct: Optional[float] = None,
):
    # 1) strict 345 alignment
    base = strict_reindex_futures_345(df_1m, tz=tz)
    X = base["aligned"].copy()
    mask_t = base["mask"].astype(int)
    session_id = base["session_id"]

    X["session_id"] = session_id.astype(str)
    X["trading_day"] = session_id.astype(str).astype(int)

    minute_index = base["minute_index"]
    day_id = base["day_id"]

    # 2) base price/vol/oi
    X["C_t"] = pd.to_numeric(X["Close"], errors="coerce").fillna(0.0).astype(float)
    X["H_t"] = pd.to_numeric(X["High"], errors="coerce").fillna(0.0).astype(float)
    X["L_t"] = pd.to_numeric(X["Low"], errors="coerce").fillna(0.0).astype(float)
    X["V_t"] = pd.to_numeric(X.get("Volume", 0.0), errors="coerce").fillna(0.0).astype(float)
    X["I_t"] = pd.to_numeric(X.get("OpenInterest", 0.0), errors="coerce").fillna(0.0).astype(float)

    # 3) ref_close: last valid close of prev session; fallback to first valid close of current
    valid_close = pd.to_numeric(X["Close"], errors="coerce").astype(float).where(mask_t == 1, np.nan)

    sid_order = pd.Index(pd.unique(session_id.astype(str)))
    first_valid_by_sid = valid_close.groupby(session_id).first()
    last_valid_by_sid = valid_close.groupby(session_id).last()

    first_valid_ordered = pd.Series([first_valid_by_sid.get(sid, np.nan) for sid in sid_order], index=sid_order)
    last_valid_ordered = pd.Series([last_valid_by_sid.get(sid, np.nan) for sid in sid_order], index=sid_order)

    prev_close_map = last_valid_ordered.shift(1)
    ref_close_t = session_id.map(prev_close_map).astype(float)

    # 缺上一日收盘时：用本 session “第一个有效 close” 兜底（避免夜盘补零分钟污染）
    ref_close_t = ref_close_t.fillna(session_id.map(first_valid_ordered).astype(float))

    # 如果 caller 传了 df_prev_session：第一交易日直接用 prev_session 的“最后有效 close”
    if df_prev_session is not None and len(df_prev_session) > 0:
        prev = strict_reindex_futures_345(df_prev_session, tz=tz)
        prev_aligned = prev["aligned"]
        prev_mask = prev["mask"].astype(int)
        prev_valid_close = pd.to_numeric(prev_aligned["Close"], errors="coerce").astype(float).where(prev_mask == 1, np.nan)
        if prev_valid_close.notna().any():
            prev_last_close = float(prev_valid_close.dropna().iloc[-1])
        else:
            prev_last_close = float(pd.to_numeric(prev_aligned["Close"].tail(1), errors="coerce").fillna(0.0).iloc[0])

        first_sid = session_id.iloc[0]
        ref_close_t.loc[session_id == first_sid] = prev_last_close

    # 仍保持 ffill/bfill（兼容极端情况：整段都 NaN）
    X["ref_close_t"] = ref_close_t.ffill().bfill().fillna(0.0).astype(float)

    # 4) cumVWAP (mask-filtered)
    valid_V = X["V_t"] * mask_t
    csum_cv = (X["C_t"] * valid_V).groupby(session_id).cumsum()
    csum_v = valid_V.groupby(session_id).cumsum()
    with np.errstate(divide="ignore", invalid="ignore"):
        cum_vwap = csum_cv / csum_v
    c_valid = X["C_t"].where(mask_t == 1, np.nan)
    cum_mean_c = c_valid.groupby(session_id).expanding().mean().reset_index(level=0, drop=True)
    X["cumVWAP_t"] = pd.Series(cum_vwap).fillna(cum_mean_c).fillna(0.0).astype(float)

    X["dC_minus_cumVWAP_t"] = (X["C_t"] - X["cumVWAP_t"]).astype(float)
    X["cmp_C_vs_cumVWAP_t"] = np.sign(X["dC_minus_cumVWAP_t"]).astype(int)

    # 5) session high/low (valid-only update)
    H_valid = pd.to_numeric(X["High"], errors="coerce").astype(float).where(mask_t == 1, np.nan)
    L_valid = pd.to_numeric(X["Low"], errors="coerce").astype(float).where(mask_t == 1, np.nan)
    session_high = H_valid.groupby(session_id).cummax()
    session_low = L_valid.groupby(session_id).cummin()
    X["session_high_t"] = session_high.groupby(session_id).ffill().fillna(0.0).astype(float)
    X["session_low_t"] = session_low.groupby(session_id).ffill().fillna(0.0).astype(float)

    # NEW: session_open_t (first valid open; fallback first valid close)
    open_valid = pd.to_numeric(X.get("Open", np.nan), errors="coerce").astype(float).where(mask_t == 1, np.nan)
    first_open = open_valid.groupby(session_id).first()
    first_close = X["C_t"].where(mask_t == 1, np.nan).groupby(session_id).first()
    session_open = session_id.map(first_open).astype(float)
    session_open = session_open.fillna(session_id.map(first_close).astype(float)).fillna(0.0).astype(float)
    X["session_open_t"] = session_open

    # NEW: range_frac_t = (H-L)/ref_close
    eps = 1e-12
    ref_s = pd.to_numeric(X["ref_close_t"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    C_np = X["C_t"].astype(float).to_numpy()
    ref_safe = np.where(ref_s > eps, ref_s, np.where(C_np > eps, C_np, eps))
    rng = np.maximum(X["session_high_t"].astype(float).to_numpy() - X["session_low_t"].astype(float).to_numpy(), 0.0)
    range_frac = np.zeros_like(rng, dtype=float)
    vmask = (np.asarray(mask_t, dtype=int) == 1)
    range_frac[vmask] = rng[vmask] / ref_safe[vmask]
    X["range_frac_t"] = range_frac.astype(float)

    # NEW: open_drift_t = log(C/session_open)
    so = pd.to_numeric(X["session_open_t"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    so_safe = np.where(so > eps, so, np.where(C_np > eps, C_np, eps))
    open_drift = np.zeros_like(C_np, dtype=float)
    open_drift[vmask] = np.log(np.where(C_np > eps, C_np, eps)[vmask] / so_safe[vmask])
    X["open_drift_t"] = open_drift.astype(float)

    # 6) bar_dir (valid-valid only; minute_index==0 -> 0)
    prev_close = X["C_t"].shift(1)
    prev_mask = pd.Series(mask_t).shift(1).fillna(0).to_numpy()
    raw_dir = np.sign(X["C_t"] - prev_close).astype(float)
    bar_dir = np.where((mask_t == 1) & (prev_mask == 1), raw_dir, 0.0)
    bar_dir = np.where(minute_index.to_numpy() == 0, 0.0, bar_dir)
    X["bar_dir_t"] = bar_dir.astype(int)

    # 7) turnover (invalid minutes contribute 0)
    cv = (X["C_t"] * (X["V_t"] * mask_t)).astype(float)
    X["turnover_t"] = cv.groupby(session_id).cumsum().astype(float)

    # 8) minute index / weekday
    X["minute_index_t"] = minute_index.astype(int)
    wd = _weekday_cyc_from_sid(session_id)
    X["weekday_sin_t"] = wd["weekday_sin_t"].astype(float)
    X["weekday_cos_t"] = wd["weekday_cos_t"].astype(float)

    # 9) limits
    if (limit_up_pct is not None) and (limit_down_pct is not None):
        up = float(limit_up_pct)
        down = float(limit_down_pct)
        if not (0.0 <= up <= 1.0):
            raise ValueError(f"limit_up_pct must be in [0, 1], got {limit_up_pct}")
        if not (0.0 <= down <= 1.0):
            raise ValueError(f"limit_down_pct must be in [0, 1], got {limit_down_pct}")
        X["limit_up_price_t"] = X["ref_close_t"] * (1.0 + up)
        X["limit_down_price_t"] = X["ref_close_t"] * (1.0 - down)
    else:
        X["limit_up_price_t"] = 0.0
        X["limit_down_price_t"] = 0.0

    # 10) dI_from_yclose
    valid_oi = X["I_t"].where(mask_t == 1, np.nan).astype(float)

    sid_order2 = pd.Index(pd.unique(session_id))
    first_oi_by_sid = valid_oi.groupby(session_id).first().reindex(sid_order2)
    last_oi_by_sid = valid_oi.groupby(session_id).last().reindex(sid_order2)

    prev_oi_map = last_oi_by_sid.shift(1)
    I_yclose = session_id.map(prev_oi_map).astype(float)

    # 兜底：没有上一日 OI 时，用本 session first-valid OI（避免无效分钟污染）
    I_yclose = I_yclose.fillna(session_id.map(first_oi_by_sid).astype(float))

    # 如果 caller 传了 df_prev_session：第一交易日直接用 prev_session 的 last-valid OI
    if df_prev_session is not None and len(df_prev_session) > 0:
        prev = strict_reindex_futures_345(df_prev_session, tz=tz)
        prev_aligned = prev["aligned"]
        prev_mask = prev["mask"].astype(int)
        prev_valid_oi = pd.to_numeric(prev_aligned.get("OpenInterest", 0.0), errors="coerce").astype(float)
        prev_valid_oi = pd.Series(prev_valid_oi).where(prev_mask == 1, np.nan)

        if prev_valid_oi.notna().any():
            prev_last_oi = float(prev_valid_oi.dropna().iloc[-1])
        else:
            prev_last_oi = float(pd.to_numeric(prev_aligned.get("OpenInterest", 0.0), errors="coerce").fillna(0.0).iloc[-1])

        first_sid = session_id.iloc[0]
        I_yclose.loc[session_id == first_sid] = prev_last_oi

    X["dI_from_yclose_t"] = (X["I_t"].astype(float) - I_yclose).fillna(0.0).astype(float)

    # 11) dP / pct
    X["dP_from_ref_t"] = (X["C_t"] - X["ref_close_t"]).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = X["C_t"] / X["ref_close_t"] - 1.0
    X["pct_chg_from_ref_t"] = pd.Series(pct).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    # 12) mask / day_id
    X["mask_t"] = mask_t.astype(float)
    X["day_id"] = day_id.astype(np.int32)

    # 12.5) RAW mask-zeroing
    m = X["mask_t"].astype(float)
    for col in FEATURES_MARKET + AUX_MARKET_COLS:
        if col == "mask_t":
            continue
        if col in ("minute_index_t", "weekday_sin_t", "weekday_cos_t"):
            continue
        if col in X.columns:
            X[col] = (X[col].astype(float) * m).astype(float)

    # 13) clean raw
    for col in FEATURES_MARKET:
        if col not in X.columns:
            X[col] = 0.0
        X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    # obs
    _add_obs_features_inplace(X)
    for col in FEATURES_MARKET_OBS:
        if col not in X.columns:
            X[col] = 0.0
        X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    return X[REQUIRED_MARKET_COLS]
