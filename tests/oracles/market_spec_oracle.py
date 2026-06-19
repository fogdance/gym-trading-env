# tests/oracles/market_spec_oracle.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd


FEATURE_TZ = "Asia/Shanghai"

# ===== 列清单（Spec 冻结） =====
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

FEATURES_RISK_CONTEXT: List[str] = [
    "atr_1m_30_price_frac",
    "atr_1m_60_price_frac",
    "atr_1m_30_rolling_percentile",
    "atr_1m_60_rolling_percentile",
    "current_bar_range_atr_30",
    "intraday_volatility_percentile",
    "atr_ready_flag",
]

FEATURES_MARKET: List[str] = [
    "C_t",
    "V_t",
    "I_t",
    "cumVWAP_t",
    "dC_minus_cumVWAP_t",
    "cmp_C_vs_cumVWAP_t",
    "ref_close_t",
    "session_open_t",
    "session_high_t",
    "session_low_t",
    "range_frac_t",
    "bar_dir_t",
    "turnover_t",
    "minute_index_t",
    "limit_up_price_t",
    "limit_down_price_t",
    "dI_from_yclose_t",
    "dP_from_ref_t",
    "pct_chg_from_ref_t",
    "open_drift_t",
    "mask_t",
    "weekday_sin_t",
    "weekday_cos_t",
]

AUX_MARKET_COLS = ["H_t", "L_t"]
REQUIRED_MARKET_COLS = (
    ["day_id", "session_id", "trading_day"]
    + AUX_MARKET_COLS
    + FEATURES_MARKET
    + FEATURES_MARKET_OBS
    + FEATURES_RISK_CONTEXT
)

SEGMENTS: List[Tuple[str, str]] = [
    ("21:00", "23:00"),
    ("09:00", "10:15"),
    ("10:30", "11:30"),
    ("13:30", "15:00"),
]


def _build_clock_for_trading_day(trading_day: pd.Timestamp, tz: str) -> pd.DatetimeIndex:
    trading_day = pd.Timestamp(trading_day, tz=tz).normalize()
    night_date = (trading_day - pd.Timedelta(days=1)).normalize()
    parts = []
    for i, (s_str, e_str) in enumerate(SEGMENTS):
        base = night_date if i == 0 else trading_day
        s = pd.Timestamp(f"{base.date()} {s_str}", tz=tz)
        e = pd.Timestamp(f"{base.date()} {e_str}", tz=tz)
        rng = pd.date_range(start=s, end=e, freq="1min", inclusive="left", tz=tz) + pd.Timedelta(minutes=1)
        parts.append(rng)
    clock = parts[0]
    for p in parts[1:]:
        clock = clock.append(p)
    assert len(clock) == 345
    return clock


def strict_reindex_futures_345_spec(df_1m: pd.DataFrame, tz: str = FEATURE_TZ) -> Dict[str, object]:
    """
    Spec 版 strict 345：
    - 优先使用 df_1m['trading_day'] 做 session_id
    - mask = Close notna（fillna 前）
    - OHLCVI 缺失填 0
    """
    if not isinstance(df_1m.index, pd.DatetimeIndex):
        raise ValueError("df_1m.index must be DatetimeIndex")
    if df_1m.index.tz is None:
        raise ValueError("df_1m.index must be tz-aware (FEATURE_TZ)")

    df = df_1m.copy()
    df.index = df.index.tz_convert(tz)

    if "trading_day" in df.columns:
        td = pd.to_numeric(df["trading_day"], errors="coerce").astype("Int64")
        if td.isna().all():
            raise ValueError("df_1m.trading_day exists but all NaN")
        session_id_series = td.astype(str)
        session_id_series.index = df.index
    else:
        # 没有 trading_day：退化策略（测试里尽量别走这里）
        session_id_series = pd.Series(df.index.tz_convert(tz).strftime("%Y%m%d"), index=df.index)

    out_frames = []
    out_masks = []
    out_sid = []
    out_minidx = []

    for sid_str, loc in session_id_series.groupby(session_id_series, sort=False):
        day_date = pd.to_datetime(sid_str, format="%Y%m%d").date()
        day_ts = pd.Timestamp(day_date, tz=tz)

        df_day = df.loc[loc.index].copy()
        # 如果有夜盘真实数据，可用真实夜盘日期，否则默认 day-1
        night_rows = df_day.index[df_day.index.hour >= 21]
        if len(night_rows) > 0:
            night_date = pd.Timestamp(night_rows.min()).normalize()
        else:
            night_date = (day_ts - pd.Timedelta(days=1)).normalize()

        # 构造 clock（夜盘用 night_date，其余用 day_ts）
        # 为了对齐实现，这里复用同样的 clock 生成逻辑
        # 但 night_date 影响第一段日期
        parts = []
        for i, (s_str, e_str) in enumerate(SEGMENTS):
            base = night_date if i == 0 else day_ts
            s = pd.Timestamp(f"{base.date()} {s_str}", tz=tz)
            e = pd.Timestamp(f"{base.date()} {e_str}", tz=tz)
            rng = pd.date_range(start=s, end=e, freq="1min", inclusive="left", tz=tz) + pd.Timedelta(minutes=1)
            parts.append(rng)
        clock = parts[0]
        for p in parts[1:]:
            clock = clock.append(p)
        assert len(clock) == 345

        df_aligned = df_day.reindex(clock)
        has_data = df_aligned["Close"].notna().astype(int)

        for col in ["Open", "High", "Low", "Close", "Volume", "OpenInterest"]:
            if col in df_aligned.columns:
                df_aligned[col] = pd.to_numeric(df_aligned[col], errors="coerce").fillna(0.0).astype(float)
            else:
                df_aligned[col] = 0.0

        minute_index = pd.Series(np.arange(len(clock), dtype=int), index=clock, name="minute_index")
        sid_full = pd.Series(sid_str, index=clock, name="session_id")

        out_frames.append(df_aligned)
        out_masks.append(has_data.rename("mask_t"))
        out_sid.append(sid_full)
        out_minidx.append(minute_index)

    aligned = pd.concat(out_frames).sort_index()
    mask = pd.concat(out_masks).astype(int)
    sid_all = pd.concat(out_sid).astype(str)
    minidx_all = pd.concat(out_minidx).astype(int)

    day_id_values, _ = pd.factorize(sid_all, sort=False)
    day_id = pd.Series(day_id_values.astype(np.int32), index=aligned.index, name="day_id")

    return {
        "aligned": aligned,
        "mask": mask,
        "session_id": sid_all,
        "minute_index": minidx_all,
        "day_id": day_id,
    }


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


def _rolling_percentile_causal_spec(values: np.ndarray, window: int = 240, min_periods: int = 30) -> np.ndarray:
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


def _rolling_slope_spec(values: np.ndarray, window: int) -> np.ndarray:
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


def _ema_next_spec(prev: Optional[float], value: float, span: int) -> float:
    if prev is None:
        return float(value)
    alpha = 2.0 / (float(span) + 1.0)
    return float(prev + alpha * (float(value) - prev))


def _dynamic_5m_macd_spec(close: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            fast_completed = _ema_next_spec(fast_completed, last_valid_close, 12)
            slow_completed = _ema_next_spec(slow_completed, last_valid_close, 26)
            completed_macd = float(fast_completed - slow_completed)
            signal_completed = _ema_next_spec(signal_completed, completed_macd, 9)
            completed_5m += 1

        c = float(close[i])
        fast_dyn = _ema_next_spec(fast_completed, c, 12)
        slow_dyn = _ema_next_spec(slow_completed, c, 26)
        macd_dyn = float(fast_dyn - slow_dyn)
        signal_dyn = _ema_next_spec(signal_completed, macd_dyn, 9)
        macd[i] = macd_dyn
        signal[i] = signal_dyn
        hist[i] = macd_dyn - signal_dyn
        ready[i] = 1.0 if completed_5m >= 35 else 0.0
        last_valid_close = c

    return macd, signal, hist, ready


def _add_obs_features_spec(df: pd.DataFrame) -> None:
    eps = 1e-12
    m = pd.to_numeric(df.get("mask_t", 1.0), errors="coerce").fillna(1.0).astype(float)
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

    rf = pd.to_numeric(df.get("range_frac_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
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

    od = pd.to_numeric(df.get("open_drift_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    df["obs_open_drift_t"] = np.clip(od, -1.0, 1.0)

    df["obs_cmp_C_vs_cumVWAP_t"] = pd.to_numeric(df.get("cmp_C_vs_cumVWAP_t", 0.0), errors="coerce").fillna(0.0).astype(float)
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
    df["volume_surprise_rolling_percentile"] = _rolling_percentile_causal_spec(volume_surprise, window=240, min_periods=30)
    df["volume_impulse_recent"] = (
        pd.Series(volume_surprise, index=df.index).rolling(5, min_periods=1).mean().fillna(0.0).to_numpy(dtype=float)
    )
    df["volume_impulse_slope_3"] = _rolling_slope_spec(volume_surprise, 3)
    df["volume_impulse_slope_5"] = _rolling_slope_spec(volume_surprise, 5)
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
    df["oi_rolling_percentile"] = _rolling_percentile_causal_spec(oi_z, window=240, min_periods=30)
    df["oi_delta_slope_3"] = _rolling_slope_spec(oi_z, 3)
    df["oi_delta_slope_5"] = _rolling_slope_spec(oi_z, 5)

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
        df[dst] = _rolling_percentile_causal_spec(
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
        df["obs_session_phase_t"] = (mi >= 120.0).astype(float)
    else:
        df["obs_session_phase_t"] = 0.0

    C_valid = C_s.where(m > 0.0, np.nan)
    prev = C_valid.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.log(C_valid / prev)
    logret = logret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raw_vol = logret.rolling(window=30, min_periods=5).std().fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    df["vol_rolling_percentile"] = _rolling_percentile_causal_spec(raw_vol, window=240, min_periods=30)

    prev_close = np.roll(C, 1)
    prev_close[0] = C[0] if len(C) else 0.0
    tr = np.maximum.reduce([
        np.maximum(H - L, 0.0),
        np.abs(H - prev_close),
        np.abs(L - prev_close),
    ])
    tr_valid = pd.Series(np.where(valid, tr, np.nan), index=df.index)
    atr30 = tr_valid.rolling(30, min_periods=30).mean().fillna(0.0).to_numpy(dtype=float)
    atr60 = tr_valid.rolling(60, min_periods=60).mean().fillna(0.0).to_numpy(dtype=float)
    valid_count = pd.Series(valid.astype(float), index=df.index)
    atr30_ready = valid_count.rolling(30, min_periods=1).sum().to_numpy(dtype=float) >= 30.0
    atr60_ready = valid_count.rolling(60, min_periods=1).sum().to_numpy(dtype=float) >= 60.0
    atr_ready = (valid & atr30_ready & atr60_ready).astype(float)

    atr30_frac = np.zeros_like(atr30, dtype=float)
    atr60_frac = np.zeros_like(atr60, dtype=float)
    price_ok = valid & (C > eps)
    atr30_frac[price_ok & atr30_ready] = atr30[price_ok & atr30_ready] / C[price_ok & atr30_ready]
    atr60_frac[price_ok & atr60_ready] = atr60[price_ok & atr60_ready] / C[price_ok & atr60_ready]
    df["atr_1m_30_price_frac"] = np.clip(atr30_frac, 0.0, 1.0) * mask_np
    df["atr_1m_60_price_frac"] = np.clip(atr60_frac, 0.0, 1.0) * mask_np
    df["atr_1m_30_rolling_percentile"] = _rolling_percentile_causal_spec(
        df["atr_1m_30_price_frac"].to_numpy(dtype=float),
        window=240,
        min_periods=30,
    ) * mask_np
    df["atr_1m_60_rolling_percentile"] = _rolling_percentile_causal_spec(
        df["atr_1m_60_price_frac"].to_numpy(dtype=float),
        window=240,
        min_periods=30,
    ) * mask_np
    range_over_atr = np.zeros_like(atr30, dtype=float)
    range_ok = valid & atr30_ready
    range_over_atr[range_ok] = np.maximum(H[range_ok] - L[range_ok], 0.0) / np.maximum(atr30[range_ok], eps)
    df["current_bar_range_atr_30"] = np.clip(range_over_atr, 0.0, 20.0) * mask_np
    df["intraday_volatility_percentile"] = df["vol_rolling_percentile"].to_numpy(dtype=float) * mask_np
    df["atr_ready_flag"] = atr_ready

    denom_macd = np.maximum(atr30, np.maximum(ref_safe * 1e-6, eps))
    macd, signal, hist, macd_ready = _dynamic_5m_macd_spec(C, valid)
    hist_delta = np.zeros_like(hist)
    hist_delta[1:] = hist[1:] - hist[:-1]
    hist_slope3 = _rolling_slope_spec(hist, 3)
    hist_slope5 = _rolling_slope_spec(hist, 5)
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

    for col in FEATURES_RISK_CONTEXT:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )


def build_market_features_spec(
    df_1m: pd.DataFrame,
    *,
    tz: str = FEATURE_TZ,
    df_prev_session: Optional[pd.DataFrame] = None,
    limit_up_pct: Optional[float] = None,
    limit_down_pct: Optional[float] = None,
) -> pd.DataFrame:
    base = strict_reindex_futures_345_spec(df_1m, tz=tz)
    X = base["aligned"].copy()
    mask_t = base["mask"].astype(int)
    session_id = base["session_id"].astype(str)
    minute_index = base["minute_index"].astype(int)
    day_id = base["day_id"].astype(np.int32)

    X["session_id"] = session_id
    X["trading_day"] = session_id.astype(int)

    X["C_t"] = pd.to_numeric(X["Close"], errors="coerce").fillna(0.0).astype(float)
    X["H_t"] = pd.to_numeric(X["High"], errors="coerce").fillna(0.0).astype(float)
    X["L_t"] = pd.to_numeric(X["Low"], errors="coerce").fillna(0.0).astype(float)
    X["V_t"] = pd.to_numeric(X.get("Volume", 0.0), errors="coerce").fillna(0.0).astype(float)
    X["I_t"] = pd.to_numeric(X.get("OpenInterest", 0.0), errors="coerce").fillna(0.0).astype(float)

    # ref_close
    valid_close = pd.to_numeric(X["Close"], errors="coerce").astype(float).where(mask_t == 1, np.nan)

    sid_order = pd.Index(pd.unique(session_id))
    first_valid_by_sid = valid_close.groupby(session_id).first().reindex(sid_order)
    last_valid_by_sid = valid_close.groupby(session_id).last().reindex(sid_order)

    prev_close_map = last_valid_by_sid.shift(1)
    ref_close_t = session_id.map(prev_close_map).astype(float)
    ref_close_t = ref_close_t.fillna(session_id.map(first_valid_by_sid).astype(float))

    if df_prev_session is not None and len(df_prev_session) > 0:
        prev = strict_reindex_futures_345_spec(df_prev_session, tz=tz)
        prev_aligned = prev["aligned"]
        prev_mask = prev["mask"].astype(int)
        prev_valid_close = pd.to_numeric(prev_aligned["Close"], errors="coerce").astype(float).where(prev_mask == 1, np.nan)
        prev_last_close = float(prev_valid_close.dropna().iloc[-1]) if prev_valid_close.notna().any() else float(
            pd.to_numeric(prev_aligned["Close"].tail(1), errors="coerce").fillna(0.0).iloc[0]
        )
        first_sid = session_id.iloc[0]
        ref_close_t.loc[session_id == first_sid] = prev_last_close

    X["ref_close_t"] = ref_close_t.ffill().bfill().fillna(0.0).astype(float)

    # cumVWAP
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

    # session high/low
    H_valid = pd.to_numeric(X["High"], errors="coerce").astype(float).where(mask_t == 1, np.nan)
    L_valid = pd.to_numeric(X["Low"], errors="coerce").astype(float).where(mask_t == 1, np.nan)
    X["session_high_t"] = H_valid.groupby(session_id).cummax().groupby(session_id).ffill().fillna(0.0).astype(float)
    X["session_low_t"] = L_valid.groupby(session_id).cummin().groupby(session_id).ffill().fillna(0.0).astype(float)

    # session open
    open_valid = pd.to_numeric(X.get("Open", np.nan), errors="coerce").astype(float).where(mask_t == 1, np.nan)
    first_open = open_valid.groupby(session_id).first()
    first_close = X["C_t"].where(mask_t == 1, np.nan).groupby(session_id).first()
    session_open = session_id.map(first_open).astype(float)
    session_open = session_open.fillna(session_id.map(first_close).astype(float)).fillna(0.0).astype(float)
    X["session_open_t"] = session_open

    # range_frac + open_drift
    eps = 1e-12
    ref_s = pd.to_numeric(X["ref_close_t"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    C_np = X["C_t"].astype(float).to_numpy()
    ref_safe = np.where(ref_s > eps, ref_s, np.where(C_np > eps, C_np, eps))
    rng = np.maximum(X["session_high_t"].astype(float).to_numpy() - X["session_low_t"].astype(float).to_numpy(), 0.0)

    vmask = (np.asarray(mask_t, dtype=int) == 1)
    range_frac = np.zeros_like(rng, dtype=float)
    range_frac[vmask] = rng[vmask] / np.maximum(ref_safe[vmask], eps)
    X["range_frac_t"] = range_frac.astype(float)

    so = pd.to_numeric(X["session_open_t"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    so_safe = np.where(so > eps, so, np.where(C_np > eps, C_np, eps))
    open_drift = np.zeros_like(C_np, dtype=float)
    open_drift[vmask] = np.log(np.where(C_np > eps, C_np, eps)[vmask] / so_safe[vmask])
    X["open_drift_t"] = open_drift.astype(float)

    # bar_dir
    prev_close = X["C_t"].shift(1)
    prev_mask = pd.Series(mask_t).shift(1).fillna(0).to_numpy()
    raw_dir = np.sign(X["C_t"] - prev_close).astype(float)
    bar_dir = np.where((mask_t == 1) & (prev_mask == 1), raw_dir, 0.0)
    bar_dir = np.where(minute_index.to_numpy() == 0, 0.0, bar_dir)
    X["bar_dir_t"] = bar_dir.astype(int)

    # turnover
    cv = (X["C_t"] * (X["V_t"] * mask_t)).astype(float)
    X["turnover_t"] = cv.groupby(session_id).cumsum().astype(float)

    # minute / weekday
    X["minute_index_t"] = minute_index.astype(int)
    wd = _weekday_cyc_from_sid(session_id)
    X["weekday_sin_t"] = wd["weekday_sin_t"].astype(float)
    X["weekday_cos_t"] = wd["weekday_cos_t"].astype(float)

    # limits
    if (limit_up_pct is not None) and (limit_down_pct is not None):
        up = float(limit_up_pct)
        down = float(limit_down_pct)
        X["limit_up_price_t"] = X["ref_close_t"] * (1.0 + up)
        X["limit_down_price_t"] = X["ref_close_t"] * (1.0 - down)
    else:
        X["limit_up_price_t"] = 0.0
        X["limit_down_price_t"] = 0.0

    # dI_from_yclose
    valid_oi = X["I_t"].where(mask_t == 1, np.nan).astype(float)
    first_oi_by_sid = valid_oi.groupby(session_id).first().reindex(sid_order)
    last_oi_by_sid = valid_oi.groupby(session_id).last().reindex(sid_order)

    prev_oi_map = last_oi_by_sid.shift(1)
    I_yclose = session_id.map(prev_oi_map).astype(float)
    I_yclose = I_yclose.fillna(session_id.map(first_oi_by_sid).astype(float))

    if df_prev_session is not None and len(df_prev_session) > 0:
        prev = strict_reindex_futures_345_spec(df_prev_session, tz=tz)
        prev_aligned = prev["aligned"]
        prev_mask = prev["mask"].astype(int)
        prev_valid_oi = pd.to_numeric(prev_aligned.get("OpenInterest", 0.0), errors="coerce").astype(float)
        prev_valid_oi = pd.Series(prev_valid_oi).where(prev_mask == 1, np.nan)
        prev_last_oi = float(prev_valid_oi.dropna().iloc[-1]) if prev_valid_oi.notna().any() else float(
            pd.to_numeric(prev_aligned.get("OpenInterest", 0.0), errors="coerce").fillna(0.0).iloc[-1]
        )
        first_sid = session_id.iloc[0]
        I_yclose.loc[session_id == first_sid] = prev_last_oi

    X["dI_from_yclose_t"] = (X["I_t"].astype(float) - I_yclose).fillna(0.0).astype(float)

    # dP / pct
    X["dP_from_ref_t"] = (X["C_t"] - X["ref_close_t"]).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = X["C_t"] / X["ref_close_t"] - 1.0
    X["pct_chg_from_ref_t"] = pd.Series(pct).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    # mask/day_id
    X["mask_t"] = mask_t.astype(float)
    X["day_id"] = day_id.astype(np.int32)

    # RAW mask-zeroing
    m = X["mask_t"].astype(float)
    for col in FEATURES_MARKET + AUX_MARKET_COLS:
        if col == "mask_t":
            continue
        if col in ("minute_index_t", "weekday_sin_t", "weekday_cos_t"):
            continue
        if col in X.columns:
            X[col] = (X[col].astype(float) * m).astype(float)

    # clean raw
    for col in FEATURES_MARKET:
        if col not in X.columns:
            X[col] = 0.0
        X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    # obs
    _add_obs_features_spec(X)
    for col in FEATURES_MARKET_OBS:
        if col not in X.columns:
            X[col] = 0.0
        X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    for col in FEATURES_RISK_CONTEXT:
        if col not in X.columns:
            X[col] = 0.0
        X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    # 需要的辅助列
    X["H_t"] = pd.to_numeric(X.get("H_t", 0.0), errors="coerce").fillna(0.0).astype(float)
    X["L_t"] = pd.to_numeric(X.get("L_t", 0.0), errors="coerce").fillna(0.0).astype(float)

    return X[REQUIRED_MARKET_COLS].copy()


def window_slice_spec(df_market_full: pd.DataFrame, *, row_i: int, window_size: int) -> np.ndarray:
    """
    Spec 窗口规则（建议冻结为“左侧 0 padding，右侧包含当前行”）：
      - 取 [row_i-window_size+1, row_i]（含 row_i）
      - 不足左侧用全 0 行补齐
    返回 float32 [window_size, F]
    """
    F = len(FEATURES_MARKET)  # 这里窗口用于 raw/obs 时，测试侧会改成对应 list
    raise RuntimeError("Use window_slice_spec_by_cols(...)")


def window_slice_spec_by_cols(df_market_full: pd.DataFrame, *, row_i: int, window_size: int, cols: List[str]) -> np.ndarray:
    F = len(cols)
    start = row_i - window_size + 1
    if start >= 0:
        w = df_market_full.iloc[start : row_i + 1][cols].to_numpy(dtype=np.float32, copy=False)
        return w
    # pad
    pad = np.zeros((abs(start), F), dtype=np.float32)
    w = df_market_full.iloc[0 : row_i + 1][cols].to_numpy(dtype=np.float32, copy=False)
    return np.concatenate([pad, w], axis=0)
