# tests/oracles/market_spec_oracle.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd


FEATURE_TZ = "Asia/Shanghai"

# ===== 列清单（Spec 冻结） =====
FEATURES_MARKET_OBS: List[str] = [
    "obs_V_t",
    "obs_I_t",
    "obs_cumVWAP_t",
    "obs_dC_minus_cumVWAP_t",
    "obs_cmp_C_vs_cumVWAP_t",
    "obs_session_high_t",
    "obs_session_low_t",
    "obs_range_t",
    "obs_open_drift_t",
    "obs_bar_dir_t",
    "obs_minute_index_t",
    "obs_session_phase_t",
    "obs_volatility_t",
    "obs_dI_from_yclose_t",
    "obs_pct_chg_from_ref_t",
    "obs_mask_t",
    "obs_weekday_sin_t",
    "obs_weekday_cos_t",
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
REQUIRED_MARKET_COLS = ["day_id", "session_id", "trading_day"] + AUX_MARKET_COLS + FEATURES_MARKET + FEATURES_MARKET_OBS

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


def _add_obs_features_spec(df: pd.DataFrame) -> None:
    eps = 1e-12
    m = pd.to_numeric(df.get("mask_t", 1.0), errors="coerce").fillna(1.0).astype(float)
    valid = (m.to_numpy(dtype=float, copy=False) > 0.0)

    C = pd.to_numeric(df.get("C_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    ref = pd.to_numeric(df.get("ref_close_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    ref_safe = np.where(ref > eps, ref, np.where(C > eps, C, eps))

    LOG_CLIP = 1.0

    def _log_ratio(x: np.ndarray) -> np.ndarray:
        x_safe = np.where(x > eps, x, eps)
        out = np.zeros_like(x_safe, dtype=float)
        out[valid] = np.log(x_safe[valid] / ref_safe[valid])
        return np.clip(out, -LOG_CLIP, LOG_CLIP)

    def _signed_log1p_ratio(z: np.ndarray) -> np.ndarray:
        out = np.zeros_like(z, dtype=float)
        ratio = np.abs(z) / np.maximum(ref_safe, eps)
        out[valid] = np.sign(z[valid]) * np.log1p(ratio[valid])
        return np.clip(out, -LOG_CLIP, LOG_CLIP)

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
    df["obs_range_t"] = np.clip(obs_range, 0.0, LOG_CLIP)

    dC = pd.to_numeric(df.get("dC_minus_cumVWAP_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    df["obs_dC_minus_cumVWAP_t"] = _signed_log1p_ratio(dC)

    I = pd.to_numeric(df.get("I_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    dI = pd.to_numeric(df.get("dI_from_yclose_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    I_yclose = I - dI
    denom = np.maximum(np.abs(I_yclose), eps)
    out_dI = np.zeros_like(dI, dtype=float)
    ratio_I = np.abs(dI) / denom
    out_dI[valid] = np.sign(dI[valid]) * np.log1p(ratio_I[valid])
    df["obs_dI_from_yclose_t"] = np.clip(out_dI, -LOG_CLIP, LOG_CLIP)

    od = pd.to_numeric(df.get("open_drift_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    df["obs_open_drift_t"] = np.clip(od, -LOG_CLIP, LOG_CLIP)

    df["obs_cmp_C_vs_cumVWAP_t"] = pd.to_numeric(df.get("cmp_C_vs_cumVWAP_t", 0.0), errors="coerce").fillna(0.0).astype(float)
    df["obs_bar_dir_t"] = pd.to_numeric(df.get("bar_dir_t", 0.0), errors="coerce").fillna(0.0).astype(float)

    V_s = pd.to_numeric(df.get("V_t", 0.0), errors="coerce").fillna(0.0).astype(float)
    V_valid = V_s.where(m > 0.0, np.nan)
    sid = df.get("session_id", pd.Series(0, index=df.index)).astype(str)
    ema = V_valid.groupby(sid).transform(
        lambda x: x.ewm(span=30, adjust=False, min_periods=5, ignore_na=True).mean()
    )
    ema_prev = ema.groupby(sid).shift(1)
    exp_mean = V_valid.groupby(sid).transform(lambda x: x.expanding(min_periods=1).mean())
    base = ema_prev.fillna(exp_mean.groupby(sid).shift(1)).fillna(0.0)

    ratio = V_s.to_numpy(dtype=float, copy=False) / (base.to_numpy(dtype=float, copy=False) + eps)
    spike = np.log1p(np.clip(ratio, 0.0, None))
    df["obs_V_t"] = (np.clip(spike, 0.0, 3.0) * m.to_numpy(dtype=float, copy=False)).astype(float)

    # keep obs_I_t as before
    oi = np.log1p(np.clip(I, 0.0, None))
    df["obs_I_t"] = np.tanh(oi / 5.0).astype(float)


    df["obs_mask_t"] = m.astype(float)
    df["obs_weekday_sin_t"] = pd.to_numeric(df.get("weekday_sin_t", 0.0), errors="coerce").fillna(0.0).astype(float)
    df["obs_weekday_cos_t"] = pd.to_numeric(df.get("weekday_cos_t", 0.0), errors="coerce").fillna(0.0).astype(float)

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

    # volatility
    VOL_WIN = 30
    VOL_MINP = 5
    VOL_SCALE = 0.01

    C_s = pd.to_numeric(df.get("C_t", 0.0), errors="coerce").astype(float)
    C_valid = C_s.where(m > 0.0, np.nan)

    sid = df.get("session_id", pd.Series("0", index=df.index))
    prev = C_valid.groupby(sid).shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.log(C_valid / prev)
    logret = logret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol = (
        logret.groupby(sid)
        .rolling(window=VOL_WIN, min_periods=VOL_MINP)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    vol = pd.to_numeric(vol, errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)
    df["obs_volatility_t"] = np.tanh(vol / max(VOL_SCALE, 1e-12)).astype(float)

    # apply mask except NO_MASK
    NO_MASK = {
        "obs_mask_t",
        "obs_minute_index_t",
        "obs_weekday_sin_t",
        "obs_weekday_cos_t",
        "obs_session_phase_t",
    }
    for col in FEATURES_MARKET_OBS:
        if col in NO_MASK:
            continue
        if col in df.columns:
            df[col] = (pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float) * m).astype(float)


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
