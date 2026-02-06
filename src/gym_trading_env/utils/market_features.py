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
    "obs_V_t",
    "obs_I_t",
    "obs_cumVWAP_t",
    "obs_dC_minus_cumVWAP_t",
    "obs_cmp_C_vs_cumVWAP_t",
    "obs_session_high_t",
    "obs_session_low_t",
    "obs_range_t",             # NEW: log1p((H-L)/ref_close), clipped, masked
    "obs_open_drift_t",        # NEW: log(C/session_open), clipped, masked (direction kept)
    "obs_bar_dir_t",
    "obs_minute_index_t",
    "obs_session_phase_t",     # NEW: 0/1 phase indicator (futures: night/day)
    "obs_volatility_t",        # NEW: rolling volatility proxy (bounded)
    "obs_dI_from_yclose_t",    # NEW stable scaling (ratio to I_yclose)
    "obs_pct_chg_from_ref_t",  # NEW: log-ratio(C/ref_close), clipped (keep old name)
    "obs_mask_t",
    "obs_weekday_sin_t",
    "obs_weekday_cos_t",
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


def _add_obs_features_inplace(df: pd.DataFrame) -> None:
    """
    基于 raw FEATURES_MARKET 生成 obs_ 归一化特征列（写回 df）。

    v2 口径：
      - 价格类：log-ratio: clip(log(x / ref_close), [-LOG_CLIP, LOG_CLIP])
      - 差值类：signed-log: clip(sign(z)*log1p(|z|/ref_close), [-LOG_CLIP, LOG_CLIP])
      - 量：obs_V_t = clip((V/I*100), [0,2])；持仓：log1p 后 tanh squash
      - dI：sign(dI)*log1p(|dI|/max(|I_yclose|, eps))（更稳，不依赖合约量级）
      - obs_pct_chg_from_ref_t：使用 log(C/ref_close)（保留旧列名）
      - NEW obs_range_t：clip(log1p((H-L)/ref_close), [0, LOG_CLIP])
      - NEW obs_open_drift_t：clip(log(C/session_open), [-LOG_CLIP, LOG_CLIP])
      - time/weekday/phase：保留（不乘 mask）
      - 其余 obs 特征：乘 mask（无效分钟置 0）
    """
    eps = 1e-12

    # --- mask ---
    m = df.get("mask_t", pd.Series(1.0, index=df.index)).astype(float)
    valid = (m.to_numpy(dtype=float, copy=False) > 0.0)

    # --- ref_close safe ---
    C = pd.to_numeric(df.get("C_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    ref = pd.to_numeric(df.get("ref_close_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    ref_safe = np.where(ref > eps, ref, np.where(C > eps, C, eps))

    # tight clip to keep obs stable
    _LOG_CLIP = 1.0

    def _log_ratio(x: np.ndarray) -> np.ndarray:
        x_safe = np.where(x > eps, x, eps)
        out = np.zeros_like(x_safe, dtype=float)
        out[valid] = np.log(x_safe[valid] / ref_safe[valid])
        return np.clip(out, -_LOG_CLIP, _LOG_CLIP)

    def _signed_log1p_ratio(z: np.ndarray) -> np.ndarray:
        out = np.zeros_like(z, dtype=float)
        ratio = np.abs(z) / np.maximum(ref_safe, eps)
        out[valid] = np.sign(z[valid]) * np.log1p(ratio[valid])
        return np.clip(out, -_LOG_CLIP, _LOG_CLIP)

    # -----------------------------
    # price-like (log-ratio)
    # -----------------------------
    cumVWAP = pd.to_numeric(df.get("cumVWAP_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    df["obs_cumVWAP_t"] = _log_ratio(cumVWAP)

    sh = pd.to_numeric(df.get("session_high_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    sl = pd.to_numeric(df.get("session_low_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    df["obs_session_high_t"] = _log_ratio(sh)
    df["obs_session_low_t"] = _log_ratio(sl)

    # pct_chg_from_ref -> log(C/ref_close), keep old name
    df["obs_pct_chg_from_ref_t"] = _log_ratio(C)

    # -----------------------------
    # NEW: intraday range (non-negative)
    # range_frac_t raw = (H-L)/ref_close
    # obs_range_t = clip(log1p(range_frac), [0, LOG_CLIP])
    # -----------------------------
    if "range_frac_t" in df.columns:
        rf = pd.to_numeric(df["range_frac_t"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    else:
        rf = np.maximum(sh - sl, 0.0) / np.maximum(ref_safe, eps)

    obs_range = np.zeros_like(rf, dtype=float)
    rf_pos = np.clip(rf, 0.0, None)
    obs_range[valid] = np.log1p(rf_pos[valid])
    df["obs_range_t"] = np.clip(obs_range, 0.0, _LOG_CLIP)

    # -----------------------------
    # delta-like (signed log)
    # -----------------------------
    dC = pd.to_numeric(df.get("dC_minus_cumVWAP_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    df["obs_dC_minus_cumVWAP_t"] = _signed_log1p_ratio(dC)

    # -----------------------------
    # dI stable scaling: sign(dI)*log1p(|dI|/max(|I_yclose|, eps))
    # where I_yclose = I_t - dI_from_yclose_t
    # -----------------------------
    I = pd.to_numeric(df.get("I_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    dI = pd.to_numeric(df.get("dI_from_yclose_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    I_yclose = I - dI
    denom = np.maximum(np.abs(I_yclose), eps)

    out_dI = np.zeros_like(dI, dtype=float)
    ratio_I = np.abs(dI) / denom
    out_dI[valid] = np.sign(dI[valid]) * np.log1p(ratio_I[valid])
    df["obs_dI_from_yclose_t"] = np.clip(out_dI, -_LOG_CLIP, _LOG_CLIP)

    # -----------------------------
    # NEW: open -> current drift (direction kept)
    # open_drift_t raw = log(C/session_open)
    # obs_open_drift_t = clip(open_drift_t, [-LOG_CLIP, LOG_CLIP])
    # -----------------------------
    if "open_drift_t" in df.columns:
        od = pd.to_numeric(df["open_drift_t"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    else:
        so = pd.to_numeric(df.get("session_open_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
        so_safe = np.where(so > eps, so, np.where(C > eps, C, eps))
        od = np.zeros_like(C, dtype=float)
        od[valid] = np.log(np.where(C > eps, C, eps)[valid] / so_safe[valid])

    df["obs_open_drift_t"] = np.clip(od, -_LOG_CLIP, _LOG_CLIP)

    # -----------------------------
    # discrete
    # -----------------------------
    df["obs_cmp_C_vs_cumVWAP_t"] = (
        pd.to_numeric(df.get("cmp_C_vs_cumVWAP_t", 0.0), errors="coerce").fillna(0.0).astype(float)
    )
    df["obs_bar_dir_t"] = pd.to_numeric(df.get("bar_dir_t", 0.0), errors="coerce").fillna(0.0).astype(float)

    # -----------------------------
    # volume / open interest -> bounded (LN(1 + V/OI*100) clipped to [0, 1])
    # -----------------------------
    V = pd.to_numeric(df.get("V_t", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()

    # -----------------------------
    # volume surprise (keep spikes, reduce fingerprint)
    # obs_V_t = clip(log1p(V / EMA_prev + eps), [0, VOL_CLIP]) * mask
    # -----------------------------
    V_s = pd.to_numeric(df.get("V_t", 0.0), errors="coerce").fillna(0.0).astype(float)
    V_valid = V_s.where(m > 0.0, np.nan)

    # EMA within session (use only past -> shift(1))
    # span=30 means ~30 minutes smoothing; tweak 20/30/60 as you like
    sid = df.get("session_id", pd.Series(0, index=df.index)).astype(str)
    ema = (
        V_valid.groupby(sid)
        .apply(lambda x: x.ewm(span=30, adjust=False, min_periods=5, ignore_na=True).mean())
        .reset_index(level=0, drop=True)
    )
    ema_prev = ema.groupby(sid).shift(1)

    # fallback baseline for early minutes: expanding mean of past valid bars
    exp_mean = (
        V_valid.groupby(sid)
        .expanding(min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    base = ema_prev.fillna(exp_mean.groupby(sid).shift(1)).fillna(0.0)

    eps_v = 1e-12
    VOL_CLIP = 3.0  # 3 对应 e^(3)-1≈19x 的“放量倍率”，已经很夸张了
    ratio = (V_s.to_numpy(dtype=float, copy=False) / (base.to_numpy(dtype=float, copy=False) + eps_v))
    spike = np.log1p(np.clip(ratio, 0.0, None))

    df["obs_V_t"] = (np.clip(spike, 0.0, VOL_CLIP) * m.to_numpy(dtype=float, copy=False)).astype(float)


    # keep obs_I_t as before (bounded, scale-free-ish)
    oi = np.log1p(np.clip(I, 0.0, None))
    df["obs_I_t"] = np.tanh(oi / 5.0).astype(float)


    # -----------------------------
    # time / weekday
    # -----------------------------
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

    # -----------------------------
    # NEW: session_phase
    # futures 345: night=0..119, day=120..344
    # for non-futures / longer sessions: set 0 (avoid injecting fake structure)
    # -----------------------------
    mi_np = mi.to_numpy(dtype=float, copy=False)
    mi_max = float(np.nanmax(mi_np)) if len(mi_np) else 0.0
    if mi_max <= 400.0:
        df["obs_session_phase_t"] = (mi >= 120.0).astype(float)  # 0 night, 1 day
    else:
        df["obs_session_phase_t"] = 0.0

    # -----------------------------
    # NEW: volatility proxy (rolling std of log returns, bounded)
    # -----------------------------
    VOL_WIN = 30          # 30 minutes
    VOL_MINP = 5
    VOL_SCALE = 0.01      # tanh(std / scale) -> [0,1); adjust if needed

    C_s = pd.to_numeric(df.get("C_t", 0.0), errors="coerce").astype(float)
    C_valid = C_s.where(m > 0.0, np.nan)

    if "session_id" in df.columns:
        sid = df["session_id"]
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
    else:
        prev = C_valid.shift(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            logret = np.log(C_valid / prev)
        logret = logret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        vol = logret.rolling(window=VOL_WIN, min_periods=VOL_MINP).std().fillna(0.0)

    vol = pd.to_numeric(vol, errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)
    df["obs_volatility_t"] = np.tanh(vol / max(VOL_SCALE, 1e-12)).astype(float)

    # -----------------------------
    # apply mask to obs features except:
    #   - obs_mask_t
    #   - clock-like: minute_index/weekday/session_phase
    # -----------------------------
    _NO_MASK = {
        "obs_mask_t",
        "obs_minute_index_t",
        "obs_weekday_sin_t",
        "obs_weekday_cos_t",
        "obs_session_phase_t",
    }
    for col in FEATURES_MARKET_OBS:
        if col in _NO_MASK:
            continue
        if col in df.columns:
            df[col] = (pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float) * m).astype(float)


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
