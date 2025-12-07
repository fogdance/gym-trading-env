# src/gym_trading_env/utils/daily_features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, List, Optional

import numpy as np
import pandas as pd


DAILY_SEQ_LEN = 7

# === daily_context (RAW) ===
FEATURES_DAILY_CONTEXT: List[str] = [
    "prev_high",
    "prev_low",
    "prev_close",
    "prev_close_pdiff_1d",
    "prev_volume",
    "prev_volume_pdiff_1d",
    "prev_oi",
    "prev_oi_pdiff_1d",
]

# === daily_context (OBS) ===
FEATURES_DAILY_CONTEXT_OBS: List[str] = [
    "obs_prev_high",
    "obs_prev_low",
    "obs_prev_close",            # debug 对齐：恒 0
    "obs_prev_close_pdiff_1d",   # log(prev_close / prev2_close)
    "obs_prev_volume",           # log1p(prev_volume)
    "obs_prev_volume_pdiff_1d",  # signed log1p diff
    "obs_prev_oi",               # log1p(prev_oi)
    "obs_prev_oi_pdiff_1d",      # signed log1p diff
]

# daily_seq_7: 7 x 3
FEATURES_DAILY_SEQ_7: List[str] = ["close", "volume", "oi"]
FEATURES_DAILY_SEQ_7_OBS: List[str] = ["obs_close", "obs_volume", "obs_oi"]


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _safe_log_ratio(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a2 = np.maximum(a, eps)
    b2 = np.maximum(b, eps)
    return np.log(a2 / b2)


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def build_daily_context_and_seq(
    df_market: pd.DataFrame,
    *,
    day_key_fn: Callable[[object], str],
    days_order: Optional[np.ndarray] = None,
    day_id_col: str = "day_id",
    mask_col: str = "mask_t",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    只用过去日线信息（shift），不偷看未来。
    返回：
      ctx_raw:  (num_days, F_CTX)
      ctx_obs:  (num_days, F_CTX)
      seq_raw:  (num_days, 7, 3)  # lag1..lag7
      seq_obs:  (num_days, 7, 3)
      daily_summary: (num_days, columns=[close, high, low, volume, oi]) 方便 debug
    """
    if day_id_col not in df_market.columns:
        raise ValueError(f"df_market missing {day_id_col}")
    if mask_col not in df_market.columns:
        raise ValueError(f"df_market missing {mask_col}")

    df = df_market.copy()

    # day_key 做成与 env 一致的字符串形式（防止 20250101 vs 20250101.0）
    day_key = df[day_id_col].map(day_key_fn)
    df["_day_key"] = day_key

    m = _to_float(df[mask_col]).fillna(0.0).to_numpy() >= 0.5

    # 需要的列：优先 H_t/L_t，否则用 C_t 兜底
    C = _to_float(df.get("C_t", df.get("Close", 0.0)))
    H = _to_float(df.get("H_t", df.get("High", C)))
    L = _to_float(df.get("L_t", df.get("Low", C)))
    V = _to_float(df.get("V_t", df.get("Volume", 0.0))).clip(lower=0.0)
    I = _to_float(df.get("I_t", df.get("OpenInterest", 0.0))).clip(lower=0.0)

    df["_C"] = C
    df["_H"] = H
    df["_L"] = L
    df["_V"] = V
    df["_I"] = I
    df["_m"] = m.astype(np.int8)

    # 只用 valid 分钟来做日线统计（尤其 futures 夜盘补齐分钟）
    dv = df[df["_m"] == 1]

    # daily close / oi close：按时间顺序 last valid
    g = dv.groupby("_day_key", sort=False)
    day_close = g["_C"].last()
    day_oi = g["_I"].last()

    # daily high/low：valid 内 max/min
    day_high = g["_H"].max()
    day_low = g["_L"].min()

    # daily volume：valid 内 sum
    day_vol = g["_V"].sum()

    summary = pd.DataFrame({
        "close": day_close,
        "high": day_high,
        "low": day_low,
        "volume": day_vol,
        "oi": day_oi,
    }).astype(float)

    # 对齐 days_order（用 env 的 self._days 顺序最稳）
    if days_order is None:
        days_order = summary.index.to_numpy()
    else:
        days_order = np.asarray(days_order, dtype=object)

    summary = summary.reindex(days_order)

    # shift 得到 prev / prev2
    prev = summary.shift(1)
    prev2 = summary.shift(2)

    # ===== daily_context RAW =====
    ctx = pd.DataFrame(index=summary.index)
    ctx["prev_high"] = prev["high"]
    ctx["prev_low"] = prev["low"]
    ctx["prev_close"] = prev["close"]
    ctx["prev_close_pdiff_1d"] = prev["close"] - prev2["close"]

    ctx["prev_volume"] = prev["volume"]
    ctx["prev_volume_pdiff_1d"] = prev["volume"] - prev2["volume"]

    ctx["prev_oi"] = prev["oi"]
    ctx["prev_oi_pdiff_1d"] = prev["oi"] - prev2["oi"]

    ctx_raw = ctx[FEATURES_DAILY_CONTEXT].to_numpy(dtype=np.float32)
    ctx_raw = np.nan_to_num(ctx_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # ===== daily_context OBS（方式A）=====
    eps = 1e-12
    prev_close_np = prev["close"].to_numpy(dtype=float)
    prev2_close_np = prev2["close"].to_numpy(dtype=float)

    prev_high_np = prev["high"].to_numpy(dtype=float)
    prev_low_np = prev["low"].to_numpy(dtype=float)

    prev_vol_np = prev["volume"].to_numpy(dtype=float)
    prev2_vol_np = prev2["volume"].to_numpy(dtype=float)

    prev_oi_np = prev["oi"].to_numpy(dtype=float)
    prev2_oi_np = prev2["oi"].to_numpy(dtype=float)

    obs_ctx = np.zeros((len(summary), len(FEATURES_DAILY_CONTEXT_OBS)), dtype=np.float32)

    # price-like => log ratio to prev_close
    obs_ctx[:, 0] = _safe_log_ratio(prev_high_np, prev_close_np, eps=eps).astype(np.float32)  # obs_prev_high
    obs_ctx[:, 1] = _safe_log_ratio(prev_low_np, prev_close_np, eps=eps).astype(np.float32)   # obs_prev_low
    obs_ctx[:, 2] = 0.0                                                                      # obs_prev_close

    # log(prev_close / prev2_close)
    obs_ctx[:, 3] = _safe_log_ratio(prev_close_np, prev2_close_np, eps=eps).astype(np.float32)

    # volume / oi
    obs_ctx[:, 4] = np.log1p(np.clip(prev_vol_np, 0.0, None)).astype(np.float32)
    obs_ctx[:, 5] = _signed_log1p(prev_vol_np - prev2_vol_np).astype(np.float32)

    obs_ctx[:, 6] = np.log1p(np.clip(prev_oi_np, 0.0, None)).astype(np.float32)
    obs_ctx[:, 7] = _signed_log1p(prev_oi_np - prev2_oi_np).astype(np.float32)

    obs_ctx = np.nan_to_num(obs_ctx, nan=0.0, posinf=0.0, neginf=0.0)

    # ===== daily_seq_7 RAW/OBS =====
    n = len(summary)
    seq_raw = np.zeros((n, DAILY_SEQ_LEN, 3), dtype=np.float32)
    seq_obs = np.zeros((n, DAILY_SEQ_LEN, 3), dtype=np.float32)

    base_close = summary["close"].shift(1).to_numpy(dtype=float)  # prev_close（当前日的基准）
    base_vol = summary["volume"].shift(1).to_numpy(dtype=float)
    base_oi = summary["oi"].shift(1).to_numpy(dtype=float)

    base_logv = np.log1p(np.clip(base_vol, 0.0, None))
    base_logi = np.log1p(np.clip(base_oi, 0.0, None))

    for lag in range(1, DAILY_SEQ_LEN + 1):
        c_lag = summary["close"].shift(lag).to_numpy(dtype=float)
        v_lag = summary["volume"].shift(lag).to_numpy(dtype=float)
        i_lag = summary["oi"].shift(lag).to_numpy(dtype=float)

        # RAW：直接值
        seq_raw[:, lag - 1, 0] = np.nan_to_num(c_lag, nan=0.0).astype(np.float32)
        seq_raw[:, lag - 1, 1] = np.nan_to_num(v_lag, nan=0.0).astype(np.float32)
        seq_raw[:, lag - 1, 2] = np.nan_to_num(i_lag, nan=0.0).astype(np.float32)

        # OBS：
        # close => log(c_lag / prev_close)（lag1=0）
        seq_obs[:, lag - 1, 0] = _safe_log_ratio(c_lag, base_close, eps=eps).astype(np.float32)

        # volume/oi => log1p(x_lag) - log1p(prev_x)（lag1=0）
        logv = np.log1p(np.clip(v_lag, 0.0, None))
        logi = np.log1p(np.clip(i_lag, 0.0, None))

        seq_obs[:, lag - 1, 1] = (logv - base_logv).astype(np.float32)
        seq_obs[:, lag - 1, 2] = (logi - base_logi).astype(np.float32)

    seq_obs = np.nan_to_num(seq_obs, nan=0.0, posinf=0.0, neginf=0.0)

    return ctx_raw, obs_ctx, seq_raw, seq_obs, summary
