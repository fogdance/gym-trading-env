#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from bisect import bisect_right, insort
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from gym_trading_env.envs.trading_env import CustomTradingEnv
from gym_trading_env.utils.agent_features import FEATURES_AGENT_OBS
from gym_trading_env.utils.observation_schema import (
    HTF_CONTEXT_FEATURES,
    MARKET_V2_CONTINUOUS_FEATURES,
    RISK_CONTEXT_FEATURES,
)


DEFAULT_CONTRACT_ROOT = Path("/data/logdir/trading_contracts/jm_walk_forward_20240603_20251202")
DEFAULT_ENTRY_EVAL_DIR = DEFAULT_CONTRACT_ROOT / "artifacts/entry_eval/jm_walk_forward_20240603_20251202"
DEFAULT_SPLIT_MANIFEST = DEFAULT_CONTRACT_ROOT / "artifacts/walk_forward_splits/jm_walk_forward_20240603_20251202/split_manifest.json"
DEFAULT_ENV_CONFIG = Path("artifacts/observation_audit/v2/env_obs_v2_audit.yaml")
DEFAULT_FIT_PARAMS = Path("artifacts/observation_audit/v2/sidecar_fit_params.json")
DEFAULT_DREAMER_AUDIT_ROOT = Path("/data/logdir/audits/action-mask-jm-futures-reward-ms-20260618")
DEFAULT_OUT = Path("artifacts/obs_v2_formula_reference_audit/v1")
DEFAULT_DOC = Path("docs/obs_v2_formula_reference_audit_v1.md")


MARKET_GROUPS = {
    "volume": {
        "volume_surprise_logratio_floor",
        "volume_surprise_rolling_percentile",
        "volume_impulse_recent",
        "volume_impulse_slope_3",
        "volume_impulse_slope_5",
        "volume_impulse_ready_flag",
    },
    "oi": {
        "oi_rel_yclose_log",
        "oi_rel_session_open_log",
        "oi_delta_rolling_z",
        "oi_rolling_percentile",
        "oi_delta_slope_3",
        "oi_delta_slope_5",
        "oi_price_confirm_short_continuous",
        "oi_price_confirm_long_continuous",
        "oi_impulse_ready_flag",
    },
    "price_local": {
        "obs_cumVWAP_t",
        "obs_cumVWAP_t_train_percentile",
        "obs_dC_minus_cumVWAP_t",
        "obs_cmp_C_vs_cumVWAP_t",
        "obs_session_high_t",
        "obs_session_high_t_train_percentile",
        "obs_session_low_t",
        "obs_range_t",
        "obs_range_t_train_percentile",
        "obs_open_drift_t",
        "obs_bar_dir_t",
        "obs_minute_index_t",
        "obs_session_phase_t",
        "vol_rolling_percentile",
        "obs_dI_from_yclose_t",
        "obs_pct_chg_from_ref_t",
        "obs_pct_chg_from_ref_t_train_percentile",
        "obs_mask_t",
    },
    "macd": {name for name in MARKET_V2_CONTINUOUS_FEATURES if name.startswith("dyn5m_macd_")},
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _to_float(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    return (
        pd.to_numeric(df[col] if col in df.columns else default, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
        .astype(float)
        .to_numpy()
    )


def _to_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return pd.Series(_to_float(df, col, default), index=df.index, dtype=float)


def _load_split(path: Path) -> tuple[dict[int, str], dict[str, list[int]], set[int], str]:
    data = _read_json(path)
    fold = data[-1] if isinstance(data, list) else data
    roles: dict[int, str] = {}
    by_role: dict[str, list[int]] = {}
    for role in ("train", "validation", "test"):
        days = [int(x) for x in fold.get(f"{role}_days", [])]
        by_role[role] = days
        for day in days:
            roles[day] = role
    return roles, by_role, set(by_role.get("train", [])), _sha256_file(path)


def _quantile_grid(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [0.0]
    return [float(x) for x in np.quantile(values, np.linspace(0.0, 1.0, 101))]


def _pct_grid(values: np.ndarray, grid: list[float]) -> np.ndarray:
    grid_arr = np.asarray(grid, dtype=float)
    if grid_arr.size == 0:
        return np.zeros_like(values, dtype=float)
    pct = np.searchsorted(grid_arr, values, side="right") / float(grid_arr.size)
    return np.clip(2.0 * pct - 1.0, -1.0, 1.0)


def _rolling_percentile_reference(values: np.ndarray, valid: np.ndarray, *, window: int, min_periods: int) -> np.ndarray:
    # Independent implementation: uses a sorted sliding row window with valid items only.
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    out = np.zeros_like(values, dtype=float)
    ordered: list[float] = []
    q: deque[float | None] = deque()
    for i, value in enumerate(values):
        item: float | None = None
        if valid[i] and np.isfinite(value):
            item = float(value)
            insort(ordered, item)
        q.append(item)
        if len(q) > int(window):
            old = q.popleft()
            if old is not None:
                pos = bisect_right(ordered, old) - 1
                if pos >= 0:
                    ordered.pop(pos)
        if item is None or len(ordered) < int(min_periods):
            continue
        rank = bisect_right(ordered, item) / float(len(ordered))
        out[i] = 2.0 * rank - 1.0
    return np.clip(out, -1.0, 1.0)


def _slope_reference(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.zeros_like(values, dtype=float)
    x = np.arange(window, dtype=float)
    x -= x.mean()
    denom = float(np.sum(x * x)) or 1.0
    for i in range(window - 1, len(values)):
        y = values[i - window + 1:i + 1].astype(float)
        if np.any(~np.isfinite(y)):
            continue
        out[i] = float(np.sum((y - y.mean()) * x) / denom)
    return out


def _rolling_mean_valid_reference(values: np.ndarray, valid: np.ndarray, window: int, min_periods: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    out = np.zeros_like(values, dtype=float)
    counts = np.zeros_like(values, dtype=float)
    q: deque[float | None] = deque()
    running = 0.0
    valid_count = 0
    for i, value in enumerate(values):
        item: float | None = None
        if valid[i] and np.isfinite(value):
            item = float(value)
            running += item
            valid_count += 1
        q.append(item)
        if len(q) > int(window):
            old = q.popleft()
            if old is not None:
                running -= old
                valid_count -= 1
        counts[i] = valid_count
        if valid_count >= int(min_periods):
            out[i] = running / float(valid_count)
    return out, counts


def _signed_log1p_relative(delta: np.ndarray, base: np.ndarray, valid: np.ndarray) -> np.ndarray:
    eps = 1e-12
    delta = np.nan_to_num(delta.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    base = np.nan_to_num(base.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros_like(delta, dtype=float)
    ratio = np.abs(delta) / np.maximum(np.abs(base), eps)
    out[valid] = np.sign(delta[valid]) * np.log1p(ratio[valid])
    return np.clip(out, -1.0, 1.0)


def _log_ratio_reference(x: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> np.ndarray:
    eps = 1e-12
    x_safe = np.where(x > eps, x, eps)
    ref_safe = np.where(ref > eps, ref, np.where(x > eps, x, eps))
    out = np.zeros_like(x_safe, dtype=float)
    out[valid] = np.log(x_safe[valid] / ref_safe[valid])
    return np.clip(out, -1.0, 1.0)


def _signed_log_ratio_reference(delta: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> np.ndarray:
    eps = 1e-12
    out = np.zeros_like(delta, dtype=float)
    ratio = np.abs(delta) / np.maximum(ref, eps)
    out[valid] = np.sign(delta[valid]) * np.log1p(ratio[valid])
    return np.clip(out, -1.0, 1.0)


def _true_range_reference(df: pd.DataFrame, valid: np.ndarray) -> np.ndarray:
    high = _to_float(df, "H_t")
    low = _to_float(df, "L_t")
    close = _to_float(df, "C_t")
    prev = np.roll(close, 1)
    if len(prev):
        prev[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)])
    return np.where(valid, np.clip(tr, 0.0, None), np.nan)


def _small_slope(vals: list[float]) -> float:
    if len(vals) == 3:
        return (float(vals[2]) - float(vals[0])) / 2.0
    if len(vals) == 5:
        return (-2.0 * float(vals[0]) - float(vals[1]) + float(vals[3]) + 2.0 * float(vals[4])) / 10.0
    return 0.0


@dataclass
class ReferenceBuild:
    market: pd.DataFrame
    risk: pd.DataFrame
    htf: pd.DataFrame
    fit_params: dict[str, list[float]]
    internals: dict[str, Any]


def build_reference(df: pd.DataFrame, train_days: set[int]) -> ReferenceBuild:
    eps = 1e-12
    n = len(df)
    idx = df.index
    valid = _to_float(df, "mask_t") > 0.0
    trading_day = _to_float(df, "trading_day").astype(int)
    train_mask = np.asarray([int(x) in train_days for x in trading_day], dtype=bool) & valid
    fit: dict[str, list[float]] = {}

    close = _to_float(df, "C_t")
    high = _to_float(df, "H_t")
    low = _to_float(df, "L_t")
    ref = _to_float(df, "ref_close_t")
    ref_safe = np.where(ref > eps, ref, np.where(close > eps, close, eps))

    market = pd.DataFrame(index=idx)

    # Price/local features.
    market["obs_cumVWAP_t"] = _log_ratio_reference(_to_float(df, "cumVWAP_t"), ref_safe, valid)
    market["obs_dC_minus_cumVWAP_t"] = _signed_log_ratio_reference(_to_float(df, "dC_minus_cumVWAP_t"), ref_safe, valid)
    market["obs_cmp_C_vs_cumVWAP_t"] = _to_float(df, "cmp_C_vs_cumVWAP_t")
    market["obs_session_high_t"] = _log_ratio_reference(_to_float(df, "session_high_t"), ref_safe, valid)
    market["obs_session_low_t"] = _log_ratio_reference(_to_float(df, "session_low_t"), ref_safe, valid)
    if "range_frac_t" in df.columns:
        range_frac = _to_float(df, "range_frac_t")
    else:
        range_frac = np.maximum(_to_float(df, "session_high_t") - _to_float(df, "session_low_t"), 0.0) / ref_safe
    obs_range = np.zeros_like(range_frac)
    obs_range[valid] = np.log1p(np.clip(range_frac[valid], 0.0, None))
    market["obs_range_t"] = np.clip(obs_range, 0.0, 1.0)
    if "open_drift_t" in df.columns:
        od = _to_float(df, "open_drift_t")
    else:
        session_open = _to_float(df, "session_open_t")
        od = np.zeros_like(close)
        od[valid] = np.log(np.maximum(close[valid], eps) / np.maximum(session_open[valid], eps))
    market["obs_open_drift_t"] = np.clip(od, -1.0, 1.0)
    market["obs_bar_dir_t"] = _to_float(df, "bar_dir_t")
    market["obs_mask_t"] = _to_float(df, "mask_t")

    minute = _to_series(df, "minute_index_t")
    if "day_id" in df.columns:
        denom = minute.groupby(df["day_id"]).transform("max").astype(float).clip(lower=1.0)
    else:
        denom = pd.Series(max(float(np.nanmax(minute.to_numpy())), 1.0), index=idx)
    market["obs_minute_index_t"] = (minute / denom).clip(0.0, 1.0).to_numpy(dtype=float)
    market["obs_session_phase_t"] = (minute >= 120.0).astype(float).to_numpy() if float(np.nanmax(minute.to_numpy())) <= 400.0 else 0.0

    dI = _to_float(df, "dI_from_yclose_t")
    oi = _to_float(df, "I_t")
    oi_yclose = oi - dI
    market["obs_dI_from_yclose_t"] = _signed_log1p_relative(dI, oi_yclose, valid)
    market["obs_pct_chg_from_ref_t"] = _log_ratio_reference(close, ref_safe, valid)

    # Volume surprise.
    volume = _to_series(df, "V_t")
    mask_s = pd.Series(valid, index=idx)
    vol_valid = volume.where(mask_s, np.nan)
    sid = df["session_id"].astype(str) if "session_id" in df.columns else pd.Series("0", index=idx)
    ema = vol_valid.groupby(sid).transform(lambda x: x.ewm(span=30, adjust=False, min_periods=5, ignore_na=True).mean())
    ema_prev = ema.groupby(sid).shift(1)
    exp_mean = vol_valid.groupby(sid).transform(lambda x: x.expanding(min_periods=1).mean())
    base = ema_prev.fillna(exp_mean.groupby(sid).shift(1)).fillna(0.0).to_numpy(dtype=float)
    vol_np = volume.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_log = np.log((np.clip(vol_np, 0.0, None) + 1.0) / (np.maximum(base, 1.0) + 1.0))
    vol_log = np.nan_to_num(vol_log, nan=0.0, posinf=5.0, neginf=-5.0)
    volume_feature = np.clip(vol_log, -5.0, 5.0) * valid.astype(float)
    market["volume_surprise_logratio_floor"] = volume_feature
    market["volume_surprise_rolling_percentile"] = _rolling_percentile_reference(
        volume_feature, valid, window=240, min_periods=30)
    volume_base = volume_feature
    market["volume_impulse_recent"] = pd.Series(volume_base).rolling(5, min_periods=1).mean().fillna(0.0).to_numpy(dtype=float)
    market["volume_impulse_slope_3"] = _slope_reference(volume_base, 3)
    market["volume_impulse_slope_5"] = _slope_reference(volume_base, 5)
    vcount = pd.Series(np.where(valid, 1.0, np.nan)).rolling(30, min_periods=1).count().fillna(0.0).to_numpy(dtype=float)
    market["volume_impulse_ready_flag"] = (vcount >= 30).astype(float)

    # OI.
    market["oi_rel_yclose_log"] = _signed_log1p_relative(dI, oi_yclose, valid)
    oi_s = pd.Series(oi, index=idx, dtype=float)
    oi_valid = oi_s.where(mask_s, np.nan)
    session_open_oi = sid.map(oi_valid.groupby(sid).first()).astype(float).to_numpy()
    market["oi_rel_session_open_log"] = _signed_log1p_relative(oi - session_open_oi, session_open_oi, valid)
    oi_delta = oi_s.diff()
    prev_valid = mask_s.shift(1, fill_value=False).astype(bool)
    delta_valid = oi_delta.where(mask_s & prev_valid, np.nan)
    hist = delta_valid.shift(1)
    mu = hist.rolling(window=240, min_periods=30).mean()
    sd = hist.rolling(window=240, min_periods=30).std(ddof=0).replace(0.0, np.nan)
    oi_z = ((delta_valid - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    market["oi_delta_rolling_z"] = np.clip(oi_z, -5.0, 5.0)
    market["oi_rolling_percentile"] = _rolling_percentile_reference(market["oi_delta_rolling_z"].to_numpy(dtype=float), valid, window=240, min_periods=30)
    market["oi_delta_slope_3"] = _slope_reference(market["oi_delta_rolling_z"].to_numpy(dtype=float), 3)
    market["oi_delta_slope_5"] = _slope_reference(market["oi_delta_rolling_z"].to_numpy(dtype=float), 5)
    ret5 = pd.Series(np.log(np.maximum(close, eps))).diff(5).fillna(0.0).to_numpy(dtype=float)
    oi_pos = np.clip(market["oi_delta_rolling_z"].to_numpy(dtype=float), 0.0, None)
    market["oi_price_confirm_short_continuous"] = np.clip(np.maximum(-ret5, 0.0) * oi_pos, 0.0, 5.0)
    market["oi_price_confirm_long_continuous"] = np.clip(np.maximum(ret5, 0.0) * oi_pos, 0.0, 5.0)
    market["oi_impulse_ready_flag"] = (vcount >= 30).astype(float)

    # Volatility and risk.
    log_close = np.log(np.maximum(close, eps))
    log_close_s = pd.Series(log_close)
    raw_vol = (
        log_close_s.where(valid, np.nan)
        .rolling(window=30, min_periods=30)
        .std()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    fit["vol_raw_30"] = _quantile_grid(raw_vol[train_mask])
    close_valid = pd.Series(close, index=idx, dtype=float).where(valid, np.nan)
    prev_close_in_session = close_valid.groupby(sid).shift(1)
    session_logret = np.log(close_valid / prev_close_in_session)
    session_logret = session_logret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    session_raw_vol = (
        session_logret.groupby(sid)
        .rolling(window=30, min_periods=5)
        .std()
        .reset_index(level=0, drop=True)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=float)
    )
    market["vol_rolling_percentile"] = _rolling_percentile_reference(session_raw_vol, valid, window=240, min_periods=30)

    tr = _true_range_reference(df, valid)
    atr30, c30 = _rolling_mean_valid_reference(tr, valid, 30, 30)
    atr60, c60 = _rolling_mean_valid_reference(tr, valid, 60, 60)
    atr30_frac = atr30 / ref_safe
    atr60_frac = atr60 / ref_safe
    fit["atr_1m_30_price_frac"] = _quantile_grid(atr30_frac[train_mask])
    fit["atr_1m_60_price_frac"] = _quantile_grid(atr60_frac[train_mask])
    risk = pd.DataFrame(index=idx)
    risk["atr_1m_30_price_frac"] = atr30_frac
    risk["atr_1m_60_price_frac"] = atr60_frac
    risk["atr_1m_30_train_percentile"] = _pct_grid(atr30_frac, fit["atr_1m_30_price_frac"])
    risk["atr_1m_60_train_percentile"] = _pct_grid(atr60_frac, fit["atr_1m_60_price_frac"])
    risk["vol_train_percentile"] = _pct_grid(raw_vol, fit["vol_raw_30"])
    risk["current_bar_range_atr_30"] = np.clip((high - low) / np.maximum(atr30, 1e-6), 0.0, 20.0)
    risk["intraday_volatility_percentile"] = market["vol_rolling_percentile"].to_numpy(dtype=float)
    risk["atr_ready_flag"] = ((c30 >= 30) & (c60 >= 60)).astype(float)

    # Price train percentiles.
    for name in ("obs_cumVWAP_t", "obs_session_high_t", "obs_range_t", "obs_pct_chg_from_ref_t"):
        vals = market[name].to_numpy(dtype=float)
        fit[name] = _quantile_grid(vals[train_mask])
        market[f"{name}_train_percentile"] = _pct_grid(vals, fit[name])

    macd = _macd_reference(df, atr30_frac)
    for col in macd.columns:
        market[col] = macd[col].to_numpy(dtype=float)

    daily, daily_fit = _daily_reference(df, train_days)
    h1, h1_fit = _h1_reference(df, train_days)
    fit.update(daily_fit)
    fit.update(h1_fit)
    htf = pd.concat([daily, h1], axis=1)

    # Mask non-clock/non-ready values exactly like env contract.
    mask_float = valid.astype(float)
    for frame in (market, risk, htf):
        for col in frame.columns:
            if col.endswith("_ready_flag") or col in ("obs_mask_t", "obs_minute_index_t", "obs_session_phase_t"):
                continue
            frame[col] = frame[col].to_numpy(dtype=float) * mask_float
    market = market.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    risk = risk.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    htf = htf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return ReferenceBuild(
        market=market[list(MARKET_V2_CONTINUOUS_FEATURES)],
        risk=risk[list(RISK_CONTEXT_FEATURES)],
        htf=htf[list(HTF_CONTEXT_FEATURES)],
        fit_params=fit,
        internals={"raw_vol": raw_vol, "atr30_frac": atr30_frac, "atr60_frac": atr60_frac},
    )


def _macd_reference(df: pd.DataFrame, atr30_frac: np.ndarray) -> pd.DataFrame:
    names = [name for name in MARKET_V2_CONTINUOUS_FEATURES if name.startswith("dyn5m_macd_")]
    out = np.zeros((len(df), len(names)), dtype=float)
    valid = _to_float(df, "mask_t") > 0.0
    close = _to_float(df, "C_t")
    ref = np.maximum(_to_float(df, "ref_close_t"), 1e-12)
    denom = np.maximum(ref * np.maximum(atr30_frac, 1e-6), 1e-6)
    af, aslow, asig = 2 / 13, 2 / 27, 2 / 10
    ema_f = ema_s = signal = None
    commit_f = commit_s = commit_sig = None
    current_bucket = -1
    completed = 0
    prev_hist = 0.0
    prev_sign = 0
    cross_age = 0
    hist_window: deque[float] = deque(maxlen=5)
    valid_count = 0
    for i in range(len(df)):
        if not valid[i]:
            continue
        bucket = valid_count // 5
        price = float(close[i])
        if ema_f is None:
            ema_f = ema_s = commit_f = commit_s = price
            signal = commit_sig = 0.0
        if bucket != current_bucket:
            if current_bucket >= 0:
                commit_f, commit_s, commit_sig = ema_f, ema_s, signal
                completed += 1
            current_bucket = bucket
        base_f = commit_f if commit_f is not None else price
        base_s = commit_s if commit_s is not None else price
        base_sig = commit_sig if commit_sig is not None else 0.0
        ema_f = af * price + (1 - af) * base_f
        ema_s = aslow * price + (1 - aslow) * base_s
        macd = ema_f - ema_s
        signal = asig * macd + (1 - asig) * base_sig
        hist = macd - signal
        sign = int(np.sign(macd - signal))
        if sign and prev_sign and sign != prev_sign:
            cross_age = 0
        else:
            cross_age += 1
        if sign:
            prev_sign = sign
        hist_window.append(float(hist))
        vals = list(hist_window)
        col = {name: j for j, name in enumerate(names)}
        out[i, col["dyn5m_macd_line_norm"]] = float(np.clip(macd / denom[i], -5, 5))
        out[i, col["dyn5m_macd_signal_norm"]] = float(np.clip(signal / denom[i], -5, 5))
        out[i, col["dyn5m_macd_hist_norm"]] = float(np.clip(hist / denom[i], -5, 5))
        out[i, col["dyn5m_macd_hist_delta"]] = float(np.clip((hist - prev_hist) / denom[i], -5, 5))
        out[i, col["dyn5m_macd_distance_norm"]] = float(np.clip((macd - signal) / denom[i], -5, 5))
        if len(vals) >= 3:
            out[i, col["dyn5m_macd_hist_slope_3"]] = float(np.clip(_small_slope(vals[-3:]) / denom[i], -5, 5))
        if len(vals) >= 5:
            out[i, col["dyn5m_macd_hist_slope_5"]] = float(np.clip(_small_slope(vals[-5:]) / denom[i], -5, 5))
        out[i, col["dyn5m_macd_cross_age_frac"]] = float(np.clip(cross_age / 120.0, 0.0, 1.0))
        out[i, col["dyn5m_macd_cross_dir"]] = float(sign)
        out[i, col["dyn5m_macd_ready_flag"]] = 1.0 if completed >= 35 else 0.0
        prev_hist = hist
        valid_count += 1
    return pd.DataFrame(out, index=df.index, columns=names)


def _daily_reference(df: pd.DataFrame, train_days: set[int]) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    eps = 1e-12
    names = list(HTF_CONTEXT_FEATURES[:9])
    valid = _to_float(df, "mask_t") > 0.0
    days = _to_series(df, "trading_day").astype(int)
    close = _to_series(df, "C_t").where(valid, np.nan)
    high = _to_series(df, "H_t").where(valid, np.nan)
    low = _to_series(df, "L_t").where(valid, np.nan)
    summary = pd.DataFrame({
        "close": close.groupby(days).last(),
        "high": high.groupby(days).max(),
        "low": low.groupby(days).min(),
    }).dropna()
    ordered = list(summary.index.astype(int))
    pos = {day: i for i, day in enumerate(ordered)}
    daily_range = (summary["high"] - summary["low"]).clip(lower=0.0).to_numpy(dtype=float)
    fit = {"daily_range": _quantile_grid(daily_range[[d in train_days for d in ordered]])}
    per_day = np.zeros((len(ordered), len(names)), dtype=float)
    closes = summary["close"].to_numpy(dtype=float)
    highs = summary["high"].to_numpy(dtype=float)
    lows = summary["low"].to_numpy(dtype=float)
    log_close = np.log(np.maximum(closes, eps))
    for p in range(len(ordered)):
        for n in (1, 3, 5):
            if p - n >= 0:
                per_day[p, names.index(f"daily_ret_{n}")] = float(log_close[p] - log_close[p - n])
        for n in (5, 10):
            if p - n + 1 >= 0:
                x = np.arange(n, dtype=float)
                x -= x.mean()
                y = log_close[p - n + 1:p + 1]
                denom = float(np.sum(x * x)) or 1.0
                per_day[p, names.index(f"daily_slope_{n}")] = float(np.sum((y - y.mean()) * x) / denom)
                lo = float(np.min(lows[p - n + 1:p + 1]))
                hi = float(np.max(highs[p - n + 1:p + 1]))
                per_day[p, names.index(f"daily_close_pos_in_range_{n}")] = float(2.0 * (closes[p] - lo) / max(hi - lo, eps) - 1.0)
        per_day[p, names.index("daily_range_percentile_train")] = _pct_grid(np.asarray([daily_range[p]]), fit["daily_range"])[0]
        per_day[p, names.index("daily_ready_flag")] = 1.0 if p >= 10 else 0.0
    row_days = days.to_numpy(dtype=int)
    prev_pos = np.asarray([pos.get(int(day), -1) - 1 for day in row_days], dtype=int)
    out = np.zeros((len(df), len(names)), dtype=float)
    valid_prev = prev_pos >= 0
    out[valid_prev, :] = per_day[prev_pos[valid_prev], :]
    return pd.DataFrame(out, index=df.index, columns=names), fit


def _h1_reference(df: pd.DataFrame, train_days: set[int]) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    eps = 1e-12
    names = list(HTF_CONTEXT_FEATURES[9:])
    out = np.zeros((len(df), len(names)), dtype=float)
    valid_rows = np.flatnonzero(_to_float(df, "mask_t") > 0.0)
    close = _to_float(df, "C_t")
    high = _to_float(df, "H_t")
    low = _to_float(df, "L_t")
    trading_day = _to_float(df, "trading_day").astype(int)
    bars = []
    for start in range(0, len(valid_rows), 60):
        rows = valid_rows[start:start + 60]
        if len(rows) < 60:
            break
        bars.append({
            "end_row": int(rows[-1]),
            "close": float(close[rows[-1]]),
            "high": float(np.max(high[rows])),
            "low": float(np.min(low[rows])),
            "trading_day": int(trading_day[rows[-1]]),
        })
    if not bars:
        return pd.DataFrame(out, index=df.index, columns=names), {"h1_range": [0.0]}
    end_rows = np.asarray([b["end_row"] for b in bars], dtype=int)
    closes = np.asarray([b["close"] for b in bars], dtype=float)
    highs = np.asarray([b["high"] for b in bars], dtype=float)
    lows = np.asarray([b["low"] for b in bars], dtype=float)
    ranges = np.clip(highs - lows, 0.0, None)
    bar_train = np.asarray([b["trading_day"] in train_days for b in bars], dtype=bool)
    fit = {"h1_range": _quantile_grid(ranges[bar_train])}
    log_close = np.log(np.maximum(closes, eps))
    per_h1 = np.zeros((len(bars), len(names)), dtype=float)
    for p in range(len(bars)):
        for n in (1, 3, 6, 12):
            if p - n >= 0:
                per_h1[p, names.index(f"h1_ret_{n}")] = float(log_close[p] - log_close[p - n])
        for n in (6, 12):
            if p - n + 1 >= 0:
                x = np.arange(n, dtype=float)
                x -= x.mean()
                y = log_close[p - n + 1:p + 1]
                denom = float(np.sum(x * x)) or 1.0
                per_h1[p, names.index(f"h1_slope_{n}")] = float(np.sum((y - y.mean()) * x) / denom)
                lo = float(np.min(lows[p - n + 1:p + 1]))
                hi = float(np.max(highs[p - n + 1:p + 1]))
                per_h1[p, names.index(f"h1_close_pos_in_range_{n}")] = float(2.0 * (closes[p] - lo) / max(hi - lo, eps) - 1.0)
        per_h1[p, names.index("h1_range_percentile_train")] = _pct_grid(np.asarray([ranges[p]]), fit["h1_range"])[0]
        per_h1[p, names.index("h1_ready_flag")] = 1.0 if p >= 12 else 0.0
    row_pos = np.searchsorted(end_rows, np.arange(len(df)), side="right") - 1
    valid_row_pos = row_pos >= 0
    out[valid_row_pos, :] = per_h1[row_pos[valid_row_pos], :]
    valid_seen = np.searchsorted(valid_rows, np.arange(len(df)), side="right")
    bars_since_h1 = np.maximum(0, valid_seen - ((row_pos + 1) * 60))
    out[valid_row_pos, names.index("last_completed_h1_age_frac")] = np.clip(bars_since_h1[valid_row_pos] / 60.0, 0.0, 1.0)
    return pd.DataFrame(out, index=df.index, columns=names), fit


def select_audit_rows(store, candidates: pd.DataFrame, roles: dict[int, str], dreamer_root: Path, sample_size: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    n = int(store.n_rows)
    row_days = np.asarray(store.row_trading_day, dtype=int)
    valid_rows = np.flatnonzero(store.row_mask >= 0.5)
    for role in ("train", "validation", "test"):
        role_valid = valid_rows[np.asarray([roles.get(int(row_days[r])) == role for r in valid_rows])]
        if role_valid.size:
            take = min(max(1, sample_size // 3), role_valid.size)
            for r in rng.choice(role_valid, size=take, replace=False):
                rows.append({"row": int(r), "source": f"random_{role}"})
    for _, rec in candidates.sample(min(1000, len(candidates)), random_state=seed).iterrows():
        rows.append({"row": int(rec["decision_row"]), "source": "entry_candidate"})
    long_pos = candidates[candidates["long_net_pnl"] > 0]
    short_pos = candidates[candidates["short_net_pnl"] > 0]
    weak = candidates[(candidates[["long_net_pnl", "short_net_pnl"]].max(axis=1) <= 0)]
    for label, frame in (("oracle_positive_long", long_pos), ("oracle_positive_short", short_pos), ("oracle_negative", weak)):
        if not frame.empty:
            for _, rec in frame.sample(min(300, len(frame)), random_state=seed + len(label)).iterrows():
                rows.append({"row": int(rec["decision_row"]), "source": label})
    if dreamer_root.exists():
        for path in sorted(dreamer_root.glob("*/dreamer_candidate_attribution.csv"))[:20]:
            try:
                df = pd.read_csv(path, usecols=["candidate_id"])
            except Exception:
                continue
            ids = set(pd.to_numeric(df["candidate_id"], errors="coerce").dropna().astype(int).tolist())
            sub = candidates[candidates["candidate_id"].isin(ids)]
            for _, rec in sub.head(300).iterrows():
                rows.append({"row": int(rec["decision_row"]), "source": "dreamer_selected"})
    ts = pd.DatetimeIndex(store.index)
    boundary_masks = {
        "night_open": (ts.hour == 21) & (ts.minute <= 5),
        "day_open": ((ts.hour == 9) & (ts.minute <= 5)) | ((ts.hour == 13) & (ts.minute >= 30) & (ts.minute <= 35)),
        "before_break": (ts.hour == 10) & (ts.minute >= 10) & (ts.minute <= 14),
        "after_break": ((ts.hour == 10) & (ts.minute >= 30) & (ts.minute <= 35)) | ((ts.hour == 13) & (ts.minute >= 30) & (ts.minute <= 35)),
        "near_eod": (ts.hour == 14) & (ts.minute >= 50),
        "cross_day_boundary": (ts.hour == 21) & (ts.minute >= 0) & (ts.minute <= 10),
    }
    for label, mask in boundary_masks.items():
        idx = np.flatnonzero(mask & (store.row_mask >= 0.5))
        for r in idx[:20]:
            rows.append({"row": int(r), "source": label})
    for key, arr, names in (
        ("market_seq", store.X_schema_market_seq, list(store.obs_feature_names_by_key["market_seq"])),
        ("risk_context", store.X_schema_risk_context, list(store.obs_feature_names_by_key["risk_context"])),
        ("htf_context", store.X_schema_htf_context, list(store.obs_feature_names_by_key["htf_context"])),
    ):
        for name in names:
            if not name.endswith("_ready_flag"):
                continue
            col = names.index(name)
            vals = arr[:, col]
            true_rows = np.flatnonzero(vals >= 0.5)
            false_rows = np.flatnonzero(vals < 0.5)
            if true_rows.size:
                r = int(true_rows[0])
                rows.append({"row": r, "source": f"{name}_first_ready"})
                for rr in range(max(0, r - 3), min(n, r + 4)):
                    rows.append({"row": int(rr), "source": f"{name}_ready_boundary"})
            if false_rows.size:
                rows.append({"row": int(false_rows[min(len(false_rows) - 1, 10)]), "source": f"{name}_not_ready"})
    df_rows = pd.DataFrame(rows).drop_duplicates("row").sort_values("row")
    df_rows = df_rows[(df_rows["row"] >= 0) & (df_rows["row"] < n)].copy()
    df_rows["timestamp"] = [str(store.index[int(r)]) for r in df_rows["row"]]
    df_rows["trading_day"] = [int(store.row_trading_day[int(r)]) for r in df_rows["row"]]
    df_rows["split_role"] = [roles.get(int(x), "unknown") for x in df_rows["trading_day"]]
    return df_rows


def compare_features(
    *,
    actual: np.ndarray,
    expected: np.ndarray,
    names: tuple[str, ...],
    rows: np.ndarray,
    key: str,
    family_override: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = []
    mismatches = []
    for i, name in enumerate(names):
        got = actual[rows, i].astype(float)
        exp = expected[rows, i].astype(float)
        diff = np.abs(got - exp)
        tol = _feature_tol(name)
        mismatch = diff > tol
        family = family_override or _feature_family(key, name)
        status = "PASS" if not np.any(mismatch) else ("FLOAT_TOLERANCE_ONLY" if float(diff.max()) <= max(tol * 10, 1e-5) else "FAIL")
        summary.append({
            "key": key,
            "feature_name": name,
            "family": family,
            "tolerance": tol,
            "max_abs_error": float(diff.max()) if diff.size else 0.0,
            "mean_abs_error": float(diff.mean()) if diff.size else 0.0,
            "p95_abs_error": float(np.quantile(diff, 0.95)) if diff.size else 0.0,
            "mismatch_count": int(mismatch.sum()),
            "mismatch_ratio": float(mismatch.mean()) if mismatch.size else 0.0,
            "status": status,
            "mismatch_type": _mismatch_type(key, name, bool(np.any(mismatch)), float(diff.max()) if diff.size else 0.0, tol),
        })
        if np.any(mismatch):
            worst = np.argsort(diff)[-20:][::-1]
            for pos in worst:
                if diff[pos] <= tol:
                    continue
                mismatches.append({
                    "key": key,
                    "feature_name": name,
                    "family": family,
                    "row": int(rows[pos]),
                    "actual": float(got[pos]),
                    "expected": float(exp[pos]),
                    "abs_error": float(diff[pos]),
                    "tolerance": tol,
                    "mismatch_type": _mismatch_type(key, name, True, float(diff[pos]), tol),
                })
    return pd.DataFrame(summary), pd.DataFrame(mismatches)


def _feature_tol(name: str) -> float:
    if name.startswith("dyn5m_macd_") or "percentile" in name or "slope" in name:
        return 1e-5
    return 1e-6


def _feature_family(key: str, name: str) -> str:
    if key == "risk_context":
        return "atr_risk"
    if key == "htf_context":
        return "daily_htf" if name.startswith("daily_") else "h1_htf"
    for family, names in MARKET_GROUPS.items():
        if name in names:
            return family
    return key


def _mismatch_type(key: str, name: str, mismatched: bool, max_err: float, tol: float) -> str:
    if not mismatched:
        return "NONE"
    if max_err <= max(tol * 10.0, 1e-5):
        return "FLOAT_TOLERANCE_ONLY"
    if "percentile" in name:
        return "FIT_PARAM_MISMATCH"
    if name.endswith("_ready_flag"):
        return "WARMUP_MISMATCH"
    if key == "htf_context":
        return "TIMING_ALIGNMENT_MISMATCH"
    if key == "risk_context":
        return "NORMALIZATION_DENOMINATOR_MISMATCH"
    return "FORMULA_MISMATCH"


def fit_param_check(saved: dict[str, Any], ref: dict[str, list[float]], split_hash: str) -> pd.DataFrame:
    records = []
    records.append({
        "param_name": "split_manifest_hash",
        "saved_hash": saved.get("split_manifest_hash", ""),
        "expected_hash": split_hash,
        "max_abs_error": 0.0,
        "status": "PASS" if saved.get("split_manifest_hash", "") == split_hash else "FAIL",
    })
    for key, exp in sorted(ref.items()):
        got = saved.get(key)
        if got is None:
            records.append({"param_name": key, "saved_hash": "", "expected_hash": "", "max_abs_error": math.inf, "status": "MISSING"})
            continue
        got_arr = np.asarray(got, dtype=float)
        exp_arr = np.asarray(exp, dtype=float)
        if got_arr.shape != exp_arr.shape:
            max_err = math.inf
            status = "SHAPE_MISMATCH"
        else:
            max_err = float(np.max(np.abs(got_arr - exp_arr))) if got_arr.size else 0.0
            status = "PASS" if max_err <= 1e-9 else "FAIL"
        records.append({
            "param_name": key,
            "saved_hash": hashlib.sha256(json.dumps(got, sort_keys=True).encode()).hexdigest() if got is not None else "",
            "expected_hash": hashlib.sha256(json.dumps(exp, sort_keys=True).encode()).hexdigest(),
            "max_abs_error": max_err,
            "status": status,
        })
    return pd.DataFrame(records)


def warmup_check(actual_frames: dict[str, tuple[np.ndarray, tuple[str, ...]]], ref_frames: dict[str, tuple[np.ndarray, tuple[str, ...]]]) -> pd.DataFrame:
    records = []
    for key, (actual, names) in actual_frames.items():
        ref, _ = ref_frames[key]
        for name in names:
            if not name.endswith("_ready_flag"):
                continue
            i = names.index(name)
            got = actual[:, i]
            exp = ref[:, i]
            diff = got != exp
            got_first = int(np.flatnonzero(got >= 0.5)[0]) if np.any(got >= 0.5) else -1
            exp_first = int(np.flatnonzero(exp >= 0.5)[0]) if np.any(exp >= 0.5) else -1
            records.append({
                "key": key,
                "feature_name": name,
                "actual_first_ready_row": got_first,
                "expected_first_ready_row": exp_first,
                "mismatch_count": int(diff.sum()),
                "status": "PASS" if not np.any(diff) else "FAIL",
            })
    return pd.DataFrame(records)


def agent_state_check(env_config: Path, out_dir: Path) -> pd.DataFrame:
    env = CustomTradingEnv(config_path=str(env_config))
    rows = []
    try:
        obs, _ = env.reset(seed=123)
        rows.append(_agent_state_row(env, obs, "reset"))
        for label, action in (("hold_flat", 1), ("try_long", 2), ("hold_after_long", 1), ("try_flat", 1), ("try_short", 0)):
            obs, _, terminated, truncated, _ = env.step(action)
            rows.append(_agent_state_row(env, obs, label))
            if terminated or truncated:
                break
    finally:
        env.close()
    return pd.DataFrame(rows)


def _agent_state_row(env: CustomTradingEnv, obs: dict[str, Any], label: str) -> dict[str, Any]:
    names = list(FEATURES_AGENT_OBS)
    vec = np.asarray(obs["agent_state"], dtype=float)
    action_mask = np.asarray(obs["action_mask"], dtype=float)
    expected = np.zeros_like(vec)
    idx = {name: i for i, name in enumerate(names)}
    expected[idx["obs_market_open_t"]] = 1.0 if float(env.bar_source.store.row_mask[int(env.current_step)]) >= 0.5 else 0.0
    have_long = float(env.user_accounts.long_position > 0)
    have_short = float(env.user_accounts.short_position > 0)
    is_flat = not have_long and not have_short
    expected[idx["obs_can_long_open_t"]] = 1.0 if is_flat and action_mask[2] >= 0.5 else 0.0
    expected[idx["obs_can_short_open_t"]] = 1.0 if is_flat and action_mask[0] >= 0.5 else 0.0
    expected[idx["obs_can_long_close_t"]] = 1.0 if have_long and action_mask[1] >= 0.5 else 0.0
    expected[idx["obs_can_short_close_t"]] = 1.0 if have_short and action_mask[1] >= 0.5 else 0.0
    expected[idx["obs_pos_side_t"]] = 1.0 if have_long else (-1.0 if have_short else 0.0)
    max_entries = max(1, int(getattr(env.config.trading, "max_entries_per_day", 1)))
    used = max(0, int(getattr(env, "_entries_used_today", 0)))
    expected[idx["obs_entries_left_frac_t"]] = max(0, max_entries - used) / float(max_entries)
    eod_idx = int(getattr(env, "_eod_idx", env.end_idx))
    expected[idx["obs_minutes_to_eod_frac_t"]] = min(max(0, eod_idx - int(env.current_step)), int(env.DAY_LEN)) / float(max(1, int(env.DAY_LEN)))
    raw_hold = 0.0
    for pos in list(env.position_manager.long_positions) + list(env.position_manager.short_positions):
        if pos is not None:
            raw_hold = max(raw_hold, float(int(env.current_step) - int(pos.open_step)))
    expected[idx["obs_holding_frac_t"]] = min(max(raw_hold / float(max(1, int(env.DAY_LEN))), 0.0), 1.0)
    r_cash = float(getattr(env, "_R_cash_last", 1.0) or 1.0)
    if r_cash > 0:
        expected[idx["obs_upnl_R_t"]] = float(np.clip(float(env.user_accounts.unrealized_pnl) / r_cash, -5.0, 5.0))
        day_start = getattr(env, "_day_start_realized_cum", env.user_accounts.realized_pnl)
        realized_today = float(env.user_accounts.realized_pnl - day_start)
        expected[idx["obs_realized_today_R_t"]] = float(np.clip(realized_today / r_cash, -10.0, 10.0))
    b0 = float(env.config.trading.initial_balance)
    equity = float(env.user_accounts.equity())
    drawdown = float(getattr(env, "drawdown", 0.0))
    expected[idx["obs_equity_frac_t"]] = float(np.clip((equity - b0) / max(b0, 1e-12), -1.0, 1.0))
    expected[idx["obs_drawdown_frac_t"]] = float(np.clip(drawdown / max(b0, 1e-12), 0.0, 1.0))
    from gym_trading_env.envs.action import ForexCode
    max_code = max(int(x.value) for x in ForexCode)
    code = int(getattr(getattr(env, "action_result", None), "value", 0))
    expected[idx["obs_action_result_t"]] = min(max(code, 0), max_code) / float(max(max_code, 1))
    diff = np.abs(vec - expected)
    tol = 1e-5
    return {
        "label": label,
        "row": int(env.current_step),
        "timestamp": str(env.bar_source.store.index[int(env.current_step)]),
        "max_abs_error": float(diff.max()) if diff.size else 0.0,
        "tolerance": tol,
        "mismatch_count": int((diff > tol).sum()),
        "status": "PASS" if not np.any(diff > tol) else "FAIL",
    }


def _write_report(path: Path, summary: dict[str, Any], feature_summary: pd.DataFrame) -> None:
    failed = feature_summary[feature_summary["status"] == "FAIL"]
    by_family = feature_summary.groupby("family", dropna=False).agg(
        features=("feature_name", "count"),
        failed=("status", lambda x: int((x == "FAIL").sum())),
        max_abs_error=("max_abs_error", "max"),
    ).reset_index()
    lines = [
        "# Obs v2 Formula Reference Audit v1",
        "",
        f"Date: {summary['generated_at']}",
        "",
        "## Final Answer",
        "",
        f"Classification: `{summary['classification']}`.",
        "",
        "This audit independently recomputes obs_v2 formulas from aligned raw 1m rows and compares them against env/store outputs. "
        "No Dreamer training, reward change, execution-contract change, or feature addition was performed.",
        "",
        "## Scope",
        "",
        f"- Audited rows: {summary['rows']['unique_reference_rows']}",
        f"- Decision rows: {summary['rows']['decision_rows']}",
        f"- Market features: {summary['schema']['market_features']}",
        f"- Risk context features: {summary['schema']['risk_features']}",
        f"- HTF context features: {summary['schema']['htf_features']}",
        f"- Agent state features: {summary['schema']['agent_features']}",
        "",
        "## Result By Family",
        "",
        "| family | features | failed | max_abs_error |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in by_family.sort_values("family").iterrows():
        lines.append(f"| {row['family']} | {int(row['features'])} | {int(row['failed'])} | {float(row['max_abs_error']):.6g} |")
    lines += [
        "",
        "## Fit Params",
        "",
        f"- Split manifest hash match: `{summary['fit_params']['split_manifest_hash_match']}`",
        f"- Failed fit params: {summary['fit_params']['failed']}",
        "",
        "## Warmup",
        "",
        f"- Failed ready flags: {summary['warmup']['failed']}",
        "",
        "## Agent State",
        "",
        f"- Scripted trajectory status: `{summary['agent_state']['status']}`",
        f"- Max abs error: {summary['agent_state']['max_abs_error']:.6g}",
        "",
        "## Mismatch Summary",
        "",
        f"- Failed features: {len(failed)}",
        f"- Mismatch rows written: {summary['mismatches']['rows']}",
        "",
        "## Artifacts",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines += [
        "",
        "## Notes",
        "",
        "- Reference code is implemented in `tools/audit_obs_v2_formula_reference.py` and does not call production `build_observation_schema()` for expected values.",
        "- The audit uses env/store only for actual outputs, aligned raw row access, and scripted env state for agent-state verification.",
        "- A PASS here verifies formula implementation, not strategy readiness.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default=str(DEFAULT_ENV_CONFIG))
    parser.add_argument("--entry-eval-dir", default=str(DEFAULT_ENTRY_EVAL_DIR))
    parser.add_argument("--split-manifest", default=str(DEFAULT_SPLIT_MANIFEST))
    parser.add_argument("--fit-params", default=str(DEFAULT_FIT_PARAMS))
    parser.add_argument("--dreamer-audit-root", default=str(DEFAULT_DREAMER_AUDIT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--doc-path", default=str(DEFAULT_DOC))
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260619)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    env_config = Path(args.env_config)
    split_manifest = Path(args.split_manifest)
    fit_path = Path(args.fit_params)
    roles, days_by_role, train_days, split_hash = _load_split(split_manifest)
    saved_fit = _read_json(fit_path)

    env = CustomTradingEnv(config_path=str(env_config))
    try:
        store = env.bar_source.store
        df = store.df_market.copy()
    finally:
        env.close()
    candidates = pd.read_csv(Path(args.entry_eval_dir) / "candidates.csv")

    audit_rows = select_audit_rows(
        store, candidates, roles, Path(args.dreamer_audit_root), sample_size=int(args.sample_size), seed=int(args.seed))
    audit_rows.to_csv(out / "audit_rows_manifest.csv", index=False)

    reference = build_reference(df, train_days)
    decision_rows = audit_rows["row"].to_numpy(dtype=int)
    window_rows = np.unique(np.concatenate([
        np.arange(max(0, int(r) - 59), int(r) + 1, dtype=int) for r in decision_rows
    ]))

    market_summary, market_mismatch = compare_features(
        actual=store.X_schema_market_seq,
        expected=reference.market.to_numpy(dtype=np.float32),
        names=tuple(MARKET_V2_CONTINUOUS_FEATURES),
        rows=window_rows,
        key="market_seq",
    )
    risk_summary, risk_mismatch = compare_features(
        actual=store.X_schema_risk_context,
        expected=reference.risk.to_numpy(dtype=np.float32),
        names=tuple(RISK_CONTEXT_FEATURES),
        rows=decision_rows,
        key="risk_context",
    )
    htf_summary, htf_mismatch = compare_features(
        actual=store.X_schema_htf_context,
        expected=reference.htf.to_numpy(dtype=np.float32),
        names=tuple(HTF_CONTEXT_FEATURES),
        rows=decision_rows,
        key="htf_context",
    )
    feature_summary = pd.concat([market_summary, risk_summary, htf_summary], ignore_index=True)
    mismatches = pd.concat([market_mismatch, risk_mismatch, htf_mismatch], ignore_index=True)
    feature_summary.to_csv(out / "feature_formula_check_summary.csv", index=False)
    mismatches.to_csv(out / "feature_formula_mismatches.csv", index=False)

    feature_summary[feature_summary["family"] == "macd"].to_csv(out / "macd_reference_check.csv", index=False)
    feature_summary[feature_summary["family"] == "atr_risk"].to_csv(out / "atr_reference_check.csv", index=False)
    feature_summary[feature_summary["family"] == "daily_htf"].to_csv(out / "daily_htf_reference_check.csv", index=False)
    feature_summary[feature_summary["family"] == "h1_htf"].to_csv(out / "h1_htf_reference_check.csv", index=False)

    fit_df = fit_param_check(saved_fit, reference.fit_params, split_hash)
    fit_df.to_csv(out / "fit_params_reference_check.csv", index=False)

    warmup = warmup_check(
        {
            "market_seq": (store.X_schema_market_seq, tuple(MARKET_V2_CONTINUOUS_FEATURES)),
            "risk_context": (store.X_schema_risk_context, tuple(RISK_CONTEXT_FEATURES)),
            "htf_context": (store.X_schema_htf_context, tuple(HTF_CONTEXT_FEATURES)),
        },
        {
            "market_seq": (reference.market.to_numpy(dtype=np.float32), tuple(MARKET_V2_CONTINUOUS_FEATURES)),
            "risk_context": (reference.risk.to_numpy(dtype=np.float32), tuple(RISK_CONTEXT_FEATURES)),
            "htf_context": (reference.htf.to_numpy(dtype=np.float32), tuple(HTF_CONTEXT_FEATURES)),
        },
    )
    warmup.to_csv(out / "warmup_ready_flag_check.csv", index=False)

    agent = agent_state_check(env_config, out)
    agent.to_csv(out / "agent_state_reference_check.csv", index=False)

    failed_features = int((feature_summary["status"] == "FAIL").sum())
    failed_fit = int((fit_df["status"] != "PASS").sum())
    failed_warmup = int((warmup["status"] != "PASS").sum())
    agent_status = "PASS" if not agent.empty and (agent["status"] == "PASS").all() else "FAIL"
    critical_families = {"macd", "atr_risk", "daily_htf", "h1_htf", "oi", "volume"}
    critical_failed = int(((feature_summary["family"].isin(critical_families)) & (feature_summary["status"] == "FAIL")).sum())
    if failed_fit:
        classification = "OBS_V2_FORMULA_FIT_PARAM_FAIL"
    elif critical_failed or agent_status != "PASS" or failed_warmup:
        classification = "OBS_V2_FORMULA_REFERENCE_FAIL"
    elif failed_features:
        classification = "OBS_V2_FORMULA_REFERENCE_PARTIAL"
    elif float(feature_summary["max_abs_error"].max()) > 0:
        classification = "OBS_V2_FORMULA_REFERENCE_PASS_WITH_MINOR_FLOAT_DIFFS"
    else:
        classification = "OBS_V2_FORMULA_REFERENCE_PASS"

    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "classification": classification,
        "inputs": {
            "env_config": str(env_config),
            "entry_eval_dir": str(args.entry_eval_dir),
            "split_manifest": str(split_manifest),
            "split_manifest_hash": split_hash,
            "fit_params": str(fit_path),
            "fit_params_hash": _sha256_file(fit_path),
        },
        "schema": {
            "market_features": len(MARKET_V2_CONTINUOUS_FEATURES),
            "risk_features": len(RISK_CONTEXT_FEATURES),
            "htf_features": len(HTF_CONTEXT_FEATURES),
            "agent_features": len(FEATURES_AGENT_OBS),
        },
        "split": {k: {"days": len(v), "first": min(v) if v else None, "last": max(v) if v else None} for k, v in days_by_role.items()},
        "rows": {
            "decision_rows": int(len(decision_rows)),
            "unique_reference_rows": int(len(window_rows)),
            "candidates": int(len(candidates)),
        },
        "features": {
            "total": int(len(feature_summary)),
            "failed": failed_features,
            "critical_failed": critical_failed,
            "max_abs_error": float(feature_summary["max_abs_error"].max()) if not feature_summary.empty else 0.0,
        },
        "mismatches": {
            "rows": int(len(mismatches)),
        },
        "fit_params": {
            "failed": failed_fit,
            "split_manifest_hash_match": bool(saved_fit.get("split_manifest_hash") == split_hash),
        },
        "warmup": {
            "failed": failed_warmup,
        },
        "agent_state": {
            "status": agent_status,
            "max_abs_error": float(agent["max_abs_error"].max()) if not agent.empty else math.inf,
        },
        "outputs": {
            "feature_formula_check_summary": str(out / "feature_formula_check_summary.csv"),
            "feature_formula_mismatches": str(out / "feature_formula_mismatches.csv"),
            "macd_reference_check": str(out / "macd_reference_check.csv"),
            "atr_reference_check": str(out / "atr_reference_check.csv"),
            "daily_htf_reference_check": str(out / "daily_htf_reference_check.csv"),
            "h1_htf_reference_check": str(out / "h1_htf_reference_check.csv"),
            "agent_state_reference_check": str(out / "agent_state_reference_check.csv"),
            "fit_params_reference_check": str(out / "fit_params_reference_check.csv"),
            "warmup_ready_flag_check": str(out / "warmup_ready_flag_check.csv"),
            "audit_rows_manifest": str(out / "audit_rows_manifest.csv"),
            "summary": str(out / "summary.json"),
            "report": str(args.doc_path),
        },
    }
    _write_json(out / "summary.json", summary)
    _write_report(Path(args.doc_path), summary, feature_summary)


if __name__ == "__main__":
    main()
