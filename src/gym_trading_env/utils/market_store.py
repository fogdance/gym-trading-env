# src/gym_trading_env/utils/market_store.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gym_trading_env.utils.market_features import (
    FEATURES_MARKET,
    FEATURES_MARKET_OBS,
)
from gym_trading_env.utils.daily_features import build_daily_context_and_seq, DAILY_SEQ_LEN
from gym_trading_env.utils.session_futures_strict import DEFAULT_TZ


def _day_key_default(val) -> str:
    """Normalize day_id to a canonical string.
    Matches your env._day_key behavior, but standalone & deterministic."""
    import numpy as _np
    if isinstance(val, _np.generic):
        val = val.item()
    if isinstance(val, (int, _np.integer)):
        return str(int(val))
    if isinstance(val, (float, _np.floating)):
        return f"{float(val):g}"
    return str(val)


@dataclass(frozen=True)
class MarketStore:
    """
    Immutable, production-stable data store for env reading.

    env MUST read only from numpy arrays in this store (not DataFrames),
    so that:
      - df replacement won't break views
      - updates can be done atomically (swap store)
      - contract tests (future-zero padding) stay stable
    """

    # debug / inspection
    df_raw: pd.DataFrame
    df_market: pd.DataFrame

    # core index
    index: pd.DatetimeIndex
    n_rows: int
    day_len: int
    tz: str

    # per-row arrays (global indexing)
    row_mask: np.ndarray          # float32 [n_rows]
    row_minute: np.ndarray        # int32   [n_rows]
    row_day_key: np.ndarray       # object  [n_rows]
    row_day_i: np.ndarray         # int32   [n_rows]

    # NEW: keep raw day_id for metrics/debug (same length as rows)
    row_day_id: np.ndarray        # int32   [n_rows]

    row_C: np.ndarray             # float32 [n_rows]
    row_H: np.ndarray             # float32 [n_rows]
    row_L: np.ndarray             # float32 [n_rows]

    # NEW: global feature matrices (aligned with df_market row order)
    X_market_raw: np.ndarray      # float32 [n_rows, F_raw]  (FEATURES_MARKET)
    X_market_obs: np.ndarray      # float32 [n_rows, F_obs]  (FEATURES_MARKET_OBS)


    # day structures
    days: np.ndarray              # object [num_days]
    sid_to_dayi: Dict[str, int]
    day_ranges: List[Tuple[int, int]]   # half-open [start,end)

    # daily tensors (always size DAY_LEN)
    daily_mask: np.ndarray        # float32 [num_days, DAY_LEN]
    daily_X_raw: np.ndarray       # float32 [num_days, DAY_LEN, F_raw]
    daily_X_obs: np.ndarray       # float32 [num_days, DAY_LEN, F_obs]

    # daily extras (optional)
    daily_ctx_raw: Optional[np.ndarray]     # float32 [num_days, F_ctx]
    daily_ctx_obs: Optional[np.ndarray]
    daily_seq7_raw: Optional[np.ndarray]   # float32 [num_days, 7, 3]
    daily_seq7_obs: Optional[np.ndarray]

    # helper for daily_features
    day_key_fn: Callable[[object], str]

    @staticmethod
    def from_frames(
        *,
        df_raw: pd.DataFrame,
        df_market: pd.DataFrame,
        is_future: bool,
        tz: str = DEFAULT_TZ,
        day_key_fn: Callable[[object], str] = _day_key_default,
        build_daily: bool = True,
    ) -> "MarketStore":
        if df_market is None or df_market.empty:
            raise ValueError("df_market is empty")

        if not isinstance(df_market.index, pd.DatetimeIndex):
            raise TypeError("df_market.index must be DatetimeIndex")

        # Decide DAY_LEN policy
        day_len = 345 if bool(is_future) else 1440

        # ---- per-row arrays (copy=True for stability) ----
        idx = df_market.index
        n_rows = int(len(df_market))

        # Required columns
        for col in ("day_id", "minute_index_t", "mask_t", "C_t"):
            if col not in df_market.columns:
                raise ValueError(f"df_market missing required column: {col}")

        # day_key normalize (object array, stable)
        day_ids = df_market["day_id"].to_numpy(copy=False)
        row_day_key = np.array([day_key_fn(x) for x in day_ids], dtype=object)

        # NEW: keep numeric day_id for metrics (stable copy)
        row_day_id = df_market["day_id"].to_numpy(dtype=np.int32, copy=True)

        # preserve day order by first appearance
        unique_days, first_idx = np.unique(row_day_key, return_index=True)
        order = np.argsort(first_idx)
        days = unique_days[order].astype(object)

        sid_to_dayi = {str(dk): int(i) for i, dk in enumerate(days)}

        # contiguous ranges by day (assumes day blocks are contiguous, which is true for strict futures reindex)
        # fallback: compute via boolean selection if non-contiguous (rare)
        day_ranges: List[Tuple[int, int]] = []
        row_day_i = np.empty((n_rows,), dtype=np.int32)
        for di, dk in enumerate(days):
            sel = (row_day_key == dk)
            if not np.any(sel):
                raise RuntimeError(f"day_key {dk} has no rows")
            s = int(np.argmax(sel))
            e = int(s + int(sel.sum()))
            day_ranges.append((s, e))
            row_day_i[s:e] = di

        row_minute = df_market["minute_index_t"].to_numpy(dtype=np.int32, copy=True)
        row_mask = df_market["mask_t"].to_numpy(dtype=np.float32, copy=True)

        # Optional low/high for stop loss
        row_C = df_market["C_t"].to_numpy(dtype=np.float32, copy=True)
        row_H = df_market.get("H_t", pd.Series(0.0, index=df_market.index)).to_numpy(dtype=np.float32, copy=True)
        row_L = df_market.get("L_t", pd.Series(0.0, index=df_market.index)).to_numpy(dtype=np.float32, copy=True)

        # ---- daily tensors ----
        F_raw = len(FEATURES_MARKET)
        F_obs = len(FEATURES_MARKET_OBS)

        num_days = int(len(days))
        daily_mask = np.zeros((num_days, day_len), dtype=np.float32)
        daily_X_raw = np.zeros((num_days, day_len, F_raw), dtype=np.float32)
        daily_X_obs = np.zeros((num_days, day_len, F_obs), dtype=np.float32)

        X_raw_all = df_market[FEATURES_MARKET].to_numpy(dtype=np.float32, copy=True)
        X_obs_all = df_market[FEATURES_MARKET_OBS].to_numpy(dtype=np.float32, copy=True)

        # Fill per-day blocks
        for di, (s, e) in enumerate(day_ranges):
            sub_len = int(e - s)

            if bool(is_future):
                # futures should be strictly day_len rows per session
                if sub_len != day_len:
                    # production: fail-fast (better than silent misalignment)
                    raise ValueError(f"futures day slice length={sub_len} != day_len={day_len} for day={days[di]}")
                # direct reshape is safe if aligned
                daily_X_raw[di, :, :] = X_raw_all[s:e, :]
                daily_X_obs[di, :, :] = X_obs_all[s:e, :]
                daily_mask[di, :] = row_mask[s:e]
            else:
                # FX: placeholder for later (you said暂不处理)
                # We still try a safe pack by minute_index into a day canvas.
                m = row_minute[s:e]
                m = np.clip(m, 0, day_len - 1)
                daily_X_raw[di, m, :] = X_raw_all[s:e, :]
                daily_X_obs[di, m, :] = X_obs_all[s:e, :]
                daily_mask[di, m] = row_mask[s:e]

        # ---- daily extras ----
        daily_ctx_raw = daily_ctx_obs = None
        daily_seq7_raw = daily_seq7_obs = None
        if build_daily:
            ctx_raw, ctx_obs, seq_raw, seq_obs, _summary = build_daily_context_and_seq(
                df_market,
                day_key_fn=day_key_fn,
                days_order=days,
                day_id_col="day_id",
                mask_col="mask_t",
            )
            daily_ctx_raw = ctx_raw.astype(np.float32, copy=False)
            daily_ctx_obs = ctx_obs.astype(np.float32, copy=False)
            daily_seq7_raw = seq_raw.astype(np.float32, copy=False)
            daily_seq7_obs = seq_obs.astype(np.float32, copy=False)

        # Make arrays contiguous (avoid surprises in downstream ops)
        daily_X_raw = np.ascontiguousarray(daily_X_raw)
        daily_X_obs = np.ascontiguousarray(daily_X_obs)
        daily_mask = np.ascontiguousarray(daily_mask)

        # NEW: keep global matrices contiguous as well
        X_raw_all = np.ascontiguousarray(X_raw_all)
        X_obs_all = np.ascontiguousarray(X_obs_all)

        return MarketStore(
            df_raw=df_raw,
            df_market=df_market,

            index=idx,
            n_rows=n_rows,
            day_len=day_len,
            tz=tz,

            row_mask=np.ascontiguousarray(row_mask),
            row_minute=np.ascontiguousarray(row_minute),
            row_day_key=row_day_key,
            row_day_i=np.ascontiguousarray(row_day_i),

            # NEW: day_id and global X matrices for env runtime
            row_day_id=np.ascontiguousarray(row_day_id),
            X_market_raw=X_raw_all,
            X_market_obs=X_obs_all,

            row_C=np.ascontiguousarray(row_C),
            row_H=np.ascontiguousarray(row_H),
            row_L=np.ascontiguousarray(row_L),

            days=days,
            sid_to_dayi=sid_to_dayi,
            day_ranges=day_ranges,

            daily_mask=daily_mask,
            daily_X_raw=daily_X_raw,
            daily_X_obs=daily_X_obs,

            daily_ctx_raw=daily_ctx_raw,
            daily_ctx_obs=daily_ctx_obs,
            daily_seq7_raw=daily_seq7_raw,
            daily_seq7_obs=daily_seq7_obs,

            day_key_fn=day_key_fn,
        )

    # --------- optional helpers (for env) ---------

    def candidate_start_rows_by_clock(self, start_clock: str) -> np.ndarray:
        """
        Returns row indices with mask==1 and local timestamp at given clock.
        Note: futures uses 21:01/09:01 anchors in your existing env logic.
        """
        if start_clock == "any":
            return np.flatnonzero(self.row_mask >= 0.5)

        idx_local = self.index
        if getattr(idx_local, "tz", None) is None:
            idx_local = idx_local.tz_localize(self.tz)
        else:
            idx_local = idx_local.tz_convert(self.tz)

        hours = idx_local.hour
        minutes = idx_local.minute

        valid_rows = np.flatnonzero(self.row_mask >= 0.5)

        def rows_at(hh: int, mm: int) -> np.ndarray:
            sel = np.flatnonzero((hours == hh) & (minutes == mm))
            return np.intersect1d(sel, valid_rows, assume_unique=False)

        if start_clock == "future_day":
            return rows_at(9, 1)

        if start_clock == "future_night":
            return rows_at(21, 1)

        if start_clock == "random_9_or_21":
            r9 = rows_at(9, 1)
            r21 = rows_at(21, 1)
            return np.concatenate([r9, r21]) if (r9.size + r21.size) > 0 else np.array([], dtype=int)

        # fallback
        return valid_rows

    def eod_idx_for_day(self, day_i: int) -> int:
        """Half-open end index for a day/session. For futures strict 345, it's day_ranges[day_i][1]."""
        return int(self.day_ranges[int(day_i)][1])
