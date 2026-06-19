# src/gym_trading_env/utils/market_store.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gym_trading_env.utils.market_features import (
    FEATURES_MARKET,
    FEATURES_MARKET_OBS,
    FEATURES_RISK_CONTEXT,
    FEATURES_HTF_CONTEXT,
)
from gym_trading_env.utils.daily_features import build_daily_context_and_seq, DAILY_SEQ_LEN
from gym_trading_env.utils.timebase import FEATURE_TZ as DEFAULT_TZ
from gym_trading_env.utils.atr_util import compute_daily_atr_from_summary_talib

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

    实盘时：
      - 当 df_raw / df_market 更新时，不要 in-place 改这些 numpy，
        而是重新构建一个新的 MarketStore，再整体替换引用：
            new_store = MarketStore.rebuild(prev_store, df_raw=new_df_raw, df_market=new_df_market, is_future=True)
            bar_source.store = new_store
            env.store = new_store
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
    row_session_id: np.ndarray    # object [n_rows]  # "YYYYMMDD"
    row_trading_day: np.ndarray   # int32  [n_rows]  # YYYYMMDD

    row_C: np.ndarray             # float32 [n_rows]
    row_H: np.ndarray             # float32 [n_rows]
    row_L: np.ndarray             # float32 [n_rows]

    # NEW: global feature matrices (aligned with df_market row order)
    X_market_raw: np.ndarray      # float32 [n_rows, F_raw]  (FEATURES_MARKET)
    X_market_obs: np.ndarray      # float32 [n_rows, F_obs]  (FEATURES_MARKET_OBS)
    X_risk_context: np.ndarray    # float32 [n_rows, F_risk] (FEATURES_RISK_CONTEXT)
    X_htf_context: np.ndarray     # float32 [n_rows, F_htf]  (FEATURES_HTF_CONTEXT)

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

    atr_period_days: int

    daily_atr_price: Optional[np.ndarray]       # float32 [num_days] (shift(1))


    # ------------------------------------------------------------------ #
    #   初次构建：从 df_raw / df_market 生成一个新的 MarketStore
    # ------------------------------------------------------------------ #
    @staticmethod
    def from_frames(
        *,
        df_raw: pd.DataFrame,
        df_market: pd.DataFrame,
        is_future: bool,
        tz: str = DEFAULT_TZ,
        day_key_fn: Callable[[object], str] = _day_key_default,
        build_daily: bool = True,
        atr_period_days: int = 14,
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

        # Required columns: day_id, session_id, trading_day, minute_index_t, mask_t, C_t
        for col in ("day_id", "session_id", "trading_day", "minute_index_t", "mask_t", "C_t"):
            if col not in df_market.columns:
                raise ValueError(f"df_market missing required column: {col}")

        row_session_id = df_market["session_id"].astype(str).to_numpy(dtype=object, copy=True)
        row_trading_day = df_market["trading_day"].to_numpy(dtype=np.int32, copy=True)

        # day_key 用 session_id（稳定、可读、可查错）
        row_day_key = row_session_id

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
        X_risk_all = df_market[FEATURES_RISK_CONTEXT].to_numpy(dtype=np.float32, copy=True)
        X_htf_all = df_market[FEATURES_HTF_CONTEXT].to_numpy(dtype=np.float32, copy=True)

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
                # FX: placeholder for later（目前你没用 FX 实盘）
                # We still try a safe pack by minute_index into a day canvas.
                m = row_minute[s:e]
                m = np.clip(m, 0, day_len - 1)
                daily_X_raw[di, m, :] = X_raw_all[s:e, :]
                daily_X_obs[di, m, :] = X_obs_all[s:e, :]
                daily_mask[di, m] = row_mask[s:e]

        # ---- daily extras ----
        daily_ctx_raw = daily_ctx_obs = None
        daily_seq7_raw = daily_seq7_obs = None

        daily_atr_price = None

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


            atr_s = compute_daily_atr_from_summary_talib(
                _summary,
                period=int(atr_period_days),
                shift=1,
            )
            daily_atr_price = np.nan_to_num(
                atr_s.to_numpy(dtype=np.float32),
                nan=0.0, posinf=0.0, neginf=0.0
            )


        # NEW: keep global matrices contiguous as well
        X_raw_all = np.ascontiguousarray(X_raw_all)
        X_obs_all = np.ascontiguousarray(X_obs_all)
        X_risk_all = np.ascontiguousarray(X_risk_all)
        X_htf_all = np.ascontiguousarray(X_htf_all)

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
            row_session_id=np.ascontiguousarray(row_session_id),
            row_trading_day=np.ascontiguousarray(row_trading_day),
            X_market_raw=X_raw_all,
            X_market_obs=X_obs_all,
            X_risk_context=X_risk_all,
            X_htf_context=X_htf_all,

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

            atr_period_days=int(atr_period_days),
            daily_atr_price=daily_atr_price,
        )

    # ------------------------------------------------------------------ #
    #   实盘/增量场景：基于已有 store 的配置重建一个新的 store
    # ------------------------------------------------------------------ #
    @classmethod
    def rebuild(
        cls,
        prev_store: "MarketStore",
        *,
        df_raw: pd.DataFrame,
        df_market: pd.DataFrame,
        is_future: bool,
        build_daily: Optional[bool] = None,
        atr_period_days: int = 14,
    ) -> "MarketStore":
        """
        使用与 prev_store 相同的 tz / day_key_fn / daily 开关，重新构建一个新的 MarketStore。

        场景：
          - 实盘中 JuejinBarSource 从 DB 增量拉到了新的 1m bar，
            并已更新/重建出一份新的 df_raw + df_market
          - 希望用统一的方式生成新的 store，并一次性替换引用

        用法示例：
            new_store = MarketStore.rebuild(
                prev_store=self.store,
                df_raw=new_df_raw,
                df_market=new_df_market,
                is_future=True,
            )
            self.store = new_store
        """
        if build_daily is None:
            # 默认保持和原 store 一致：如果之前有 daily_ctx/seq，就继续构建；否则不构建
            has_daily = (prev_store.daily_ctx_raw is not None) or (prev_store.daily_seq7_raw is not None)
            build_daily = has_daily

        return cls.from_frames(
            df_raw=df_raw,
            df_market=df_market,
            is_future=is_future,
            tz=prev_store.tz,
            day_key_fn=prev_store.day_key_fn,
            build_daily=build_daily,
            atr_period_days=atr_period_days,
        )

    def inplace_overwrite_day_from_df_market(self, day_i: int, df_market_day: pd.DataFrame) -> None:
        """
        In-place overwrite a whole futures session block (strict 345 rows) into numpy tensors.

        Contract:
        - futures strict: len(df_market_day) must == self.day_len (345)
        - df_market_day must be aligned to the SAME index order as store.index slice [s:e)

        This mutates numpy arrays IN PLACE (store object remains the same).
        """
        di = int(day_i)
        s, e = self.day_ranges[di]
        sub_len = int(e - s)

        if int(len(df_market_day)) != sub_len:
            raise ValueError(f"inplace_overwrite_day: len(df_market_day)={len(df_market_day)} != slice_len={sub_len}")

        # --- per-row arrays ---
        self.row_mask[s:e] = df_market_day["mask_t"].to_numpy(dtype=np.float32, copy=False)
        self.row_minute[s:e] = df_market_day["minute_index_t"].to_numpy(dtype=np.int32, copy=False)
        self.row_C[s:e] = df_market_day["C_t"].to_numpy(dtype=np.float32, copy=False)

        if "H_t" in df_market_day.columns:
            self.row_H[s:e] = df_market_day["H_t"].to_numpy(dtype=np.float32, copy=False)
        if "L_t" in df_market_day.columns:
            self.row_L[s:e] = df_market_day["L_t"].to_numpy(dtype=np.float32, copy=False)

        # --- global feature matrices ---
        self.X_market_raw[s:e, :] = df_market_day[FEATURES_MARKET].to_numpy(dtype=np.float32, copy=False)
        self.X_market_obs[s:e, :] = df_market_day[FEATURES_MARKET_OBS].to_numpy(dtype=np.float32, copy=False)
        self.X_risk_context[s:e, :] = df_market_day[FEATURES_RISK_CONTEXT].to_numpy(dtype=np.float32, copy=False)
        self.X_htf_context[s:e, :] = df_market_day[FEATURES_HTF_CONTEXT].to_numpy(dtype=np.float32, copy=False)

        # --- daily tensors (futures strict reshape is safe) ---
        self.daily_mask[di, :] = self.row_mask[s:e]
        self.daily_X_raw[di, :, :] = self.X_market_raw[s:e, :]
        self.daily_X_obs[di, :, :] = self.X_market_obs[s:e, :]


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
            raise RuntimeError("MarketStore.index is naive. This is a bug: index must be tz-aware at ingestion.")
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
