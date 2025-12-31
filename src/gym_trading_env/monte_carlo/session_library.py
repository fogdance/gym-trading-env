# src/gym_trading_env/monte_carlo/session_library.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from gym_trading_env.utils.timebase import FEATURE_TZ, ensure_index_tz_strict
from gym_trading_env.utils.session_futures_strict import strict_reindex_futures_345


_OHLCVI_COLS = ["Open", "High", "Low", "Close", "Volume", "OpenInterest"]


@dataclass(frozen=True)
class SessionLibrary:
    """
    A compact library of complete futures sessions (345 bars each).

    sessions: float32 [N, 345, 6] with columns [O,H,L,C,V,OI]
    session_id: object[str] [N]  format 'YYYYMMDD'
    tz: exchange timezone string
    """
    sessions: np.ndarray
    session_id: np.ndarray
    tz: str = FEATURE_TZ

    @property
    def n_sessions(self) -> int:
        return int(self.sessions.shape[0])

    @property
    def day_len(self) -> int:
        return int(self.sessions.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.sessions.shape[2])


def build_session_library(
    df_raw: pd.DataFrame,
    *,
    tz: str = FEATURE_TZ,
    require_complete: bool = True,
    min_valid_ratio: float = 1.0,
) -> SessionLibrary:
    """
    Build SessionLibrary from raw 1m OHLCVI.

    Key policy (per your requirement):
    - We do NOT patch missing bars inside trading clock.
    - If a session is not complete within 345 clock (mask ratio < min_valid_ratio), DROP it.

    Assumptions:
    - df_raw.index is tz-aware eob timestamps (or will error).
    - df_raw has at least Open/High/Low/Close. Volume/OI strongly recommended for futures.
    """
    if df_raw is None or df_raw.empty:
        raise ValueError("df_raw is empty")

    for c in ["Open", "High", "Low", "Close"]:
        if c not in df_raw.columns:
            raise ValueError(f"df_raw missing required col: {c}")

    # tz strict (ingestion should have done this already)
    _ = ensure_index_tz_strict(df_raw.index, target_tz=tz)

    aligned_pack = strict_reindex_futures_345(df_raw, tz=tz)
    aligned: pd.DataFrame = aligned_pack["aligned"]
    mask: pd.Series = aligned_pack["mask"]            # int 0/1 on 345 grid
    sid: pd.Series = aligned_pack["session_id"]       # 'YYYYMMDD' per row

    if aligned.empty:
        raise ValueError("strict_reindex_futures_345 returned empty aligned data")

    # Ensure all OHLCVI cols exist in aligned (strict_reindex already fills, but be defensive)
    for c in _OHLCVI_COLS:
        if c not in aligned.columns:
            aligned[c] = 0.0

    # Group by session_id in the canonical grid order (sort=False keeps first-appearance order)
    sessions_list = []
    sid_list = []

    for sid_str, loc in sid.groupby(sid, sort=False):
        idx = loc.index
        sub = aligned.loc[idx, _OHLCVI_COLS]
        sub_mask = mask.loc[idx].to_numpy(dtype=np.int8, copy=False)

        # Contract: should be exactly 345 rows per session in strict grid
        if require_complete and len(sub) != 345:
            continue

        valid_ratio = float(sub_mask.mean()) if len(sub_mask) > 0 else 0.0
        if valid_ratio < float(min_valid_ratio):
            # drop incomplete sessions
            continue

        # We require complete trading-clock bars: mask all ones
        if require_complete and not np.all(sub_mask == 1):
            continue

        # Basic sanity: OHLC must be finite
        arr = sub.to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(arr[:, 0:4]).all():
            continue

        sessions_list.append(arr)
        sid_list.append(str(sid_str))

    if not sessions_list:
        raise ValueError(
            "No complete sessions found. "
            "Either your raw data has gaps within trading clock, or trading_day inference mismatched."
        )

    sessions_np = np.stack(sessions_list, axis=0).astype(np.float32, copy=False)
    sid_np = np.array(sid_list, dtype=object)

    return SessionLibrary(sessions=sessions_np, session_id=sid_np, tz=tz)
