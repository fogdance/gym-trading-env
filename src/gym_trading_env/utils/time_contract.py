# src/gym_trading_env/utils/time_contract.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pandas as pd

from gym_trading_env.utils.timebase import FEATURE_TZ, DB_TZ


def from_db_naive_dt(x: Any) -> pd.Timestamp:
    """
    DB DATETIME(naive) -> tz-aware (FEATURE_TZ)
    - x can be datetime / str / pd.Timestamp
    - if x already tz-aware, just convert to FEATURE_TZ
    """
    ts = pd.Timestamp(x)
    if ts.tz is None:
        return ts.tz_localize(DB_TZ).tz_convert(FEATURE_TZ)
    return ts.tz_convert(FEATURE_TZ)


def from_db_naive_series(s: pd.Series) -> pd.Series:
    """
    Vectorized: DB naive datetime series -> tz-aware FEATURE_TZ series.
    Handles mixed inputs defensively.
    """
    ts = pd.to_datetime(s, errors="coerce")
    # ts is Series[datetime64] (naive) in most cases from pymysql
    try:
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(DB_TZ)
        ts = ts.dt.tz_convert(FEATURE_TZ)
        return ts
    except Exception:
        # fallback: elementwise (handles occasional tz-aware values)
        out = []
        for v in s:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                out.append(pd.NaT)
            else:
                out.append(from_db_naive_dt(v))
        return pd.Series(out, index=s.index)


def to_db_naive_dt(ts: Any) -> datetime:
    """
    tz-aware -> DB DATETIME(naive) in DB_TZ
    """
    t = pd.Timestamp(ts)
    if t.tz is None:
        raise ValueError("to_db_naive_dt requires tz-aware timestamp")
    return t.tz_convert(DB_TZ).tz_localize(None).to_pydatetime()


def ensure_feature_tz_index(
    idx: pd.DatetimeIndex,
    *,
    assume_tz: Optional[str] = None,
) -> pd.DatetimeIndex:
    """
    Ingestion-only helper:
    - If idx is naive, localize to assume_tz then convert to FEATURE_TZ.
    - If idx is tz-aware, convert to FEATURE_TZ.
    """
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError(f"idx must be pd.DatetimeIndex, got {type(idx)}")

    if idx.tz is None:
        if not assume_tz:
            raise ValueError("DatetimeIndex is naive. Provide assume_tz at ingestion.")
        return idx.tz_localize(assume_tz).tz_convert(FEATURE_TZ)

    return idx.tz_convert(FEATURE_TZ)
