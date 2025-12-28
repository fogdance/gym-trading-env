# src/gym_trading_env/utils/timebase.py
from __future__ import annotations
import pandas as pd
from typing import Optional
from datetime import datetime

EXCHANGE_TZ = "Asia/Shanghai"
FEATURE_TZ  = EXCHANGE_TZ
DB_TZ       = EXCHANGE_TZ

def ensure_index_tz_strict(idx: pd.DatetimeIndex, target_tz: str = FEATURE_TZ) -> pd.DatetimeIndex:
    """STRICT: idx 必须 tz-aware，否则直接报错。"""
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError(f"idx must be pd.DatetimeIndex, got {type(idx)}")
    if idx.tz is None:
        raise ValueError("DatetimeIndex is naive. Localize it in ingestion layer (BarSource/Downloader) explicitly.")
    return idx.tz_convert(target_tz)

def localize_index_explicit(
    idx: pd.DatetimeIndex,
    assume_tz: str,
    target_tz: str = FEATURE_TZ,
) -> pd.DatetimeIndex:
    """EXPLICIT: 只有 ingestion 层可用。"""
    if idx.tz is None:
        return idx.tz_localize(assume_tz).tz_convert(target_tz)
    return idx.tz_convert(target_tz)

def ts_from_db_naive(x, db_tz: str = DB_TZ) -> pd.Timestamp:
    """EXPLICIT: DB DATETIME(naive) -> tz-aware (db_tz)."""
    ts = pd.Timestamp(x)
    if ts.tz is not None:
        return ts.tz_convert(db_tz)
    return ts.tz_localize(db_tz)

def to_db_naive_dt_strict(x: pd.Timestamp, db_tz: str = DB_TZ) -> datetime:
    """STRICT: 输入必须 tz-aware（通常来自 store/index），再转成 DB naive。"""
    ts = pd.Timestamp(x)
    if ts.tz is None:
        raise ValueError("to_db_naive_dt_strict requires tz-aware timestamp.")
    return ts.tz_convert(db_tz).tz_localize(None).to_pydatetime()

def ts_to_naive_str(ts, tz: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    t = pd.Timestamp(ts)
    if t.tz is None:
        raise ValueError("ts is naive; must be tz-aware before serialization")
    t = t.tz_convert(tz)
    return t.tz_localize(None).strftime(fmt)

def yyyymmdd_int(d) -> int:
    """date/datetime/Timestamp -> YYYYMMDD int"""
    return int(pd.Timestamp(d).strftime("%Y%m%d"))

def to_local_ts_strict(x, target_tz: str = FEATURE_TZ) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tz is None:
        raise ValueError("to_local_ts_strict: got naive ts; localize explicitly in ingestion layer.")
    return ts.tz_convert(target_tz)
