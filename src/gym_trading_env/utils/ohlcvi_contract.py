# src/gym_trading_env/utils/ohlcvi_contract.py
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

REQUIRED = ["Open", "High", "Low", "Close"]
OPTIONAL = ["Volume", "OpenInterest"]


def normalize_ohlcvi(
    df: pd.DataFrame,
    *,
    rename_map: Optional[Dict[str, str]] = None,
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Normalize OHLCVI dataframe:
    - Accept either DatetimeIndex or a date column (default 'Date')
    - Optional rename_map applied first
    - Ensure required columns exist & are float
    - Ensure optional columns exist & are float (missing -> 0.0)
    - Sort by time, drop duplicated timestamps keep last
    """
    if df is None:
        raise ValueError("df is None")

    df = df.copy()

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Build index
    if not isinstance(df.index, pd.DatetimeIndex):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.set_index(date_col)
        else:
            raise TypeError("DataFrame must have DatetimeIndex or a date column (default 'Date').")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be DatetimeIndex after normalization")

    df = df[~df.index.isna()].copy()
    df = df.sort_index()

    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")].copy()

    # Required OHLC
    for c in REQUIRED:
        if c not in df.columns:
            raise ValueError(f"missing required col: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    # Optional VI
    for c in OPTIONAL:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
        else:
            df[c] = 0.0

    return df
