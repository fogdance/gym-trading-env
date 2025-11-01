# utils/session_fx.py
import numpy as np
import pandas as pd
from typing import Dict

def _localize_index(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    """Ensure index is timezone-aware in tz."""
    if index.tz is None:
        return index.tz_localize(tz)
    return index.tz_convert(tz)

def compute_session_meta(df: pd.DataFrame,
                         tz: str = "Asia/Singapore",
                         rollover_hour_local: int = 5) -> Dict[str, pd.Series]:
    """
    Compute FX session metadata with rollover at `rollover_hour_local`.
    Returns Series aligned to df.index:
      - session_id: str YYYYMMDD per row
      - day_id: int factorized id for each session (fast numeric key)
      - minute_index: int [0..] within session
      - weekday_sin, weekday_cos: float cyclical weekday
      - prev_session_close: float previous session's close aligned
      - is_hard_gap: int 1 if gap > 60s, else 0
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex")

    idx_local = _localize_index(df.index, tz)

    hours = idx_local.hour
    dates = idx_local.date
    shifted_dates = np.where(
        hours < rollover_hour_local,
        pd.to_datetime(dates) - np.timedelta64(1, "D"),
        pd.to_datetime(dates)
    )
    shifted_dates = pd.to_datetime(shifted_dates)

    # String session id (readable, stable for groupby)
    session_id = pd.Series(shifted_dates.strftime("%Y%m%d"), index=df.index)

    # Numeric day id (fast, compact key for env tensors)
    day_id_values, _ = pd.factorize(session_id, sort=False)
    day_id = pd.Series(day_id_values.astype(np.int32), index=df.index)

    # Minute index within session
    minute_index = pd.Series(0, index=df.index, dtype=int)
    for sid, loc in session_id.groupby(session_id):
        minute_index.loc[loc.index] = np.arange(len(loc), dtype=int)

    # Weekday cyclical encoding
    weekday = pd.Series(shifted_dates.weekday, index=df.index).astype(int)
    weekday_sin = np.sin(2 * np.pi * (weekday / 7.0))
    weekday_cos = np.cos(2 * np.pi * (weekday / 7.0))

    # Previous session close mapped onto current session
    last_close = df["Close"].groupby(session_id).tail(1)
    by_sid_last_close = last_close.groupby(session_id[last_close.index]).first()
    prev_close_by_sid = by_sid_last_close.shift(1)
    prev_session_close = session_id.map(prev_close_by_sid).astype(float)
    # bootstrap first session with its first Close
    first_sid = session_id.iloc[0]
    first_val = float(df.loc[session_id == first_sid, "Close"].iloc[0])
    prev_session_close.loc[session_id == first_sid] = first_val

    # Hard gap detection
    delta = df.index.to_series().diff().dt.total_seconds().fillna(60.0)
    is_hard_gap = (delta > 60.0).astype(int)

    return {
        "session_id": session_id,
        "day_id": day_id,
        "minute_index": minute_index,
        "weekday_sin": weekday_sin,
        "weekday_cos": weekday_cos,
        "prev_session_close": prev_session_close,
        "is_hard_gap": is_hard_gap,
    }
