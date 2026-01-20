# utils/validate_juejin_minutes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gym_trading_env.utils.timebase import FEATURE_TZ as DEFAULT_TZ, ensure_index_tz_strict
from utils.session_futures_strict import build_clock_for_trading_day, SEGMENTS


@dataclass(frozen=True)
class MissingBarsError(RuntimeError):
    """
    Raised when expected 1m bars are missing during required trading minutes.

    Attributes:
      trading_day: int YYYYMMDD
      missing: tz-aware DatetimeIndex of missing eob timestamps
      required_count: number of required minutes
      actual_count: number of minutes present in raw DB slice
      has_night: whether this trading_day slice contains any eob with hour>=21
      note: extra debug info
    """
    trading_day: int
    missing: pd.DatetimeIndex
    required_count: int
    actual_count: int
    has_night: bool
    note: str = ""

    def __str__(self) -> str:
        head = [str(x) for x in self.missing[:20]]
        return (
            f"[JuejinBarSource] Missing 1m bars during required trading minutes: "
            f"trading_day={self.trading_day}, missing_count={len(self.missing)}, "
            f"required_count={self.required_count}, actual_count={self.actual_count}, "
            f"has_night={self.has_night}, missing_head={head}. {self.note}"
        )


def _normalize_trading_day_series(
    s: pd.Series,
    *,
    name: str = "trading_day",
) -> pd.Series:
    """
    Normalize trading_day column into int YYYYMMDD, aligned with df.index.
    Accepts:
      - int/Int64 YYYYMMDD
      - datetime/date-like (will format to YYYYMMDD)
      - str "YYYY-MM-DD" / "YYYYMMDD"
    """
    if pd.api.types.is_integer_dtype(s.dtype):
        out = pd.to_numeric(s, errors="coerce").astype("Int64")
    else:
        # Try parse datetime-like then format
        dt = pd.to_datetime(s, errors="coerce")
        if dt.notna().any():
            out = dt.dt.strftime("%Y%m%d").astype("Int64")
        else:
            out = pd.to_numeric(s.astype(str).str.replace("-", "", regex=False), errors="coerce").astype("Int64")

    if out.isna().any():
        bad_n = int(out.isna().sum())
        raise ValueError(f"{name} has {bad_n} NaN after normalization; cannot validate completeness.")
    return out.astype(np.int32)


def _segment_lengths_minutes(segments: Sequence[Tuple[str, str]]) -> List[int]:
    """
    Each segment is [start,end) in minutes, but clock built by build_clock_for_trading_day
    is eob timestamps, generated as:
      date_range(start=s, end=e, freq=1min, inclusive='left') + 1min
    So the count is (end-start) minutes.
    """
    out: List[int] = []
    for start_str, end_str in segments:
        s = pd.Timestamp(f"2000-01-01 {start_str}")
        e = pd.Timestamp(f"2000-01-01 {end_str}")
        mins = int((e - s) / pd.Timedelta(minutes=1))
        if mins <= 0:
            raise ValueError(f"Invalid segment {start_str}-{end_str}: non-positive length.")
        out.append(mins)
    return out


def validate_juejin_minutes_complete(
    df_raw: pd.DataFrame,
    *,
    tz: str = DEFAULT_TZ,
    trading_day_col: str = "trading_day",
    segments: Sequence[Tuple[str, str]] = SEGMENTS,
    allow_full_missing_night: bool = True,
    # if True: when "no night bars at all" for that trading_day, we allow the whole night segment to be missing
    # and only require day segments
    require_day_segments: bool = True,
    # if False: you can choose to validate nothing (not recommended); kept for flexibility
    raise_on_duplicates: bool = True,
    # if duplicates exist, either raise (recommended) or keep last
    max_report: int = 200,
) -> None:
    """
    Production-grade completeness validator for JuejinBarSource raw 1m bars.

    Contract:
    - df_raw.index is eob timestamps (tz-aware or convertible), 1-minute frequency source.
    - df_raw contains a trading day column per row (YYYYMMDD or date-like).
    - For each trading_day session, required eob timestamps must exist; otherwise raise MissingBarsError.

    Policy:
    - Day segments are always required when require_day_segments=True.
    - Night segment requirement:
        - If allow_full_missing_night=True:
            * If this trading_day has ANY night eob rows (hour>=21), then night must be complete.
            * If this trading_day has NO night rows at all, then the entire night segment may be missing
              (often due to holiday / exchange schedule / disabled night trading).
        - If allow_full_missing_night=False:
            * Night segment is always required (strict).

    Raises:
      - ValueError for schema/time issues
      - MissingBarsError for missing required minutes
    """
    if df_raw is None or not isinstance(df_raw, pd.DataFrame):
        raise TypeError("df_raw must be a pandas DataFrame")
    if df_raw.empty:
        raise ValueError("df_raw is empty; cannot validate completeness")
    if trading_day_col not in df_raw.columns:
        raise ValueError(f"df_raw missing required column: {trading_day_col}")

    # ---- normalize index to tz-aware local (FEATURE_TZ) ----
    if not isinstance(df_raw.index, pd.DatetimeIndex):
        raise TypeError("df_raw.index must be a DatetimeIndex (eob timestamps)")

    df = df_raw.copy()
    df.index = ensure_index_tz_strict(df.index, target_tz=tz)
    df = df.sort_index()

    # ---- handle duplicates ----
    if df.index.has_duplicates:
        if raise_on_duplicates:
            dup = df.index[df.index.duplicated(keep=False)]
            head = [str(x) for x in dup[:20]]
            raise ValueError(f"df_raw index has duplicated eob timestamps (count={len(dup)}). head={head}")
        df = df[~df.index.duplicated(keep="last")]

    # ---- normalize trading_day series ----
    td = _normalize_trading_day_series(df[trading_day_col], name=trading_day_col)
    td.index = df.index  # align

    # ---- precompute segment lengths & day-only cutoff ----
    seg_lens = _segment_lengths_minutes(segments)
    total_len = int(sum(seg_lens))
    if total_len <= 0:
        raise ValueError("segments total length must be positive")

    night_len = int(seg_lens[0]) if len(seg_lens) > 0 else 0
    if night_len <= 0:
        raise ValueError("night segment length computed as <=0; check SEGMENTS")
    if total_len != 345 and segments is SEGMENTS:
        # Not fatal, but your strict futures contract expects 345
        # Keep as ValueError to avoid silent mismatch in production.
        raise ValueError(f"Expected total session length=345 minutes, got {total_len}. Check SEGMENTS.")

    # ---- group by trading_day and validate ----
    # NOTE: td is int YYYYMMDD; grouping preserves order by appearance (sort=False)
    for td_int, idx_group in td.groupby(td, sort=False).groups.items():
        td_int = int(td_int)
        sub_idx = pd.DatetimeIndex(idx_group)
        sub_idx = sub_idx.sort_values()

        # Determine session trading date (day segments use this date)
        day_ts = pd.Timestamp(str(td_int), format="%Y%m%d", tz=tz).normalize()

        # Night presence for this trading_day slice (based on actual DB rows)
        has_night = bool(np.any(sub_idx.hour >= 21))

        # Choose night_date for expected clock:
        # - If has night data, use the real night calendar date of earliest night row.
        # - Else default to natural day-1 (so expected clock still builds deterministically).
        if has_night:
            night_date = pd.Timestamp(sub_idx[sub_idx.hour >= 21].min()).normalize()
        else:
            night_date = (day_ts - pd.Timedelta(days=1)).normalize()

        clock = build_clock_for_trading_day(day_ts, tz=tz, night_date=night_date)
        if len(clock) != total_len:
            raise ValueError(
                f"Clock length mismatch for trading_day={td_int}: len(clock)={len(clock)} != total_len={total_len}"
            )

        # Required minutes selection
        if not require_day_segments:
            # Nothing required (not recommended); skip
            continue

        if allow_full_missing_night and (not has_night):
            # Require day segments only => drop first (night) block
            required = clock[night_len:]
        else:
            required = clock

        # Missing required timestamps
        # Use set difference on DatetimeIndex (fast enough for 345)
        missing = required.difference(sub_idx)

        if len(missing) > 0:
            # Clip for message size but keep full missing in exception field
            note = ""
            if len(missing) > max_report:
                note = f"missing list truncated in message to first {max_report}; see exception.missing for full."
            raise MissingBarsError(
                trading_day=td_int,
                missing=missing,
                required_count=int(len(required)),
                actual_count=int(len(sub_idx)),
                has_night=has_night,
                note=note,
            )
