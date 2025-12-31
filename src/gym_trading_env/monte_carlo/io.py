# src/gym_trading_env/monte_carlo/io.py
from __future__ import annotations

from typing import Union
from pathlib import Path
import numpy as np
import pandas as pd

from gym_trading_env.monte_carlo.session_library import SessionLibrary
from gym_trading_env.utils.timebase import FEATURE_TZ


def save_session_library_npz(lib: SessionLibrary, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        sessions=lib.sessions.astype(np.float32, copy=False),
        session_id=lib.session_id.astype(object),
        tz=np.array([lib.tz], dtype=object),
    )


def load_session_library_npz(path: Union[str, Path]) -> SessionLibrary:
    z = np.load(str(path), allow_pickle=True)
    tz = str(z["tz"][0]) if "tz" in z else FEATURE_TZ
    return SessionLibrary(
        sessions=z["sessions"].astype(np.float32, copy=False),
        session_id=z["session_id"].astype(object),
        tz=tz,
    )


def write_synth_csv(
    df_raw: pd.DataFrame,
    out_path: Union[str, Path],
    *,
    tz: str = FEATURE_TZ,
    include_trading_day: bool = False,
) -> None:
    """
    Write CSV in STRICT format:

    Default (as you required):
      Date,Open,High,Low,Close,Volume,OpenInterest
      2025-06-09 09:01:00,792.5,793.5,789.0,792.5,1010,93619

    Notes:
    - Date is local exchange time string (no timezone suffix).
    - Volume/OpenInterest are written as integers (no .0).
    - trading_day can be optionally appended at the end.
    """
    if df_raw is None or df_raw.empty:
        raise ValueError("df_raw is empty")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    need_cols = ["Open", "High", "Low", "Close", "Volume", "OpenInterest"]
    for c in need_cols:
        if c not in df_raw.columns:
            raise ValueError(f"df_raw missing required col: {c}")

    df = df_raw.copy()

    # 1) build Date column from index (tz-aware -> local naive -> string)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(tz).tz_localize(None)
    df.insert(0, "Date", idx.strftime("%Y-%m-%d %H:%M:%S"))

    # 2) enforce Volume/OI integer output
    for c in ["Volume", "OpenInterest"]:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.isna().any():
            # 这里不应该发生（你要求 session 完整且 OHLCVI 全量），直接报错更安全
            bad = int(s.isna().sum())
            raise ValueError(f"{c} has NaN after numeric cast: {bad} rows")
        df[c] = s.round().astype("int64")

    # 3) strict column order
    cols = ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInterest"]
    if include_trading_day and "trading_day" in df.columns:
        cols.append("trading_day")

    df = df[cols]

    # 4) write without index (because Date is already a column)
    df.to_csv(out_path, index=False)
