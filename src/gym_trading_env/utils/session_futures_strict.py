# utils/session_futures_strict.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from gym_trading_env.utils.timebase import FEATURE_TZ as DEFAULT_TZ, ensure_index_tz_strict

# 固定的 4 段交易时间（按“分钟收盘时刻”）
SEGMENTS: List[Tuple[str, str]] = [
    ("21:00", "23:00"),   # 夜盘（可能不存在 -> mask=0）
    ("09:00", "10:15"),
    ("10:30", "11:30"),
    ("13:30", "15:00"),
]


def _infer_trading_days(idx_local: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    从数据里推断“交易日集合”（用日盘时间段出现的日期作为交易日）。
    这样周末/法定假期会自动被跳过（因为数据里没有日盘）。
    """
    hours = idx_local.hour
    day_mask = (hours >= 6) & (hours < 18)  # 覆盖 09:00-15:00
    days = pd.DatetimeIndex(idx_local[day_mask].normalize().unique())
    if len(days) == 0:
        days = pd.DatetimeIndex(idx_local.normalize().unique())
    return days.sort_values()

def _next_trading_day(d0: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    """返回 trading_days 中严格大于 d0 的最早日期；找不到则 d0+1D。"""
    if len(trading_days) == 0:
        return d0 + pd.Timedelta(days=1)
    pos = int(np.searchsorted(trading_days.asi8, d0.value, side="right"))
    if pos >= len(trading_days):
        return d0 + pd.Timedelta(days=1)
    return pd.Timestamp(trading_days[pos])

def _prev_trading_day(d0: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    """返回 trading_days 中严格小于 d0 的最晚日期；找不到则 d0-1D。"""
    if len(trading_days) == 0:
        return d0 - pd.Timedelta(days=1)
    pos = int(np.searchsorted(trading_days.asi8, d0.value, side="left")) - 1
    if pos < 0:
        return d0 - pd.Timedelta(days=1)
    return pd.Timestamp(trading_days[pos])

def _trading_day_for_ts(ts_local: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    """
    将真实时间戳归属到“交易日 T”（T 用日盘日期表示）：
    - 夜盘(21:00-23:59) 归入 “下一交易日”（自动跳过周末/假期）
    - 凌晨(00:00-05:59) 归入 “上一交易日”
    - 其余归入当天
    """
    d0 = ts_local.normalize()
    h = ts_local.hour
    if 21 <= h <= 23:
        return _next_trading_day(d0, trading_days)
    if 0 <= h <= 5:
        return _prev_trading_day(d0, trading_days)
    return d0

def build_clock_for_trading_day(
    trading_day: pd.Timestamp,
    tz: str,
    night_date: Optional[pd.Timestamp] = None,
) -> pd.DatetimeIndex:
    """
    造一个 session 的 345 “分钟收盘时钟”：
    - 夜盘段用 night_date 的日期（如果该交易日没有夜盘数据，则 night_date 默认为 trading_day-1 的自然日）
    - 日盘段用 trading_day 的日期
    """
    trading_day = pd.Timestamp(trading_day).tz_convert(tz) if pd.Timestamp(trading_day).tzinfo else pd.Timestamp(trading_day, tz=tz)
    trading_day = trading_day.normalize()

    if night_date is None:
        night_date = (trading_day - pd.Timedelta(days=1)).normalize()
    else:
        night_date = pd.Timestamp(night_date).tz_convert(tz) if pd.Timestamp(night_date).tzinfo else pd.Timestamp(night_date, tz=tz)
        night_date = night_date.normalize()

    parts = []
    for i, (start_str, end_str) in enumerate(SEGMENTS):
        base = night_date if i == 0 else trading_day
        s = pd.Timestamp(f"{base.date()} {start_str}", tz=tz)
        e = pd.Timestamp(f"{base.date()} {end_str}", tz=tz)
        rng = pd.date_range(start=s, end=e, freq="1min", inclusive="left", tz=tz) + pd.Timedelta(minutes=1)
        parts.append(rng)

    clock = parts[0]
    for p in parts[1:]:
        clock = clock.append(p)

    assert len(clock) == 345
    return clock

def strict_reindex_futures_345(
    df_1m: pd.DataFrame,
    tz: str = DEFAULT_TZ,
) -> Dict[str, pd.DataFrame]:
    """
    固定造“345 分钟收盘时钟”（夜盘 + 三段日盘），不管有没有数据，时点都存在；
    然后把原始 df 按 session 对齐上去（有就填 OHLCV/OI，没有就 0），并产出 mask/session_id/minute_index/day_id。

    关键改动：
    - 夜盘归属“下一交易日”，且“下一交易日”来自数据推断的交易日集合（自动处理：周五夜盘 -> 周一）
    - 对齐索引用真实时间戳（不再把夜盘硬 shift 到同一天），从而支持长假“无夜盘”（对应分钟 mask=0）
    """
    if not isinstance(df_1m.index, pd.DatetimeIndex):
        raise ValueError("df_1m.index must be a DatetimeIndex")
    if not {"Open", "High", "Low", "Close"}.issubset(df_1m.columns):
        raise ValueError("df_1m must contain ['Open','High','Low','Close']")

    df = df_1m.copy()
    idx_local = ensure_index_tz_strict(df.index, target_tz=tz)
    df.index = idx_local

    if df.empty:
        return {
            "aligned": df,
            "mask": pd.Series([], dtype=int),
            "session_id": pd.Series([], dtype=str),
            "minute_index": pd.Series([], dtype=int),
            "day_id": pd.Series([], dtype=np.int32),
        }

    # 分流：DB 有 trading_day -> 直接用；CSV 没有 -> 走原推断
    if "trading_day" in df.columns:
        td = pd.to_numeric(df["trading_day"], errors="coerce").astype("Int64")
        if td.isna().all():
            raise ValueError("df_1m.trading_day exists but all NaN")

        # session_id_series：每条原始 bar 的 session_id（YYYYMMDD 字符串）
        session_id_series = td.astype(str)
        session_id_series.index = df.index
        session_id_series.name = "session_id"
    else:
        trading_days = _infer_trading_days(idx_local)
        td_list = [_trading_day_for_ts(ts, trading_days) for ts in idx_local]
        trading_day_idx = pd.DatetimeIndex(td_list)
        session_id_series = pd.Series(trading_day_idx.strftime("%Y%m%d"), index=df.index, name="session_id")


    out_frames: List[pd.DataFrame] = []
    out_masks: List[pd.Series] = []
    out_sid: List[pd.Series] = []
    out_minidx: List[pd.Series] = []

    for sid_str, loc in session_id_series.groupby(session_id_series, sort=False):
        day_date = pd.to_datetime(sid_str, format="%Y%m%d").date()
        day_ts = pd.Timestamp(day_date, tz=tz)

        df_day = df.loc[loc.index].copy()

        # 该 session 是否真的有夜盘数据？有的话用真实夜盘日期；否则使用 day-1 的自然日（长假/停夜盘会全 mask=0）
        night_rows = df_day.index[df_day.index.hour >= 21]
        if len(night_rows) > 0:
            night_date = pd.Timestamp(night_rows.min()).normalize()
        else:
            night_date = (day_ts - pd.Timedelta(days=1)).normalize()

        clock = build_clock_for_trading_day(day_ts, tz=tz, night_date=night_date)

        df_aligned = df_day.reindex(clock)
        has_data = df_aligned["Close"].notna().astype(int)  # 在 fillna 前算 mask

        for col in ["Open", "High", "Low", "Close", "Volume", "OpenInterest"]:
            if col in df_aligned.columns:
                df_aligned[col] = df_aligned[col].fillna(0.0).astype(float)
            else:
                df_aligned[col] = 0.0

        minute_index = pd.Series(np.arange(len(clock), dtype=int), index=clock, name="minute_index")
        sid_full = pd.Series(sid_str, index=clock, name="session_id")

        out_frames.append(df_aligned)
        out_masks.append(has_data.rename("mask_t"))
        out_sid.append(sid_full)
        out_minidx.append(minute_index)

    aligned = pd.concat(out_frames)
    mask = pd.concat(out_masks).astype(int)
    sid_all = pd.concat(out_sid).astype(str)
    minidx_all = pd.concat(out_minidx).astype(int)

    day_id_values, _ = pd.factorize(sid_all, sort=False)
    day_id = pd.Series(day_id_values.astype(np.int32), index=aligned.index, name="day_id")

    for col in ["Open", "High", "Low", "Close", "Volume", "OpenInterest"]:
        if col not in aligned.columns:
            aligned[col] = 0.0

    return {
        "aligned": aligned,
        "mask": mask,
        "session_id": sid_all,
        "minute_index": minidx_all,
        "day_id": day_id,
    }
