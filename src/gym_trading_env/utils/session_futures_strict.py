# utils/session_futures_strict.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

DEFAULT_TZ = "Asia/Shanghai"

# 固定的 4 段交易时间（按“分钟收盘时刻”）
SEGMENTS: List[Tuple[str, str]] = [
    ("21:00", "23:00"),
    ("09:00", "10:15"),
    ("10:30", "11:30"),
    ("13:30", "15:00"),
]

def _localize_index(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.tz_localize(tz)
    return index.tz_convert(tz)

def _trading_day_for_timestamp(ts_local: pd.Timestamp) -> pd.Timestamp:
    """
    将真实时间戳归属到“交易日 T”：
    - 夜盘(21:00-23:59) 归入 “次日” 交易日
    - 凌晨(00:00-05:59) 归入 “前一日” 交易日
    - 其余归入当天
    返回一个带时区的 00:00:00 时间戳（仅日期有意义）
    """
    d = ts_local.normalize()
    h = ts_local.hour
    if 21 <= h <= 23:
        return (d + pd.Timedelta(days=1)).tz_localize(ts_local.tz) if d.tz is None else (d + pd.Timedelta(days=1))
    if 0 <= h <= 5:
        return (d - pd.Timedelta(days=1)).tz_localize(ts_local.tz) if d.tz is None else (d - pd.Timedelta(days=1))
    return d

def build_clock_for_trading_day(day: pd.Timestamp, tz: str) -> pd.DatetimeIndex:
    parts = []
    for start_str, end_str in SEGMENTS:
        s = pd.Timestamp(f"{day.date()} {start_str}", tz=tz)
        e = pd.Timestamp(f"{day.date()} {end_str}", tz=tz)
        rng = pd.date_range(start=s, end=e, freq="1min", inclusive="left", tz=tz) + pd.Timedelta(minutes=1)
        parts.append(rng)
    # 关键：不要用 .values（会丢 tz）
    clock = parts[0]
    for p in parts[1:]:
        clock = clock.append(p)  # pandas>=2.2 可用 union_many / concat 其他方式避免弃用
    assert len(clock) == 345
    return clock


def strict_reindex_futures_345(
    df_1m: pd.DataFrame,
    tz: str = DEFAULT_TZ,
) -> Dict[str, pd.DataFrame]:
    """
    固定造“345 分钟收盘时钟”（夜盘 + 三段日盘），不管有没有数据，时点都存在。
    把传进来的 df_1m 直接按时点对齐上去（有就填 OHLCV/OI，没有就 0），
    再产出 mask_t / session_id / minute_index / day_id。

    仅需要原始列：Open, High, Low, Close, Volume(可缺省), OpenInterest(可缺省)
    """
    # ---- 基本校验 ----
    if not isinstance(df_1m.index, pd.DatetimeIndex):
        raise ValueError("df_1m.index must be a DatetimeIndex")
    if not {"Open","High","Low","Close"}.issubset(df_1m.columns):
        raise ValueError("df_1m must contain ['Open','High','Low','Close']")

    # 复制并统一时区
    df = df_1m.copy()
    idx_local = _localize_index(df.index, tz)

    # 用本地化后的索引来计算交易日，确保索引对齐
    trading_day_series = pd.Series([_trading_day_for_timestamp(ts) for ts in idx_local],
                                index=idx_local)

    # 计算 canonical_index（tz-aware），并立刻用它作为 df 的索引
    time_of_day = (idx_local - idx_local.normalize())
    trading_day_idx = pd.DatetimeIndex(trading_day_series)
    if trading_day_idx.tz is None:
        trading_day_idx = trading_day_idx.tz_localize(tz)
    else:
        trading_day_idx = trading_day_idx.tz_convert(tz)

    canonical_index = trading_day_idx.normalize() + time_of_day

    df_can = df.copy()
    df_can.index = canonical_index

    # 现在基于 df_can.index（即 canonical_index）来生成 session_id_series（保证索引对齐）
    session_id_series = pd.Series(trading_day_idx.strftime("%Y%m%d"),
                                index=df_can.index,
                                name="session_id")


    # 规范化索引：用“交易日的日期 + 原始时分秒”
    time_of_day = (idx_local - idx_local.normalize())  # TimedeltaIndex（时分秒）
    canonical_index = trading_day_idx.normalize() + time_of_day  # tz-aware
    df_can = df.copy()
    df_can.index = canonical_index

    # 若原始为空，直接返回空结构
    if df_can.empty:
        return {
            "aligned": df_can,
            "mask": pd.Series([], dtype=int),
            "session_id": pd.Series([], dtype=str),
            "minute_index": pd.Series([], dtype=int),
            "day_id": pd.Series([], dtype=np.int32),
        }

    # ---- 用 session_id 分组，逐日造 clock 并对齐 ----
    out_frames: List[pd.DataFrame] = []
    out_masks: List[pd.Series] = []
    out_sid:   List[pd.Series] = []
    out_minidx:List[pd.Series] = []


    for sid_str, loc in session_id_series.groupby(session_id_series):
        # 该交易日的 YYYYMMDD 和其对应的日期对象
        day_date = pd.to_datetime(sid_str, format="%Y%m%d").date()
        day_ts = pd.Timestamp(day_date, tz=tz)

        # 固定 345 收盘时钟
        clock = build_clock_for_trading_day(day_ts, tz)

        # 取该交易日的原始行（按 loc 的 index 直接截取）
        df_day = df_can.loc[loc.index].copy()

        # 按收盘时钟 reindex：有则命中，无则 NaN → 下面补 0
        df_aligned = df_day.reindex(clock)

        # 掩码：原始有数据（Close 非空）=1
        has_data = df_aligned["Close"].notna().astype(int)

        # 缺失分钟补 0（OHLCV/OI）
        for col in ["Open","High","Low","Close","Volume","OpenInterest"]:
            if col in df_aligned.columns:
                df_aligned[col] = df_aligned[col].fillna(0.0).astype(float)
            else:
                df_aligned[col] = 0.0

        # 固定元数据
        minute_index = pd.Series(np.arange(len(clock), dtype=int), index=clock, name="minute_index")
        sid_full = pd.Series(sid_str, index=clock, name="session_id")

        out_frames.append(df_aligned)
        out_masks.append(has_data.rename("mask_t"))
        out_sid.append(sid_full)
        out_minidx.append(minute_index)

    # ---- 拼接所有交易日 ----
    aligned = pd.concat(out_frames)
    mask = pd.concat(out_masks).astype(int)
    sid_all = pd.concat(out_sid).astype(str)
    minidx_all = pd.concat(out_minidx).astype(int)

    # day_id（factorize）
    day_id_values, _ = pd.factorize(sid_all, sort=False)
    day_id = pd.Series(day_id_values.astype(np.int32), index=aligned.index, name="day_id")

    # 确保必需列存在（Volume/OpenInterest 可缺省）
    for col in ["Open","High","Low","Close","Volume","OpenInterest"]:
        if col not in aligned.columns:
            aligned[col] = 0.0

    return {
        "aligned": aligned,
        "mask": mask,
        "session_id": sid_all,
        "minute_index": minidx_all,
        "day_id": day_id,
    }
