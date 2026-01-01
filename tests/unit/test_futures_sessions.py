# tests/unit/test_futures_sessions.py

import pandas as pd
import pytest
from types import SimpleNamespace

pytestmark = pytest.mark.unit

from gym_trading_env.utils.timebase import FEATURE_TZ as DEFAULT_TZ
from gym_trading_env.utils.bar_source import CsvBarSource


def _mk_cfg_future():
    """
    Minimal config stub for CsvBarSource.
    CsvBarSource._build() only reads fields under config.trading.
    """
    return SimpleNamespace(
        trading=SimpleNamespace(
            data_path="",
            data_interval="1m",
            data_tz=str(DEFAULT_TZ),  # assume tz for naive Date values
            is_future=True,
            use_daily_context=False,
            use_daily_seq_7=False,
        ),
        training=SimpleNamespace(),
    )


def _make_ohlc_df(idx):
    """
    CsvBarSource.normalize_ohlcvi(..., date_col="Date") expects a Date column.
    Close non-NaN => mask=1 for those input rows; synthetic rows will be mask=0.
    """
    dt = pd.to_datetime(idx)
    n = len(dt)
    return pd.DataFrame(
        {
            "Date": dt,
            "Open": [100.0] * n,
            "High": [101.0] * n,
            "Low": [99.0] * n,
            "Close": [100.0] * n,
            "Volume": [1.0] * n,
            "OpenInterest": [0.0] * n,
        }
    )


def test_weekend_friday_night_belongs_to_monday_trading_day():
    """
    场景1：周五夜盘 + 周一日盘 属于同一交易日（周一）。
    """
    idx = [
        "2025-09-19 22:44:00",  # Fri night
        "2025-09-19 23:00:00",  # Fri night end
        "2025-09-22 09:01:00",  # Mon day open
        "2025-09-22 09:02:00",
    ]
    df = _make_ohlc_df(idx)

    bs = CsvBarSource(config=_mk_cfg_future(), df=df)
    store = bs.store

    t_fri_2244 = pd.Timestamp("2025-09-19 22:44:00", tz=DEFAULT_TZ)
    t_mon_0901 = pd.Timestamp("2025-09-22 09:01:00", tz=DEFAULT_TZ)

    assert t_fri_2244 in store.index
    assert t_mon_0901 in store.index

    i_fri = int(store.index.get_loc(t_fri_2244))
    i_mon = int(store.index.get_loc(t_mon_0901))

    # 核心：两者 trading_day 同为周一 20250922
    assert int(store.row_trading_day[i_fri]) == 20250922
    assert int(store.row_trading_day[i_mon]) == 20250922

    # 且两者都是真实数据命中（mask=1）
    assert int(store.row_mask[i_fri] >= 0.5) == 1
    assert int(store.row_mask[i_mon] >= 0.5) == 1

    # minute_index 顺序：夜盘120分钟在前 -> 09:01 应是 120
    assert int(store.row_minute[i_mon]) == 120


def test_normal_night_belongs_to_next_day_trading_day():
    """
    场景2：普通工作日夜盘 + 次日日盘，同一交易日（次日）。
    """
    idx = [
        "2025-09-18 22:00:00",  # Thu night
        "2025-09-19 09:01:00",  # Fri day
        "2025-09-19 09:02:00",
    ]
    df = _make_ohlc_df(idx)

    bs = CsvBarSource(config=_mk_cfg_future(), df=df)
    store = bs.store

    t_thu_2200 = pd.Timestamp("2025-09-18 22:00:00", tz=DEFAULT_TZ)
    t_fri_0901 = pd.Timestamp("2025-09-19 09:01:00", tz=DEFAULT_TZ)

    assert t_thu_2200 in store.index
    assert t_fri_0901 in store.index

    i_thu = int(store.index.get_loc(t_thu_2200))
    i_fri = int(store.index.get_loc(t_fri_0901))

    assert int(store.row_trading_day[i_thu]) == 20250919
    assert int(store.row_trading_day[i_fri]) == 20250919
    assert int(store.row_mask[i_thu] >= 0.5) == 1
    assert int(store.row_mask[i_fri] >= 0.5) == 1


def test_holiday_may_have_no_night_session_minutes_masked():
    """
    场景3：国庆长假后复市，可能没有“前一晚夜盘数据”。
    我们仍然生成该交易日的夜盘时段点位，但全部 mask=0。
    """
    idx = [
        "2025-09-30 15:00:00",  # 假期前最后日盘收盘
        "2025-10-09 09:01:00",  # 假期后第一天日盘
        "2025-10-09 09:02:00",
    ]
    df = _make_ohlc_df(idx)

    bs = CsvBarSource(config=_mk_cfg_future(), df=df)
    store = bs.store

    t_0930_1500 = pd.Timestamp("2025-09-30 15:00:00", tz=DEFAULT_TZ)
    t_1009_0901 = pd.Timestamp("2025-10-09 09:01:00", tz=DEFAULT_TZ)
    t_1008_2101 = pd.Timestamp("2025-10-08 21:01:00", tz=DEFAULT_TZ)  # 复市日(10/9)“夜盘段”对应自然日前一晚

    assert t_0930_1500 in store.index
    assert t_1009_0901 in store.index
    assert t_1008_2101 in store.index

    i_0930 = int(store.index.get_loc(t_0930_1500))
    i_1009 = int(store.index.get_loc(t_1009_0901))
    i_1008 = int(store.index.get_loc(t_1008_2101))

    # 09/30 属于 20250930
    assert int(store.row_trading_day[i_0930]) == 20250930
    assert int(store.row_mask[i_0930] >= 0.5) == 1

    # 10/09 属于 20251009
    assert int(store.row_trading_day[i_1009]) == 20251009
    assert int(store.row_mask[i_1009] >= 0.5) == 1

    # 10/08 夜盘点位存在且归属 20251009，但因为没有输入数据所以 mask=0
    assert int(store.row_trading_day[i_1008]) == 20251009
    assert int(store.row_mask[i_1008] >= 0.5) == 0
