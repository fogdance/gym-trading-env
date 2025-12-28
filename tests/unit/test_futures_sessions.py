# tests/unit/test_futures_sessions.py

import pandas as pd
import pytest

pytestmark = pytest.mark.unit

from gym_trading_env.utils.session_futures_strict import strict_reindex_futures_345
from gym_trading_env.utils.timebase import FEATURE_TZ as DEFAULT_TZ


def _make_ohlc_df(idx):
    # 用简单常数填充即可；只要 Close 非 NaN 就会 mask=1
    n = len(idx)
    return pd.DataFrame(
        {
            "Open":  [100.0] * n,
            "High":  [101.0] * n,
            "Low":   [99.0] * n,
            "Close": [100.0] * n,
            "Volume": [1] * n,
        },
        index=pd.DatetimeIndex(idx),
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

    out = strict_reindex_futures_345(df, tz=DEFAULT_TZ)
    sid = out["session_id"]
    msk = out["mask"]
    midx = out["minute_index"]

    t_fri_2244 = pd.Timestamp("2025-09-19 22:44:00", tz=DEFAULT_TZ)
    t_mon_0901 = pd.Timestamp("2025-09-22 09:01:00", tz=DEFAULT_TZ)

    # 核心：两者 session_id 同为周一 20250922
    assert sid.loc[t_fri_2244] == "20250922"
    assert sid.loc[t_mon_0901] == "20250922"

    # 且两者都是真实数据命中（mask=1）
    assert int(msk.loc[t_fri_2244]) == 1
    assert int(msk.loc[t_mon_0901]) == 1

    # minute_index 顺序：夜盘120分钟在前 -> 09:01 应是 120
    assert int(midx.loc[t_mon_0901]) == 120


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

    out = strict_reindex_futures_345(df, tz=DEFAULT_TZ)
    sid = out["session_id"]
    msk = out["mask"]

    t_thu_2200 = pd.Timestamp("2025-09-18 22:00:00", tz=DEFAULT_TZ)
    t_fri_0901 = pd.Timestamp("2025-09-19 09:01:00", tz=DEFAULT_TZ)

    assert sid.loc[t_thu_2200] == "20250919"
    assert sid.loc[t_fri_0901] == "20250919"
    assert int(msk.loc[t_thu_2200]) == 1
    assert int(msk.loc[t_fri_0901]) == 1


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

    out = strict_reindex_futures_345(df, tz=DEFAULT_TZ)
    sid = out["session_id"]
    msk = out["mask"]

    t_0930_1500 = pd.Timestamp("2025-09-30 15:00:00", tz=DEFAULT_TZ)
    t_1009_0901 = pd.Timestamp("2025-10-09 09:01:00", tz=DEFAULT_TZ)
    t_1008_2101 = pd.Timestamp("2025-10-08 21:01:00", tz=DEFAULT_TZ)  # 复市日(10/9)“夜盘段”对应自然日前一晚

    # 09/30 属于 20250930
    assert sid.loc[t_0930_1500] == "20250930"
    assert int(msk.loc[t_0930_1500]) == 1

    # 10/09 属于 20251009
    assert sid.loc[t_1009_0901] == "20251009"
    assert int(msk.loc[t_1009_0901]) == 1

    # 10/08 夜盘点位存在且归属 20251009，但因为没有输入数据所以 mask=0
    assert sid.loc[t_1008_2101] == "20251009"
    assert int(msk.loc[t_1008_2101]) == 0
