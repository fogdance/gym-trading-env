from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture(scope="module")
def market():
    segments = [["21:00", "23:00"], ["09:00", "10:15"],
                ["10:30", "11:30"], ["13:30", "15:00"]]
    times, days, sessions = [], [], []
    for day in ("2026-01-06", "2026-01-07"):
        td = pd.Timestamp(day)
        for start, end in segments:
            wall_date = td - pd.Timedelta(days=1) if start == "21:00" else td
            a, b = (pd.Timestamp(str(wall_date.date()) + " " + x, tz="Asia/Shanghai") for x in (start, end))
            idx = pd.date_range(a + pd.Timedelta(minutes=1), b, freq="min")
            times.extend(idx); days.extend([day] * len(idx))
            sessions.append(dict(start=a.isoformat(), end=b.isoformat()))
    i = np.arange(len(times))
    close = 1000 + (i // 345) * 20 + np.sin(i / 11) * 3
    frame = pd.DataFrame(dict(Date=times, Open=close-.25, High=close+1.5,
        Low=close-1.5, Close=close, Volume=100+(i % 67)*10,
        OpenInterest=100000+i*20, symbol="CZCE.FG809", trading_date=days,
        bob=pd.DatetimeIndex(times)-pd.Timedelta(minutes=1)))
    schedule = dict(timezone="Asia/Shanghai", minutes_per_day=345,
                    segments=segments, dated_sessions=sessions)
    return frame, schedule


@pytest.fixture(scope="module")
def prepared(market):
    from e_data_v1 import EDataConfig, MarketData
    frame, schedule = market
    return MarketData(frame, schedule, EDataConfig(audit_1m=True))
