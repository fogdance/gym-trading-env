import pandas as pd
import pytest

from gym_trading_env.research.fg_history_study import inspect_frame


def test_raw_audit_keeps_missing_and_zero_duration_evidence():
    times = pd.to_datetime(['2018-05-15 21:01', '2018-05-15 21:03', '2018-05-15 21:04'])
    frame = pd.DataFrame({'Date': times, 'bob': times - pd.to_timedelta([1, 1, 0], unit='m'),
                          'trading_date': '2018-05-16', 'symbol': 'CZCE.FG809',
                          'provider': 'JUEJIN', 'source': 'HISTORY', 'Open': 900,
                          'High': 901, 'Low': 899, 'Close': 900, 'Volume': 3,
                          'OpenInterest': 100})
    record, holes = inspect_frame(frame, 'CZCE.FG809', {'segments': [['21:00', '23:00']]})
    assert record['standard_minutes'] == 2
    assert record['zero_duration'] == 1
    assert record['internal_missing_minutes'] == 1
    assert holes[0]['missing_minutes'] == [1262]
    assert record['outside_345_clock'] == 0
    with pytest.raises(ValueError, match='OpenInterest'):
        inspect_frame(frame.drop(columns='OpenInterest'), 'CZCE.FG809', {'segments': []})
