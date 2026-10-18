"""Regression for the first FG809 blocker: certified one-minute interval."""
import copy

import numpy as np
import pandas as pd
import pytest

from gym_trading_env.utils.macd_5m_causal import CausalMACD5m, ema
from test_wp02_macd import push


def test_certified_short_interval_does_not_commit_its_partial_close():
    stream = CausalMACD5m()
    for i in range(180):
        push(stream, i)
    before = copy.deepcopy(stream.snapshot())
    # Observed FG809 calendar: (2018-05-15 21:00,21:01] then (21:02,23:00].
    stream.push(1365., bucket='short/0', session='short', slot=1,
        time_ns=1000*60_000_000_000, available_ns=1000*60_000_000_000,
        session_end_ns=1000*60_000_000_000)
    after_tail = CausalMACD5m.restore(stream.snapshot())
    args = dict(bucket='next/0', session='next', slot=1,
        time_ns=1002*60_000_000_000, available_ns=1002*60_000_000_000,
        session_end_ns=1120*60_000_000_000)
    assert stream.push(1364., **args) == after_tail.push(1364., **args)
    state = before['state']
    assert stream.completed == state['completed']
    assert (stream.fast,stream.slow,stream.signal) == (state['fast'],state['slow'],state['signal'])
    dif = ema(state['fast'],1364.,12)-ema(state['slow'],1364.,26)
    signal = ema(state['signal'],dif,9)
    np.testing.assert_array_equal(stream.raw,[dif,signal,dif-signal])
    assert stream.discarded_partial_buckets == 1


def test_missing_last_minute_cannot_be_disguised_as_a_session_break():
    stream = CausalMACD5m()
    stream.push(1000.,bucket='a/0',session='a',slot=1,time_ns=60_000_000_000,
        available_ns=60_000_000_000,session_end_ns=120_000_000_000)
    with pytest.raises(ValueError,match='incomplete'):
        stream.push(1001.,bucket='b/0',session='b',slot=1,time_ns=240_000_000_000,
            available_ns=240_000_000_000,session_end_ns=540_000_000_000)


def test_previous_schema_snapshot_cannot_resume_with_new_tail_semantics():
    state = CausalMACD5m().snapshot()
    state['schema'] = 'FG_S3_E_MACD5M_CAUSAL_V2'
    with pytest.raises(ValueError,match='schema'):
        CausalMACD5m.restore(state)


def test_zero_duration_interval_end_is_a_marker_not_a_fifth_close():
    from e_data_v1.core import EDataError, MarketData

    intervals = [('2026-01-02 21:00','2026-01-02 23:00'),
                 ('2026-01-05 09:00','2026-01-05 10:15'),
                 ('2026-01-05 10:30','2026-01-05 11:30')]
    sessions = [dict(start=str(pd.Timestamp(a,tz='Asia/Shanghai')),
                     end=str(pd.Timestamp(b,tz='Asia/Shanghai'))) for a,b in intervals]
    times = pd.DatetimeIndex([t for a,b in intervals for t in pd.date_range(
        pd.Timestamp(a,tz='Asia/Shanghai')+pd.Timedelta(minutes=1),
        pd.Timestamp(b,tz='Asia/Shanghai'),freq='min')])
    close = 1000+np.sin(np.arange(len(times))/3)
    frame = pd.DataFrame(dict(Date=times,bob=times-pd.Timedelta(minutes=1),
        Open=close,High=close+2,Low=close-2,Close=close,Volume=1,OpenInterest=100,
        symbol='CZCE.FG2701',trading_date='2026-01-05'))
    # Real FG809 has bob == eob at 10:15. It is not an executable 1m bar.
    frame.loc[194,'bob'] = times[194]
    schedule = dict(timezone='Asia/Shanghai',minutes_per_day=345,
        segments=[['21:00','23:00'],['09:00','10:15'],['10:30','11:30'],['13:30','15:00']],
        dated_sessions=sessions)
    market = MarketData(frame,schedule,compute_features=False)
    before,bmeta = market.macd_snapshot(times[193])
    marker,mmeta = market.macd_snapshot(times[194])
    np.testing.assert_array_equal(before,marker)
    assert mmeta['anchor_close'] == bmeta['anchor_close'] == frame.Close.iloc[193]
    assert mmeta['completed_buckets'] == bmeta['completed_buckets'] == 38
    current,cmeta = market.macd_snapshot(times[195])
    assert cmeta['state']['state']['discarded_partial_buckets'] == 1
    assert cmeta['completed_buckets'] == 38 and current[3] == 1
    restored = CausalMACD5m.restore(mmeta['state'])
    restored.push(float(frame.Close.iloc[195]),bucket=cmeta['bucket_id'],
        session=pd.Timestamp(sessions[2]['start']).isoformat(),slot=1,
        time_ns=times[195].value,available_ns=times[195].value,
        session_end_ns=pd.Timestamp(sessions[2]['end']).value)
    assert restored.snapshot() == cmeta['state']
    # Marker OHLC is never an EMA input, even though its visible price differs.
    changed = frame.copy()
    changed.loc[194,['Open','High','Low','Close']] = [2000,2000,2000,2000]
    other = MarketData(changed,schedule,compute_features=False)
    np.testing.assert_array_equal(current,other.macd_snapshot(times[195])[0])
    assert cmeta == other.macd_snapshot(times[195])[1]
    # An observed closing marker cannot excuse a missing real minute before it.
    broken = MarketData(frame.drop(index=193),schedule,compute_features=False)
    with pytest.raises(EDataError,match='missing'):
        broken.macd_snapshot(times[195])
