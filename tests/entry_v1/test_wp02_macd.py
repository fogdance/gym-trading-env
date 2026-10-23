"""A: causal math, versioning and actual native Env; no model or real prices."""
import copy
import json
import numpy as np
import pandas as pd
import pytest
from gym_trading_env.utils.macd_5m_causal import CausalMACD5m, ema
from gym_trading_env.research.fg_s3_e_observation import e_array_hash, observation_contract
from test_s3_e_native_env import _profile
from test_entry_market_cache import csv_env_config
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv


def push(s,i,close=None,session='day',offset=0):
    return s.push(1000+i if close is None else close, bucket=f'{session}/{i//5}',session=session,
        slot=i%5+1,time_ns=(i+offset+1)*60_000_000_000,available_ns=(i+offset+1)*60_000_000_000)


def test_A01_partial_from_committed_ema_and_hist_without_double():
    s=CausalMACD5m()
    for i in range(180):push(s,i)
    state=copy.deepcopy(s.snapshot())
    for j in range(4):
        push(s,180+j)
        assert s.completed==36
        f=ema(state['state']['fast'],1180+j,12);slow=ema(state['state']['slow'],1180+j,26)
        signal=ema(state['state']['signal'],f-slow,9)
        np.testing.assert_array_equal(s.raw,[f-slow,signal,f-slow-signal])
    push(s,184);assert s.completed==37
    assert s.fast==ema(state['state']['fast'],1184,12)


def test_A02_warmup_and_price_scale():
    s=CausalMACD5m()
    for i in range(175):assert push(s,i)==[0.]*4
    out=push(s,175)
    assert out[3]==1
    np.testing.assert_array_equal(out[:3],[1000*x/1175 for x in s.raw])


def test_A03_stream_restore_and_prefix_invariance():
    s=CausalMACD5m()
    for i in range(182):push(s,i)
    restored=CausalMACD5m.restore(json.loads(json.dumps(s.snapshot())))
    for i in range(182,195):assert push(s,i)==push(restored,i)
    assert s.snapshot()==restored.snapshot()
    prefix=[];other=CausalMACD5m()
    for i in range(200):
        x=push(other,i,1000+i if i<195 else 10000+i)
        if i<195:prefix.append(x)
    original=CausalMACD5m()
    assert prefix==[push(original,i) for i in range(195)]


@pytest.mark.parametrize('kind',['duplicate','missing','order','late','nonfinite','negative'])
def test_A04_bad_minutes_fail_closed(kind):
    s=CausalMACD5m();push(s,0)
    args=dict(close=1001,bucket='day/0',session='day',slot=2,time_ns=120_000_000_000,available_ns=120_000_000_000)
    if kind=='duplicate':args.update(slot=1,time_ns=60_000_000_000)
    if kind=='missing':args.update(slot=3,time_ns=180_000_000_000)
    if kind=='order':args.update(time_ns=0)
    if kind=='late':args.update(available_ns=180_000_000_000)
    if kind=='nonfinite':args.update(close=float('nan'))
    if kind=='negative':args.update(close=-1)
    with pytest.raises(ValueError):s.push(**args)


def test_A05_session_break_carries_ema_but_cannot_join_incomplete_buckets():
    s=CausalMACD5m()
    for i in range(180):push(s,i)
    previous=s.fast
    push(s,0,1200,session='night_next_official_day',offset=1000)
    assert s.fast==previous and s.completed==36 and s.output[3]==1
    with pytest.raises(ValueError,match='incomplete'):push(s,0,1200,session='after_lunch',offset=2000)


def test_A06_native_snapshot_repeated_queries_cold_restore_and_macd_hash(tmp_path):
    config,market=csv_env_config(tmp_path/'source')
    profile=_profile(market)
    a=FixedDirectionEntryEnv(config_path=config,e_profile=profile)
    b=FixedDirectionEntryEnv(config_path=config,e_profile=profile)
    try:
        a.step(0);a.step(0)
        before=a._get_obs();state=a.export_state()
        for _ in range(3):
            for k,v in before.items():np.testing.assert_array_equal(a._get_obs()[k],v)
        b.import_state(json.loads(json.dumps(state)))
        for k,v in before.items():np.testing.assert_array_equal(b._get_obs()[k],v)
        changed={**before,'market_macd_5m':np.array([1,2,3,1],np.float32)}
        assert e_array_hash(before)!=e_array_hash(changed)
    finally:a.close();b.close()


def test_A07_course_v1_and_sidecar_cannot_masquerade_as_v2():
    from dreamerv3.tests.test_fg_s3_e_quality_course_contract import draft_spec
    from dreamerv3.fg_s3_e_quality_course_contract import validate_course_spec,OBSERVATION
    spec=draft_spec();spec['schema']='FG_S3_E_QUALITY_COURSE_V1'
    with pytest.raises(ValueError):validate_course_spec(spec)
    assert OBSERVATION['macd_5m']==observation_contract()['macd_5m']
    assert len(OBSERVATION['policy_observation_fields'])==5


def test_A08_actual_calendar_snapshot_future_ohlc_and_arrivals_do_not_change_prefix(tmp_path):
    from wp02_fixture import native_factory
    factory,path,profile=native_factory(tmp_path/'market',minutes=210)
    frame=pd.read_csv(path.parent/'market.csv')
    frame['Close']=1000+np.sin(np.arange(len(frame))/3)*.5
    market=path.parent/'market.csv';frame.to_csv(market,index=False)
    a=factory()
    frame.loc[190:,['High','Low','Close']]=[1010,990,1003]
    frame['available_at']=frame.Date
    frame.loc[190:,'available_at']=(pd.to_datetime(frame.Date.iloc[190:])+pd.Timedelta(minutes=5)).astype(str)
    frame.to_csv(market,index=False)
    b=factory()
    try:
        for i in [0,1,2,3,4,174,175,176,177,178,179,180,185]:
            cutoff=a._raw.index[i]
            left,lmeta=a._e_market.macd_snapshot(cutoff)
            right,rmeta=b._e_market.macd_snapshot(cutoff)
            np.testing.assert_array_equal(left,right)
            assert lmeta==rmeta
        assert a._e_market.macd_snapshot(a._raw.index[185])[0][3]==1
        assert any(a._e_market.macd_snapshot(a._raw.index[185])[0][:3]!=0)
    finally:a.close();b.close()


def test_A09_native_dated_calendar_midnight_weekend_holiday_and_lunch(tmp_path):
    from wp02_fixture import native_factory
    from e_data_v1.core import MarketData,EDataError
    from e_data_v1.schema import EDataConfig
    factory,path,_=native_factory(tmp_path/'market',minutes=690)
    env=factory();frame=pd.read_csv(path.parent/'market.csv')
    sessions=[];times=[];days=[]
    for night,day in [('2026-01-02','2026-01-05'),('2026-01-05','2026-01-08')]:
        for a,b in [(night+' 21:00',night+' 23:00'),
                    (day+' 09:00',day+' 10:15'),(day+' 10:30',day+' 11:30'),(day+' 13:30',day+' 15:00')]:
            start,end=pd.Timestamp(a,tz='Asia/Shanghai'),pd.Timestamp(b,tz='Asia/Shanghai')
            sessions.append(dict(start=str(start),end=str(end)))
            rows=list(pd.date_range(start+pd.Timedelta(minutes=1),end,freq='min'))
            times+=rows;days += [day]*len(rows)
    frame['Date']=times;frame['bob']=[t-pd.Timedelta(minutes=1) for t in times];frame['trading_date']=days
    schedule={**env._e_schedule,'dated_sessions':sessions}
    market=MarketData(frame,schedule,EDataConfig())
    _,night=market.macd_snapshot(times[119]);assert night['completed_buckets']==24
    _,morning=market.macd_snapshot(times[194]);assert morning['completed_buckets']==39
    _,lunch=market.macd_snapshot(times[195]);assert lunch['completed_buckets']==39
    _,day_end=market.macd_snapshot(times[344]);assert day_end['completed_buckets']==69
    _,holiday=market.macd_snapshot(times[465]);assert holiday['completed_buckets']==93
    broken=MarketData(frame.drop(index=10),schedule,EDataConfig())
    with pytest.raises(EDataError,match='missing'):broken.macd_snapshot(times[20])
    # Frozen FG 345-minute features reject an invented midnight session. The
    # actual night-to-next-official-day gap above crosses midnight/weekends
    # without manufacturing bars, EMA updates or a continuous wall-time bucket.
    midnight=frame.copy();midnight.loc[:119,'Date']=pd.Series(times[:120])+pd.Timedelta(hours=2)
    midnight.loc[:119,'bob']=pd.Series(times[:120])+pd.Timedelta(hours=2,minutes=-1)
    midnight_schedule=copy.deepcopy(schedule)
    midnight_schedule['dated_sessions'][0]={k:str(pd.Timestamp(v)+pd.Timedelta(hours=2)) for k,v in sessions[0].items()}
    with pytest.raises(ValueError,match='345-minute clock'):
        MarketData(midnight,midnight_schedule,EDataConfig())
    # A zero-duration session-open row cannot become a new MACD price anchor.
    opening=frame.iloc[[195]].copy();opening['Date']=pd.Timestamp(sessions[2]['start']).tz_convert('Asia/Shanghai')
    opening['bob']=opening['Date'];opening[['Open','High','Low','Close']]=2000
    with_open=MarketData(pd.concat([frame,opening],ignore_index=True),schedule,EDataConfig())
    prior,pmeta=with_open.macd_snapshot(times[194])
    value,meta=with_open.macd_snapshot(pd.Timestamp(sessions[2]['start']))
    np.testing.assert_array_equal(prior,value)
    assert meta['anchor_close']==pmeta['anchor_close']==float(frame.Close.iloc[194])
    env.close()
