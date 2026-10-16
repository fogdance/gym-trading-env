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
        assert a.e_snapshot==b.e_snapshot
        for k,v in before.items():np.testing.assert_array_equal(b._get_obs()[k],v)
        changed={**before,'market_macd_5m':np.array([1,2,3,1],np.float32)}
        assert e_array_hash(before)!=e_array_hash(changed)
        with pytest.raises(ValueError,match='E V2'):b.import_state({k:v for k,v in state.items() if k!='e_state'})
    finally:a.close();b.close()


def test_A07_course_v1_and_sidecar_cannot_masquerade_as_v2():
    from dreamerv3.tests.test_fg_s3_e_quality_course_contract import draft_spec
    from dreamerv3.fg_s3_e_quality_course_contract import validate_course_spec,OBSERVATION
    spec=draft_spec();spec['schema']='FG_S3_E_QUALITY_COURSE_V1'
    with pytest.raises(ValueError):validate_course_spec(spec)
    assert OBSERVATION['macd_5m']==observation_contract()['macd_5m']
    assert len(OBSERVATION['policy_observation_fields'])==5
