# tests/utils/test_market_features_match_spec.py

import numpy as np
import pandas as pd

from tests.factories.market_data_factory import MarketDataFactory
from tests.oracles.market_spec_oracle import (
    build_market_features_spec,
    REQUIRED_MARKET_COLS,
)
from tests.utils.assertions import assert_frame_allclose

from gym_trading_env.utils.market_features import build_market_features


def test_fg_all_observation_fields_match_every_live_prefix_and_planned_clock():
    from gym_trading_env.envs.entry_contract import read_yaml
    from gym_trading_env.utils.market_features import FEATURES_MARKET_OBS_OUTPUT
    schedule=read_yaml('configs/fg_entry_v1/fg.yaml')['market_clock']
    # Both old and recent dates use the same explicitly truncated 345-minute clock.
    for day,night,end,denominator in (
            ('2019-12-11','2019-12-10','23:00',344),
            ('2019-12-12','2019-12-11','23:00',344)):
        index=pd.DatetimeIndex([],tz='Asia/Shanghai')
        for date,start,stop in ((night,'21:00',end),(day,'09:00','10:15'),
                                (day,'10:30','11:30'),(day,'13:30','15:00')):
            index=index.append(pd.date_range(f'{date} {start}',f'{date} {stop}',
                freq='min',inclusive='right',tz='Asia/Shanghai'))
        close=1000+np.arange(len(index))%7
        frame=pd.DataFrame(dict(Open=close,High=close+1,Low=close-1,Close=close,
            Volume=10,OpenInterest=100,trading_date=day),index=index)
        full=build_market_features(frame,is_future=True,real_futures_clock=True,futures_schedule=schedule)
        np.testing.assert_allclose(full.obs_minute_index_t,np.arange(len(index))/denominator)
        for stop in range(1,len(frame)+1):
            prefix=build_market_features(frame.iloc[:stop],is_future=True,
                real_futures_clock=True,futures_schedule=schedule)
            np.testing.assert_allclose(prefix[FEATURES_MARKET_OBS_OUTPUT].iloc[-1],
                full[FEATURES_MARKET_OBS_OUTPUT].iloc[stop-1],rtol=0,atol=1e-12)
        # Missing night rows must retain their planned positions, not start at 0.
        day_frame=frame.loc[frame.index.hour<18]
        day_features=build_market_features(day_frame,is_future=True,real_futures_clock=True,futures_schedule=schedule)
        night_minutes=denominator+1-225
        assert day_features.minute_index_t.iloc[0]==night_minutes
        assert day_features.obs_minute_index_t.iloc[0]==night_minutes/denominator


def test_obs_volume_uses_clipped_absolute_volume_log_tanh():
    bundle = MarketDataFactory.make_futures_bundle(num_days=1)
    frame = bundle.df_1m.copy()
    volumes = np.array([-7.0, 0.0, 10.0, 10.0, 1e9])
    frame.loc[frame.index[:5], "Volume"] = volumes
    got = build_market_features(frame, tz=bundle.tz, is_future=True)
    values = got.loc[frame.index[:5], "obs_V_t"].to_numpy()
    expected = np.tanh(np.log1p(np.clip(volumes, 0.0, None)) / 5.0)
    np.testing.assert_allclose(values, expected)
    assert values[2] == values[3]  # Same volume remains equal after different history.


def test_market_features_match_spec_futures_full_table():
    bundle = MarketDataFactory.make_futures_bundle(num_days=3)
    df = bundle.df_1m
    df_prev = bundle.df_prev_session

    got = build_market_features(
        df,
        tz=bundle.tz,
        is_future=True,
        df_prev_session=df_prev,
        limit_up_pct=None,
        limit_down_pct=None,
    )

    exp = build_market_features_spec(
        df,
        tz=bundle.tz,
        df_prev_session=df_prev,
        limit_up_pct=None,
        limit_down_pct=None,
    )

    # 1) 必须包含所有 REQUIRED_MARKET_COLS
    assert list(got.columns) == list(REQUIRED_MARKET_COLS)
    assert list(exp.columns) == list(REQUIRED_MARKET_COLS)

    # 2) 逐字段逐行 allclose
    assert_frame_allclose(got, exp, cols=list(REQUIRED_MARKET_COLS), rtol=1e-6, atol=1e-6)
