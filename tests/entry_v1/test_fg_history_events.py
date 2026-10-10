import numpy as np
import pandas as pd
import pytest

from gym_trading_env.research.fg_history_events import RangeMax, generate_events
from tests.entry_v1.test_entry_contract import make_env


SCHEDULE = {'minutes_per_day': 345, 'segments': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']]}
COSTS = dict(spread=1, slippage=0, open_fee=3, close_fee=3, close_today_fee=3,
             lot_size=20, margin_rate=.1)


def market(prices, times=None, volume=100):
    times = pd.date_range('2026-01-05 09:31', periods=len(prices), freq='min') if times is None else pd.DatetimeIndex(times)
    frame = pd.DataFrame(prices, columns=['Open', 'High', 'Low', 'Close'])
    frame['Date'], frame['bob'] = times, times - pd.Timedelta(minutes=1)
    frame['trading_date'], frame['symbol'] = '2026-01-05', 'CZCE.FG2701'
    frame['Volume'], frame['OpenInterest'] = volume, 1000
    return frame


def events(frame, **kwargs):
    return list(generate_events(frame, contract='CZCE.FG2701', role='train',
                                costs=COSTS, schedule=SCHEDULE, **kwargs))


def first(frame, direction='LONG', **kwargs):
    return next(e for e in events(frame, **kwargs) if e['direction'] == direction)


@pytest.mark.parametrize('minutes', [1, 2, 5])
@pytest.mark.parametrize('side', ['LONG', 'SHORT'])
def test_rapid_conservative_probe_matches_real_env(tmp_path, minutes, side):
    prices = [(1000, 1000, 1000, 1000)] * (minutes + 2)
    prices[minutes] = (1000, 1008, 1000, 1008)
    env = make_env(tmp_path, prices, side=side, spread=1, fees=3)
    env.step(1)
    for _ in range(minutes - 1):
        env.step(0)
    frame = market(prices if side == 'LONG' else [(2000-o, 2000-l, 2000-h, 2000-c) for o,h,l,c in prices])
    event = first(frame, direction=side, mode='conservative_spread')
    assert event['completion_status'] == 'success'
    assert event['duration_minutes'] == minutes
    assert event['reward'] == env.total_reward == 60 / (60 + minutes)
    assert event['entry_price'] == float(env.attributions[0]['entry_price'])
    assert event['account_net_pnl'] == env._get_info()['account_net_pnl']


def test_entry_bar_extreme_is_touch_not_conservative_fill_and_mae_is_bounded():
    frame = market([(1000,1000,1000,1000), (1000,1010,990,1000), (1000,1000,1000,1000)])
    touch = first(frame, mode='touch_spread')
    conservative = first(frame, mode='conservative_spread')
    assert touch['duration_minutes'] == 1
    assert touch['MAE_lower'] == 0 and touch['MAE_upper'] == 11
    assert conservative['completion_status'] == 'right_censored'
    assert conservative['reward'] is None
    frame.loc[0, 'High'] = 2000  # Completed signal bar must not fill its own trade.
    assert first(frame, mode='touch_spread') == touch


def test_terminal_risk_wins_and_forced_price_matches_env(tmp_path):
    prices = [(1000,1000,1000,1000), (1000,1000,1000,1000), (1000,1010,1000,1008)]
    frame = market(prices)
    event = first(frame, mode='conservative_spread', terminal=True)
    env = make_env(tmp_path, prices, spread=1, fees=3, main_exit=str(frame.Date.iloc[-1]))
    env.step(1); env.step(0)
    assert event['terminal_reason'] == 'main_qualification_end'
    assert event['reward'] == env.total_reward == 0
    assert event['account_net_pnl'] == env._get_info()['account_net_pnl']


def test_gap_and_policy_truncation_censor_without_reading_future_prices():
    frame = market([(1000,1000,1000,1000)] * 4,
                   times=['2026-01-05 09:31', '2026-01-05 09:32', '2026-01-05 09:34', '2026-01-05 09:35'])
    event = first(frame, mode='touch_spread')
    assert event['censor_reason'] == 'missing_minute_record' and event['reward'] is None
    frame.loc[2:, ['High', 'Close']] = 1200
    assert first(frame, mode='touch_spread') == event
    night = market([(1000,1000,1000,1000)] * 4,
                   times=['2026-01-04 22:59', '2026-01-04 23:00', '2026-01-05 09:01', '2026-01-05 09:02'])
    event = first(night, mode='touch_spread', omitted_bobs=['2026-01-04 23:00'])
    assert event['censor_reason'] == 'policy_truncated_night'
    rejected = [e for e in events(night, mode='touch_spread', omitted_bobs=['2026-01-04 23:00'])
                if e['decision_timestamp'] == pd.Timestamp('2026-01-04 23:00')]
    assert all(e['rejection_reason'] == 'unobservable_before_entry' for e in rejected)


def test_break_wait_uses_natural_time_and_gap_open_excludes_future_mae():
    frame = market([(1000,1000,1000,1000), (1000,1000,1000,1000), (1010,1010,900,1010)],
                   times=['2026-01-05 10:14', '2026-01-05 10:15', '2026-01-05 10:31'])
    event = first(frame, mode='touch_spread')
    assert event['duration_minutes'] == 16
    assert event['duration_trading_bars'] == 1
    assert event['MAE'] == 1  # Spread exposure on the preceding bar, not gap bar Low.
    assert event['first_touch_timestamp'] == pd.Timestamp('2026-01-05 10:30')


def test_no_entry_for_zero_duration_or_insufficient_quantity():
    frame = market([(1000,1000,1000,1000)] * 3, volume=2)
    assert first(frame, mode='touch_spread')['rejection_reason'] == 'insufficient_entry_volume'
    frame['Volume'] = 100
    frame.loc[1, 'bob'] = frame.loc[1, 'Date']
    assert first(frame, mode='touch_spread')['rejection_reason'] == 'unverified_entry_bar'
    assert next(e for e in events(frame, mode='touch_spread') if e['decision_timestamp'] == frame.Date.iloc[1])['rejection_reason'] == 'unverified_decision_bar'


def test_margin_priority_and_low_exit_liquidity_match_env(tmp_path):
    prices = [(10000,10000,10000,10000), (10000,10000,10000,10000), (10000,10010,1,10008)]
    frame = market(prices)
    event = first(frame, mode='conservative_spread')
    env = make_env(tmp_path, prices, spread=1, fees=3)
    env.step(1); env.step(0)
    assert event['terminal_reason'] == 'margin_call'
    assert event['account_net_pnl'] == env._get_info()['account_net_pnl']
    assert event['reward'] == 0
    frame = market([(1000,1000,1000,1000), (1000,1010,1000,1010), (1000,1010,1000,1010)])
    frame.loc[1, 'Volume'] = 3  # Opening consumed all volume; cannot close on same bar.
    event = first(frame, mode='conservative_spread')
    assert event['duration_minutes'] == 2


def test_range_queries_and_full_outcomes_against_small_brute_scan():
    frame = market([(1000+i%3, 1008+i%7, 990-i%5, 1000+i%3) for i in range(30)])
    values = frame.High.to_numpy()
    tree = RangeMax(values)
    for start, end, threshold in [(0,29,1013), (6,17,1014), (20,29,2000), (2,2,1000)]:
        hits = np.flatnonzero(values[start:end+1] >= threshold)
        assert tree.first(start, end, threshold) == (start + hits[0] if len(hits) else -1)
        assert tree.maximum(start, end) == max(values[start:end+1])
    for e in events(frame, mode='touch_spread'):
        if e['completion_status'] == 'no_entry':
            continue
        j = int(frame.index[frame.bob == e['entry_timestamp']][0])
        sign = 1 if e['direction'] == 'LONG' else -1
        hits = [k for k in range(j, len(frame)) if sign * (frame.High[k] if sign == 1 else frame.Low[k]) >= sign * e['tp_price']]
        assert e['completion_status'] == ('success' if hits else 'right_censored')
        if hits:
            k = hits[0]
            expected = frame.bob[k] if sign * frame.Open[k] >= sign * e['tp_price'] else frame.Date[k]
            assert e['first_touch_timestamp'] == expected


def test_cross_year_and_five_day_horizon_uses_trading_dates_not_bar_count():
    frame = market([(1000,1000,1000,1000)] * 7,
                   times=['2025-12-31 14:59', '2025-12-31 15:00', '2026-01-05 09:01',
                          '2026-01-06 09:01', '2026-01-07 09:01', '2026-01-08 09:01', '2026-01-09 09:01'])
    frame['trading_date'] = pd.to_datetime(frame.Date).dt.strftime('%Y-%m-%d')
    frame.loc[2, ['Open', 'High', 'Low', 'Close']] = 1010
    event = first(frame, mode='touch_spread')
    assert event['first_touch_timestamp'] == pd.Timestamp('2026-01-05 09:00')
    assert event['duration_minutes'] == (pd.Timestamp('2026-01-05 09:00') - pd.Timestamp('2025-12-31 14:59')).total_seconds() / 60
    assert event['horizon_1day_end'] == pd.Timestamp('2025-12-31 15:00')
    assert event['horizon_5day_end'] == pd.Timestamp('2026-01-08 15:00')


def test_primary_touch_needs_no_exit_volume_but_conservative_does():
    frame = market([(1000,1000,1000,1000)] * 4)
    frame.loc[2, ['High', 'Close', 'Volume']] = [1010, 1010, 1]
    assert first(frame, mode='touch_spread')['completion_status'] == 'success'
    assert first(frame, mode='conservative_spread')['completion_status'] == 'right_censored'


def test_unfilled_main_exit_is_not_reported_as_a_committed_fill():
    frame = market([(1000,1000,1000,1000)] * 3)
    frame.loc[2, 'Volume'] = 1
    event = first(frame, mode='conservative_spread', terminal=True)
    assert event['completion_status'] == 'terminal_failure'
    assert event['terminal_reason'] == 'main_qualification_end_unfilled'
    assert event['reward'] == 0 and event['exit_price'] is None
