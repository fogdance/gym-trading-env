import numpy as np
import pandas as pd

from gym_trading_env.research.fg_history_statistics import horizon_flags, competing_risk_curve, contract_intervals


def test_censoring_and_trading_day_horizons_do_not_become_failures():
    frame = pd.DataFrame(dict(
        completion_status=['success', 'terminal_failure', 'right_censored', 'right_censored'],
        duration_minutes=[2, np.nan, np.nan, np.nan], followup_minutes=[2, 1, 1, 5160],
        entry_timestamp=['2026-01-05 09:00']*4,
        first_touch_timestamp=['2026-01-05 09:02', None, None, None],
        observation_end=['2026-01-05 09:02', '2026-01-05 09:01', '2026-01-05 09:01', '2026-01-08 23:00'],
        horizon_1day_end=['2026-01-05 15:00']*4, trading_date=['2026-01-05']*4))
    days = ['2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08', '2026-01-09']
    mapping = {pd.Timestamp('2026-01-05 09:02'): '2026-01-05', pd.Timestamp('2026-01-05 09:01'): '2026-01-05',
               pd.Timestamp('2026-01-08 23:00'): '2026-01-09'}
    flags = horizon_flags(frame, days, mapping)
    np.testing.assert_array_equal(flags['5m'][0], [True, False, False, False])
    np.testing.assert_array_equal(flags['5m'][1], [False, False, True, False])
    # Fifth trading day's night occurs BEFORE that trading day's day close.
    np.testing.assert_array_equal(flags['5day'][1], [False, False, True, True])
    frame.loc[3, 'observation_end'] = '2026-01-09 15:00'
    mapping[pd.Timestamp('2026-01-09 15:00')] = '2026-01-09'
    assert not horizon_flags(frame, days, mapping)['5day'][1][3]


def test_competing_terminal_and_cluster_resampling_have_known_results():
    curve = competing_risk_curve({1: [1, 0, 1], 2: [1, 1, 0]})
    assert curve == [(1, .25), (2, .625)]
    values = np.array([[100, 20, 30], [100, 20, 30]])
    assert contract_intervals(values, 1000, np.random.default_rng(0)) == (.2, .5)
