"""Explicit P1/P2 boundaries absent from the supplied static-only review."""
import pandas as pd
import pytest
import yaml

from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from gym_trading_env.research.fg_s3_quality_teacher import quality_record
from test_fg_single3_env import single_env


def test_chunk_metadata_and_censor_are_not_failed_entry_labels(tmp_path, monkeypatch):
    env = single_env(tmp_path, [(1000,1000,1000,1000)] * 4 +
                     [(1000,1008,1000,1007)], candidate='C1', chunk=1,
                     zero_duration_indices=(2,))
    env._raw['trading_date'] = '2026-01-05'
    state_path = tmp_path/'must_not_write.json'
    state_path.write_text('original state file\n')
    env.entry_config['state_path'] = str(state_path)
    before = env.export_state()

    def forbidden_close(self):
        raise AssertionError('Read-only counterfactual called Env.close()')

    monkeypatch.setattr(type(env), 'close', forbidden_close)
    censored = quality_record(env, fold='TRAIN', max_market_steps=2)
    assert censored['outcome'] == 'RIGHT_CENSORED'
    assert censored['last_observed_cursor'] == 3  # metadata does not consume budget
    assert not censored['quality_valid'] and censored['quality_score'] is None
    assert not censored['immediate_add_outcome_valid']
    complete = quality_record(env, fold='TRAIN')
    assert complete['outcome'] == 'TAKE_PROFIT' and complete['market_minutes'] == 3
    assert complete['quality_score'] == 100
    assert env.export_state() == before
    assert state_path.read_text() == 'original state file\n'


@pytest.mark.parametrize('cross_official_day', [False, True])
def test_native_official_day_and_session_break(cross_official_day, tmp_path):
    if cross_official_day:
        times = ['2026-01-05 14:59', '2026-01-05 15:00', '2026-01-05 21:01']
        calendar = [dict(start='2026-01-05 14:58', end='2026-01-05 15:00'),
                    dict(start='2026-01-05 21:00', end='2026-01-05 21:01')]
        days = ['2026-01-05', '2026-01-05', '2026-01-06']
    else:
        times = ['2026-01-05 22:59', '2026-01-05 23:00', '2026-01-06 09:31']
        calendar = [dict(start='2026-01-05 22:58', end='2026-01-05 23:00'),
                    dict(start='2026-01-06 09:30', end='2026-01-06 09:31')]
        days = ['2026-01-06'] * 3
    seed = single_env(tmp_path, [(1000,1000,1000,1000), (1000,1000,996,1000),
                                (1000,1008,1000,1007)], candidate='C1', fees=3)
    # Supply official dates and the real futures clock before construction.
    raw = seed._raw.copy()
    raw.index = pd.DatetimeIndex(times, tz='Asia/Shanghai')
    raw['bob'] = raw.index - pd.Timedelta(minutes=1)
    raw['trading_date'] = days
    contract_path = tmp_path/'contract.yaml'
    terms = yaml.safe_load(contract_path.read_text())
    terms['sessions'] = calendar
    contract_path.write_text(yaml.safe_dump(terms))
    config_path = tmp_path/'env.yaml'
    config = yaml.safe_load(config_path.read_text())
    config['entry']['market_clock'] = dict(timezone='Asia/Shanghai', minutes_per_day=345,
        segments=[['21:00','23:00'],['09:00','10:15'],['10:30','11:30'],['13:30','15:00']])
    config_path.write_text(yaml.safe_dump(config))
    env = FixedDirectionEntryEnv(df=raw, config_path=config_path)
    record = quality_record(env, fold='TRAIN')
    assert record['outcome'] == 'TAKE_PROFIT' and record['market_minutes'] == 2
    assert record['later_trading_days'] == int(cross_official_day)
    assert record['mae_quote_envelope_points'] == 4
    assert record['quality_score'] == (45 if cross_official_day else 100)
    assert record['tp_event_reward_new'] == (.45 if cross_official_day else 1)


@pytest.mark.parametrize('exit_kind', ['MAIN_ROLL', 'MARGIN_CALL'])
def test_other_exit_is_observed_but_unscored(exit_kind, tmp_path):
    if exit_kind == 'MAIN_ROLL':
        env = single_env(tmp_path, [(1000,1000,1000,1000), (1000,1000,1000,1000),
                                   (995,995,995,995)], candidate='C1', fees=3,
                         main_exit='2026-01-05 09:33')
    else:
        env = single_env(tmp_path, [(50000,50000,50000,50000),
                                   (50000,50010,49960,50010)], candidate='C1')
    env._raw['trading_date'] = '2026-01-05'
    record = quality_record(env, fold='TRAIN')
    assert record['outcome'] == exit_kind
    assert record['immediate_add_outcome_valid']
    assert not record['quality_valid'] and record['quality_score'] is None
    assert record['tp_event_reward_new'] is None
    assert record['old_reward_undiscounted'] == record['new_reward_undiscounted']


def test_rejected_fill_never_becomes_negative_teacher(tmp_path):
    env = single_env(tmp_path, [(1000,1000,1000,1000)] * 3, candidate='C1')
    env._raw['trading_date'] = '2026-01-05'
    env._raw.loc[env._raw.index[1], 'Volume'] = 2
    before = env.export_state()
    record = quality_record(env, fold='TRAIN')
    assert record['actual_action_mask'] == [1, 1]
    assert record['outcome'] == 'REJECTED'
    assert not record['immediate_add_outcome_valid']
    assert not record['quality_valid'] and record['quality_score'] is None
    assert env.export_state() == before


def test_missing_official_day_and_e_profile_fail_closed(tmp_path):
    env = single_env(tmp_path, [(1000,1000,1000,1000), (1000,1000,1000,1000),
                               (1000,1008,1000,1007)], candidate='C1')
    before = env.export_state()
    with pytest.raises(ValueError, match='official trading dates'):
        quality_record(env, fold='TRAIN')
    assert env.export_state() == before
    env._e_profile = object()
    with pytest.raises(ValueError, match='E-profile'):
        quality_record(env, fold='TRAIN')
