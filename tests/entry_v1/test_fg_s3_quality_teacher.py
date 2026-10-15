"""Opt-in offline Quality Teacher regression tests; no GPU or old-Run writes."""
import pytest
import pandas as pd
from types import SimpleNamespace

from gym_trading_env.research.fg_s3_quality_score import tp_base_points, tp_event_reward, quality_for_exit
from gym_trading_env.research.fg_s3_quality_teacher import quality_record, later_official_days
from test_fg_single3_env import single_env


@pytest.mark.parametrize('t,expected', [
    (4,100),(10,100),(30,96.875),(60,92.1875),(74,90),
    (120,85),(240,75),(480,60+15*__import__('math').exp(-2)),
])
def test_quality_score_anchor(t,expected):
    assert tp_base_points(t,0)==pytest.approx(expected)
    assert tp_event_reward(t,0)==pytest.approx(expected/100)


def test_cross_day_and_refuse_invalid():
    assert [tp_base_points(74,d) for d in range(1,9)]==[45,40,35,30,25,20,20,20]
    for t,d in ((0,0),(-1,0),(10,-1),(1.5,0),(True,0),(1,True)):
        with pytest.raises(ValueError):tp_base_points(t,d)
    assert quality_for_exit('STOP_LOSS', 5, 0)==(0.,'OBSERVED_SL')
    assert quality_for_exit('MAIN_ROLL',5,0)==(None,'OTHER_OR_CENSORED')
    assert quality_for_exit('RIGHT_CENSORED',None,None)[0] is None


def test_real_env_tp_no_oracle_and_no_mutation(tmp_path):
    env=single_env(tmp_path,[(1000,1000,1000,1000),(1000,1000,1000,1000),
                             (1000,1007,1000,1007)],candidate='C1',fees=3)
    env._raw['trading_date']='2026-01-05'  # fixture has no official-day column
    original=env.export_state()
    record=quality_record(env,fold='TRAIN')
    assert env.export_state()==original
    assert record['outcome']=='TAKE_PROFIT' and record['quality_valid']
    assert record['quality_score']==100 and record['market_minutes']==2
    assert record['tp_event_reward_new']==1
    assert record['old_tp_event_reward']==pytest.approx(60/62)
    assert record['new_reward_undiscounted']==pytest.approx(1)
    assert record['net_trade_pnl']=='282'
    assert not record['strict_decision_valid']
    assert record['training_action_label_allowed'] is False


def test_real_env_sl_and_losing_state_continues_to_grade(tmp_path):
    env=single_env(tmp_path,[(1000,1000,1000,1000),(1000,1008,965,1007),
                             (1000,1000,1000,1000)],candidate='C1',fees=3)
    env._raw['trading_date']='2026-01-05'
    record=quality_record(env,fold='TRAIN')
    assert record['outcome']=='STOP_LOSS' and record['quality_score']==0
    assert record['tp_event_reward_new'] is None
    assert record['new_reward_undiscounted']==record['old_reward_undiscounted']
    # A real loss now places the original account BELOW the old 300k
    # Oracle certificate, but a valid single-entry outcome remains possible.
    env.step(1)
    assert env._calculate_equity()<300000
    assert env._published_mask[1]
    after=quality_record(env,fold='TRAIN')
    assert after['starting_equity']==str(env._calculate_equity())
    assert not after['strict_decision_valid']
    assert after['outcome'] in ('STOP_LOSS','TAKE_PROFIT','RIGHT_CENSORED')


def test_no_label_for_unexecuted_or_live_position_and_test_fold(tmp_path):
    env=single_env(tmp_path,[(1000,1000,1000,1000)]*3,candidate='C1')
    with pytest.raises(ValueError,match='TEST'):
        quality_record(env,fold='TEST')
    env.step(1)
    with pytest.raises(ValueError,match='ACTIVE flat'):
        quality_record(env,fold='TRAIN')


def test_official_trading_dates_night_session_not_wall_midnight():
    fake=SimpleNamespace(_raw=pd.DataFrame({'trading_date':
        ['2026-01-06','2026-01-06','2026-01-06','2026-01-07']}))
    assert later_official_days(fake,0,2)==0
    assert later_official_days(fake,1,3)==1


def test_quality_can_be_valid_when_strict_oracle_rejects(tmp_path):
    env=single_env(tmp_path,[(1000,1000,1000,1000), (1000,1000,1000,1000),
                             (1000,1007,1000,1007)],candidate='C1',fees=3)
    env._raw['trading_date']='2026-01-05'
    class Uncertified:
        def query(self, env, mask, *, snapshot):
            return dict(action=0,valid=False,certified=False,
                        reason='FUNDS_OUTSIDE_ORIGINAL_CERTIFICATE')
    record=quality_record(env,fold='TRAIN',strict_query=Uncertified())
    assert record['quality_valid']
    assert record['quality_score']==100
    assert not record['strict_decision_valid']
    assert record['strict_oracle']['action'] is None
    assert record['add_better_than_wait'] is None
