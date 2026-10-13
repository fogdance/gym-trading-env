"""Controlled truths, separate from real-data and real-GPU acceptance."""
import itertools
import json
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from test_fg_single3_env import single_env
from gym_trading_env.research.fg_single3_oracle import CandidateIndex, GAMMA, empty_at, optimize, verify_candidates
from gym_trading_env.research.fg_s3b_advantage import RelaxedValues
from gym_trading_env.rewards.single3_reward import Single3RewardConfig
from gym_trading_env.research.fg_reward_candidates import timeboxed_tp
from gym_trading_env.utils.s3_market_clock import MarketClock, PROTOCOL


@pytest.mark.parametrize('bars', [1, 2, 5, 6, 15, 16, 60, 61])
def test_inclusive_fill_reward_age_and_sparse_oracle(tmp_path, bars):
    prices = [(1000,1000,1000,1000)] * (bars+2)
    prices[bars] = (1000,1007,1000,1007)
    env = single_env(tmp_path, prices, candidate='C1',
        times=pd.date_range('2026-01-05 09:01', periods=len(prices), freq='min'),
        calendar=[dict(start='2026-01-05 09:00', end='2026-01-05 10:15')])
    candidate = CandidateIndex(env).candidate(0)
    empty_at(env,0)
    total = 0
    for i in range(1,bars+1):
        obs, reward, *_ = env.step(int(i==1))
        total += reward
        if i < bars:
            assert obs['position_state'][5] == pytest.approx(np.log1p(i)/np.log1p(43200))
    event = env.attributions[0]
    assert event['clock_protocol'] == PROTOCOL
    assert event['held_market_bars'] == event['tp_market_minutes'] == bars
    assert candidate['held_market_bars'] == candidate['tp_market_minutes'] == bars
    assert total == pytest.approx(timeboxed_tp(bars))
    assert candidate['rewards']['C1'] == [[bars, float(np.float32(total))]]
    assert env._get_info()['mean_tp_market_minutes'] == bars


@pytest.mark.parametrize('restart', ['lunch', 'overnight'])
def test_closed_hours_metadata_zero_volume_and_chunk_restore(tmp_path, restart):
    later = '2026-01-05 13:31' if restart=='lunch' else '2026-01-06 09:31'
    times = pd.DatetimeIndex(['2026-01-05 11:28','2026-01-05 11:29','2026-01-05 11:30',later])
    calendar = [dict(start='2026-01-05 11:27',end='2026-01-05 11:30'),
                dict(start=str(pd.Timestamp(later)-pd.Timedelta(minutes=1)),end=later)]
    env = single_env(tmp_path, [(1000,1000,1000,1000)]*3+[(1000,1007,1000,1007)],
        times=times, calendar=calendar, zero_duration_indices=(2,), chunk=1)
    env.step(1); env.reset()
    before = env._get_obs()['market_seq'].copy()
    obs, reward, *_ = env.step(0)
    assert reward == 0 and obs['position_state'][5] == pytest.approx(np.log1p(1)/np.log1p(43200))
    np.testing.assert_array_equal(obs['market_seq'], before)
    state = json.loads(json.dumps(env.export_state()))
    env.import_state(state)
    assert env.export_state() == state
    env.reset(); env.step(0)
    assert env.attributions[0]['tp_market_minutes'] == 2
    assert env.total_reward == pytest.approx(60/62)


def test_zero_volume_real_bar_counts_and_censored_or_rejected_is_not_tp(tmp_path):
    env=single_env(tmp_path, [(1000,1000,1000,1000)]*4)
    env.step(1)
    env._raw.loc[env._raw.index[2], 'Volume']=0
    obs, reward, *_ = env.step(0)
    assert reward == 0 and obs['position_state'][5] == pytest.approx(np.log1p(2)/np.log1p(43200))
    env.step(0); _,reward,terminal,truncated,_=env.step(0)
    assert reward==0 and truncated and not terminal and not env.attributions
    rejected=tmp_path/'rejected';rejected.mkdir()
    other=single_env(rejected,[(1000,1000,1000,1000)]*3,zero_duration_indices=(1,))
    _,_,_,_,info=other.step(1)
    assert info['reject_reason']=='UNVERIFIED_BAR_TIME' and not other.entries and not other.attributions
    gap=tmp_path/'gap';gap.mkdir()
    other=single_env(gap,[(1000,1000,1000,1000)]*3,
        times=pd.DatetimeIndex(['2026-01-05 09:31','2026-01-05 09:32','2026-01-05 09:34']))
    other.step(1); _,r,terminal,truncated,info=other.step(0)
    assert r==0 and truncated and not terminal and info['status']=='DATA_GAP' and not other.attributions


@pytest.mark.parametrize('bar,penalty', [(5,-.5),(6,-.2),(15,-.2),(16,0.)])
def test_first_adverse_market_bar_boundaries(tmp_path,bar,penalty):
    prices=[(1000,1000,1000,1000)]*(bar+2)
    prices[bar]=(1000,1000,990,1000)
    env=single_env(tmp_path,prices,candidate='C1')
    rewards=[env.step(1)[1]]+[env.step(0)[1] for _ in range(bar-1)]
    assert sum(rewards)==pytest.approx(penalty)
    assert next(iter(env.reward_function.scorer.active.values())).adverse_first_bar==bar


def test_metadata_discount_matches_independent_env_enumeration_and_s3b(tmp_path):
    prices=[(1000,1000,1000,1000),(1000,1000,1000,1000),(1000,1000,1000,1000),
            (1000,1007,1000,1007),(1000,1000,1000,1000)]
    env=single_env(tmp_path,prices,candidate='C1',zero_duration_indices=(2,))
    index=CandidateIndex(env); rows=[index.candidate(t) for t in range(index.flat_end)]
    verify_candidates(index,rows,count=100)
    gold=Fraction(0)
    for actions in itertools.product((0,1),repeat=4):
        empty_at(env,0); total=Fraction(0)
        for action in actions:
            before=env.current_step
            _,reward,*_=env.step(action)
            exponent=index.clock.reward_exponent(0,env.current_step)
            total+=GAMMA**exponent*Fraction.from_float(float(np.float32(reward)))
        gold=max(gold,total)
    plan=optimize(rows,index.flat_end,'C1',clock=index.clock)
    assert plan['value_interval'][0] <= float(gold) <= plan['value_interval'][1]
    relaxed=RelaxedValues(rows,index.flat_end,clock=index.clock)
    assert list(relaxed.chain(int(relaxed.head[0]))) == plan['add_cursors']
    assert index.clock.advance(1,2)==0 and index.clock.held(1,3)==2
    with pytest.raises(TypeError,match='clock'):
        optimize(rows,index.flat_end,'C1')


def test_cross_semantic_snapshot_rejected_before_model_use(tmp_path):
    env=single_env(tmp_path)
    state=env.export_state(); state['schema']=2
    with pytest.raises(ValueError,match='Incompatible'):
        env.import_state(state)
    with pytest.raises(ValueError,match='frozen'):
        Single3RewardConfig(schema='FG_SINGLE3_REWARD_V1')


def test_model_return_uses_feedback_aligned_market_discounts():
    import jax.numpy as jnp
    from dreamerv3.agent import lambda_return
    zeros=jnp.zeros((1,4)); rew=jnp.array([[0.,0.,0.,2.]])
    disc=jnp.array([[1.,float(GAMMA),1.,float(GAMMA)]])
    result=lambda_return(zeros,zeros,rew,zeros,zeros,disc,1.)
    np.testing.assert_allclose(np.asarray(result), [[2*float(GAMMA),2.,2.]],rtol=1e-6)
    one=jnp.full((1,4),float(GAMMA))
    np.testing.assert_array_equal(lambda_return(zeros,zeros,rew,zeros,zeros,one,.95),
                                 lambda_return(zeros,zeros,rew,zeros,zeros,float(GAMMA),.95))
