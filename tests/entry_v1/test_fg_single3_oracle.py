"""Exact optimization is checked against independent original-account action enumeration."""
from fractions import Fraction
import itertools

import numpy as np
import pytest

from gym_trading_env.research.fg_single3_oracle import (CandidateIndex, GAMMA, empty_at,
    funds_certificate, optimize, reference_candidate, verify_candidates)
from test_fg_single3_env import single_env


def test_index_pending_gap_and_reference_boundaries(tmp_path):
    prices=[(1000,1000,1000,1000),(1000,1000,965,980),
            (1010,1010,1010,1010),(1000,1000,1000,1000),
            (950,951,949,950),(1000,1008,1000,1007),(1000,1000,1000,1000)]
    env=single_env(tmp_path,prices,fees=3,spread=1,chunk=2)
    env._raw.loc[env._raw.index[1],'Volume']=5
    index=CandidateIndex(env)
    rows=[index.candidate(t) for t in range(index.flat_end)]
    receipt=verify_candidates(index,rows,count=100)
    assert receipt['status']=='PASS'
    assert rows[0]['exit']['reason']=='STOP_LOSS' and rows[0]['exit']['sl_pending_retry']
    assert rows[2]['exit']['gap_slippage_points']>0
    assert rows[-1]['right_censored']


@pytest.mark.parametrize('side', ['LONG','SHORT'])
def test_exact_dp_matches_independent_real_env_exhaustive_actions(tmp_path,side):
    prices=[(1000,1000,1000,1000),(1000,1007,1000,1006),
            (1002,1003,991,1001),(1000,1007,995,1007),
            (1001,1003,970,995),(995,1003,994,1000)]
    env=single_env(tmp_path,prices,side=side,fees=3,chunk=2)
    index=CandidateIndex(env)
    rows=[index.candidate(t) for t in range(index.flat_end)]
    verify_candidates(index,rows,count=100)
    for candidate in ('C0','C1','C2','C3'):
        directory=tmp_path/candidate;directory.mkdir()
        original=single_env(directory,prices,side=side,candidate=candidate,fees=3,chunk=2)
        for gamma in (GAMMA,Fraction(1)):
            optimum=Fraction(0)
            # Enumerate requested actions, keeping the original funding/margin
            # and masked rejection behavior. No candidate/DP logic is used here.
            for actions in itertools.product((0,1),repeat=len(prices)-1):
                empty_at(original,0);total=Fraction(0)
                for t,action in enumerate(actions):
                    if original.truncated:original.reset()
                    if original.terminated:break
                    _,r,_,_,_=original.step(action)
                    total+=gamma**t*Fraction.from_float(float(np.float32(r)))
                optimum=max(optimum,total)
            plan=optimize(rows,index.flat_end,candidate,gamma,clock=index.clock)
            empty_at(original,0);total=Fraction(0)
            for t in range(len(prices)-1):
                if original.truncated:original.reset()
                if original.terminated:break
                _,r,_,_,_=original.step(int(t in plan['add_cursors']))
                total+=gamma**t*Fraction.from_float(float(np.float32(r)))
            assert total==optimum
            assert plan['value_interval'][0]<=float(optimum)<=plan['value_interval'][1]
            assert plan['oracle_optimality_status']=='exact_proven'


def test_proof_failure_and_exact_tie_fail_closed(tmp_path):
    env=single_env(tmp_path,[(50000,50000,50000,50000)]*3)
    with pytest.raises(ValueError,match='NOT_PROVEN'):
        funds_certificate(env)
    # ADD and WAIT both lead to exactly the same future float32 feedback.
    rows=[dict(cursor=0,filled=True,exit={'reason':'TAKE_PROFIT','net_trade_pnl':'282'},
               release_cursor=2,rewards={'C0':[[2,1.]]}),
          dict(cursor=1,filled=True,exit={'reason':'TAKE_PROFIT','net_trade_pnl':'282'},
               release_cursor=2,rewards={'C0':[[2,1.]]})]
    plan=optimize(rows,2,'C0',clock=env.market_clock_counter)
    assert plan['add_cursors']==[1] and plan['exact_ties']>=1
    with pytest.raises(ValueError,match='budget'):
        optimize(rows,2,'C0',clock=env.market_clock_counter,max_states=2)


def test_index_global_priority_with_metadata_and_insufficient_volume(tmp_path):
    env=single_env(tmp_path,[(1000,1000,1000,1000)]*8,
        main_exit='2026-01-05 09:34',zero_duration_indices=(3,),chunk=2,fees=3)
    env._raw.loc[env._raw.index[4],'Volume']=2
    index=CandidateIndex(env)
    rows=[index.candidate(t) for t in range(index.flat_end)]
    assert verify_candidates(index,rows,count=100)['status']=='PASS'
    assert rows[0]['exit']['reason']=='MAIN_ROLL' and rows[0]['exit_cursor']==5
    assert rows[0]['exit']['net_trade_pnl']=='-18'
    assert optimize(rows,index.flat_end,'C2',clock=index.clock)['add_cursors']==[]
