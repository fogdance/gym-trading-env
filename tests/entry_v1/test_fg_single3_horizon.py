import copy
import numpy as np
import pytest

from gym_trading_env.research.fg_single3_horizon import fair_pair,discounted_feedback
from test_fg_single3_env import single_env
from gym_trading_env.utils.s3_market_clock import MarketClock


def test_same_account_fair_fork_uses_same_causal_suffix_and_first_feedback_zero(tmp_path):
    prices=[(1000,1000,1000,1000),(1000,1008,1000,1007),
            (1000,1000,1000,1000),(1000,1008,1000,1007),
            (1000,1000,1000,1000),(1000,1000,960,980)]
    env=single_env(tmp_path,prices,fees=3,chunk=2)
    state=copy.deepcopy(env.export_state())
    pair=fair_pair(env,state)
    assert pair['add']['starting_account_hash']==pair['wait']['starting_account_hash']==pair['snapshot_hash']
    assert pair['add']['first_action']==1 and pair['wait']['first_action']==0
    assert pair['add']['first_new_entry_time']!=pair['wait']['first_new_entry_time']
    assert pair['add']['subsequent_policy']==pair['wait']['subsequent_policy']=='frozen_causal_P1'
    assert pair['add']['counts']['STOP_LOSS']==pair['wait']['counts']['STOP_LOSS']==1
    assert env.export_state()==state
    value=float(np.float32(60/61))
    assert discounted_feedback([[1,value]],0,MarketClock(np.ones(1002,bool)))['333']==value
    assert discounted_feedback([[1001,-6.]],0,MarketClock(np.ones(1002,bool)))['333']==pytest.approx(-6*(332/333)**1000)
    with pytest.raises(ValueError,match='follow'):
        discounted_feedback([[0,1.]],0,MarketClock(np.ones(1002,bool)))
