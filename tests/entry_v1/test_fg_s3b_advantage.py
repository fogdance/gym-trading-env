"""Independent FULL original-account action enumeration, including loss cash."""
import copy
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from test_fg_single3_env import single_env
from gym_trading_env.research.fg_single3_oracle import CandidateIndex, GAMMA
from gym_trading_env.research.fg_s3b_advantage import (
    ExactBudget, RelaxedValues, build_value_evidence, certify_advantage,
    funding_terms, rational_interval, verify_counterfactual)


def candidates(env):
    index = CandidateIndex(env)
    rows = [index.candidate(t) for t in range(index.flat_end)]
    return rows, RelaxedValues(rows, index.flat_end), funding_terms(env)


def prepare(env, state):
    env.import_state(state)
    if env.truncated and env.status == 'CHUNK_END':
        env.reset()
    return env.export_state()


def terminal(state):
    return state['terminated'] or state['status'] in ('DATA_END', 'DATA_GAP')


def gold(env, state, seen=None):
    """Never uses candidates/DP; legal original Env actions with actual ledger."""
    state = prepare(env, state)
    if terminal(state):
        return Fraction(0)
    env.import_state(state)
    mask = env._get_obs()['action_mask']
    options = {}
    for action in np.flatnonzero(mask):
        env.import_state(state)
        old = env.current_step
        _, reward, _, _, _ = env.step(int(action))
        following = env.export_state()
        value = Fraction.from_float(float(np.float32(reward)))
        if env.current_step > old and not terminal(following):
            value += GAMMA*gold(env, following, seen)
        options[int(action)] = value
    if seen is not None and not state['entries'] and mask.all():
        seen.append((state, options))
    return max(options.values())


def reduced_cash(env, amount):
    env._initialized = False
    env.reset(seed=0)
    state = env.export_state()
    cash = Decimal(str(amount))
    difference = Decimal(state['balances']['user_cash'])-cash
    state['balances']['user_cash'] = str(cash)
    state['balances']['broker_pnl'] = str(Decimal(state['balances']['broker_pnl'])+difference)
    state['realized_pnl'] = str(-difference)
    env.import_state(state)
    return env.export_state()


@pytest.mark.parametrize('side', ['LONG', 'SHORT'])
def test_all_reachable_account_actions_enclosed_and_actual_forks(tmp_path, side):
    prices = [(1000,1000,1000,1000),(1000,1008,965,1007),
              (1000,1007,989,1007),(1000,1007,1000,1007),
              (1000,1000,970,980),(1000,1008,1000,1007)]
    env = single_env(tmp_path, prices, side=side, candidate='C1', fees=3, chunk=2)
    rows, dp, terms = candidates(env)
    env._initialized = False
    env.reset(seed=0)
    states = []
    gold(env, env.export_state(), states)
    checked = 0
    for state, truth in states:
        t = state['current_step']
        if t >= dp.end:
            continue
        cash = state['balances']['user_cash']
        e = build_value_evidence(rows[t], dp, cash, terms['required_cash'])
        if e['q_add'] is not None:
            for action, key in ((0, 'q_wait'), (1, 'q_add')):
                lo, hi = e[key]
                # Exact rational comparison, not a float epsilon oracle.
                assert Fraction.from_float(lo) <= truth[action] <= Fraction.from_float(hi)
            assert e['margin_lower'] <= float(abs(truth[1]-truth[0]))
            if e['certificate'] == 'ADD_BETTER': assert truth[1] > truth[0]
            if e['certificate'] == 'WAIT_BETTER': assert truth[1] < truth[0]
            if e['certificate'] == 'TIE_EXACT': assert truth[1] == truth[0]
        fork = verify_counterfactual(env, state, rows[t])
        assert fork['status'] == 'PASS' and fork['full_actual_account_snapshot']
        checked += 1
    assert checked >= 5


def test_stop_loss_limited_future_is_only_upper_bound(tmp_path):
    # SL completes without margin call; a later higher-price entry needs more
    # cash than the post-SL account. Fresh-account relaxed suffix overestimates.
    prices = [(1000,1000,1000,1000),(1000,1000,969,970),
              (1100,1108,1100,1107),(1100,1108,1100,1107)]
    env = single_env(tmp_path, prices, candidate='C1', fees=3)
    rows, dp, terms = candidates(env)
    state = reduced_cash(env, 8000)
    fork = verify_counterfactual(env, state, rows[0])
    assert rows[0]['exit']['reason'] == 'STOP_LOSS' and Decimal(fork['cash_after']) < 6600
    unknown = build_value_evidence(rows[0], dp, 8000, terms['required_cash'])
    assert unknown['q_add'] is None and unknown['certificate'] == 'UNCERTIFIED'
    evidence = build_value_evidence(rows[0], dp, 8000, terms['required_cash'], branch_verified=True)
    assert evidence['proof'] == 'CERTIFIED_SIGN_ONLY'
    seen = []
    gold(env, state, seen)
    actual = next(q for s, q in seen if s['current_step'] == 0)
    assert Fraction.from_float(evidence['q_add'][0]) <= actual[1] < Fraction.from_float(evidence['q_add'][1])
    assert evidence['certificate'] == 'WAIT_BETTER' and evidence['margin_lower'] > 0
    continuation = list(dp.chain(int(dp.head[rows[0]['release_cursor']])))
    with pytest.raises(ValueError,match='not actually filled|restricted real mask'):
        verify_counterfactual(env,state,rows[0],continuation=continuation)
    # The same original forced SL, with sufficient actual cash, has an attained
    # relaxed continuation and therefore receives EXACT_Q.
    funded = build_value_evidence(rows[0], dp, 300000, terms['required_cash'])
    assert funded['proof'] == 'EXACT_Q'
    full_state = reduced_cash(env,300000)
    witness = verify_counterfactual(env,full_state,rows[0],continuation=continuation)
    assert witness['suffix_add'] == continuation and witness['status'] == 'PASS'
    env.import_state(state)
    corrupt = copy.deepcopy(rows[0]); corrupt['exit']['reason'] = 'TAKE_PROFIT'
    with pytest.raises(ValueError, match='mismatch'):
        verify_counterfactual(env, state, corrupt)


def test_pending_main_roll_nonstandard_rejection_and_censor(tmp_path):
    prices = [(1000,1000,1000,1000),(1000,1000,965,980),
              (1010,1010,990,990),(1000,1000,1000,1000)]
    env = single_env(tmp_path, prices, candidate='C1', fees=3, chunk=1,
                     main_exit='2026-01-05 09:33')
    env._raw.loc[env._raw.index[1], 'Volume'] = 5
    rows, dp, terms = candidates(env)
    env._initialized = False; env.reset(seed=0)
    fork = verify_counterfactual(env, env.export_state(), rows[0])
    assert fork['status'] == 'PASS' and rows[0]['exit']['reason'] == 'MAIN_ROLL'
    e = build_value_evidence(rows[0], dp, 300000, terms['required_cash'])
    assert e['proof'] == 'EXACT_Q' and e['q_add'][1] < 0
    other = tmp_path/'other'; other.mkdir()
    env = single_env(other, [(1000,1000,1000,1000)]*4, candidate='C1', chunk=2)
    env._raw.loc[env._raw.index[1], 'Volume'] = 2
    rows, dp, terms = candidates(env)
    env._initialized = False; env.reset(seed=0)
    state = env.export_state()
    fork = verify_counterfactual(env, state, rows[0])
    assert fork['rejection_equivalent']
    e = build_value_evidence(rows[0], dp, 300000, terms['required_cash'], teacher_action=0)
    assert e['certificate'] == 'TIE_EXACT' and e['branch'] == 'REJECTED'
    e = build_value_evidence(rows[1], dp, 300000, terms['required_cash'])
    assert e['q_add'] is None and e['proof'] == 'CENSORED'


def test_exact_tie_budget_overlap_teacher_conflict_and_numeric_bounds():
    def row(t):
        return dict(cursor=t, eligible=True, filled=True, entry_cursor=t+1,
                    release_cursor=2, exit_cursor=2, observed_until_cursor=2,
                    right_censored=False, exit=dict(reason='TAKE_PROFIT', committed=True,
                        closed_quantity=3, net_trade_pnl='282'), rewards={'C1':[[2, 1.]]})
    rows = [row(0), row(1)]
    dp = RelaxedValues(rows, 2)
    e = build_value_evidence(rows[0], dp, 300000, 1000, teacher_action=0)
    assert e['certificate'] == 'TIE_EXACT' and e['delta'] == [0.,0.]
    dp = RelaxedValues(rows, 2, budget=ExactBudget(0,8))
    e = build_value_evidence(rows[0], dp, 300000, 1000, teacher_action=0)
    assert e['certificate'] == 'UNCERTIFIED' and e['margin_lower'] == 0
    assert dp.budget.exhausted > 0
    assert certify_advantage([0.,2.],[1.,3.])['certificate'] == 'UNCERTIFIED'
    with pytest.raises(ValueError, match='Teacher conflict'):
        certify_advantage([2.,3.],[0.,1.],0)
    with pytest.raises(ValueError, match='Invalid value'):
        certify_advantage([float('nan'),3.],[0.,1.])
    lo, hi = rational_interval(GAMMA)
    assert Fraction.from_float(lo) <= GAMMA <= Fraction.from_float(hi)
    malformed = copy.deepcopy(rows); malformed[0]['rewards']['C1'] = [[0,1.]]
    with pytest.raises(ValueError, match='timing'):
        RelaxedValues(malformed,2)


def test_early_penalty_plus_tp_discount_and_administrative_clock(tmp_path):
    prices = [(1000,1000,1000,1000),(1000,1000,989,1000),
              (1000,1008,1000,1007),(1000,1008,1000,1007)]
    env = single_env(tmp_path, prices, candidate='C1', chunk=1, fees=3)
    rows, dp, terms = candidates(env)
    assert rows[0]['rewards']['C1'][0][1] == -.5
    assert rows[0]['rewards']['C1'][1][1] > 0
    env._initialized = False; env.reset(seed=0)
    state = env.export_state()
    fork = verify_counterfactual(env, state, rows[0])
    exact = sum((GAMMA**(s-1)*Fraction.from_float(v) for s,v in fork['local_rewards']),Fraction(0))
    lo, hi = dp.local_value(rows[0])
    assert Fraction.from_float(lo) <= exact <= Fraction.from_float(hi)
    assert rows[0]['release_cursor'] == 2
    assert len(fork['local_rewards']) == 2  # reset contributed no discount step


def test_exact_budget_event_limit_fails_closed():
    b = ExactBudget(2,1)
    assert b.difference([(1,1.),(2,2.)],[],0) is None and b.exhausted == 1
