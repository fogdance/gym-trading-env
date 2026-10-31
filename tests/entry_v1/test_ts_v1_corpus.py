"""Thinking Student V1 teaching material on the synthetic Native market."""
import json
import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from wp02_fixture import native_factory, ts_corpus

from gym_trading_env.research import ts_v1_corpus as C
from gym_trading_env.research.fg_single3_oracle import reference_candidate

CONTRACT = 'CZCE.FG2701'
# The Thinking Student market pool of the sibling dreamerv3 checkout.
POOL = Path(__file__).resolve().parents[3] / 'dreamerv3/configs/thinking_student/markets/pool.json'


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
    return ts_corpus(tmp_path_factory.mktemp('ts_corpus'))


def test_pool_rows_keep_train_and_dev_apart(tmp_path):
    roles = dict(TRAIN=('train', True), DEV=('validation', False), TEST=('test', False))
    pool = tmp_path / 'markets' / 'pool.json'
    pool.parent.mkdir()
    pool.write_text(json.dumps(dict(files=[
        dict(contract=name, role=role, eligible_for_training=eligible, env_config=f'envs/{name}.yaml')
        for name, (role, eligible) in roles.items()])))
    row = C.train_row('TRAIN', pool)
    assert row['env_config'] == str(tmp_path / 'markets' / 'envs' / 'TRAIN.yaml')
    assert C.dev_row('DEV', pool)['role'] == 'validation'
    for name in ('DEV', 'TEST'):
        with pytest.raises(ValueError, match='not a TRAIN member'):
            C.train_row(name, pool)
    for name in ('TRAIN', 'TEST'):
        with pytest.raises(ValueError, match='not a VALIDATION row'):
            C.dev_row(name, pool)


def test_inputs_are_self_contained_and_load_after_the_folder_moved(tmp_path):
    _, config, _ = native_factory(tmp_path / 'market')
    row = dict(contract=CONTRACT, path=str(config.parent / 'market.csv'), env_config=str(config))
    C.write_inputs(row, tmp_path / 'written')
    (tmp_path / 'written').rename(tmp_path / 'moved')
    for name in ('env.yaml', 'contract.yaml', 'directions.yaml'):
        (config.parent / name).unlink()  # only the market data stays outside
    env = C.native_env(tmp_path / 'moved' / 'env.yaml', market=False)
    entry = C.read_yaml(tmp_path / 'moved' / 'env.yaml')['entry']
    assert (entry['contract_config'], entry['direction_config']) == ('contract.yaml', 'directions.yaml')
    assert env.chunk_steps == 0 and env.reward_config.schema == C.QUALITY_TP_SCHEMA


@pytest.mark.skipif(not POOL.exists(), reason='Thinking Student market pool not present')
def test_pool_inputs_with_a_strategy_master_load_after_the_folder_moved(tmp_path):
    for row in (C.train_row('CZCE.FG809', POOL), C.dev_row('CZCE.FG409', POOL)):
        written = C.write_inputs(row, tmp_path / 'written').parent
        written.rename(tmp_path / 'moved')
        env = C.native_env(tmp_path / 'moved' / 'env.yaml', market=False)
        assert env.strategy_path == tmp_path / 'moved' / 'single3.yaml'
        assert env.chunk_steps == 0 and env.reward_config.schema == C.QUALITY_TP_SCHEMA
        assert len(env._raw) == row['rows']
        shutil.rmtree(tmp_path / 'moved')


def test_facts_equal_a_native_minute_trace_at_every_cursor(corpus):
    folder, source = corpus
    facts = C.load_arrays(folder / 'facts')
    env = C.corpus_env(folder, market=False)
    assert env.chunk_steps == 0
    eligible = np.flatnonzero(facts['eligible'])
    for t in eligible:
        assert C.native_check(facts, t, reference_candidate(env, int(t))) is None, t
    closed = facts['filled'] & (facts['exit_cursor'] >= 0)
    assert closed.sum() > 20 and {0, 1} <= set(facts['kind'][closed].tolist()), 'needs TP and SL'
    for t in np.flatnonzero(closed):
        events = facts['reward_events']
        assert facts['reward'][t] == events[events[:, 0] == t][:, 2].sum(), t
    # Production replayed its own sample, and a changed fact is caught.
    assert source['native_check']['agree'] == len(source['native_check']['cursors']) > 0
    t = int(np.flatnonzero(closed)[0])
    broken = {**facts, 'exit_cursor': facts['exit_cursor'].copy()}
    broken['exit_cursor'][t] += 1
    assert C.native_check(broken, t, reference_candidate(env, t)) == 'exit_cursor'


def test_other_tp_sl_distances_reach_the_facts_grades_and_inputs(tmp_path, corpus):
    """TP / SL are configurable (user 2026-10-31): the corpus inputs carry the
    distances and every indexed fact is the Native minute trace at them."""
    rules = dict(take_profit_points=4, stop_loss_points=15)
    folder, source = ts_corpus(tmp_path, rules=rules)
    assert source['inputs']['rules'] == rules
    assert corpus[1]['inputs']['rules'] == dict(take_profit_points=5, stop_loss_points=30)
    entry = C.read_yaml(folder / 'inputs' / 'env.yaml')['entry']
    assert {k: entry[k] for k in rules} == rules
    env = C.corpus_env(folder, market=False)
    facts = C.load_arrays(folder / 'facts')
    closed = facts['filled'] & (facts['exit_cursor'] >= 0)
    assert {0, 1} <= set(facts['kind'][closed].tolist()), 'needs TP and SL'
    for t in np.flatnonzero(facts['eligible']):
        native = reference_candidate(env, int(t))
        assert C.native_check(facts, t, native) is None, t
        trade = native.get('exit')
        if trade and trade['reason'] in ('TAKE_PROFIT', 'STOP_LOSS'):
            sign = 1 if trade['side'] == 'LONG' else -1
            move = sign * (float(trade['exit_price']) - float(trade['entry_price']))
            assert (move == 4 if trade['reason'] == 'TAKE_PROFIT' else move <= -15), (t, move)
    default = C.load_arrays(corpus[0] / 'facts')
    assert not np.array_equal(default['exit_cursor'], facts['exit_cursor'])
    # The graded teacher checks each TP against the TP distance of the env.
    graded = C.load_arrays(folder / 'graded')
    assert (graded['grade'][closed & (facts['kind'] == 1)] == 0).all(), 'an SL is POOR'


def test_always_add_account_trades_exactly_the_facts(corpus):
    folder, source = corpus
    facts, p1 = C.load_arrays(folder / 'facts'), C.load_arrays(folder / 'behaviors' / 'P1')
    floor = source['funds_floor']
    adds = np.flatnonzero(p1['action'] == 1)
    assert len(adds) > 3
    for row in adds:
        t = int(p1['cursor'][row])
        assert p1['account_native'][row, 0] == 1 and p1['equity'][row] >= floor
        assert facts['eligible'][t] and facts['filled'][t]
        held = np.flatnonzero(p1['account_native'][row + 1:, 0] == 1)
        if not len(held):
            continue
        exit_row = row + 1 + held[0]
        assert p1['cursor'][exit_row] == facts['exit_cursor'][t]
        assert p1['reward'][row:exit_row].astype(np.float64).sum() == pytest.approx(
            facts['reward'][t], abs=1e-6)
    # A funded flat account can ADD exactly where the flat-account fact is eligible.
    for name in C.BEHAVIORS:
        b = C.load_arrays(folder / 'behaviors' / name)
        flat = (b['account_native'][:, 0] == 1) & (b['equity'] >= floor) & ~b['is_last']
        cursors = b['cursor'][flat]
        np.testing.assert_array_equal(b['action_mask'][flat, 1] > 0.5, facts['eligible'][cursors])


def test_strict_plan_account_and_labels(corpus):
    folder, source = corpus
    strict, p4 = C.load_arrays(folder / 'strict'), C.load_arrays(folder / 'behaviors' / 'P4')
    np.testing.assert_array_equal(p4['cursor'][p4['action'] == 1], strict['plan'])
    assert len(strict['plan']) > 0 and source['behaviors']['P4']['exits'] == {
        'TAKE_PROFIT': len(strict['plan'])}
    # Every planned ADD is a strict ADD choice from the flat account.
    assert strict['action'][strict['plan']].all() and strict['valid'][strict['plan']].all()
    facts = C.load_arrays(folder / 'facts')
    assert (strict['valid'] == facts['eligible']).all()
    assert strict['tie'][facts['eligible'] & ~facts['filled']].all()


def test_strict_choice_and_ties_are_the_exact_add_and_wait_values(corpus):
    """An independent exact DP over every filled trade, not only the TP
    trades the optimizer keeps: ADD exactly where its value is larger, a tie
    exactly where both values are equal."""
    folder, source = corpus
    facts, strict = C.load_arrays(folder / 'facts'), C.load_arrays(folder / 'strict')
    clock = C.corpus_env(folder, market=False).market_clock_counter
    gamma, end, events = Fraction(332, 333), source['flat_end'], facts['reward_events']
    value, add, wait = [Fraction(0)] * (end + 1), {}, {}
    for t in range(end - 1, -1, -1):
        wait[t] = gamma ** clock.advance(t, t + 1) * value[t + 1]
        value[t] = wait[t]
        if not facts['eligible'][t]:
            continue
        add[t] = wait[t]
        if facts['filled'][t]:
            add[t] = sum((gamma ** clock.reward_exponent(t, int(s)) * Fraction(float(v))
                          for _, s, v in events[events[:, 0] == t]), Fraction(0))
            if facts['exit_cursor'][t] >= 0:
                u = int(facts['exit_cursor'][t])
                add[t] += gamma ** clock.advance(t, u) * value[u]
        value[t] = max(wait[t], add[t])
    assert set(add) == set(np.flatnonzero(strict['valid']).tolist())
    for t in add:
        assert strict['action'][t] == (add[t] > wait[t]), t
        assert strict['tie'][t] == (add[t] == wait[t]), t
    losing = [t for t in add if facts['filled'][t] and add[t] < wait[t]]
    assert losing and not strict['tie'][losing].any(), 'a losing ADD is a strict WAIT'


def test_a_trade_without_feedback_ties_only_when_it_blocks_nothing():
    from types import SimpleNamespace
    from gym_trading_env.utils.s3_market_clock import MarketClock

    def trade(t, reason, release, feedback=()):
        return dict(cursor=t, eligible=True, filled=True, rewards={'C1': [list(f) for f in feedback]},
                    exit=dict(reason=reason, net_trade_pnl='0'), release_cursor=release)
    rows = [trade(0, 'MAIN_ROLL', 5),                      # blocks the TP trade at 2
            trade(1, 'STOP_LOSS', 2, [(2, -0.5)]),         # loses
            trade(2, 'TAKE_PROFIT', 3, [(3, 1.0)]),
            dict(cursor=3, eligible=True, filled=False, rewards={'C1': []}, exit=None,
                 release_cursor=None),                      # rejected: equals WAIT
            trade(6, 'MAIN_ROLL', 8)]                       # nothing to block before 8
    env = SimpleNamespace(reward_config=SimpleNamespace(candidate='C1'),
                          market_clock_counter=MarketClock(np.ones(10, bool)))
    strict, _ = C.strict_arrays(env, rows, 9, 10)
    assert np.flatnonzero(strict['valid']).tolist() == [0, 1, 2, 3, 6]
    assert np.flatnonzero(strict['action']).tolist() == [2]
    assert np.flatnonzero(strict['tie']).tolist() == [3, 6]


def test_market_rows_are_account_free_and_one_per_cursor(corpus):
    folder, source = corpus
    market = C.load_arrays(folder / 'market')
    n = source['rows']
    assert all(market[k].shape[0] == n for k in C.MARKET_KEYS)
    for name in C.BEHAVIORS:
        b = C.load_arrays(folder / 'behaviors' / name)
        np.testing.assert_array_equal(b['cursor'], np.arange(len(b['cursor'])))
        assert b['is_first'][0] and b['is_last'][-1] and b['is_last'].sum() == 1
        assert b['account_native'].shape == (len(b['cursor']), 14)
    # Another account sees the same market observation at the same cursor.
    arrays, views, _ = C.run_behavior(
        C.corpus_env(folder), C.Behavior(C.BEHAVIORS['P2_mid']), market=True)
    for key in C.MARKET_KEYS:
        np.testing.assert_array_equal(views[key], market[key][arrays['cursor']])
    assert (arrays['action'] == 1).any()
