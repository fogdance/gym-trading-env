"""Thinking Student V2 minimal change (V0.5, user 2026-10-31): the Native
env's structural stop and 2R target, the net PnL reward, the longer 15m
context, the strategy overlay and the V2 positive teacher of a corpus. The
V1 rules and identities stay as they were."""
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from gym_trading_env.research import ts_v1_corpus as C
from gym_trading_env.research import ts_v2_positive as V
from gym_trading_env.research.fg_single3_oracle import CandidateIndex, empty_at, reference_candidate
from gym_trading_env.rewards.single3_reward import NET_PNL_SCHEMA, Single3RewardConfig

DREAMER = Path(__file__).resolve().parents[3] / 'dreamerv3'
POOL = DREAMER / 'configs/thinking_student/markets/pool.json'
STRATEGY = DREAMER / 'configs/thinking_student/markets/strategy_v2_short_structural.yaml'
MATERIAL = Path('/data/logdir/fg_single3/thinking_student/V2_1_POSITIVE_ENTRY_20261031')
CORPUS = Path('/data/logdir/fg_single3/thinking_student/corpus_v2_minimal_20261031')
real = pytest.mark.skipif(not (POOL.exists() and MATERIAL.exists()), reason='pool or V2_1 material missing')
SCALE = 618.
# FG809 V1 inputs under the code before V0.5 (dev 0e1ef3b): nothing of V1 moves.
V1_IDENTITY = dict(
    config_hash='3cea5c55fa19944e337a1976749d22fbcd62a935650c3fd22d0a915880be08d4',
    observation_identity='a44954169ff619105758cf423706b0aeebe3b6966dcf7f0006ff95c30fb28a33',
    reward_hash='0e00322430aa8760d798ed6192076badd6d54cf316c3ee72ad5b29a31775175b')


@pytest.fixture(scope='module')
def fg809(tmp_path_factory):
    row = C.train_row('CZCE.FG809', POOL)
    v2 = C.write_inputs(row, tmp_path_factory.mktemp('v2'), strategy=C.load_strategy(STRATEGY))
    entries = V.load_positive(MATERIAL)['positive_entries']
    return dict(row=row, config=v2, env=C.native_env(v2, market=False),
                entries=entries[entries.contract == 'CZCE.FG809'])


def _fill(env, s):
    return env._price(env._raw.Open.iloc[s + 1], 'SHORT', env.bobs[s + 1], opening=True)


@real
def test_the_stop_is_the_prior_high_of_the_material_or_capped(fg809):
    env, entries = fg809['env'], fg809['entries']
    assert env.stop_rule['max_risk_points'] == 10 and env.context_15m_buckets == 48
    assert env.fill_model == 'minute_structural_risk_first_v1'
    seen = set()
    for e in entries.itertuples():
        fill = _fill(env, int(e.cursor))
        assert float(fill) == e.label_fill_expected
        p = env.planned_exit(int(e.cursor), fill, 'SHORT')
        if e.label_R <= 10:  # SL = H + 5 of the material's line, TP = fill - 2R
            assert p['stop_source'] == 'STRUCTURAL' and float(p['stop_loss_price']) == e.obs_sl
            assert float(p['take_profit_price']) == e.label_tp
        else:  # R 11-12 of the material: capped at 10 below the line
            assert p['stop_source'] == 'CAPPED' and p['stop_loss_price'] - fill == 10
            assert fill - p['take_profit_price'] == 20 and float(p['line']) == e.obs_H
        seen.add(p['stop_source'])
    assert seen == {'STRUCTURAL', 'CAPPED'}


@real
def test_without_a_line_at_or_above_the_fill_the_stop_is_max_risk(fg809):
    env = fg809['env']
    lines = env._stop_lines()
    for s in range(200, len(env._raw) - 1, 97):
        fill = _fill(env, s)
        live = ((lines['confirm'] <= s) & (lines['departed'] <= s) & (s < lines['broken'])
                & (s < lines['expire']))
        if not (live & (lines['H'] >= float(fill))).any():
            p = env.planned_exit(s, fill, 'SHORT')
            assert p['stop_source'] == 'NO_LINE' and p['line'] is None
            assert (p['stop_loss_price'], p['take_profit_price']) == (fill + 10, fill - 20)
            break
    else:
        pytest.fail('no cursor without a line')
    with pytest.raises(ValueError, match='SHORT only'):
        env.planned_exit(s, fill, 'LONG')


@real
def test_an_add_trades_its_planned_exit_and_is_paid_its_net_pnl(fg809):
    env, entries = fg809['env'], fg809['entries']
    e = entries[entries.label_R <= 10].iloc[0]
    trace = reference_candidate(env, int(e.cursor))
    entry = next(f for f in env.fills if f['reason'] == 'ENTRY')
    assert entry['stop_source'] == 'STRUCTURAL' and float(entry['stop_loss_price']) == e.obs_sl
    assert float(entry['take_profit_price']) == e.label_tp
    assert trace['exit']['reason'] == 'TAKE_PROFIT' and trace['exit_cursor'] == e.label_exit_cursor
    net = float(trace['exit']['net_trade_pnl'])
    assert net == e.label_net_pnl == (e.label_fill_expected - e.label_tp) * 60 - 18
    # The only reward of the trade: its net PnL / 618 at the exit step.
    assert trace['rewards'] == [[e.label_exit_cursor, float(np.float32(net / SCALE))]]
    fact = C.candidate_facts(env, CandidateIndex(env), C.official_day_index(env), int(e.cursor))
    assert fact['facts']['exit_cursor'] == e.label_exit_cursor and fact['facts']['net_pnl'] == net
    assert fact['rewards']['C1'] == trace['rewards']


@real
def test_a_stop_loss_pays_the_planned_loss(fg809):
    env = fg809['env']
    index = CandidateIndex(env)
    for t in range(300, len(env._raw), 211):
        row = index.candidate(t)
        if row.get('filled') and row['exit'] and row['exit']['reason'] == 'STOP_LOSS':
            entry = next(f for f in env.fills if f['reason'] == 'ENTRY')
            loss = Decimal(row['exit']['exit_price']) - Decimal(entry['price'])
            assert loss >= Decimal(entry['risk_points']) and Decimal(entry['risk_points']) <= 10
            assert row['rewards']['C1'] == [[row['exit_cursor'], float(np.float32(
                float(row['exit']['net_trade_pnl']) / SCALE))]]
            assert float(row['exit']['net_trade_pnl']) <= -(5 * 60 + 18)
            return
    pytest.fail('no stop loss found')


@real
def test_the_15m_context_is_48_buckets_whose_newest_24_are_the_v1_context(fg809, tmp_path):
    v1 = C.native_env(C.write_inputs(fg809['row'], tmp_path / 'v1'))
    v2 = C.native_env(fg809['config'])
    assert v2.observation_space['market_context_15m'].shape == (48, 7)
    for t in (100, 5000, len(v2._raw) // 2):
        a, b = empty_at(v1, t), empty_at(v2, t)
        np.testing.assert_array_equal(b['market_context_15m'][24:], a['market_context_15m'])
        np.testing.assert_array_equal(b['market_seq_e_shared'], a['market_seq_e_shared'])
        np.testing.assert_array_equal(b['market_macd_5m'], a['market_macd_5m'])
    assert b['market_context_15m'][:24, 6].all()  # older buckets are real, not padding


@real
def test_the_48_bucket_context_and_the_stop_are_causal(fg809):
    """Other prices and volumes after the fill bar t + 1 change neither the
    observation at t (the 48 x 7 context included) nor the planned exit of
    an ADD decided at t."""
    config = fg809['config']
    full = C.native_env(config)
    frame = pd.read_csv(full.config.trading.data_path)
    for t in (len(full._raw) // 3, 2 * len(full._raw) // 3):
        obs, fill = empty_at(full, t), _fill(full, t)
        plan = full.planned_exit(t, fill, 'SHORT')
        last = full._raw.index[t + 1].tz_convert('Asia/Shanghai').tz_localize(None)
        later = frame.copy()
        future = pd.to_datetime(later.Date) > last
        later.loc[future, ['Open', 'High', 'Low', 'Close']] += 40
        later.loc[future, 'Volume'] *= 3
        other = FixedDirectionEntryEnv(df=later, config_path=config, e_profile=full._e_profile)
        assert other._raw.High.iloc[t + 2] != full._raw.High.iloc[t + 2]
        for k, v in empty_at(other, t).items():
            np.testing.assert_array_equal(v, obs[k], err_msg=k)
        assert other.planned_exit(t, _fill(other, t), 'SHORT') == plan


@real
def test_the_v1_rules_and_identities_are_unchanged(fg809, tmp_path):
    env = C.native_env(C.write_inputs(fg809['row'], tmp_path / 'v1'), market=False)
    assert env.stop_rule is None and env.context_15m_buckets == 24 and env._e_market is None
    assert (env.take_profit_points, env.stop_loss_points) == (5, 30)
    assert env.fill_model == 'minute_tp5_sl30_risk_first_v1'
    assert dict(config_hash=env.config_hash, observation_identity=env.observation_identity,
                reward_hash=env.reward_config.hash) == V1_IDENTITY


def test_the_net_pnl_reward_needs_its_scale_and_only_it_does():
    Single3RewardConfig('C1', .5, .4, 3., NET_PNL_SCHEMA, reward_scale_cny=618)
    with pytest.raises(ValueError):
        Single3RewardConfig('C1', .5, .4, 3., NET_PNL_SCHEMA)
    with pytest.raises(ValueError):
        Single3RewardConfig('C1', .5, .4, 3., reward_scale_cny=618)
    with pytest.raises(ValueError):
        Single3RewardConfig('C1', .5, .4, 3., NET_PNL_SCHEMA, reward_scale_cny=0)


@real
def test_the_strategy_refuses_a_long_contract_and_unknown_keys(tmp_path):
    long_row = C.train_row('CZCE.FG909', POOL)
    with pytest.raises(ValueError, match='not traded SHORT'):
        C.write_inputs(long_row, tmp_path / 'long', strategy=C.load_strategy(STRATEGY))
    bad = tmp_path / 'bad.yaml'
    bad.write_text(STRATEGY.read_text().replace('direction: SHORT', 'direction: LONG'))
    with pytest.raises(ValueError, match='SHORT strategy'):
        C.load_strategy(bad)
    config = C.write_inputs(C.train_row('CZCE.FG809', POOL), tmp_path / 'v2',
                            strategy=C.load_strategy(STRATEGY))
    env = C.read_yaml(config)
    del env['entry']['target_r']
    broken = tmp_path / 'v2' / 'broken.yaml'
    broken.write_text(json.dumps(env))
    with pytest.raises(ValueError, match='stop_rule'):
        C.native_env(broken, market=False)


@pytest.mark.skipif(not (CORPUS / 'CZCE.FG809' / 'source.json').exists(), reason='V2 corpus not built')
def test_the_corpus_teacher_is_the_material_and_nothing_else():
    folder = CORPUS / 'CZCE.FG809'
    graded, facts = C.load_arrays(folder / 'graded'), C.load_arrays(folder / 'facts')
    tables = V.load_positive(MATERIAL)
    entries = tables['positive_entries'][tables['positive_entries'].contract == 'CZCE.FG809']
    timeline = tables['positive_timeline'][tables['positive_timeline'].contract == 'CZCE.FG809']
    summary = json.loads((folder / 'graded' / 'summary.json').read_text())
    assert summary['dropped']['STOP_LINE_DIFFERS'] == 0  # FG809: the env takes every material line
    kept = entries[entries.label_R <= 10]
    adds = np.flatnonzero(graded['valid'] & (graded['action'] == 1))
    assert adds.tolist() == sorted(kept.cursor)
    assert (facts['kind'][adds] == C.KINDS.index('TP')).all()
    np.testing.assert_array_equal(facts['exit_cursor'][kept.cursor], kept.label_exit_cursor)
    assert (graded['ready_at'][kept.cursor] == kept.label_exit_cursor).all()
    waits = timeline[timeline.teach.isin(V.WAIT_ROWS) & timeline.event_id.isin(set(kept.event_id))]
    assert np.flatnonzero(graded['valid'] & (graded['action'] == 0)).tolist() == sorted(waits.cursor)
    # ACTIVATION rows, the R > 10 positives and every other minute: no demonstration.
    others = timeline[timeline.teach == 'ACTIVATION'].cursor.tolist() + entries[entries.label_R > 10].cursor.tolist()
    assert not graded['valid'][others].any()
    assert graded['valid'].sum() == len(kept) + len(waits)
    assert summary['label']['source'] == 'V2_POSITIVE_TEACHER' and summary['grades']['max_risk_points'] == 10
    source = json.loads((folder / 'source.json').read_text())
    assert source['inputs']['reward_schema'] == NET_PNL_SCHEMA
    # The exit rule recorded is the structural stop, not the unused TP5 / SL30.
    entry = C.load_strategy(STRATEGY)['entry']
    assert source['inputs']['rules'] == {'stop_rule': {k: v for k, v in entry.items() if k != 'context_15m_buckets'}}
    assert np.load(folder / 'market' / 'market_context_15m.npy', mmap_mode='r').shape[1:] == (48, 7)


@real
def test_the_corpus_cli_refuses_mixed_exit_rules_and_teachers(tmp_path):
    base = ['--contract', 'CZCE.FG809', '--pool', str(POOL), '--out', str(tmp_path)]
    for extra in (['--strategy', str(STRATEGY), '--teacher', str(MATERIAL), '--take-profit-points', '5'],
                  ['--teacher', str(MATERIAL)],
                  ['--strategy', str(STRATEGY), '--teacher', str(MATERIAL), '--grades', 'x.yaml']):
        with pytest.raises(SystemExit):
            C.main(base + extra)
    assert not any(tmp_path.iterdir())
