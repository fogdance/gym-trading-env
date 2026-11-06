"""Thinking Student rule gate (V2.3, user 2026-11-05) with rules V5 (V2.4: "止损固定5点",
"全部：SL = 成交价+5"): the env's rule gate opens an ADD exactly at the rule
positions of the V5 material, on both sides; it is causal (a live market's
newest minute can be one); every gated stop is fixed at the fill +/- 5 with TP
at 2R = 10; the timeline teaches WAIT before a touch's first positive only;
the real costs come from a CZCE_FG cost preset and stay configurable."""
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from gym_trading_env.research import ts_v1_corpus as C
from gym_trading_env.research import ts_v2_positive as V
from gym_trading_env.research.fg_single3_oracle import CandidateIndex, empty_at

DREAMER = Path(__file__).resolve().parents[3] / 'dreamerv3'
CONFIGS = DREAMER / 'configs/thinking_student'
POOL = CONFIGS / 'markets/pool.json'
STRATEGY = CONFIGS / 'markets/strategy_v2_4_fixed_stop.yaml'
STRATEGY_V21 = CONFIGS / 'markets/strategy_v2_1_structural.yaml'
PRESET = CONFIGS / 'markets/costs_fg_real_20261105.yaml'
RULES = {'SHORT': CONFIGS / 'v2_positive_entry_v5.yaml', 'LONG': CONFIGS / 'v2_positive_entry_v5_long.yaml'}
real = pytest.mark.skipif(not (POOL.exists() and STRATEGY.exists()), reason='Thinking Student V2.4 configs missing')
SIDES = {'CZCE.FG205': 'SHORT', 'CZCE.FG401': 'LONG'}
# FG205's inputs in the V2.2 corpus, written before V2.3 (V2.1 structural strategy).
CORPUS_V22 = Path('/data/logdir/fg_single3/thinking_student/corpus_v2_2_20261104/CZCE.FG205/inputs')


@pytest.fixture(scope='module')
def pair(tmp_path_factory):
    """Per contract: the V5 material's own scan (_scan_job) and the gated env."""
    out = {}
    for contract, side in SIDES.items():
        row, cfg = C.train_row(contract, POOL), V.load_config(RULES[side])
        tmp = tmp_path_factory.mktemp(contract)
        _, events, candidates, _, stats = V._scan_job((row, cfg, str(tmp / 'material')))
        env = C.native_env(C.write_inputs(row, tmp / 'gate', strategy=C.load_strategy(STRATEGY)), market=False)
        out[contract] = dict(row=row, cfg=cfg, side=side, events=events, candidates=candidates,
                             end=min(stats['flat_end'], stats['rows'] - 1), env=env, config=tmp / 'gate' / 'env.yaml')
    return out


@real
@pytest.mark.parametrize('contract', sorted(SIDES))
def test_the_gate_opens_add_exactly_at_the_material_rule_positions(pair, contract):
    p = pair[contract]
    env, side, end = p['env'], p['side'], p['end']
    g = env._gate(side)
    cursors = [c['cursor'] for c in p['candidates']]
    assert len(cursors) > 500 and set(np.flatnonzero(g['open'][:end]).tolist()) == set(cursors)
    sign = V.sign(side)
    for c in p['candidates']:  # the same touch line, in short space
        assert g['H'][g['event'][c['cursor']]] == sign * c['obs_H']
        assert c['label_R'] == c['obs_planned_risk'] == 5 and c['obs_sl'] == c['obs_quote'] + sign * 5
    # A flat account may ADD at a rule position (where the ungated env would)
    # and nowhere else.
    rng = np.random.default_rng(5)
    closed = np.flatnonzero(~g['open'][:end])
    seen = dict(opened=0, blocked=0)
    for t in sorted(rng.choice(cursors, 80, replace=False).tolist() + rng.choice(closed, 80, replace=False).tolist()):
        free = add_mask(env, t, gated=False)
        assert add_mask(env, t) == (free and bool(g['open'][t]))
        seen['opened' if g['open'][t] else 'blocked'] += free
    assert seen['opened'] > 60 and seen['blocked'] > 20, seen


def add_mask(env, t, gated=True):
    """mask[1] of a flat account at t, with the env's rule gate or without it."""
    gate = env.rule_gate
    env.rule_gate = gate if gated else None
    try:
        return bool(empty_at(env, t)['action_mask'][1])
    finally:
        env.rule_gate = gate


@real
def test_the_gate_is_causal_and_recomputed_for_a_longer_market(pair):
    p = pair['CZCE.FG205']
    side, full = p['side'], p['env']._gate('SHORT')
    frame = pd.read_csv(p['row']['path'])
    opens = np.flatnonzero(full['open'])
    cuts = [int(opens[40]) + 1, int(opens[len(opens) // 2]) + 1]  # the newest minute is a rule position
    # The env of the full market (its main roll minute is in it), then a
    # shorter and a longer market loaded the way live data arrives.
    env = FixedDirectionEntryEnv(config_path=p['config'])
    for k in cuts:
        env._load_market(frame.iloc[:k].reset_index(drop=True))
        assert env._gates is None
        g = env._gate(side)
        assert g['open'][k - 1] and np.array_equal(g['open'], full['open'][:k])
        assert np.array_equal(g['event'], full['event'][:k])
        assert np.array_equal(g['H'], full['H'][:len(g['H'])])


@real
@pytest.mark.parametrize('contract', sorted(SIDES))
def test_every_gated_stop_is_fixed_at_the_fill(pair, contract):
    p = pair[contract]
    env, side = p['env'], p['side']
    s, g = V.sign(side), env._gate(side)
    for c in p['candidates'][:300]:
        t = c['cursor']
        fill = env._price(env._raw.Open.iloc[t + 1], side, env.bobs[t + 1], opening=True)
        assert float(fill) == float(env._raw.Open.iloc[t + 1]) == c['label_fill_expected']  # no spread
        plan = env.planned_exit(t, fill, side)
        assert plan['rule_event'] == c['event_id'] and plan['touch'] == c['obs_touch']
        assert plan['stop_source'] == 'FIXED' and plan['risk_points'] == '5'
        assert plan['stop_loss_price'] == fill + s * 5 and plan['take_profit_price'] == fill - s * 10
    # A gap fill far beyond the line keeps the fixed stop too ("全部").
    t, line = p['candidates'][0]['cursor'], Decimal(str(p['candidates'][0]['obs_H']))
    far = env.planned_exit(t, line + s * 9, side)
    assert far['stop_source'] == 'FIXED' and far['stop_loss_price'] == line + s * 14
    with pytest.raises(ValueError, match='not a rule position'):
        env.planned_exit(int(np.flatnonzero(~g['open'])[100]), line, side)


@real
def test_the_timeline_waits_before_a_touchs_first_positive_only():
    cfg = V.load_config(RULES['SHORT'])

    def cand(event, cursor, reason):
        return dict(event_id=event, contract='X', cursor=cursor, time=str(cursor), direction_source='D',
                    obs_H=100.0, obs_close=97.0, obs_quote=97.0, obs_quote_below_h=3.0, obs_rv=1.0,
                    obs_rv_activation=2.5, obs_minutes_since_activation=cursor - 10, obs_touch=1,
                    obs_line_spike_rv=2.5, obs_sl=105.0, obs_planned_risk=8.0, obs_planned_tp=81.0,
                    obs_explain='x', reason=reason, label_grade='GOOD' if reason == 'POSITIVE' else '',
                    label_net_pnl=1.0, label_fill_expected=97.0)

    def event(name, activation):
        return dict(event_id=name, contract='X', direction_source='D', H=100.0, level_bucket_end='b',
                    level_confirm=0, level_departed=0, touch=1, line_spike_time='t', line_spike_rv=2.5,
                    activation=activation, activation_time=str(activation), activation_rv=2.5, end=40,
                    end_time='40', end_reason='DEPARTED', cursors=30, wait=[],
                    activation_obs=dict(obs_H=100.0, obs_explain='a'))

    events = [event('A', 10), event('B', 50)]
    # A MASKED minute (an ADD there is not legal) is no WAIT: there is no decision.
    candidates = [cand('A', 10, 'STOP_LOSS'), cand('A', 11, 'TP_SLOW'), cand('A', 12, 'MASKED'),
                  cand('A', 13, 'POSITIVE'), cand('A', 14, 'STOP_LOSS'), cand('A', 15, 'POSITIVE'),
                  cand('B', 51, 'STOP_LOSS')]
    timeline = V.assemble(events, candidates, cfg)['positive_timeline']
    assert list(zip(timeline.cursor, timeline.teach)) == [(10, 'WAIT_IN_RULE'), (11, 'WAIT_IN_RULE'),
                                                          (13, 'ADD'), (15, 'ADD')]
    assert timeline.obs_explain.iloc[0].startswith('rule position: quote 97 (H-3)')
    plain = V.assemble(events, candidates, dict(cfg, timeline={'wait_rows_per_phase': 8}))['positive_timeline']
    assert list(plain.teach) == ['ACTIVATION', 'ADD', 'ADD']


@real
def test_the_real_costs_come_from_a_configurable_preset(pair, tmp_path):
    costs = C.load_costs(str(PRESET), CONFIGS / 'markets' / 'x.yaml')
    assert {k: costs[k] for k in ('open_fee', 'close_fee', 'close_today_fee', 'spread', 'slippage', 'lot_size')} == dict(
        open_fee=2.1, close_fee=2.1, close_today_fee=2.1, spread=0.0, slippage=0.0, lot_size=20.0)
    p = pair['CZCE.FG205']
    env = p['env']
    master = yaml.safe_load((p['config'].parent / 'single3.yaml').read_text())
    assert master['costs'] == costs
    # A rule ADD through env.step: it fills at the next open, carries its
    # touch and the touch's stop, and pays 2.1 CNY per lot each way.
    index = CandidateIndex(env)
    for c in p['candidates']:
        row = index.candidate(c['cursor'])
        if row.get('filled') and row['exit'] is not None:
            break
    t, trade = c['cursor'], row['exit']
    opening = next(f for f in env.fills if f['reason'] == 'ENTRY')
    assert Decimal(row['entry_price']) == Decimal(str(env._raw.Open.iloc[t + 1]))
    assert opening['rule_event'] == c['event_id'] and opening['stop_source'] == 'FIXED'
    assert Decimal(opening['stop_loss_price']) == Decimal(opening['price']) + 5
    assert Decimal(trade['open_fee']) == Decimal(trade['close_fee']) == Decimal('6.3')
    assert Decimal(trade['net_trade_pnl']) == Decimal(trade['gross_trade_pnl']) - Decimal('12.6')
    # Another preset, other costs: nothing about them is fixed in the code.
    other = tmp_path / 'other.yaml'
    other.write_text(yaml.safe_dump(dict(CZCE_FG=dict(lot_size=20, leverage=10, trading_fee_per_lot=1.5,
                                                      is_round_turn=True, spread=1))))
    cheap = C.native_env(C.write_inputs(p['row'], tmp_path / 'cheap', costs=C.load_costs(str(other), tmp_path / 'x')),
                         market=False)
    terms = cheap._terms_at(cheap._raw.index[100])
    assert (terms['open_fee'], terms['close_fee'], terms['spread']) == (Decimal('1.5'), Decimal('1.5'), Decimal('1'))
    # Without costs a strategy keeps the pool master's: V2.1 / V2.2 inputs
    # are written as they were, byte for byte.
    old = C.write_inputs(p['row'], tmp_path / 'v21', strategy=C.load_strategy(STRATEGY_V21))
    pool_env = C.read_yaml(p['row']['env_config'])
    pool_master = C.read_yaml(C.resolve_path(p['row']['env_config'], pool_env['entry']['strategy_config']))
    assert yaml.safe_load((old.parent / 'single3.yaml').read_text())['costs'] == pool_master['costs']
    if CORPUS_V22.exists():
        for name in ('env.yaml', 'single3.yaml', 'contract.yaml'):
            assert (old.parent / name).read_bytes() == (CORPUS_V22 / name).read_bytes(), name
    # A rules file whose costs are not the env's is refused.
    bad = yaml.safe_load((p['config'].parent / 'single3.yaml').read_text())
    bad['costs']['spread'] = 1.0
    (p['config'].parent / 'single3.yaml').write_text(yaml.safe_dump(bad, sort_keys=False))
    try:
        with pytest.raises(ValueError, match="costs are not this env's"):
            C.native_env(p['config'], market=False)
    finally:
        (p['config'].parent / 'single3.yaml').write_text(yaml.safe_dump(master, sort_keys=False))


@real
def test_v5_rules_refuse_a_fixed_stop_beyond_max_risk(tmp_path):
    bad = yaml.safe_load(RULES['SHORT'].read_text())
    bad['entry']['fixed_stop_points'] = 11
    bad.pop('costs')
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match='fixed_stop_points'):
        V.load_config(path)



STRATEGY_FG = CONFIGS / 'markets/strategy_v3_trend_trail_zone_h2.yaml'


@real
@pytest.mark.parametrize('contract', sorted(SIDES))
def test_fg_zone_above_the_line(contract, tmp_path):
    """The FG rules since 2026-11-06 (user: "明白了，FG只扩区间到 H+2， SA保持不变"): the V5 rules with
    zone_above_points 2. Every rule position's quote is within [H - 5, H + 2] (short space), some are
    above the line, none follows an SL quote at H + 5 in its touch, a touch ends INVALIDATED exactly
    where it first reaches H + 5, every touch starts on a 2x bar; the gate opens ADD exactly there,
    with SL H + 5."""
    row, side = C.train_row(contract, POOL), SIDES[contract]
    env = C.native_env(C.write_inputs(row, tmp_path, strategy=C.load_strategy(STRATEGY_FG)), market=False)
    cfg = env.rule_gate[side]
    assert cfg['entry']['zone_above_points'] == 2 and 'touch_break_points' not in cfg['entry']
    a = V.market_arrays(env, cfg)
    levels, _ = V.prior_highs(env, a, cfg)
    r = V.rule_pass(a, levels, cfg, a['n'], env.contract)
    assert np.array_equal(r['in_zone'], env._gate(side)['open'])
    cursors = np.flatnonzero(r['in_zone'])
    H = np.array([r['events'][i]['H'] for i in r['event_of'][cursors]])
    q = a['sell_close'][cursors]
    assert len(cursors) > 200 and ((H - 5 <= q) & (q <= H + 2)).all() and (q > H).any()
    for s, h in zip(cursors, H):
        x = r['events'][r['event_of'][s]]
        assert not (a['valid'][x['activation'] + 1:s + 1] & (a['buy_high'][x['activation'] + 1:s + 1] >= h + 5)).any()
    for x in r['events']:
        t, end = x['activation'], x['end']
        hit = np.flatnonzero(a['valid'][t + 1:end + 1] & (a['buy_high'][t + 1:end + 1] >= x['H'] + 5))
        assert (x['end_reason'] == 'INVALIDATED') == (len(hit) > 0 and t + 1 + hit[0] == end)
    assert all(x['activation_rv'] >= 2 for x in r['events'])
    s = int(cursors[len(cursors) // 2])
    plan = env.planned_exit(s, a['sell_open'][s + 1] * a['sign'], side)
    if plan['stop_source'] == 'RULE':
        assert V.sign(side) * float(plan['stop_loss_price']) == r['events'][r['event_of'][s]]['H'] + 5
