"""V3 long trend exit (user 2026-11-06: "这个是新的策略，做长线，而不是短线"; "入场方式没变"):
under entry.exit_rule there is no TP; the stop of an open position moves with the 15m
troughs (a LONG: peaks) after the fill (research.swing_trail): 0 troughs the gate's own
stop, 8 points past the fill or 1 trough break-even, 2 troughs fill -/+ 5, >= 3 troughs the
last trough but one; it never moves back, and a swing counts only once its confirming
bucket is available."""
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from gym_trading_env.market_structure import analyze_market_structure
from gym_trading_env.research import ts_v1_corpus as C
from gym_trading_env.research.swing_trail import Swings, trail_stop

CONFIGS = Path(__file__).resolve().parents[3] / 'dreamerv3' / 'configs/thinking_student'
POOL = CONFIGS / 'markets/pool.json'
STRATEGY = CONFIGS / 'markets/strategy_v3_trend_trail.yaml'
FIXED = {'SHORT': CONFIGS / 'v2_positive_entry_v5.yaml', 'LONG': CONFIGS / 'v2_positive_entry_v5_long.yaml'}
CFG = dict(kind='swing_trail', timeframe='15m', pivot_left=2, pivot_right=2, breakeven_points=8, lock_points=5)
real = pytest.mark.skipif(not (POOL.exists() and STRATEGY.exists()), reason='Thinking Student V3 configs missing')


def test_trail_stop_stages():
    # Short space: fill 100, initial stop 105 (H + 5 or fill + 5).
    assert trail_stop(100, 105, 'INITIAL', 95, [], CFG) == (105, 'INITIAL')
    assert trail_stop(100, 105, 'INITIAL', 92, [], CFG) == (100, 'BREAKEVEN')  # 8 points past the fill
    assert trail_stop(100, 105, 'INITIAL', 97, [96], CFG) == (100, 'BREAKEVEN')
    assert trail_stop(100, 100, 'BREAKEVEN', 90, [96, 91], CFG) == (95, 'LOCK')
    assert trail_stop(100, 95, 'LOCK', 85, [96, 91, 86], CFG) == (91, 'TROUGH')  # the last trough but one
    assert trail_stop(100, 91, 'TROUGH', 80, [96, 91, 86, 81], CFG) == (86, 'TROUGH')
    # Never back: a shallow second trough keeps the locked stop; a higher candidate is ignored.
    assert trail_stop(100, 95, 'LOCK', 90, [96, 97, 90], CFG) == (95, 'LOCK')
    assert trail_stop(100, 95, 'LOCK', 99, [96], CFG) == (95, 'LOCK')



def test_breakeven_minus_one_then_hold():
    # User 2026-11-08: 8 points from the fill or 1 trough -> fill - 1, then held to the stop or the forced close.
    cfg = dict(CFG, breakeven_offset_points=1, hold_after_breakeven=True)
    assert trail_stop(100, 105, 'INITIAL', 95, [], cfg) == (105, 'INITIAL')
    assert trail_stop(100, 105, 'INITIAL', 92, [], cfg) == (99, 'BREAKEVEN')
    assert trail_stop(100, 105, 'INITIAL', 97, [96], cfg) == (99, 'BREAKEVEN')
    assert trail_stop(100, 99, 'BREAKEVEN', 85, [96, 91], cfg) == (99, 'BREAKEVEN')  # no LOCK
    assert trail_stop(100, 99, 'BREAKEVEN', 80, [96, 91, 86, 81], cfg) == (99, 'BREAKEVEN')  # no TROUGH
    assert trail_stop(100, 105, 'INITIAL', 97, [96, 91, 86], cfg) == (99, 'BREAKEVEN')  # troughs seen at once
    # Without the keys V3 is unchanged.
    assert trail_stop(100, 105, 'INITIAL', 92, [], CFG) == (100, 'BREAKEVEN')
    assert trail_stop(100, 100, 'BREAKEVEN', 90, [96, 91], CFG) == (95, 'LOCK')
    from gym_trading_env.research.swing_trail import check_config
    for bad in (dict(breakeven_offset_points=8), dict(breakeven_offset_points=-1), dict(hold_after_breakeven='yes')):
        with pytest.raises(ValueError, match='breakeven_offset_points'):
            check_config(dict(CFG, **bad))

def test_step_tighten():
    # User 2026-11-08: "+2 -1; +4 -2; +6 -3", on top of break-even -1 and hold. Fill 100, initial stop 105.
    cfg = dict(CFG, breakeven_offset_points=1, hold_after_breakeven=True, step_tighten=[[2, 1], [4, 2], [6, 3]])
    assert trail_stop(100, 105, 'INITIAL', 99, [], cfg, initial=105) == (105, 'INITIAL')
    assert trail_stop(100, 105, 'INITIAL', 98, [], cfg, initial=105) == (104, 'TIGHTEN')
    assert trail_stop(100, 104, 'TIGHTEN', 96, [], cfg, initial=105) == (103, 'TIGHTEN')
    assert trail_stop(100, 103, 'TIGHTEN', 94, [], cfg, initial=105) == (102, 'TIGHTEN')
    assert trail_stop(100, 105, 'INITIAL', 93, [], cfg, initial=105) == (102, 'TIGHTEN')  # several steps in one bar
    assert trail_stop(100, 102, 'TIGHTEN', 92, [], cfg, initial=105) == (99, 'BREAKEVEN')
    assert trail_stop(100, 102, 'TIGHTEN', 97, [96], cfg, initial=105) == (99, 'BREAKEVEN')
    # Never back: the price returning keeps the tighter stop; a later step below a tighter stop is ignored.
    assert trail_stop(100, 103, 'TIGHTEN', 100, [], cfg, initial=105) == (103, 'TIGHTEN')
    assert trail_stop(100, 99, 'BREAKEVEN', 94, [], cfg, initial=105) == (99, 'BREAKEVEN')
    # A small risk: initial 103, +6 -> the fill.
    assert trail_stop(100, 103, 'INITIAL', 94, [], cfg, initial=103) == (100, 'TIGHTEN')
    from gym_trading_env.research.swing_trail import check_config
    for bad in ([[2, 1], [2, 2]], [[2, 2], [4, 1]], [[2, 0]], [[2]], 'x'):
        with pytest.raises(ValueError, match='step_tighten'):
            check_config(dict(CFG, step_tighten=bad))
    with pytest.raises(ValueError, match='initial stop'):
        trail_stop(100, 105, 'INITIAL', 98, [], cfg)


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    """Per contract and initial stop: the always-ADD account of the whole contract."""
    out = {}
    for contract in ('CZCE.FG205', 'CZCE.FG401'):
        for stop in ('H5', 'FIXED'):
            strategy = C.load_strategy(STRATEGY)
            if stop == 'FIXED':
                strategy['entry']['rule_gate'] = {k: str(v) for k, v in FIXED.items()}
            row = C.train_row(contract, POOL)
            env = C.native_env(C.write_inputs(row, tmp_path_factory.mktemp(contract) / stop, strategy=strategy),
                               market=False)
            obs, _ = env.reset()
            while True:
                obs, _, term, trunc, _ = env.step(int(obs['action_mask'][1] > 0.5))
                if trunc and env.status == 'CHUNK_END':
                    env.status, env.truncated = 'ACTIVE', False
                    continue
                if term or trunc:
                    break
            out[contract, stop] = env
    return out


@real
def test_replayed_swings_are_the_prefix_analysis(runs):
    env = runs['CZCE.FG205', 'H5']
    buckets = env._e_market.complete_buckets
    swings = Swings(buckets, 2, 2)
    bars = [dict(open=b['Open'], high=b['High'], low=b['Low'], close=b['Close']) for b in buckets]
    assert len(buckets) > 1000
    for b in range(len(buckets)):
        assert analyze_market_structure(bars[:b + 1])['swings'] == swings.chain(b)


@real
@pytest.mark.parametrize('stop', ['H5', 'FIXED'])
@pytest.mark.parametrize('contract', ['CZCE.FG205', 'CZCE.FG401'])
def test_the_stop_trails_the_known_swings(runs, contract, stop):
    env = runs[contract, stop]
    buckets = env._e_market.complete_buckets
    trades = env.attributions
    assert len(trades) > 50 and not any(t['reason'] == 'TAKE_PROFIT' for t in trades)
    stages = {t['stop_stage'] for t in trades}
    assert {'INITIAL', 'BREAKEVEN', 'LOCK', 'TROUGH'} <= stages, stages
    fills = {f['entry_id']: f for f in env.fills if f['reason'] == 'ENTRY'}
    for t in trades:
        s = 1 if t['side'] == 'SHORT' else -1
        fill, plan = Decimal(t['entry_price']), fills[t['entry_id']]
        initial = Decimal(plan['stop_loss_price'])
        assert plan['take_profit_price'] is None and plan['exit_rule'] == 'swing_trail'
        if stop == 'FIXED':
            assert plan['stop_source'] == 'FIXED' and initial == fill + s * 5
        else:
            assert plan['stop_source'] in ('RULE', 'CAPPED', 'GAP')
        path = [initial] + [Decimal(h['stop']) for h in t['stop_history']]
        assert all(s * (b - a) < 0 for a, b in zip(path, path[1:]))  # only in the position's favour
        final = Decimal(t['final_stop_price'])
        assert final == path[-1]
        assert {'INITIAL': initial, 'BREAKEVEN': fill, 'LOCK': fill - s * 5}.get(t['stop_stage'], final) == final
        if t['reason'] == 'STOP_LOSS':  # at the stop, or worse through a gap
            assert s * (Decimal(t['exit_price']) - final) >= 0
        entry = pd.Timestamp(t['entry_time'])
        for h in t['stop_history']:
            if h['stage'] != 'TROUGH':
                continue
            # A trough after the fill whose confirmation (2 buckets on) had arrived by then.
            key = 'Low' if s > 0 else 'High'
            assert any(Decimal(str(b[key])) == Decimal(h['stop']) and pd.Timestamp(b['bucket_start']) >= entry
                       and j + 2 < len(buckets) and pd.Timestamp(buckets[j + 2]['available_at']) <= pd.Timestamp(h['time'])
                       for j, b in enumerate(buckets)), h


@pytest.mark.skipif(not POOL.exists(), reason='Thinking Student V3 configs missing')
def test_rule_stops_capped_at_five(tmp_path):
    # User 2026-11-08 "止损上限5点": max_risk_points 5 with cap_rule_risk; the rule positions stay.
    path = CONFIGS / 'markets/strategy_v3_trend_trail_be1_step_risk5.yaml'
    row = C.train_row('CZCE.FG205', POOL)
    strategy = C.load_strategy(path)
    assert strategy['entry']['max_risk_points'] == 5 and strategy['entry']['cap_rule_risk'] is True
    env = C.native_env(C.write_inputs(row, tmp_path / 'capped', strategy=strategy), market=False)
    obs, _ = env.reset()
    while True:
        obs, _, term, trunc, _ = env.step(int(obs['action_mask'][1] > 0.5))
        if trunc and env.status == 'CHUNK_END':
            env.status, env.truncated = 'ACTIVE', False
            continue
        if term or trunc:
            break
    plans = [f for f in env.fills if f['reason'] == 'ENTRY']
    assert len(plans) > 50
    for p in plans:
        s = 1 if p['side'] == 'SHORT' else -1
        risk = Decimal(p['risk_points'])
        assert p['stop_source'] in ('RULE', 'CAPPED', 'GAP') and 0 < risk <= 5
        assert Decimal(p['stop_loss_price']) == Decimal(p['price']) + s * risk
        if p['stop_source'] == 'RULE':  # the touch's line H + 5 (LONG L - 5)
            assert Decimal(p['stop_loss_price']) == Decimal(p['line']) + s * 5
        if p['stop_source'] == 'CAPPED':  # the line's stop farther than 5
            assert risk == 5 and s * (Decimal(p['line']) + s * 5 - Decimal(p['price'])) > 5
    assert {p['stop_source'] for p in plans} >= {'RULE', 'CAPPED'}
    strategy = C.load_strategy(path)
    del strategy['entry']['cap_rule_risk']
    with pytest.raises(ValueError, match='risk'):
        C.native_env(C.write_inputs(row, tmp_path / 'refused', strategy=strategy), market=False)
