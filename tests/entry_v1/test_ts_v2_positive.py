"""Thinking Student V2 positive entry corpus: causal levels and events, the
Native facts of the positives, the teacher reader and the TRAIN SHORT scope."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from gym_trading_env.research import ts_v2_positive as V
from gym_trading_env.research.fg_single3_oracle import reference_candidate
from gym_trading_env.research.ts_v1_corpus import native_env, read_yaml, train_row, write_inputs

DREAMER = Path(__file__).resolve().parents[3] / 'dreamerv3'
POOL = DREAMER / 'configs/thinking_student/markets/pool.json'
CONFIG = DREAMER / 'configs/thinking_student/v2_positive_entry.yaml'
SHORT = {'CZCE.FG809', 'CZCE.FG901', 'CZCE.FG905', 'CZCE.FG005', 'CZCE.FG201', 'CZCE.FG205',
         'CZCE.FG209', 'CZCE.FG301', 'CZCE.FG405'}
real = pytest.mark.skipif(not (POOL.exists() and CONFIG.exists()), reason='Thinking Student pool not present')

# The rules of v2_positive_entry.yaml, so the synthetic tests stand alone.
CFG = yaml.safe_load("""
version: V2_POSITIVE_ENTRY_V1
direction: SHORT
direction_source: DIRECTION_ASSUMED
prior_high: {pivot_left: 2, pivot_right: 2, lookback_buckets: 46, departure_points: 10}
volume: {baseline_minutes: 20, ratio: 2.0, exclude_segment_open: true}
event: {near_points: 5, window_minutes: 60}
entry: {zone_points: 5, stop_above_points: 5, max_risk_points: 10, target_r: 2.0, parallel_r: 1.5}
quality: {excellent: {mae_r: 0.5, held_minutes: 120}, good: {mae_r: 0.8, held_minutes: 345}}
timeline: {wait_rows_per_phase: 8}
figures: {max_events: 12, per_contract: 2}
""")


def _bars(price, high=None, n=15, volume=100.):
    return [(price, price if high is None else high, price - 1, price, volume)] * n


def synthetic(spike=(96, 99, 96, 98, 500.)):
    """15-bar buckets: a pivot high H = 100 (bucket 2), a departure to 87, a
    rebound (far, approach, near on low volume), a volume spike in the zone
    at cursor 96, three more in-zone minutes on low volume, then a drop."""
    bars = (_bars(88, 90) + _bars(88, 90) + _bars(95, 100) + _bars(93, 95) + _bars(92, 94)
            + _bars(87, 89) + [(89, 89, 88, 89, 100.)] + _bars(92, 93, 3) + _bars(96, 97, 2)
            + [spike, (98, 99, 97, 98, 100.), (98, 98, 96, 97, 100.),
               (97, 97, 96, 97, 100.), (97, 97, 92, 93, 100.)] + _bars(92, 93, 4)
            + _bars(88, 89) + _bars(85, 86) * 3)
    o, h, lo, c, v = (np.array(x, float) for x in zip(*bars))
    n = len(bars)
    times = pd.date_range('2027-01-04 09:01', periods=n, freq='min', tz='Asia/Shanghai')
    valid = np.ones(n, bool)
    a = dict(n=n, times=times, bobs=times - pd.Timedelta(minutes=1), times_ns=times.asi8,
             bob_ns=(times - pd.Timedelta(minutes=1)).asi8, open=o, high=h, low=lo, close=c, volume=v,
             valid=valid, segment_open=np.zeros(n, bool), sell_close=c - 1, sell_open=o - 1, buy_high=h + 1,
             rv=V.volume_ratio(v, valid, 20), prefix=np.arange(n + 1), multiplier=20.)
    buckets = [dict(High=float(h[i:i + 15].max()), bucket_end=times[i + 14], available_at=times[i + 14])
               for i in range(0, n - 14, 15)]
    return a, buckets


def run(a, buckets, cfg=CFG):
    env = SimpleNamespace(_e_market=SimpleNamespace(complete_buckets=buckets))
    levels, _ = V.prior_highs(env, a, cfg)
    return V.scan(a, levels, cfg, a['n'], 'SYN')


def truncate(a, buckets, m):
    """The market as it was with bars [0, m) only."""
    cut = {k: (v[:m + 1] if k == 'prefix' else v[:m]) for k, v in a.items() if k not in ('n', 'multiplier')}
    last = a['times'][m - 1]
    return (dict(cut, n=m, multiplier=a['multiplier']),
            [b for b in buckets if pd.Timestamp(b['available_at']) <= last])


def assert_causal(a, buckets, cursors, cfg=CFG):
    events, candidates = run(a, buckets, cfg)
    for m in cursors:
        e2, c2 = run(*truncate(a, buckets, m), cfg)
        known = [x for x in events if x['activation'] < m - 1]
        assert [x['event_id'] for x in e2] == [x['event_id'] for x in known]
        for x, y in zip(known, e2):
            assert (y['H'], y['activation_rv'], y['wait']) == (x['H'], x['activation_rv'], x['wait'])
            if x['end'] < m - 1:
                assert (y['end'], y['end_reason']) == (x['end'], x['end_reason'])
        # Frames compare NaN (RV of a segment-open minute) as equal.
        pd.testing.assert_frame_equal(pd.DataFrame(c2), pd.DataFrame([c for c in candidates if c['cursor'] < m - 1]))
    return events, candidates


def test_volume_ratio_skips_ineligible_bars_and_needs_a_full_baseline():
    volume = np.array([10., 999., 10., 10., 30., 10.])
    eligible = np.array([True, False, True, True, True, True])
    rv = V.volume_ratio(volume, eligible, 3)
    assert np.isnan(rv[:4]).all()
    assert rv[4] == 3. and rv[5] == pytest.approx(10 / (50 / 3))


def test_one_event_activates_on_the_spike_and_keeps_later_low_volume_minutes():
    a, buckets = synthetic()
    events, candidates = run(a, buckets)
    assert len(events) == 1
    x = events[0]
    assert (x['H'], x['activation'], x['end'], x['end_reason']) == (100., 96, 105, 'DEPARTED')
    assert x['activation_rv'] == 5.
    assert [(w['cursor'], w['teach']) for w in x['wait']] == [
        (90, 'WAIT_FAR'), (91, 'WAIT_APPROACH'), (92, 'WAIT_APPROACH'), (93, 'WAIT_APPROACH'),
        (94, 'WAIT_NEAR_NO_VOLUME'), (95, 'WAIT_NEAR_NO_VOLUME')]
    assert [c['cursor'] for c in candidates] == [96, 97, 98, 99]
    assert candidates[0]['obs_rv'] >= 2 and all(c['obs_rv'] < 2 for c in candidates[1:])
    assert [c['obs_minutes_since_activation'] for c in candidates] == [0, 1, 2, 3]
    for c in candidates:
        s = c['cursor']
        # The fill is the next bar's open less the spread, never a bar High.
        assert c['label_fill_expected'] == a['open'][s + 1] - 1 and c['label_fill_cursor'] == s + 1
        assert c['obs_quote'] == a['close'][s] - 1 and c['obs_sl'] == 105.
        assert c['label_R'] == 105. - c['label_fill_expected'] and c['label_fill_in_zone']
        assert not any(k.startswith('obs_') and 'fill' in k for k in c)


def test_a_level_is_never_used_before_it_is_confirmed_or_after_it_broke():
    a, buckets = synthetic()
    env = SimpleNamespace(_e_market=SimpleNamespace(complete_buckets=buckets))
    levels, _ = V.prior_highs(env, a, CFG)
    level = next(x for x in levels if x['H'] == 100.)
    # Visible at the close of the second bucket after the pivot, departed at
    # the first close at H - 10.
    assert (level['start'], level['confirm'], level['departed']) == (45, 74, 75)
    # A spike whose short SL quote reaches H + 5 breaks the level instead.
    a, buckets = synthetic(spike=(96, 104, 96, 98, 500.))
    assert run(a, buckets) == ([], [])


def test_the_scan_is_causal_on_a_synthetic_market():
    a, buckets = synthetic()
    events, _ = assert_causal(a, buckets, range(40, a['n'] + 1))
    assert len(events) == 1


@pytest.fixture(scope='module')
def fg809(tmp_path_factory):
    folder = tmp_path_factory.mktemp('fg809')
    env = native_env(write_inputs(train_row('CZCE.FG809', POOL), folder), market=True)
    cfg = V.load_config(CONFIG)
    a = V.market_arrays(env, cfg)
    return env, a, env._e_market.complete_buckets, cfg


@real
def test_the_scan_is_causal_on_a_real_train_market(fg809):
    _, a, buckets, cfg = fg809
    events, candidates = run(a, buckets, cfg)
    assert events and candidates
    # Cut inside events, just after activations and at a random cursor.
    cuts = sorted({events[0]['activation'] + 2, events[len(events) // 2]['activation'] + 1,
                   candidates[len(candidates) // 3]['cursor'] + 1,
                   int(np.random.default_rng(0).integers(1000, a['n']))})
    assert_causal(a, buckets, cuts, cfg)


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp('v2') / 'corpus'
    rows = V.short_rows(POOL, ['CZCE.FG809'])
    V.produce(rows, V.load_config(CONFIG), out, workers=8, log=lambda *_: None)
    return out


@real
def test_positives_keep_their_risk_and_their_fee_inclusive_pnl(corpus):
    entries = V.load_positive(corpus)['positive_entries']
    cfg = V.load_config(CONFIG)
    assert len(entries) and set(entries.label_grade) <= {'EXCELLENT', 'GOOD'}
    inputs = corpus / 'inputs' / 'CZCE.FG809' / 'base'
    contract = read_yaml(inputs / 'contract.yaml')
    lot, lots, (terms,) = float(contract['lot_size']), V.LOTS, contract['terms']
    assert terms['close_fee'] == terms['close_today_fee']
    fee = lots * (terms['open_fee'] + terms['close_fee'])
    for _, e in entries.iterrows():
        assert 5 <= e.label_R <= 10 and e.obs_sl == e.obs_H + 5 == e.label_sl
        assert e.obs_H - 5 <= e.label_fill_expected <= e.obs_H
        assert e.label_tp == e.label_fill_expected - 2 * e.label_R
        assert e.label_exit_reason == 'TAKE_PROFIT' and e.label_exit_price == e.label_tp
        assert e.label_net_pnl == pytest.approx((e.label_fill_expected - e.label_exit_price) * lot * lots - fee)
        assert e.label_mae_points <= cfg['quality']['good']['mae_r'] * e.label_R
        assert e.label_held_minutes <= cfg['quality']['good']['held_minutes']
        assert e.label_native_verified


@real
def test_positives_match_their_native_minute_trace(corpus):
    entries = V.load_positive(corpus)['positive_entries']
    for _, e in entries.groupby('label_R').head(1).iterrows():
        R = int(e.label_R)
        env = native_env(corpus / 'inputs' / 'CZCE.FG809' / f'R{R}_TP{2 * R}' / 'env.yaml', market=False)
        assert (env.stop_loss_points, env.take_profit_points) == (R, 2 * R)
        trace = reference_candidate(env, int(e.cursor))
        assert trace['filled'] and trace['exit']['reason'] == 'TAKE_PROFIT'
        assert float(trace['exit']['entry_price']) == e.label_fill_expected
        assert trace['exit_cursor'] == e.label_exit_cursor
        assert float(trace['exit']['net_trade_pnl']) == e.label_net_pnl


@real
def test_the_reader_returns_teacher_tables_only(corpus, tmp_path):
    out = V.load_positive(corpus)
    assert set(out) == set(V.TEACHER)
    for name, frame in out.items():
        for c in frame.columns:
            assert c in V.META or c.startswith(('obs_', 'label_')) or name == 'positive_events', (name, c)
            assert not (c.startswith('obs_') and any(w in c for w in ('grade', 'quality', 'pnl', 'exit'))), c
    assert not any(c.startswith('label_') for c in out['positive_timeline'].columns)
    assert 'reason' not in out['positive_entries'].columns
    # Each entry says why it is placed there (decision-time facts only) and
    # why it is graded so (offline).
    entries = out['positive_entries']
    for note, why, grade in zip(entries.obs_explain, entries.label_grade_reason, entries.label_grade):
        assert note.startswith('quote ') and not any(w in note for w in ('fill', 'Native', 'MAE', 'GOOD'))
        assert f'{grade}: ' in why and 'Native TAKE_PROFIT' in why
    assert out['positive_timeline'].obs_explain.notna().all()
    assert set(out['positive_timeline'].teach) <= set(V.TEACH)
    events = out['positive_events']
    assert 'net_pnl' not in events.columns and (events.positives >= 1).all()
    candidates = pd.read_csv(corpus / 'audit' / 'candidates.csv')
    assert (candidates.reason == 'POSITIVE').sum() == len(out['positive_entries']) < len(candidates)
    # Rows of a WAIT phase are decided before activation; nothing after
    # activation is taught as WAIT.
    timeline = out['positive_timeline'].merge(events[['event_id', 'activation']], on='event_id')
    wait = timeline.teach.str.startswith('WAIT')
    assert (timeline.cursor[wait] < timeline.activation[wait]).all()
    assert (timeline.cursor[~wait] >= timeline.activation[~wait]).all()
    summary = json.loads((corpus / 'audit' / 'audit_summary.json').read_text())
    assert summary['total']['reasons']['POSITIVE'] == len(out['positive_entries'])
    # A teacher table that carries an outcome among its inputs is refused.
    bad = tmp_path / 'bad'
    (bad / 'teacher').mkdir(parents=True)
    for name, frame in out.items():
        frame.assign(**({'obs_net_pnl': 0.} if name == 'positive_entries' else {})).to_csv(
            bad / 'teacher' / f'{name}.csv', index=False)
    with pytest.raises(ValueError, match='not a decision-time input'):
        V.load_positive(bad)


@real
def test_only_short_train_contracts_are_read():
    assert {r['contract'] for r in V.short_rows(POOL)} == SHORT
    longs = [f['contract'] for f in json.loads(POOL.read_text())['files']
             if f.get('role') == 'train' and f['contract'] not in SHORT]
    with pytest.raises(ValueError, match='not SHORT contracts'):
        V.short_rows(POOL, ['CZCE.FG809', longs[0]])
    with pytest.raises(ValueError, match='not a TRAIN member'):
        V.short_rows(POOL, ['CZCE.FG409'])


def test_the_config_must_be_the_short_version(tmp_path):
    for change in (dict(direction='LONG'), dict(version='V0')):
        path = tmp_path / 'cfg.yaml'
        path.write_text(yaml.safe_dump(dict(CFG, **change)))
        with pytest.raises(ValueError, match='not a V2_POSITIVE_ENTRY_V1 SHORT config'):
            V.load_config(path)
