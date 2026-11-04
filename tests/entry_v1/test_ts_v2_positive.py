"""Thinking Student V2 positive entry corpus: causal levels and events, the
Native facts of the positives, the teacher reader, the TRAIN scope of each
side, the LONG mirror (V2.1), the combined SHORT + LONG material and the
per-touch volume spike of rules V3 (V2.2)."""
import json
import shutil
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
CONFIGS_V3 = [DREAMER / f'configs/thinking_student/v2_positive_entry_v3{x}.yaml' for x in ('', '_long')]
SHORT = {'CZCE.FG809', 'CZCE.FG901', 'CZCE.FG905', 'CZCE.FG005', 'CZCE.FG201', 'CZCE.FG205',
         'CZCE.FG209', 'CZCE.FG301', 'CZCE.FG405'}
LONG = {'CZCE.FG001', 'CZCE.FG009', 'CZCE.FG101', 'CZCE.FG105', 'CZCE.FG109', 'CZCE.FG305',
        'CZCE.FG309', 'CZCE.FG401', 'CZCE.FG909'}
real = pytest.mark.skipif(not (POOL.exists() and CONFIG.exists()), reason='Thinking Student pool not present')

# The rules of v2_positive_entry.yaml, so the synthetic tests stand alone.
CFG = yaml.safe_load("""
version: V2_POSITIVE_ENTRY_V2
direction: SHORT
direction_source: DIRECTION_ASSUMED
prior_high: {pivot_left: 2, pivot_right: 2, lookback_buckets: 115, departure_points: 10}
volume: {baseline_minutes: 20, ratio: 2.0, exclude_segment_open: true}
event: {near_points: 5}
entry: {zone_points: 7, stop_above_points: 5, max_risk_points: 12, target_r: 2.0, parallel_r: 1.5}
quality: {excellent: {mae_r: 0.5, held_minutes: 120}, good: {mae_r: 0.8, held_minutes: 345}}
timeline: {wait_rows_per_phase: 8}
figures: {max_events: 12, per_contract: 2}
""")
# Rules V3 (V2.2, user 2026-11-04): each touch needs its own volume spike near H.
CFG3 = dict(CFG, version='V2_POSITIVE_ENTRY_V3', volume=dict(CFG['volume'], activation='touch'))


def _bars(price, high=None, n=15, volume=100.):
    return [(price, price if high is None else high, price - 1, price, volume)] * n


QUIET_PEAK = _bars(95, 100)
# The pivot bucket with a volume spike at its high (cursor 37).
LOUD_PEAK = _bars(95, 100, 7) + [(95, 100, 94, 95, 500.)] + _bars(95, 100, 7)
# After the first touch departs (cursor 105): a quiet return near H at
# cursors 110-112, a second departure at 113.
RETURN = _bars(88, 89, 5) + [(96, 97, 95, 96, 100.)] * 3 + _bars(88, 89, 7) + _bars(85, 86) * 3
# The same return with a volume spike near H at cursor 111.
LOUD_RETURN = (_bars(88, 89, 5) + [(96, 97, 95, 96, 100.), (96, 97, 95, 96, 500.), (96, 97, 95, 96, 100.)]
               + _bars(88, 89, 7) + _bars(85, 86) * 3)


def synthetic(spike=(96, 99, 96, 98, 500.), peak=QUIET_PEAK, tail=_bars(88, 89) + _bars(85, 86) * 3):
    """15-bar buckets: a pivot high H = 100 (bucket 2), a departure to 87, a
    rebound (far, approach, near on low volume), a volume spike in the zone
    at cursor 96, three more in-zone minutes on low volume, then a drop."""
    bars = (_bars(88, 90) + _bars(88, 90) + peak + _bars(93, 95) + _bars(92, 94)
            + _bars(87, 89) + [(89, 89, 88, 89, 100.)] + _bars(92, 93, 3) + _bars(96, 97, 2)
            + [spike, (98, 99, 97, 98, 100.), (98, 98, 96, 97, 100.),
               (97, 97, 96, 97, 100.), (97, 97, 92, 93, 100.)] + _bars(92, 93, 4) + tail)
    o, h, lo, c, v = (np.array(x, float) for x in zip(*bars))
    n = len(bars)
    times = pd.date_range('2027-01-04 09:01', periods=n, freq='min', tz='Asia/Shanghai')
    valid = np.ones(n, bool)
    a = dict(n=n, times=times, bobs=times - pd.Timedelta(minutes=1), times_ns=times.asi8,
             bob_ns=(times - pd.Timedelta(minutes=1)).asi8, open=o, high=h, low=lo, close=c, volume=v,
             valid=valid, segment_open=np.zeros(n, bool), sell_close=c - 1, sell_open=o - 1, buy_high=h + 1,
             rv=V.volume_ratio(v, valid, 20), prefix=np.arange(n + 1), multiplier=20., side='SHORT', sign=1)
    buckets = [dict(High=float(h[i:i + 15].max()), bucket_end=times[i + 14], available_at=times[i + 14])
               for i in range(0, n - 14, 15)]
    return a, buckets


def run(a, buckets, cfg=CFG):
    env = SimpleNamespace(_e_market=SimpleNamespace(complete_buckets=buckets))
    levels, _ = V.prior_highs(env, a, cfg)
    return V.scan(a, levels, cfg, a['n'], 'SYN')


def truncate(a, buckets, m):
    """The market as it was with bars [0, m) only."""
    whole = ('n', 'multiplier', 'side', 'sign')
    cut = {k: (v[:m + 1] if k == 'prefix' else v[:m]) for k, v in a.items() if k not in whole}
    last = a['times'][m - 1]
    return (dict(cut, n=m, **{k: a[k] for k in whole[1:]}),
            [b for b in buckets if pd.Timestamp(b['available_at']) <= last])


def assert_causal(a, buckets, cursors, cfg=CFG):
    events, candidates = run(a, buckets, cfg)
    for m in cursors:
        e2, c2 = run(*truncate(a, buckets, m), cfg)
        known = [x for x in events if x['activation'] < m - 1]
        assert [x['event_id'] for x in e2] == [x['event_id'] for x in known]
        for x, y in zip(known, e2):
            # NaN (RV of a segment-open minute) compares equal here.
            np.testing.assert_equal([y[k] for k in ('H', 'touch', 'line_spike', 'activation_rv', 'wait')],
                                    [x[k] for k in ('H', 'touch', 'line_spike', 'activation_rv', 'wait')])
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
    assert x['activation_rv'] == 5. and (x['touch'], x['line_spike']) == (1, 96)
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
        assert (c['obs_touch'], c['obs_line_spike_rv']) == (1, 5.)


def test_a_line_is_touched_again_after_a_departure_without_a_new_spike():
    a, buckets = synthetic(tail=RETURN)
    events, candidates = run(a, buckets)
    assert [(x['H'], x['touch'], x['activation'], x['end'], x['end_reason']) for x in events] == [
        (100., 1, 96, 105, 'DEPARTED'), (100., 2, 110, 113, 'DEPARTED')]
    second = events[1]
    # The return has no volume; the line's spike (cursor 96) carries it.
    assert second['activation_rv'] < 2 and second['line_spike'] == 96
    # Its WAIT starts after the first touch ended; nothing near H waits.
    assert [w['teach'] for w in second['wait']] == ['WAIT_FAR'] and second['wait'][0]['cursor'] > 105
    late = [c for c in candidates if c['event_id'] == second['event_id']]
    assert [c['cursor'] for c in late] == [110, 111, 112]
    assert {c['obs_touch'] for c in late} == {2} and all(c['obs_rv'] < 2 for c in late)
    assert_causal(a, buckets, range(100, a['n'] + 1, 3))


def test_a_spike_at_the_high_itself_marks_the_line():
    # The pivot bucket's own spike: the first rebound near H needs no new
    # one, so no minute near H waits for volume.
    a, buckets = synthetic(spike=(96, 99, 96, 98, 100.), peak=LOUD_PEAK)
    events, candidates = run(a, buckets)
    x = events[0]
    assert (x['touch'], x['line_spike'], x['activation']) == (1, 37, 94)
    assert 'WAIT_NEAR_NO_VOLUME' not in {w['teach'] for w in x['wait']}
    assert candidates[0]['cursor'] == 94 and candidates[0]['obs_line_spike_rv'] == 5.
    # Without that spike the same rebound waits for the spike at cursor 96.
    a, buckets = synthetic(spike=(96, 99, 96, 98, 500.))
    assert run(a, buckets)[0][0]['activation'] == 96


def test_a_touch_takes_the_lowest_armed_line():
    a, buckets = synthetic()
    env = SimpleNamespace(_e_market=SimpleNamespace(complete_buckets=buckets))
    levels, _ = V.prior_highs(env, a, CFG)
    # A second line, known and departed with the line at 100, never broken.
    line = dict(levels[0], spike=0, expire=a['n'], broken=a['n'])
    above = V.scan(a, levels + [dict(line, bucket=-1, H=104.)], CFG, a['n'], 'SYN')[0]
    assert [(x['H'], x['activation']) for x in above] == [(100., 96)]
    below = V.scan(a, levels + [dict(line, bucket=-1, H=97.)], CFG, a['n'], 'SYN')[0]
    # The first bar within 5 of 97 starts a touch of the line at 97.
    assert (below[0]['H'], below[0]['activation']) == (97., 91)


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
    rows = V.side_rows(POOL, 'SHORT', ['CZCE.FG809'])
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
    e_ = cfg['entry']
    for _, e in entries.iterrows():
        assert e_['stop_above_points'] <= e.label_R <= e_['max_risk_points']
        assert e.obs_sl == e.obs_H + e_['stop_above_points'] == e.label_sl
        assert e.obs_H - e_['zone_points'] <= e.label_fill_expected <= e.obs_H
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
def test_only_train_contracts_of_the_side_are_read():
    assert {r['contract'] for r in V.side_rows(POOL, 'SHORT')} == SHORT
    assert {r['contract'] for r in V.side_rows(POOL, 'LONG')} == LONG
    with pytest.raises(ValueError, match='not SHORT contracts'):
        V.side_rows(POOL, 'SHORT', ['CZCE.FG809', 'CZCE.FG401'])
    with pytest.raises(ValueError, match='not LONG contracts'):
        V.side_rows(POOL, 'LONG', ['CZCE.FG401', 'CZCE.FG809'])
    with pytest.raises(ValueError, match='not a TRAIN member'):
        V.side_rows(POOL, 'SHORT', ['CZCE.FG409'])


def test_the_config_must_be_the_version_and_one_side(tmp_path):
    path = tmp_path / 'cfg.yaml'
    path.write_text(yaml.safe_dump(dict(CFG, direction='LONG')))
    assert V.load_config(path)['direction'] == 'LONG'
    for change in (dict(direction='BOTH'), dict(version='V0')):
        path.write_text(yaml.safe_dump(dict(CFG, **change)))
        with pytest.raises(ValueError, match='V2_POSITIVE_ENTRY_V2 / V2_POSITIVE_ENTRY_V3 SHORT or LONG config'):
            V.load_config(path)
    # The version names its volume rule: V3 with activation touch, V2 without.
    path.write_text(yaml.safe_dump(CFG3))
    assert V.load_config(path)['volume']['activation'] == 'touch'
    for cfg in (dict(CFG3, volume=CFG['volume']), dict(CFG, volume=CFG3['volume'])):
        path.write_text(yaml.safe_dump(cfg))
        with pytest.raises(ValueError, match='needs volume.activation'):
            V.load_config(path)
    with pytest.raises(ValueError, match='not SHORT or LONG'):
        V.sign('BOTH')


@pytest.fixture(scope='module')
def fg401(tmp_path_factory):
    folder = tmp_path_factory.mktemp('fg401')
    env = native_env(write_inputs(train_row('CZCE.FG401', POOL), folder), market=True)
    cfg = dict(V.load_config(CONFIG), direction='LONG')
    return env, V.market_arrays(env, cfg), cfg


@real
def test_a_long_market_is_scanned_as_its_mirror_in_short_space(fg401):
    """V2.1 (user 2026-11-04): LONG is the SHORT rules on the negated market.
    Its lows are the short-space highs, its buy quotes the short-space sell
    quotes, its sell quote at the low the quote its SL checks."""
    env, a, cfg = fg401
    raw = env._raw
    assert (a['side'], a['sign']) == ('LONG', -1)
    for key, column in (('open', 'Open'), ('high', 'Low'), ('low', 'High'), ('close', 'Close')):
        np.testing.assert_array_equal(a[key], -raw[column].to_numpy(float))
    for t in np.random.default_rng(0).integers(0, a['n'], 50):
        at = raw.index[t]
        assert a['sell_close'][t] == -float(env._price(raw.Close.iloc[t], 'LONG', at, opening=True))
        assert a['buy_high'][t] == -float(env._price(raw.Low.iloc[t], 'LONG', at, opening=False))
    # Buy above the ask: the short-space quote lies below the negated close, as a SHORT sell quote does.
    assert (a['sell_close'] <= a['close']).all()
    levels, _ = V.prior_highs(env, a, cfg)
    lows = [-float(b['Low']) for b in env._e_market.complete_buckets]
    assert levels and all(x['H'] == lows[x['bucket']] for x in levels)
    # A line in short space is a 15m pivot low: no lower low within 2 buckets either side.
    for x in levels[:50]:
        j = x['bucket']
        assert x['H'] == max(lows[max(0, j - 2):j + 3])
    events, candidates = run(a, env._e_market.complete_buckets, cfg)
    assert events and candidates
    # Converted back to real prices: the line L is a real low, the SL L - 5 below it, the quote in [L, L+7], the
    # risk the fill above the SL, the planned TP 2R above the quote.
    e = cfg['entry']
    assert all(V._real(dict(x), -1)['H'] == -x['H'] > 0 for x in events)
    for c in candidates:
        r = V._real(dict(c), a['sign'])
        L = r['obs_H']
        assert L == -c['obs_H'] > 0 and r['obs_sl'] == L - e['stop_above_points']
        assert L <= r['obs_quote'] <= L + e['zone_points']
        assert r['label_fill_expected'] - r['obs_sl'] == c['label_R']
        assert r['obs_planned_tp'] == r['obs_quote'] + e['target_r'] * r['obs_planned_risk']


@real
def test_two_sides_combine_into_one_material_byte_for_byte(corpus, tmp_path):
    # The FG809 material and a copy of it under another contract name and the LONG rules.
    other = tmp_path / 'long'
    shutil.copytree(corpus, other)
    for name in V.TEACHER:
        f = other / 'teacher' / f'{name}.csv'
        f.write_text(f.read_text().replace('CZCE.FG809', 'CZCE.FG401'))
    params = yaml.safe_load((other / 'params.yaml').read_text())
    (other / 'params.yaml').write_text(yaml.safe_dump(dict(params, direction='LONG'), sort_keys=False))
    out = tmp_path / 'both'
    source = V.combine([corpus, other], out)
    for name in V.TEACHER:
        a, b = ((f / 'teacher' / f'{name}.csv').read_text().splitlines(keepends=True) for f in (corpus, other))
        assert (out / 'teacher' / f'{name}.csv').read_text() == ''.join(a + b[1:])
    both = V.load_positive(out)
    assert set(both['positive_events'].contract) == {'CZCE.FG809', 'CZCE.FG401'}
    assert [(x['direction'], x['contracts']) for x in source['combined']] == [
        ('SHORT', ['CZCE.FG809']), ('LONG', ['CZCE.FG401'])]
    assert json.loads((out / 'source.json').read_text())['combined'][1]['sha256'] == {
        name: V._sha256(other / 'teacher' / f'{name}.csv') for name in V.TEACHER}
    with pytest.raises(FileExistsError):
        V.combine([corpus, other], out)
    with pytest.raises(ValueError, match='more than one material'):
        V.combine([corpus, corpus], tmp_path / 'twice')
    (other / 'params.yaml').write_text(yaml.safe_dump(dict(params, direction='LONG', target_r=3.0), sort_keys=False))
    with pytest.raises(ValueError, match='rules other than the direction'):
        V.combine([corpus, other], tmp_path / 'rules')
    (other / 'params.yaml').write_text(yaml.safe_dump(dict(params, direction='LONG'), sort_keys=False))
    # The same table with its columns in another order (pandas orders them by the first row a run sees).
    timeline = other / 'teacher' / 'positive_timeline.csv'
    frame = pd.read_csv(timeline, dtype=str, keep_default_na=False)
    frame[list(frame.columns[1:]) + [frame.columns[0]]].to_csv(timeline, index=False)
    V.load_positive(other)
    with pytest.raises(ValueError, match='different columns'):
        V.combine([corpus, other], tmp_path / 'columns')
    assert not (tmp_path / 'columns').exists()


def test_v3_a_touch_starts_only_on_its_own_spike_near_h():
    # The line spiked at its own high (cursor 37): V2 starts the first
    # rebound near H at 94; V3 waits there for a spike that never comes.
    a, buckets = synthetic(spike=(96, 99, 96, 98, 100.), peak=LOUD_PEAK)
    assert run(a, buckets)[0][0]['activation'] == 94
    assert run(a, buckets, CFG3) == ([], [])
    # With a spike at cursor 96, V3 starts there; 94 and 95 wait for volume.
    a, buckets = synthetic(peak=LOUD_PEAK)
    (x,), candidates = run(a, buckets, CFG3)
    assert (x['activation'], x['activation_rv'], x['touch'], x['line_spike']) == (96, 5., 1, 37)
    assert [(w['cursor'], w['teach']) for w in x['wait']][-2:] == [(94, 'WAIT_NEAR_NO_VOLUME'), (95, 'WAIT_NEAR_NO_VOLUME')]
    assert [c['cursor'] for c in candidates] == [96, 97, 98, 99]
    assert all(c['obs_rv'] < 2 for c in candidates[1:])
    # A quiet peak: the line's first spike is the touch's own, and V2 = V3.
    a, buckets = synthetic()
    assert run(a, buckets, CFG3) == run(a, buckets)


def test_v3_a_return_to_the_line_needs_a_new_spike():
    # V2 touches the line again on the quiet return (cursor 110); V3 does not.
    a, buckets = synthetic(tail=RETURN)
    assert [x['activation'] for x in run(a, buckets)[0]] == [96, 110]
    events, candidates = assert_causal(a, buckets, range(100, a['n'] + 1, 3), CFG3)
    assert [x['activation'] for x in events] == [96] and max(c['cursor'] for c in candidates) == 99
    # With its own spike at cursor 111 the return is touch 2; 110 waits.
    a, buckets = synthetic(tail=LOUD_RETURN)
    events, candidates = assert_causal(a, buckets, range(100, a['n'] + 1, 3), CFG3)
    assert [(x['touch'], x['activation'], x['end'], x['end_reason']) for x in events] == [
        (1, 96, 105, 'DEPARTED'), (2, 111, 113, 'DEPARTED')]
    assert events[1]['activation_rv'] >= 2 and events[1]['wait'][-1] == dict(cursor=110, teach='WAIT_NEAR_NO_VOLUME')
    assert [c['cursor'] for c in candidates if c['event_id'] == events[1]['event_id']] == [111, 112]
    # The timeline notes say which spike a V3 touch waits for.
    o = dict(obs_H=100., obs_high=97., obs_rv=1., obs_close=96., obs_quote=95., obs_touch=2, obs_line_spike_rv=5.)
    assert V.wait_note('WAIT_NEAR_NO_VOLUME', o, CFG3).endswith('RV 1.0 < 2: no volume spike on this touch yet; wait')
    assert V.wait_note('WAIT_NEAR_NO_VOLUME', o, CFG).endswith('and the line has had no volume spike; wait')
    assert "RV 5.0 >= 2: this touch's volume spike: touch 2 starts" in V.wait_note('ACTIVATION', dict(o, obs_rv=5.), CFG3)


@pytest.mark.skipif(not all(p.exists() for p in CONFIGS_V3), reason='Thinking Student configs not present')
def test_v3_configs_are_the_v2_rules_with_the_touch_spike():
    v2 = V.load_config(CONFIG)
    for path, side in zip(CONFIGS_V3, ('SHORT', 'LONG')):
        v3 = V.load_config(path)
        assert v3 == dict(v2, version='V2_POSITIVE_ENTRY_V3', direction=side,
                          volume=dict(v2['volume'], activation='touch'))
