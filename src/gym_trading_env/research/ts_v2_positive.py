"""Thinking Student V2 positive entry corpus (Spec V0.3, user 2026-10-31;
rules V2 after the user's review of the same day).

Teacher demonstrations of one entry only: the direction is given (the
config's `direction`, SHORT or LONG, the contracts the env trades that way);
for SHORT a known prior 15m high H is a line; the line's first 1m volume
ratio RV >= 2 near H (from its own 15m bucket on) marks it; every return to
the line after the price departed is a touch (an event); ADD at any in-zone
minute of a touch, SL = H + 5 (R <= 12), TP = fill - 2R. A line holds until
the price breaks it. The rules are docs/thinking_student_v2/DESIGN.md with
the values of dreamerv3/configs/thinking_student/v2_positive_entry.yaml.

Rules V3 (V2.2, user 2026-11-04) restore the Spec V0.3 high-position volume
spike: each touch starts only on its own first bar near H with RV >= 2
(bars near H before it wait, WAIT_NEAR_NO_VOLUME); later minutes of the
touch need no new spike. V2 lets the line's spike carry every touch, so
volume barely filtered. Every other rule is V2's.

LONG (V2.1, user 2026-11-04) is the mirror: the line is a prior 15m low L,
the zone [L, L + 7], SL = L - 5, TP = fill + 2R, a departure a close at
L + 10. It runs the same scan in short space: every price of a LONG market
negated (its lows become highs, its buy quotes sell quotes, rounding
included: floor(-x) = -ceil(x)), and the prices written out are real again.

Lines, RV, touches and candidates use only the bars up to their cursor and
the complete 15m buckets available by then. The future is used offline
only: each candidate's Native trade (the real env with SL = R points and TP
= 2R points) selects the teacher positives (EXCELLENT / GOOD). Every other
candidate stays in audit/ with its reason and never becomes material.
Candidates of one touch, and of the touches of one line, are alternatives,
each traded from a flat account; nothing adds their results.

  python -m gym_trading_env.research.ts_v2_positive \\
      --pool ../dreamerv3/configs/thinking_student/markets/pool.json \\
      --config ../dreamerv3/configs/thinking_student/v2_positive_entry.yaml \\
      --out /data/logdir/fg_single3/thinking_student/V2_POSITIVE_ENTRY_V2_20261031 \\
      --workers 12 [--contracts CZCE.FG809 ...]
  python -m gym_trading_env.research.ts_v2_positive --combine SHORT_DIR LONG_DIR --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gym_trading_env.envs.entry_contract import read_yaml, resolve_path
from .fg_single3_oracle import CandidateIndex, reference_candidate
from .ts_v1_corpus import _pool, _repos, native_env, train_row, write_inputs
from .ts_v1_grades import _room

# Material versions and the volume spike that starts a touch: V2, the
# line's first spike from its own 15m bucket on (touches after it need
# none); V3 (user 2026-11-04), each touch's own bar near the line with RV >=
# ratio, as Spec V0.3 and rules V1 (config volume.activation: touch). V4
# (user 2026-11-05): V3 with zone 5 (planned R <= 10), the WAIT_IN_RULE rows
# (timeline.wait_in_rule) and the costs of a cost preset; the V2.3 env's rule
# gate opens ADD at its rule positions (rule_cursors). V5 (user 2026-11-05,
# "止损固定5点", "全部：SL = 成交价+5"): V4 with entry.fixed_stop_points, the
# SL at the fill + 5 (R 5, TP 2R = 10) for every ADD, a gap fill included;
# the line H still places the rule positions and ends a touch it breaks.
ACTIVATION = {'V2_POSITIVE_ENTRY_V2': 'line', 'V2_POSITIVE_ENTRY_V3': 'touch', 'V2_POSITIVE_ENTRY_V4': 'touch',
              'V2_POSITIVE_ENTRY_V5': 'touch'}
SIDES = ('SHORT', 'LONG')
LOTS = 3
# Order in which a candidate's first failing condition names its reason.
REASONS = ('FILL_OUT_OF_ZONE', 'MASKED', 'NOT_FILLED', 'STOP_LOSS', 'OTHER_EXIT',
           'RIGHT_CENSORED', 'TP_DEEP_MAE', 'TP_SLOW', 'POSITIVE')
TEACH = ('WAIT_FAR', 'WAIT_APPROACH', 'WAIT_NEAR_NO_VOLUME', 'ACTIVATION', 'ACTIVATION_ADD', 'ADD', 'WAIT_IN_RULE')
# Teacher table columns other than obs_* (decision-time) and label_* (offline).
META = ('event_id', 'entry_id', 'contract', 'cursor', 'time', 'direction_source', 'teach')
TEACHER = ('positive_events', 'positive_entries', 'positive_timeline')
# Trade facts of a candidate (label_* / label_tp15_* columns).
LABELS = dict(exit_reason='exit_reason', exit_cursor='exit_cursor', exit_time='exit_time',
              exit_price='exit_price', net_pnl='net_pnl', held='held_minutes', mae='mae_points',
              mfe='mfe_points')


# ---- inputs ----

def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get('version') not in ACTIVATION or cfg.get('direction') not in SIDES:
        raise ValueError(f'{path}: not a {" / ".join(ACTIVATION)} SHORT or LONG config')
    if cfg['volume'].get('activation', 'line') != ACTIVATION[cfg['version']]:
        raise ValueError(f'{path}: {cfg["version"]} needs volume.activation {ACTIVATION[cfg["version"]]}')
    e = cfg['entry']
    if not 0 < e['stop_above_points'] <= e['stop_above_points'] + e['zone_points'] <= e['max_risk_points']:
        raise ValueError(f'{path}: the zone and SL allow a risk above max_risk_points')
    fixed = e.get('fixed_stop_points')
    if fixed is not None and not (isinstance(fixed, int) and 0 < fixed <= e['max_risk_points']):
        raise ValueError(f'{path}: entry.fixed_stop_points is a positive integer <= max_risk_points')
    if not isinstance(cfg['timeline'].get('wait_in_rule', False), bool):
        raise ValueError(f'{path}: timeline.wait_in_rule is true or false')
    if 'costs' in cfg:  # a cost preset path relative to this file, read (ts_v1_corpus.load_costs)
        from .ts_v1_corpus import load_costs
        cfg['costs'] = load_costs(cfg['costs'], path)
    return cfg


def contract_direction(row):
    """The single direction of the contract in its env's direction schedule."""
    entry = read_yaml(row['env_config'])['entry']
    schedule = (read_yaml(resolve_path(row['env_config'], entry['strategy_config']))['directions']
                if entry.get('strategy_config') else
                read_yaml(resolve_path(row['env_config'], entry['direction_config'])))
    sides = {r['direction'] for r in schedule['intervals'] if r['contract'] == row['contract']}
    if len(sides) != 1:
        raise ValueError(f'{row["contract"]}: directions {sorted(sides)}')
    return sides.pop()


def side_rows(pool, side, contracts=None):
    """TRAIN pool rows of the contracts traded `side`: the given ones (any
    other is refused) or all of the pool."""
    names = contracts or [r['contract'] for r in json.loads(Path(pool).read_text())['files']
                          if r.get('role') == 'train' and r.get('eligible_for_training')]
    rows = [train_row(c, pool) for c in names]
    if contracts:
        wrong = [r['contract'] for r in rows if contract_direction(r) != side]
        if wrong:
            raise ValueError(f'not {side} contracts: {wrong}')
        return rows
    return [r for r in rows if contract_direction(r) == side]


def sign(side):
    """+1 for SHORT, -1 for LONG: a price times the sign is its short-space price."""
    if side not in SIDES:
        raise ValueError(f'direction {side} is not SHORT or LONG')
    return 1 if side == 'SHORT' else -1


def side_buckets(buckets, s):
    """Complete 15m buckets in short space: a LONG market's negated lows are
    its highs (the only bar price pivot_levels reads)."""
    return buckets if s > 0 else [dict(b, High=-float(b['Low'])) for b in buckets]


def fixed_stop(cfg):
    """entry.fixed_stop_points (V5): R in points from the fill, or None (SL = H + stop_above)."""
    return cfg['entry'].get('fixed_stop_points')


def tp_points(cfg, risk, multiple):
    """TP distance at `multiple` R, rounded away from the fill."""
    return int(math.ceil(multiple * risk - 1e-9))


# ---- causal market facts ----

def market_arrays(env, cfg):
    """Per-cursor bars, the executable quotes and RV of one market in short
    space (module docstring): a SHORT market as it is, a LONG one negated,
    its lows in `high` and its buy quotes in `sell_*`."""
    raw, side = env._raw, cfg['direction']
    s = sign(side)
    times, bobs = raw.index, pd.DatetimeIndex(env.bobs)
    top, bottom = ('High', 'Low') if s > 0 else ('Low', 'High')

    def quotes(column, at, opening):
        q = np.array([float(env._price(v, side, t, opening=opening)) for v, t in zip(raw[column], at)])
        return q if s > 0 else -q

    def price(column):
        v = raw[column].to_numpy(float)
        return v if s > 0 else -v

    starts = {a for a, _ in env.market_clock['segments']}
    valid = np.asarray(env.valid_fill_clock, bool)
    segment_open = np.asarray(bobs.strftime('%H:%M').isin(starts), bool)
    v = cfg['volume']
    eligible = valid & ~segment_open if v['exclude_segment_open'] else valid
    volume = raw.Volume.to_numpy(float)
    return dict(
        n=len(raw), times=times, bobs=bobs, times_ns=times.asi8, bob_ns=bobs.asi8,
        open=price('Open'), high=price(top), low=price(bottom), close=price('Close'), volume=volume,
        valid=valid, segment_open=segment_open,
        sell_close=quotes('Close', times, True),   # an ADD decided at the close
        sell_open=quotes('Open', bobs, True),      # the fill of an ADD decided a bar earlier
        buy_high=quotes(top, times, False),        # the quote the position's SL checks
        rv=volume_ratio(volume, eligible, v['baseline_minutes']),
        prefix=env.market_clock_counter.prefix, multiplier=float(env.terms.multiplier),
        side=side, sign=s)


def volume_ratio(volume, eligible, baseline):
    """RV of each eligible bar against the mean volume of the previous
    `baseline` eligible bars; NaN elsewhere and without a full baseline."""
    rows = np.flatnonzero(eligible)
    rv = np.full(len(volume), np.nan)
    if len(rows) > baseline:
        v = volume[rows]
        c = np.concatenate(([0.], np.cumsum(v)))
        mean = (c[baseline:-1] - c[:-baseline - 1]) / baseline
        with np.errstate(divide='ignore', invalid='ignore'):
            rv[rows[baseline:]] = np.where(mean > 0, v[baseline:] / mean, np.nan)
    return rv


def pivot_levels(buckets, a, left, right, look, stop):
    """Pivot highs H of the complete 15m `buckets` (the lines), each with the
    cursor of its first later bar (start), the cursors at which it becomes
    visible (confirm) and expires, and the first at which it is broken (a
    short's SL quote `a['buy_high']` reaches H + stop). Shared by the
    material and the Native env's structural stop (V0.5)."""
    high = np.array([b['High'] for b in buckets], float)
    end_ns = np.array([pd.Timestamp(b['bucket_end']).value for b in buckets], np.int64)
    avail_ns = np.array([pd.Timestamp(b['available_at']).value for b in buckets], np.int64)
    n = a['n']

    def seen(j):  # first cursor whose bar end is at or after bucket j's arrival
        return int(np.searchsorted(a['times_ns'], avail_ns[j], side='left')) if j < len(high) else n

    levels = []
    for j in range(left, len(high) - right):
        if not (high[j] > high[j - left:j].max() and high[j] >= high[j + 1:j + right + 1].max()):
            continue
        H = float(high[j])
        start = int(np.searchsorted(a['bob_ns'], end_ns[j], side='left'))
        expire = seen(j + look + 1)
        levels.append(dict(
            bucket=j, H=H, bucket_end=str(pd.Timestamp(end_ns[j], tz='UTC').tz_convert(a['times'].tz)),
            start=start, confirm=seen(j + right), expire=expire,
            broken=_first(a['valid'] & (a['buy_high'] >= H + stop), start, expire, n)))
    return levels


def _first(mask, lo, hi, n):
    hit = np.flatnonzero(mask[lo:hi])
    return lo + int(hit[0]) if len(hit) else n


def departed_at(a, level, points):
    """First cursor, from the line's start, of a 1m close at least `points`
    below its H (the price has left the line; a touch may follow). Shared by
    the material and the Native env's structural stop."""
    return _first(a['valid'] & (a['close'] <= level['H'] - points), level['start'], level['expire'], a['n'])


def prior_highs(env, a, cfg):
    """The lines (`pivot_levels`), each also with the cursors at which it is
    first departed (a close at H - departure) and first spiked (RV >= ratio
    on a bar near H, from the bars of its own bucket on)."""
    p, stop = cfg['prior_high'], cfg['entry']['stop_above_points']
    near, ratio = cfg['event']['near_points'], cfg['volume']['ratio']
    buckets = side_buckets(env._e_market.complete_buckets, a['sign'])
    end_ns = np.array([pd.Timestamp(b['bucket_end']).value for b in buckets], np.int64)
    spiky = a['valid'] & (np.nan_to_num(a['rv']) >= ratio)
    levels = pivot_levels(buckets, a, p['pivot_left'], p['pivot_right'], p['lookback_buckets'], stop)
    for x in levels:
        H, expire, j = x['H'], x['expire'], x['bucket']
        own = int(np.searchsorted(a['bob_ns'], end_ns[j - 1], side='left'))  # first bar of bucket j
        x.update(departed=departed_at(a, x, p['departure_points']),
                 spike=_first(spiky & (a['high'] >= H - near), own, min(x['broken'], expire), a['n']))
    return levels, len(buckets)


def deadline(env):
    """First cursor at which a flat account can no longer ADD for good."""
    times = env._raw.index
    ends = [env.terms.last_open] + ([env.main_exit_at] if env.main_exit_at is not None else [])
    return int(min(np.searchsorted(times, t, side='left') for t in ends))


def rule_pass(a, levels, cfg, end, contract):
    """The rule's causal pass over the cursors of one market (DESIGN
    sections 2-5): its touches (events, WAIT rows not attached) and, per
    cursor, `event_of` (index into the events, -1 outside a touch),
    `in_zone` (a rule position: inside a touch, a valid minute, the decision
    quote in [H - zone, H]) and `active` (the bucket of the active line
    outside touches, read by the WAIT rows); `lines` holds each event's
    line. No touch continues or starts at `end` or later (BOUNDARY): the
    material passes min(flat_end, n - 1), so every rule position has its
    fill bar; the env's rule gate passes n, so the newest minute of a live
    market can be one."""
    e, ev, p = cfg['entry'], cfg['event'], cfg['prior_high']
    stop, zone, near = e['stop_above_points'], e['zone_points'], ev['near_points']
    far, n = p['departure_points'], a['n']
    ratio, own_spike = cfg['volume']['ratio'], ACTIVATION[cfg['version']] == 'touch'
    pending = sorted(levels, key=lambda x: x['confirm'])
    live, k = [], 0
    # A line may start a touch from its first departure on, and again from
    # the bar after each of its touches that ended by a departure.
    armed = {x['bucket']: x['departed'] for x in levels}
    touches = dict.fromkeys(armed, 0)
    active = np.full(n, -1, np.int64)  # bucket of the active line of a cursor outside touches
    event_of = np.full(n, -1, np.int64)
    in_zone = np.zeros(n, bool)
    events, lines, event = [], [], None
    for s in range(n):
        while k < len(pending) and pending[k]['confirm'] <= s:
            live.append(pending[k])
            k += 1
        live = [x for x in live if x['expire'] > s and x['broken'] > s]
        if event is not None:
            t, H = event['activation'], event['H']
            reason = None
            if s > t and a['valid'][s] and a['buy_high'][s] >= H + stop:
                reason = 'INVALIDATED'
            elif s > t and a['valid'][s] and a['close'][s] <= H - far:
                reason = 'DEPARTED'
            elif s >= event['level_expire']:
                reason = 'EXPIRED'
            elif s >= end:
                reason = 'BOUNDARY'
            if reason is None:
                event['cursors'] += 1
                event_of[s] = len(events)
                in_zone[s] = a['valid'][s] and H - zone <= a['sell_close'][s] <= H
                continue
            event.update(end=s, end_time=str(a['times'][s]), end_reason=reason)
            if reason == 'DEPARTED':
                armed[event['level_bucket']] = s + 1
            events.append(event)
            event = None
        ready = [x for x in live if armed[x['bucket']] <= s]
        if not ready or s >= end:
            continue
        # The first line above the price: the lowest armed one.
        level = min(ready, key=lambda x: (x['H'], -x['bucket']))
        active[s] = level['bucket']
        H = level['H']
        # Near H, not through the stop, and with a volume spike: V3, this
        # bar's own RV >= ratio; V2, the line's spike (this bar may be it).
        spiked = a['rv'][s] >= ratio if own_spike else level['spike'] <= s
        if a['valid'][s] and a['high'][s] >= H - near and a['buy_high'][s] < H + stop and spiked:
            armed[level['bucket']] = n
            touches[level['bucket']] += 1
            spike = level['spike']
            event = dict(
                event_id=f'{contract}@{s}', contract=contract, direction_source=cfg['direction_source'],
                H=H, level_bucket=level['bucket'], level_bucket_end=level['bucket_end'],
                level_confirm=level['confirm'], level_departed=level['departed'], level_expire=level['expire'],
                touch=touches[level['bucket']], line_spike=spike, line_spike_time=str(a['times'][spike]),
                line_spike_rv=float(a['rv'][spike]),
                activation=s, activation_time=str(a['times'][s]), activation_rv=float(a['rv'][s]),
                cursors=0, wait=None)
            lines.append(level)
            active[s] = -1
            # The activation minute is itself the first decision of the event.
            event['cursors'] += 1
            event_of[s] = len(events)
            in_zone[s] = H - zone <= a['sell_close'][s] <= H
    if event is not None:
        event.update(end=n - 1, end_time=str(a['times'][n - 1]), end_reason='BOUNDARY')
        events.append(event)
    return dict(events=events, lines=lines, event_of=event_of, in_zone=in_zone, active=active)


def scan(a, levels, cfg, flat_end, contract):
    """Touches (events), candidates and the pre-touch WAIT rows of one
    market (DESIGN sections 2-5, 8): rule_pass with its boundary at
    flat_end and before the last minute (a candidate needs its fill bar);
    the candidates are its rule positions, and each touch's WAIT rows read
    the active lines before it."""
    r = rule_pass(a, levels, cfg, min(flat_end, a['n'] - 1), contract)
    events = r['events']
    for x, level in zip(events, r['lines']):
        x['wait'] = _wait_rows(a, level, x['activation'], r['active'], cfg)
    candidates = [_candidate(a, events[r['event_of'][s]], int(s), cfg, contract)
                  for s in np.flatnonzero(r['in_zone'])]
    return events, candidates


def rule_cursors(env, cfg):
    """The rule positions of one env's market in the direction of `cfg`
    (V2.3 rule gate, user 2026-11-05): the material's own rule_pass over
    the whole market, with no boundary before its last minute. Per cursor
    `open` (a rule position) and `event` (its touch, -1 outside); per touch
    its line `H` (short space: a LONG's prior low negated), `touch` (the
    line's touch count) and `event_id`; `sign` of the side."""
    a = market_arrays(env, cfg)
    levels, _ = prior_highs(env, a, cfg)
    r = rule_pass(a, levels, cfg, a['n'], env.contract)
    return dict(open=r['in_zone'], event=r['event_of'], sign=a['sign'],
                H=np.array([x['H'] for x in r['events']], float),
                touch=np.array([x['touch'] for x in r['events']], np.int64),
                event_id=[x['event_id'] for x in r['events']])


def _candidate(a, event, s, cfg, contract):
    e = cfg['entry']
    H, t, q = event['H'], event['activation'], float(a['sell_close'][s])
    fill, fixed = float(a['sell_open'][s + 1]), fixed_stop(cfg)
    planned = H + e['stop_above_points'] - q if fixed is None else float(fixed)
    risk = H + e['stop_above_points'] - fill if fixed is None else float(fixed)
    return dict(
        event_id=event['event_id'], contract=contract, cursor=s, time=str(a['times'][s]),
        direction_source=event['direction_source'],
        obs_H=H, obs_close=float(a['close'][s]), obs_quote=q, obs_quote_below_h=H - q,
        obs_rv=float(a['rv'][s]), obs_rv_activation=event['activation_rv'],
        obs_minutes_since_activation=int(a['prefix'][s + 1] - a['prefix'][t + 1]),
        obs_touch=event['touch'], obs_line_spike_rv=event['line_spike_rv'],
        obs_sl=q + planned if fixed is not None else H + e['stop_above_points'], obs_planned_risk=planned,
        obs_planned_tp=q - tp_points(cfg, planned, e['target_r']),
        label_fill_cursor=s + 1, label_fill_time=str(a['bobs'][s + 1]), label_fill_expected=fill,
        label_R=risk, label_fill_in_zone=bool(H - e['zone_points'] <= fill <= H))


def _wait_rows(a, level, t, active, cfg):
    """The WAIT rows before touch start t: from the last close far below H
    (and no earlier than the run of cursors where this line is the active
    one), each phase cut to its first, last and a few evenly spaced rows."""
    H, far = level['H'], cfg['prior_high']['departure_points']
    near, keep = cfg['event']['near_points'], cfg['timeline']['wait_rows_per_phase']
    run = t
    while run > 0 and active[run - 1] == level['bucket']:
        run -= 1
    span = np.arange(run, t)
    span = span[a['valid'][span]]
    far_rows = span[a['close'][span] <= H - far]
    if len(far_rows):
        span = span[span >= far_rows[-1]]
    phases = np.where(a['close'][span] <= H - far, 0, np.where(a['high'][span] >= H - near, 2, 1))
    chosen = set()
    for phase in range(3):
        rows = span[phases == phase]
        if len(rows):
            chosen |= {int(rows[0]), int(rows[-1])}
            chosen |= {int(rows[i]) for i in np.linspace(0, len(rows) - 1, min(keep, len(rows))).round().astype(int)}
    return [dict(cursor=s, teach=TEACH[int(phases[np.searchsorted(span, s)])]) for s in sorted(chosen)]


# ---- Native trades (offline) ----

def _trade_job(job):
    """Native trades "ADD at cursor s, then hold" from a flat account at one
    SL / TP (points): the indexed trade and, for `verify` cursors, its
    minute-by-minute Native trace."""
    config, cursors, verify, side = job
    env = native_env(config, market=False)
    index = CandidateIndex(env)
    volume = env._raw.Volume.to_numpy()
    out = {}
    for s in cursors:
        row = index.candidate(s)
        rec = dict(eligible=bool(row['eligible']), filled=bool(row.get('filled')),
                   reject=row.get('reject_reason'))
        if rec['filled']:
            i, fill, exit = int(row['entry_cursor']), float(row['entry_price']), row['exit']
            rec.update(entry_cursor=i, fill=fill, right_censored=bool(row['right_censored']),
                       exit_reason=exit['reason'] if exit else 'RIGHT_CENSORED',
                       exit_cursor=int(row['exit_cursor']) if exit else -1,
                       exit_time=exit['exit_time'] if exit else '',
                       exit_price=float(exit['exit_price']) if exit else math.nan,
                       net_pnl=float(exit['net_trade_pnl']) if exit else math.nan,
                       held=int(row['held_market_bars']), mae=float(row['MAE_points']),
                       mfe=float(_room(index, side, fill, i, int(row['observed_until_cursor']), volume)),
                       ambiguous=bool(exit and exit.get('ambiguous_tp_sl')))
        out[s] = rec
    checks = {}
    for s in verify:
        native = reference_candidate(env, s)
        rec, trade = out[s], native.get('exit')
        checks[s] = (native['filled'] == rec['filled'] and trade is not None
                     and trade['reason'] == rec['exit_reason'] and native['exit_cursor'] == rec['exit_cursor']
                     and float(trade['entry_price']) == rec['fill']
                     and float(trade['net_trade_pnl']) == rec['net_pnl'])
    return config, out, checks


def classify(c, trade, cfg):
    """(reason, grade, quality) of a candidate from its target trade."""
    q = cfg['quality']
    if not c['label_fill_in_zone'] and fixed_stop(cfg) is None:  # V5 trades a gap fill too
        return 'FILL_OUT_OF_ZONE', '', math.nan
    if not trade['eligible']:
        return 'MASKED', '', math.nan
    if not trade['filled']:
        return 'NOT_FILLED', '', math.nan
    if trade['fill'] != c['label_fill_expected']:
        raise ValueError(f'{c["event_id"]} cursor {c["cursor"]}: Native fill {trade["fill"]} '
                         f'differs from the expected {c["label_fill_expected"]}')
    reason = trade['exit_reason']
    if reason != 'TAKE_PROFIT':
        return {'STOP_LOSS': 'STOP_LOSS', 'RIGHT_CENSORED': 'RIGHT_CENSORED'}.get(reason, 'OTHER_EXIT'), '', math.nan
    risk, held, mae = c['label_R'], trade['held'], trade['mae']
    if trade['ambiguous']:
        raise ValueError('A TP exit on an ambiguous TP / SL bar contradicts the risk-first rule')
    if mae > q['good']['mae_r'] * risk:
        return 'TP_DEEP_MAE', '', math.nan
    if held > q['good']['held_minutes']:
        return 'TP_SLOW', '', math.nan
    grade = ('EXCELLENT' if mae <= q['excellent']['mae_r'] * risk and held <= q['excellent']['held_minutes']
             else 'GOOD')
    day = q['good']['held_minutes']
    return 'POSITIVE', grade, 0.5 * (1 - mae / risk) + 0.5 * (1 - min(held, day) / day)


# The prices of a record; a LONG market's are negated back from short space.
PRICES = ('H', 'obs_H', 'obs_close', 'obs_high', 'obs_quote', 'obs_sl', 'obs_planned_tp', 'label_fill_expected')


def _real(record, s):
    """`record` with real prices (short space for SHORT is real already)."""
    if s < 0:
        record.update({k: -record[k] for k in PRICES if k in record})
    return record


def _words(cfg):
    """The line's name, its zone and its stop in the texts."""
    e, fixed = cfg['entry'], fixed_stop(cfg)
    if sign(cfg['direction']) > 0:
        return 'H', f'[H-{e["zone_points"]}, H]', f'H+{e["stop_above_points"]}' if fixed is None else f'fill+{fixed}'
    return 'L', f'[L, L+{e["zone_points"]}]', f'L-{e["stop_above_points"]}' if fixed is None else f'fill-{fixed}'


def _at(price, H, name='H'):
    return f'{price:g} ({name}{price - H:+g})'


def _rv(rv):
    return f'RV {rv:.1f}' if np.isfinite(rv) else 'RV n/a'


def explain(c, cfg):
    """Why a candidate is placed where it is, from its obs_* facts only."""
    e, H = cfg['entry'], c['obs_H']
    N, zone, stop = _words(cfg)
    return (f'quote {_at(c["obs_quote"], H, N)} in {zone}; touch {c["obs_touch"]} of the line '
            f'(line volume spike {_rv(c["obs_line_spike_rv"])}); {_rv(c["obs_rv"])}, touch began at '
            f'{_rv(c["obs_rv_activation"])} {c["obs_minutes_since_activation"]} min ago; '
            f'SL {stop} = {c["obs_sl"]:g}, planned R {c["obs_planned_risk"]:g}, '
            f'planned TP {c["obs_planned_tp"]:g} ({e["target_r"]:g}R)')


def wait_note(teach, o, cfg, spike_now=False):
    """Why a timeline row before an ADD waits or starts a touch (obs_*
    facts); `spike_now`: the touch's first bar is the line's volume spike."""
    H, far = o['obs_H'], cfg['prior_high']['departure_points']
    near, ratio = cfg['event']['near_points'], cfg['volume']['ratio']
    N = _words(cfg)[0]
    # obs_high is the bar's extreme toward the line: its high (SHORT), its low (LONG).
    if sign(cfg['direction']) > 0:
        far_from, not_near, move, end = f'at or below H-{far}', f'below H-{near}', 'rebound, high', 'high'
    else:
        far_from, not_near, move, end = f'at or above L+{far}', f'above L+{near}', 'pullback, low', 'low'
    if teach == 'WAIT_FAR':
        return f'close {_at(o["obs_close"], H, N)} {far_from}: far from {N}; wait'
    if teach == 'WAIT_IN_RULE':
        return (f'rule position: quote {_at(o["obs_quote"], H, N)} in {_words(cfg)[1]}, touch {o["obs_touch"]}, '
                f'{_rv(o["obs_rv"])}, {o["obs_minutes_since_activation"]} min after the touch began; '
                f'the teacher waits for a later entry of this touch')
    if teach == 'WAIT_APPROACH':
        return f'{move} {_at(o["obs_high"], H, N)} {not_near}: not near {N} yet; wait'
    own = ACTIVATION[cfg['version']] == 'touch'
    if teach == 'WAIT_NEAR_NO_VOLUME':
        return (f'{end} {_at(o["obs_high"], H, N)} near {N} but {_rv(o["obs_rv"])} < {ratio:g}'
                + (': no volume spike on this touch yet; wait' if own else ' and the line has had no volume spike; wait'))
    why = (f'{_rv(o["obs_rv"])} >= {ratio:g}: this touch\'s volume spike' if own else
           f'{_rv(o["obs_rv"])} >= {ratio:g}: the line\'s volume spike' if spike_now else
           f'the line had its volume spike ({_rv(o["obs_line_spike_rv"])}) before')
    return (f'{end} {_at(o["obs_high"], H, N)} near {N}, {why}: touch {o["obs_touch"]} starts; '
            f'quote {_at(o["obs_quote"], H, N)}')


def grade_reason(c, cfg):
    """The offline reason of a candidate's reason / grade."""
    q, H, reason = cfg['quality'], c['obs_H'], c['reason']
    N, zone, _ = _words(cfg)
    fill = f'fill {_at(c["label_fill_expected"], H, N)}'
    if reason == 'FILL_OUT_OF_ZONE':
        return f'{fill} (next open {"-" if sign(cfg["direction"]) > 0 else "+"} spread) outside {zone}'
    if reason == 'MASKED':
        return 'ADD masked for a flat account'
    if reason == 'NOT_FILLED':
        return 'the ADD did not fill'
    if reason == 'RIGHT_CENSORED':
        return f'{fill}; the data ends before an exit'
    held, mae, mae_r = int(c['label_held_minutes']), c['label_mae_points'], c['label_mae_R']
    trade = f'{fill}, R {c["label_R"]:g}; Native {c["label_exit_reason"]} {c["label_exit_price"]:g} after {held} min'
    if reason in ('STOP_LOSS', 'OTHER_EXIT'):
        return trade
    trade += f', MAE {mae:g} = {mae_r:.2f}R'
    if reason == 'TP_DEEP_MAE':
        return f'{trade} > {q["good"]["mae_r"]:g}R'
    if reason == 'TP_SLOW':
        return f'{trade}; held > {q["good"]["held_minutes"]} min'
    x, g = q['excellent'], q['good']
    if c['label_grade'] == 'EXCELLENT':
        return f'{trade}; EXCELLENT: MAE <= {x["mae_r"]:g}R and held <= {x["held_minutes"]} min'
    miss = [f'held > {x["held_minutes"]} min'] * (held > x['held_minutes']) + [
        f'MAE > {x["mae_r"]:g}R'] * (mae > x['mae_r'] * c['label_R'])
    return (f'{trade}; GOOD: MAE <= {g["mae_r"]:g}R and held <= {g["held_minutes"]} min, '
            f'not EXCELLENT ({", ".join(miss)})')


# ---- production ----

def _scan_job(job):
    row, cfg, folder = job
    config = write_inputs(row, Path(folder) / 'inputs' / row['contract'] / 'base', costs=cfg.get('costs'))
    env = native_env(config, market=True)
    if (env.take_profit_points, env.stop_loss_points) != (5, 30) or env.direction_at(
            env._raw.index[0], 0) not in (cfg['direction'], 'NO_TRADE'):
        raise ValueError(f'{row["contract"]}: unexpected base env')
    a = market_arrays(env, cfg)
    levels, buckets = prior_highs(env, a, cfg)
    flat_end = min(CandidateIndex(env).flat_end, deadline(env))
    events, candidates = scan(a, levels, cfg, flat_end, row['contract'])
    def seen(x, s):  # decision-time facts of cursor s against the line of touch x (real prices)
        H, q = x['H'], float(a['sell_close'][s])
        # The touch number is the count of earlier touches plus one; the
        # line's spike is known once it happened.
        return _real(dict(obs_H=H, obs_close=float(a['close'][s]), obs_high=float(a['high'][s]),
                          obs_quote=q, obs_quote_below_h=H - q, obs_rv=float(a['rv'][s]),
                          obs_in_zone=bool(0 <= H - q <= cfg['entry']['zone_points']), obs_touch=x['touch'],
                          obs_line_spike_rv=x['line_spike_rv'] if x['line_spike'] <= s else math.nan), a['sign'])

    for x in events:
        x['wait'] = [dict(w, time=str(a['times'][w['cursor']]), **seen(x, w['cursor']),
                          obs_minutes_since_activation=-1) for w in x['wait']]
        for w in x['wait']:
            w['obs_explain'] = wait_note(w['teach'], w, cfg)
        x['activation_obs'] = seen(x, x['activation'])
        x['activation_obs']['obs_explain'] = wait_note('ACTIVATION', x['activation_obs'], cfg,
                                                       spike_now=x['line_spike'] == x['activation'])
    for record in (*events, *candidates):
        _real(record, a['sign'])
    raw = env._raw
    market = pd.DataFrame(dict(time=a['times'].astype(str), bob=a['bobs'].astype(str),
                               open=raw.Open.to_numpy(float), high=raw.High.to_numpy(float),
                               low=raw.Low.to_numpy(float), close=raw.Close.to_numpy(float), volume=a['volume'],
                               rv=a['rv'], valid=a['valid'], segment_open=a['segment_open']))
    stats = dict(rows=a['n'], buckets=buckets, levels=len(levels),
                 levels_departed=sum(x['departed'] < min(x['broken'], x['expire']) for x in levels),
                 levels_spiked=sum(x['spike'] < min(x['broken'], x['expire']) for x in levels),
                 levels_touched=len({x['level_bucket'] for x in events}),
                 flat_end=flat_end, first_time=str(a['times'][0]), last_time=str(a['times'][-1]),
                 multiplier=a['multiplier'], data_sha256=_sha256(row['path']))
    return row['contract'], events, candidates, market, stats


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _trade_inputs(rows, cfg, folder, candidates):
    """One env inputs folder per contract, risk and target; the jobs of the
    in-zone candidates of each."""
    jobs = []
    e = cfg['entry']
    by_row = {r['contract']: r for r in rows}
    for contract in sorted({c['contract'] for c in candidates}):
        mine = [c for c in candidates if c['contract'] == contract
                and (c['label_fill_in_zone'] or fixed_stop(cfg) is not None)]
        for risk in sorted({int(c['label_R']) for c in mine}):
            cursors = [c['cursor'] for c in mine if int(c['label_R']) == risk]
            for name, multiple in (('target', e['target_r']), ('parallel', e['parallel_r'])):
                tp = tp_points(cfg, risk, multiple)
                rules = dict(take_profit_points=tp, stop_loss_points=risk)
                config = write_inputs(by_row[contract], Path(folder) / 'inputs' / contract / f'R{risk}_TP{tp}', rules,
                                      costs=cfg.get('costs'))
                jobs.append(dict(contract=contract, risk=risk, name=name, tp=tp,
                                 config=str(config), cursors=cursors))
    return jobs


def produce(rows, cfg, out, workers, log=print, command=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'{out} exists; remove it to regenerate')
    tmp = out.with_name(out.name + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    began, seconds = time.monotonic(), {}
    with _pool(max(1, min(workers, len(rows)))) as pool:
        scanned = list(pool.map(_scan_job, [(r, cfg, str(tmp)) for r in rows]))
    seconds['scan'] = round(time.monotonic() - began, 1)
    events = [x for _, ev, _, _, _ in scanned for x in ev]
    candidates = [c for _, _, cs, _, _ in scanned for c in cs]
    markets = {name: m for name, _, _, m, _ in scanned}
    stats = {name: s for name, _, _, _, s in scanned}
    log(f'scan: {len(events)} events, {len(candidates)} candidates ({seconds["scan"]} s)')

    jobs = _trade_inputs(rows, cfg, tmp, candidates)
    side, s = cfg['direction'], sign(cfg['direction'])
    started = time.monotonic()
    with _pool(max(1, min(workers, len(jobs)))) as pool:
        done = list(pool.map(_trade_job, [(j['config'], j['cursors'], [], side) for j in jobs]))
    trades = {(j['contract'], j['name'], s): rec for j, (_, out_, _) in zip(jobs, done)
              for s, rec in out_.items()}
    seconds['trades'] = round(time.monotonic() - started, 1)

    for c in candidates:
        key = (c['contract'], 'target', c['cursor'])
        c['reason'], c['label_grade'], c['label_quality'] = (
            classify(c, trades[key], cfg) if key in trades else ('FILL_OUT_OF_ZONE', '', math.nan))
        for name, prefix in (('target', 'label_'), ('parallel', 'label_tp15_')):
            rec = trades.get((c['contract'], name, c['cursor']), {})
            c.update({prefix + new: rec.get(old, '' if old in ('exit_reason', 'exit_time') else math.nan)
                      for old, new in LABELS.items()})
            c[prefix + 'filled'] = bool(rec.get('filled', False))
        c['label_tp15_points'] = tp_points(cfg, c['label_R'], cfg['entry']['parallel_r']) \
            if c['label_fill_in_zone'] or fixed_stop(cfg) is not None else math.nan
        c['label_tp'] = c['label_fill_expected'] - s * tp_points(cfg, c['label_R'], cfg['entry']['target_r'])
        # V5: the SL of the actual fill (obs_sl is the one planned at the quote).
        c['label_sl'] = c['obs_sl'] if fixed_stop(cfg) is None else c['label_fill_expected'] + s * c['label_R']
        mult = stats[c['contract']]['multiplier']
        c['label_net_R'] = c['label_net_pnl'] / (c['label_R'] * LOTS * mult) if c['label_filled'] else math.nan
        c['label_mae_R'] = c['label_mae_points'] / c['label_R'] if c['label_filled'] else math.nan
        # An exit on the first minute of a market segment (21:00, 09:00, 10:30,
        # 13:30) is set by that minute's open, often a gap past the TP.
        exit_at = c['label_exit_cursor']
        c['label_exit_at_segment_open'] = bool(
            c['label_filled'] and exit_at >= 0 and markets[c['contract']].segment_open.iat[int(exit_at)])
        c['obs_explain'], c['label_grade_reason'] = explain(c, cfg), grade_reason(c, cfg)

    # Every positive is replayed minute by minute through its Native env.
    started = time.monotonic()
    verify = []
    for j in jobs:
        if j['name'] == 'target':
            cursors = [c['cursor'] for c in candidates if c['contract'] == j['contract']
                       and c['reason'] == 'POSITIVE' and int(c['label_R']) == j['risk']]
            if cursors:
                verify.append((j['config'], cursors, cursors, side))
    with _pool(max(1, min(workers, len(verify) or 1))) as pool:
        checks = {}
        for config, _, check in pool.map(_trade_job, verify):
            contract = Path(config).parents[1].name
            checks.update({(contract, s): ok for s, ok in check.items()})
    for c in candidates:
        c['label_native_verified'] = checks.get((c['contract'], c['cursor']), False)
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise ValueError(f'Indexed positives differ from their Native minute trace: {bad}')
    seconds['native_verify'] = round(time.monotonic() - started, 1)

    tables = assemble(events, candidates, cfg)
    write_tables(tmp, tables)
    summary = audit_summary(events, candidates, stats, cfg)
    (tmp / 'audit' / 'audit_summary.json').write_text(json.dumps(summary, indent=1))
    (tmp / 'params.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    started = time.monotonic()
    from .ts_v2_positive_plot import plot_events
    chosen = choose_figures(tables['positive_events'], cfg)
    figures = plot_events(tmp / 'figures', chosen, tables, markets, cfg)
    seconds['figures'] = round(time.monotonic() - started, 1)
    source = dict(version=cfg['version'], direction=side, command=command, fold='TRAIN', mode='OFFLINE_TRAIN',
                  rows=rows, stats=stats, figures=figures, workers=workers,
                  seconds=dict(seconds, total=round(time.monotonic() - began, 1)),
                  created=time.strftime('%Y-%m-%d %H:%M:%S'), repos=_repos())
    (tmp / 'source.json').write_text(json.dumps(source, indent=1, default=str))
    tmp.rename(out)
    log(json.dumps(summary['total'], indent=1))
    return summary


def assemble(events, candidates, cfg=None):
    """Teacher tables (positives only) and audit tables (everything). With
    `cfg` timeline.wait_in_rule (V4, user 2026-11-05: "正例之前的位置标
    WAIT"), every candidate of a touch before its first positive is a
    WAIT_IN_RULE row (at the activation minute in place of its ACTIVATION
    row), but a MASKED one: no ADD is legal there, so there is no decision
    to teach; a touch without a positive keeps none."""
    in_rule = bool(cfg and cfg['timeline'].get('wait_in_rule'))
    by_event = {}
    for c in candidates:
        by_event.setdefault(c['event_id'], []).append(c)
    entries, timeline, positive_events = [], [], []
    for x in events:
        mine = by_event.get(x['event_id'], [])
        good = [c for c in mine if c['reason'] == 'POSITIVE']
        x.update(candidates=len(mine), positives=len(good),
                 excellent=sum(c['label_grade'] == 'EXCELLENT' for c in good))
        if not good:
            continue
        for k, c in enumerate(good):
            c['entry_id'] = f'{x["event_id"]}:{_letter(k)}'
        pnl = np.array([c['label_net_pnl'] for c in good])
        rule_wait = ([c for c in mine if c['cursor'] < good[0]['cursor'] and c['reason'] != 'MASKED']
                     if in_rule else [])
        positive_events.append(dict(
            {k: x[k] for k in ('event_id', 'contract', 'direction_source', 'H', 'level_bucket_end',
                               'level_confirm', 'level_departed', 'touch', 'line_spike_time', 'line_spike_rv',
                               'activation', 'activation_time', 'activation_rv', 'end', 'end_time',
                               'end_reason', 'cursors', 'candidates', 'positives', 'excellent')},
            entries=' '.join(c['entry_id'].rsplit(':', 1)[1] for c in good),
            first_positive_minutes=good[0]['obs_minutes_since_activation'],
            last_positive_minutes=good[-1]['obs_minutes_since_activation'],
            fill_range=max(c['label_fill_expected'] for c in good) - min(c['label_fill_expected'] for c in good),
            best_net_pnl=float(pnl.max()), median_net_pnl=float(np.median(pnl)),
            **(dict(wait_in_rule=len(rule_wait)) if in_rule else {})))
        entries += [dict(c) for c in good]
        add = {c['cursor']: c for c in good}
        for w in x['wait']:
            timeline.append(dict(event_id=x['event_id'], contract=x['contract'],
                                 direction_source=x['direction_source'], **w))
        act = x['activation']
        if act not in add and not any(c['cursor'] == act for c in rule_wait):
            timeline.append(dict(event_id=x['event_id'], contract=x['contract'], cursor=act,
                                 time=x['activation_time'], direction_source=x['direction_source'],
                                 teach='ACTIVATION', **x['activation_obs'], obs_minutes_since_activation=0))
        for c in rule_wait:
            timeline.append(dict({k: v for k, v in c.items() if k in META or k.startswith('obs_')},
                                 obs_explain=wait_note('WAIT_IN_RULE', c, cfg), teach='WAIT_IN_RULE',
                                 obs_in_zone=True))
        for c in good:
            timeline.append(dict({k: v for k, v in c.items() if k in META or k.startswith('obs_')},
                                 teach='ACTIVATION_ADD' if c['cursor'] == act else 'ADD', obs_in_zone=True))
    audit_events = [{k: v for k, v in x.items() if k != 'wait'} for x in events]
    return dict(positive_events=pd.DataFrame(positive_events),
                positive_entries=pd.DataFrame(entries),
                positive_timeline=pd.DataFrame(timeline),
                events=pd.DataFrame(audit_events), candidates=pd.DataFrame(candidates))


def _letter(k):
    """A..Z, then AA, AB, ..."""
    return chr(ord('A') + k) if k < 26 else _letter(k // 26 - 1) + chr(ord('A') + k % 26)


def _order(frame):
    """META first, then obs_*, then label_*, then the rest."""
    cols = list(frame.columns)
    rank = lambda c: (0 if c in META else 1 if c.startswith('obs_') else 2 if c.startswith('label_') else 3)
    return frame[sorted(cols, key=lambda c: (rank(c), cols.index(c)))]


def write_tables(folder, tables):
    for sub, names in (('teacher', TEACHER), ('audit', ('events', 'candidates'))):
        (Path(folder) / sub).mkdir(parents=True, exist_ok=True)
        for name in names:
            frame = tables[name]
            if name in TEACHER:
                frame = teacher_columns(frame, name)
            _order(frame).to_csv(Path(folder) / sub / f'{name}.csv', index=False)


def teacher_columns(frame, name):
    """The columns a teacher table may carry; the audit-only fields go."""
    if name == 'positive_entries':
        drop = [c for c in frame.columns if c in ('reason', 'label_fill_in_zone')]
        return frame.drop(columns=drop)
    if name == 'positive_timeline':
        return frame[[c for c in frame.columns if c in META or c.startswith('obs_')]]
    return frame


def load_positive(folder):
    """The teacher tables of a V2 corpus, and nothing of its audit. obs_*
    columns are the decision-time inputs; label_* are offline facts."""
    folder = Path(folder) / 'teacher'
    out = {name: pd.read_csv(folder / f'{name}.csv') for name in TEACHER}
    for name, frame in out.items():
        for c in frame.columns:
            if c.startswith('obs_') and any(w in c for w in ('pnl', 'grade', 'quality', 'exit', 'mae', 'mfe')):
                raise ValueError(f'{name}: {c} is not a decision-time input')
        if name == 'positive_timeline' and any(c.startswith('label_') for c in frame.columns):
            raise ValueError('positive_timeline carries offline labels')
    if not set(out['positive_timeline']['teach']) <= set(TEACH):
        raise ValueError('unknown teach rows')
    return out


# ---- the teacher of a Thinking Student corpus (V0.5, user 2026-10-31) ----

TEACHER_LABEL = dict(source='V2_POSITIVE_TEACHER', ready_at='ready_at: the TP exit of an ADD; '
                     'the last kept TP exit of its touch for a WAIT')
WAIT_ROWS = ('WAIT_FAR', 'WAIT_APPROACH', 'WAIT_NEAR_NO_VOLUME')
# The WAIT rows of a gated env (V2.3): at the rule positions before a touch's
# first positive; the pre-touch rows above are masked minutes there.
RULE_WAIT_ROWS = ('WAIT_IN_RULE',)


def graded_teacher(env, facts, folder, contract, side, index=None):
    """(graded arrays, summary) of one corpus from the positive material in
    `folder`, in the arrays of ts_v1_grades.grade so L3 / L5 read it as they
    are. ADD (grade GOOD / EXCELLENT, ready at its exit) on a positive with
    R <= the env's max_risk_points whose Native trade the corpus facts
    reproduce (TP, the same exit, the same net PnL; any other is an error);
    WAIT on the WAIT rows of the touches that keep a positive; no
    demonstration anywhere else (ACTIVATION and every other cursor). A
    positive whose line the env's stateless stop does not take (a lower line
    confirmed during its touch) is dropped as STOP_LINE_DIFFERS: its trade
    is not the env's. Entry price, side and MFE are measured as for the
    graded teacher. `side` is the contract's direction (V2.1: SHORT or LONG).

    A gated env (V2.3, entry.rule_gate; user 2026-11-05) needs the material
    of its own rules: every label is at a rule position (a legal minute);
    the stop is the touch's own line, so a positive whose SL the env does
    not reproduce is an error; WAIT is taught on the WAIT_IN_RULE rows only,
    and the pre-touch WAIT rows, masked minutes there, are dropped
    (WAIT_OFF_RULE). Under rules V5 the stop is the fill's (label_sl)."""
    from .ts_v1_corpus import KINDS
    from .ts_v1_grades import EXCELLENT, GOOD, MFE_MINUTES, measure
    out, m = measure(env, facts, index)
    n, max_risk = m['n'], env.stop_rule['max_risk_points']
    gate = getattr(env, 'rule_gate', None)
    if gate is not None and _material_rules(folder) != _json({k: v for k, v in gate[side].items() if k != 'direction'}):
        raise ValueError(f'{contract}: the material {folder} is not of the env\'s rule gate')
    tables = load_positive(folder)
    entries = tables['positive_entries'][tables['positive_entries'].contract == contract]
    timeline = tables['positive_timeline'][tables['positive_timeline'].contract == contract]
    adds = timeline[timeline.teach.isin(('ADD', 'ACTIVATION_ADD'))]
    if sorted(adds.cursor) != sorted(entries.cursor):
        raise ValueError(f'{contract}: the ADD rows of the timeline are not the positive entries')
    out.update(grade=np.full(n, -1, np.int8), ready_at=np.full(n, -1, np.int32),
               valid=np.zeros(n, bool), action=np.zeros(n, np.int8))
    dropped, kept, ready = {'RISK_ABOVE_MAX': 0, 'STOP_LINE_DIFFERS': 0}, set(), {}
    tp = KINDS.index('TP')
    for r in entries.itertuples():
        t = int(r.cursor)
        if r.label_R > max_risk:
            dropped['RISK_ABOVE_MAX'] += 1
            continue
        if gate is not None and not facts['eligible'][t]:
            raise ValueError(f'{contract} cursor {t}: a positive at a minute the rule gate masks')
        plan = env.planned_exit(t, Decimal(str(r.label_fill_expected)), side)
        want = r.label_sl if gate is not None and fixed_stop(gate[side]) is not None else r.obs_sl
        if float(plan['stop_loss_price']) != want:
            if gate is not None:
                raise ValueError(f'{contract} cursor {t}: the gated stop {plan["stop_loss_price"]} is not SL {want}')
            dropped['STOP_LINE_DIFFERS'] += 1
            continue
        if not (facts['filled'][t] and facts['kind'][t] == tp and facts['exit_cursor'][t] == r.label_exit_cursor
                and abs(facts['net_pnl'][t] - r.label_net_pnl) < 1e-6):
            raise ValueError(f'{contract} cursor {t}: the Native trade differs from the positive')
        out['grade'][t] = EXCELLENT if r.label_grade == 'EXCELLENT' else GOOD
        out['ready_at'][t] = int(r.label_exit_cursor)
        out['valid'][t], out['action'][t] = True, 1
        kept.add(r.event_id)
        ready[r.event_id] = max(ready.get(r.event_id, -1), int(r.label_exit_cursor))
    waits = timeline[timeline.teach.isin(RULE_WAIT_ROWS if gate is not None else WAIT_ROWS)]
    if gate is not None:
        dropped['WAIT_OFF_RULE'] = int(timeline.teach.isin(WAIT_ROWS).sum())
    dropped['WAIT_OF_DROPPED_TOUCH'] = int((~waits.event_id.isin(kept)).sum())
    for r in waits[waits.event_id.isin(kept)].itertuples():
        t = int(r.cursor)
        if out['valid'][t] or not facts['eligible'][t]:
            raise ValueError(f'{contract} cursor {t}: a WAIT row on an ADD or a masked cursor')
        out['ready_at'][t], out['valid'][t] = ready[r.event_id], True
    dropped['ACTIVATION'] = int((timeline.teach == 'ACTIVATION').sum())
    teacher = Path(folder) / 'teacher'
    summary = dict(
        contract=contract, label=TEACHER_LABEL, mfe_minutes=list(MFE_MINUTES),
        grades=dict(teacher=json.loads((Path(folder) / 'source.json').read_text())['version'], max_risk_points=max_risk, material=str(Path(folder).resolve()),
                    sha256={name: _sha256(teacher / f'{name}.csv') for name in TEACHER}),
        graded={'GOOD': int((out['grade'] == GOOD).sum()), 'EXCELLENT': int((out['grade'] == EXCELLENT).sum())},
        demo=dict(add=int((out['valid'] & (out['action'] == 1)).sum()),
                  wait=int((out['valid'] & (out['action'] == 0)).sum())),
        touches=dict(kept=len(kept), material=int(entries.event_id.nunique())),
        dropped=dropped, tp_checked=m['tp_checked'])
    if gate is not None:
        rule = int(env._gate(side)['open'][:n].sum())
        labels = summary['demo']['add'] + summary['demo']['wait']
        summary['rule_gate'] = dict(rule_positions=rule, labels=labels,
                                    label_coverage=round(labels / rule, 4) if rule else None)
    return out, summary


def _json(x):
    return json.loads(json.dumps(x, default=str))


def _material_rules(folder):
    """The rules of a material but its direction: a combined material's
    source.json `rules`, or a single one's params.yaml."""
    source = json.loads((Path(folder) / 'source.json').read_text())
    if 'rules' in source:
        return _json(source['rules'])
    return _json({k: v for k, v in yaml.safe_load((Path(folder) / 'params.yaml').read_text()).items()
                  if k != 'direction'})


def audit_summary(events, candidates, stats, cfg):
    """Counts of every stage and the base rates of all candidates, per
    contract and in total (no trade lists)."""
    def part(evs, cs):
        filled = [c for c in cs if c['label_filled']]
        first = {}
        for c in cs:
            first.setdefault(c['event_id'], c)
        firsts = [c for c in first.values() if c['label_filled']]

        def rates(rows, prefix):
            done = [c for c in rows if c[prefix + 'filled']]
            tp = [c for c in done if c[prefix + 'exit_reason'] == 'TAKE_PROFIT']
            pnl = [c[prefix + 'net_pnl'] for c in done if np.isfinite(c[prefix + 'net_pnl'])]
            return dict(trades=len(done), tp=len(tp), tp_rate=len(tp) / len(done) if done else None,
                        mean_net_pnl=float(np.mean(pnl)) if pnl else None)

        reasons = {r: sum(c['reason'] == r for c in cs) for r in REASONS}
        good = [c for c in cs if c['reason'] == 'POSITIVE']
        per_event = {}
        for c in good:
            per_event[c['event_id']] = per_event.get(c['event_id'], 0) + 1
        lines = {(x['contract'], x['level_bucket']) for x in evs if x['event_id'] in per_event}
        spike_now = {x['event_id'] for x in evs if x['line_spike'] == x['activation']}
        return dict(
            events=len(evs), events_with_candidates=sum(x['candidates'] > 0 for x in evs),
            events_with_positive=len(per_event),
            events_with_2plus_positives=sum(v >= 2 for v in per_event.values()),
            positives_per_event=_counts(per_event.values()),
            lines_with_positive=len(lines),
            events_by_touch=_counts(min(x['touch'], 4) for x in evs),
            positives_by_touch=_counts(min(c['obs_touch'], 4) for c in good),
            positives_spike_at_touch_start=sum(c['event_id'] in spike_now for c in good),
            end_reasons=_counts(x['end_reason'] for x in evs),
            candidates=len(cs), filled=len(filled), reasons=reasons,
            excellent=sum(c['label_grade'] == 'EXCELLENT' for c in good),
            good=sum(c['label_grade'] == 'GOOD' for c in good),
            positives_tp_at_segment_open=sum(c['label_exit_at_segment_open'] for c in good),
            base_rate_all_candidates=dict(target=rates(cs, 'label_'), parallel=rates(cs, 'label_tp15_')),
            base_rate_first_candidate=dict(target=rates(list(first.values()), 'label_'),
                                           parallel=rates(list(first.values()), 'label_tp15_')),
            first_filled=len(firsts))

    contracts = {name: dict(stats[name], **part([x for x in events if x['contract'] == name],
                                                [c for c in candidates if c['contract'] == name]))
                 for name in stats}
    total = part(events, candidates)
    total.update(contracts=len(stats), rows=sum(s['rows'] for s in stats.values()),
                 **{k: sum(s[k] for s in stats.values()) for k in ('levels', 'levels_spiked', 'levels_touched')})
    return dict(version=cfg['version'], selection_bias=(
        'Positives are chosen by their own Native outcome: their TP rate is 100% by construction. '
        'The base rates of all candidates and of the first candidate of each event are the '
        'rule\'s own record; neither shows a verified edge, and the direction is assumed.'),
        total=total, contracts=contracts)


def _counts(values):
    out = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


def choose_figures(positive_events, cfg):
    """Events with the most positives first, then by time, at most
    `per_contract` per contract and `max_events` in all; never by PnL."""
    f = cfg['figures']
    if positive_events.empty:
        return []
    order = positive_events.sort_values(['positives', 'activation_time'], ascending=[False, True])
    chosen, per = [], {}
    for _, x in order.iterrows():
        if per.get(x['contract'], 0) < f['per_contract'] and len(chosen) < f['max_events']:
            chosen.append(x['event_id'])
            per[x['contract']] = per.get(x['contract'], 0) + 1
    return chosen


def combine(folders, out):
    """One teacher material of several (V2.1, user 2026-11-04: the V2_1 SHORT
    material and its LONG mirror), so that every corpus has one graded
    teacher: each teacher table is the header and then the rows of every
    folder's, byte for byte and in the order given. The folders must share
    the version and every rule but the direction, and no contract."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'{out} exists; remove it to combine again')
    folders = [Path(f) for f in folders]
    params = [yaml.safe_load((f / 'params.yaml').read_text()) for f in folders]
    rules = [{k: v for k, v in x.items() if k != 'direction'} for x in params]
    if any(x['version'] not in ACTIVATION for x in params) or any(r != rules[0] for r in rules):
        raise ValueError('the materials differ in version or in rules other than the direction')
    contracts = [set(load_positive(f)['positive_events'].contract) for f in folders]
    if sum(map(len, contracts)) != len(set().union(*contracts)):
        raise ValueError('a contract is in more than one material')
    tmp = out.with_name(out.name + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / 'teacher').mkdir(parents=True)
    for name in TEACHER:
        texts = [(f / 'teacher' / f'{name}.csv').read_text().splitlines(keepends=True) for f in folders]
        if any(t[0] != texts[0][0] for t in texts):
            raise ValueError(f'{name}: the materials have different columns')
        (tmp / 'teacher' / f'{name}.csv').write_text(texts[0][0] + ''.join(x for t in texts for x in t[1:]))
    load_positive(tmp)
    source = dict(version=params[0]['version'], combined=[
        dict(folder=str(f.resolve()), direction=x['direction'], contracts=sorted(c),
             sha256={name: _sha256(f / 'teacher' / f'{name}.csv') for name in TEACHER})
        for f, x, c in zip(folders, params, contracts)],
        rules=rules[0], created=time.strftime('%Y-%m-%d %H:%M:%S'), repos=_repos())
    (tmp / 'source.json').write_text(json.dumps(source, indent=1, default=str))
    tmp.rename(out)
    return source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--pool', help='market pool (dreamerv3/configs/thinking_student/markets/pool.json)')
    parser.add_argument('--config', help='rules (dreamerv3/configs/thinking_student/v2_positive_entry.yaml)')
    parser.add_argument('--combine', nargs='+', metavar='MATERIAL',
                        help='combine these materials into --out (combine) instead of producing one')
    parser.add_argument('--out', required=True)
    parser.add_argument('--contracts', nargs='*',
                        help="TRAIN contracts of the config's direction (default: all of the pool)")
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args(argv)
    if args.combine:
        if args.pool or args.config or args.contracts:
            parser.error('--combine takes only --out')
        print(json.dumps(combine(args.combine, args.out)['combined'], indent=1))
        return
    if not (args.pool and args.config):
        parser.error('--pool and --config are required (or --combine)')
    cfg = load_config(args.config)
    rows = side_rows(args.pool, cfg['direction'], args.contracts)
    produce(rows, cfg, args.out, args.workers, command=' '.join(sys.argv))


if __name__ == '__main__':
    main()
