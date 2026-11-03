"""Thinking Student V2 positive entry corpus (Spec V0.3, user 2026-10-31;
rules V2 after the user's review of the same day).

Teacher demonstrations of one entry only: the direction is given as SHORT;
a known prior 15m high H is a line; the line's first 1m volume ratio
RV >= 2 near H (from its own 15m bucket on) marks it; every return to the
line after the price departed is a touch (an event); ADD at any in-zone
minute of a touch, SL = H + 5 (R <= 12), TP = fill - 2R. A line holds until
the price breaks it. The rules are docs/thinking_student_v2/DESIGN.md with
the values of dreamerv3/configs/thinking_student/v2_positive_entry.yaml.

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

VERSION = 'V2_POSITIVE_ENTRY_V2'
SIDE = 'SHORT'
LOTS = 3
# Order in which a candidate's first failing condition names its reason.
REASONS = ('FILL_OUT_OF_ZONE', 'MASKED', 'NOT_FILLED', 'STOP_LOSS', 'OTHER_EXIT',
           'RIGHT_CENSORED', 'TP_DEEP_MAE', 'TP_SLOW', 'POSITIVE')
TEACH = ('WAIT_FAR', 'WAIT_APPROACH', 'WAIT_NEAR_NO_VOLUME', 'ACTIVATION', 'ACTIVATION_ADD', 'ADD')
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
    if cfg.get('version') != VERSION or cfg.get('direction') != SIDE:
        raise ValueError(f'{path}: not a {VERSION} {SIDE} config')
    e = cfg['entry']
    if not 0 < e['stop_above_points'] <= e['stop_above_points'] + e['zone_points'] <= e['max_risk_points']:
        raise ValueError(f'{path}: the zone and SL allow a risk above max_risk_points')
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


def short_rows(pool, contracts=None):
    """TRAIN pool rows of SHORT contracts: the given ones (any other is
    refused) or all of the pool."""
    names = contracts or [r['contract'] for r in json.loads(Path(pool).read_text())['files']
                          if r.get('role') == 'train' and r.get('eligible_for_training')]
    rows = [train_row(c, pool) for c in names]
    if contracts:
        wrong = [r['contract'] for r in rows if contract_direction(r) != SIDE]
        if wrong:
            raise ValueError(f'not {SIDE} contracts: {wrong}')
        return rows
    return [r for r in rows if contract_direction(r) == SIDE]


def tp_points(cfg, risk, multiple):
    """TP distance of a short at `multiple` R, rounded away from the fill."""
    return int(math.ceil(multiple * risk - 1e-9))


# ---- causal market facts ----

def market_arrays(env, cfg):
    """Per-cursor bars, the executable short quotes and RV of one market."""
    raw, side = env._raw, SIDE
    times, bobs = raw.index, pd.DatetimeIndex(env.bobs)

    def quotes(column, at, opening):
        return np.array([float(env._price(v, side, t, opening=opening)) for v, t in zip(raw[column], at)])

    starts = {a for a, _ in env.market_clock['segments']}
    valid = np.asarray(env.valid_fill_clock, bool)
    segment_open = np.asarray(bobs.strftime('%H:%M').isin(starts), bool)
    v = cfg['volume']
    eligible = valid & ~segment_open if v['exclude_segment_open'] else valid
    volume = raw.Volume.to_numpy(float)
    return dict(
        n=len(raw), times=times, bobs=bobs, times_ns=times.asi8, bob_ns=bobs.asi8,
        open=raw.Open.to_numpy(float), high=raw.High.to_numpy(float),
        low=raw.Low.to_numpy(float), close=raw.Close.to_numpy(float), volume=volume,
        valid=valid, segment_open=segment_open,
        sell_close=quotes('Close', times, True),   # an ADD decided at the close
        sell_open=quotes('Open', bobs, True),      # the fill of an ADD decided a bar earlier
        buy_high=quotes('High', times, False),     # the quote a short's SL checks
        rv=volume_ratio(volume, eligible, v['baseline_minutes']),
        prefix=env.market_clock_counter.prefix, multiplier=float(env.terms.multiplier))


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
    buckets = env._e_market.complete_buckets
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


def scan(a, levels, cfg, flat_end, contract):
    """Touches (events), candidates and the pre-touch WAIT rows of one
    market, in one causal pass over the cursors (DESIGN sections 2-5, 8)."""
    e, ev, p = cfg['entry'], cfg['event'], cfg['prior_high']
    stop, zone, near = e['stop_above_points'], e['zone_points'], ev['near_points']
    far, n = p['departure_points'], a['n']
    pending = sorted(levels, key=lambda x: x['confirm'])
    live, k = [], 0
    # A line may start a touch from its first departure on, and again from
    # the bar after each of its touches that ended by a departure.
    armed = {x['bucket']: x['departed'] for x in levels}
    touches = dict.fromkeys(armed, 0)
    active = np.full(n, -1, np.int64)  # bucket of the active line of a cursor outside touches
    events, candidates, event = [], [], None
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
            elif s >= flat_end or s + 1 >= n:
                reason = 'BOUNDARY'
            if reason is None:
                event['cursors'] += 1
                q = a['sell_close'][s]
                if a['valid'][s] and H - zone <= q <= H:
                    candidates.append(_candidate(a, event, s, cfg, contract))
                continue
            event.update(end=s, end_time=str(a['times'][s]), end_reason=reason)
            if reason == 'DEPARTED':
                armed[event['level_bucket']] = s + 1
            events.append(event)
            event = None
        ready = [x for x in live if armed[x['bucket']] <= s]
        if not ready or s >= flat_end or s + 1 >= n:
            continue
        # The first line above the price: the lowest armed one.
        level = min(ready, key=lambda x: (x['H'], -x['bucket']))
        active[s] = level['bucket']
        H = level['H']
        # Near H, not through the stop, and the line has had its volume
        # spike (this bar may be that spike).
        if (a['valid'][s] and a['high'][s] >= H - near and a['buy_high'][s] < H + stop
                and level['spike'] <= s):
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
                cursors=0, wait=_wait_rows(a, level, s, active, cfg))
            active[s] = -1
            # The activation minute is itself the first decision of the event.
            event['cursors'] += 1
            if H - zone <= a['sell_close'][s] <= H:
                candidates.append(_candidate(a, event, s, cfg, contract))
    if event is not None:
        event.update(end=n - 1, end_time=str(a['times'][n - 1]), end_reason='BOUNDARY')
        events.append(event)
    return events, candidates


def _candidate(a, event, s, cfg, contract):
    e = cfg['entry']
    H, t, q = event['H'], event['activation'], float(a['sell_close'][s])
    planned = H + e['stop_above_points'] - q
    fill = float(a['sell_open'][s + 1])
    risk = H + e['stop_above_points'] - fill
    return dict(
        event_id=event['event_id'], contract=contract, cursor=s, time=str(a['times'][s]),
        direction_source=event['direction_source'],
        obs_H=H, obs_close=float(a['close'][s]), obs_quote=q, obs_quote_below_h=H - q,
        obs_rv=float(a['rv'][s]), obs_rv_activation=event['activation_rv'],
        obs_minutes_since_activation=int(a['prefix'][s + 1] - a['prefix'][t + 1]),
        obs_touch=event['touch'], obs_line_spike_rv=event['line_spike_rv'],
        obs_sl=H + e['stop_above_points'], obs_planned_risk=planned,
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
    config, cursors, verify = job
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
                       mfe=float(_room(index, SIDE, fill, i, int(row['observed_until_cursor']), volume)),
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
    if not c['label_fill_in_zone']:
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


def _at(price, H):
    return f'{price:g} (H{price - H:+g})'


def _rv(rv):
    return f'RV {rv:.1f}' if np.isfinite(rv) else 'RV n/a'


def explain(c, cfg):
    """Why a candidate is placed where it is, from its obs_* facts only."""
    e, H = cfg['entry'], c['obs_H']
    return (f'quote {_at(c["obs_quote"], H)} in [H-{e["zone_points"]}, H]; touch {c["obs_touch"]} of the line '
            f'(line volume spike {_rv(c["obs_line_spike_rv"])}); {_rv(c["obs_rv"])}, touch began at '
            f'{_rv(c["obs_rv_activation"])} {c["obs_minutes_since_activation"]} min ago; '
            f'SL H+{e["stop_above_points"]} = {c["obs_sl"]:g}, planned R {c["obs_planned_risk"]:g}, '
            f'planned TP {c["obs_planned_tp"]:g} ({e["target_r"]:g}R)')


def wait_note(teach, o, cfg, spike_now=False):
    """Why a timeline row before an ADD waits or starts a touch (obs_*
    facts); `spike_now`: the touch's first bar is the line's volume spike."""
    H, far = o['obs_H'], cfg['prior_high']['departure_points']
    near, ratio = cfg['event']['near_points'], cfg['volume']['ratio']
    if teach == 'WAIT_FAR':
        return f'close {_at(o["obs_close"], H)} at or below H-{far}: far from H; wait'
    if teach == 'WAIT_APPROACH':
        return f'rebound, high {_at(o["obs_high"], H)} below H-{near}: not near H yet; wait'
    if teach == 'WAIT_NEAR_NO_VOLUME':
        return (f'high {_at(o["obs_high"], H)} near H but {_rv(o["obs_rv"])} < {ratio:g} and the line has '
                'had no volume spike; wait')
    why = (f'{_rv(o["obs_rv"])} >= {ratio:g}: the line\'s volume spike' if spike_now else
           f'the line had its volume spike ({_rv(o["obs_line_spike_rv"])}) before')
    return (f'high {_at(o["obs_high"], H)} near H, {why}: touch {o["obs_touch"]} starts; '
            f'quote {_at(o["obs_quote"], H)}')


def grade_reason(c, cfg):
    """The offline reason of a candidate's reason / grade."""
    q, e, H, reason = cfg['quality'], cfg['entry'], c['obs_H'], c['reason']
    fill = f'fill {_at(c["label_fill_expected"], H)}'
    if reason == 'FILL_OUT_OF_ZONE':
        return f'{fill} (next open - spread) outside [H-{e["zone_points"]}, H]'
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
    config = write_inputs(row, Path(folder) / 'inputs' / row['contract'] / 'base')
    env = native_env(config, market=True)
    if (env.take_profit_points, env.stop_loss_points) != (5, 30) or env.direction_at(
            env._raw.index[0], 0) not in (SIDE, 'NO_TRADE'):
        raise ValueError(f'{row["contract"]}: unexpected base env')
    a = market_arrays(env, cfg)
    levels, buckets = prior_highs(env, a, cfg)
    flat_end = min(CandidateIndex(env).flat_end, deadline(env))
    events, candidates = scan(a, levels, cfg, flat_end, row['contract'])
    def seen(x, s):  # decision-time facts of cursor s against the line of touch x
        H, q = x['H'], float(a['sell_close'][s])
        # The touch number is the count of earlier touches plus one; the
        # line's spike is known once it happened.
        return dict(obs_H=H, obs_close=float(a['close'][s]), obs_high=float(a['high'][s]),
                    obs_quote=q, obs_quote_below_h=H - q, obs_rv=float(a['rv'][s]),
                    obs_in_zone=bool(0 <= H - q <= cfg['entry']['zone_points']), obs_touch=x['touch'],
                    obs_line_spike_rv=x['line_spike_rv'] if x['line_spike'] <= s else math.nan)

    for x in events:
        x['wait'] = [dict(w, time=str(a['times'][w['cursor']]), **seen(x, w['cursor']),
                          obs_minutes_since_activation=-1) for w in x['wait']]
        for w in x['wait']:
            w['obs_explain'] = wait_note(w['teach'], w, cfg)
        x['activation_obs'] = seen(x, x['activation'])
        x['activation_obs']['obs_explain'] = wait_note('ACTIVATION', x['activation_obs'], cfg,
                                                       spike_now=x['line_spike'] == x['activation'])
    market = pd.DataFrame(dict(time=a['times'].astype(str), bob=a['bobs'].astype(str), open=a['open'],
                               high=a['high'], low=a['low'], close=a['close'], volume=a['volume'],
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
        mine = [c for c in candidates if c['contract'] == contract and c['label_fill_in_zone']]
        for risk in sorted({int(c['label_R']) for c in mine}):
            cursors = [c['cursor'] for c in mine if int(c['label_R']) == risk]
            for name, multiple in (('target', e['target_r']), ('parallel', e['parallel_r'])):
                tp = tp_points(cfg, risk, multiple)
                rules = dict(take_profit_points=tp, stop_loss_points=risk)
                config = write_inputs(by_row[contract], Path(folder) / 'inputs' / contract / f'R{risk}_TP{tp}', rules)
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
    started = time.monotonic()
    with _pool(max(1, min(workers, len(jobs)))) as pool:
        done = list(pool.map(_trade_job, [(j['config'], j['cursors'], []) for j in jobs]))
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
            if c['label_fill_in_zone'] else math.nan
        c['label_tp'] = c['label_fill_expected'] - tp_points(cfg, c['label_R'], cfg['entry']['target_r'])
        c['label_sl'] = c['obs_sl']
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
                verify.append((j['config'], cursors, cursors))
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

    tables = assemble(events, candidates)
    write_tables(tmp, tables)
    summary = audit_summary(events, candidates, stats, cfg)
    (tmp / 'audit' / 'audit_summary.json').write_text(json.dumps(summary, indent=1))
    (tmp / 'params.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    started = time.monotonic()
    from .ts_v2_positive_plot import plot_events
    chosen = choose_figures(tables['positive_events'], cfg)
    figures = plot_events(tmp / 'figures', chosen, tables, markets, cfg)
    seconds['figures'] = round(time.monotonic() - started, 1)
    source = dict(version=VERSION, command=command, fold='TRAIN', mode='OFFLINE_TRAIN',
                  rows=rows, stats=stats, figures=figures, workers=workers,
                  seconds=dict(seconds, total=round(time.monotonic() - began, 1)),
                  created=time.strftime('%Y-%m-%d %H:%M:%S'), repos=_repos())
    (tmp / 'source.json').write_text(json.dumps(source, indent=1, default=str))
    tmp.rename(out)
    log(json.dumps(summary['total'], indent=1))
    return summary


def assemble(events, candidates):
    """Teacher tables (positives only) and audit tables (everything)."""
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
        positive_events.append(dict(
            {k: x[k] for k in ('event_id', 'contract', 'direction_source', 'H', 'level_bucket_end',
                               'level_confirm', 'level_departed', 'touch', 'line_spike_time', 'line_spike_rv',
                               'activation', 'activation_time', 'activation_rv', 'end', 'end_time',
                               'end_reason', 'cursors', 'candidates', 'positives', 'excellent')},
            entries=' '.join(c['entry_id'].rsplit(':', 1)[1] for c in good),
            first_positive_minutes=good[0]['obs_minutes_since_activation'],
            last_positive_minutes=good[-1]['obs_minutes_since_activation'],
            fill_range=max(c['label_fill_expected'] for c in good) - min(c['label_fill_expected'] for c in good),
            best_net_pnl=float(pnl.max()), median_net_pnl=float(np.median(pnl))))
        entries += [dict(c) for c in good]
        add = {c['cursor']: c for c in good}
        for w in x['wait']:
            timeline.append(dict(event_id=x['event_id'], contract=x['contract'],
                                 direction_source=x['direction_source'], **w))
        act = x['activation']
        if act not in add:
            timeline.append(dict(event_id=x['event_id'], contract=x['contract'], cursor=act,
                                 time=x['activation_time'], direction_source=x['direction_source'],
                                 teach='ACTIVATION', **x['activation_obs'], obs_minutes_since_activation=0))
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


def graded_teacher(env, facts, folder, contract, index=None):
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
    graded teacher."""
    from .ts_v1_corpus import KINDS
    from .ts_v1_grades import EXCELLENT, GOOD, MFE_MINUTES, measure
    out, m = measure(env, facts, index)
    n, max_risk = m['n'], env.stop_rule['max_risk_points']
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
        plan = env.planned_exit(t, Decimal(str(r.label_fill_expected)), SIDE)
        if float(plan['stop_loss_price']) != r.obs_sl:
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
    waits = timeline[timeline.teach.isin(WAIT_ROWS)]
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
        grades=dict(teacher=VERSION, max_risk_points=max_risk, material=str(Path(folder).resolve()),
                    sha256={name: _sha256(teacher / f'{name}.csv') for name in TEACHER}),
        graded={'GOOD': int((out['grade'] == GOOD).sum()), 'EXCELLENT': int((out['grade'] == EXCELLENT).sum())},
        demo=dict(add=int((out['valid'] & (out['action'] == 1)).sum()),
                  wait=int((out['valid'] & (out['action'] == 0)).sum())),
        touches=dict(kept=len(kept), material=int(entries.event_id.nunique())),
        dropped=dropped, tp_checked=m['tp_checked'])
    return out, summary


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
    return dict(version=VERSION, selection_bias=(
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--pool', required=True,
                        help='market pool (dreamerv3/configs/thinking_student/markets/pool.json)')
    parser.add_argument('--config', required=True,
                        help='rules (dreamerv3/configs/thinking_student/v2_positive_entry.yaml)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--contracts', nargs='*', help='SHORT TRAIN contracts (default: all of the pool)')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    rows = short_rows(args.pool, args.contracts)
    produce(rows, cfg, args.out, args.workers, command=' '.join(sys.argv))


if __name__ == '__main__':
    main()
