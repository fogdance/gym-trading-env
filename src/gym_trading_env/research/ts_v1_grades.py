"""Graded entry teacher of a Thinking Student corpus (graded/, user decision
2026-10-28): the demonstrations of L3/L5 and the Outcome MFE targets.

The strict oracle (strict/) demonstrates ADD only where the exact TRAIN
optimum of a flat account starts a trade at that very minute: of several
equally good neighbouring entries one is ADD and the others WAIT, decided by
scheduling the student cannot see. The graded teacher grades the Native
trade "ADD now, then hold" of every flat funded cursor on its own instead.
P4 still executes the strict plan.

A filled ADD at cursor t fills at i = t + 1 with the side and price env.step
opens it at. Its trade gets three grades, EXCELLENT 3 / GOOD 2 / FAIR 1 /
POOR 0:

  time  a TP within time_minutes[0] / [1] market minutes of the same
        official trading day, a later TP that day, or a TP on another day
  mae   mae_full of the facts (fill to exit, executable quotes, both
        spreads) at most mae_points[0] / [1] / [2], or more
  room  the room at least room_points[0] / [1] / [2], or less

The room of a trade is the largest executable favourable move from its fill
price over the quotes a TP can fill at, as env.step exits: the close of the
fill minute when its volume leaves an exit, then the high (low) of every
later valid minute with Volume >= 3 (the TP index of fg_single3_oracle), up
to room_after_tp_minutes market minutes after its TP, within the official
trading day of the fill. A TP fills at the first such quote the env's TP
distance (entry.take_profit_points, TP5 by default) and a tick ahead, so the
room of a same-day TP is at least that much.

The grade of a trade is its worst grade. SL and other exits are POOR, and
so is a TP on a later day. A right-censored trade, an ADD that did not fill
and a same-day TP whose room window the data cuts have none. GOOD or better
demonstrates ADD, POOR demonstrates WAIT, FAIR demonstrates nothing.

The Outcome MFE targets mfe_15/60/120 measure the same room over the first
15/60/120 market minutes from the fill (the fill minute counts, as in
held_minutes), whatever the exit, cut at the end of the official trading
day of the fill. A window the data cuts first is not complete.

ts_v1_corpus.produce writes graded/ with the rest of a corpus; this CLI
(re)grades corpora that already exist:

  python -m gym_trading_env.research.ts_v1_grades \\
      --grades ../dreamerv3/configs/thinking_student/entry_grades.yaml \\
      --corpus-root /data/logdir/fg_single3/thinking_student/corpus --workers 18
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import yaml

from .fg_s3_quality_teacher_v2 import official_day_index
from .fg_single3_oracle import CandidateIndex
from .ts_v1_corpus import KINDS, LABELS, _pool, _repos, corpus_env, load_arrays, save_arrays

GRADES = ('POOR', 'FAIR', 'GOOD', 'EXCELLENT')
POOR, FAIR, GOOD, EXCELLENT = range(4)
MFE_MINUTES = (15, 60, 120)
KEYS = ('time_minutes', 'mae_points', 'room_points', 'room_after_tp_minutes')
# env.step exits on the fill minute only from the volume left after opening:
# the 3 lots it opened and the 3 lots to close.
FILL_MINUTE_EXIT_VOLUME = 6


def check_grades(grades, origin='grades'):
    """The thresholds of the teacher; anything else is refused."""
    if set(grades) != set(KEYS):
        raise ValueError(f'{origin}: keys {sorted(grades)} are not {sorted(KEYS)}')
    t, m, r, w = (grades[k] for k in KEYS)
    if len(t) != 2 or len(m) != 3 or len(r) != 3:
        raise ValueError(f'{origin}: 2 time, 3 mae and 3 room thresholds are needed')
    if list(t) != sorted(t) or list(m) != sorted(m) or list(r) != sorted(r, reverse=True):
        raise ValueError(f'{origin}: time and mae thresholds rise, room thresholds fall')
    if not (type(w) is int and w > 0 and min(t) > 0):
        raise ValueError(f'{origin}: market minutes are positive integers')
    return {k: grades[k] if k == 'room_after_tp_minutes' else [float(x) for x in grades[k]]
            for k in KEYS}


def load_grades(path):
    return check_grades(yaml.safe_load(Path(path).read_text()), str(path))


def _window_end(clock, day_end, first, minutes, stop, n):
    """(last cursor, complete) of `minutes` market minutes whose first is
    valid row number `first`, cut at `day_end`, the last cursor of the
    fill's official trading day. A window the day end cuts is complete when
    a later row shows the day ended; one the data (stop) cuts is not."""
    k = first + minutes - 1
    if k < len(clock.rows) and clock.rows[k] <= day_end:
        end = int(clock.rows[k])
        return end, end < stop
    return day_end, day_end < stop and day_end + 1 < n


def _room(index, side, price, i, end, volume):
    """Largest executable favourable move from `price` on the quotes a TP can
    fill at from the fill minute i through `end`; 0 without one."""
    sign = 1 if side == 'LONG' else -1
    best = sign * index.quotes[side]['close'][i] if volume[i] >= FILL_MINUTE_EXIT_VOLUME else -np.inf
    if end > i:
        best = max(best, -index.indices[side]['tp'].minimum(i + 1, end + 1))
    return max(0., float(best) - sign * price)


def measure(env, facts, index=None, calendar=None):
    """Fill price, side and MFE of every filled cursor, checked against the
    facts: the TP of each TP trade is the first quote a TP can fill at that
    is the env's TP distance (the trade's own TP under entry.stop_rule) and
    a tick ahead of the measured fill."""
    index = index or CandidateIndex(env)
    calendar = calendar or official_day_index(env)
    clock, stop, n = index.clock, index.stop, len(env._raw)
    day_last = np.array([calendar['last'][d] for d in calendar['days']], np.int64)
    opens, volume = env._raw.Open.to_numpy(), env._raw.Volume.to_numpy()
    tick, tp_points = float(env.terms.tick), env.take_profit_points
    out = dict(entry_price=np.full(n, np.nan), sign=np.zeros(n, np.int8),
               **{f'mfe_{h}': np.full(n, np.nan, np.float32) for h in MFE_MINUTES},
               **{f'mfe_{h}_complete': np.zeros(n, bool) for h in MFE_MINUTES})
    tp, checked = KINDS.index('TP'), 0
    for t in np.flatnonzero(facts['filled']):
        i = int(facts['entry_cursor'][t])
        side = env.direction_at(env.bobs[i], row_index=i)
        sign = 1 if side == 'LONG' else -1
        price = float(env._price(opens[i], side, env.bobs[i], opening=True))
        out['entry_price'][t], out['sign'][t] = price, sign
        for h in MFE_MINUTES:
            end, complete = _window_end(clock, int(day_last[i]), int(clock.prefix[i]), h, stop, n)
            out[f'mfe_{h}'][t] = _room(index, side, price, i, min(end, stop - 1), volume)
            out[f'mfe_{h}_complete'][t] = complete
        if facts['kind'][t] == tp:
            if env.stop_rule:  # this trade's own TP distance
                tp_points = sign * float(env.planned_exit(i - 1, price, side)['take_profit_price'] - Decimal(str(price)))
            first = (i if _room(index, side, price, i, i, volume) >= tp_points + tick else
                     index.indices[side]['tp'].first(i + 1, stop, -sign * (price + sign * tp_points) - tick))
            if first != facts['exit_cursor'][t]:
                raise ValueError(f'cursor {t}: TP at {facts["exit_cursor"][t]}, but the measured '
                                 f'{side} fill {price} reaches TP first at {first}')
            checked += 1
    return out, dict(index=index, clock=clock, day_last=day_last, stop=stop, n=n,
                     volume=volume, tp_checked=checked)


def grade(env, facts, grades, index=None, calendar=None, strict=None):
    """(graded arrays, summary) of every cursor of one corpus; the summary
    compares the demonstrations with the strict ones when given."""
    grades = check_grades(grades)
    out, m = measure(env, facts, index, calendar)
    n, clock, day_last, stop = m['n'], m['clock'], m['day_last'], m['stop']
    t_lim, m_lim, r_lim = (grades[k] for k in KEYS[:3])
    after = grades['room_after_tp_minutes']
    out.update(room=np.full(n, np.nan, np.float32), room_complete=np.zeros(n, bool),
               **{f'grade_{k}': np.full(n, -1, np.int8) for k in ('time', 'mae', 'room')},
               grade=np.full(n, -1, np.int8), ready_at=np.full(n, -1, np.int32))
    for t in np.flatnonzero(facts['filled']):
        kind, exit_at = KINDS[facts['kind'][t]], int(facts['exit_cursor'][t])
        if kind in ('SL', 'OTHER_EXIT'):
            out['grade'][t], out['ready_at'][t] = POOR, exit_at
            continue
        if kind != 'TP':
            continue  # RIGHT_CENSORED: no grade
        if facts['crossed_days'][t] > 0:
            out['grade_time'][t] = out['grade'][t] = POOR
            out['ready_at'][t] = exit_at
            continue
        held, i = int(facts['held_minutes'][t]), int(facts['entry_cursor'][t])
        out['grade_time'][t] = EXCELLENT - sum(held > x for x in t_lim)
        out['grade_mae'][t] = EXCELLENT - sum(float(facts['mae_full'][t]) > x for x in m_lim)
        # The room window runs from the fill to `after` market minutes after the TP.
        end, complete = _window_end(clock, int(day_last[i]), int(clock.prefix[exit_at + 1]),
                                    after, stop, n)
        side = 'LONG' if out['sign'][t] == 1 else 'SHORT'
        out['room'][t] = _room(m['index'], side, float(out['entry_price'][t]), i,
                               min(end, stop - 1), m['volume'])
        out['room_complete'][t] = complete
        if not complete:
            continue
        out['grade_room'][t] = EXCELLENT - sum(float(out['room'][t]) < x for x in r_lim)
        out['grade'][t] = min(out['grade_time'][t], out['grade_mae'][t], out['grade_room'][t])
        out['ready_at'][t] = end
    out['valid'] = (out['grade'] >= GOOD) | (out['grade'] == POOR)
    out['action'] = (out['grade'] >= GOOD).astype(np.int8)
    return out, dict(grades=grades, tp_checked=m['tp_checked'], **counts(out, strict))


def counts(out, strict=None):
    """Grade and demonstration counts, and the strict demonstrations of the
    same cursors when `strict` is given."""
    g, valid, action = out['grade'], out['valid'], out['action']
    res = dict(graded={GRADES[k]: int((g == k).sum()) for k in range(4)},
               ungraded_filled=int((np.isfinite(out['entry_price']) & (g < 0)).sum()),
               demo=dict(add=int((valid & (action == 1)).sum()),
                         wait=int((valid & (action == 0)).sum())))
    if strict is not None:
        s_valid = strict['valid'] & ~strict['tie']
        res['strict'] = dict(add=int((s_valid & (strict['action'] == 1)).sum()),
                             wait=int((s_valid & (strict['action'] == 0)).sum()))
        res['strict_vs_graded'] = {
            f'{s}->{d}': int((sm & dm).sum())
            for s, sm in (('ADD', s_valid & (strict['action'] == 1)),
                          ('WAIT', s_valid & (strict['action'] == 0)))
            for d, dm in (('ADD', valid & (action == 1)), ('WAIT', valid & (action == 0)),
                          ('none', ~valid))}
    return res


def save(folder, contract, arrays, summary, began):
    """graded arrays and summary.json of one corpus into `folder`."""
    summary = dict(contract=contract, label=LABELS['graded'], mfe_minutes=list(MFE_MINUTES),
                   **summary, seconds=round(time.monotonic() - began, 1),
                   created=time.strftime('%Y-%m-%d %H:%M:%S'), repos=_repos())
    save_arrays(folder, arrays)
    (folder / 'summary.json').write_text(json.dumps(summary, indent=1))
    return summary


def write(folder, grades, replace=False, log=print):
    """graded/ of an existing corpus folder, written whole (graded.tmp, then
    renamed)."""
    folder = Path(folder)
    out, tmp = folder / 'graded', folder / 'graded.tmp'
    if out.exists() and not replace:
        raise FileExistsError(f'{out} exists; pass --replace to regrade')
    began = time.monotonic()
    arrays, summary = grade(corpus_env(folder, market=False), load_arrays(folder / 'facts'),
                            grades, strict=load_arrays(folder / 'strict'))
    if tmp.exists():
        shutil.rmtree(tmp)
    summary = save(tmp, json.loads((folder / 'source.json').read_text())['contract'],
                   arrays, summary, began)
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    log(json.dumps({k: summary[k] for k in ('contract', 'graded', 'demo', 'seconds')}))
    return summary


def _write_job(job):
    return write(*job, log=lambda *_: None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--grades', required=True,
                        help='thresholds (dreamerv3/configs/thinking_student/entry_grades.yaml)')
    parser.add_argument('--corpus', action='append', default=None, help='one corpus folder')
    parser.add_argument('--corpus-root', default=None, help='every corpus folder under it')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--replace', action='store_true', help='regrade an existing graded/')
    args = parser.parse_args(argv)
    folders = [Path(f) for f in args.corpus or ()]
    if args.corpus_root:
        folders += sorted(p for p in Path(args.corpus_root).iterdir() if (p / 'source.json').exists())
    if not folders:
        parser.error('give --corpus or --corpus-root')
    grades = load_grades(args.grades)
    with _pool(min(args.workers or len(folders), len(folders))) as pool:
        for summary in pool.map(_write_job, [(f, grades, args.replace) for f in folders]):
            print(json.dumps({k: summary[k] for k in ('contract', 'graded', 'demo', 'strict',
                                                      'tp_checked', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
