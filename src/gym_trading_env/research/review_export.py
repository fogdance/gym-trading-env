"""review-v1 export of a Thinking Student corpus's teacher (generation side;
user 2026-11-04, dreamerv3/docs/thinking_student_v2/review_export_v1.md).

Reads only finished training facts and writes only CORPUS/review/: one JSONL
of the valid teacher ADD / WAIT labels (graded.valid) per contract and one
manifest. Every field is a recorded fact; nothing is recomputed:
- the market bars (bob, eob = Date, trading_date) come from the pool CSV of
  the corpus, whose sha256, rows and times must match the corpus;
- direction, entry price, grade and ready_at come from graded/, the exit and
  net PnL from facts/;
- SL / TP, exit price and the details come from the positive material named
  in graded/summary.json, whose teacher CSVs must have the recorded sha256.
Any disagreement stops the export. Training never reads review/ (it has no
source.json, so corpus discovery skips it).

Publishing is deterministic and never overwrites: a JSONL that exists with
the same bytes is reused, with other bytes it is a conflict; the manifest is
published once, after every file passed, and a rerun with the same content
leaves it untouched.

The run part (user 2026-11-04: run manifest, stage status and student trades
belong to the generation side) writes RUN/review/: the run manifest, fixed
to the published corpus review of paths.corpus_root (its id and manifest
sha256), with progress and status from run_state.json (formal_train records
status), and one student dataset per finished result of a frozen policy:
its TRAIN collection, TRAIN_SMOKE and DEV exams, and PAPER exams registered
by hand. A student record is one committed Native entry (its ENTRY fill,
open trade, closing trades and attribution, which must agree); an entry
still open when the result ended is CENSORED. A registered file is never
rewritten; the run manifest is replaced atomically, its revision raised
only when its content changed. formal_train calls `export_run` after each
state write and never lets a failed export stop the training; by hand, run
it only while no formal_train call runs on the same run (one writer).

  python -m gym_trading_env.research.review_export corpus --input CORPUS_DIR \\
      --label "V2.1 teacher" --expect-contracts 18
  python -m gym_trading_env.research.review_export run --input RUN_DIR
  python -m gym_trading_env.research.review_export paper --input RUN_DIR \\
      --results PAPER_DIR/TAG --policy pi_0000 --result r0001
  python -m gym_trading_env.research.review_export validate --input CORPUS_DIR|RUN_DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCHEMA = 'review-v1'
TIMEZONE = 'Asia/Shanghai'
CURRENCY = 'CNY'
FIELDS = ('id', 'symbol', 'action', 'direction', 'decision_time', 'entry_time', 'bar_bob', 'bar_eob',
          'trading_day', 'entry_price', 'volume', 'stop_price', 'target_price', 'grade', 'label_ready_at',
          'outcome', 'details', 'note')
OUTCOME = ('status', 'exit_time', 'exit_price', 'exit_reason', 'net_pnl')
STATUSES = ('CLOSED', 'OPEN', 'PARTIAL', 'CENSORED')
# graded.grade codes (ts_v1_grades: POOR, FAIR, GOOD, EXCELLENT = 0..3; WAIT -1).
GRADES = {2: 'GOOD', 3: 'EXCELLENT'}
# Text the exporter writes is English (user 2026-11-04); Risk R is the
# fill's risk (SL - fill), not the decision-time plan in the explanation.
NOTE_ADD, NOTE_WAIT = 'Native minute trace verified', 'Valid teacher WAIT'
LINE = dict(SHORT='Prior high H', LONG='Prior low L')
DATASET_LABEL = 'Teacher demonstrations of this training'
MANIFEST_KEYS = ('schema', 'kind', 'id', 'revision', 'label', 'strategy', 'status', 'updated_at', 'timezone',
                 'contracts', 'datasets')
RUN_KEYS = ('schema', 'kind', 'id', 'revision', 'label', 'strategy', 'status', 'updated_at', 'timezone', 'corpus',
            'progress', 'latest_policy_id', 'best_dev_policy_id', 'contracts', 'datasets')
RUN_STATUSES = ('running', 'completed', 'failed', 'cancelled')
SCOPES = ('TRAIN', 'TRAIN_SMOKE', 'DEV', 'PAPER')
STUDENT_SUMMARY = ('entries', 'closed', 'open', 'partial', 'censored')
RESULT = re.compile(r'^[A-Za-z0-9_]+$')
DAY = re.compile(r'^\d{4}-\d{2}-\d{2}$')


class ReviewError(ValueError):
    """A source disagreement or an invalid review file: nothing is published."""


class ConflictError(ReviewError):
    """A published file or manifest exists with other content."""


# ---- values ----

def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _iso(text):
    """ISO 8601 of a zoned source time ('2021-12-14 11:27:00+08:00')."""
    t = pd.Timestamp(text)
    if t.tzinfo is None:
        raise ReviewError(f'{text}: a time without its zone')
    return t.isoformat()


def _number(x):
    """A finite JSON number (an integral value as an int); unknown is an error."""
    x = float(x)
    if not math.isfinite(x):
        raise ReviewError(f'{x}: not a finite number')
    return int(x) if x.is_integer() else x


def _optional(x):
    """A finite number, or None for a missing one (NaN / empty)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return _number(x)


def dumps(record):
    return json.dumps(record, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n'


# ---- sources ----

def _read_json(path):
    return json.loads(Path(path).read_text())


def corpus_folders(root):
    return sorted(p for p in Path(root).iterdir() if (p / 'source.json').exists())


def _pool_market(row):
    """The bars of a pool row's CSV (bob, eob = Date, trading_date), after
    its sha256, rows and symbol are checked against the row. A row without
    rows (the SA paper pool, user 2026-11-06: "V3 paper SA 没有发布复盘数据") is
    pinned by its sha256 alone."""
    if _sha256(row['path']) != row['sha256']:
        raise ReviewError(f'{row["contract"]}: {row["path"]} is not the pool row\'s market')
    raw = pd.read_csv(row['path'], usecols=['Date', 'bob', 'trading_date', 'symbol'], dtype=str)
    if 'rows' in row and len(raw) != row['rows']:
        raise ReviewError(f'{row["contract"]}: market rows differ from the pool row')
    if set(raw.symbol) != {row['contract']}:
        raise ReviewError(f'{row["contract"]}: the market CSV holds another symbol')
    # bob < eob is checked on the bars a record marks (check_record): the
    # source has a few closing-auction bars with bob = eob (FG001 2019-09-24
    # 11:30) that no label uses.
    if not raw.trading_date.str.match(DAY.pattern).all():
        raise ReviewError(f'{row["contract"]}: a bar without a trading_date')
    eob = pd.to_datetime(raw.Date).dt.tz_localize(TIMEZONE)
    bob = pd.to_datetime(raw.bob).dt.tz_localize(TIMEZONE)
    return dict(eob=[t.isoformat() for t in eob], bob=[t.isoformat() for t in bob],
                day=raw.trading_date.tolist(), times=eob)


def _market(folder, source):
    """The pool CSV of the corpus: its bob, eob and trading_date per cursor,
    after its sha256, rows and times are checked against the corpus."""
    row = source['pool_row']
    m = _pool_market(row)
    times = np.load(Path(folder) / 'market' / 'time.npy')
    if len(m['eob']) != source['rows'] or len(times) != len(m['eob']):
        raise ReviewError(f'{row["contract"]}: market rows differ from the corpus')
    if not (m.pop('times') == pd.to_datetime(pd.Series(times))).all():
        raise ReviewError(f'{row["contract"]}: market times differ from the corpus')
    return m


def _arrays(folder, group, names):
    return {k: np.load(Path(folder) / group / f'{k}.npy') for k in names}


def material(summary):
    """The positive material a corpus's graded teacher names, its teacher
    CSVs checked against the recorded sha256."""
    from .ts_v2_positive import TEACHER, load_positive
    grades = summary['grades']
    folder = Path(grades['material'])
    for name in TEACHER:
        if _sha256(folder / 'teacher' / f'{name}.csv') != grades['sha256'][name]:
            raise ReviewError(f'{folder}: {name}.csv is not the material the corpus was built from')
    return load_positive(folder)


def _one(frame, contract, cursor, what):
    rows = frame[(frame.contract == contract) & (frame.cursor == cursor)]
    if len(rows) != 1:
        raise ReviewError(f'{contract} cursor {cursor}: {len(rows)} material {what} rows, not 1')
    return rows.iloc[0]


# ---- records ----

def contract_records(folder, tables):
    """The review-v1 records of one corpus's valid teacher labels, by
    cursor, and its file summary; `tables` is its material (`material`)."""
    from .ts_v2_positive import RULE_WAIT_ROWS, WAIT_ROWS, contract_direction
    folder = Path(folder)
    source = _read_json(folder / 'source.json')
    summary = _read_json(folder / 'graded' / 'summary.json')
    symbol = source['contract']
    side = contract_direction(dict(contract=symbol, env_config=str(folder / 'inputs' / 'env.yaml')))
    lots = yaml.safe_load((folder / 'inputs' / 'env.yaml').read_text())['trading']['trade_lot']
    g = _arrays(folder, 'graded', ('valid', 'action', 'grade', 'ready_at', 'entry_price', 'sign'))
    f = _arrays(folder, 'facts', ('entry_cursor', 'exit_cursor', 'exit_reason', 'net_pnl'))
    m = _market(folder, source)
    n = len(m['eob'])
    entries = tables['positive_entries']
    timeline = tables['positive_timeline']
    # The pre-touch WAIT rows; under a rule gate (V2.3) those at the rule
    # positions before a touch's first positive.
    waits = timeline[timeline.teach.isin(WAIT_ROWS + RULE_WAIT_ROWS)]
    records = []
    for s in np.flatnonzero(g['valid']):
        s = int(s)
        # sign is a fill fact (ts_v1_grades.measure): a WAIT label at a legal
        # minute whose ADD would not fill (NOT_FILLED, V2.4 corpus build
        # 2026-11-05: FG005 cursor 9175) has none and keeps the env's side.
        unfilled_wait = int(g['action'][s]) == 0 and int(g['sign'][s]) == 0
        direction = side if unfilled_wait else {1: 'LONG', -1: 'SHORT'}.get(int(g['sign'][s]))
        if direction != side:
            raise ReviewError(f'{symbol} cursor {s}: sign {g["sign"][s]} is not the env\'s direction {side}')
        ready = int(g['ready_at'][s])
        common = dict(id=f'teacher:{symbol}:{s}', symbol=symbol, direction=direction,
                      decision_time=m['eob'][s], label_ready_at=m['eob'][ready] if 0 <= ready < n else None)
        if int(g['action'][s]) == 1:
            e = _one(entries, symbol, s, 'positive entry')
            t, x = int(f['entry_cursor'][s]), int(f['exit_cursor'][s])
            grade = GRADES.get(int(g['grade'][s]))
            checks = {
                'grade': (grade, e.label_grade),
                'fill time': (m['bob'][t], _iso(e.label_fill_time)),
                'fill price': (_number(g['entry_price'][s]), _number(e.label_fill_expected)),
                'exit time': (m['eob'][x], _iso(e.label_exit_time)),
                'exit reason': (str(f['exit_reason'][s]), e.label_exit_reason),
                'net PnL': (_number(f['net_pnl'][s]), _number(e.label_net_pnl))}
            for what, (corpus, taught) in checks.items():
                if corpus != taught or corpus is None:
                    raise ReviewError(f'{symbol} cursor {s}: corpus {what} {corpus} != material {taught}')
            records.append(dict(
                common, action='ADD', entry_time=m['bob'][t], bar_bob=m['bob'][t], bar_eob=m['eob'][t],
                trading_day=m['day'][t], entry_price=_number(g['entry_price'][s]), volume=_number(lots),
                stop_price=_number(e.label_sl), target_price=_number(e.label_tp), grade=grade,
                outcome=dict(status='CLOSED', exit_time=m['eob'][x], exit_price=_number(e.label_exit_price),
                             exit_reason=str(f['exit_reason'][s]), net_pnl=_number(f['net_pnl'][s])),
                details={'Positive ID': e.entry_id, LINE[side]: _number(e.obs_H),
                         'Touch': _number(e.obs_touch), 'Line spike RV': _optional(e.obs_line_spike_rv),
                         'Touch start RV': _optional(e.obs_rv_activation), 'Decision quote': _number(e.obs_quote),
                         'Risk R (points)': _number(e.label_R), 'Held (market minutes)': _number(e.label_held_minutes),
                         'MAE (points)': _number(e.label_mae_points), 'MFE (points)': _number(e.label_mfe_points),
                         'Explanation': str(e.obs_explain)},
                note=NOTE_ADD if bool(e.label_native_verified) else ''))
        else:
            w = _one(waits, symbol, s, 'WAIT')
            records.append(dict(
                common, action='WAIT', entry_time=None, bar_bob=m['bob'][s], bar_eob=m['eob'][s],
                trading_day=m['day'][s], entry_price=None, volume=None, stop_price=None, target_price=None,
                grade=None, outcome=None,
                details={'Wait type': w.teach, LINE[side]: _number(w.obs_H), 'Touch': _number(w.obs_touch),
                         'RV': _optional(w.obs_rv),
                         **({'Decision quote': _number(w.obs_quote), 'Planned R (points)': _number(w.obs_planned_risk),
                             'Minutes since touch began': _number(w.obs_minutes_since_activation)}
                            if w.teach in RULE_WAIT_ROWS else {}),
                         'Explanation': str(w.obs_explain)},
                note=NOTE_WAIT))
    records = [{k: r[k] for k in FIELDS} for r in records]
    counts = file_summary(records)
    expect = dict(add=summary['demo']['add'], wait=summary['demo']['wait'],
                  excellent=summary['graded']['EXCELLENT'], good=summary['graded']['GOOD'])
    if counts != expect:
        raise ReviewError(f'{symbol}: exported {counts}, graded/summary.json {expect}')
    for r in records:
        check_record(r, symbol)
    return records, counts


def file_summary(records):
    grades = [r['grade'] for r in records]
    return dict(add=sum(r['action'] == 'ADD' for r in records), wait=sum(r['action'] == 'WAIT' for r in records),
                excellent=grades.count('EXCELLENT'), good=grades.count('GOOD'))


# ---- student records (Native results) ----

def native_state(collect):
    """The Native account state (export_state) a finished collection or exam
    left in its collect folder."""
    import pickle
    with open(Path(collect) / 'state.pkl', 'rb') as f:
        return pickle.load(f)['run']['env']


def _entry_order(entry_id):
    _, account, entry = entry_id.rsplit(':', 2)
    return int(account), int(entry)


def _dec(x):
    return Decimal(str(x))


def _student_record(symbol, eid, fill, exits, closed, still, trades, m):
    """One committed Native entry: its ENTRY fill, its open and closing
    trades, and its attribution (closed) or its open entry (still)."""
    n = len(m['eob'])
    source = closed if closed is not None else still
    t = source.get('entry_cursor')
    # Native fills an ADD of bar t - 1 at the open (bob) of bar t.
    if not isinstance(t, int) or not 0 < t < n:
        raise ReviewError(f'{eid}: no entry cursor to mark its bar')
    entry_time = _iso(fill['time'])
    if entry_time != m['bob'][t] or _iso(source['entry_time']) != entry_time:
        raise ReviewError(f'{eid}: the fill at {entry_time} is not the open of bar {t} ({m["bob"][t]})')
    if fill['side'] not in ('LONG', 'SHORT') or source['side'] != fill['side'] or (
            _dec(source['entry_price']) != _dec(fill['price'])):
        raise ReviewError(f'{eid}: the entry fill disagrees with its entry')
    opens = [x for x in trades if x['operation_type'].endswith('_OPEN')]
    closes = [x for x in trades if x['operation_type'].endswith('_CLOSE')]
    if len(opens) + len(closes) != len(trades) or len(opens) != 1 or (
            _dec(opens[0]['open_price']) != _dec(fill['price'])) or (
            _dec(opens[0]['position_size']) != _dec(fill['quantity'])):
        raise ReviewError(f'{eid}: its open trade disagrees with the entry fill')
    if len(closes) != len(exits):
        raise ReviewError(f'{eid}: {len(closes)} closing trades, {len(exits)} exit fills')
    gross = sum((_dec(x['pnl']) for x in closes), Decimal(0))
    fees = sum((_dec(x['fee']) for x in trades), Decimal(0))
    shut = sum((_dec(x['closed_size']) for x in closes), Decimal(0))
    details = {'Native entry ID': eid, 'Account': eid.rsplit(':', 1)[0], 'Stop source': fill.get('stop_source'),
               'Line': _optional(fill.get('line')), 'Risk (points)': _optional(fill.get('risk_points'))}
    if fill.get('rule_event') is not None:  # a rule-gated ADD (V2.3): the touch of its rule position
        details.update({'Rule touch': fill['rule_event'], 'Touch': _number(fill.get('touch'))})
    if closed is not None:
        x = closed.get('exit_cursor')
        if not isinstance(x, int) or not t <= x < n:
            raise ReviewError(f'{eid}: no exit cursor')
        if not exits or shut != _dec(fill['quantity']) or _dec(closed['closed_quantity']) != shut:
            raise ReviewError(f'{eid}: closed {shut} of {fill["quantity"]}')
        exit_time = _iso(closed['exit_time'])
        if exit_time != m['eob'][x] or _iso(exits[-1]['time']) != exit_time:
            raise ReviewError(f'{eid}: the exit at {exit_time} is not the end of bar {x} ({m["eob"][x]})')
        net = gross - fees
        if _dec(closed['net_trade_pnl']) != net:
            raise ReviewError(f'{eid}: attributed net {closed["net_trade_pnl"]}, its trades {net}')
        prices, reasons = {_dec(f['price']) for f in exits}, {f['reason'] for f in exits}
        if len(prices) == 1 and _dec(closed['exit_price']) != next(iter(prices)) or (
                len(reasons) == 1 and closed['reason'] != next(iter(reasons))):
            raise ReviewError(f'{eid}: the attribution disagrees with its exit fill')
        outcome = dict(status='CLOSED', exit_time=exit_time,
                       exit_price=_number(closed['exit_price']) if len(prices) == 1 else None,
                       exit_reason=closed['reason'] if len(reasons) == 1 else None, net_pnl=_number(net))
        details.update({'Held (market minutes)': _optional(closed.get('held_market_bars')),
                        'Gross PnL (CNY)': _number(gross), 'Fees (CNY)': _number(fees),
                        'Gap slippage (points)': _optional(closed.get('gap_slippage_points')),
                        'Exit fills': len(exits)})
    else:
        # The result ended (its collection or exam is finished) with the
        # entry not wholly closed: its later outcome is not visible.
        outcome = dict(status='CENSORED', exit_time=None, exit_price=None, exit_reason=None, net_pnl=None)
        details['Open at the end of the result'] = True
        if shut:
            details.update({'Closed quantity': _number(shut), 'Realized gross PnL (CNY)': _number(gross)})
    return dict(id=f'student:{eid}', symbol=symbol, action='ADD', direction=fill['side'],
                decision_time=m['eob'][t - 1], entry_time=entry_time, bar_bob=m['bob'][t], bar_eob=m['eob'][t],
                trading_day=m['day'][t], entry_price=_number(fill['price']), volume=_number(fill['quantity']),
                stop_price=_optional(fill.get('stop_loss_price')),
                target_price=_optional(fill.get('take_profit_price')), grade=None, label_ready_at=None,
                outcome=outcome, details=details, note='')


def student_records(env, m, symbol):
    """The review-v1 records of every committed entry of a Native state (its
    completed accounts first), by account and entry; `m` its market bars."""
    records = []
    for account in [*env['completed_accounts'], env]:
        if account['contract'] != symbol:
            raise ReviewError(f'{symbol}: a Native account of {account["contract"]}')
        fills = account['fills']
        if not all(f.get('committed') for f in fills):
            raise ReviewError(f'{symbol}: an uncommitted fill')
        entries = [f for f in fills if f['reason'] == 'ENTRY']
        opened = {f['entry_id']: f for f in entries}
        exits = {}
        for f in fills:
            if f['reason'] != 'ENTRY':
                exits.setdefault(f['entry_id'], []).append(f)
        closed = {a['entry_id']: a for a in account['attributions']}
        still = account['entries']
        trades = {}
        for x in account['trade_history']:
            eid = (x.get('meta') or {}).get('entry_id')
            if eid is None:
                raise ReviewError(f'{symbol}: a trade without its entry')
            trades.setdefault(eid, []).append(x)
        if (len(opened) != len(entries) or len(closed) != len(account['attributions'])
                or set(closed) & set(still) or set(closed) | set(still) != set(opened)
                or not set(exits) <= set(opened) or set(trades) != set(opened)):
            raise ReviewError(f'{symbol}: entry fills, trades, attributions and open entries disagree')
        for eid in sorted(opened, key=_entry_order):
            records.append(_student_record(symbol, eid, opened[eid], exits.get(eid, []), closed.get(eid),
                                           still.get(eid), trades[eid], m))
    for r in records:
        check_record(r, symbol, 'student')
    return records


def student_summary(records):
    status = [r['outcome']['status'] for r in records]
    return dict(entries=len(records), closed=status.count('CLOSED'), open=status.count('OPEN'),
                partial=status.count('PARTIAL'), censored=status.count('CENSORED'))


# ---- checks (shared by the export and validate) ----

def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_time(x):
    if not isinstance(x, str):
        return False
    try:
        return pd.Timestamp(x).tzinfo is not None and 'T' in x
    except ValueError:
        return False


def check_record(r, symbol, kind='teacher'):
    """One review-v1 record of a teacher or student file of `symbol`."""
    def bad(why):
        raise ReviewError(f'{r.get("id", "?") if isinstance(r, dict) else "?"}: {why}')
    if not isinstance(r, dict) or tuple(r) != FIELDS:
        bad(f'fields {list(r) if isinstance(r, dict) else type(r).__name__}, not {list(FIELDS)}')
    if not isinstance(r['id'], str) or not r['id'] or r['symbol'] != symbol:
        bad(f'id / symbol not of {symbol}')
    if r['action'] not in ('ADD', 'WAIT') or r['direction'] not in ('LONG', 'SHORT', None):
        bad('action / direction')
    for k in ('decision_time', 'entry_time', 'label_ready_at'):
        if r[k] is not None and not _is_time(r[k]):
            bad(f'{k} is not a zoned ISO time')
    if not (_is_time(r['bar_bob']) and _is_time(r['bar_eob'])
            and pd.Timestamp(r['bar_bob']) < pd.Timestamp(r['bar_eob'])):
        bad('bar_bob / bar_eob')
    if not isinstance(r['trading_day'], str) or not DAY.match(r['trading_day']):
        bad('trading_day')
    for k in ('entry_price', 'volume', 'stop_price', 'target_price'):
        if r[k] is not None and not _is_number(r[k]):
            bad(f'{k} is not a finite number')
    if r['grade'] is not None and not isinstance(r['grade'], str):
        bad('grade')
    if not isinstance(r['details'], dict) or not all(
            isinstance(k, str) and (v is None or isinstance(v, (str, bool)) or _is_number(v))
            for k, v in r['details'].items()):
        bad('details')
    if not isinstance(r['note'], str):
        bad('note')
    if r['action'] == 'WAIT':
        if any(r[k] is not None for k in ('entry_time', 'entry_price', 'volume', 'stop_price', 'target_price',
                                          'grade', 'outcome')):
            bad('a WAIT with an entry, a grade or an outcome')
    else:
        o = r['outcome']
        if not isinstance(o, dict) or tuple(o) != OUTCOME or o['status'] not in STATUSES:
            bad('an ADD outcome needs status and the exit fields')
        if any(o[k] is not None and not _is_time(o[k]) for k in ('exit_time',)) or any(
                o[k] is not None and not _is_number(o[k]) for k in ('exit_price', 'net_pnl')):
            bad('outcome exit time / price / net PnL')
    if kind == 'student':
        # A committed Native entry: its fill, no teacher label; a CLOSED one
        # has its exit time and net PnL, the others no exit at all.
        if r['action'] != 'ADD' or r['grade'] is not None or r['label_ready_at'] is not None or any(
                r[k] is None for k in ('direction', 'decision_time', 'entry_time', 'entry_price', 'volume')):
            bad('a student record is a committed ADD with its fill and no teacher label')
        o = r['outcome']
        if o['status'] == 'CLOSED':
            if o['exit_time'] is None or o['net_pnl'] is None:
                bad('a CLOSED entry needs its exit time and net PnL')
        elif any(o[k] is not None for k in OUTCOME[1:]):
            bad(f'a {o["status"]} entry has no exit fields')


def _strict_loads(line):
    def reject(constant):
        raise ReviewError(f'{constant} is not a JSON number')
    return json.loads(line, parse_constant=reject)


def check_file(path, symbol, records, sha256, kind='teacher'):
    """A published JSONL: its bytes, line count and every record."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ReviewError(f'{path}: sha256 differs from the manifest')
    text = data.decode('utf-8')
    if text and not text.endswith('\n'):
        raise ReviewError(f'{path}: the last line has no newline')
    rows = [_strict_loads(x) for x in text.splitlines()]
    if len(rows) != records:
        raise ReviewError(f'{path}: {len(rows)} lines, the manifest says {records}')
    ids = [r.get('id') if isinstance(r, dict) else None for r in rows]
    if len(set(ids)) != len(ids):
        raise ReviewError(f'{path}: duplicate record ids')
    for r in rows:
        check_record(r, symbol, kind)
    return rows


def _inside(review, rel):
    p = Path(rel)
    if p.is_absolute() or '..' in p.parts or not rel or rel != p.as_posix():
        raise ReviewError(f'{rel}: a data path must be relative and inside review/')
    return Path(review) / p


def validate(root):
    """A published corpus or run review (ROOT/review): its manifest, every
    registered file and record. Returns the totals; any failure raises
    ReviewError."""
    return validate_review(Path(root) / 'review')


def _load_manifest(review):
    path = Path(review) / 'manifest.json'
    if not path.exists():
        raise ReviewError(f'{path} does not exist')
    m = _strict_loads(path.read_text())
    if not isinstance(m, dict) or m.get('schema') != SCHEMA:
        raise ReviewError(f'{path}: schema is not {SCHEMA}')
    return path, m


def _check_contracts(path, contracts):
    symbols = [c.get('symbol') for c in contracts]
    if len(set(symbols)) != len(symbols) or not all(
            set(c) == {'symbol', 'price_tick', 'multiplier', 'currency'} and _is_number(c['price_tick'])
            and _is_number(c['multiplier']) for c in contracts):
        raise ReviewError(f'{path}: contracts')
    return symbols


def validate_review(review, corpus=True):
    """A review/ folder of either kind; `corpus` False skips the full check
    of the corpus a run is fixed to (its manifest sha256 is still checked)."""
    review = Path(review)
    path, m = _load_manifest(review)
    if m.get('kind') == 'run':
        return validate_run(review, m, corpus)
    if m.get('kind') != 'corpus' or tuple(m) != MANIFEST_KEYS or m['status'] != 'ready':
        raise ReviewError(f'{path}: not a ready corpus manifest of the review-v1 fields')
    symbols = _check_contracts(path, m['contracts'])
    totals = dict(files=0, records=0, add=0, wait=0, excellent=0, good=0)
    for d in m['datasets']:
        if d.get('kind') != 'teacher' or d.get('policy_id') is not None:
            raise ReviewError(f'{path}: dataset {d.get("id")} is not a teacher dataset')
        for f in d['files']:
            if f['symbol'] not in symbols:
                raise ReviewError(f'{path}: {f["symbol"]} is not a registered contract')
            rows = check_file(_inside(review, f['path']), f['symbol'], f['records'], f['sha256'])
            if file_summary(rows) != f['summary']:
                raise ReviewError(f'{f["path"]}: the summary differs from its records')
            totals['files'] += 1
            totals['records'] += len(rows)
            for k, v in f['summary'].items():
                totals[k] += v
    return totals


def _corpus_ref(review, ref):
    """The corpus manifest a run manifest is fixed to: the file it names
    (relative to the run's review/) must have its id and sha256."""
    if not isinstance(ref, dict) or set(ref) != {'id', 'manifest', 'sha256'}:
        raise ReviewError(f'{review}: the corpus reference needs id, manifest and sha256')
    target = (Path(review) / ref['manifest']).resolve()
    if target.name != 'manifest.json' or target.parent.name != 'review' or not target.exists():
        raise ReviewError(f'{review}: the corpus manifest {ref["manifest"]} does not exist')
    if _sha256(target) != ref['sha256']:
        raise ReviewError(f'{review}: the corpus manifest {target} is not the one this run is fixed to')
    if _read_json(target).get('id') != ref['id']:
        raise ReviewError(f'{review}: the corpus manifest {target} is not of {ref["id"]}')
    return target.parent


def validate_run(review, m=None, corpus=True):
    """A run review: the fixed corpus, the progress and status, every
    registered student file and its summary (entries = records = closed +
    open + partial + censored); the policy ids refer to published results."""
    review = Path(review)
    path, m = (review / 'manifest.json', m) if m is not None else _load_manifest(review)
    if m.get('kind') != 'run' or tuple(m) != RUN_KEYS or m['status'] not in RUN_STATUSES or not (
            isinstance(m['revision'], int) and m['revision'] >= 1) or not _is_time(m['updated_at']):
        raise ReviewError(f'{path}: not a run manifest of the review-v1 fields')
    folder = _corpus_ref(review, m['corpus'])
    if corpus:
        validate_review(folder)
    p = m['progress']
    if not isinstance(p, dict) or set(p) != {'phase', 'step', 'round'}:
        raise ReviewError(f'{path}: progress needs phase, step and round')
    symbols = _check_contracts(path, m['contracts'])
    ids = [d.get('id') for d in m['datasets']]
    if len(set(ids)) != len(ids):
        raise ReviewError(f'{path}: duplicate dataset ids')
    totals = dict(files=0, records=0, **{k: 0 for k in STUDENT_SUMMARY})
    policies = set()
    for d in m['datasets']:
        if d.get('kind') != 'student' or d.get('scope') not in SCOPES or not isinstance(d.get('policy_id'), str) or (
                d['id'] != f'student:{d["policy_id"]}:{d["scope"]}:{d["id"].rsplit(":", 1)[-1]}'):
            raise ReviewError(f'{path}: dataset {d.get("id")} is not a student dataset')
        policies.add(d['policy_id'])
        for f in d['files']:
            if f['symbol'] not in symbols or f.get('market_complete') not in (True, False, None) or set(f) != {
                    'symbol', 'path', 'records', 'sha256', 'summary', 'market_complete'}:
                raise ReviewError(f'{path}: file {f.get("path")} of {d["id"]}')
            rows = check_file(_inside(review, f['path']), f['symbol'], f['records'], f['sha256'], 'student')
            if student_summary(rows) != f['summary']:
                raise ReviewError(f'{f["path"]}: the summary differs from its records')
            totals['files'] += 1
            totals['records'] += len(rows)
            for k, v in f['summary'].items():
                totals[k] += v
    for k in ('latest_policy_id', 'best_dev_policy_id'):
        if m[k] is not None and m[k] not in policies:
            raise ReviewError(f'{path}: {k} {m[k]} has no published result')
    return dict(totals, datasets=len(m['datasets']), revision=m['revision'], status=m['status'])


# ---- publishing ----

def _publish(data, final):
    """Write `data` to `final` without ever overwriting: 'published', or
    'reused' if `final` already holds these bytes; other bytes: conflict."""
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f'.{final.name}.{os.getpid()}.tmp')
    tmp.write_bytes(data)
    try:
        os.link(tmp, final)
        return 'published'
    except FileExistsError:
        if final.read_bytes() != data:
            raise ConflictError(f'{final} exists with other content; nothing overwritten') from None
        return 'reused'
    finally:
        tmp.unlink()


def _contract_meta(folder):
    c = yaml.safe_load((Path(folder) / 'inputs' / 'contract.yaml').read_text())
    return dict(symbol=c['contract'], price_tick=_number(c['tick_size']), multiplier=_number(c['lot_size']),
                currency=CURRENCY)


def export_corpus(root, label, expect_contracts, out=None, now=None, log=print):
    """Export every corpus of `root` (exactly `expect_contracts`) into
    `out` (default ROOT/review) and publish its manifest once."""
    root = Path(root)
    review = Path(out) if out is not None else root / 'review'
    folders = corpus_folders(root)
    if len(folders) != expect_contracts:
        raise ReviewError(f'{root}: {len(folders)} corpora, expected {expect_contracts}')
    sources = [_read_json(f / 'source.json') for f in folders]
    summaries = [_read_json(f / 'graded' / 'summary.json') for f in folders]
    strategies = {json.dumps(s['inputs'].get('strategy'), sort_keys=True) for s in sources}
    teachers = {json.dumps({k: v for k, v in s['grades'].items()}, sort_keys=True) for s in summaries}
    if len(strategies) != 1 or len(teachers) != 1:
        raise ReviewError(f'{root}: the corpora differ in strategy or teacher')
    tables = material(summaries[0])
    files, contracts, data = [], [], {}
    for folder in folders:
        records, counts = contract_records(folder, tables)
        meta = _contract_meta(folder)
        rel = f'teacher/{meta["symbol"]}.jsonl'
        data[rel] = ''.join(map(dumps, records)).encode('utf-8')
        files.append(dict(symbol=meta['symbol'], path=rel, records=len(records),
                          sha256=hashlib.sha256(data[rel]).hexdigest(), summary=counts))
        contracts.append(meta)
        log(f'{meta["symbol"]}: {len(records)} records {counts}')
    strategy = sources[0]['inputs'].get('strategy') or {}
    manifest = dict(schema=SCHEMA, kind='corpus', id=root.name, revision=1, label=label,
                    strategy=strategy.get('version'), status='ready',
                    updated_at=now or pd.Timestamp.now(tz=TIMEZONE).floor('s').isoformat(), timezone=TIMEZONE,
                    contracts=contracts,
                    datasets=[dict(id='teacher', label=DATASET_LABEL, kind='teacher', scope='TRAIN',
                                   policy_id=None, files=files)])
    path = review / 'manifest.json'
    if path.exists():
        old = _read_json(path)
        if {k: v for k, v in old.items() if k != 'updated_at'} != {
                k: v for k, v in manifest.items() if k != 'updated_at'}:
            raise ConflictError(f'{path} is published with other content; a new teacher needs a new corpus id')
    states = {rel: _publish(blob, review / rel) for rel, blob in data.items()}
    for f in files:
        check_file(review / f['path'], f['symbol'], f['records'], f['sha256'])
    if path.exists():
        status = 'unchanged'
    else:
        _publish((json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8'), path)
        status = 'published'
    totals = validate_review(review)
    return dict(manifest=status, files={k: list(states.values()).count(k) for k in ('published', 'reused')},
                totals=totals)


# ---- run export ----

def _exam_part(folder):
    """(symbol, collect folder, pool row, folder of inputs/contract.yaml) of
    a finished exam (exam.json is written last)."""
    folder = Path(folder)
    source = _read_json(folder / 'exam.json')['source']
    if 'row' in source:
        row, corpus = source['row'], None
    else:
        corpus = Path(source['corpus'])
        row = _read_json(corpus / 'source.json')['pool_row']
    inputs = folder if (folder / 'inputs' / 'contract.yaml').exists() else corpus
    if inputs is None or row['contract'] != source.get('contract', row['contract']):
        raise ReviewError(f'{folder}: the exam names no market of its contract')
    return row['contract'], folder / 'collect', row, inputs


def run_results(root, cfg, state):
    """The finished student results of a formal run, per frozen policy:
    (policy, scope, result, parts). A result counts when all of it is done:
    the TRAIN collection of every contract of the policy's round, the
    TRAIN smoke of `smoke_contracts` contracts, the DEV exam of every DEV
    contract. Each is the policy's one result of that scope (r0001); its
    files are ordered by symbol."""
    root = Path(root)
    corpus_root = Path(cfg['paths']['corpus_root'])
    out = []
    for policy in sorted((root / 'policies').glob('pi_[0-9][0-9][0-9][0-9]')):
        if not (policy / 'policy.json').exists():
            continue
        name = policy.name
        rec = (state.get('rounds') or {}).get(str(int(name[3:]) + 1))
        if rec and rec.get('policy_in') == name and rec.get('contracts'):
            parts = []
            for c in rec['contracts']:
                collect = root / 'experience' / name / 'collect' / c
                meta = _read_json(collect / 'meta.json') if (collect / 'meta.json').exists() else None
                if not (meta and meta.get('done')):
                    break
                parts.append((c, collect, _read_json(corpus_root / c / 'source.json')['pool_row'], corpus_root / c))
            else:
                out.append((name, 'TRAIN', 'r0001', sorted(parts, key=lambda x: x[0])))
        smoke = sorted((root / 'eval' / name / 'TRAIN_SMOKE').glob('*/exam.json'))
        if smoke and len(smoke) == cfg['evaluation']['smoke_contracts']:
            out.append((name, 'TRAIN_SMOKE', 'r0001', sorted((_exam_part(e.parent) for e in smoke), key=lambda x: x[0])))
        dev = [root / 'eval' / name / 'DEV' / c for c in cfg['dev']['contracts']]
        if dev and all((d / 'exam.json').exists() for d in dev):
            out.append((name, 'DEV', 'r0001', sorted(map(_exam_part, dev), key=lambda x: x[0])))
    return out


def paper_results(folder, policy, result):
    """A paper result: one exam (exam --paper-pool) per contract folder."""
    folder = Path(folder)
    parts = sorted(p for p in folder.iterdir() if p.is_dir())
    if not parts or not all((p / 'exam.json').exists() for p in parts):
        raise ReviewError(f'{folder}: not every paper exam is finished')
    return policy, 'PAPER', result, sorted(map(_exam_part, parts), key=lambda x: x[0])


def _dataset_label(policy, scope, result):
    return f'Student {policy} · {scope}' + ('' if result == 'r0001' else f' · {result}')


def export_run(root, paper=(), now=None, log=print):
    """Publish or update RUN/review: the run manifest fixed to the corpus
    review of paths.corpus_root, its progress and status from run_state.json,
    and a student dataset for each finished result not yet registered
    (`paper`: (folder, policy, result) of paper results to register too).
    Registered datasets and files are kept as they are."""
    root = Path(root)
    review = root / 'review'
    cfg = yaml.safe_load((root / 'config.yaml').read_text())
    state = _read_json(root / 'run_state.json')
    if state.get('status') not in RUN_STATUSES:
        raise ReviewError(f'{root}: run_state.json records no status {RUN_STATUSES}')
    path = review / 'manifest.json'
    old = _read_json(path) if path.exists() else None
    corpus_review = Path(cfg['paths']['corpus_root']) / 'review'
    target = corpus_review / 'manifest.json'
    if not target.exists():
        raise ReviewError(f'{target}: the corpus review is not published')
    ref = dict(id=_read_json(target)['id'], manifest=os.path.relpath(target, review), sha256=_sha256(target))
    if old is None:
        validate_review(corpus_review)
    elif old.get('kind') != 'run' or old.get('corpus') != ref:
        raise ConflictError(f'{path} is fixed to the corpus {old.get("corpus")}, not {ref}')
    datasets = {d['id']: d for d in (old['datasets'] if old else [])}
    contracts = {c['symbol']: c for c in (old['contracts'] if old else [])}
    for d in datasets.values():
        for f in d['files']:
            if _sha256(_inside(review, f['path'])) != f['sha256']:
                raise ReviewError(f'{f["path"]}: a registered file changed')
    for _, policy, _ in paper:
        if not (root / 'policies' / policy / 'policy.json').exists():
            raise ReviewError(f'{policy} is not a frozen policy of {root.name}')
    new = []
    for policy, scope, result, parts in run_results(root, cfg, state) + [paper_results(*p) for p in paper]:
        did = f'student:{policy}:{scope}:{result}'
        if not RESULT.match(result) or did in datasets:
            continue
        files = []
        for symbol, collect, row, inputs in parts:
            meta = _contract_meta(inputs)
            if meta['symbol'] != symbol or contracts.get(symbol, meta) != meta:
                raise ReviewError(f'{symbol}: contract terms differ between results')
            env = native_state(collect)
            records = student_records(env, _pool_market(row), symbol)
            rel = f'student/{policy}/{scope}/{result}/{symbol}.jsonl'
            blob = ''.join(map(dumps, records)).encode('utf-8')
            _publish(blob, review / rel)
            sha = hashlib.sha256(blob).hexdigest()
            check_file(review / rel, symbol, len(records), sha, 'student')
            files.append(dict(symbol=symbol, path=rel, records=len(records), sha256=sha,
                              summary=student_summary(records), market_complete=bool(env['terminated'])))
            contracts[symbol] = meta
        datasets[did] = dict(id=did, label=_dataset_label(policy, scope, result), kind='student', scope=scope,
                             policy_id=policy, files=files)
        new.append(did)
        log(f'{did}: ' + ', '.join(f'{f["symbol"]} {f["summary"]}' for f in files))
    published = {d['policy_id'] for d in datasets.values()}
    latest = Path(state['latest_candidate']).name if state.get('latest_candidate') else None
    best = Path(state['best_dev']['policy']).name if state.get('best_dev') else None
    manifest = dict(schema=SCHEMA, kind='run', id=root.name, revision=None,
                    label=(cfg.get('review') or {}).get('label') or root.name,
                    strategy=_read_json(target)['strategy'], status=state['status'], updated_at=None,
                    timezone=TIMEZONE, corpus=ref,
                    progress=dict(phase=state.get('phase'), step=state.get('step'), round=state.get('round')),
                    latest_policy_id=latest if latest in published else None,
                    best_dev_policy_id=best if best in published else None,
                    contracts=[contracts[s] for s in sorted(contracts)],
                    datasets=sorted(datasets.values(),
                                    key=lambda d: (d['policy_id'], SCOPES.index(d['scope']), d['id'])))
    fixed = ('revision', 'updated_at')
    if old is not None and {k: v for k, v in old.items() if k not in fixed} == {
            k: v for k, v in manifest.items() if k not in fixed}:
        return dict(manifest='unchanged', revision=old['revision'], datasets=len(datasets), new=new)
    manifest['revision'] = old['revision'] + 1 if old else 1
    manifest['updated_at'] = now or pd.Timestamp.now(tz=TIMEZONE).floor('s').isoformat()
    review.mkdir(parents=True, exist_ok=True)
    validate_run(review, manifest, corpus=False)
    tmp = path.with_name('manifest.json.tmp')
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)
    return dict(manifest='published' if old is None else 'updated', revision=manifest['revision'],
                datasets=len(datasets), new=new)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('corpus', help='export the teacher of a corpus root into ROOT/review')
    c.add_argument('--input', required=True, help='corpus root (one folder per contract)')
    c.add_argument('--label', required=True, help='manifest label, e.g. "V2.1 teacher"')
    c.add_argument('--expect-contracts', type=int, required=True, help='corpora the root must hold')
    r = sub.add_parser('run', help='publish or update RUN/review from the run\'s finished results')
    r.add_argument('--input', required=True, help='formal run root (run_state.json, config.yaml)')
    p = sub.add_parser('paper', help='register a paper result of a frozen policy of the run')
    p.add_argument('--input', required=True, help='formal run root')
    p.add_argument('--results', required=True, help='paper folder of one policy: one exam folder per contract')
    p.add_argument('--policy', required=True, help='the frozen policy that traded, e.g. pi_0000')
    p.add_argument('--result', required=True, help='result id of this paper run, e.g. r0001')
    v = sub.add_parser('validate', help='check ROOT/review/manifest.json and its files')
    v.add_argument('--input', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'corpus':
            out = export_corpus(args.input, args.label, args.expect_contracts)
        elif args.command == 'run':
            out = export_run(args.input)
        elif args.command == 'paper':
            out = export_run(args.input, paper=[(args.results, args.policy, args.result)])
        else:
            out = dict(totals=validate(args.input))
    except ReviewError as e:
        print(f'review export failed: {e}', file=sys.stderr)
        sys.exit(1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
