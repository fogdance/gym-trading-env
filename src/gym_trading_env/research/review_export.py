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
leaves it untouched. The run / student part of review-v1 is not here.

  python -m gym_trading_env.research.review_export corpus --input CORPUS_DIR \\
      --label "V2.1 teacher" --expect-contracts 18
  python -m gym_trading_env.research.review_export validate --input CORPUS_DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
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
WAIT_ROWS = ('WAIT_FAR', 'WAIT_APPROACH', 'WAIT_NEAR_NO_VOLUME')
# Text the exporter writes is English (user 2026-11-04); Risk R is the
# fill's risk (SL - fill), not the decision-time plan in the explanation.
NOTE_ADD, NOTE_WAIT = 'Native minute trace verified', 'Valid teacher WAIT'
LINE = dict(SHORT='Prior high H', LONG='Prior low L')
DATASET_LABEL = 'Teacher demonstrations of this training'
MANIFEST_KEYS = ('schema', 'kind', 'id', 'revision', 'label', 'strategy', 'status', 'updated_at', 'timezone',
                 'contracts', 'datasets')
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


def _market(folder, source):
    """The pool CSV of the corpus: its bob, eob and trading_date per cursor,
    after its sha256, rows and times are checked against the corpus."""
    row = source['pool_row']
    if _sha256(row['path']) != row['sha256']:
        raise ReviewError(f'{row["contract"]}: {row["path"]} is not the pool row\'s market')
    raw = pd.read_csv(row['path'], usecols=['Date', 'bob', 'trading_date', 'symbol'], dtype=str)
    times = np.load(Path(folder) / 'market' / 'time.npy')
    if len(raw) != row['rows'] or len(raw) != source['rows'] or len(times) != len(raw):
        raise ReviewError(f'{row["contract"]}: market rows differ from the corpus')
    if set(raw.symbol) != {row['contract']}:
        raise ReviewError(f'{row["contract"]}: the market CSV holds another symbol')
    eob = pd.to_datetime(raw.Date).dt.tz_localize(TIMEZONE)
    bob = pd.to_datetime(raw.bob).dt.tz_localize(TIMEZONE)
    if not (eob == pd.to_datetime(pd.Series(times))).all():
        raise ReviewError(f'{row["contract"]}: market times differ from the corpus')
    # bob < eob is checked on the bars a record marks (check_record): the
    # source has a few closing-auction bars with bob = eob (FG001 2019-09-24
    # 11:30) that no label uses.
    if not raw.trading_date.str.match(DAY.pattern).all():
        raise ReviewError(f'{row["contract"]}: a bar without a trading_date')
    return dict(eob=[t.isoformat() for t in eob], bob=[t.isoformat() for t in bob],
                day=raw.trading_date.tolist())


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
    from .ts_v2_positive import contract_direction
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
    waits = timeline[timeline.teach.isin(WAIT_ROWS)]
    records = []
    for s in np.flatnonzero(g['valid']):
        s = int(s)
        direction = {1: 'LONG', -1: 'SHORT'}.get(int(g['sign'][s]))
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
                         'RV': _optional(w.obs_rv), 'Explanation': str(w.obs_explain)},
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


def check_record(r, symbol):
    """One review-v1 record of a teacher file of `symbol`."""
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


def _strict_loads(line):
    def reject(constant):
        raise ReviewError(f'{constant} is not a JSON number')
    return json.loads(line, parse_constant=reject)


def check_file(path, symbol, records, sha256):
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
        check_record(r, symbol)
    return rows


def _inside(review, rel):
    p = Path(rel)
    if p.is_absolute() or '..' in p.parts or not rel or rel != p.as_posix():
        raise ReviewError(f'{rel}: a data path must be relative and inside review/')
    return Path(review) / p


def validate(root):
    """A published corpus review (ROOT/review): its manifest, every
    registered file and record. Returns the totals; any failure raises
    ReviewError."""
    return validate_review(Path(root) / 'review')


def validate_review(review):
    review = Path(review)
    path = review / 'manifest.json'
    if not path.exists():
        raise ReviewError(f'{path} does not exist')
    m = _strict_loads(path.read_text())
    if not isinstance(m, dict) or m.get('schema') != SCHEMA:
        raise ReviewError(f'{path}: schema is not {SCHEMA}')
    if m.get('kind') != 'corpus' or tuple(m) != MANIFEST_KEYS or m['status'] != 'ready':
        raise ReviewError(f'{path}: not a ready corpus manifest of the review-v1 fields')
    symbols = [c['symbol'] for c in m['contracts']]
    if len(set(symbols)) != len(symbols) or not all(
            set(c) == {'symbol', 'price_tick', 'multiplier', 'currency'} and _is_number(c['price_tick'])
            and _is_number(c['multiplier']) for c in m['contracts']):
        raise ReviewError(f'{path}: contracts')
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('corpus', help='export the teacher of a corpus root into ROOT/review')
    c.add_argument('--input', required=True, help='corpus root (one folder per contract)')
    c.add_argument('--label', required=True, help='manifest label, e.g. "V2.1 teacher"')
    c.add_argument('--expect-contracts', type=int, required=True, help='corpora the root must hold')
    v = sub.add_parser('validate', help='check ROOT/review/manifest.json and its files')
    v.add_argument('--input', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'corpus':
            out = export_corpus(args.input, args.label, args.expect_contracts)
        else:
            out = dict(totals=validate(args.input))
    except ReviewError as e:
        print(f'review export failed: {e}', file=sys.stderr)
        sys.exit(1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
