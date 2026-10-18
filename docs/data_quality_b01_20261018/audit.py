"""Read-only B01 publication audit. Reuses the existing coverage reporter.

No producer, model, optimizer, G3 replay or additional batch is launched.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import runpy
import sqlite3

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = Path('/data/s3_corpus/early_train_20261018_cpu20_v2')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def timing(path):
    values = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.strip().partition(': ')
        if separator:
            values[key] = value
    wall = values['Elapsed (wall clock) time (h:mm:ss or m:ss)']
    seconds = 0.
    for part in wall.split(':'):
        seconds = seconds*60+float(part)
    return dict(wall_seconds=seconds,
        user_cpu_seconds=float(values['User time (seconds)']),
        system_cpu_seconds=float(values['System time (seconds)']),
        average_cpu_percent=float(values['Percent of CPU this job got'].rstrip('%')),
        max_rss_KiB=int(values['Maximum resident set size (kbytes)']),
        exit_status=int(values['Exit status']))


def audit():
    corpus = PRODUCTION/'batches/B01_FG809'
    receipts = PRODUCTION/'receipts/B01_FG809'
    batch = read(receipts/'BATCH_MANIFEST.json')
    body = {k:v for k,v in batch.items() if k != 'content_hash'}
    assert hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest() == batch['content_hash']
    checked = 0
    for name, expected in batch['files'].items():
        kind, relative = name.split('/',1)
        assert kind in ('corpus','receipts') and not Path(relative).is_absolute() and '..' not in Path(relative).parts
        assert sha((corpus if kind == 'corpus' else receipts)/relative) == expected, name
        checked += 1
    manifest = read(corpus/'corpus_manifest.json')
    native = read(receipts/'native/native_verification.json')
    reader = read(receipts/'reader/WP03_COURSE_READER_COVERAGE.json')
    coverage = read(corpus/'coverage.json')
    assert manifest['fold'] == 'TRAIN' and manifest['execution_mode'] == 'data'
    assert batch['corpus_identity'] == manifest['content_hash'] == native['result']['corpus_identity'] == reader['corpus_identity']
    assert native['exit_code'] == 0 and native['result']['G3'] == 'GO'
    helper = ROOT/'docs/train_corpus_20261018/audit_coverage.py'
    quality = runpy.run_path(str(helper))['summarize'](corpus,receipts/'reader')
    assert quality == read(receipts/'quality_coverage.json')
    extra = Counter(); unique = defaultdict(set); reasons = Counter()
    numbers = defaultdict(list); raw_rows = 0
    for policy in manifest['behaviors']:
        with gzip.open(corpus/'policies'/policy/'steps.jsonl.gz','rt') as stream:
            for line in stream:
                row = json.loads(line); raw_rows += 1
                facts, strict = row['teacher']['L1'], row['teacher']['L2']
                if not strict['teacher_valid']:
                    reasons['L2/'+str(strict.get('reason'))] += 1
                if not facts['immediate_outcome_verified']:
                    reasons['L1/'+str(facts.get('invalid_reason'))] += 1
                    continue
                if not facts['score_valid']:
                    reasons['score/'+str(facts.get('score_reason'))] += 1
                metrics = []
                if facts['exit_type'] == 'TP5':
                    metrics.append('TP_SAME_DAY' if facts['crossed_trading_days'] == 0 else 'TP_CROSS_DAY')
                    numbers['tp_market_minutes'].append(facts['held_market_minutes'])
                for horizon in (5,15):
                    if facts[f'early10_{horizon}_valid'] and facts[f'early10_{horizon}']:
                        metrics.append(f'early10_{horizon}')
                for metric in metrics:
                    extra[metric] += 1; unique[metric].add(row['cursor'])
                if facts['score_valid']:
                    numbers['score'].append(facts['score'])
                if facts['mae_15_complete']:
                    numbers['mae_15'].append(facts['mae_15'])
                if facts['path_complete']:
                    numbers['mae_full'].append(facts['mae_full'])
                if facts['underwater_full_valid']:
                    numbers['underwater_minutes'].append(facts['underwater_minutes'])
    projected = Counter(); invalid_flags = Counter(); physical = 0
    db = sqlite3.connect((receipts/'reader/join.sqlite').as_uri()+'?mode=ro&immutable=1',uri=True)
    try:
        db.execute('PRAGMA query_only=ON')
        for labels, audit_json, first, last in db.execute('SELECT labels,audit,first,last FROM rows'):
            labels, row = json.loads(labels), json.loads(audit_json); physical += 1
            for field in ('quality_outcome_valid','quality_score_valid','teacher_valid','quality_q_v2_valid','quality_soft_valid'):
                projected[field] += int(labels[field])
            if labels['teacher_valid']:
                projected['strict_ADD' if labels['teacher_action'] else 'strict_WAIT'] += 1
            else:
                for flag, value in [('first',first),('last',last),('nonexecuted',not row['decision_executed']),
                                    ('ADD_illegal',not row['legal_mask'][1])]:
                    invalid_flags[flag] += int(bool(value))
    finally:
        db.close()
    assert physical == raw_rows == quality['projected_mask_denominator_physical_rows']
    assert projected['strict_ADD']+projected['strict_WAIT'] == projected['teacher_valid']
    assert sha(receipts/'reader/join.sqlite') == batch['index_sha256']
    distributions = {k:dict(n=len(v),p50=float(np.quantile(v,.5)),p90=float(np.quantile(v,.9)),
        minimum=min(v),maximum=max(v)) for k,v in numbers.items()}
    costs = {name:timing(receipts/(name+'.time.txt')) for name in ('validate','generate','native_g3','reader','coverage')}
    assert all(v['exit_status'] == 0 for v in costs.values())
    return dict(protocol='FG_S3_DATA_QUALITY_B01_AUDIT_V1',
        status='B01_PUBLISHED_AND_AUDITED',batch_manifest=batch,hashed_files_checked=checked,
        batch_manifest_sha256=sha(receipts/'BATCH_MANIFEST.json'),existing_quality_report=quality,
        extra_candidate_occurrences=dict(extra),extra_unique_cursor_support={k:len(v) for k,v in unique.items()},
        distributions=distributions,distribution_denominator='all observed L1 account-state candidates; TP duration uses TP5 only',
        raw_invalid_reasons=dict(reasons),projected_valid_counts=dict(projected),physical_rows=physical,
        projected_invalid_flags=dict(invalid_flags),invalid_flags_overlap=True,
        actual_exits={key:sum(v['counts'].get('executed_exit/'+key,0) for v in coverage['behaviors'].values())
                      for key in ('TAKE_PROFIT','STOP_LOSS','MAIN_ROLL')},
        behavior_counts={k:v['counts'] for k,v in coverage['behaviors'].items()},
        reader_policy_support=reader['policy_support'],reader_stage_support={k:v.get('status','SUPPORTED') for k,v in reader['stage_views'].items()},
        costs=costs,cost_wall_seconds=sum(v['wall_seconds'] for v in costs.values()),
        published_bytes=reader['corpus_published_bytes'],reader_index_bytes=reader['index_bytes'],
        native_g3='GO_FROM_EXISTING_PRODUCTION_RECEIPT',new_native_g3='NOT_RUN',
        new_production='NOT_RUN',optimizer_updates=0,
        next_batch=dict(batch_id='B02_FG009',official_days=['2020-08-07','2020-08-10','2020-08-11'],
            status='SELECTED_NOT_RUN',reason='prelocked cross-year TRAIN batch; no future outcome selection'),
        raw_receipts={str(p):sha(p) for p in [receipts/'BATCH_MANIFEST.json',receipts/'quality_coverage.json',
            receipts/'native/native_verification.json',receipts/'reader/WP03_COURSE_READER_COVERAGE.json',corpus/'coverage.json']},
        source_task=dict(path='/home/v/Downloads/FG_S3_E_Quality_Parallel_Workstreams_20261018_V1.md',
                         scope='First P0 DATA-QUALITY only'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    args = parser.parse_args()
    output = Path(args.out).absolute()
    if output.is_relative_to(PRODUCTION) or output.exists():
        raise ValueError('Use a new report outside the immutable production tree')
    result = audit()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps({k:result[k] for k in ['status','hashed_files_checked','physical_rows','extra_candidate_occurrences',
        'extra_unique_cursor_support','projected_valid_counts','raw_invalid_reasons','cost_wall_seconds']}))
