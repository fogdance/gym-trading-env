#!/usr/bin/env python3
"""Summarize existing native labels and reader masks; never create labels."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def summarize(corpus, reader_dir):
    corpus = Path(corpus).resolve(strict=True)
    reader_dir = Path(reader_dir).resolve(strict=True)
    manifest = read(corpus/'corpus_manifest.json')
    reader = read(reader_dir/'WP03_COURSE_READER_COVERAGE.json')
    coverage = read(corpus/'coverage.json')
    source = read(corpus/'source_lock.json')
    if (manifest['fold'] != 'TRAIN' or manifest['training_admitted'] is not False
            or reader['corpus_identity'] != manifest['content_hash']
            or reader['status'] not in ('PASS_READER_ONLY', 'PASS_READER_CPU')
            or sha(reader_dir/'join.sqlite') != reader['index_sha256']):
        raise ValueError('TRAIN corpus/reader/index identity mismatch')
    candidates = Counter()
    unique = defaultdict(set)
    for policy in manifest['behaviors']:
        rows = count = 0
        with gzip.open(corpus/'policies'/policy/'steps.jsonl.gz', 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                rows += 1
                facts = row['teacher']['L1']
                if not facts['immediate_outcome_verified']:
                    continue
                count += 1
                key = (source['contract'], row['cursor'])
                metrics = ['L1_candidates', 'outcome/'+facts['exit_type']]
                if facts['exit_type'] == 'SL30':
                    for window in (5, 15):
                        if facts[f'early10_{window}_valid'] and facts[f'early10_{window}']:
                            metrics.append(f'early10_{window}_then_SL30')
                if facts['exit_type'] == 'TP5':
                    metrics.append('TP_FAST' if facts['held_market_minutes'] <= 5 else 'TP_SLOW')
                    if facts['mae_15_complete'] and facts['mae_15'] >= 10:
                        metrics.append('TP_with_MAE15_at_least_10')
                for metric in metrics:
                    candidates[metric] += 1
                    unique[metric].add(key)
        reference = coverage['behaviors'][policy]['counts']
        if rows != reference['replay_rows'] or count != reference['l1_candidates']:
            raise ValueError('Recomputed L1 counts differ from native coverage')
    masks = {key: None for key in ('event', 'touch', 'adverse')}
    physical_rows = 0
    db = sqlite3.connect((reader_dir/'join.sqlite').as_uri()+'?mode=ro', uri=True)
    try:
        db.execute('PRAGMA query_only=ON')
        for (value,) in db.execute('SELECT labels FROM rows'):
            physical_rows += 1
            labels = json.loads(value)
            for name in masks:
                raw = labels['quality_'+name+'_valid']
                flat = [v for row in raw for v in row] if raw and isinstance(raw[0], list) else raw
                if masks[name] is None:
                    masks[name] = [0]*len(flat)
                if len(flat) != len(masks[name]):
                    raise ValueError('Reader mask shape changed')
                masks[name] = [total+int(v) for total, v in zip(masks[name], flat)]
    finally:
        db.close()
    if not physical_rows:
        raise ValueError('UNSUPPORTED: empty Reader index')
    return dict(status='OBSERVED_COVERAGE_ONLY', corpus_identity=manifest['content_hash'],
        corpus_manifest_sha256=sha(corpus/'corpus_manifest.json'),
        reader_report_sha256=sha(reader_dir/'WP03_COURSE_READER_COVERAGE.json'),
        source_lock=source['content_hash'],
        market_denominators={k:coverage[k] for k in ('unique_market_cursors',
            'unique_advanced_market_minutes', 'unique_contract_official_days',
            'distinct_account_decision_states')},
        candidate_occurrences=dict(candidates),
        unique_contract_cursor_support={k:len(v) for k,v in unique.items()},
        candidate_denominator='account-state counterfactuals; overlapping metrics, not independent trades',
        projected_mask_denominator_physical_rows=physical_rows,
        projected_mask_valid_counts=masks,
        projected_mask_valid_rates={k:[n/physical_rows for n in v] for k,v in masks.items()},
        event_window_order=[5,15,60,120,240,'entry_day_end','next_day_end'],
        actual_optimizer_exposures=reader['actual_optimizer_exposures'],
        training_admitted=False, new_native_g3='NOT_RUN_BY_THIS_REPORTER')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--reader-dir', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = Path(args.out).absolute()
    if output.is_relative_to(Path(args.corpus).resolve()):
        raise ValueError('Coverage output must remain outside immutable Corpus')
    report = summarize(args.corpus, args.reader_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report))
