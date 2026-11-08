"""Bounded, resumable process jobs with sealed blind roles and atomic outputs."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import multiprocessing
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

import pandas as pd
import numpy as np
import yaml

from gym_trading_env.research.fg_history_study import audit_snapshot, digest
from gym_trading_env.research.fg_history_events import generate_events


def run_job(task):
    """One real contract, one pre-blind role, one mode; serialize rows as streamed gzip."""
    from gym_trading_env.research.fg_entry_v1 import write_json
    from gym_trading_env.utils.market_features import build_market_features, FEATURES_MARKET_OBS_OUTPUT

    if task['role'] not in {'train', 'validation'} or task['end'] >= task['blind_start']:
        raise ValueError('Blind labels remain sealed at the worker boundary')
    started = time.monotonic()
    directory = Path(task['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    done = directory / 'complete.json'
    if done.exists():
        record = json.loads(done.read_text())
        if record['signature'] != task['signature']:
            raise ValueError('Resume signature mismatch')
        if all(digest(directory / name) == value for name, value in record['output_hashes'].items()):
            return dict(record, resumed=True)
        raise ValueError(f'Completed output changed: {directory}')
    frames, omitted = [], []
    for item in task['files']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Market changed after audit')
        # Slice BEFORE features, range queries, terminal decisions and labels.
        raw = pd.read_csv(item['path'])
        days = pd.to_datetime(raw.trading_date).dt.strftime('%Y-%m-%d')
        frames.append(raw.loc[(days >= task['start']) & (days <= task['end'])].copy())
        if item.get('truncation_audit'):
            if digest(item['truncation_audit']) != item['truncation_audit_sha256']:
                raise ValueError('Truncation identity audit changed')
            # Do NOT read removed prices, volume, or interest.
            identity = pd.read_csv(item['truncation_audit'], usecols=['bob', 'trading_date', 'symbol'])
            if not identity.empty:
                selected = identity[(identity.trading_date >= task['start']) & (identity.trading_date <= task['end'])]
                omitted.extend(selected.groupby('trading_date').bob.min().tolist())
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        raise ValueError('Empty job after role slicing')
    times = pd.DatetimeIndex(pd.to_datetime(frame.Date))
    if not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError('Repeated/overlapping contract segments require explicit lifecycle review')
    frame.index = times.tz_localize(task['schedule']['timezone'])
    features = build_market_features(frame, is_future=True, real_futures_clock=True,
                                    futures_schedule=task['schedule'])[FEATURES_MARKET_OBS_OUTPUT]
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError('Nonfinite research observation')
    feature_rows = features.to_numpy()
    # File cutoff is not terminal evidence; real main change must exist in this role.
    final = task['files'][-1]
    terminal = bool(final.get('next_main_contract') and final['next_main_contract'] != task['contract']
                    and final.get('main_exit_at')
                    and pd.Timestamp(final['main_exit_at']) == times[-1])
    outputs = {'events.csv.gz': gzip.open(directory / 'events.csv.gz.tmp', 'wt', newline=''),
               'rejected.csv.gz': gzip.open(directory / 'rejected.csv.gz.tmp', 'wt', newline='')}
    writers = {}
    counts = Counter()
    try:
        for number, event in enumerate(generate_events(frame, contract=task['contract'], role=task['role'],
                    mode=task['mode'], costs=task['costs'], schedule=task['schedule'],
                    terminal=terminal, omitted_bobs=omitted, take_profit=task['take_profit'], quantity=task['quantity'])):
            event['source_manifest_sha256'] = task['manifest_sha256']
            event.update(zip(FEATURES_MARKET_OBS_OUTPUT, feature_rows[number % len(frame)]))
            name = 'rejected.csv.gz' if event['completion_status'] == 'no_entry' else 'events.csv.gz'
            if not writers:
                for output_name in outputs:
                    writers[output_name] = csv.DictWriter(outputs[output_name], fieldnames=list(event))
                    writers[output_name].writeheader()
            writers[name].writerow(event)
            counts[event['completion_status']] += 1
            if event['censor_reason']:
                counts['censor:' + event['censor_reason']] += 1
            if event['rejection_reason']:
                counts['reject:' + event['rejection_reason']] += 1
    finally:
        for stream in outputs.values():
            stream.close()
    for name in outputs:
        (directory / (name + '.tmp')).replace(directory / name)
    result = dict(schema=1, signature=task['signature'], contract=task['contract'], role=task['role'],
                  mode=task['mode'], input_rows=len(frame), counts=dict(counts),
                  seconds=round(time.monotonic() - started, 3), resumed=False,
                  output_hashes={name: digest(directory / name) for name in outputs},
                  terminal_evidence=terminal, directory=str(directory))
    write_json(done, result)
    return result


def run_events(config_path, *, workers=None, roles=('train', 'validation'), contracts=None, plan_only=False):
    """Blind event execution has no CLI switch. Only pre-blind tasks are admitted."""
    config = yaml.safe_load(Path(config_path).read_text())
    master = yaml.safe_load(Path(config['master_config']).read_text())
    # Costs are the master's product profile's (envs.entry_product, user 2026-11-08).
    from gym_trading_env.envs.entry_product import master_profile
    master['costs'] = master_profile(master, config['master_config'])['costs']
    if not roles or set(roles) - {'train', 'validation'}:
        raise ValueError('Blind labels remain sealed; only train/validation allowed')
    if config['parallel']['start_method'] != 'spawn' or config['parallel']['native_threads_per_worker'] != 1:
        raise ValueError('Research requires spawn and one native thread per worker')
    workers = config['parallel']['workers'] if workers is None else workers
    if not isinstance(workers, int) or workers < 1 or workers > 8:
        raise ValueError('workers must be between 1 and 8')
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[key] = '1'
    output = Path(config['output']).resolve()
    root = Path(__file__).resolve().parents[3]
    if not output.is_relative_to(root / 'artifacts'):
        raise ValueError('Generated research data must stay in ignored project artifacts')
    if master['market_clock']['minutes_per_day'] != 345 or config['take_profit_points'] != 5 or config['quantity'] != 3:
        raise ValueError('Frozen study clock/target/quantity changed')
    audit = audit_snapshot(config_path)
    for record in audit['contracts_detail']:
        failures = ['invalid_duration', 'duplicate_timestamps', 'bad_ohlc', 'nonfinite_values',
                    'nonpositive_prices', 'negative_volume_oi', 'identity_mismatch',
                    'provider_mismatch', 'outside_345_clock', 'mapping_mismatch']
        if any(record[key] for key in failures) or not record['ascending'] or not record['manifest_rows_match']:
            raise ValueError(f"Input audit failed: {record['contract']}")
    if audit['mapping_days_without_bars']:
        raise ValueError('Whole mapped trading days lack data')
    manifest = json.loads(Path(config['export_manifest']).read_text())
    seen_contracts, previous = set(), None
    for item in manifest['files']:
        if item['contract'] != previous:
            if item['contract'] in seen_contracts:
                raise ValueError('Nonadjacent main lifecycles cannot be merged')
            seen_contracts.add(item['contract'])
        previous = item['contract']
    if plan_only:
        return audit
    split = master['data']['split']
    a, b = split['train_end_inclusive'], split['validation_end_inclusive']
    if not a < b < config['blind_start']:
        raise ValueError('Pre-blind role boundaries overlap sealed period')
    boundaries = {'train': (master['data']['start'], a),
                  'validation': ((pd.Timestamp(a) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), b)}
    signature_payload = dict(config=digest(config_path), master=digest(config['master_config']),
                             manifest=digest(config['export_manifest']),
                             engine=digest(Path(__file__).with_name('fg_history_events.py')),
                             runner=digest(__file__),
                             features=digest(root / 'src/gym_trading_env/utils/market_features.py'))
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    run_directory = output / 'events' / signature[:16]
    run_directory.mkdir(parents=True, exist_ok=True)
    groups = {}
    for item, coverage in zip(manifest['files'], audit['contracts_detail']):
        if contracts and item['contract'] not in contracts:
            continue
        for role in roles:
            start, end = boundaries[role]
            if coverage['last_trading_date'] >= start and coverage['first_trading_date'] <= end:
                groups.setdefault((item['contract'], role), []).append(item)
    if not groups:
        raise ValueError('No pre-blind contracts selected')
    tasks = []
    for (contract, role), files in groups.items():
        for mode in config['execution_modes']:
            start, end = boundaries[role]
            tasks.append(dict(contract=contract, role=role, mode=mode, files=files, start=start, end=end,
                              blind_start=config['blind_start'],
                              directory=str(run_directory / f'{role}_{contract.replace(".", "_")}_{mode}'),
                              signature=signature, manifest_sha256=signature_payload['manifest'],
                              schedule=master['market_clock'], costs=master['costs'],
                              take_profit=config['take_profit_points'], quantity=config['quantity']))
    return execute_tasks(tasks, workers, run_directory, signature, signature_payload,
                         selected_roles=list(roles), selected_contracts=contracts)


def execute_tasks(tasks, workers, run_directory, signature, signature_payload, **metadata):
    """Shared locked, atomic, resumable execution for descriptive and forward jobs."""
    import fcntl
    from gym_trading_env.research.fg_entry_v1 import write_json

    with (run_directory / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        print(f'[FG events] jobs={len(tasks)} workers={workers} run={run_directory}', flush=True)
        results = []
        write_json(run_directory / 'plan.json', dict(signature=signature, provenance=signature_payload,
                                                    blind_sealed=True, tasks=tasks, workers=workers))
        if workers == 1:
            for task in tasks:
                result = run_job(task)
                results.append(result)
                print(f"[FG events] {len(results)}/{len(tasks)} {result['contract']} {result['role']} {result['mode']} resumed={result['resumed']}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = [pool.submit(run_job, task) for task in tasks]
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    print(f"[FG events] {len(results)}/{len(tasks)} {result['contract']} {result['role']} {result['mode']} resumed={result['resumed']}", flush=True)
        results.sort(key=lambda r: (r['role'], r['contract'], r['mode']))
        summary = dict(schema=1, signature=signature, blind_sealed=True,
                       **metadata,
                       seconds=round(time.monotonic() - started, 3), jobs=results)
        write_json(run_directory / 'summary.json', summary)
        print(f'[FG events] complete seconds={summary["seconds"]}', flush=True)
        return summary
