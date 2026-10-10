"""Forward research datasets: slice before labels, share only identical price queries."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from gym_trading_env.research.fg_history_parallel import execute_tasks
from gym_trading_env.research.fg_history_study import digest
from gym_trading_env.research.fg_history_windows import window_coverage
from gym_trading_env.research.fg_entry_v1 import write_json


def build_tasks(coverage, files, config, master, directory, signature, manifest_sha):
    """Return dataset references and deduplicated, role-specific contract slices."""
    jobs, datasets = {}, []
    for row in coverage.to_dict('records'):
        if not row['planned_start'] <= row['cutoff'] < row['validation_start'] <= row['validation_end'] < config['blind_start']:
            raise ValueError('Forward fold overlaps sealed blind period or reverses time')
        references = {}
        for role, start, end in [('train', row['planned_start'], row['cutoff']),
                                 ('validation', row['validation_start'], row['validation_end'])]:
            if start > end or end >= config['blind_start']:
                raise ValueError('Forward fold overlaps sealed blind period or reverses time')
            references[role] = []
            for contract, entries in files.items():
                selected = [item for item, first, last in entries if first <= end and last >= start]
                if not selected:
                    continue
                # Canonicalize only to available bounds: identical inputs/features/labels.
                actual_start = max(start, min(first for _, first, last in entries if first <= end and last >= start))
                actual_end = min(end, max(last for _, first, last in entries if first <= end and last >= start))
                for mode in config['execution_modes']:
                    payload = dict(contract=contract, role=role, mode=mode, files=selected,
                        start=actual_start, end=actual_end, blind_start=config['blind_start'],
                        manifest_sha256=manifest_sha, schedule=master['market_clock'], costs=master['costs'],
                        take_profit=config['take_profit_points'], quantity=config['quantity'])
                    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
                    jobs.setdefault(key, dict(payload, directory=str(directory / 'jobs' / key), signature=signature))
                    references[role].append(key)
            if not references[role]:
                raise ValueError(f'Empty {role} dataset for cutoff {row["cutoff"]}')
        datasets.append(dict(row, jobs=references))
    return list(jobs.values()), datasets


def run_folds(config_path, *, workers=None, plan_only=False):
    # Reuse the existing admission audit and guards without executing descriptive jobs.
    from gym_trading_env.research.fg_history_parallel import run_events
    config = yaml.safe_load(Path(config_path).read_text())
    master = yaml.safe_load(Path(config['master_config']).read_text())
    workers = config['parallel']['workers'] if workers is None else workers
    audit = run_events(config_path, workers=workers, plan_only=True)
    manifest = json.loads(Path(config['export_manifest']).read_text())
    files = {}
    for item, record in zip(manifest['files'], audit['contracts_detail']):
        files.setdefault(item['contract'], []).append((item, record['first_trading_date'], record['last_trading_date']))
    coverage = pd.read_csv(window_coverage(config_path), dtype={'window': str})
    provenance = dict(config=digest(config_path), master=digest(config['master_config']),
        manifest=digest(config['export_manifest']), folds=digest(__file__),
        runner=digest(Path(__file__).with_name('fg_history_parallel.py')),
        windows=digest(Path(__file__).with_name('fg_history_windows.py')),
        engine=digest(Path(__file__).with_name('fg_history_events.py')),
        features=digest(Path(__file__).parents[1] / 'utils/market_features.py'))
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    directory = Path(config['output']).resolve() / 'forward_folds' / signature[:16]
    directory.mkdir(parents=True, exist_ok=True)
    tasks, datasets = build_tasks(coverage, files, config, master, directory, signature, provenance['manifest'])
    write_json(directory / 'datasets.json', dict(schema=1, signature=signature, provenance=provenance,
        blind_sealed=True, datasets=datasets, unique_jobs=len(tasks),
        notes=['Research event labels only; no model training.',
               'Features restart at each contract slice; 60-bar warmup is not supplied.',
               'Independent LONG/SHORT probes overlap; rewards cannot be summed as portfolio PnL.']))
    print(f'[FG folds] datasets={len(datasets)} unique_jobs={len(tasks)} directory={directory}', flush=True)
    if plan_only:
        return dict(directory=str(directory), datasets=len(datasets), unique_jobs=len(tasks))
    summary = execute_tasks(tasks, workers, directory, signature, provenance, datasets=len(datasets))
    summary['directory'] = str(directory)
    return summary
