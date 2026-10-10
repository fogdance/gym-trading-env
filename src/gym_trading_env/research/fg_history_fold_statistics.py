"""Forward-window descriptive metrics; frozen horizons, contract clusters and censor bounds."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gym_trading_env.research.fg_entry_v1 import write_json
from gym_trading_env.research.fg_history_statistics import HORIZONS, horizon_flags, contract_intervals, block_intervals
from gym_trading_env.research.fg_history_study import digest

FIELDS = ['n', 'success', 'terminal', 'censored', 'reward_sum', 'reward_known',
          'censored_reward_upper', 'wait_sum', 'mae_lower_sum', 'mae_upper_sum'] + [
              f'{h}_{kind}' for h in HORIZONS for kind in ['hit', 'unknown']]


def summarize(values, *, replicates, rng):
    """Each row is one whole contract contribution, not one independent minute."""
    values = np.asarray(values, dtype=float)
    totals = dict(zip(FIELDS, values.sum(axis=0)))
    n = totals['n']
    if n <= 0:
        raise ValueError('Empty group cannot be summarized')
    result = dict(totals, contracts=len(values),
        inference_status='insufficient_contract_clusters' if len(values) < 6 else 'descriptive_cluster_intervals',
        mean_success_wait=totals['wait_sum'] / totals['success'] if totals['success'] else None,
        mean_known_reward=totals['reward_sum'] / totals['reward_known'] if totals['reward_known'] else None,
        mean_reward_lower=totals['reward_sum'] / n,
        mean_reward_upper=(totals['reward_sum'] + totals['censored_reward_upper']) / n,
        censored_fraction=totals['censored'] / n,
        mean_mae_lower=totals['mae_lower_sum'] / n, mean_mae_upper=totals['mae_upper_sum'] / n)
    for h in HORIZONS:
        result[h + '_lower'] = totals[h + '_hit'] / n
        result[h + '_upper'] = (totals[h + '_hit'] + totals[h + '_unknown']) / n
        selected = [FIELDS.index(k) for k in ['n', h + '_hit', h + '_unknown']]
        result[h + '_cluster_ci_low'], result[h + '_cluster_ci_high'] = contract_intervals(
            values[:, selected], replicates, rng)
    return result


def aggregate_job(task):
    directory = Path(task['directory'])
    done = json.loads((directory / 'complete.json').read_text())
    if done['signature'] != task['signature']:
        raise ValueError('Job signature changed')
    for name, expected in done['output_hashes'].items():
        if digest(directory / name) != expected:
            raise ValueError('Event output changed')
    identities = []
    for item in task['files']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Trading-date source changed')
        frame = pd.read_csv(item['path'], usecols=['Date', 'bob', 'trading_date'])
        identities.append(frame.loc[frame.trading_date.between(task['start'], task['end'])])
    identity = pd.concat(identities)
    days = list(dict.fromkeys(identity.trading_date))
    mapping = dict(zip(pd.to_datetime(identity.bob), identity.trading_date))
    mapping.update(zip(pd.to_datetime(identity.Date), identity.trading_date))
    totals = defaultdict(lambda: np.zeros(len(FIELDS)))
    trade_days = defaultdict(set)
    daily = defaultdict(lambda: np.zeros(3))
    for chunk in pd.read_csv(directory / 'events.csv.gz', chunksize=20000, low_memory=False):
        if not (chunk.role.eq(task['role']).all() and chunk.contract.eq(task['contract']).all()
                and chunk.trading_date.between(task['start'], task['end']).all()
                and (chunk.trading_date < task['blind_start']).all()):
            raise ValueError('Event role, identity or date boundary mismatch')
        success = chunk.completion_status.eq('success')
        terminal = chunk.completion_status.eq('terminal_failure')
        censored = chunk.completion_status.eq('right_censored')
        if not (success | terminal | censored).all() or not chunk.loc[censored, 'reward'].isna().all():
            raise ValueError('Invalid event completion or censor reward')
        flags = horizon_flags(chunk, days, mapping)
        metrics = pd.DataFrame(dict(n=1, success=success.astype(int), terminal=terminal.astype(int),
            censored=censored.astype(int), reward_sum=chunk.reward.fillna(0),
            reward_known=chunk.reward.notna().astype(int),
            censored_reward_upper=(60 / (60 + chunk.followup_minutes)).where(censored, 0),
            wait_sum=chunk.duration_minutes.where(success, 0),
            mae_lower_sum=chunk.MAE_lower, mae_upper_sum=chunk.MAE_upper), index=chunk.index)
        for h in HORIZONS:
            metrics[h + '_hit'], metrics[h + '_unknown'] = (a.astype(int) for a in flags[h])
        for direction, part in metrics.assign(direction=chunk.direction).groupby('direction'):
            totals[direction] += part[FIELDS].sum().to_numpy(float)
            for day, sums in part.assign(day=chunk.loc[part.index, 'trading_date']).groupby('day')[['n', '60m_hit', '60m_unknown']].sum().iterrows():
                daily[(direction, day)] += sums.to_numpy(float)
            trade_days[direction].update(chunk.loc[chunk.direction.eq(direction), 'trading_date'])
    n = sum(v[0] for v in totals.values())
    if n != sum(done['counts'].get(k, 0) for k in ['success', 'terminal_failure', 'right_censored']):
        raise ValueError('Event count mismatch')
    rejected = pd.read_csv(directory / 'rejected.csv.gz', usecols=['direction']).direction.value_counts().to_dict()
    return dict(id=directory.name, rejected=rejected, daily=[dict(direction=k[0], day=k[1], values=v.tolist()) for k, v in sorted(daily.items())], values={k: v.tolist() for k, v in totals.items()},
                days={k: sorted(v) for k, v in trade_days.items()}, counts=done['counts'])


def describe_folds(run_path, config_path, *, workers=None):
    directory = Path(run_path)
    plan = json.loads((directory / 'plan.json').read_text())
    datasets = json.loads((directory / 'datasets.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    config = yaml.safe_load(Path(config_path).read_text())
    if not all(x['blind_sealed'] for x in [plan, datasets, summary]):
        raise ValueError('Blind data remain sealed')
    if not plan['signature'] == datasets['signature'] == summary['signature']:
        raise ValueError('Forward manifest signature mismatch')
    for key, path in [('config', config_path), ('master', config['master_config']), ('manifest', config['export_manifest'])]:
        if plan['provenance'][key] != digest(path):
            raise ValueError('Forward configuration or export changed')
    workers = config['parallel']['workers'] if workers is None else workers
    if not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError('workers must be between 1 and 8')
    tasks = {Path(t['directory']).name: t for t in plan['tasks']}
    if len(tasks) != len(plan['tasks']) or len(tasks) != len(summary['jobs']):
        raise ValueError('Incomplete or duplicated jobs')
    if any(t['role'] not in {'train', 'validation'} or t['end'] >= config['blind_start'] for t in tasks.values()):
        raise ValueError('Blind labels remain sealed')
    source_hash = digest(__file__)
    shared_hash = digest(Path(__file__).with_name('fg_history_statistics.py'))
    signature = hashlib.sha256((plan['signature'] + source_hash + shared_hash).encode()).hexdigest()[:16]
    output = directory / 'statistics' / signature
    output.mkdir(parents=True, exist_ok=True)
    contributions = {}
    if workers == 1:
        for task in tasks.values():
            result = aggregate_job(task)
            contributions[result['id']] = result
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(aggregate_job, t) for t in tasks.values()]
            for future in as_completed(futures):
                result = future.result()
                contributions[result['id']] = result
                print(f'[FG fold statistics] {len(contributions)}/{len(tasks)}', flush=True)
    rows, block_rows, block_cache = [], [], {}
    for dataset in datasets['datasets']:
        for role, keys in dataset['jobs'].items():
            if len(keys) != len(set(keys)) or role not in {'train', 'validation'}:
                raise ValueError('Duplicate dataset reference or forbidden role')
            block_key = hashlib.sha256('|'.join(sorted(keys)).encode()).hexdigest()[:16]
            if block_key not in block_cache:
                daily = {}
                for key in keys:
                    task = tasks[key]
                    for item in contributions[key]['daily']:
                        daily[(task['mode'], role, item['direction'], task['contract'], item['day'])] = np.array(item['values'])
                block_dir = output / 'day_blocks' / block_key
                block_dir.mkdir(parents=True, exist_ok=True)
                block_intervals(daily, config, block_dir)
                block_cache[block_key] = pd.read_csv(block_dir / 'day_block_sensitivity.csv')
            block_rows.append(block_cache[block_key].assign(cutoff=dataset['cutoff'], window=str(dataset['window'])))
            for mode in config['execution_modes']:
                selected = [key for key in keys if tasks[key]['mode'] == mode]
                if any(tasks[key]['role'] != role for key in selected):
                    raise ValueError('Dataset role mismatch')
                for direction in ['LONG', 'SHORT']:
                    available = [key for key in selected if direction in contributions[key]['values']]
                    if not available:
                        continue
                    contracts = [tasks[key]['contract'] for key in available]
                    if len(contracts) != len(set(contracts)):
                        raise ValueError('Contract cluster duplicated within dataset')
                    # Same validation references must give identical bootstrap intervals.
                    local_seed = int(hashlib.sha256('|'.join(sorted(available)).encode()).hexdigest()[:8], 16)
                    rng = np.random.default_rng(config['bootstrap_seed'] + local_seed)
                    result = summarize([contributions[k]['values'][direction] for k in available],
                        replicates=config['bootstrap_replicates'], rng=rng)
                    days = set().union(*(set(contributions[k]['days'][direction]) for k in available))
                    result.update(cutoff=dataset['cutoff'], window=str(dataset['window']), role=role,
                                  mode=mode, direction=direction, trading_days=len(days),
                                  rejected=sum(contributions[k]['rejected'].get(direction, 0) for k in selected))
                    rows.append(result)
    table = pd.DataFrame(rows)
    table.to_csv(output / 'window_statistics.csv', index=False)
    pd.concat(block_rows, ignore_index=True).to_csv(output / 'day_block_sensitivity.csv', index=False)
    # Validation is repeated by reference, never four independent replications.
    validation = table[table.role.eq('validation')]
    metrics = [c for c in table if c not in {'window', 'cutoff', 'role', 'mode', 'direction'}]
    if (validation.groupby(['cutoff', 'mode', 'direction'])[metrics].nunique(dropna=False) > 1).any().any():
        raise ValueError('Identical validation slices yielded different statistics')
    validation.drop_duplicates(['cutoff', 'mode', 'direction']).to_csv(output / 'validation_statistics.csv', index=False)
    write_json(output / 'report_manifest.json', dict(schema=1, event_signature=plan['signature'],
        statistics_sha256=source_hash, shared_statistics_sha256=shared_hash, blind_sealed=True,
        datasets=len(datasets['datasets']), unique_jobs=len(tasks), summary_rows=len(table),
        validation_unique_rows=len(validation.drop_duplicates(['cutoff', 'mode', 'direction'])),
        seed=config['bootstrap_seed'], replicates=config['bootstrap_replicates'],
        limitations=['Descriptive overlapping probes are not executable portfolio returns.',
                    'Shared validation cannot rank training windows without model predictions.',
                    'Contract clusters below six are insufficient for drift significance.',
                    'Reward bounds do not impute censored labels; waiting means condition on success.']))
    print(f'[FG fold statistics] output={output}', flush=True)
    return output
