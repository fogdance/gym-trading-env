"""Descriptive clustered statistics; censored outcomes are never filled as failures."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gym_trading_env.research.fg_history_study import digest


HORIZONS = ['5m', '15m', '60m', '1day', '5day']
METRICS = ['n', 'success', 'terminal', 'censored', 'reward_sum', 'reward_known',
           'wait_sum', 'mae_lower_sum', 'mae_upper_sum'] + [f'{h}_{x}' for h in HORIZONS for x in ['hit', 'unknown']]
DURATION_BINS = [0, 1, 2, 5, 15, 60, 240, 1440, 7200, 43200, np.inf]
MAE_BINS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, np.inf]


def horizon_flags(frame, trading_days, timestamp_days):
    """Known successes/failures and unresolved outcomes at each fixed horizon."""
    success = frame.completion_status.eq('success').to_numpy()
    censored = frame.completion_status.eq('right_censored').to_numpy()
    elapsed = frame.duration_minutes.to_numpy(float)
    followup = frame.followup_minutes.to_numpy(float)
    result = {}
    for label, minutes in [('5m', 5), ('15m', 15), ('60m', 60)]:
        result[label] = (success & (elapsed <= minutes), censored & (followup < minutes))
    touch = pd.to_datetime(frame.first_touch_timestamp)
    observed = pd.to_datetime(frame.observation_end)
    close = pd.to_datetime(frame.horizon_1day_end)
    result['1day'] = (success & (touch <= close).to_numpy(), censored & (observed < close).to_numpy())
    ordinal = {day: i for i, day in enumerate(trading_days)}
    entry_day = frame.trading_date.astype(str).map(ordinal).to_numpy(float)
    # Timestamp->DB trading_date includes night gaps; wall date is not a trading date.
    observed_day = observed.map(timestamp_days).map(ordinal).to_numpy(float)
    touch_day = touch.map(timestamp_days).map(ordinal).to_numpy(float)
    if not np.isfinite(entry_day).all() or not np.isfinite(observed_day).all():
        raise ValueError('Missing trading-date metadata for an event')
    if not np.isfinite(touch_day[success]).all():
        raise ValueError('Missing trading-date metadata for a touch')
    observed_close = pd.to_datetime(observed.map(timestamp_days)) + pd.Timedelta(hours=15)
    day_close_passed = (observed >= observed_close).to_numpy()
    fifth_observed = (observed_day > entry_day + 4) | ((observed_day == entry_day + 4) & day_close_passed)
    result['5day'] = (success & (touch_day <= entry_day + 4), censored & ~fifth_observed)
    return result


def competing_risk_curve(counts):
    """Discrete Aalen-Johansen: TP, terminal failure, and right censoring.

    Ties use the risk set immediately before that timestamp. Estimates assume
    noninformative censoring; the data's policy censoring can violate that.
    """
    risk = sum(sum(row) for row in counts.values())
    survival, cumulative = 1.0, 0.0
    curve = []
    for minute, (touch, terminal, censored) in sorted(counts.items()):
        if risk <= 0:
            break
        cumulative += survival * touch / risk
        survival *= 1 - (touch + terminal) / risk
        curve.append((minute, cumulative))
        risk -= touch + terminal + censored
    return curve


def contract_intervals(values, replicates, rng):
    """Resample entire contract contributions, never individual minute rows."""
    n = len(values)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=replicates)
    totals = weights @ values
    denominator = totals[:, 0]
    good = denominator > 0
    lower = totals[good, 1] / denominator[good]
    upper = (totals[good, 1] + totals[good, 2]) / denominator[good]
    return float(np.quantile(lower, .025)), float(np.quantile(upper, .975))


def describe_events(run_path, config_path):
    from gym_trading_env.research.fg_entry_v1 import write_json
    from gym_trading_env.utils.market_features import FEATURES_MARKET_OBS_OUTPUT

    statistics_hash = digest(__file__)
    directory = Path(run_path).resolve()
    summary = json.loads((directory / 'summary.json').read_text())
    plan = json.loads((directory / 'plan.json').read_text())
    config = yaml.safe_load(Path(config_path).read_text())
    if not summary['blind_sealed'] or not plan['blind_sealed']:
        raise ValueError('Blind data cannot enter development statistics')
    if plan['provenance']['config'] != digest(config_path):
        raise ValueError('Study configuration changed after event generation')
    if plan['provenance']['master'] != digest(config['master_config']):
        raise ValueError('Master configuration changed after event generation')
    tasks = {(t['contract'], t['role'], t['mode']): t for t in plan['tasks']}
    if len(summary['jobs']) != len(tasks):
        raise ValueError('Incomplete event run')
    jobs = summary['jobs']
    for job in jobs:
        if job['role'] not in {'train', 'validation'}:
            raise ValueError('Blind labels remain sealed')
        for name, value in job['output_hashes'].items():
            if digest(Path(job['directory']) / name) != value:
                raise ValueError('Event output changed')
    reference_jobs = sorted([j for j in jobs if j['mode'] == 'touch_spread' and j['role'] == 'train'],
                            key=lambda j: tasks[(j['contract'], j['role'], j['mode'])]['files'][0]['first_timestamp'])[:3]
    if len(reference_jobs) != 3:
        raise ValueError('Need the three earliest development contracts for frozen bins')
    reference_contracts = {j['contract'] for j in reference_jobs}
    reference_features = []
    for job in reference_jobs:
        frame = pd.read_csv(Path(job['directory']) / 'events.csv.gz', usecols=['direction'] + FEATURES_MARKET_OBS_OUTPUT)
        reference_features.append(frame.loc[frame.direction.eq('LONG'), FEATURES_MARKET_OBS_OUTPUT])
    reference = pd.concat(reference_features, ignore_index=True)
    vol_edges = np.unique(np.quantile(reference.obs_volatility_t, [.25, .75]))
    feature_edges = {field: np.r_[-np.inf, np.unique(np.quantile(reference[field], [.1, .25, .5, .75, .9])), np.inf]
                     for field in FEATURES_MARKET_OBS_OUTPUT}
    aggregate = defaultdict(lambda: np.zeros(len(METRICS)))
    durations, maes, rewards, feature_hist = CounterTable(), CounterTable(), CounterTable(), CounterTable()
    feature_sums = defaultdict(lambda: np.zeros(3))
    survival = defaultdict(lambda: defaultdict(lambda: np.zeros(3, dtype=np.int64)))
    daily = defaultdict(lambda: np.zeros(3))
    unique_days = defaultdict(set)
    checked_inputs = set()
    for job_number, job in enumerate(jobs, 1):
        task = tasks[(job['contract'], job['role'], job['mode'])]
        for item in task['files']:
            if item['path'] not in checked_inputs:
                if digest(item['path']) != item['sha256']:
                    raise ValueError('Trading-date source changed after event generation')
                checked_inputs.add(item['path'])
        identity = pd.concat([pd.read_csv(item['path'], usecols=['Date', 'bob', 'trading_date']) for item in task['files']])
        identity = identity[(identity.trading_date >= task['start']) & (identity.trading_date <= task['end'])]
        trading_days = list(dict.fromkeys(identity.trading_date))
        timestamp_days = dict(zip(pd.to_datetime(identity.bob), identity.trading_date))
        timestamp_days.update(zip(pd.to_datetime(identity.Date), identity.trading_date))
        observed_rows = 0
        for chunk in pd.read_csv(Path(job['directory']) / 'events.csv.gz', chunksize=20000, low_memory=False):
            observed_rows += len(chunk)
            if (chunk.trading_date >= config['blind_start']).any():
                raise ValueError('Blind event reached development summary')
            flags = horizon_flags(chunk, trading_days, timestamp_days)
            success = chunk.completion_status.eq('success')
            terminal = chunk.completion_status.eq('terminal_failure')
            censored = chunk.completion_status.eq('right_censored')
            year = chunk.trading_date.str[:4]
            quarter = pd.to_datetime(chunk.trading_date).dt.to_period('Q').astype(str)
            volatility = pd.Series(np.searchsorted(vol_edges, chunk.obs_volatility_t.to_numpy(), side='right'), index=chunk.index).astype(str)
            drift = pd.Series(np.sign(chunk.obs_open_drift_t).astype(int), index=chunk.index).astype(str)
            state = year + '|' + job['contract'][-2:] + '|' + chunk.entry_session + '|' + volatility + '|' + drift
            metrics = pd.DataFrame({'n': 1, 'success': success.astype(int), 'terminal': terminal.astype(int),
                                    'censored': censored.astype(int), 'reward_sum': chunk.reward.fillna(0),
                                    'reward_known': chunk.reward.notna().astype(int),
                                    'wait_sum': chunk.duration_minutes.where(success, 0),
                                    'mae_lower_sum': chunk.MAE_lower, 'mae_upper_sum': chunk.MAE_upper}, index=chunk.index)
            for horizon in HORIZONS:
                metrics[horizon + '_hit'], metrics[horizon + '_unknown'] = (v.astype(int) for v in flags[horizon])
            for dimension, labels in [('overall', pd.Series('all', index=chunk.index)), ('year', year),
                                      ('quarter', quarter), ('delivery_month', pd.Series(job['contract'][-2:], index=chunk.index)),
                                      ('session', chunk.entry_session), ('volatility', volatility), ('state', state)]:
                grouped = metrics.assign(direction=chunk.direction, group=labels).groupby(['direction', 'group'])[METRICS].sum()
                for (direction, label), values in grouped.iterrows():
                    key = (job['mode'], job['role'], direction, dimension, label, job['contract'])
                    aggregate[key] += values.to_numpy(float)
                for direction in ['LONG', 'SHORT']:
                    for label in labels[chunk.direction.eq(direction)].unique():
                        key = (job['mode'], job['role'], direction, dimension, label)
                        unique_days[key].update(chunk.loc[chunk.direction.eq(direction) & labels.eq(label), 'trading_date'])
            for direction, part in metrics.assign(direction=chunk.direction, day=chunk.trading_date).groupby('direction'):
                for day, values in part.groupby('day')[['n', '60m_hit', '60m_unknown']].sum().iterrows():
                    daily[(job['mode'], job['role'], direction, job['contract'], day)] += values.to_numpy(float)
            risk_minutes = chunk.duration_minutes.where(success, chunk.followup_minutes)
            for dimension, labels in [('overall', pd.Series('all', index=chunk.index)), ('year', year)]:
                risk_rows = pd.DataFrame({'direction': chunk.direction, 'group': labels, 'minute': risk_minutes,
                                         'type': np.where(success, 0, np.where(terminal, 1, 2))})
                for (direction, label, minute, kind), count in risk_rows.groupby(['direction', 'group', 'minute', 'type']).size().items():
                    survival[(job['mode'], job['role'], direction, dimension, label)][minute][kind] += count
            hist_keys = pd.DataFrame({'year': year, 'direction': chunk.direction, 'status': chunk.completion_status})
            for store, field, bins in [(durations, 'followup_minutes', DURATION_BINS), (maes, 'MAE_upper', MAE_BINS),
                                       (rewards, 'reward', [-np.inf, 0, .1, .25, .5, .75, .9, .98, 1, np.inf])]:
                for key, count in hist_keys.assign(bin=pd.cut(chunk[field], bins, include_lowest=True).astype(str)).groupby(['year', 'direction', 'status', 'bin']).size().items():
                    store.add((job['mode'], job['role']) + key, count)
            if job['mode'] == 'touch_spread':
                one_direction = chunk.direction.eq('LONG')
                for field in FEATURES_MARKET_OBS_OUTPUT:
                    for yr, values in chunk.loc[one_direction].groupby(year[one_direction])[field]:
                        feature_sums[(yr, field)] += [len(values), values.sum(), (values ** 2).sum()]
                        for bucket, count in pd.cut(values, feature_edges[field], include_lowest=True).astype(str).value_counts().items():
                            feature_hist.add((yr, field, bucket), int(count))
        if observed_rows != sum(job['counts'].get(status, 0) for status in ['success', 'terminal_failure', 'right_censored']):
            raise ValueError('Event row count disagrees with completion manifest')
        print(f'[FG statistics] {job_number}/{len(jobs)} {job["contract"]} {job["mode"]}', flush=True)
    grouped_contracts = defaultdict(list)
    for key, values in aggregate.items():
        grouped_contracts[key[:-1]].append((key[-1], values))
    rng = np.random.default_rng(config['bootstrap_seed'])
    rows = []
    for key, contributions in sorted(grouped_contracts.items()):
        values = sum((x[1] for x in contributions), np.zeros(len(METRICS)))
        metrics = dict(zip(METRICS, values))
        row = dict(zip(['mode', 'role', 'direction', 'dimension', 'group'], key))
        row.update(metrics, contracts=len(contributions), trading_days=len(unique_days[key]),
                   inference_status='insufficient_contract_clusters' if len(contributions) < 6 else 'descriptive_cluster_intervals',
                   mean_success_wait=metrics['wait_sum'] / metrics['success'] if metrics['success'] else None,
                   mean_known_reward=metrics['reward_sum'] / metrics['reward_known'] if metrics['reward_known'] else None)
        for horizon in HORIZONS:
            row[horizon + '_lower'] = metrics[horizon + '_hit'] / metrics['n']
            row[horizon + '_upper'] = (metrics[horizon + '_hit'] + metrics[horizon + '_unknown']) / metrics['n']
            selected = [METRICS.index(k) for k in ['n', horizon + '_hit', horizon + '_unknown']]
            low, high = contract_intervals(np.array([x[1][selected] for x in contributions]), config['bootstrap_replicates'], rng)
            row[horizon + '_cluster_ci_low'], row[horizon + '_cluster_ci_high'] = low, high
        rows.append(row)
    table = pd.DataFrame(rows)
    signature = hashlib.sha256((summary['signature'] + statistics_hash).encode()).hexdigest()[:16]
    output = directory / 'statistics' / signature
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / 'group_statistics.csv', index=False)
    for store, name in [(durations, 'duration_distribution'), (maes, 'mae_upper_distribution'), (rewards, 'reward_distribution')]:
        store.save(output / (name + '.csv'), ['mode', 'role', 'year', 'direction', 'status', 'bin'])
    feature_hist.save(output / 'feature_distribution.csv', ['year', 'feature', 'reference_bin'])
    pd.DataFrame([dict(year=key[0], feature=key[1], n=v[0], mean=v[1]/v[0], std=np.sqrt(max(0, v[2]/v[0]-(v[1]/v[0])**2)))
                  for key, v in sorted(feature_sums.items())]).to_csv(output / 'feature_moments.csv', index=False)
    curve_rows = [dict(zip(['mode', 'role', 'direction', 'dimension', 'group'], key), minute=t, aj_touch_probability=p)
                  for key, counts in sorted(survival.items()) for t, p in competing_risk_curve(counts)]
    pd.DataFrame(curve_rows).to_csv(output / 'competing_risk_curves.csv', index=False)
    block_intervals(daily, config, output)
    conditional_analysis(aggregate, reference_contracts, output)
    plot_statistics(table, pd.DataFrame(curve_rows), output)
    write_json(output / 'report_manifest.json', dict(schema=1, event_signature=summary['signature'],
        statistics_sha256=statistics_hash, blind_sealed=True, selected_roles=summary['selected_roles'],
        selected_contracts=summary['selected_contracts'],
        reference_contracts=sorted(reference_contracts), volatility_edges=vol_edges.tolist(),
        seed=config['bootstrap_seed'], replicates=config['bootstrap_replicates'],
        limitations=['Informative censoring can bias AJ estimates; bounds do not impute failures.',
                     'Overlapping probes are not executable combined PnL.', 'Small contract clusters cannot support drift significance.',
                     'Five-day calendar follows observed DB trading dates, not an independently verified exchange calendar.']))
    print(f'[FG statistics] output={output}', flush=True)
    return output


class CounterTable:
    def __init__(self):
        self.counts = defaultdict(int)

    def add(self, key, count):
        self.counts[key] += count

    def save(self, path, columns):
        pd.DataFrame([dict(zip(columns, key), count=value) for key, value in sorted(self.counts.items())]).to_csv(path, index=False)


def block_intervals(daily, config, output):
    groups = defaultdict(lambda: defaultdict(list))
    for key, values in sorted(daily.items()):
        groups[key[:3]][key[3]].append(values)
    rng = np.random.default_rng(config['bootstrap_seed'])
    rows = []
    for key, contracts in sorted(groups.items()):
        for length in [5, 20]:
            totals = np.zeros((config['bootstrap_replicates'], 3))
            for values in contracts.values():
                days = np.array(values)
                n = len(days)
                starts = rng.integers(n, size=(config['bootstrap_replicates'], int(np.ceil(n / length))))
                indices = (starts[:, :, None] + np.arange(length)[None, None, :]) % n
                totals += days[indices.reshape(len(totals), -1)[:, :n]].sum(axis=1)
            low = totals[:, 1] / totals[:, 0]
            high = (totals[:, 1] + totals[:, 2]) / totals[:, 0]
            rows.append(dict(zip(['mode', 'role', 'direction'], key), block_days=length,
                             contracts=len(contracts), ci_low=float(np.quantile(low, .025)), ci_high=float(np.quantile(high, .975))))
    pd.DataFrame(rows).to_csv(output / 'day_block_sensitivity.csv', index=False)


def conditional_analysis(aggregate, reference_contracts, output):
    """Fixed reference weights, explicit common-support loss; descriptive only."""
    reference, cells = defaultdict(float), defaultdict(lambda: np.zeros(3))
    for (mode, role, direction, dimension, label, contract), values in aggregate.items():
        if mode != 'touch_spread' or dimension != 'state':
            continue
        year, cell = label.split('|', 1)
        selected = values[[METRICS.index(x) for x in ['n', '60m_hit', '60m_unknown']]]
        cells[(direction, year, cell)] += selected
        if contract in reference_contracts:
            reference[(direction, cell)] += values[0]
    rows = []
    for direction, year in sorted({key[:2] for key in cells}):
        reference_total = sum(v for (d, _), v in reference.items() if d == direction)
        lower, upper, supported_weight = 0.0, 0.0, 0.0
        for (d, cell), weight in reference.items():
            if d != direction:
                continue
            values = cells.get((direction, year, cell))
            if values is None or values[0] < 100:
                continue
            w = weight / reference_total
            lower += w * values[1] / values[0]
            upper += w * (values[1] + values[2]) / values[0]
            supported_weight += w
        # Missing/sparse cells keep their fixed reference weight as unresolved
        # [0,1], rather than renormalizing to a different population each year.
        rows.append(dict(direction=direction, year=year, reference_weight_covered=supported_weight,
                         standardized_60m_lower=lower,
                         standardized_60m_upper=min(1.0, upper + 1 - supported_weight),
                         conclusion='descriptive_only_not_full_18D_control'))
    pd.DataFrame(rows).to_csv(output / 'conditional_standardization.csv', index=False)


def plot_statistics(table, curves, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    annual = table[(table['mode'] == 'touch_spread') & (table.dimension == 'year')].sort_values(['group', 'role'])
    for direction in ['LONG', 'SHORT']:
        part = annual[annual.direction == direction]
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        labels = part.group + '/' + part.role
        axes[0].plot(labels, part['60m_lower'], marker='o', label='Observed lower bound')
        axes[0].fill_between(np.arange(len(part)), part['60m_lower'], part['60m_upper'], alpha=.25, label='Censoring bounds')
        axes[0].set_ylabel('P(touch <= 60 natural minutes)'); axes[0].legend()
        axes[1].plot(labels, part.censored / part.n, marker='o', color='orange')
        axes[1].set_ylabel('Right-censoring fraction'); axes[1].tick_params(axis='x', rotation=45)
        fig.suptitle(direction + ': descriptive overlapping probes, blind sealed')
        fig.tight_layout()
        for extension in ['png', 'svg']:
            fig.savefig(output / f'annual_{direction}.{extension}')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for (mode, role, direction), part in curves[curves.dimension == 'overall'].groupby(['mode', 'role', 'direction']):
        selected = part[part.minute <= 1440]
        ax.step(selected.minute, selected.aj_touch_probability, where='post', label=f'{mode}/{role}/{direction}')
    ax.set_xscale('symlog', linthresh=5); ax.set_xlabel('Natural minutes'); ax.set_ylabel('AJ touch probability')
    ax.legend(fontsize=7); ax.set_title('Competing terminal risk; noninformative censoring assumption')
    fig.tight_layout()
    for extension in ['png', 'svg']:
        fig.savefig(output / f'touch_curves.{extension}')
    plt.close(fig)
    feature_table = pd.read_csv(output / 'feature_moments.csv')
    fig, axes = plt.subplots(6, 3, figsize=(13, 18))
    for ax, (field, part) in zip(axes.flat, feature_table.groupby('feature')):
        part = part.sort_values('year')
        ax.plot(part.year, part['mean'], marker='o')
        ax.fill_between(part.year, part['mean'] - part['std'], part['mean'] + part['std'], alpha=.2)
        ax.set_title(field, fontsize=9)
    fig.suptitle('18 existing observations: yearly mean +/- descriptive SD (not confidence intervals)')
    fig.tight_layout()
    for extension in ['png', 'svg']:
        fig.savefig(output / f'feature_changes.{extension}')
    plt.close(fig)
    condition = pd.read_csv(output / 'conditional_standardization.csv')
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for direction, part in condition.groupby('direction'):
        part = part.sort_values('year')
        axes[0].plot(part.year, part.standardized_60m_lower, marker='o', label=direction)
        axes[0].fill_between(part.year, part.standardized_60m_lower, part.standardized_60m_upper, alpha=.2)
        axes[1].plot(part.year, part.reference_weight_covered, marker='o', label=direction)
    axes[0].set_ylabel('Fixed reference 60m bounds'); axes[0].legend()
    axes[1].set_ylabel('Supported reference weight'); axes[1].set_xlabel('Year')
    fig.suptitle('Simple state controls; missing support remains unresolved, blind sealed')
    fig.tight_layout()
    for extension in ['png', 'svg']:
        fig.savefig(output / f'conditional_changes.{extension}')
    plt.close(fig)
