"""Price-identity coverage for proposed folds; never read event outcomes."""
import json
from pathlib import Path

import pandas as pd
import yaml

from gym_trading_env.envs.entry_contract import DirectionSchedule
from gym_trading_env.research.fg_history_study import digest


def window_coverage(config_path):
    config = yaml.safe_load(Path(config_path).read_text())
    master = yaml.safe_load(Path(config['master_config']).read_text())
    manifest = json.loads(Path(config['export_manifest']).read_text())
    if not manifest.get('main_only') or manifest.get('minute_clock') != 345:
        raise ValueError('Window coverage requires the current main-only 345 export')
    schedule = DirectionSchedule(master['directions'])
    daily = []
    for item in manifest['files']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Input changed after export')
        frame = pd.read_csv(item['path'], usecols=['Date', 'trading_date', 'symbol'])
        frame = frame[frame.trading_date < config['blind_start']]
        for day, part in frame.groupby('trading_date', sort=False):
            daily.append(dict(day=day, contract=item['contract'], month=item['contract'][-2:], rows=len(part),
                              direction=schedule.at(item['contract'], part.Date.iloc[0], day)))
    days = pd.DataFrame(daily)
    first = pd.Timestamp(days.day.min())
    rows = []
    cutoffs = ['2021-12-31', master['data']['split']['train_end_inclusive'], '2023-12-31', '2024-04-30']
    for cutoff_text in cutoffs:
        cutoff = pd.Timestamp(cutoff_text)
        validation_end = cutoff + pd.DateOffset(months=3)
        if validation_end >= pd.Timestamp(config['blind_start']):
            raise ValueError('Proposed fold reaches sealed blind period')
        for months in config['training_windows_months']:
            start = first if months == 'all' else cutoff + pd.Timedelta(days=1) - pd.DateOffset(months=int(months))
            selected = days[(days.day >= str(start.date())) & (days.day <= cutoff_text)]
            if selected.empty:
                continue
            row = dict(cutoff=cutoff_text, window=str(months), planned_start=str(start.date()),
                       actual_start=selected.day.min(), actual_end=selected.day.max(),
                       validation_start=str((cutoff + pd.Timedelta(days=1)).date()),
                       validation_end=str(validation_end.date()), full_requested_history_available=start >= first,
                       contracts=selected.contract.nunique(), trading_days=selected.day.nunique(), rows=int(selected.rows.sum()))
            for month in ['01', '05', '09']:
                row['month_' + month + '_rows'] = int(selected.loc[selected.month == month, 'rows'].sum())
            for direction in ['LONG', 'SHORT', 'NO_TRADE']:
                row[direction + '_rows'] = int(selected.loc[selected.direction == direction, 'rows'].sum())
            rows.append(row)
    output = Path(config['output']) / 'window_coverage.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output
