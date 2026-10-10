"""Raw price scale and bar ranges for interpretation; never add model indicators."""
import json
from pathlib import Path

import pandas as pd
import yaml

from gym_trading_env.research.fg_history_study import digest


def market_profile(config_path):
    config = yaml.safe_load(Path(config_path).read_text())
    manifest = json.loads(Path(config['export_manifest']).read_text())
    if not manifest.get('main_only') or manifest.get('minute_clock') != 345:
        raise ValueError('Market profile requires the current main-only 345 export')
    frames = []
    for item in manifest['files']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Market changed after export')
        frame = pd.read_csv(item['path'], usecols=['Date', 'bob', 'trading_date', 'Open', 'High', 'Low', 'Close'])
        frame = frame[frame.trading_date < config['blind_start']].copy()
        frame = frame[pd.to_datetime(frame.Date) - pd.to_datetime(frame.bob) == pd.Timedelta(minutes=1)]
        frame['year'] = frame.trading_date.str[:4]
        frame['contract'] = item['contract']
        frame['bar_range'] = frame.High - frame.Low
        frame['open_close_move'] = (frame.Close - frame.Open).abs()
        frames.append(frame)
    all_bars = pd.concat(frames, ignore_index=True)
    rows = []
    for year, part in all_bars.groupby('year'):
        rows.append(dict(year=year, bars=len(part), contracts=part.contract.nunique(),
                         trading_days=part.trading_date.nunique(), mean_price=part.Close.mean(),
                         mean_bar_range=part.bar_range.mean(), median_bar_range=part.bar_range.median(),
                         p90_bar_range=part.bar_range.quantile(.9), mean_abs_open_close=part.open_close_move.mean()))
    output = Path(config['output']) / 'market_profile.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output
