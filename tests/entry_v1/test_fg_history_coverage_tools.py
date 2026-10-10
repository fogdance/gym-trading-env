import json
from pathlib import Path

import pandas as pd
import yaml

from gym_trading_env.research.fg_history_profile import market_profile
from gym_trading_env.research.fg_history_windows import window_coverage
from gym_trading_env.research.fg_history_study import digest
from tests.entry_v1.test_fg_history_events import market


def test_profile_and_window_coverage_exclude_blind_rows_without_labels(tmp_path):
    frame = market([(1000,1002,999,1000)] * 3,
                   times=['2021-01-04 09:31', '2021-01-04 09:32', '2024-08-20 09:31'])
    frame['trading_date'] = pd.to_datetime(frame.Date).dt.strftime('%Y-%m-%d')
    source = tmp_path / 'bars.csv'
    frame.to_csv(source, index=False)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(main_only=True, minute_clock=345,
        files=[dict(path=str(source), sha256=digest(source), contract='CZCE.FG2701')])))
    config = tmp_path / 'study.yaml'
    config.write_text(yaml.safe_dump(dict(master_config=str(Path('configs/fg_entry_v1/fg.yaml').resolve()),
        export_manifest=str(manifest), output=str(tmp_path / 'output'), blind_start='2024-08-20',
        training_windows_months=[12,24,36,'all'])))
    profile = pd.read_csv(market_profile(config))
    assert profile.bars.sum() == 2
    assert list(profile.year) == [2021]
    assert profile.mean_bar_range.iloc[0] == 3
    windows = pd.read_csv(window_coverage(config))
    assert (windows.actual_end < '2024-08-20').all()
    assert (windows.rows == 2).all()
    assert (windows.validation_end < '2024-08-20').all()
