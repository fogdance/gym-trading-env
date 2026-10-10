from pathlib import Path

import pandas as pd
import pytest

from gym_trading_env.research.fg_history_folds import build_tasks
from tests.entry_v1.test_fg_history_parallel import task_for
from gym_trading_env.research.fg_history_parallel import run_job


def test_forward_queries_share_exact_slices_and_censor_at_training_boundary(tmp_path):
    task = task_for(tmp_path)
    source = task['files'][0]
    frame = pd.read_csv(source['path'])
    frame.loc[2, ['Date', 'bob', 'trading_date']] = ['2026-01-06 09:31', '2026-01-06 09:30', '2026-01-06']
    frame.to_csv(source['path'], index=False)
    from gym_trading_env.research.fg_history_study import digest
    source['sha256'] = digest(source['path'])
    # Apparent main expiry lies OUTSIDE the training slice; it cannot become a failure.
    source.update(next_main_contract='CZCE.FG2705', main_exit_at='2026-01-06 09:31')
    coverage = pd.DataFrame([dict(window=w, planned_start=start, cutoff='2026-01-05',
        validation_start='2026-01-06', validation_end='2026-01-06')
        for w, start in [('12', '2025-01-01'), ('24', '2024-01-01')]])
    config = dict(blind_start='2027-01-01', execution_modes=['conservative_spread'],
                  take_profit_points=5, quantity=3)
    files = {task['contract']: [(source, '2026-01-05', '2026-01-06')]}
    jobs, datasets = build_tasks(coverage, files, config,
        dict(market_clock=task['schedule'], costs=task['costs']), tmp_path, 'test', 'fixture')
    assert len(jobs) == 2  # train and validation shared across the two windows
    assert datasets[0]['jobs'] == datasets[1]['jobs']
    train = next(job for job in jobs if job['role'] == 'train')
    result = run_job(train)
    assert not result['terminal_evidence']
    events = pd.read_csv(Path(train['directory']) / 'events.csv.gz')
    assert set(events.completion_status) == {'right_censored'}
    assert events.reward.isna().all()
    assert events.first_touch_timestamp.isna().all()
    coverage.loc[0, 'validation_end'] = config['blind_start']
    with pytest.raises(ValueError, match='sealed blind'):
        build_tasks(coverage, files, config, {}, tmp_path, 'test', 'fixture')
