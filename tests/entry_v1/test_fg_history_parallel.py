from concurrent.futures import ProcessPoolExecutor
import gzip
import multiprocessing
from pathlib import Path

import pandas as pd
import pytest

from gym_trading_env.research.fg_history_parallel import run_job, run_events
from gym_trading_env.research.fg_history_study import digest
from tests.entry_v1.test_fg_history_events import COSTS, SCHEDULE, market


def task_for(tmp_path, name='probe'):
    frame = market([(1000,1000,1000,1000), (1000,1000,1000,1000), (1000,1010,990,1008)])
    source = tmp_path / (name + '.csv')
    frame.to_csv(source, index=False)
    return dict(directory=str(tmp_path / name), signature='fixed-test-signature', contract='CZCE.FG2701',
                role='train', mode='conservative_spread', start='2026-01-05', end='2026-01-05',
                blind_start='2027-01-01',
                files=[dict(path=str(source), sha256=digest(source), next_main_contract=None)],
                schedule=dict(SCHEDULE, timezone='Asia/Shanghai'), costs=COSTS,
                take_profit=5, quantity=3, manifest_sha256='fixture')


def test_spawn_matches_serial_payload_and_resume_verifies_outputs(tmp_path):
    sequential = task_for(tmp_path, 'serial')
    parallel = dict(sequential, directory=str(tmp_path / 'parallel'))
    serial_result = run_job(sequential)
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        parallel_result = pool.submit(run_job, parallel).result(timeout=60)
    assert serial_result['counts'] == parallel_result['counts']
    for name in ['events.csv.gz', 'rejected.csv.gz']:
        with gzip.open(Path(sequential['directory']) / name, 'rt') as left, gzip.open(Path(parallel['directory']) / name, 'rt') as right:
            assert left.read() == right.read()
    assert run_job(parallel)['resumed']
    (Path(parallel['directory']) / 'events.csv.gz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Completed output changed'):
        run_job(parallel)


def test_role_slice_censors_before_future_and_blind_request_is_rejected(tmp_path):
    task = task_for(tmp_path)
    frame = pd.read_csv(task['files'][0]['path'])
    frame.loc[2, ['Date', 'bob', 'trading_date']] = ['2026-01-06 09:31', '2026-01-06 09:30', '2026-01-06']
    frame.to_csv(task['files'][0]['path'], index=False)
    task['files'][0]['sha256'] = digest(task['files'][0]['path'])
    run_job(task)
    events = pd.read_csv(Path(task['directory']) / 'events.csv.gz')
    assert set(events.completion_status) == {'right_censored'}
    assert events.reward.isna().all()
    assert events.first_touch_timestamp.isna().all()
    assert events.horizon_5day_end.isna().all()
    assert set(events.observation_end) == {'2026-01-05 09:32:00'}
    with pytest.raises(ValueError, match='Blind labels remain sealed'):
        run_events('configs/fg_entry_v1/history_study.yaml', roles=['test'])
    with pytest.raises(ValueError, match='worker boundary'):
        run_job(dict(task, role='blind'))
