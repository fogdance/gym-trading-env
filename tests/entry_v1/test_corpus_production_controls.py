"""Explicit production scope, CPU budget and frozen verifier identity."""
from pathlib import Path

import pandas as pd
import pytest

from gym_trading_env.research import fg_s3_e_quality_corpus_cli as cli
from gym_trading_env.research import fg_s3_e_quality_supervision as supervision
from gym_trading_env.research.fg_replay import digest, signed, write_json


def test_explicit_range_can_select_earlier_days_without_changing_suffix():
    days = pd.Series(['2024-04-01'] * 3 + ['2024-04-02'] * 3 + ['2024-04-03'] * 3)
    selected, start, end, scope = cli._sample_bounds(days, '2024-04-01', '2024-04-02')
    assert selected == ['2024-04-01', '2024-04-02']
    assert (start, end) == (0, 5)
    assert scope == 'explicit_official_day_range_with_full_train_suffix'
    assert cli._sample_bounds(days)[:3] == (['2024-04-02', '2024-04-03'], 3, 8)


@pytest.mark.parametrize('start,end', [
    ('2024-04-01', None), ('2024-04-02', '2024-04-01'),
    ('2024-03-29', '2024-04-02'), ('2024-04-01', '2024-04-04'),
])
def test_scope_never_silently_clips_or_substitutes_days(start, end):
    with pytest.raises(ValueError, match='official|together'):
        cli._sample_bounds(pd.Series(['2024-04-01'] * 2 + ['2024-04-02'] * 2), start, end)


def test_explicit_24_workers_distribute_across_six_accounts_and_labels():
    limits = cli._worker_limits(6, cpus=24, requested_workers=24,
        available=20*1024**3, process_memory=400*1024**2, peak_rss=500*1024**2)
    assert limits['workers'] == 6 and limits['label_workers'] == [3]*6
    assert limits['total_worker_processes'] == 24
    assert limits['requested_worker_processes'] == 24
    assert limits['native_threads_per_worker'] == 1
    with pytest.raises(ValueError, match='affinity'):
        cli._worker_limits(6, cpus=12, requested_workers=24)


def test_cpu_budget_remains_memory_bounded():
    limits = cli._worker_limits(6, cpus=24, requested_workers=24,
        available=4*1024**3, process_memory=1024**3, peak_rss=1024**3)
    assert limits['total_worker_processes'] == 2
    assert limits['worker_limit_reason'] == 'memory'


def test_external_output_reaches_source_authentication_without_writing(tmp_path, monkeypatch):
    output = tmp_path/'external_disk'/'new_batch'
    def source(*args, **kwargs):
        raise ValueError('SOURCE_AUTHENTICATION_REACHED')
    monkeypatch.setattr(cli, 'source_inputs', source)
    with pytest.raises(ValueError, match='SOURCE_AUTHENTICATION_REACHED'):
        cli.generate_small(tmp_path/'pool.json', tmp_path/'receipt', 'CZCE.FG809', output)
    assert not output.exists()


def test_cli_routes_dates_workers_and_external_output(monkeypatch, tmp_path):
    captured = {}
    def generate(*args, **kwargs):
        captured.update(kwargs)
        return {'status': 'ROUTING_ONLY', 'training_admitted': False}
    monkeypatch.setattr(cli, 'generate_small', generate)
    assert cli.main(['generate', '--source-pool', 'pool.json', '--env-receipt', 'receipt',
        '--contract', 'CZCE.FG809', '--out', str(tmp_path/'batch'),
        '--start-day', '2018-08-06', '--end-day', '2018-08-08', '--workers', '24']) == 0
    assert captured == dict(seed=202, start_day='2018-08-06', end_day='2018-08-08', workers=24)


def test_bounded_native_replay_preserves_complete_train_suffix_and_account(tmp_path):
    from wp02_fixture import native_factory
    from gym_trading_env.research.fg_s3_e_quality_corpus import generate_behavior, behavior_specs
    from gym_trading_env.research.fg_s3_e_quality_strict import QualityStrictTeacher
    from gym_trading_env.research.fg_replay import iter_sidecar
    factory, _, _ = native_factory(tmp_path/'market', minutes=45, chunk=9)
    env, initial = cli._prefix(factory, 5)
    source = dict(fold='TRAIN', state_cache_entries=0)
    strict = QualityStrictTeacher(env, source=source, start_cursor=5)
    try:
        assert strict.query.end == 44
        trace, receipt = generate_behavior(factory, behavior_specs()[0], tmp_path/'bounded',
            Path(__file__).resolve().parents[3]/'dreamerv3', source=source,
            initial_state=initial, end_cursor=20, strict=strict)
        rows = list(iter_sidecar(tmp_path/'bounded'))
        assert (rows[0]['cursor'], rows[-1]['cursor']) == (5, 20)
        assert trace['bounded_observation_end'] == 20 and not trace['terminal']
        assert receipt['status'] == 'PASS' and receipt['teacher_labels_verified'] == trace['records']
        assert all(row['teacher']['L2']['suffix_boundary_identity']['cursor'] == 44 for row in rows)
    finally:
        env.close()


def test_explicit_producer_pin_requires_external_sha_and_clean_runtime(tmp_path):
    import subprocess
    producer = tmp_path/'gym-trading-env'
    heads = {}
    for name, root, filename in [('gym', producer, 'src/gym_trading_env/source.py'),
                                 ('dreamer', producer.parent/'dreamerv3', 'dreamerv3/source.py')]:
        root.mkdir()
        subprocess.run(['git', 'init', '-q', str(root)], check=True)
        path = root/filename
        path.parent.mkdir(parents=True)
        path.write_text('VALUE = 1\n')
        subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(root), '-c', 'user.name=Corpus test', '-c',
            'user.email=corpus-test@example.invalid', 'commit', '-qm', 'controlled fixture'], check=True)
        heads[name] = dict(sha=subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip())
    lock = tmp_path/'producer_lock.json'
    write_json(lock, signed(dict(protocol='FG_S3_E_QUALITY_PRODUCER_PIN_V1', repositories=heads)))
    with pytest.raises(ValueError, match='PIN_SHA256'):
        supervision.frozen_producer_audit(producer, producer_lock=lock)
    result = supervision.frozen_producer_audit(producer, producer_lock=lock, producer_lock_sha256=digest(lock))
    assert result['gym']['head'] == heads['gym']['sha']
    with pytest.raises(ValueError, match='PIN_SHA256_MISMATCH'):
        supervision.frozen_producer_audit(producer, producer_lock=lock, producer_lock_sha256='0'*64)
    (producer/'src/gym_trading_env/source.py').write_text('VALUE = 2\n')
    with pytest.raises(ValueError, match='FROZEN_PRODUCER_CODE_CHANGED'):
        supervision.frozen_producer_audit(producer, producer_lock=lock, producer_lock_sha256=digest(lock))
