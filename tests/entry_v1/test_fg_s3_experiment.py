"""Managed configuration/snapshots over the actual completed A/B CPU adapters."""
import copy
import concurrent.futures
from pathlib import Path
import subprocess
import sys
import threading

import pytest
import yaml

from gym_trading_env.research import fg_s3_experiment as manager
from gym_trading_env.research.fg_replay import digest, read_manifest

DREAMER = Path(__file__).resolve().parents[3]/'dreamerv3'
sys.path.insert(0, str(DREAMER / 'embodied/tests'))
from test_fg_s3b import bundle  # Reuse the existing original-neural source fixture.


def experiment(bundle, root, trainer='s3a', existing_features=False):
    root.mkdir()
    source = bundle['source']
    template = root / 'template.yaml'
    template.write_text(yaml.safe_dump({'defaults': source['config']}))
    lm = read_manifest(bundle['labels'] / 'manifest.json')
    cfg = dict(schema=manager.SCHEMA, experiment='managed_' + trainer, trainer=trainer,
        diagnostic=True, prepared_dir=str(root / 'prepared'), dreamer_root=str(DREAMER),
        model=dict(template=str(template), presets=['defaults']),
        paths=dict(dataset=lm['source']['dataset'], oracle_receipt=lm['source']['oracle_receipt'],
            labels=str(bundle['labels']), source_checkpoint=source['path'],
            features=str(bundle['cache'].root if existing_features else root / 'features'),
            weights=str(bundle['weights']) if trainer == 's3b' else None),
        features=dict(seed=0, shard_rows=3, max_rows=None, device='cpu'),
        train=dict(mode='S3B_CERTIFIED' if trainer == 's3b' else 'S3A_FULL', seed=4,
            batch_size=3, updates=2, comparison_updates=2, save_every=1,
            device='cpu', logdir=str(root / 'train'), optimizer=dict(lr=0.00004, agc=0.3)))
    path = root / 'experiment.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg


@pytest.mark.parametrize('trainer', ['s3a', 's3b'])
def test_prepare_start_resume_original_actor_and_frozen_snapshot(bundle, tmp_path, trainer):
    config, cfg = experiment(bundle, tmp_path / trainer, trainer, existing_features=trainer == 's3b')
    prepared, logdir = Path(cfg['prepared_dir']), Path(cfg['train']['logdir'])
    receipt = manager.prepare(config)
    assert receipt['status'] == ('READY_FOR_BC' if trainer == 's3b' else 'READY_FOR_FEATURES')
    assert not logdir.exists()
    effective = yaml.safe_load((prepared / 'effective_config.yaml').read_text())
    assert effective['train']['optimizer']['warmup'] == 0
    assert yaml.safe_load((prepared / 'source.config.yaml').read_text()) == bundle['source']['config']
    for field in ('paths', 'train', 'features'):
        assert field in effective
    result = manager.start(prepared, config=config)
    assert result['status'] == 'TRAINING_CHECKPOINT_VERIFIED' and result['bc_updates'] == 2
    assert result['frozen_keys'] > 100
    assert (logdir / 'managed_snapshot/preparation_receipt.json').is_file()
    assert read_manifest(logdir / 'managed_snapshot/manifest.json')['binding']['prepared_manifest_sha256'] == digest(prepared / 'manifest.json')
    assert len(list((prepared / 'attempts').glob('*.json'))) == (2 if trainer == 's3a' else 1)
    checkpoint = logdir / 'ckpt' / (logdir / 'ckpt/latest').read_text().strip()
    saved = digest(checkpoint / 'state.pkl')
    # Simulate a lost launcher snapshot copy while preserving user artifacts.
    # Resume reconstructs it only after verifying the actual saved checkpoint.
    (logdir / 'managed_snapshot').rename(logdir / 'snapshot_before_launcher_crash')
    assert manager.start(prepared, resume=True, config=config)['bc_updates'] == 2
    assert digest(checkpoint / 'state.pkl') == saved
    identity_file = logdir / 'run_identity.json'
    identity_file.rename(logdir / 'run_identity.saved.json')
    with pytest.raises(ValueError, match='no saved training identity'):
        manager.audit(prepared)
    (logdir / 'run_identity.saved.json').rename(identity_file)
    # A deliberate central-YAML edit cannot alter the saved run through --config.
    changed = copy.deepcopy(cfg); changed['train']['seed'] += 1
    config.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match='YAML differs'):
        manager.require_prepared(prepared, config)
    # Explicit --prepared uses the immutable saved configuration instead.
    assert manager.require_prepared(prepared)[1]['train']['seed'] == 4
    with pytest.raises(ValueError, match='new logdir'):
        manager.start(prepared)


def test_preparation_refuses_template_drift_overlap_and_snapshot_tamper(bundle, tmp_path):
    config, cfg = experiment(bundle, tmp_path / 'guards', existing_features=True)
    bad = copy.deepcopy(cfg)
    bad['train']['logdir'] = str(Path(bundle['source']['path']).parent.parent / 'new_logdir')
    config.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match='immutable source'):
        manager.read_experiment(config)
    config.write_text(yaml.safe_dump(cfg))
    defaults = copy.deepcopy(bundle['source']['config'])
    defaults['agent']['policy']['units'] += 1
    template = Path(cfg['model']['template'])
    template.write_text(yaml.safe_dump({'defaults': defaults}))
    with pytest.raises(ValueError, match='architecture: policy'):
        manager.prepare(config)
    assert not Path(cfg['prepared_dir']).exists()
    template.write_text(yaml.safe_dump({'defaults': bundle['source']['config']}))
    manager.prepare(config)
    prepared = Path(cfg['prepared_dir'])
    assert manager.audit(prepared)['status'] == 'READY_FOR_BC'
    assert not (prepared / 'features_binding.json').exists()
    with pytest.raises(FileExistsError, match='Prepared experiment'):
        manager.prepare(config)
    target = prepared / 'effective_config.yaml'
    target.write_text(target.read_text() + '\n# changed\n')
    with pytest.raises(ValueError, match='changed'):
        manager.require_prepared(prepared)


def test_central_configuration_interfaces_and_formal_partial_guard(tmp_path):
    names = ('S3A_ORACLE_BC', 'S3A_FULL', 'S3A_MATCHED', 'S3B_CERTIFIED')
    configs = [manager.read_experiment(name)[1] for name in names]
    assert len({c['prepared_dir'] for c in configs}) == len(names)
    assert len({c['train']['logdir'] for c in configs}) == len(names)
    for cfg in configs:
        assert cfg['model']['template'] == str(DREAMER / 'dreamerv3/configs.yaml')
    for key in ('paths', 'features'):
        assert configs[1][key] == configs[2][key] == configs[3][key]
    assert configs[1]['train']['optimizer'] == configs[2]['train']['optimizer'] == configs[3]['train']['optimizer']
    bad = copy.deepcopy(configs[0]); bad['features']['max_rows'] = 1024
    path = tmp_path / 'partial.yaml'; path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match='partial features'):
        manager.read_experiment(path)


def test_worker_keeps_exclusive_lock_after_launcher_descriptor_closes(tmp_path, monkeypatch):
    # Exercise actual Popen descriptor forwarding with a stdlib-only worker;
    # no training, signal, or process termination is needed for this boundary.
    prepared = tmp_path / 'prepared'; prepared.mkdir()
    (prepared / 'manifest.json').write_text('{}')
    cfg = dict(experiment='lock_probe', dreamer_root=str(DREAMER),
               train=dict(device='cpu', logdir=str(tmp_path / 'run')))
    real_popen, started, processes = subprocess.Popen, threading.Event(), []
    def worker(argv, **kwargs):
        assert kwargs['pass_fds']
        process = real_popen([sys.executable, '-c',
            'import sys; print("ready", flush=True); sys.stdin.readline()'],
            stdin=subprocess.PIPE, **kwargs)
        processes.append(process); started.set()
        return process
    monkeypatch.setattr(manager.subprocess, 'Popen', worker)
    lock_path = tmp_path / '.run.s3.lock'
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        with manager._lock(lock_path) as lock:
            future = pool.submit(manager._run_phase, prepared, cfg, 'train', lock_fds=(lock.fileno(),))
            assert started.wait(5)
        try:
            with pytest.raises(BlockingIOError):
                with manager._lock(lock_path):
                    pytest.fail('Launcher close released a live worker lock')
        finally:
            processes[0].stdin.write('done\n'); processes[0].stdin.flush(); processes[0].stdin.close()
        future.result(timeout=10)
    with manager._lock(lock_path):
        pass
