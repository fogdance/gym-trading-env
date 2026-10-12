"""Managed S3 experiments: immutable preparation, separate A/B trainers."""
from contextlib import contextmanager
import copy
import fcntl
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

import yaml

from .fg_replay import check_files, digest, read_manifest, signed, write_json

GYM_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = GYM_ROOT / 'configs/experiments/fg_single3'
SCHEMA = 'FG_S3_EXPERIMENT_V1'
PROTOCOL = 'FG_S3_PREPARED_V1'
SNAPSHOT_FILES = {'experiment.requested.yaml', 'model.template.yaml', 'model.defaults.yaml',
                  'source.config.yaml', 'effective_config.yaml', 'input_hashes.json', 'code_hashes.json'}
TRAINERS = {'s3a': ('S3A_FULL',),
            's3b': ('S3A_FULL', 'S3A_MATCHED', 'S3B_CERTIFIED')}


def _mapping(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'{name} requires exactly these keys: {sorted(keys)}')


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _overlap(a, b):
    return a.is_relative_to(b) or b.is_relative_to(a)


def read_experiment(config):
    path = Path(config)
    if not path.is_file() and not path.is_absolute():
        path = CONFIG_ROOT / (str(config) if path.suffix else str(config) + '.yaml')
    path = path.resolve()
    cfg = yaml.safe_load(path.read_text())
    _mapping(cfg, ('schema', 'experiment', 'trainer', 'diagnostic', 'prepared_dir',
                   'dreamer_root', 'model', 'paths', 'features', 'train'), 'experiment')
    if cfg['schema'] != SCHEMA or cfg['trainer'] not in TRAINERS:
        raise ValueError('Foreign S3 experiment schema/trainer')
    if not isinstance(cfg['experiment'], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', cfg['experiment']):
        raise ValueError('Experiment must be a safe, nonempty run name')
    if type(cfg['diagnostic']) is not bool:
        raise ValueError('diagnostic must be an explicit boolean')
    _mapping(cfg['model'], ('template', 'presets'), 'model')
    if (not isinstance(cfg['model']['presets'], list) or not cfg['model']['presets']
            or any(not isinstance(x, str) for x in cfg['model']['presets'])):
        raise ValueError('Model presets must be a nonempty list of names')
    _mapping(cfg['paths'], ('dataset', 'oracle_receipt', 'labels', 'source_checkpoint',
                            'features', 'weights'), 'paths')
    _mapping(cfg['features'], ('seed', 'shard_rows', 'max_rows', 'device'), 'features')
    _mapping(cfg['train'], ('mode', 'seed', 'batch_size', 'updates', 'comparison_updates',
                            'save_every', 'device', 'logdir', 'optimizer'), 'train')
    if cfg['train']['mode'] not in TRAINERS[cfg['trainer']]:
        raise ValueError('Mode does not belong to the selected trainer')
    for section in ('features', 'train'):
        if type(cfg[section]['seed']) is not int or cfg[section]['seed'] < 0:
            raise ValueError('Seeds must be nonnegative integers')
        if cfg[section]['device'] not in ('cpu', 'gpu'):
            raise ValueError('S3 device must be cpu or gpu')
        if cfg['diagnostic'] and cfg[section]['device'] != 'cpu':
            raise ValueError('Diagnostic experiments are CPU-only')
    _positive(cfg['features']['shard_rows'], 'shard_rows')
    if cfg['features']['max_rows'] is not None:
        _positive(cfg['features']['max_rows'], 'max_rows')
        if not cfg['diagnostic']:
            raise ValueError('Formal experiments cannot use partial features')
    for name in ('batch_size', 'updates', 'comparison_updates', 'save_every'):
        _positive(cfg['train'][name], name)
    if cfg['train']['comparison_updates'] < cfg['train']['updates']:
        raise ValueError('Comparison budget is below training budget')
    if not isinstance(cfg['train']['optimizer'], dict):
        raise ValueError('Optimizer overrides must be a mapping')

    def resolve(value):
        if not isinstance(value, str) or not value:
            raise ValueError('Configured paths must be nonempty strings')
        p = Path(value).expanduser()
        return str((p if p.is_absolute() else GYM_ROOT / p).resolve())

    cfg['prepared_dir'] = resolve(cfg['prepared_dir'])
    cfg['dreamer_root'] = resolve(cfg['dreamer_root'])
    cfg['model']['template'] = resolve(cfg['model']['template'])
    cfg['train']['logdir'] = resolve(cfg['train']['logdir'])
    for key, value in cfg['paths'].items():
        cfg['paths'][key] = None if key == 'weights' and value is None else resolve(value)
    if cfg['trainer'] == 's3b' and cfg['paths']['weights'] is None:
        raise ValueError('S3-B comparison modes require frozen weights')
    if cfg['trainer'] == 's3a' and cfg['paths']['weights'] is not None:
        raise ValueError('Native S3-A does not consume value weights')
    _check_locations(cfg)
    return path, cfg


def _check_locations(cfg):
    prepared, logdir = Path(cfg['prepared_dir']), Path(cfg['train']['logdir'])
    features = Path(cfg['paths']['features'])
    inputs = [Path(cfg['paths'][n]) for n in ('dataset', 'oracle_receipt', 'labels')]
    inputs += [Path(cfg['paths']['source_checkpoint']).parent.parent,
               Path(cfg['dreamer_root']), Path(cfg['model']['template'])]
    if cfg['paths']['weights']:
        inputs.append(Path(cfg['paths']['weights']))
    if any(_overlap(output, p) for output in (prepared, logdir, features) for p in inputs):
        raise ValueError('S3 outputs overlap an immutable source/model/data directory')
    if any(_overlap(a, b) for a, b in ((prepared, logdir), (prepared, features), (logdir, features))):
        raise ValueError('Prepared artifacts, feature cache and training logdir must be separate')
    if not cfg['diagnostic']:
        if not prepared.is_relative_to(GYM_ROOT / 'artifacts') or prepared == GYM_ROOT / 'artifacts':
            raise ValueError('Formal prepared snapshots must be under this repository artifacts/')
        allowed = Path('/data/logdir/fg_single3')
        if not logdir.is_relative_to(allowed) or logdir == allowed:
            raise ValueError('Formal training logs/checkpoints must be under /data/logdir/fg_single3/')


def activate(cfg, device='cpu'):
    """Select the backend before any JAX imports, including preparation."""
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['JAX_PLATFORMS'] = 'cpu' if device == 'cpu' else 'cuda'
    os.environ['JAX_PLATFORM_NAME'] = 'cpu' if device == 'cpu' else 'gpu'
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(name, '1')
    root = cfg['dreamer_root']
    if root not in sys.path:
        sys.path.insert(0, root)
    import dreamerv3
    if Path(dreamerv3.__file__).resolve().parent.parent != Path(root):
        raise ValueError('Imported Dreamer belongs to a different configured repository')


def runtime_files(cfg):
    files = [Path(__file__), GYM_ROOT / 'tools/fg_s3_experiment.py']
    for root, names in ((Path(cfg['dreamer_root']), ('dreamerv3', 'embodied')),
                        (GYM_ROOT / 'src', ('gym_trading_env',))):
        for name in names:
            files.extend(p for p in (root / name).rglob('*.py') if 'tests' not in p.parts)
    files.append(Path(cfg['model']['template']))
    return {str(p): digest(p) for p in sorted(set(files))}


def resolve_defaults(cfg, source):
    import elements
    from dreamerv3.configuration import read_config_yaml
    from dreamerv3.fg_s3a_actor import DEFAULT_OPT
    from embodied.run.fg_checkpoint import plain
    definitions = read_config_yaml(cfg['model']['template'])
    presets = cfg['model']['presets']
    resolved = elements.Config(definitions[presets[0]])
    for name in presets[1:]:
        resolved = resolved.update(definitions[name])
    defaults = plain(resolved)
    # The checkpoint remains the exact model authority. A changed public
    # architecture must be reviewed instead of silently reshaping its tensors.
    for name in ('enc', 'dyn', 'dec', 'rewhead', 'conhead', 'availhead', 'policy', 'value'):
        if defaults['agent'][name] != source['config']['agent'][name]:
            raise ValueError('Shared model template differs from source architecture: ' + name)
    inherited = {k: v for k, v in defaults['agent']['opt'].items()
                 if k in DEFAULT_OPT and k not in ('wd', 'schedule', 'warmup', 'anneal')}
    opt = dict(DEFAULT_OPT, **inherited)
    overrides = cfg['train']['optimizer']
    if set(overrides) - set(DEFAULT_OPT):
        raise ValueError('Unknown Actor-only optimizer override')
    opt.update(overrides)
    if opt['wd'] or opt['warmup'] or opt['anneal'] or opt['schedule'] != 'const':
        raise ValueError('Actor-only baseline requires constant LR without decay/warmup')
    import math
    if any(not math.isfinite(float(opt[k])) for k in ('lr', 'agc', 'eps', 'beta1', 'beta2')):
        raise ValueError('Nonfinite Actor-only optimizer configuration')
    if opt['lr'] <= 0 or opt['agc'] < 0 or opt['eps'] <= 0 or not 0 <= opt['beta1'] < 1 or not 0 <= opt['beta2'] < 1:
        raise ValueError('Invalid Actor-only optimizer configuration')
    return defaults, opt


def admission(cfg, *, feature_required=False):
    from gym_trading_env.research.fg_s3a_labels import load_labels
    from embodied.run.fg_s3a_checkpoint import validate_source
    from embodied.run.fg_s3a_features import FeatureCache
    # Missing source is reported before hashing the full TRAIN input again.
    if not (Path(cfg['paths']['source_checkpoint']) / 'manifest.json').is_file():
        raise FileNotFoundError('Source checkpoint is not available: ' + cfg['paths']['source_checkpoint'])
    labels = load_labels(cfg['paths']['labels'])
    for name in ('dataset', 'oracle_receipt'):
        if cfg['paths'][name] != labels['source'][name]:
            raise ValueError('Experiment input differs from frozen S3-A labels: ' + name)
    source = validate_source(cfg['paths']['source_checkpoint'], labels, diagnostic=cfg['diagnostic'])
    weights = None
    if cfg['trainer'] == 's3b':
        from gym_trading_env.research.fg_s3b_normalization import audit_weights
        weights = audit_weights(cfg['paths']['weights'])
        if Path(weights['labels']).resolve() != Path(cfg['paths']['labels']):
            raise ValueError('Weights belong to a different S3-A label index')
        if not cfg['diagnostic'] and weights['summary']['training_admission'] != 'PASS':
            raise ValueError(weights['summary']['training_admission'])
    cache = None
    if Path(cfg['paths']['features']).exists() or feature_required:
        cache = FeatureCache(cfg['paths']['features'], source, cfg['paths']['labels'])
        if cache.manifest['seed'] != cfg['features']['seed']:
            raise ValueError('Frozen feature seed differs from experiment configuration')
        if cfg['trainer'] == 's3b':
            from embodied.run.fg_s3b_train import TrainingRows
            TrainingRows(cache, cfg['paths']['weights'], cfg['train']['mode'], diagnostic=source['diagnostic'])
    return labels, source, weights, cache


@contextmanager
def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield stream


def prepare(config):
    requested, cfg = read_experiment(config)
    activate(cfg)
    root = Path(cfg['prepared_dir'])
    if root.exists():
        raise FileExistsError('Prepared experiment already exists; choose a new experiment')
    if Path(cfg['train']['logdir']).exists():
        raise FileExistsError('Training output already exists; preparation cannot adopt another run')
    labels, source, weights, cache = admission(cfg)
    template_hash = digest(cfg['model']['template'])
    defaults, opt = resolve_defaults(cfg, source)
    if digest(cfg['model']['template']) != template_hash:
        raise ValueError('Shared model template changed during preparation')
    effective = copy.deepcopy(cfg)
    effective['train']['optimizer'] = opt
    inputs = dict(labels['source']['frozen_inputs'])
    inputs[str(Path(cfg['paths']['labels']) / 'manifest.json')] = digest(Path(cfg['paths']['labels']) / 'manifest.json')
    checkpoint = Path(source['path'])
    for p in checkpoint.iterdir():
        if p.is_file():
            inputs[str(p)] = digest(p)
    for name in ('single3_identity.json', 's3a_diagnostic_identity.json', 'training_summary.json'):
        p = checkpoint.parent.parent / name
        if p.is_file():
            inputs[str(p)] = digest(p)
    if weights:
        for name in ('manifest.json', 'normalization.json'):
            p = Path(cfg['paths']['weights']) / name
            inputs[str(p)] = digest(p)
        for name in ('manifest.json', 'scale.json'):
            p = Path(weights['base_weights']) / name
            inputs[str(p)] = digest(p)
        p = Path(weights['values']) / 'manifest.json'
        inputs[str(p)] = digest(p)
    if cache:
        p = cache.root / 'manifest.json'
        inputs[str(p)] = digest(p)
    code = runtime_files(cfg)
    from embodied.run.fg_s3a_checkpoint import package_versions
    with _lock(root.parent / ('.' + root.name + '.prepare.lock')):
        if root.exists():
            raise FileExistsError('Prepared experiment already exists')
        pending = root.parent / ('.' + root.name + '.pending_' + uuid.uuid4().hex)
        pending.mkdir()
        (pending / 'experiment.requested.yaml').write_bytes(requested.read_bytes())
        (pending / 'model.template.yaml').write_bytes(Path(cfg['model']['template']).read_bytes())
        (pending / 'source.config.yaml').write_bytes((checkpoint / 'config.yaml').read_bytes())
        for name, value in (('model.defaults.yaml', defaults), ('effective_config.yaml', effective)):
            (pending / name).write_text(yaml.safe_dump(value, sort_keys=False))
        write_json(pending / 'input_hashes.json', inputs)
        write_json(pending / 'code_hashes.json', code)
        receipt = signed(dict(protocol=PROTOCOL,
            status='READY_FOR_BC' if cache else 'READY_FOR_FEATURES',
            experiment=cfg['experiment'], trainer=cfg['trainer'], mode=cfg['train']['mode'],
            diagnostic=cfg['diagnostic'], requested_config=str(requested),
            requested_config_sha256=digest(requested), source_manifest_sha256=source['manifest_sha256'],
            source_identity_hash=source['identity_hash'], source_model_config_sha256=digest(checkpoint / 'config.yaml'),
            packages=package_versions(),
            source_versions={str(r): subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=r, text=True).strip()
                             for r in (GYM_ROOT, Path(cfg['dreamer_root']))},
            files={p.name: digest(p) for p in pending.iterdir()},
            formal_gpu_training='NOT_RUN', online_rl_training='NOT_RUN'))
        check_files(inputs)
        check_files(code)
        if (read_experiment(requested)[1] != cfg
                or digest(requested) != digest(pending / 'experiment.requested.yaml')
                or digest(cfg['model']['template']) != template_hash
                or digest(pending / 'model.template.yaml') != template_hash):
            raise ValueError('Experiment/template changed during preparation')
        write_json(pending / 'manifest.json', receipt)
        (pending / 'done').write_text('')
        pending.rename(root)
    return receipt


def require_prepared(prepared, config=None):
    root = Path(prepared).resolve()
    receipt = read_manifest(root / 'manifest.json')
    if (not (root / 'done').is_file() or receipt['protocol'] != PROTOCOL
            or receipt['status'] not in ('READY_FOR_BC', 'READY_FOR_FEATURES')
            or set(receipt['files']) != SNAPSHOT_FILES):
        raise ValueError('Incomplete or foreign prepared S3 experiment')
    check_files({str(root / name): h for name, h in receipt['files'].items()})
    cfg = yaml.safe_load((root / 'effective_config.yaml').read_text())
    _check_locations(cfg)
    if Path(cfg['prepared_dir']) != root or (cfg['experiment'], cfg['trainer'], cfg['train']['mode'], cfg['diagnostic']) != (
            receipt['experiment'], receipt['trainer'], receipt['mode'], receipt['diagnostic']):
        raise ValueError('Prepared S3 experiment location/mode differs')
    if config is not None:
        requested, current = read_experiment(config)
        if digest(requested) != receipt['requested_config_sha256'] or Path(current['prepared_dir']) != root:
            raise ValueError('Selected experiment YAML differs from its prepared snapshot')
    if runtime_files(cfg) != read_manifest_hashes(root / 'code_hashes.json'):
        raise ValueError('Prepared S3 runtime code/template changed')
    check_files(read_manifest_hashes(root / 'input_hashes.json'))
    from embodied.run.fg_s3a_checkpoint import package_versions
    if receipt['packages'] != package_versions():
        raise ValueError('Prepared S3 package versions changed')
    return receipt, cfg


def read_manifest_hashes(path):
    import json
    return json.loads(path.read_text())


def _bind_features(prepared, cache, *, create=True):
    root = Path(prepared)
    binding = signed(dict(prepared_manifest_sha256=digest(root / 'manifest.json'),
                          features=str(cache.root), feature_manifest_sha256=digest(cache.root / 'manifest.json')))
    target = root / 'features_binding.json'
    if target.exists():
        if read_manifest(target) != binding:
            raise ValueError('Prepared experiment feature binding changed')
    elif create:
        write_json(target, binding)
    return binding


def _training_identity(prepared, receipt, cfg, source, cache):
    run_id = cfg['experiment'] + '_' + receipt['content_hash'][:16]
    args = dict(seed=cfg['train']['seed'], run_id=run_id, batch_size=cfg['train']['batch_size'],
                opt_config=cfg['train']['optimizer'])
    if cfg['trainer'] == 's3a':
        from embodied.run.fg_s3a_train import training_identity
        return run_id, training_identity(source, cache, **args)
    from embodied.run.fg_s3b_train import TrainingRows, training_identity
    rows = TrainingRows(cache, cfg['paths']['weights'], cfg['train']['mode'], diagnostic=source['diagnostic'])
    return run_id, training_identity(source, cache, rows, cfg['paths']['weights'], **args,
        comparison_updates=cfg['train']['comparison_updates'], config_sha256=digest(Path(prepared) / 'effective_config.yaml'))


def execute_phase(prepared, phase, resume=False):
    # Called only by a separately configured child process.
    raw = yaml.safe_load((Path(prepared) / 'effective_config.yaml').read_text())
    activate(raw, raw['features' if phase == 'features' else 'train']['device'])
    receipt, cfg = require_prepared(prepared)
    _, source, _, cache = admission(cfg, feature_required=phase == 'train')
    if phase == 'features':
        if cache:
            return {'status': 'FEATURES_REUSED'}
        from embodied.run.fg_s3a_checkpoint import load_source
        from embodied.run.fg_s3a_features import build_features
        root = cfg['paths']['features']
        agent = load_source(source, root, device=cfg['features']['device'])
        return build_features(agent, source, cfg['paths']['labels'], root,
            seed=cfg['features']['seed'], run_id='features_' + str(cfg['features']['seed']),
            shard_rows=cfg['features']['shard_rows'], max_rows=cfg['features']['max_rows'])
    _bind_features(prepared, cache)
    run_id, _ = _training_identity(prepared, receipt, cfg, source, cache)
    args = dict(seed=cfg['train']['seed'], run_id=run_id, batch_size=cfg['train']['batch_size'],
        updates=cfg['train']['updates'], opt_config=cfg['train']['optimizer'],
        resume=resume, save_every=cfg['train']['save_every'], device=cfg['train']['device'])
    if cfg['trainer'] == 's3a':
        from embodied.run.fg_s3a_train import train
        return train(source, cache, cfg['train']['logdir'], **args)
    from embodied.run.fg_s3b_train import train
    return train(source, cache, cfg['paths']['weights'], cfg['train']['logdir'], **args,
                 mode=cfg['train']['mode'], comparison_updates=cfg['train']['comparison_updates'],
                 config_sha256=digest(Path(prepared) / 'effective_config.yaml'))


def _run_phase(prepared, cfg, phase, resume=False, *, lock_fds=()):
    logdir = Path(cfg['train']['logdir'])
    logs = logdir.parent / 'launch_logs'
    logs.mkdir(parents=True, exist_ok=True)
    attempt = uuid.uuid4().hex
    output = logs / (cfg['experiment'] + '_' + phase + '_' + attempt + '.stdout.log')
    argv = [sys.executable, str(GYM_ROOT / 'tools/fg_s3_experiment.py'), '_execute',
            '--prepared', str(prepared), '--phase', phase]
    if resume:
        argv.append('--resume-worker')
    env = dict(os.environ)
    device = cfg['features' if phase == 'features' else 'train']['device']
    env.update(PYTHONDONTWRITEBYTECODE='1', JAX_PLATFORMS='cpu' if device == 'cpu' else 'cuda',
               JAX_PLATFORM_NAME='cpu' if device == 'cpu' else 'gpu',
               PYTHONPATH=str(GYM_ROOT / 'src') + ':' + cfg['dreamer_root'])
    # The parent always prepares/audits on CPU; preserve the caller's selected
    # visible GPU separately rather than inheriting its CPU-only empty value.
    env['CUDA_VISIBLE_DEVICES'] = '' if device == 'cpu' else os.environ.get('FG_S3_VISIBLE_GPUS', '0')
    with output.open('x') as stream:
        process = subprocess.Popen(argv, cwd=GYM_ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   pass_fds=lock_fds)
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end='', flush=True)
        code = process.wait()
    attempts = Path(prepared) / 'attempts'
    attempts.mkdir(exist_ok=True)
    write_json(attempts / (attempt + '.json'), signed(dict(phase=phase, resume=resume,
        argv=argv, stdout=str(output), stdout_sha256=digest(output), exit_code=code,
        prepared_manifest_sha256=digest(Path(prepared) / 'manifest.json'))))
    if code:
        raise subprocess.CalledProcessError(code, argv)


def _save_run_snapshot(prepared, cfg, identity, *, create=False):
    root = Path(prepared)
    logdir = Path(cfg['train']['logdir'])
    if not logdir.exists():
        return
    if not (logdir / 'run_identity.json').is_file():
        raise ValueError('Managed logdir has no saved training identity')
    if read_manifest(logdir / 'run_identity.json')['identity'] != identity:
        raise ValueError('Managed logdir belongs to another training identity')
    dest = logdir / 'managed_snapshot'
    expected = dict(prepared_manifest_sha256=digest(root / 'manifest.json'),
                    feature_binding_sha256=digest(root / 'features_binding.json'))
    if dest.exists():
        saved = read_manifest(dest / 'manifest.json')
        if (saved['protocol'] != 'FG_S3_RUN_SNAPSHOT_V1' or saved['binding'] != expected
                or set(saved['files']) != SNAPSHOT_FILES | {'features_binding.json', 'preparation_receipt.json'}):
            raise ValueError('Managed run snapshot differs from prepared experiment')
        check_files({str(dest / p): h for p, h in saved['files'].items()})
        return
    if not create:
        raise ValueError('Managed run snapshot is missing')
    pending = logdir / ('.managed_snapshot.pending_' + uuid.uuid4().hex)
    pending.mkdir()
    names = list(read_manifest(root / 'manifest.json')['files']) + ['features_binding.json']
    for name in names:
        shutil.copyfile(root / name, pending / name)
    shutil.copyfile(root / 'manifest.json', pending / 'preparation_receipt.json')
    write_json(pending / 'manifest.json', signed(dict(protocol='FG_S3_RUN_SNAPSHOT_V1', binding=expected,
        files={p.name: digest(p) for p in pending.iterdir() if p.name != 'manifest.json'})))
    pending.rename(dest)


def _audit_checkpoint(cfg, identity):
    logdir = Path(cfg['train']['logdir'])
    name = (logdir / 'ckpt/latest').read_text().strip()
    if not name or Path(name).name != name:
        raise ValueError('Managed checkpoint pointer escaped logdir')
    if cfg['trainer'] == 's3a':
        from embodied.run.fg_s3a_checkpoint import audit_bc
    else:
        from embodied.run.fg_s3b_checkpoint import audit_bc
    return audit_bc(logdir / 'ckpt' / name, identity)


def start(prepared, *, resume=False, config=None):
    receipt, cfg = require_prepared(prepared, config)
    logdir = Path(cfg['train']['logdir'])
    with _lock(logdir.parent / ('.' + logdir.name + '.s3.lock')) as run_lock:
        if logdir.exists() != resume:
            raise ValueError('start requires a new logdir; resume requires an existing managed run')
        featuredir = Path(cfg['paths']['features'])
        if not featuredir.exists():
            if resume:
                raise ValueError('Resume cannot reconstruct a missing frozen feature cache')
            with _lock(featuredir.parent / ('.' + featuredir.name + '.s3.lock')) as feature_lock:
                if not featuredir.exists():
                    _run_phase(prepared, cfg, 'features',
                               lock_fds=(run_lock.fileno(), feature_lock.fileno()))
        _, source, _, cache = admission(cfg, feature_required=True)
        _bind_features(prepared, cache)
        _, identity = _training_identity(prepared, receipt, cfg, source, cache)
        if resume:
            if read_manifest(logdir / 'run_identity.json')['identity'] != identity:
                raise ValueError('Managed resume training identity differs')
            # A launcher crash may precede its final snapshot copy. Recover
            # only after the real checkpoint matches this frozen preparation.
            _audit_checkpoint(cfg, identity)
            _save_run_snapshot(prepared, cfg, identity, create=True)
        try:
            _run_phase(prepared, cfg, 'train', resume, lock_fds=(run_lock.fileno(),))
        finally:
            _save_run_snapshot(prepared, cfg, identity, create=True)
    return audit(prepared)


def audit(prepared, config=None):
    receipt, cfg = require_prepared(prepared, config)
    _, source, _, cache = admission(cfg)
    result = dict(status=receipt['status'], experiment=cfg['experiment'], trainer=cfg['trainer'],
                  mode=cfg['train']['mode'], diagnostic=cfg['diagnostic'],
                  prepared=str(Path(prepared).resolve()), logdir=cfg['train']['logdir'])
    if cache:
        _bind_features(prepared, cache, create=False)
        _, identity = _training_identity(prepared, receipt, cfg, source, cache)
        logdir = Path(cfg['train']['logdir'])
        if logdir.exists():
            _save_run_snapshot(prepared, cfg, identity)
            checked = _audit_checkpoint(cfg, identity)
            result.update(status='TRAINING_CHECKPOINT_VERIFIED', bc_updates=checked['bc_updates'],
                          frozen_keys=checked['freeze_audit']['frozen_keys'])
    return result
