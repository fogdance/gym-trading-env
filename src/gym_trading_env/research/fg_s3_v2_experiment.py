"""Immutable preparation and one entrypoint for bounded S3-V2 experiments."""
import copy
from pathlib import Path
import subprocess

from dreamerv3.configuration import read_config_yaml
from embodied.run.fg_checkpoint import plain, file_hash
from embodied.run.fg_s3_source_admission import inspect_source
from embodied.run.fg_s3_v2_train import execution_code
from .fg_replay import signed, write_json, read_manifest
from .fg_s3_oracle_state_query import load_train_query
from .fg_s3_teacher_catalog import load_catalog


PROJECT = Path(__file__).resolve().parents[3]
SCHEMA = 'FG_S3_V2_DIAGNOSTIC_CONFIG_V1'


def read_settings(path):
    path = Path(path)
    if not path.is_file():
        path = PROJECT/'configs/experiments/fg_s3_v2'/f'{path}.yaml'
    path = path.resolve()
    s = read_config_yaml(path)
    if s['schema'] != SCHEMA or s['mode'] not in ('a2', 'c2') or s['device'] != 'gpu':
        raise ValueError('S3-V2 requires an explicit A2/C2 GPU diagnostic config')
    for key in ('env_steps', 'collect_every', 'updates_per_collect', 'save_every', 'batch_size', 'replay_seed'):
        if type(s[key]) is not int or s[key] < (0 if key == 'replay_seed' else 1):
            raise ValueError('Invalid integer setting: '+key)
    if (not 65 <= s['env_steps'] <= 10000 or s['collect_every'] < 65
            or s['updates_per_collect'] * ((s['env_steps']+s['collect_every']-1)//s['collect_every']) > 1000):
        raise ValueError('V2 diagnostics are bounded to 65..10000 real steps and at most 1000 model updates')
    for key in ('beta', 'bc_lambda'):
        if isinstance(s[key], bool) or not isinstance(s[key], (int, float)):
            raise ValueError('Invalid numerical setting: '+key)
    import math
    if not math.isfinite(s['beta']) or not 0 <= s['beta'] <= 1 or not math.isfinite(s['bc_lambda']) or s['bc_lambda'] < 0:
        raise ValueError('Invalid beta or BC lambda')
    for key in ('dreamer_root', 'env_receipt', 'oracle_receipt', 'catalog', 'source_checkpoint', 'prepared_dir', 'run_dir', 'dev_source_manifest'):
        s[key] = str((PROJECT/s[key]).resolve())
    if Path(s['dreamer_root']) != PROJECT.parent/'dreamerv3':
        raise ValueError('Foreign Dreamer worktree')
    for key, subdir in (('prepared_dir', 'prepared'), ('run_dir', 'runs')):
        if not Path(s[key]).is_relative_to(PROJECT/'artifacts/fg_s3_v2'/subdir):
            raise ValueError('V2 output escaped isolated '+subdir)
    s.update(config_path=str(path), config_sha256=file_hash(path))
    return s


def prepare(config):
    s = read_settings(config)
    output = Path(s['prepared_dir'])
    if output.exists():
        raise FileExistsError('Preparation retained; use a new config/preparation name')
    catalog = load_catalog(s['catalog'])
    try:
        initial = inspect_source('direct_full', catalog=catalog, path=s['source_checkpoint'])
        target = copy.deepcopy(initial['target_config'])
        target['agent']['avail_threshold'] = .30
        spec = inspect_source('direct_full', catalog=catalog, path=s['source_checkpoint'],
            config=target, gate_protocol='FG_SINGLE3_MASK_GATE_V1')
        model = copy.deepcopy(spec['target_config'])
        model.update(batch_size=s['batch_size'], batch_length=64, report_length=64)
        query, env, _, _, _ = load_train_query(s['env_receipt'], s['oracle_receipt'], s['contract'])
        env.close()
        output.mkdir(parents=True, exist_ok=False)
        (output/'experiment.yaml').write_bytes(Path(s['config_path']).read_bytes())
        write_json(output/'settings.json', signed(dict(settings=s)))
        write_json(output/'model_config.json', signed(dict(config=plain(model))))
        write_json(output/'source_admission.json', spec)
        result = signed(dict(protocol='FG_S3_V2_PREPARED_V1', status='PREPARED_NO_TRAINING',
            code=execution_code(), oracle_identity=query.identity_hash,
            settings_path=str(output/'settings.json'), source_identity=spec['target_model_identity_hash'],
            versions={str(r): subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=r, text=True).strip()
                      for r in (PROJECT, PROJECT.parent/'dreamerv3')},
            dirty={str(r): subprocess.check_output(['git', 'status', '--porcelain'], cwd=r, text=True).splitlines()
                   for r in (PROJECT, PROJECT.parent/'dreamerv3')},
            files={p.name: file_hash(p) for p in output.iterdir() if p.is_file()},
            source_update_offset=spec['origin']['step'], new_environment_steps=0,
            diagnostic=True, budget_unit='environment_advances', formal_1m='NOT_RUN'))
        write_json(output/'manifest.json', result)
        return result
    finally:
        for a in catalog.arrays:
            if getattr(a, '_mmap', None) is not None:
                a._mmap.close()


def settings_from_prepared(prepared):
    root = Path(prepared).resolve()
    manifest = read_manifest(root/'manifest.json')
    from .fg_replay import check_files
    check_files({str(root/k): v for k, v in manifest['files'].items()})
    return read_manifest(root/'settings.json')['settings']
