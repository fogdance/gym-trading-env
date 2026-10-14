"""Immutable preparation and one entrypoint for bounded S3-V2 experiments."""
import copy
from pathlib import Path
import subprocess

from dreamerv3.configuration import read_config_yaml
from dreamerv3.fg_s3_composition import (CONFIG_SCHEMA, normalize_config, identity_fields,
    settings_composition, validate_records)
from embodied.run.fg_checkpoint import plain, file_hash
from embodied.run.fg_s3_source_admission import inspect_source, LiveOracleTraining
from embodied.run.fg_s3_v2_train import execution_code
from .fg_replay import signed, write_json, read_manifest
from .fg_s3_oracle_state_query import load_train_query
from .fg_s3_teacher_catalog import load_catalog
from .fg_s3_v2_beta import BetaSchedule
from dreamerv3.fg_s3_tutor_loss import DEFAULTS, agent_extensions


PROJECT = Path(__file__).resolve().parents[3]
SCHEMA = 'FG_S3_V2_DIAGNOSTIC_CONFIG_V1'
FORMAL_ROOT = Path('/data/logdir/fg_single3/s3_v2')


def require_training_authorization(settings, authorized):
    if settings['scope'] == 'formal' and not authorized:
        raise PermissionError('Formal S3-V2 training requires the explicit --authorize-training flag')


def read_settings(path):
    path = Path(path)
    if not path.is_file():
        path = PROJECT/'configs/experiments/fg_s3_v2'/f'{path}.yaml'
    path = path.resolve()
    s = normalize_config(read_config_yaml(path))
    return _validate_settings(s, path)


def _validate_settings(s, path):
    """Shared settings validation; public read_settings requires readiness."""
    if s.get('clock_protocol') != 'S3_MARKET_MINUTE':
        raise ValueError('New S3 settings require S3_MARKET_MINUTE and rebuilt inputs')
    if s['schema'] not in (SCHEMA, CONFIG_SCHEMA) or s['mode'] not in ('a2', 'c2', 'tutor', 'b2') or s['device'] != 'gpu':
        raise ValueError('S3-V2 requires an explicit A2/C2/Tutor/B2 GPU config')
    s.setdefault('scope', 'diagnostic')
    if s['mode'] == 'tutor':
        extension = agent_extensions(s)
        s.update({k: extension[k] for k in DEFAULTS})
    if s['mode'] == 'b2':
        extension = agent_extensions(s)
        s.update({k:v for k,v in extension.items() if k != 's3_v2_mode'})
        s.setdefault('catalog', None)
    s.setdefault('oracle_receipt_scope', 'aggregate')
    if s['oracle_receipt_scope'] not in ('aggregate', 'diagnostic_contract') or (
            s['oracle_receipt_scope'] == 'diagnostic_contract' and s['scope'] != 'diagnostic'):
        raise ValueError('Contract-only Oracle evidence is diagnostic-only')
    s.setdefault('contracts', None)
    if s['mode'] == 'b2' and s['contracts'] is not None and not s['catalog']:
        raise ValueError('B2 pooled training requires the shared TeacherCatalog source admission')
    s.setdefault('env_slice_bars', 1024)
    s.setdefault('replay_capacity', 2048 if s['scope'] == 'diagnostic' else 32768)
    s.setdefault('beta_schedule', None)
    if s['scope'] not in ('diagnostic', 'formal'):
        raise ValueError('Explicit diagnostic or formal scope required')
    for key in ('env_steps', 'collect_every', 'updates_per_collect', 'save_every', 'batch_size',
                'replay_seed', 'env_slice_bars', 'replay_capacity'):
        if type(s[key]) is not int or s[key] < (0 if key == 'replay_seed' else 1):
            raise ValueError('Invalid integer setting: '+key)
    if s['collect_every'] < 65 or s['replay_capacity'] < 65:
        raise ValueError('Collection and Replay must provide a complete causal sequence')
    if s['scope'] == 'diagnostic' and (not 65 <= s['env_steps'] <= 10000
            or s['updates_per_collect'] * ((s['env_steps']+s['collect_every']-1)//s['collect_every']) > 1000):
        raise ValueError('V2 diagnostics are bounded to 65..10000 real steps and at most 1000 model updates')
    if s['scope'] == 'formal' and (s['env_steps'] != 1000000 or s['contracts'] is None):
        raise ValueError('Formal S3-V2 requires exactly 1M actual Env advances and an explicit TRAIN pool')
    if s['contracts'] is not None and s['contracts'] != 'ALL_TRAIN' and (
            not isinstance(s['contracts'], list) or not s['contracts']
            or any(not isinstance(c, str) for c in s['contracts'])):
        raise ValueError('contracts must be ALL_TRAIN or an explicit nonempty list')
    for key in ('beta', 'bc_lambda'):
        if isinstance(s[key], bool) or not isinstance(s[key], (int, float)):
            raise ValueError('Invalid numerical setting: '+key)
    import math
    if not math.isfinite(s['beta']) or not 0 <= s['beta'] <= 1 or not math.isfinite(s['bc_lambda']) or s['bc_lambda'] < 0:
        raise ValueError('Invalid beta or BC lambda')
    schedule = BetaSchedule(s['beta'], s['beta_schedule'])
    if schedule.points is not None and schedule.points[-1][0] > s['env_steps']:
        raise ValueError('Beta schedule exceeds the frozen Env budget')
    is_e = bool(s.get('resolved_composition', {}).get('use_e'))
    if is_e:
        from dreamerv3.fg_s3_e_config import resolve_model_init
        if s.get('source_kind') != 'direct_init' or s.get('source_checkpoint') is not None:
            raise ValueError('E fresh-init requires direct_init without source_checkpoint')
        resolve_model_init(s.get('model_init'))
        if not s.get('catalog') or s['contracts'] is None:
            raise ValueError('E fresh-init requires an explicit TeacherCatalog TRAIN pool')
    paths = ('dreamer_root', 'env_receipt', 'oracle_receipt', 'catalog',
             'prepared_dir', 'run_dir', 'dev_source_manifest')
    paths += ('e_data_manifest',) if is_e else ('source_checkpoint',)
    for key in paths:
        if key == 'catalog' and s['mode'] == 'b2' and s[key] is None:
            continue
        s[key] = str((PROJECT/s[key]).resolve())
    if Path(s['dreamer_root']) != PROJECT.parent/'dreamerv3':
        raise ValueError('Foreign Dreamer worktree')
    if not Path(s['prepared_dir']).is_relative_to(PROJECT/'artifacts/fg_s3_v2/prepared'):
        raise ValueError('Preparation escaped isolated artifacts')
    allowed = FORMAL_ROOT if s['scope'] == 'formal' else PROJECT/'artifacts/fg_s3_v2/runs'
    if not Path(s['run_dir']).is_relative_to(allowed):
        raise ValueError('Run escaped its isolated '+s['scope']+' directory')
    s.update(config_path=str(path), config_sha256=file_hash(path))
    return s


def prepare(config):
    s = read_settings(config)
    output = Path(s['prepared_dir'])
    if output.exists():
        raise FileExistsError('Preparation retained; use a new config/preparation name')
    query, env = None, None
    if s['mode'] == 'b2' and s['contracts'] is None:
        query, env, _, _, candidates = load_train_query(s['env_receipt'], s['oracle_receipt'],
            s['contract'], receipt_scope=s['oracle_receipt_scope'])
        catalog = None
        binding = LiveOracleTraining(query, env)
        admission_args = dict(live_oracle=binding, diagnostic=True, agent_extension=agent_extensions(s))
    else:
        catalog = load_catalog(s['catalog'])
        admission_args = dict(catalog=catalog)
        if s['mode'] == 'b2':
            admission_args['agent_extension'] = agent_extensions(s)
    composition = settings_composition(s)
    if composition is not None:
        admission_args['composition'] = composition
    try:
        kind = s.get('source_kind', 'warm_start')
        if composition is not None and composition['use_e']:
            from dreamerv3.fg_s3_e_config import resolve_model_init
            from embodied.run.fg_s3_e_source import make_binding
            target = resolve_model_init(s['model_init'])
            binding = make_binding(target, composition, s['e_data_manifest'])
            spec = inspect_source(kind, seed=target['seed'], config=target,
                diagnostic=s['scope'] == 'diagnostic', e_binding=binding,
                gate_protocol='FG_SINGLE3_MASK_GATE_V1', **admission_args)
        else:
            initial = inspect_source(kind, path=s['source_checkpoint'], **admission_args)
            target = copy.deepcopy(initial['target_config'])
            target['agent']['avail_threshold'] = .30
            spec = inspect_source(kind, path=s['source_checkpoint'],
                config=target, gate_protocol='FG_SINGLE3_MASK_GATE_V1', **admission_args)
        model = copy.deepcopy(spec['target_config'])
        model.update(batch_size=s['batch_size'], batch_length=64, report_length=64)
        output.mkdir(parents=True, exist_ok=False)
        if s['contracts'] is None:
            if query is None:
                query, env, _, _, _ = load_train_query(s['env_receipt'], s['oracle_receipt'], s['contract'],
                                                    receipt_scope=s['oracle_receipt_scope'])
            if s['mode'] == 'b2':
                from .fg_s3_b2_outcome_catalog import OutcomeCatalog
                OutcomeCatalog(query, env, candidates).write_receipt(output/'outcome_catalog.json')
            env.close()
        else:
            from .fg_s3_v2_train_pool import prepare_pool
            query = prepare_pool(s, output)
        (output/'experiment.yaml').write_bytes(Path(s['config_path']).read_bytes())
        write_json(output/'settings.json', signed(dict(settings=s)))
        frozen_model = dict(config=plain(model), **identity_fields(s))
        if 'e_binding' in spec:
            frozen_model['e_binding'] = spec['e_binding']
        if s['mode'] in ('tutor', 'b2'):
            frozen_model['agent_extension'] = agent_extensions(s)
        write_json(output/'model_config.json', signed(frozen_model))
        write_json(output/'source_admission.json', spec)
        result = signed(dict(protocol='FG_S3_V2_PREPARED_V1', status='PREPARED_NO_TRAINING',
            **({'e_binding': spec['e_binding']} if 'e_binding' in spec else {}),
            code=execution_code(), oracle_identity=query.identity_hash,
            **identity_fields(s),
            settings_path=str(output/'settings.json'), source_identity=spec['target_model_identity_hash'],
            versions={str(r): subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=r, text=True).strip()
                      for r in (PROJECT, PROJECT.parent/'dreamerv3')},
            dirty={str(r): subprocess.check_output(['git', 'status', '--porcelain'], cwd=r, text=True).splitlines()
                   for r in (PROJECT, PROJECT.parent/'dreamerv3')},
            files={p.name: file_hash(p) for p in output.iterdir() if p.is_file()},
            source_update_offset=spec['target_update_offset'], source_model_updates=spec['origin']['step'],
            clock_protocol='S3_MARKET_MINUTE', new_environment_steps=0,
            training_identity=dict(candidate='C1', pool='C1_live_dagger' if catalog is None else 'C1_natural_oracle',
                teacher_source_kind=spec.get('teacher_source_kind', 'catalog'),
                dataset_sha256=spec['dataset_sha256'], reward_params_hash=spec['reward_params_hash'],
                observation_identity=spec['observation_identity'],
                training_hash=(binding if catalog is None else catalog).manifest['source']['training_hash']),
            diagnostic=s['scope'] == 'diagnostic', budget_unit='environment_advances', formal_1m='NOT_RUN'))
        write_json(output/'manifest.json', result)
        return result
    finally:
        if env is not None:
            env.close()
        if catalog is not None:
            for a in catalog.arrays:
                if getattr(a, '_mmap', None) is not None:
                    a._mmap.close()


def settings_from_prepared(prepared):
    root = Path(prepared).resolve()
    manifest = read_manifest(root/'manifest.json')
    settings = read_manifest(root/'settings.json')['settings']
    settings_composition(settings)
    from .fg_replay import check_files
    check_files({str(root/k): v for k, v in manifest['files'].items()})
    validate_records(settings, manifest)
    if settings_composition(settings) is not None:
        model = read_manifest(root/'model_config.json')
        source = read_manifest(root/'source_admission.json')
        validate_records(settings, model, source, source['target_model_identity'])
        if settings['resolved_composition']['use_e']:
            from embodied.run.fg_s3_e_source import validate_e_records
            validate_e_records(manifest, model, source, source['target_model_identity'])
    return settings
