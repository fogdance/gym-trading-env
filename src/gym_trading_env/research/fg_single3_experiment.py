"""Prepare user-started Single3 world-model pretraining from accepted data."""
from pathlib import Path
import copy
import os
import subprocess
import sys

import yaml

from .fg_replay import check_files, digest, read_manifest, signed, write_json


GYM_ROOT = Path(__file__).resolve().parents[3]
DREAMER_ROOT = GYM_ROOT.parent / 'dreamerv3'
ACCEPTED_ROOT = GYM_ROOT / 'artifacts/fg_single3_v1'
DEFAULT_SETTINGS = GYM_ROOT / 'configs/experiments/fg_single3/C1_M1_10K_then_online_1M.yaml'
SETTINGS_PATH = Path(os.environ.get('FG_SINGLE3_TRAINING_CONFIG',str(DEFAULT_SETTINGS))).resolve()
if str(DREAMER_ROOT) not in sys.path:
    sys.path.insert(0,str(DREAMER_ROOT))
from dreamerv3.configuration import read_config_yaml
SETTINGS = read_config_yaml(SETTINGS_PATH)
EXPERIMENT = SETTINGS['experiment']
UPDATES = SETTINGS['offline']['updates']
MILESTONES = tuple(SETTINGS['offline']['key_updates'])
DEFAULT_OUTPUT = GYM_ROOT / SETTINGS['prepared_dir']
DEFAULT_LOGDIR = Path(SETTINGS['offline']['logdir'])
ONLINE_LOGDIR = Path(SETTINGS['online']['logdir'])
DATASET = GYM_ROOT / SETTINGS['paths']['dataset']
COVERAGE = GYM_ROOT / SETTINGS['paths']['coverage']
HORIZON = GYM_ROOT / SETTINGS['paths']['horizon']
SOURCE_MANIFEST = GYM_ROOT / SETTINGS['paths']['dev_source_manifest']
NEURAL_SMOKE = GYM_ROOT / SETTINGS['paths']['neural_smoke']
NEURAL_SMOKE_LOG = NEURAL_SMOKE.with_suffix('.log')


def resolved_config(logdir=DEFAULT_LOGDIR):
    if str(DREAMER_ROOT) not in sys.path:
        sys.path.insert(0,str(DREAMER_ROOT))
    import elements
    from dreamerv3.main import _resolve_seed_protocol
    from dreamerv3.configuration import read_config_yaml
    definitions = read_config_yaml(DREAMER_ROOT / 'dreamerv3/configs.yaml')
    config = elements.Config(definitions[SETTINGS['dreamer_presets'][0]])
    for name in SETTINGS['dreamer_presets'][1:]:
        config = config.update(definitions[name])
    data = read_manifest(DATASET / 'manifest.json')
    inputs = read_manifest(DATASET / 'inputs.json')
    view = inputs['training_views']['C1']
    capacity = sum(data['traces'][i]['sequence_starts'] for i in data['pool_members']['C1_natural_oracle'])
    return _resolve_seed_protocol(config.update({
        'fg_single3_offline.profile': EXPERIMENT, 'run.steps': UPDATES,
        'run.fg_key_steps': MILESTONES,
        'run.fg_report_every_steps': SETTINGS['offline']['report_every_updates'],
        'run.fg_save_every_steps': SETTINGS['offline']['save_every_updates'],
        'logdir': str(logdir), 'fg_entry.manifest': view['path'],
        'env.gymnasium.config_path': view['rows'][0]['env_config'],
        'replay.size': capacity, 'run.replay_dir': str(DATASET / data['views']['C1_natural_oracle']),
        'fg_single3_offline.dataset': str(DATASET), 'fg_single3_offline.coverage': str(COVERAGE),
        'fg_single3_offline.horizon': str(HORIZON), 'fg_single3_offline.authorized': True}))


def source_files():
    """The launcher/evaluator as well as Agent/optimizer are experiment inputs."""
    gym = [SETTINGS_PATH, Path(__file__), Path(__file__).with_name('fg_single3_evaluation.py'),
        Path(__file__).with_name('fg_replay.py'), GYM_ROOT / 'tools/fg_single3_experiment.py',
        GYM_ROOT / 'tools/smoke_fg_single3_neural.py']
    dreamer = [DREAMER_ROOT / name for name in (
        'dreamerv3/agent.py','dreamerv3/rssm.py','dreamerv3/configs.yaml',
        'dreamerv3/fg_single3_offline.py','dreamerv3/main.py','dreamerv3/configuration.py',
        'embodied/jax/agent.py','embodied/jax/opt.py','embodied/jax/outs.py','embodied/run/offline_pretrain.py',
        'embodied/run/fg_single3_audit.py','embodied/run/fg_checkpoint.py',
        'embodied/run/action_mask_warmup.py','embodied/run/train.py','dreamerv3/fg_stage_transfer.py')]
    return gym + dreamer


def prepare_dev(directory):
    """Derive a Single3 DEV config; never open a TEST member or relabel TRAIN."""
    source = read_manifest(SOURCE_MANIFEST)
    roles = source['roles']
    if not roles['train'] or not roles['validation'] or max(roles['train']) >= min(roles['validation']):
        raise ValueError('DEV must be strictly after TRAIN')
    rows = [r for r in source['files'] if r['role'] == 'validation']
    if len(rows) != 1 or rows[0]['contract'] != 'CZCE.FG409' or rows[0]['eligible_for_training']:
        raise ValueError('Expected the agreed FG409 DEV member')
    row = copy.deepcopy(rows[0])
    inputs = read_manifest(DATASET / 'inputs.json')
    train_env = yaml.safe_load(Path(inputs['training_views']['C1']['rows'][0]['env_config']).read_text())
    master_path = Path(train_env['entry']['strategy_config'])
    master = yaml.safe_load(master_path.read_text())
    original = yaml.safe_load(Path(row['env_config']).read_text())
    env = copy.deepcopy(original)
    env['entry'].update(protocol='fg_single3_v1', strategy_config=str(master_path),
        contract_config=row['contract_config'], chunk_steps=1024)
    env['trading'].update(master['environment']['trading'])
    filename = directory / 'dev_env.yaml'
    filename.write_text(yaml.safe_dump(env,sort_keys=False))
    frozen = {str(SOURCE_MANIFEST):digest(SOURCE_MANIFEST), str(master_path):digest(master_path),
        row['path']:row['sha256'], row['env_config']:row['env_sha256'],
        row['contract_config']:row['contract_sha256'], str(filename):digest(filename)}
    check_files(frozen)
    row.update(env_config=str(filename), env_sha256=digest(filename))
    return signed(dict(role='DEV', contract='CZCE.FG409', rows=[row], frozen_inputs=frozen,
        train_days=roles['train'], dev_days=roles['validation'],
        train_last_timestamp=max(r['last_timestamp'] for r in source['files'] if r['role']=='train'),
        dev_first_timestamp=row['first_timestamp'], test_prices_accessed=False,
        oracle_labels_used=False, strategies=['always_wait','always_add','random_add','agent']))


def prepare(output=DEFAULT_OUTPUT, logdir=DEFAULT_LOGDIR):
    """CPU admission only; no neural initialization and no formal directory writes."""
    from .fg_replay import cpu_runtime
    cpu_runtime(DREAMER_ROOT)
    from dreamerv3.fg_single3_offline import validate
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preparation is immutable; choose a new output directory')
    config = resolved_config(logdir)
    checked = validate(config, persist=False)
    config = config.update({'fg_single3_offline.identity_hash':checked['identity_hash']})
    proof=read_manifest(NEURAL_SMOKE/'manifest.json')
    if (proof['status']!='REAL_NEURAL_CPU_SMOKE_PASS' or proof['formal_gradient_updates']!=0
            or proof['smoke_gradient_updates']!=2 or proof['batch_shape']!=[16,65]
            or not proof['exact_neural_restore'] or not proof['evaluation_policy_sync']
            or not proof['neural_policy_executed'] or not proof['report_executed']
            or proof['audit']['wm_effective_updates']<1 or proof['audit']['actor_effective_updates']!=0):
        raise ValueError('The full real neural CPU wiring proof is missing or incomplete')
    check_files(proof['neural_source_sha256'])
    check_files({str(NEURAL_SMOKE/p):h for p,h in proof['files'].items()})
    output.mkdir(parents=True)
    (output / 'config.yaml').write_text(yaml.safe_dump(dict(config),sort_keys=False))
    (output / 'online_env.yaml').write_text(yaml.safe_dump(dict(
        strategy_mode='entry_dataset',role='train',manifest_path=config.fg_entry.manifest)))
    online = online_config(output,config)
    (output / 'online_config.yaml').write_text(yaml.safe_dump(dict(online),sort_keys=False))
    dev = prepare_dev(output)
    write_json(output / 'dev_manifest.json',dev)
    frozen = {str(p):digest(p) for p in source_files()}
    frozen.update(proof['neural_source_sha256'])
    frozen.update({str(NEURAL_SMOKE/p):h for p,h in proof['files'].items()})
    frozen[str(NEURAL_SMOKE/'manifest.json')]=digest(NEURAL_SMOKE/'manifest.json')
    frozen[str(NEURAL_SMOKE_LOG)]=digest(NEURAL_SMOKE_LOG)
    for name in ('env/run_20261011_04','reward/run_20261011_02','oracle/run_20261011_01',
            'replay/run_20261011_01','coverage/run_20261011_01','horizon/run_20261011_01',
            'pilot/run_20261011_02'):
        path = ACCEPTED_ROOT / name / 'manifest.json'
        frozen[str(path)] = digest(path)
    frozen[str(ACCEPTED_ROOT / 'cpu_acceptance_run_20261011.json')] = digest(ACCEPTED_ROOT / 'cpu_acceptance_run_20261011.json')
    versions = {str(root):subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        for root in (GYM_ROOT,DREAMER_ROOT)}
    check_files(frozen)
    files = {str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}
    receipt = signed(dict(schema='FG_SINGLE3_OFFLINE_PREPARED_V1',status='CPU_ADMISSION_PASS',
        experiment=EXPERIMENT,formal_training_status='NOT_RUN',formal_gradient_updates=0,
        gpu_status='NOT_RUN',timing_status='NOT_ESTABLISHED',
        identity=checked,logdir=str(logdir),gradient_updates_budget=UPDATES,
        online_logdir=str(ONLINE_LOGDIR),online_env_steps_budget=SETTINGS['online']['env_steps'],
        phases=['offline_world_model_pretraining','online_training'],start_from_zero=True,
        training_positions_budget=UPDATES*16*64,milestones=list(MILESTONES),
        source_versions=versions,frozen_inputs=frozen,dev_manifest_sha256=digest(output/'dev_manifest.json'),
        training_settings_path=str(SETTINGS_PATH),
        neural_cpu_proof=str(NEURAL_SMOKE/'manifest.json'),
        argv=training_argv(output,config),online_argv=training_argv(output,online),files=files))
    # No PASS receipt is published before all output/input byte checks succeed.
    check_files({str(output/p):h for p,h in files.items()})
    write_json(output / 'manifest.json',receipt)
    return receipt


def require_prepared(directory):
    directory = Path(directory).resolve()
    receipt = read_manifest(directory / 'manifest.json')
    if receipt['experiment'] != EXPERIMENT or receipt['gradient_updates_budget'] != UPDATES:
        raise ValueError('Foreign experiment preparation')
    check_files(receipt['frozen_inputs'])
    check_files({str(directory/p):h for p,h in receipt['files'].items()})
    return receipt


def training_argv(directory, config):
    argv = [sys.executable,'-m','dreamerv3.main','--configs',*SETTINGS['dreamer_presets']]
    keys = ['logdir','fg_entry.manifest','env.gymnasium.config_path','replay.size',
            'fg_single3_offline.profile','run.steps','run.fg_key_steps',
            'run.replay_dir','fg_single3_offline.dataset','fg_single3_offline.coverage',
            'fg_single3_offline.horizon','fg_single3_offline.authorized','fg_single3_offline.identity_hash',
            'run.fg_report_every_steps','run.fg_save_every_steps']
    if config.script=='train':
        keys += ['script','task','fg_single3_offline.enabled','run.fg_single3_offline',
            'run.fg_single3_audit','run.env_checkpoint','run.fg_full_checkpoint',
            'run.fg_stage_transfer','run.fg_source_steps','run.fg_learning_audit',
            'run.fg_keep_all_checkpoints','replay.online']
    for key in keys:
        value = config
        for part in key.split('.'):
            value = value[part]
        argv += ['--'+key] + ([str(x) for x in value] if isinstance(value,(tuple,list)) else [str(value)])
    return argv


def online_config(directory, offline):
    directory=Path(directory).resolve()
    online=SETTINGS['online']
    return offline.update({'script':'train','task':'gymnasium_FGEntryDatasetEnv-v0',
        'logdir':str(ONLINE_LOGDIR),'env.gymnasium.config_path':str(directory/'online_env.yaml'),
        'fg_single3_offline.enabled':False,'run.fg_single3_offline':False,
        'run.fg_single3_audit':False,'run.steps':online['env_steps'],'run.env_checkpoint':True,
        'run.fg_full_checkpoint':True,'run.fg_stage_transfer':str(directory/'online_transfer.json'),
        'run.fg_source_steps':UPDATES,'run.fg_learning_audit':True,'run.replay_dir':'',
        'run.fg_report_every_steps':online['report_every_env_steps'],
        'run.fg_save_every_steps':online['save_every_env_steps'],
        'run.fg_key_steps':online['key_env_steps'],'run.fg_keep_all_checkpoints':False,
        'replay.online':True})
