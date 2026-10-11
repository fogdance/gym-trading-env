"""Prepare one user-started C1_M1_1M experiment from accepted immutable data."""
from pathlib import Path
import copy
import os
import shlex
import subprocess
import sys

import yaml

from .fg_replay import check_files, digest, read_manifest, signed, write_json


EXPERIMENT = 'C1_M1_1M'
UPDATES = 62500
MILESTONES = (0,2000,10000,25000,50000,62500)
GYM_ROOT = Path(__file__).resolve().parents[3]
DREAMER_ROOT = GYM_ROOT.parent / 'dreamerv3'
ACCEPTED_ROOT = GYM_ROOT / 'artifacts/fg_single3_v1'
DEFAULT_OUTPUT = ACCEPTED_ROOT / 'experiments/C1_M1_1M'
DEFAULT_LOGDIR = Path('/data/logdir/fg_single3/gate_margin_v2/C1_M1_1M')
DATASET = ACCEPTED_ROOT / 'replay/run_20261011_01'
COVERAGE = ACCEPTED_ROOT / 'coverage/run_20261011_01'
HORIZON = ACCEPTED_ROOT / 'horizon/run_20261011_01'
SOURCE_MANIFEST = GYM_ROOT / 'artifacts/fg_abcd_v1/v3_shared_prefix/FG409/all/manifest.json'
NEURAL_SMOKE = ACCEPTED_ROOT / 'experiments/C1_M1_1M_cpu_smoke_20261011_06'


def resolved_config(logdir=DEFAULT_LOGDIR):
    if str(DREAMER_ROOT) not in sys.path:
        sys.path.insert(0,str(DREAMER_ROOT))
    import elements
    from dreamerv3.main import _resolve_seed_protocol
    from dreamerv3.configuration import read_config_yaml
    definitions = read_config_yaml(DREAMER_ROOT / 'dreamerv3/configs.yaml')
    config = elements.Config(definitions['defaults']).update(definitions['fg_entry_v1'])
    config = config.update(definitions['fg_single3_joint_offline']).update(definitions['fg_single3_1m'])
    data = read_manifest(DATASET / 'manifest.json')
    inputs = read_manifest(DATASET / 'inputs.json')
    view = inputs['training_views']['C1']
    capacity = sum(data['traces'][i]['sequence_starts'] for i in data['pool_members']['C1_natural_oracle'])
    return _resolve_seed_protocol(config.update({
        'logdir': str(logdir), 'fg_entry.manifest': view['path'],
        'env.gymnasium.config_path': view['rows'][0]['env_config'],
        'replay.size': capacity, 'run.replay_dir': str(DATASET / data['views']['C1_natural_oracle']),
        'fg_single3_offline.dataset': str(DATASET), 'fg_single3_offline.coverage': str(COVERAGE),
        'fg_single3_offline.horizon': str(HORIZON), 'fg_single3_offline.authorized': True}))


def source_files():
    """The launcher/evaluator as well as Agent/optimizer are experiment inputs."""
    gym = [Path(__file__), Path(__file__).with_name('fg_single3_evaluation.py'),
        Path(__file__).with_name('fg_replay.py'), GYM_ROOT / 'tools/fg_single3_experiment.py',
        GYM_ROOT / 'tools/smoke_fg_single3_neural.py']
    dreamer = [DREAMER_ROOT / name for name in (
        'dreamerv3/agent.py','dreamerv3/rssm.py','dreamerv3/configs.yaml',
        'dreamerv3/fg_single3_offline.py','dreamerv3/main.py','dreamerv3/configuration.py',
        'embodied/jax/agent.py','embodied/jax/opt.py','embodied/run/offline_pretrain.py',
        'embodied/run/fg_single3_audit.py','embodied/run/fg_checkpoint.py',
        'embodied/run/action_mask_warmup.py')]
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
    dev = prepare_dev(output)
    write_json(output / 'dev_manifest.json',dev)
    frozen = {str(p):digest(p) for p in source_files()}
    frozen.update(proof['neural_source_sha256'])
    frozen.update({str(NEURAL_SMOKE/p):h for p,h in proof['files'].items()})
    frozen[str(NEURAL_SMOKE/'manifest.json')]=digest(NEURAL_SMOKE/'manifest.json')
    frozen[str(NEURAL_SMOKE.parent/'cpu_smoke_06.log')]=digest(NEURAL_SMOKE.parent/'cpu_smoke_06.log')
    for name in ('env/run_20261011_04','reward/run_20261011_02','oracle/run_20261011_01',
            'replay/run_20261011_01','coverage/run_20261011_01','horizon/run_20261011_01',
            'pilot/run_20261011_02'):
        path = ACCEPTED_ROOT / name / 'manifest.json'
        frozen[str(path)] = digest(path)
    frozen[str(ACCEPTED_ROOT / 'cpu_acceptance_run_20261011.json')] = digest(ACCEPTED_ROOT / 'cpu_acceptance_run_20261011.json')
    versions = {str(root):subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        for root in (GYM_ROOT,DREAMER_ROOT)}
    for command in ('start','resume','evaluate','report'):
        argv = [sys.executable,str(GYM_ROOT / 'tools/fg_single3_experiment.py'),command,'--prepared',str(output)]
        script = '#!/usr/bin/env bash\nset -euo pipefail\nexec ' + shlex.join(argv) + '\n'
        (output / (command + '.sh')).write_text(script)
    check_files(frozen)
    files = {str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}
    receipt = signed(dict(schema='FG_SINGLE3_C1_M1_1M_PREPARED_V1',status='CPU_ADMISSION_PASS',
        experiment=EXPERIMENT,formal_training_status='NOT_RUN',formal_gradient_updates=0,
        gpu_status='NOT_RUN',timing_status='NOT_ESTABLISHED',
        identity=checked,logdir=str(logdir),gradient_updates_budget=UPDATES,
        training_positions_budget=UPDATES*16*64,milestones=list(MILESTONES),
        source_versions=versions,frozen_inputs=frozen,dev_manifest_sha256=digest(output/'dev_manifest.json'),
        neural_cpu_proof=str(NEURAL_SMOKE/'manifest.json'),
        argv=training_argv(output,config),files=files))
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
    argv = [sys.executable,'-m','dreamerv3.main','--configs','defaults','fg_entry_v1',
        'fg_single3_joint_offline','fg_single3_1m']
    for key in ('logdir','fg_entry.manifest','env.gymnasium.config_path','replay.size',
            'run.replay_dir','fg_single3_offline.dataset','fg_single3_offline.coverage',
            'fg_single3_offline.horizon','fg_single3_offline.authorized','fg_single3_offline.identity_hash'):
        value = config
        for part in key.split('.'):
            value = value[part]
        argv += ['--'+key,str(value)]
    return argv
