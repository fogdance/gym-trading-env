"""Explicit full-state continuation under a new Availability decision threshold.

This is neither an offline-to-online weight transfer nor a relaxation of exact
FullCheckpoint identity checks. Only the threshold and reviewed diagnostic /
parameter publication wiring can differ from the frozen online source.
"""
import copy
import contextlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

from .fg_replay import check_files, digest, read_manifest, signed, write_json

GYM_ROOT = Path(__file__).resolve().parents[3]
DREAMER_ROOT = GYM_ROOT.parent / 'dreamerv3'
if str(DREAMER_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMER_ROOT))

SCHEMA = 'FG_SINGLE3_GATE_CALIBRATION_V1'
RECEIPT_NAME = 'gate_source_transfer.json'
ALLOWED_CODE_PATHS = {
    'dreamerv3/dreamerv3/agent.py',
    'dreamerv3/embodied/run/train.py',
    'dreamerv3/embodied/run/action_mask_warmup.py',
    'dreamerv3/embodied/jax/agent.py',
    'dreamerv3/dreamerv3/fg_stage_transfer.py',
}


def code_hashes():
    from embodied.run.fg_checkpoint import file_hash
    result = {}
    for root, names in ((DREAMER_ROOT, ('embodied', 'dreamerv3')),
            (GYM_ROOT / 'src/gym_trading_env', ('envs', 'utils'))):
        for name in names:
            for path in sorted((root / name).rglob('*.py')):
                if 'tests' not in path.parts:
                    result[f'{root.name}/{path.relative_to(root)}'] = file_hash(path)
    return result


def check_config_delta(source, target):
    """Only run location and the one probability threshold may change."""
    from embodied.run.fg_checkpoint import plain
    from dreamerv3.agent import validate_availability_threshold
    old, new = plain(source), plain(target)
    validate_availability_threshold(new['agent']['avail_threshold'])
    if Path(old['logdir']).resolve() == Path(new['logdir']).resolve():
        raise ValueError('Gate calibration requires an independent experiment directory')
    new['logdir'] = old['logdir']
    new['agent']['avail_threshold'] = old['agent']['avail_threshold']
    # Derived by main after admission, rather than a trainable setting.
    new.pop('walk_forward_guard_result', None)
    old.pop('walk_forward_guard_result', None)
    if old != new:
        raise ValueError('Gate calibration forbids other configuration changes')


def calibrated_runtime(runtime, warmup):
    """Remove only old-rule observations; retain financial and random state."""
    runtime, warmup = copy.deepcopy(runtime), copy.deepcopy(warmup)
    if warmup['actor_enabled']:
        raise ValueError('This migration is restricted to the stopped NOT_READY source')
    runtime['samples'] = 0
    runtime['aggregates']['warmup'] = {}
    # The ordinary loss logger can otherwise emit old-threshold statistics.
    runtime['aggregates']['train'] = {k:v for k,v in runtime['aggregates']['train'].items()
        if not k.startswith('train/avail/')}
    warmup.update(consecutive_passes=0, consecutive_regressions=0,
        last_passed=False, last_add_support_sufficient=False)
    return runtime, warmup


def artifact_analysis(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    check_files({str(directory/name):h for name,h in manifest['files'].items()})
    result = json.loads((directory / 'analysis.json').read_text())
    if result['status'] != 'READ_ONLY_PREFLIGHT_COMPLETE' or not result['source_unchanged']:
        raise ValueError('Frozen preflight did not complete')
    return result


def preflight_decision(settings, source_meta):
    from embodied.run.action_mask_warmup import METRICS, ADD_RECALL_METRICS
    t = float(settings['threshold'])
    scan_dir = GYM_ROOT / settings['preflight']['scan']
    windows_dir = GYM_ROOT / settings['preflight']['windows']
    scan, windows = artifact_analysis(scan_dir), artifact_analysis(windows_dir)
    for artifact in (scan, windows):
        if artifact['source_manifest_sha256'] != digest(Path(settings['source_checkpoint'])/'manifest.json'):
            raise ValueError('Preflight used a different source checkpoint')
        if artifact['source'] != str(Path(settings['source_checkpoint'])):
            raise ValueError('Preflight source path changed')
        if artifact['training_modified'] or artifact['dev_test_accessed']:
            raise ValueError('Preflight accessed forbidden data or modified training')
    reasons = []
    candidate = scan['candidates'][str(t)]
    values = {}
    for head in ('post','prior'):
        values.update({f'avail/{head}_{k}':candidate['metrics'][head][name]
            for k,name in (('accuracy','accuracy'),('exact_accuracy','exact_accuracy'),
                ('fpr','fpr'),('add_recall','add_recall'))})
    values['avail/img_fallback_rate'] = candidate['imagination']['all']['fallback_rate']
    quality = scan['actual_quality_thresholds']
    for name,(key,mode) in {**METRICS,**ADD_RECALL_METRICS}.items():
        value = values.get(key)
        if value is None or not math.isfinite(value) or (value < quality[name] if mode=='min' else value > quality[name]):
            reasons.append('full_sample_'+name)
    complete = windows.get('imagination_threshold') == t
    if not complete:
        reasons.append('candidate_window_imagination_not_run')
    records = windows['windows']
    from dreamerv3.configuration import read_config_yaml
    from embodied.run.action_mask_warmup import ActionMaskWarmup
    import elements
    source_config=read_config_yaml(Path(settings['source_checkpoint'])/'config.yaml')
    gate=ActionMaskWarmup.for_single3_1m(elements.Config(source_config['action_mask_warmup']))
    if quality!=gate.opening_thresholds or windows['actual_quality_thresholds']!=quality:
        raise ValueError('Preflight changed the original quality thresholds')
    longest=0
    for r in records:
        evaluated=r['candidates'][str(t)]
        if not gate.ready(r['positions']):
            reasons.append('window_train_sample_support_insufficient')
        gate.update(evaluated['metrics'],r['report_step'],allow_activation=False)
        if gate.last_passed!=evaluated['passed'] or gate.consecutive_passes!=evaluated['consecutive_passes']:
            raise ValueError('Recorded Gate streak disagrees with the original Gate evaluator')
        longest=max(longest,gate.consecutive_passes)
    if longest < 20:
        reasons.append('twenty_consecutive_full_windows_not_passed')
    if any(r['candidates'][str(t)]['metrics'].get('avail/img_action_invalid_rate',0) != 0 for r in records):
        reasons.append('imagined_action_invalid')
    return dict(status='GO' if not reasons else 'NO_GO', threshold=t, reasons=reasons,
        full_window_max_streak=longest, windows=len(records), source_step=source_meta['step'],
        scan=str(scan_dir), windows_artifact=str(windows_dir),
        input_hashes={str(d/'manifest.json'):digest(d/'manifest.json') for d in (scan_dir,windows_dir)})


def require_cpu_certificate(directory, receipt):
    proof=read_manifest(Path(directory)/'cpu_restore_proof.json')
    check_files(proof['input_hashes'])
    if (proof['status']!='FULL_GATE_MIGRATION_CPU_RESTORE_PASS'
            or proof['receipt_sha256']!=digest(Path(directory)/RECEIPT_NAME)
            or proof['target_identity']!=receipt['target_identity']):
        raise ValueError('Full Gate migration CPU certificate does not match the prepared experiment')
    return proof


def require_formal_admission(config):
    """Also protect direct official CLI launches, before GPU initialization."""
    receipt=validate_branch(config)
    if receipt['preflight']['status']!='GO':
        raise RuntimeError('Gate CPU preflight is NO_GO')
    directory=Path(receipt['migrated_checkpoint']).parents[1]
    require_cpu_certificate(directory,receipt)
    from embodied.run.fg_checkpoint import digest as state_digest
    if os.environ.get('FG_SINGLE3_GATE_AUTHORIZED_IDENTITY')!=state_digest(receipt['target_identity']):
        raise PermissionError('Gate GPU continuation requires the explicitly authorized shared runner')
    return receipt


def validate_branch(config, receipt_path=None):
    """Used by official Dreamer admission before model/device initialization."""
    from dreamerv3.configuration import read_config_yaml
    from embodied.run.fg_checkpoint import validate_bundle, provenance, digest as state_digest
    path = Path(receipt_path or (Path(config.logdir)/RECEIPT_NAME))
    receipt = read_manifest(path)
    if receipt['schema'] != SCHEMA or receipt['gate_threshold_version'] != 2:
        raise ValueError('Unsupported Gate continuation receipt')
    check_files(receipt['frozen_inputs'])
    source = Path(receipt['source_checkpoint'])
    if digest(source/'manifest.json') != receipt['source_manifest_sha256']:
        raise ValueError('Gate source checkpoint identity changed')
    old_meta = validate_bundle(source)
    old = read_config_yaml(source/'config.yaml')
    check_config_delta(old,config)
    if float(config.agent.avail_threshold) != receipt['new_threshold'] or str(config.logdir) != receipt['target_logdir']:
        raise ValueError('Gate target threshold or location mismatch')
    actual = provenance(config)
    if actual != receipt['target_identity']:
        raise ValueError('Gate target runtime identity changed')
    if old_meta['identity'] != receipt['source_identity']:
        raise ValueError('Gate source runtime identity changed')
    if state_digest(receipt['source_code_hashes']) != old_meta['identity']['code_hash']:
        raise ValueError('Gate source code provenance mismatch')
    current = code_hashes()
    if set(current) != set(receipt['source_code_hashes']):
        raise ValueError('Gate migration cannot add or remove training modules')
    changed = {p for p,h in current.items() if h != receipt['source_code_hashes'][p]}
    if not changed <= ALLOWED_CODE_PATHS:
        raise ValueError('Gate migration contains an unapproved training code change')
    if current != receipt['target_code_hashes']:
        raise ValueError('Gate migration code changed after preparation')
    return receipt


def prepare(config_path, output=None):
    """Create an immutable CPU recovery package; no formal logdir/GPU writes."""
    from dreamerv3.configuration import read_config_yaml
    from embodied.run.fg_checkpoint import validate_bundle, provenance, digest as state_digest
    import elements
    import yaml
    config_path = Path(config_path).resolve()
    settings = read_config_yaml(config_path)
    if settings['schema'] != SCHEMA:
        raise ValueError('Expected Gate calibration configuration')
    source = Path(settings['source_checkpoint']).resolve()
    meta = validate_bundle(source)
    if meta['step'] != 180000:
        raise ValueError('This approved calibration requires the frozen online 180k node')
    before = digest(source/'manifest.json')
    old = read_config_yaml(source/'config.yaml')
    target = elements.Config(old).update(logdir=settings['logdir'],
        **{'agent.avail_threshold':settings['threshold']})
    check_config_delta(old,target)
    target_root = Path(target.logdir)
    if target_root.exists():
        raise FileExistsError('Gate target experiment already exists; never overwrite it')
    source_code_path = GYM_ROOT/settings['source_code_hashes']
    old_code = json.loads(source_code_path.read_text())
    new_code = code_hashes()
    if state_digest(old_code) != meta['identity']['code_hash']:
        raise ValueError('Frozen source code map does not match checkpoint provenance')
    changed = {p for p,h in new_code.items() if old_code.get(p) != h}
    if set(old_code)!=set(new_code) or not changed <= ALLOWED_CODE_PATHS:
        raise ValueError('Unapproved code change in Gate calibration branch')
    identity = provenance(target)
    decision = preflight_decision(settings,meta)
    out = Path(output or (GYM_ROOT/settings['prepared_dir'])).resolve()
    if out.exists():
        raise FileExistsError('Gate preparation is immutable; choose a new experiment')
    out.mkdir(parents=True)
    try:
        (out/'config.yaml').write_text(yaml.safe_dump(dict(target),sort_keys=False))
        original_prepared = GYM_ROOT/settings['source_prepared']
        original = read_manifest(original_prepared/'manifest.json')
        argv = list(original['online_argv'])
        argv[argv.index('--logdir')+1] = str(target.logdir)
        argv += ['--agent.avail_threshold',str(settings['threshold'])]
        # Resolve through the official parser; never hand-parse CLI scalar types.
        from dreamerv3.main import _resolve_seed_protocol
        definitions=read_config_yaml(DREAMER_ROOT/'dreamerv3/configs.yaml')
        parsed,other=elements.Flags(configs=['defaults']).parse_known(argv[3:])
        resolved=elements.Config(definitions['defaults'])
        for name in parsed.configs:
            resolved=resolved.update(definitions[name])
        resolved=_resolve_seed_protocol(elements.Flags(resolved).parse(other))
        check_config_delta(old,resolved)
        if str(resolved.logdir) != str(target.logdir) or resolved.agent.avail_threshold != target.agent.avail_threshold:
            raise ValueError('Formal CLI does not reproduce the calibrated configuration')
        cp = out/'ckpt'/source.name
        cp.parent.mkdir()
        shutil.copytree(source,cp)
        shutil.copyfile(out/'config.yaml',cp/'config.yaml')
        runtime=pickle.loads((source/'runtime.pkl').read_bytes())
        warmup=pickle.loads((source/'action_mask_warmup.pkl').read_bytes())
        updated_runtime,updated_warmup=calibrated_runtime(runtime,warmup)
        import cloudpickle
        (cp/'runtime.pkl').write_bytes(cloudpickle.dumps(updated_runtime,protocol=5))
        (cp/'action_mask_warmup.pkl').write_bytes(pickle.dumps(updated_warmup,protocol=5))
        changed_files=('config.yaml','runtime.pkl','action_mask_warmup.pkl')
        target_meta=copy.deepcopy(meta)
        target_meta.update(identity=identity,protected=True)
        for name in changed_files:
            target_meta['files'][name]=digest(cp/name)
        write_json(cp/'manifest.json',target_meta)
        validate_bundle(cp,identity)
        (cp.parent/'latest').write_text(source.name)
        assert all(digest(cp/name)==sha for name,sha in meta['files'].items() if name not in changed_files)
        frozen={str(p):digest(p) for p in (config_path,source_code_path,
            original_prepared/'manifest.json',Path(__file__),GYM_ROOT/'tools/fg_single3_experiment.py',
            DREAMER_ROOT/'dreamerv3/fg_stage_transfer.py',DREAMER_ROOT/'dreamerv3/configs.yaml',
            Path(old['run']['fg_stage_transfer']),out/'config.yaml')}
        frozen.update(decision['input_hashes'])
        receipt=signed(dict(schema=SCHEMA,status='FULL_STATE_GATE_BRANCH_PREPARED',
            experiment=settings['experiment'],target_logdir=str(target.logdir),
            source_checkpoint=str(source),source_manifest_sha256=before,source_identity=meta['identity'],
            target_identity=identity,source_code_hashes=old_code,target_code_hashes=new_code,
            changed_code=sorted(changed),changed_checkpoint_files=list(changed_files),
            old_threshold=old['agent']['avail_threshold'],new_threshold=settings['threshold'],gate_threshold_version=2,
            inherited_env_steps=meta['step'],source_gate_state=warmup,target_gate_state=updated_warmup,
            source_counters=pickle.loads((source/'agent.pkl').read_bytes())['counters'],
            preflight=decision,formal_training_status='NOT_RUN',formal_training_authorized=False,
            gpu_status='NOT_RUN',dev_test_accessed=False,frozen_inputs=frozen,argv=argv,
            source_versions={str(root):subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
                for root in (GYM_ROOT,DREAMER_ROOT)},
            migrated_checkpoint=str(cp),migrated_manifest_sha256=digest(cp/'manifest.json')))
        write_json(out/RECEIPT_NAME,receipt)
        validate_branch(target,out/RECEIPT_NAME)
        assert digest(source/'manifest.json')==before
        validate_bundle(source)
        write_json(out/'manifest.json',signed(dict(status=receipt['status'],receipt_sha256=digest(out/RECEIPT_NAME),
            config_sha256=digest(out/'config.yaml'),checkpoint_manifest_sha256=digest(cp/'manifest.json'))))
        return receipt
    except BaseException as exc:
        write_json(out/'failure.json',dict(status='FAILED',error_type=type(exc).__name__,error=str(exc)))
        raise


def require_prepared(directory):
    from dreamerv3.configuration import read_config_yaml
    from embodied.run.fg_checkpoint import validate_bundle
    import elements
    directory=Path(directory).resolve()
    manifest=read_manifest(directory/'manifest.json')
    check_files({str(directory/RECEIPT_NAME):manifest['receipt_sha256'],
        str(directory/'config.yaml'):manifest['config_sha256']})
    config=elements.Config(read_config_yaml(directory/'config.yaml'))
    receipt=validate_branch(config,directory/RECEIPT_NAME)
    checkpoint=Path(receipt['migrated_checkpoint'])
    if digest(checkpoint/'manifest.json') != manifest['checkpoint_manifest_sha256']:
        raise ValueError('Prepared Gate checkpoint changed')
    validate_bundle(checkpoint,receipt['target_identity'])
    return config,receipt


def publish(directory):
    """Install the already validated recovery package into its new logdir."""
    config,receipt=require_prepared(directory)
    target=Path(config.logdir)
    if target.exists():
        raise FileExistsError('Gate formal directory already exists')
    source=Path(directory)
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_name('.pending_'+target.name)
    temp.mkdir(exist_ok=False)
    try:
        for name in ('config.yaml',RECEIPT_NAME):
            shutil.copyfile(source/name,temp/name)
        shutil.copytree(source/'ckpt',temp/'ckpt')
        from embodied.run.fg_checkpoint import validate_bundle
        validate_bundle(temp/'ckpt'/Path(receipt['migrated_checkpoint']).name,receipt['target_identity'])
        temp.rename(target)
    except BaseException as exc:
        write_json(temp/'failure.json',dict(status='FAILED',error=str(exc)))
        raise
    return target


def certify_restore(directory, source_proof, target_proof):
    """Attach actual two-path CPU restoration evidence after publication."""
    config,receipt=require_prepared(directory)
    source_proof,target_proof=Path(source_proof).resolve(),Path(target_proof).resolve()
    source=json.loads(source_proof.read_text())
    target=json.loads(target_proof.read_text())
    for result in (source,target):
        if (result['status']!='REAL_NEURAL_FULL_CPU_RESTORE_PASS'
                or result['gradient_updates_performed'] or result['dev_test_accessed']):
            raise ValueError('Actual read-only neural restoration proof is missing')
    expected={receipt['migrated_checkpoint'],str(Path(config.logdir)/'ckpt'/Path(receipt['migrated_checkpoint']).name)}
    if {r['path'] for r in target['results']} != expected:
        raise ValueError('Gate restoration proof must cover both independent recovery paths')
    original=next(r for r in source['results'] if r['path']==receipt['source_checkpoint'])
    for r in target['results']:
        if (r['identity']!=receipt['target_identity'] or r['step']!=180000
                or r['agent_counters']!=receipt['source_counters'] or r['actor_enabled']
                or r['gate']['consecutive_passes']!=0):
            raise ValueError('Migrated neural restoration has incorrect identity, counter or Gate')
        for key in ('agent_semantic_sha256','financial_account_sha256','parameter_groups'):
            if r[key]!=original[key]:
                raise ValueError('Gate migration failed to preserve '+key)
        if not all(r[k] for k in ('replay_arrays_equal','replay_sampler_rng_equal',
                'python_rng_equal','numpy_rng_equal','prefetch_restored')):
            raise ValueError('Gate migration failed full RNG/Replay restoration')
    proof=signed(dict(status='FULL_GATE_MIGRATION_CPU_RESTORE_PASS',
        target_identity=receipt['target_identity'],receipt_sha256=digest(Path(directory)/RECEIPT_NAME),
        input_hashes={str(p):digest(p) for p in (source_proof,target_proof)},
        recovery_paths=sorted(expected),gradient_updates_performed=0,gpu_training_started=False))
    output=Path(directory)/'cpu_restore_proof.json'
    if output.exists():
        raise FileExistsError('CPU restoration certificate is immutable')
    write_json(output,proof)
    return proof


@contextlib.contextmanager
def authorized_identity(identity):
    from embodied.run.fg_checkpoint import digest as state_digest
    # Authorization exists only for this requested launch, never in frozen YAML.
    previous_authorization=os.environ.get('FG_SINGLE3_GATE_AUTHORIZED_IDENTITY')
    os.environ['FG_SINGLE3_GATE_AUTHORIZED_IDENTITY']=state_digest(identity)
    try:
        yield
    finally:
        if previous_authorization is None:
            os.environ.pop('FG_SINGLE3_GATE_AUTHORIZED_IDENTITY',None)
        else:
            os.environ['FG_SINGLE3_GATE_AUTHORIZED_IDENTITY']=previous_authorization


def resume(directory, *, authorize_training=False):
    if not authorize_training:
        raise PermissionError('Formal GPU training is not authorized; explicitly pass --authorize-training after approval')
    config,receipt=require_prepared(directory)
    if receipt['preflight']['status'] != 'GO':
        raise RuntimeError('Gate preflight NO_GO: '+', '.join(receipt['preflight']['reasons']))
    require_cpu_certificate(directory,receipt)
    if not Path(config.logdir).exists():
        raise RuntimeError('Publish and CPU-verify the full recovery node before formal resume')
    with authorized_identity(receipt['target_identity']):
        return _run_authorized(config,receipt)


def _run_authorized(config,receipt):
    from embodied.run.fg_checkpoint import FullCheckpoint
    from dreamerv3.fg_entry_guard import validate_fg_entry
    validate_fg_entry(config)
    FullCheckpoint(Path(config.logdir)/'ckpt',None,receipt['target_identity'],()).latest()
    # Same shared training process/logging implementation as ordinary Single3.
    import importlib.util
    spec=importlib.util.spec_from_file_location('fg_single3_runner',GYM_ROOT/'tools/fg_single3_experiment.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    resources=runner.hardware_preflight(config,allowed_logdirs=(Path(config.logdir),))
    import fcntl
    import uuid
    target=Path(config.logdir)
    with (target.parent/('.'+target.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempt=uuid.uuid4().hex
        write_json(target/('resources_'+attempt+'.json'),signed(resources))
        code=runner.run_training(receipt['argv'],target/('stdout_'+attempt+'.log'),target)
        if code:
            raise subprocess.CalledProcessError(code,receipt['argv'])
