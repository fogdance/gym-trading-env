"""Two-batch CPU pipeline with immutable attempts and coarse stage recovery."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time

from dreamerv3.configuration import read_config_yaml
from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import check_files, digest, read_manifest, signed, write_json
from .fg_s3_corpus_resources import (Broker, Ledger, ENV, allocate, available_memory,
                                     policy_cpus, process_identity)

PYTHON = '/home/v/miniconda3/envs/dreamerv3/bin/python'
PROJECT = Path(__file__).resolve().parents[3]
STAGES = ('runtime_limits', 'validate', 'generate', 'native_g3', 'reader', 'coverage', 'publish')
CPU_ENV = dict(PYTHONPATH='.:src:../dreamerv3', CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
    PYTHONDONTWRITEBYTECODE='1', PYTHONINTMAXSTRDIGITS='0', OPENBLAS_NUM_THREADS='1',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')


def read(path):
    return json.loads(Path(path).read_text())


def last_json(path):
    # A trailing diagnostic or signal is not a successful final receipt.
    return json.loads(Path(path).read_text().splitlines()[-1])


def batch_signed(value):
    # Published batch results use the existing compact JSON contract, whereas
    # Replay/control manifests use entry_contract.fingerprint's spaced JSON.
    return dict(value,content_hash=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest())


def read_batch(path):
    value=read(path)
    if batch_signed({k:v for k,v in value.items() if k!='content_hash'})!=value:
        raise ValueError('Published batch integrity mismatch')
    return value


def checked_path(path):
    value = Path(path).absolute()
    if value.resolve() != value:
        raise ValueError(f'Symlink/relative ancestor refused: {value}')
    return value


def check_environment(config):
    if checked_path(config['producer']['gym'])!=PROJECT:
        raise ValueError('Launch the production CLI from its configured frozen Gym/Dreamer pair')
    for path in (PROJECT,PROJECT.parent/'dreamerv3',Path(config['production_root'])):
        existing=path
        while not existing.exists():existing=existing.parent
        device=subprocess.check_output(['findmnt','-n','-o','SOURCE','-T',str(existing)],text=True).strip()
        if device not in ('/dev/sdd2','/dev/sda1'):
            raise ValueError(f'Production storage must be /dev/sdd or /dev/sda: {path} -> {device}')
        if shutil.disk_usage(existing).free<5*1024**3:
            raise ValueError(f'Insufficient free production disk space: {existing}')


def load_config(path):
    value = read_config_yaml(path)
    if value.get('protocol') != 'FG_S3_CORPUS_PRODUCTION_V1':
        raise ValueError('Unknown production configuration')
    value = json.loads(json.dumps(value))
    value['production_root'] = str(checked_path(value['production_root']))
    resources = value['resources']
    resources['policy_cpus'] = policy_cpus(resources['policy_cpus'])
    affinity = resources['affinity']
    if (not isinstance(affinity, list) or not affinity or len(set(affinity)) != len(affinity)
            or any(type(cpu) is not int or cpu < 0 for cpu in affinity)
            or type(resources['workers']) is not int or not 12 <= resources['workers'] <= len(affinity)
            or resources.get('native_threads') != 1
            or value.get('max_inflight_batches') not in (1, 2)):
        raise ValueError('Invalid production CPU/phase limits')
    for key in ('worker_memory_mib', 'coordinator_memory_mib', 'headroom_mib'):
        if type(resources[key]) is not int or resources[key] < 1:
            raise ValueError('Invalid physical memory reservation')
    ids = [batch['id'] for batch in value['batches']]
    if (not ids or len(set(ids)) != len(ids)
            or any(re.fullmatch(r'[A-Za-z0-9_-]+', name) is None for name in ids)):
        raise ValueError('Unique safe batch IDs required')
    if type(value['seed']) is not int or type(value['reader_length']) is not int or value['reader_length']<1:
        raise ValueError('Explicit integer seed and positive Reader length required')
    # Resource changes are allowed between attempts. Source recipe/pins are not.
    identity = {k:v for k,v in value.items() if k not in ('resources', 'max_inflight_batches')}
    return value, fingerprint(identity)


class ProductionLock:
    def __init__(self, root):
        self.root, self.stream = Path(root), None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.stream = (self.root/'controller.lock').open('a+')
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.stream.close()
            raise ValueError('Another Corpus production controller owns this output root') from exc
        return self

    def __exit__(self, *args):
        self.stream.close()


def validate_completion(record):
    if record.get('status') != 'succeeded':
        return False
    completion = read_manifest(Path(record['attempt'])/'completion.json')
    if (completion.get('exit_code') != 0 or completion.get('stage') != record['stage']
            or completion.get('status') != 'succeeded'
            or completion['content_hash'] != record['completion_hash'] or not completion['files']):
        raise ValueError('Invalid successful stage receipt')
    check_files(completion['files'])
    stage_result(Path(record['attempt']),record['stage'])
    return True


def stage_result(directory, stage):
    result=read(directory/'result.json')
    statuses=dict(runtime_limits=('PASS_EXACT_RATIONAL_RUNTIME',),validate=('SOURCE_VALIDATED_ONLY',),
        generate=('CORPUS_PRODUCED','ADOPTED_GENERATED_CORPUS'),native_g3=('NATIVE_G3_VERIFIED',),
        reader=('PASS_READER_ONLY','PASS_READER_CPU'),coverage=('OBSERVED_COVERAGE_ONLY',),
        publish=('NATIVE_G3_AND_INDEPENDENT_READER_COMPLETED',))
    if result.get('status') not in statuses[stage]:
        raise ValueError('Missing final successful stage artifact: '+stage)
    if stage=='generate':
        generated=read(directory/'generated.json')
        manifest=read_manifest(Path(generated['corpus'])/'corpus_manifest.json')
        if generated!=result or manifest['content_hash']!=result['corpus_identity']:
            raise ValueError('Generated manifest identity differs')
    elif stage=='validate':
        if read_manifest(directory/'VALIDATED_SOURCE_LOCK.json')!=result['source']:
            raise ValueError('Validated source artifact differs')
    elif stage=='native_g3':
        native=read(directory/'native/native_verification.json')
        value=native['result']
        if (native['exit_code']!=0 or value.get('status')!='PASS' or value.get('G3')!='GO'
                or value.get('verification_state')!='NATIVE_G3_VERIFIED'
                or value['corpus_identity']!=result['corpus_identity']
                or len(value.get('native_replay',[]))!=6
                or any(r.get('status')!='PASS' or not r.get('records')
                       or r['records']!=r.get('teacher_labels_verified') for r in value['native_replay'])):
            raise ValueError('Incomplete native G3 artifact')
    elif stage=='reader':
        if (read_manifest(directory/'reader/WP03_COURSE_READER_COVERAGE.json')!=result
                or digest(directory/'reader/join.sqlite')!=result['index_sha256']):
            raise ValueError('Reader artifact differs')
    elif stage=='coverage':
        if read(directory/'quality_coverage.json')!=result:
            raise ValueError('Coverage artifact differs')
    elif stage=='publish':
        manifest=read_batch(directory/'BATCH_MANIFEST.json')
        read_manifest(directory/'IDENTITY_LOCK.json')
        if {k:v for k,v in manifest.items() if k!='files'}!=result:
            raise ValueError('Published artifact differs')
    elif result.get('python_int_max_str_digits')!=0:
        raise ValueError('Runtime rational witness missing')
    return result


def terminate_owned(identity, grace=5, known_members=()):
    """Never signal a reused PID or a process outside our recorded session."""
    current = process_identity(identity['pid'])
    if current is None:
        members = group_members(identity['pgid'])
        if not members:
            return
        if not any(process_identity(item['pid']) == item and item['pgid'] == identity['pgid'] for item in known_members):
            raise ValueError('Orphan group ownership cannot be proved; preserve state for manual process audit')
    elif current != identity or identity['pid'] != identity['pgid']:
        raise ValueError('Recorded process identity changed; refusing to signal foreign process')
    try:os.killpg(identity['pgid'], signal.SIGTERM)
    except ProcessLookupError:return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        # A zombie leader may persist until Popen.wait; descendants still need cleanup.
        members = group_members(identity['pgid'])
        if not any(state != 'Z' for _, state in members):
            return
        time.sleep(.05)
    if group_members(identity['pgid']):
        try:os.killpg(identity['pgid'], signal.SIGKILL)
        except ProcessLookupError:pass


def group_members(pgid):
    members = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
                if int(fields[2]) == pgid:
                    members.append((int(path.name), fields[0]))
            except OSError:
                pass
    return members


def producer_for(config, batch):
    return batch.get('adopt', {}).get('producer', config['producer'])


def authenticate(producer):
    from .fg_s3_e_quality_supervision import frozen_producer_audit
    return frozen_producer_audit(producer['gym'], producer_lock=producer['pin'],
                                 producer_lock_sha256=producer['pin_sha256'])


def validate_source(source, batch, config):
    if (source.get('fold') != 'TRAIN' or source['contract'] != batch['contract']
            or source['selection'] != dict(start_day=batch['start_day'], end_day=batch['end_day'])
            or source['source_pool']['path'] != config['source_pool']
            or source['env_receipt']['path'] != config['env_receipt']):
        raise ValueError('Validated source differs from configured TRAIN recipe')
    if batch.get('plan'):
        plan = batch['plan']
        check_files({plan['path']: plan['sha256']})
        expected = read_manifest(plan['path'])
        for key in ('contract', 'row', 'source_pool', 'official_days', 'start_cursor', 'end_cursor'):
            if source[key] != expected[key]:
                raise ValueError('Validated source differs from locked batch plan: '+key)
        if source['suffix_end_cursor'] != expected['end_cursor']:
            raise ValueError('Full TRAIN suffix differs from locked batch plan')


def validate_generated(corpus, source, producer, seed):
    from .fg_s3_e_quality_supervision import SupervisionSource, inspect_source
    manifest = inspect_source(SupervisionSource.from_corpus(corpus))
    check_files(manifest['source_binding']['files'])
    if (manifest.get('execution_mode', 'data') != 'data'
            or read_manifest(Path(corpus)/'source_lock.json') != source
            or manifest['source_lock_hash'] != source['content_hash']):
        raise ValueError('Generated Corpus source differs from validated source')
    audit = authenticate(producer)
    for role, record in manifest['code']['repositories'].items():
        if record['sha'] != audit[role]['head'] or record['dirty']:
            raise ValueError('Corpus producer differs from authenticated frozen producer')
    spec=read(Path(corpus)/'course_v2.json')
    if any(item['seed']!=seed for item in spec['curriculum']['behaviors']):
        raise ValueError('Corpus behavior seed differs from configured recipe')
    return manifest


def run_command(argv, out, *, cwd=PROJECT):
    with (out/'stdout.log').open('x') as log:
        result = subprocess.run(argv, cwd=cwd, env={**os.environ, **CPU_ENV},
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise ValueError(f'Production child exited {result.returncode}: {out}/stdout.log')
    return last_json(out/'stdout.log')


def execute_stage(request):
    """One phase subprocess; uses existing Producer/Reader entrypoints."""
    config, batch, stage = request['config'], request['batch'], request['stage']
    out = Path(request['out'])
    producer = producer_for(config, batch)
    if stage == 'runtime_limits':
        check_files({config['runtime_probe']['path']: config['runtime_probe']['sha256']})
        result = run_command([PYTHON, '-B', config['runtime_probe']['path']], out)
        if result.get('status') != 'PASS_EXACT_RATIONAL_RUNTIME' or result.get('python_int_max_str_digits') != 0:
            raise ValueError('Exact rational runtime failed')
    elif stage == 'validate':
        from .fg_s3_e_quality_corpus_cli import source_inputs
        source, _, _ = source_inputs(config['source_pool'], config['env_receipt'], batch['contract'],
            start_day=batch['start_day'], end_day=batch['end_day'])
        validate_source(source, batch, config)
        authenticate(producer)
        write_json(out/'VALIDATED_SOURCE_LOCK.json', source)
        result = dict(status='SOURCE_VALIDATED_ONLY', source=source)
    elif stage == 'generate':
        source = read_manifest(Path(request['previous']['validate'])/'VALIDATED_SOURCE_LOCK.json')
        if batch.get('adopt'):
            corpus = Path(batch['adopt']['corpus'])
            manifest = validate_generated(corpus, source, producer, config['seed'])
            result = dict(status='ADOPTED_GENERATED_CORPUS', corpus=str(corpus), corpus_identity=manifest['content_hash'])
        else:
            from .fg_s3_e_quality_corpus_cli import generate_small
            authenticate(producer)
            corpus = out/'corpus'
            manifest = generate_small(config['source_pool'], config['env_receipt'], batch['contract'], corpus,
                seed=config['seed'], start_day=batch['start_day'], end_day=batch['end_day'],
                workers=request['capacity'], mode='data', policy_cpus=request['policy_cpus'])
            validate_generated(corpus, source, producer, config['seed'])
            result = dict(status='CORPUS_PRODUCED', corpus=str(corpus), corpus_identity=manifest['content_hash'])
        write_json(out/'generated.json', result)
    elif stage == 'native_g3':
        from .fg_s3_e_quality_supervision import QualitySupervisionAdmission, SupervisionSource
        corpus = Path(request['corpus'])
        target = out/'native';target.mkdir()
        admission = QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(corpus),
            purpose='native-verification', producer_gym=producer['gym'], log_path=target/'native_verify.log',
            producer_lock=producer['pin'], producer_lock_sha256=producer['pin_sha256'],
            workers=request['capacity'], **({} if batch.get('adopt') else {'policy_cpus':request['policy_cpus']}))
        write_json(target/'native_verification.json', admission.native_result)
        write_json(target/'WP03_SOURCE_ADMISSION_RECEIPT.json', admission.receipt)
        result = dict(status='NATIVE_G3_VERIFIED', corpus_identity=admission.source.corpus_content_hash)
    elif stage == 'reader':
        from .fg_s3_e_quality_supervision import QualitySupervisionAdmission, SupervisionSource
        from .fg_s3_e_quality_checks import reader_evidence
        admission = QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(request['corpus']))
        target = out/'reader';target.mkdir()
        write_json(target/'WP03_SOURCE_ADMISSION_RECEIPT.json', admission.receipt)
        result = reader_evidence(admission, target, length=config.get('reader_length', 64),
                                 seed=config['seed'], with_model=False)
    elif stage == 'coverage':
        program = config['coverage_program']
        check_files({program['path']:program['sha256']})
        result = run_command([PYTHON, '-B', program['path'], '--corpus', request['corpus'],
            '--reader-dir', str(Path(request['previous']['reader'])/'reader'),
            '--out', str(out/'quality_coverage.json')], out)
        if result.get('corpus_identity') != read_manifest(Path(request['corpus'])/'corpus_manifest.json')['content_hash']:
            raise ValueError('Coverage identity differs')
    elif stage == 'publish':
        result = publish_batch(request)
    else:
        raise ValueError('Unknown production stage')
    write_json(out/'result.json', result)
    return result


def link_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def publish_batch(request):
    """Same native/Reader publication contract as the original full-main release.

    Only selected successful attempts are copied into this immutable receipt
    directory. Controller state, active logs and failed attempts stay outside.
    """
    corpus, out = Path(request['corpus']), Path(request['out'])
    previous = {key:Path(value) for key,value in request['previous'].items()}
    for stage,directory in previous.items():
        validate_completion(dict(stage=stage,status='succeeded',attempt=str(directory),
                                 completion_hash=request['previous_receipts'][stage]))
    config, batch = request['config'], request['batch']
    source = read_manifest(previous['validate']/'VALIDATED_SOURCE_LOCK.json')
    manifest = validate_generated(corpus, source, producer_for(config, batch),config['seed'])
    native = read(previous['native_g3']/'native/native_verification.json')
    reader = read_manifest(previous['reader']/'reader/WP03_COURSE_READER_COVERAGE.json')
    result = native['result']
    if (native['exit_code'] != 0 or result.get('G3') != 'GO' or result.get('status') != 'PASS'
            or result.get('verification_state') != 'NATIVE_G3_VERIFIED'
            or result['corpus_identity'] != manifest['content_hash']
            or reader['corpus_identity'] != manifest['content_hash']
            or reader['actual_optimizer_exposures'] != 0 or reader.get('training_admitted') is not False
            or reader['status'] not in ('PASS_READER_ONLY', 'PASS_READER_CPU')
            or native['producer_audit'] != authenticate(producer_for(config, batch))):
        raise ValueError('Batch native/Reader/producer receipt mismatch')
    from .fg_replay import trace_manifest, E_QUALITY_SCHEMA
    if len(result['native_replay']) != len(manifest['behaviors']):
        raise ValueError('Incomplete G3 receipts')
    for name, receipt in zip(manifest['behaviors'], result['native_replay'], strict=True):
        trace = trace_manifest(corpus/'policies'/name, schema=E_QUALITY_SCHEMA)
        if (receipt['status'] != 'PASS' or not receipt['records']
                or receipt['records'] != receipt['teacher_labels_verified']
                or receipt['records'] != trace['records'] or receipt['replay_trace_hash'] != trace['replay_trace_hash']):
            raise ValueError('Incomplete G3 label verification')
    if digest(previous['reader']/'reader/join.sqlite') != reader['index_sha256']:
        raise ValueError('Reader index changed')
    runtime = read(previous['runtime_limits']/'result.json')
    coverage = read(previous['coverage']/'quality_coverage.json')
    if (runtime.get('status') != 'PASS_EXACT_RATIONAL_RUNTIME' or runtime['python_int_max_str_digits'] != 0
            or coverage['corpus_identity'] != manifest['content_hash']):
        raise ValueError('Runtime/coverage proof missing')
    # Preserve canonical receipt paths consumed by existing multi-Corpus admission.
    for name, directory in (('native', previous['native_g3']/'native'), ('reader', previous['reader']/'reader')):
        shutil.copytree(directory, out/name, copy_function=link_copy)
    shutil.copy2(previous['coverage']/'quality_coverage.json', out/'quality_coverage.json')
    for stage, directory in previous.items():
        target = out/'stages'/stage;target.mkdir(parents=True)
        for path in directory.iterdir():
            if path.is_file():
                link_copy(path, target/path.name)
    files = {'corpus/'+str(path.relative_to(corpus)):digest(path)
             for path in corpus.rglob('*') if path.is_file()}
    files.update({'receipts/'+str(path.relative_to(out)):digest(path) for path in out.rglob('*')
                  if path.is_file() and path.parts[len(out.parts)] not in ('controller.log', 'request.json')})
    value = batch_signed(dict(protocol='FG_S3_E_TRAIN_CORPUS_BATCH_RESULT_V1', batch_id=batch['id'],
        status='NATIVE_G3_AND_INDEPENDENT_READER_COMPLETED', corpus_identity=manifest['content_hash'],
        source_lock=manifest['source_lock_hash'], course_identity=manifest['course_identity_hash'],
        teacher_identity=manifest['teacher_identity'], producer_repositories=manifest['code']['repositories'],
        producer_pin_sha256=producer_for(config, batch)['pin_sha256'], index_sha256=reader['index_sha256'],
        curriculum_support=reader['curriculum_support'], training_admitted=False, optimizer_updates=0,
        runtime_environment=dict(PYTHONINTMAXSTRDIGITS='0', python_int_max_str_digits=0), files=files))
    write_json(out/'BATCH_MANIFEST.json', value)
    write_json(out/'IDENTITY_LOCK.json', signed(dict(protocol='FG_S3_E_TRAIN_CORPUS_IDENTITY_LOCK_V1',
        batch_id=batch['id'], corpus_identity=manifest['content_hash'], source_lock=manifest['source_lock_hash'],
        course_identity=manifest['course_identity_hash'], teacher_identity=manifest['teacher_identity'],
        producer_pin_sha256=value['producer_pin_sha256'], index_sha256=reader['index_sha256'],
        files={**files, 'receipts/BATCH_MANIFEST.json':digest(out/'BATCH_MANIFEST.json')})))
    return {key:value for key,value in value.items() if key != 'files'}


class Controller:
    def __init__(self, config, recipe):
        self.config, self.recipe = config, recipe
        self.root = Path(config['production_root'])
        self.control = self.root/'control'
        self.state_path = self.control/'state.json'
        self.state_lock = threading.Lock()
        self.running, self.errors = {}, []
        self.stopping = threading.Event()
        self.ledger, self.broker = None, None

    def save(self):
        write_json(self.state_path, self.state)

    def initialize(self, resume):
        self.control.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            if not resume:
                raise ValueError('Production state exists; use resume --config')
            self.state = read(self.state_path)
            if self.state['recipe_hash'] != self.recipe:
                raise ValueError('Production source/recipe/producer identity changed; create a new production root')
            for batch in self.state['batches'].values():
                for attempts in batch['stages'].values():
                    for record in attempts:
                        if record['status'] == 'running':
                            if record.get('process'):
                                terminate_owned(record['process'],known_members=record.get('members',()))
                            record['status'] = 'interrupted'
                        if record['status'] == 'succeeded':
                            validate_completion(record)
            self.save()
        else:
            if resume and any((self.root/'attempts').glob('*')):
                raise ValueError('Attempts exist without controller state; manual audit required')
            self.state = dict(protocol='FG_S3_CORPUS_PRODUCTION_STATE_V1', recipe_hash=self.recipe,
                batches={batch['id']:dict(stages={stage:[] for stage in STAGES}) for batch in self.config['batches']})
            self.save()

    def successful(self, batch):
        result = {}
        for stage in STAGES:
            records = self.state['batches'][batch['id']]['stages'][stage]
            successes = [record for record in records if record['status'] == 'succeeded']
            if not successes:
                break
            result[stage] = successes[-1]
        return result

    def setup_resources(self):
        resources = self.config['resources']
        if not set(resources['affinity']) <= os.sched_getaffinity(0):
            raise ValueError('Configured CPU affinity is unavailable')
        worker_bytes = resources['worker_memory_mib'] * 1024**2
        headroom = resources['headroom_mib'] * 1024**2
        parent = resources['coordinator_memory_mib'] * 1024**2
        available=available_memory()
        check_memory = resources.get('memory_checks', True)
        capacity = resources['workers']
        if check_memory:
            capacity = min(capacity, (available-headroom-2*parent)//worker_bytes)
        if capacity<12:
            raise ValueError(f'Insufficient CPU/memory capacity: {capacity} workers fit, 12 required; '
                             f'MemAvailable={available}, worker_bytes={worker_bytes}, '
                             f'headroom_bytes={headroom}, coordinator_bytes={2*parent}')
        self.budgets = allocate(resources['policy_cpus'], capacity)
        self.capacity = int(capacity)
        event_lock = threading.Lock()
        def event(value):
            with event_lock, (self.control/'resources.jsonl').open('a') as stream:
                stream.write(json.dumps(value)+'\n')
        self.ledger = Ledger(self.capacity, worker_bytes, headroom, parent, event=event,
                             check_memory=check_memory)
        self.broker = Broker(self.ledger)
        print(json.dumps(dict(status='RESOURCE_PLAN', capacity=self.capacity, policy_cpus=self.budgets,
            affinity=resources['affinity'], memory_checks=check_memory,
            worker_memory_bytes=worker_bytes, available_memory_bytes=available)), flush=True)

    def request(self, batch, stage, directory, previous):
        value = dict(config=self.config, batch=batch, stage=stage, out=str(directory),
                     previous={key:item['attempt'] for key,item in previous.items()},
                     previous_receipts={key:item['completion_hash'] for key,item in previous.items()},
                     capacity=self.capacity, policy_cpus=self.budgets)
        if 'generate' in previous:
            value['corpus'] = read(Path(previous['generate']['attempt'])/'generated.json')['corpus']
        return value

    def stage_command(self, directory):
        return [PYTHON,'-B','tools/fg_s3_e_quality_corpus.py','_production-stage',
                '--request',str(directory/'request.json')]

    def run_stage(self, batch, stage, record, request):
        token, child = None, None
        directory = Path(record['attempt'])
        try:
            priority = self.config['batches'].index(batch)*10 + (0 if stage != 'generate' else 2)
            token = self.ledger.acquire(1, batch['id']+'/'+stage, priority)
            if self.stopping.is_set():
                raise RuntimeError('Controller interrupted')
            argv = self.stage_command(directory)
            env = {**os.environ, **CPU_ENV, ENV:self.broker.setting, 'FG_CORPUS_PREP_LEASE':token,
                   'FG_CORPUS_PRIORITY':str(priority)}
            env['FG_CORPUS_MEMORY_CHECKS'] = '1' if self.config['resources'].get('memory_checks', True) else '0'
            env['TMPDIR'] = str(self.control/'tmp');Path(env['TMPDIR']).mkdir(exist_ok=True)
            env['MPLCONFIGDIR'] = str(self.control/'matplotlib')
            if batch.get('adopt'):
                env['FG_CORPUS_LEGACY'] = '1'
            write_json(directory/'command.json', dict(argv=argv, cwd=str(PROJECT),
                resources=self.config['resources'], capacity=self.capacity, policy_cpus=self.budgets,
                env_overrides=CPU_ENV, recipe_hash=self.recipe))
            # taskset avoids preexec_fn in this threaded controller.
            argv = ['taskset', '-c', ','.join(map(str, self.config['resources']['affinity'])), *argv]
            with (directory/'controller.log').open('x') as log:
                child = subprocess.Popen(argv, cwd=PROJECT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                identity = process_identity(child.pid)
                with self.state_lock:
                    record['process'] = identity;self.save()
                self.ledger.register(child.pid)
                last_snapshot = 0
                while child.poll() is None:
                    if time.monotonic()-last_snapshot > .5:
                        members=[process_identity(pid) for pid,state in group_members(child.pid) if state!='Z']
                        with self.state_lock:
                            record['members']=[item for item in members if item];self.save()
                        last_snapshot=time.monotonic()
                    if self.stopping.wait(.15):
                        terminate_owned(identity,known_members=record.get('members',()));break
                exit_code = child.wait()
            # Account processes must not outlive a successful phase.
            remaining = [(pid,state) for pid,state in group_members(child.pid) if state != 'Z']
            if remaining:
                raise ValueError('Phase left live descendant processes; refusing completion')
            if exit_code != 0:
                raise ValueError(f'{batch["id"]}/{stage} exited {exit_code}; {directory}/controller.log')
            result = stage_result(directory,stage)
            files = {str(path):digest(path) for path in directory.rglob('*') if path.is_file()}
            if stage == 'generate' and batch.get('adopt'):
                corpus = Path(result['corpus'])
                files.update({str(path):digest(path) for path in corpus.rglob('*') if path.is_file()})
            completion = signed(dict(status='succeeded', stage=stage, exit_code=exit_code, files=files,
                                     recipe_hash=self.recipe, finished=time.time()))
            write_json(directory/'completion.json', completion)
            with self.state_lock:
                record.update(status='succeeded', completion_hash=completion['content_hash'], finished=time.time())
                self.save()
        except BaseException as exc:
            was_stopping=self.stopping.is_set()
            self.stopping.set();self.ledger.stop()
            if child:
                try:
                    terminate_owned(record['process'],known_members=record.get('members',()));child.wait()
                except (OSError,ValueError) as cleanup:
                    exc=RuntimeError(f'{exc}; cleanup requires audit: {cleanup}')
            with self.state_lock:
                record.update(status='interrupted' if was_stopping or isinstance(exc, KeyboardInterrupt) else 'failed',
                              error=str(exc), finished=time.time())
                self.errors.append(str(exc));self.save()
        finally:
            if token:self.ledger.release(token)
            if child:self.ledger.unregister(child.pid)
            with self.state_lock:
                self.running.pop(batch['id'], None)

    def run(self, resume):
        check_environment(self.config)
        with ProductionLock(self.root/'control'):
            self.initialize(resume)
            if all('publish' in self.successful(batch) for batch in self.config['batches']):
                print(json.dumps(dict(status='ALL_BATCHES_ALREADY_PUBLISHED')));return 0
            authenticate(self.config['producer'])
            self.setup_resources()
            interrupted = False
            def stop(signum, frame):
                nonlocal interrupted
                interrupted = True;self.stopping.set();self.ledger.stop()
            old_handlers = {sig:signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
            threads = []
            try:
                while not self.stopping.is_set():
                    if self.ledger.failure:
                        self.errors.append(self.ledger.failure);self.stopping.set();break
                    with self.state_lock:
                        active = [batch for batch in self.config['batches'] if 'publish' not in self.successful(batch)]
                        if not active and not self.running:
                            break
                        window = active[:self.config['max_inflight_batches']]
                        for index,batch in enumerate(window):
                            if batch['id'] in self.running:
                                continue
                            previous = self.successful(batch)
                            stage = STAGES[len(previous)]
                            # Batch B enters only after A's generation, and at most
                            # one generating and one native-verifying batch run.
                            if index and 'generate' not in self.successful(window[index-1]):
                                continue
                            if any(item['stage'] == stage for item in self.running.values()):
                                continue
                            records = self.state['batches'][batch['id']]['stages'][stage]
                            parent=self.root/'attempts'/batch['id']/stage
                            # mkdir can precede the last atomic state commit.
                            # Preserve such directories and advance past them.
                            existing=[int(path.name) for path in parent.glob('*') if path.name.isdigit()]
                            number=max([len(records),*existing])+1
                            directory = parent/f'{number:04d}'
                            directory.mkdir(parents=True, exist_ok=False)
                            record = dict(stage=stage, status='running', attempt=str(directory), started=time.time())
                            records.append(record);self.running[batch['id']] = record
                            request = self.request(batch, stage, directory, previous)
                            write_json(directory/'request.json', request);self.save()
                            thread = threading.Thread(target=self.run_stage, args=(batch,stage,record,request))
                            threads.append(thread);thread.start()
                    time.sleep(.1)
            except BaseException:
                self.stopping.set();self.ledger.stop()
                raise
            finally:
                if self.stopping.is_set():self.ledger.stop()
                for thread in threads:thread.join()
                self.broker.close()
                for sig, handler in old_handlers.items():signal.signal(sig, handler)
            if self.ledger.failure and self.ledger.failure not in self.errors:
                self.errors.insert(0,self.ledger.failure)
            if interrupted:
                raise RuntimeError('Production interrupted; resume reruns the incomplete stage in a fresh attempt')
            if self.errors:
                raise RuntimeError('; '.join(self.errors))
            print(json.dumps(dict(status='ALL_BATCHES_PUBLISHED', training_admitted=False)), flush=True)
            return 0


def production_main(command, path):
    config, recipe = load_config(path)
    if command == 'status':
        state_path = Path(config['production_root'])/'control/state.json'
        if not state_path.exists():
            print(json.dumps(dict(status='NOT_STARTED', recipe_hash=recipe)));return 0
        state = read(state_path)
        if state['recipe_hash'] != recipe:
            raise ValueError('Production recipe differs')
        print(json.dumps(dict(status='PRODUCTION_STATUS', batches={name:{stage:(records[-1]['status'] if records else 'pending')
            for stage,records in batch['stages'].items()} for name,batch in state['batches'].items()}), ensure_ascii=False))
        return 0
    return Controller(config, recipe).run(resume=command == 'resume')
