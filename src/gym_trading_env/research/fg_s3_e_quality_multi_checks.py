"""Read-only multi-source diagnostics and original FullCheckpoint data probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
from types import SimpleNamespace
import subprocess
import sys
import time

import numpy as np

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import digest, read_manifest, signed, write_json
from .fg_s3_e_quality_course_preflight import _local_path
from .fg_s3_e_quality_multi import MultiCorpusCourseStream, STAGES
from .fg_s3_e_quality_supervision import SupervisionSource, QualitySupervisionAdmission


CONFIG_SCHEMA = 'FG_S3_E_MULTICORPUS_CONFIG_V1'
PROJECT = Path(__file__).resolve().parents[3]


def read_config(path):
    path = _local_path(path)
    config = json.loads(path.read_text())
    required = {'schema', 'sources', 'stage', 'per_source_sampling_spec', 'source_mix_weights', 'seed'}
    if (not isinstance(config, dict) or not required <= set(config)
            or set(config) - required - {'cache_bytes', 'f3_auxiliary_sampling_spec'} or config['schema'] != CONFIG_SCHEMA
            or config['stage'] not in STAGES or not isinstance(config['sources'], list)
            or not config['sources']):
        raise ValueError('MULTI_CONFIG_INVALID')
    for source in config['sources']:
        if (not isinstance(source, dict) or not {'corpus_id', 'corpus_root', 'index_path'} <= set(source)
                or set(source) - {'corpus_id', 'corpus_root', 'index_path', 'batch_manifest'}):
            raise ValueError('MULTI_CONFIG_SOURCE_INVALID')
        for key in ('corpus_root', 'index_path', 'batch_manifest'):
            if key in source:
                value = source[key]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError('MULTI_CONFIG_PATH_INVALID')
                source[key] = str(_local_path(path.parent/value if not Path(value).is_absolute() else value))
    return config


def _sources(config):
    return [dict(corpus_id=s['corpus_id'], index_path=s['index_path'],
        admission=QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(
            s['corpus_root'], stage=STAGES[config['stage']]))) for s in config['sources']]


def open_config(path):
    config = read_config(path)
    if config.get('f3_auxiliary_sampling_spec') is not None:
        raise ValueError('MULTI_F3_ADAPTER_REQUIRED_FOR_AUXILIARY')
    sources = _sources(config)
    return MultiCorpusCourseStream.open(sources, stage=config['stage'],
        per_source_sampling_spec=config['per_source_sampling_spec'],
        source_mix_weights=config['source_mix_weights'], seed=config['seed'],
        cache_bytes=config.get('cache_bytes', 64 << 20))


def published_sources(config):
    """Verify site receipts, not fresh in-process G3 or training admission."""
    if len(config['sources']) < 2:
        raise ValueError('MULTI_TWO_PUBLISHED_CORPORA_REQUIRED')
    result = []
    for source in config['sources']:
        if 'batch_manifest' not in source:
            raise ValueError('MULTI_PUBLISHED_BATCH_MANIFEST_REQUIRED')
        path = _local_path(source['batch_manifest'])
        batch = json.loads(path.read_text())
        body = {k: v for k, v in batch.items() if k != 'content_hash'}
        if (hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                != batch.get('content_hash') or batch.get('protocol') != 'FG_S3_E_TRAIN_CORPUS_BATCH_RESULT_V1'
                or batch.get('status') != 'NATIVE_G3_AND_INDEPENDENT_READER_COMPLETED'
                or batch.get('training_admitted') is not False or batch.get('optimizer_updates') != 0):
            raise ValueError('MULTI_PUBLISHED_BATCH_IDENTITY_INVALID')
        corpus, receipts = Path(source['corpus_root']), path.parent
        required = {'corpus/corpus_manifest.json', 'corpus/source_lock.json', 'corpus/course_v2.json',
            'corpus/teacher_manifest.json', 'corpus/model_contract.json',
            'receipts/native/native_verification.json',
            'receipts/reader/WP03_COURSE_READER_COVERAGE.json',
            'receipts/reader/join.sqlite', 'receipts/reader/join.json'}
        if not required <= set(batch['files']):
            raise ValueError('MULTI_PUBLISHED_RECEIPT_INVENTORY_INCOMPLETE')
        for name, expected in batch['files'].items():
            kind, relative = name.split('/', 1)
            if kind not in ('corpus', 'receipts') or Path(relative).is_absolute() or '..' in Path(relative).parts:
                raise ValueError('MULTI_PUBLISHED_FILE_PATH_INVALID')
            target = _local_path((corpus if kind == 'corpus' else receipts)/relative)
            if digest(target) != expected:
                raise ValueError('MULTI_PUBLISHED_FILE_CHANGED: ' + name)
        manifest = read_manifest(corpus/'corpus_manifest.json')
        native = json.loads((receipts/'native/native_verification.json').read_text())
        reader = read_manifest(receipts/'reader/WP03_COURSE_READER_COVERAGE.json')
        if (native['exit_code'] != 0 or native['result']['G3'] != 'GO'
                or native['result']['corpus_identity'] != batch['corpus_identity']
                or reader['corpus_identity'] != batch['corpus_identity']
                or reader['status'] not in ('PASS_READER_ONLY', 'PASS_READER_CPU')
                or reader['actual_optimizer_exposures'] != 0
                or manifest['content_hash'] != batch['corpus_identity']
                or manifest['source_lock_hash'] != batch['source_lock']
                or manifest['course_identity_hash'] != batch['course_identity']
                or manifest['teacher_identity'] != batch['teacher_identity']
                or manifest['code']['repositories'] != batch['producer_repositories']
                or digest(Path(source['index_path'])) != batch['index_sha256']
                or digest(Path(source['index_path']).with_suffix('.json')) != batch['files']['receipts/reader/join.json']):
            raise ValueError('MULTI_PUBLISHED_RECEIPTS_DIFFER')
        # Authenticate the original TRAIN whitelist, not a self-signed claim.
        from .fg_s3_e_quality_corpus_cli import source_inputs
        lock = read_manifest(corpus/'source_lock.json')
        actual, _, _ = source_inputs(lock['source_pool']['path'], lock['env_receipt']['path'],
                                    lock['contract'], **lock['selection'])
        if actual != lock:
            raise ValueError('MULTI_ORIGINAL_TRAIN_AUTHORITY_DIFFERS')
        result.append(dict(corpus_id=source['corpus_id'], batch_manifest_sha256=digest(path),
            corpus_identity=manifest['content_hash'], original_train_lock=lock['content_hash'],
            existing_native_g3='VERIFIED_PUBLISHED_RECEIPT_ONLY', new_native_g3='NOT_RUN'))
    return result


class _ProbeWindowAdapter:
    """All native fields, for a data-only probe; never passed to an Agent."""
    def __init__(self, reader, settings):
        self.reader = reader

    def adapt(self, raw):
        return raw, self.reader.audit_windows(raw)


def _diagnostic_stream(path, *, adapter, batch_size, length, burnin_steps):
    if adapter == 'native':
        if burnin_steps is not None:
            raise ValueError('MULTI_F3_ADAPTER_REQUIRED_FOR_BURNIN')
        core = open_config(path)
        try:
            return core.bind(dict(batch_size=batch_size, target_steps=length-1, burnin_steps=1), _ProbeWindowAdapter)
        except BaseException:
            core.close()
            raise
    if adapter != 'f3' or type(burnin_steps) is not int or not 0 < burnin_steps < length:
        raise ValueError('MULTI_F3_EXPLICIT_BURNIN_REQUIRED')
    from .fg_s3_e_quality_multi_actor import MultiCorpusF3Stream
    config = read_config(path)
    if config['stage'] != 'ACTOR_STRICT':
        raise ValueError('MULTI_F3_ACTOR_STRICT_STAGE_REQUIRED')
    return MultiCorpusF3Stream.open(_sources(config),
        settings=dict(batch_size=batch_size, target_steps=length-burnin_steps, burnin_steps=burnin_steps),
        per_source_sampling_spec=config['per_source_sampling_spec'],
        aux_per_source_sampling_spec=config.get('f3_auxiliary_sampling_spec'),
        source_mix_weights=config['source_mix_weights'], seed=config['seed'],
        cache_bytes=config.get('cache_bytes', 64 << 20))


def f3_loss_evidence(data, burnin_steps):
    """Original WP06 objective over fixed logits, with no model or optimizer."""
    import jax
    import jax.numpy as jnp
    import embodied
    from dreamerv3.fg_s3_e_quality_actor import actor_objective, model_extensions, LABELS
    with jax.default_device(jax.devices('cpu')[0]):
        start = 1 + burnin_steps
        labels = {key: jnp.asarray(data[key][:, start:]) for key in LABELS}
        mask = jnp.asarray(data['action_mask'][:, start:])
        logits = jnp.zeros(mask.shape, jnp.float32)
        dist = embodied.jax.outs.MaskedCategorical(logits, mask > .5, unimix=.01, allow_fallback=False)
        loss, metrics = actor_objective(dist, labels, mask,
            *(jnp.asarray(data[k][:, start:]) for k in ('is_first', 'is_last', 'is_terminal')),
            model_extensions(data['stepid'].shape[1]-start, burnin_steps,
                             dict(balanced=.2, strict_pair=.3, quality_pair=.4)))
        values = {key: np.asarray(value).tolist() for key, value in metrics.items()}
        if not np.isfinite(float(loss)) or not all(np.isfinite(np.asarray(v)).all() for v in values.values()):
            raise ValueError('MULTI_F3_READONLY_LOSS_NONFINITE')
        return dict(status='PASS_ORIGINAL_F3_OBJECTIVE', fixed_logits=True, loss=float(loss),
                    metrics=values, optimizer_updates=0, model_forward='NOT_RUN')


class _NoModelState:
    """FullCheckpoint protocol fixture, explicitly no parameters or optimizer."""
    def save(self):
        return dict(protocol='DATA_ONLY_NO_AGENT_OR_OPTIMIZER', optimizer_updates=0)

    save_exact = save

    def load(self, state):
        if state != self.save():
            raise ValueError('MULTI_DATA_ONLY_PROBE_STATE_CHANGED')

    load_exact = load


def _wait_pending(prefetch):
    deadline = time.monotonic()+30
    with prefetch.queue.not_empty:
        while prefetch.queue._qsize() < prefetch.amount:
            if any(isinstance(x, str) for x in prefetch.queue.queue):
                raise RuntimeError('MULTI_PREFETCH_PROBE_FAILED')
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise RuntimeError('MULTI_PREFETCH_PROBE_TIMEOUT')
            prefetch.queue.not_empty.wait(min(remaining, 1))


def batch_digest(batch, audit):
    return fingerprint(dict(arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype),
        sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k, v in sorted(batch.items())}, audit=audit))


def checkpoint_probe(config_path, root, *, resume, stop, batch_size=4, length=8,
                     adapter='native', burnin_steps=None):
    """Called in a fresh process; original Checkpoint/Runtime/Prefetch only."""
    import elements
    import embodied
    from embodied.run.fg_checkpoint import FullCheckpoint, RuntimeState, StepSchedule
    from embodied.run.fg_s3_source_admission import runtime_versions
    if type(length) is not int or length < 2:
        raise ValueError('MULTI_CHECKPOINT_PROBE_LENGTH_AT_LEAST_TWO')
    random.seed(13); np.random.seed(17)
    bound = _diagnostic_stream(config_path, adapter=adapter, batch_size=batch_size,
                               length=length, burnin_steps=burnin_steps)
    coverage = bound.coverage()
    prefetch = embodied.streams.Prefetch(bound, amount=2)
    step = elements.Counter()
    schedule = StepSchedule(1, 3)
    samples = [0]
    def no_carry(value):
        if value is not None: raise ValueError('MULTI_PROBE_CARRY_MUST_BE_EMPTY')
    runtime = RuntimeState(schedule, SimpleNamespace(), [prefetch], lambda: None, no_carry, {}, samples)
    identity = dict(protocol='FG_S3_DATA_MULTI_COLD_PROBE_V1', source=coverage['identity_hash'],
        adapter=bound.identity if adapter == 'native' else bound.identity_hash,
        adapter_mode=adapter, burnin_steps=burnin_steps, batch_size=batch_size, length=length, prefetch=2,
        code_sha256=digest(Path(__file__).with_name('fg_s3_e_quality_multi.py')),
        probe_code_sha256=digest(Path(__file__)), packages=runtime_versions(), data_only=True)
    if adapter == 'f3':
        identity.update(f3_code_sha256=digest(Path(__file__).with_name('fg_s3_e_quality_multi_actor.py')),
            f3_adapter_sha256=digest(Path(__file__).with_name('fg_s3_e_quality_actor.py')),
            role_composer_sha256=digest(PROJECT.parent/'dreamerv3/embodied/run/fg_s3_e_quality_actor.py'))
    root = Path(root)
    cp = FullCheckpoint(root/'ckpt', step, identity, [prefetch], reserve_gib=0,
                        keep_all=True, backend='external_immutable_course_reader')
    cp.course_reader = bound; cp.agent = _NoModelState(); cp.step = step; cp.runtime = runtime
    history_path = root/'batch_history.json'
    history = json.loads(history_path.read_text()) if resume else []
    if resume:
        cp.load()
        if len(history) != int(step):
            raise ValueError('MULTI_PROBE_HISTORY_CHECKPOINT_MISMATCH')
    else:
        if root.exists(): raise ValueError('MULTI_PROBE_OUTPUT_MUST_BE_NEW')
        cp.save()
    if not int(step) <= stop:
        raise ValueError('MULTI_PROBE_STOP_BEFORE_RESTORED_COUNT')
    runtime.start()
    while int(step) < stop:
        batch = next(prefetch)
        audit = prefetch.state['last_batch_audit']
        history.append(batch_digest(batch, audit))
        step.increment(); samples[0] += batch_size*length
        schedule.advance(int(step))
    _wait_pending(prefetch)
    cp.save(final=True)
    with prefetch.pause():
        evidence = dict(reader=bound.save(), runtime=runtime.save(), history=history,
            consumed_batches=int(step), optimizer_updates=0,
            checkpoint=str(cp.latest()), aggregate_identity=coverage['identity'])
    history_path.write_text(json.dumps(history)+'\n')
    with (root/('evidence_'+str(stop)+'.pkl')).open('xb') as output:
        pickle.dump(evidence, output, protocol=5)
    return dict(status='DATA_ONLY_CHECKPOINT_PROBE_PASS', consumed_batches=int(step),
                optimizer_updates=0, pending_batches=len(evidence['runtime']['streams'][0]['pending']))


def cold_evidence(config_path, out, *, batch_size, length, adapter='native', burnin_steps=None):
    out = Path(out)
    argv_base = [sys.executable, '-B', '-m', __name__, '--probe-config', str(config_path),
                 '--batch-size', str(batch_size), '--length', str(length), '--adapter', adapter]
    if burnin_steps is not None:
        argv_base.extend(['--burnin-steps', str(burnin_steps)])
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(map(str, [PROJECT, PROJECT/'src', PROJECT.parent/'dreamerv3'])),
           'CUDA_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu', 'PYTHONDONTWRITEBYTECODE': '1',
           'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    calls = []
    for name, resume, stop in [('continuous', False, 6), ('split', False, 3), ('split', True, 6)]:
        argv = [*argv_base, '--probe-out', str(out/name), '--stop', str(stop)]
        if resume: argv.append('--resume')
        run = subprocess.run(argv, env=env, cwd=PROJECT, capture_output=True, text=True)
        label = name+('_resume' if resume else '_start')
        (out/(label+'.log')).write_text(run.stdout+run.stderr)
        calls.append(dict(argv=argv, exit_code=run.returncode, log_sha256=digest(out/(label+'.log'))))
        if run.returncode:
            raise RuntimeError('MULTI_COLD_PROBE_FAILED: '+run.stderr[-3000:]+run.stdout[-2000:])
    with (out/'continuous/evidence_6.pkl').open('rb') as stream: left = pickle.load(stream)
    with (out/'split/evidence_6.pkl').open('rb') as stream: right = pickle.load(stream)
    # Paths name different test outputs; every actual data/RNG/queue field is compared.
    def exact(a, b):
        if isinstance(a, dict):
            if set(a) != set(b): raise ValueError('MULTI_COLD_STATE_KEYS_DIFFER')
            for key in a: exact(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b): raise ValueError('MULTI_COLD_STATE_LENGTH_DIFFERS')
            for x, y in zip(a, b): exact(x, y)
        elif isinstance(a, np.ndarray):
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                raise ValueError('MULTI_COLD_ARRAY_BYTES_DIFFER')
        elif a != b: raise ValueError('MULTI_COLD_STATE_VALUE_DIFFERS')
    exact({k: v for k, v in left.items() if k != 'checkpoint'},
          {k: v for k, v in right.items() if k != 'checkpoint'})
    from embodied.run.fg_checkpoint import validate_bundle
    checkpoint = Path(right['checkpoint']); bundle = validate_bundle(checkpoint)
    if any(p.suffix == '.npz' or p.name == 'replay' for p in checkpoint.rglob('*')):
        raise ValueError('MULTI_CHECKPOINT_COPIED_CORPUS')
    return dict(status='PASS_EXACT_COLD_DATA_STATE', calls=calls, batches=6, split_at=3,
        prefetched_batches=2, scheduler_and_all_reader_states_equal=True,
        all_native_array_bytes_and_provenance_equal=True, runtime_and_pending_queue_equal=True,
        batch_history=right['history'], checkpoint_manifest=bundle, optimizer_updates=0,
        neural_training_checkpoint='NOT_RUN', copied_corpus_chunks=0)


def multi_evidence(config_path, output, *, batch_size=4, length=8, require_published=False,
                   adapter='native', burnin_steps=None):
    from .fg_s3_e_quality_checks import _output
    config_path = _local_path(config_path)
    config = read_config(config_path)
    if type(batch_size) is not int or batch_size < 1 or type(length) is not int or length < 2:
        raise ValueError('MULTI_DIAGNOSTIC_DIMENSIONS_INVALID')
    published = published_sources(config) if require_published else []
    for source in config['sources']:
        _output(output, source['corpus_root'])
    out = _local_path(output)
    stream = _diagnostic_stream(config_path, adapter=adapter, batch_size=batch_size,
                                length=length, burnin_steps=burnin_steps)
    try:
        raw, audit = stream.sample()
        coverage = stream.coverage()
        loss = f3_loss_evidence(raw, burnin_steps) if adapter == 'f3' else None
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/'coverage.json', signed(coverage))
        write_json(out/'sample_audit.json', signed(dict(batch_digest=batch_digest(raw, audit), rows=audit)))
        if loss is not None:
            write_json(out/'F3_READONLY_LOSS.json', signed(loss))
    finally:
        stream.close()
    cold = cold_evidence(config_path, out, batch_size=batch_size, length=length,
                         adapter=adapter, burnin_steps=burnin_steps)
    write_json(out/'CHECKPOINT_COLD_EVIDENCE.json', signed(cold))
    report = signed(dict(protocol='FG_S3_E_MULTICORPUS_READONLY_EVIDENCE_V1',
        status='REAL_READONLY_PASS' if require_published else 'DEV_PASS',
        real_source_acceptance='PASS' if require_published else 'NOT_RUN',
        sources_published=published, identity=coverage['identity'], identity_hash=coverage['identity_hash'],
        adapter_mode=adapter,
        cold_restore=cold['status'], optimizer_updates=0, training_admitted=False,
        new_native_g3='NOT_RUN', model_or_optimizer_checkpoint='NOT_RUN'))
    write_json(out/'DATA_MULTI_REPORT.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data-only original FullCheckpoint probe worker')
    parser.add_argument('--probe-config', required=True)
    parser.add_argument('--probe-out', required=True)
    parser.add_argument('--stop', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--length', type=int, default=8)
    parser.add_argument('--adapter', choices=('native', 'f3'), default='native')
    parser.add_argument('--burnin-steps', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(checkpoint_probe(args.probe_config, args.probe_out, resume=args.resume,
        stop=args.stop, batch_size=args.batch_size, length=args.length,
        adapter=args.adapter, burnin_steps=args.burnin_steps)), flush=True)
