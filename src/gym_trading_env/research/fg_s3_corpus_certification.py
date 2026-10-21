"""Reuse immutable production G3/Reader certificates; never execute a Teacher."""
from __future__ import annotations

import json
from pathlib import Path
import re

from .fg_replay import digest, read_manifest
from .fg_s3_e_quality_course_preflight import _local_path


def certification_reference(value):
    if (not isinstance(value, dict) or set(value) != {'batch_manifest', 'sha256'}
            or re.fullmatch('[0-9a-f]{64}', str(value.get('sha256'))) is None):
        raise ValueError('CORPUS_CERTIFICATION_EXTERNAL_SHA256_REQUIRED')
    return dict(batch_manifest=str(_local_path(value['batch_manifest'])), sha256=value['sha256'])


def published_corpus(source, reference, *, content_identity):
    """Authenticate original evidence and current bytes, without a live verifier.

    The external digest freezes the production release, including its original
    verifier Pin. Consumer Git HEAD is provenance and is never its trust root.
    """
    from .fg_s3_corpus_production import read_batch
    from .fg_s3_corpus_provenance import authenticate_code
    from .fg_s3_e_quality_supervision import verify_native_receipts

    reference = certification_reference(reference)
    path = Path(reference['batch_manifest'])
    if digest(path) != reference['sha256']:
        raise ValueError('CORPUS_CERTIFICATION_BATCH_SHA256_MISMATCH')
    batch = read_batch(path)
    corpus, receipts = Path(source.corpus_root), path.parent
    if (batch.get('protocol') != 'FG_S3_E_TRAIN_CORPUS_BATCH_RESULT_V1'
            or batch.get('status') != 'NATIVE_G3_AND_INDEPENDENT_READER_COMPLETED'
            or batch.get('training_admitted') is not False or batch.get('optimizer_updates') != 0):
        raise ValueError('CORPUS_CERTIFICATION_PUBLICATION_REQUIRED')
    required = {'corpus/corpus_manifest.json', 'corpus/source_lock.json', 'corpus/course_v2.json',
        'corpus/teacher_manifest.json', 'corpus/model_contract.json',
        'receipts/native/native_verification.json', 'receipts/native/native_verify.log',
        'receipts/native/VERIFIER_PIN.json', 'receipts/reader/WP03_COURSE_READER_COVERAGE.json',
        'receipts/reader/join.sqlite', 'receipts/reader/join.json'}
    if not required <= batch.get('files', {}).keys():
        raise ValueError('CORPUS_CERTIFICATION_INCOMPLETE_INVENTORY')
    for name, expected in batch['files'].items():
        kind, relative = name.split('/', 1)
        if kind not in ('corpus', 'receipts') or Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('CORPUS_CERTIFICATION_UNSAFE_PATH')
        target = _local_path((corpus if kind == 'corpus' else receipts)/relative)
        if digest(target) != expected:
            raise ValueError('CORPUS_CERTIFICATION_CONTENT_CHANGED: ' + name)
    manifest = read_manifest(corpus/'corpus_manifest.json')
    locked = read_manifest(receipts/'IDENTITY_LOCK.json')
    locked_fields = ('batch_id', 'corpus_identity', 'source_lock', 'course_identity',
                     'teacher_identity', 'producer_pin_sha256', 'native_verifier_pin_sha256', 'index_sha256')
    if (locked.get('protocol') != 'FG_S3_E_TRAIN_CORPUS_IDENTITY_LOCK_V1'
            or any(locked.get(key) != batch.get(key) for key in locked_fields)
            or locked.get('files') != dict(batch['files'], **{'receipts/BATCH_MANIFEST.json':reference['sha256']})):
        raise ValueError('CORPUS_CERTIFICATION_IDENTITY_LOCK_DIFFERS')
    native = json.loads((receipts/'native/native_verification.json').read_text())
    reader = read_manifest(receipts/'reader/WP03_COURSE_READER_COVERAGE.json')
    joined = read_manifest(receipts/'reader/join.json')
    if (manifest['content_hash'] != batch['corpus_identity']
            or manifest['source_lock_hash'] != batch['source_lock']
            or manifest['course_identity_hash'] != batch['course_identity']
            or manifest['teacher_identity'] != batch['teacher_identity']
            or manifest['code']['repositories'] != batch['producer_repositories']
            or reader['corpus_identity'] != batch['corpus_identity']
            or reader['status'] not in ('PASS_READER_ONLY', 'PASS_READER_CPU')
            or reader.get('training_admitted') is not False or reader['actual_optimizer_exposures'] != 0
            or reader['index_sha256'] != batch['index_sha256']
            or joined['index_sha256'] != batch['index_sha256']
            or batch['files']['receipts/reader/join.sqlite'] != batch['index_sha256']
            or reader['source_identity'] != content_identity or joined['source_identity'] != content_identity
            or native.get('exit_code') != 0
            or native.get('log_sha256') != batch['files']['receipts/native/native_verify.log']):
        raise ValueError('CORPUS_CERTIFICATION_RECEIPTS_DIFFER')
    verify_native_receipts(source, native['result'])
    pin = read_manifest(receipts/'native/VERIFIER_PIN.json')
    pin_sha = batch.get('native_verifier_pin_sha256')
    if (pin.get('protocol') != 'FG_S3_E_QUALITY_PRODUCER_PIN_V1'
            or set(pin.get('repositories', {})) != {'gym', 'dreamer'}
            or batch['files']['receipts/native/VERIFIER_PIN.json'] != pin_sha
            or any(row.get('dirty') is not False
                or re.fullmatch('[0-9a-f]{40}', str(row.get('sha'))) is None
                for row in pin['repositories'].values())):
        raise ValueError('CORPUS_CERTIFICATION_VERIFIER_PIN_INVALID')
    audit = {role: dict(head=row['sha'], runtime_diff_exit_code=0, untracked_runtime=[])
        for role, row in pin['repositories'].items()}
    audit['operator_producer_pin_sha256'] = pin_sha
    if native.get('producer_audit') != audit:
        raise ValueError('CORPUS_CERTIFICATION_VERIFIER_AUDIT_DIFFERS')
    authenticate_code(manifest['code'])
    # Reuse the existing TRAIN whitelist/source authority check. No Env,
    # Teacher, replay of accounts, GPU, optimizer or TEST prices are opened.
    from .fg_s3_e_quality_corpus_cli import source_inputs
    lock = read_manifest(corpus/'source_lock.json')
    actual, _, _ = source_inputs(lock['source_pool']['path'], lock['env_receipt']['path'],
                                lock['contract'], **lock['selection'])
    if actual != lock:
        raise ValueError('CORPUS_CERTIFICATION_ORIGINAL_TRAIN_AUTHORITY_DIFFERS')
    return dict(native, certification=reference, verification_mode='PUBLISHED_NATIVE_G3_REUSED',
                fresh_native_execution=False, published_index=str(receipts/'reader/join.sqlite'))
