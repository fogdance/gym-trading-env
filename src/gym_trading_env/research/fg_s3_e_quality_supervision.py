"""WP02 corpus integrity and WP03 supervision, never training authority.

Development checks content and contracts, not the current checkout's identity.
The producer's historical code inventory remains evidence, not a constraint on
adding a reader. Native verification is an explicit, expensive operation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import time

from dreamerv3 import fg_s3_e_quality_course_contract as course
from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import digest, read_manifest, signed
from .fg_s3_e_observation import observation_contract
from .fg_s3_e_quality_course_preflight import _local_path, native_contracts

PROTOCOL = 'FG_S3_E_QUALITY_SUPERVISION_V1'
POLICIES = ('P0', 'P1', 'P2_low', 'P2_mid', 'P3', 'P4')
STAGES = ('FOUNDATION_WM', 'QUALITY_HEADS', 'ACTOR_SUPERVISED')
PRODUCER_HEADS = dict(gym='e57189bad100465c9fee8d9bbb5601f3e5c376ad',
                      dreamer='44a6cc62032811501631e83be977f07137dc65b8')


def member(root, name):
    """Validate all locations before opening, hashing, or loading arrays."""
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('SUPERVISION_UNSAFE_PATH')
    path = _local_path(Path(root) / name)
    if not path.is_relative_to(Path(root).absolute()):
        raise ValueError('SUPERVISION_UNSAFE_PATH')
    if (any(re.search(r'(^|[^A-Z0-9])(TEST|DEV|VALIDATION)([^A-Z0-9]|$)',part.upper())
            for part in Path(name).parts)
            or Path(name).suffix.lower()=='.csv' and name!='coverage.csv'):
        raise ValueError('SUPERVISION_NON_TRAIN_PATH')
    return path


@dataclass(frozen=True)
class SupervisionSource:
    corpus_root: Path
    corpus_content_hash: str
    course_identity_hash: str
    teacher_identity_hash: str
    observation_contract_hash: str
    source_lock_hash: str
    stage: str = 'FOUNDATION_WM'

    @classmethod
    def from_corpus(cls, root, *, stage='FOUNDATION_WM'):
        root = _local_path(root)
        if any(part.upper() in ('TEST', 'DEV', 'VALIDATION') for part in root.parts):
            raise ValueError('SUPERVISION_NON_TRAIN_PATH')
        m = read_manifest(root / 'corpus_manifest.json')
        return cls(root, m['content_hash'], m['course_identity_hash'],
                   m['teacher_identity'], m['observation']['contract_hash'],
                   m['source_lock_hash'], stage)


@dataclass(frozen=True)
class Admission:
    source: SupervisionSource
    receipt: dict
    # Only preflight's native execution populates this. A loaded JSON PASS is
    # never promoted into evidence that a process was actually executed.
    native_result: dict | None = None


def inspect_source(source):
    root = _local_path(source.corpus_root)
    if any(part.upper() in ('TEST', 'DEV', 'VALIDATION') for part in root.parts):
        raise ValueError('SUPERVISION_NON_TRAIN_PATH')
    if source.stage not in STAGES:
        raise ValueError('SUPERVISION_STAGE_INVALID')
    m = read_manifest(root / 'corpus_manifest.json')
    if (m.get('protocol') != 'FG_S3_E_QUALITY_CORPUS_V2'
            or m.get('fold') != 'TRAIN' or m.get('training_admitted') is not False
            or m.get('behaviors') != list(POLICIES)):
        raise ValueError('SUPERVISION_CORPUS_CONTRACT')
    expected = (m['content_hash'], m['course_identity_hash'], m['teacher_identity'],
                m['observation']['contract_hash'], m['source_lock_hash'])
    actual = (source.corpus_content_hash, source.course_identity_hash,
              source.teacher_identity_hash, source.observation_contract_hash, source.source_lock_hash)
    if actual != expected:
        raise ValueError('SUPERVISION_SOURCE_IDENTITY_MISMATCH')
    paths = {name: member(root, name) for name in m['files']}
    required = {'course_v2.json', 'source_lock.json', 'teacher_manifest.json', 'model_contract.json'}
    # Validate nested paths BEFORE any array/file inventory is hashed.
    from .fg_replay import trace_manifest
    for policy in POLICIES:
        for kind in ('policies', 'student_replay'):
            folder = member(root, f'{kind}/{policy}')
            trace = trace_manifest(folder, verify=False, schema=m['observation']['replay_schema'])
            required.add(f'{kind}/{policy}/manifest.json')
            required.update(f'{kind}/{policy}/{name}' for name in trace['files'])
        member(root, f'teacher_sidecar/{policy}/steps.jsonl.gz')
        required.add(f'teacher_sidecar/{policy}/steps.jsonl.gz')
    if not required <= paths.keys():
        raise ValueError('SUPERVISION_INCOMPLETE_INVENTORY')
    for name, path in paths.items():
        if not path.is_file() or digest(path) != m['files'][name]:
            raise ValueError('SUPERVISION_CONTENT_CHANGED: ' + name)
    lock = read_manifest(root / 'source_lock.json')
    teacher = read_manifest(root / 'teacher_manifest.json')
    model = read_manifest(root / 'model_contract.json')
    spec = course.validate_course_spec(json.loads((root / 'course_v2.json').read_text()))
    if spec['spec_mode'] != 'formal':
        raise ValueError('SUPERVISION_REQUIRES_FORMAL_CORPUS_CONTRACT')
    native_contracts(spec)  # Structural/native contract and unchanged public Guard.
    if (lock.get('fold') != 'TRAIN' or lock['content_hash'] != source.source_lock_hash
            or lock['row'].get('role') != 'train'
            or lock['row'].get('eligible_for_training') is not True
            or fingerprint(course.course_identity_payload(spec)) != source.course_identity_hash
            or m['observation'] != observation_contract()
            or model.get('protocol') != 'FG_S3_E_MODEL_CONTRACT_V2'
            or model['observation'] != observation_contract()
            or model['content_hash'] != m['model_contract_hash']
            or teacher.get('protocol') != 'FG_S3_E_QUALITY_TEACHER_V2'
            or teacher.get('fold') != 'TRAIN'
            or teacher['strict_identity'] != source.teacher_identity_hash
            or teacher['source_lock_hash'] != source.source_lock_hash
            or teacher.get('objective') != 'discounted_native_float32_Quality_TP_feedback'
            or teacher.get('gamma_rational') != [332, 333]
            or teacher.get('reward_schema') != spec['reward']['schema']):
        raise ValueError('SUPERVISION_CONTRACT_IDENTITY_MISMATCH')
    return m


def frozen_producer_audit(producer_gym):
    """Authenticate the historical verifier, only for final native evidence.

    This never inspects or constrains the current development checkout.
    HEAD alone does not prove that the executed producer has those bytes.
    """
    producer_gym=_local_path(producer_gym)
    audit={}
    for role,root in (('gym',producer_gym),('dreamer',producer_gym.parent/'dreamerv3')):
        head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        if head!=PRODUCER_HEADS[role]:raise ValueError('SUPERVISION_NATIVE_PRODUCER_MISMATCH')
        paths=(['src/gym_trading_env','e_data_v1','tools/fg_s3_e_quality_corpus.py','tools/fg_s3_v2_experiment.py']
               if role=='gym' else ['dreamerv3','embodied',':(exclude)embodied/tests/**'])
        diff=subprocess.run(['git','-C',str(root),'diff','--quiet','HEAD','--',*paths],capture_output=True)
        untracked=subprocess.check_output(['git','-C',str(root),'ls-files','--others','--exclude-standard'],text=True).splitlines()
        runtime=[p for p in untracked if Path(p).suffix=='.py' and not
                 set(Path(p).parts)&{'tests','docs','artifacts'}]
        if diff.returncode or runtime:raise ValueError('SUPERVISION_FROZEN_PRODUCER_CODE_CHANGED')
        audit[role]=dict(head=head,runtime_diff_exit_code=diff.returncode,untracked_runtime=runtime)
    return audit


def native_verification(source, *, producer_gym, log_path):
    """Execute the unchanged, exact WP02 producer; no re-signing/migration.

    This is required once for final REAL_DATA evidence, not for editing or
    opening the development reader. Old absolute identities remain historical
    evidence; the new reader binds relative corpus content instead.
    """
    producer_gym = _local_path(producer_gym)
    producer_audit=frozen_producer_audit(producer_gym)
    env = {**os.environ, 'PYTHONPATH': '.:src:../dreamerv3', 'JAX_PLATFORMS': 'cpu',
           'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
           'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    argv = ['/home/v/miniconda3/envs/dreamerv3/bin/python', '-B',
            'tools/fg_s3_e_quality_corpus.py', 'verify', '--corpus', str(source.corpus_root)]
    began = time.monotonic()
    log_path = _local_path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('x') as stream:
        run = subprocess.run(argv, cwd=producer_gym, env=env, stdout=stream, stderr=subprocess.STDOUT)
    lines = log_path.read_text().splitlines()
    if run.returncode != 0:
        raise ValueError(f'SUPERVISION_NATIVE_VERIFY_FAILED: exit={run.returncode}; {log_path}')
    result = json.loads(lines[-1])
    if (result.get('status') != 'PASS' or result.get('G3') != 'GO'
            or result.get('corpus_identity') != source.corpus_content_hash
            or result.get('training_admitted') is not False):
        raise ValueError('SUPERVISION_NATIVE_RECEIPT_MISMATCH')
    if frozen_producer_audit(producer_gym)!=producer_audit:
        raise ValueError('SUPERVISION_FROZEN_PRODUCER_CODE_CHANGED')
    return dict(argv=argv, cwd=str(producer_gym), exit_code=run.returncode,producer_audit=producer_audit,
                seconds=time.monotonic() - began, log_sha256=digest(log_path), result=result)


class QualitySupervisionAdmission:
    @staticmethod
    def preflight(source, *, purpose='development', producer_gym=None, log_path=None):
        if purpose not in ('development', 'native-verification'):
            raise ValueError('SUPERVISION_EXPLICIT_PURPOSE_REQUIRED')
        m = inspect_source(source)
        native = None
        if purpose == 'native-verification':
            if producer_gym is None or log_path is None:
                raise ValueError('SUPERVISION_NATIVE_EXECUTION_ARGUMENTS_REQUIRED')
            native = native_verification(source, producer_gym=producer_gym, log_path=log_path)
            inspect_source(source)  # Reject changes while the native process ran.
        # Content identity is independent of the current filesystem location or
        # consumer Git SHA/dirty state. Legacy identities are preserved links.
        identity = dict(protocol=PROTOCOL, corpus_identity=source.corpus_content_hash,
            files=m['files'], course_identity=source.course_identity_hash,
            teacher_identity=source.teacher_identity_hash,
            observation_contract=source.observation_contract_hash, source_lock=source.source_lock_hash)
        receipt = signed(dict(protocol=PROTOCOL, purpose=purpose, stage=source.stage,
            status='SUPERVISION_DATA_READY' if native else 'DEVELOPMENT_VALIDATED',
            source_status='SOURCE_VERIFIED' if native else 'INTEGRITY_CHECKED_ONLY',
            admission_status='NOT_ADMITTED', training_admitted=False,
            prepared_certified=False, checkpoint_certified=False,
            corpus_root=str(source.corpus_root), content_identity=fingerprint(identity),
            identity=identity, native_receipt_hash=fingerprint(native) if native else None,
            blockers=['E_QUALITY_FORMAL_TRAIN_GUARD_CLOSED'] +
                ([] if native else ['NATIVE_G3_NOT_RUN_IN_THIS_ADMISSION']),
            historical_producer=m.get('code', {}).get('repositories', {})))
        return Admission(source, receipt, native)


def project_label(label, row, step, owner):
    """Project certified facts independently; never teach from behavior action.

    No Q/Soft target is authorized by this adapter. Invalid numeric targets
    are zeroed *before* entering JAX, including independently invalid windows.
    """
    import numpy as np
    from .fg_s3_quality_catalog_v2 import FIELDS, validate_corpus_label
    validate_corpus_label(label)
    facts, strict = label['L1'], label['L2']
    out = {k:np.zeros(shape, dtype) for k,(dtype,shape) in FIELDS.items()}
    out.update(teacher_action=np.array(0,np.int32), teacher_valid=np.array(False,bool))
    completed = (row['decision_executed'] and row['decision_valid_minute']
        and not any(row[k] for k in ('reset','is_first','is_last','is_terminal')))
    dual = bool(np.array_equal(step['action_mask'], [1,1]))
    eligible = bool(completed and dual and facts['actual_legal_wait'] and facts['actual_legal_add'])
    if not eligible:
        return out
    if strict['teacher_valid']:
        out['teacher_valid'][...] = True
        out['teacher_action'][...] = int(strict['strict_action'])
    out['quality_outcome_valid'][...] = bool(facts['immediate_outcome_verified'])
    score_valid = bool(facts['immediate_outcome_verified'] and facts['score_valid'])
    out['quality_score_valid'][...] = score_valid
    if score_valid:
        score = float(facts['score'])
        if not np.isfinite(score) or not 0 <= score <= 100:
            raise ValueError('SUPERVISION_SCORE_TARGET_INVALID')
        out['quality_score'][...] = score
    if facts['immediate_outcome_verified']:
        out['quality_tp'][...] = facts.get('exit_type') == 'TP5'
        out['quality_sl'][...] = facts.get('exit_type') == 'SL30'
        early = facts.get('first_adverse',{}).get('10')
        out['quality_early5'][...] = early is not None and early <= 5
    for name, classes in [('event',4), ('touch',2), ('adverse',2)]:
        valid = np.asarray(facts[name+'_valid'],bool)
        value = np.asarray(facts[name])
        safe = np.where(valid,value,0)
        if (not np.isfinite(safe).all() or not np.isin(safe,np.arange(classes)).all()):
            raise ValueError('SUPERVISION_WINDOW_TARGET_INVALID: ' + name)
        out['quality_'+name][...] = safe
        out['quality_'+name+'_valid'][...] = valid
    supported = bool(out['teacher_valid'] or out['quality_outcome_valid'] or
        any(np.any(out['quality_'+k+'_valid']) for k in ('event','touch','adverse')))
    if supported:
        owner_array = np.frombuffer(bytes.fromhex(owner), '>u4').astype(np.uint32)
        if owner_array.shape != (8,) or not owner_array.any():
            raise ValueError('SUPERVISION_NONZERO_OWNER_REQUIRED')
        out['quality_owner'][...] = owner_array
    return out


def supervision_settings():
    """Loss-only backend settings, with all prediction/Q/decision gates closed."""
    return dict(quality_supervision_protocol=PROTOCOL,
        quality_q_c1_lambda=0., quality_q_v2_lambda=0., quality_soft_lambda=0.,
        quality_q_admitted=False, quality_prediction_admitted=False,
        quality_decision_admitted=False, quality_fuse_outcome=False, quality_fuse_score=False)
