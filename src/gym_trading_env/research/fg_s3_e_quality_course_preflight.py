"""Read-only native WP01 diagnostics, never Prepared or SourceAdmission.

Only explicit JSON metadata references are read. Market bytes, labels, model
parameters, collectors and trainers are outside this adapter's authority.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from dreamerv3 import fg_s3_e_quality_course_contract as course
from dreamerv3.fg_s3_composition import resolve_composition, require_ready
from e_data_v1.io import git_identity, independent_output, sha256
from e_data_v1.reference import load_legacy
from e_data_v1.schema import EDataConfig
from e_data_v1.sources import pool_entries
from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.rewards.single3_reward import Single3RewardConfig
from .fg_replay import read_manifest, signed, write_json
from .fg_s3_e_observation import observation_contract
from .fg_s3_quality_score import PROTOCOL as SCORE_POLICY


PROJECT = Path(__file__).resolve().parents[3]
REPORT_SCHEMA = 'FG_S3_E_QUALITY_COURSE_PREFLIGHT_V2'
RECEIPT_NAME = 'course_contract_receipt.json'
RESERVED_OUTPUT_PARTS = {'prepared', 'runs', 'run', 'ckpt', 'checkpoint', 'checkpoints', 'corpus', 'corpora'}


def _local_path(value):
    if (not isinstance(value, (str, Path)) or not str(value).strip() or
            str(value) != str(value).strip() or '\x00' in str(value) or
            '\\' in str(value) or '://' in str(value) or '..' in Path(value).parts):
        raise ValueError('COURSE_PATH_INVALID: explicit local path without parent traversal required')
    path = Path(value).expanduser().absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('COURSE_PATH_INVALID: symlink path refused')
    return path


def _json_path(value):
    path = _local_path(value)
    if path.suffix != '.json' or not path.is_file():
        raise ValueError('COURSE_PATH_INVALID: existing JSON metadata file required')
    return path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('COURSE_JSON_INVALID: duplicate key ' + key)
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError('COURSE_JSON_INVALID: nonfinite number ' + value)


def _read_json(path):
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_object, parse_constant=_nonfinite)


def _metadata_reference(reference, base, *, native_signed):
    path = Path(reference['path'])
    if not path.is_absolute():
        path = base / path
    path = _json_path(path)
    if sha256(path) != reference['sha256']:
        raise ValueError('COURSE_SOURCE_IDENTITY_MISMATCH: metadata SHA256')
    value = _read_json(path)
    if type(value) is not dict:
        raise ValueError('COURSE_SOURCE_INVALID: metadata must be an object')
    def reject_approval(item):
        if isinstance(item, dict):
            if set(item) & {'certified', 'approved', 'ready', 'admitted', 'execution_admitted', 'training_authorized'}:
                raise ValueError('COURSE_SOURCE_INVALID: approval claims are not certification evidence')
            for child in item.values(): reject_approval(child)
        elif isinstance(item, list):
            for child in item: reject_approval(child)
    reject_approval(value)
    if native_signed and read_manifest(path) != value:
        raise ValueError('COURSE_SOURCE_IDENTITY_MISMATCH: native signed metadata')
    if sha256(path) != reference['sha256']:
        raise ValueError('COURSE_SOURCE_IDENTITY_MISMATCH: metadata changed during inspection')
    return path, value


def native_contracts(validated):
    """Call native validators and schema producers; no Env or Agent is built."""
    reward = Single3RewardConfig(**validated['reward'])
    if asdict(reward) != validated['reward']:
        raise ValueError('COURSE_NATIVE_REWARD_MISMATCH')
    obs = observation_contract()
    wanted = validated['observation']
    if (obs['observation_schema'] != wanted['schema'] or
            obs['encoder_keys'] != wanted['policy_observation_fields'] or
            obs['context_fields'] != wanted['context_fields'] or
            obs['macd_5m'] != wanted['macd_5m'] or
            any(obs['fields'].get(k) != v for k, v in wanted['fields'].items())):
        raise ValueError('COURSE_NATIVE_OBSERVATION_MISMATCH')
    e_schema = EDataConfig().schema(load_legacy()[1])
    if (e_schema['schema'] != wanted['visibility']['data_schema'] or
            e_schema['availability'] != wanted['visibility']['availability'] or
            e_schema['clock'] != reward.clock_protocol or
            e_schema['context_fields'] != wanted['context_fields']):
        raise ValueError('COURSE_NATIVE_E_VISIBILITY_MISMATCH')
    claimed = validated['data_identity']['e_profile_schema_sha256']
    if claimed is not None and claimed != e_schema['schema_sha256']:
        raise ValueError('COURSE_NATIVE_E_PROFILE_IDENTITY_MISMATCH')
    if validated['teacher']['score_policy'] != SCORE_POLICY:
        raise ValueError('COURSE_NATIVE_TEACHER_POLICY_MISMATCH')
    composition = resolve_composition(dict(composition=validated['composition']))
    try:
        require_ready(composition)
    except ValueError as exc:
        guard = str(exc)
    else:
        raise ValueError('COURSE_PUBLIC_GUARD_CHANGED: WP01 requires separate execution review')
    if guard != 'S3_QUALITY_E_NOT_IMPLEMENTED':
        raise ValueError('COURSE_PUBLIC_GUARD_CHANGED: ' + guard)
    return dict(reward=dict(config=asdict(reward), config_hash=reward.hash),
        e_observation=obs, e_data_schema=e_schema,
        teacher=dict(kind=validated['teacher']['kind'], score_policy=SCORE_POLICY, fold='TRAIN'),
        public_composition=dict(resolved=composition, guard=guard, execution_admitted=False))


def _source_metadata(validated, base):
    """Inspect declared metadata only; never authenticate underlying market bytes."""
    data = validated['data_identity']
    checked, inputs = {}, []
    if data['source_manifest'] is not None:
        path, _ = _metadata_reference(data['source_manifest'], base, native_signed=False)
        try:
            entries, record = pool_entries(path, data['contracts'])
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError('COURSE_SOURCE_INVALID: malformed native TRAIN pool metadata') from exc
        if any(row.get('eligible_for_training') is not True for row in entries):
            raise ValueError('COURSE_SOURCE_INVALID: explicit native TRAIN eligibility required')
        if record['sha256'] != data['source_manifest']['sha256']:
            raise ValueError('COURSE_SOURCE_IDENTITY_MISMATCH: TRAIN pool changed')
        if data['market_sources'] is None:
            checked['source_manifest'] = 'TRAIN_MEMBERS_METADATA_ONLY; DECLARED_SOURCE_HASHES_PENDING'
        else:
            index = {r['contract']: r for r in entries}
            for row in data['market_sources']:
                native = index[row['contract']]
                if any(row[key] != native.get(native_key) for key, native_key in (
                        ('source_sha256', 'sha256'), ('env_config_sha256', 'env_sha256'),
                        ('metadata_sha256', 'contract_sha256'))):
                    raise ValueError('COURSE_SOURCE_IDENTITY_MISMATCH: selected TRAIN member')
            checked['source_manifest'] = 'TRAIN_MEMBERS_AND_DECLARED_HASHES_MATCH_METADATA_ONLY'
        inputs.append(path)
    references = [('teacher_manifest', data['teacher_manifest'])]
    references.extend(('oracle:' + b['behavior_id'], b['parameters']['source'])
        for b in validated['curriculum']['behaviors'] if b['kind'] == 'quality_oracle_schedule')
    for name, reference in references:
        if reference is None:
            continue
        path, meta = _metadata_reference(reference, base, native_signed=True)
        if (meta.get('fold') != 'TRAIN' or meta.get('reward_schema') != validated['reward']['schema'] or
                meta.get('score_policy') != validated['teacher']['score_policy']):
            raise ValueError('COURSE_SOURCE_IDENTITY_MISMATCH: TRAIN/Reward/Score metadata binding')
        checked[name] = 'NATIVE_CONTENT_HASH_AND_DECLARED_BINDING_ONLY; NOT_CERTIFIED'
        inputs.append(path)
    return dict(fold='TRAIN', metadata_checks=checked, source_bytes_verified=False,
        time_bounds_verified=False, official_calendar_verified=False, authenticated=False), inputs


def _output_directory(value, inputs, source_roots):
    path = _local_path(value)
    if set(path.parts) & RESERVED_OUTPUT_PARTS:
        raise ValueError('COURSE_PATH_INVALID: report cannot be written into production artifact directories')
    if any(path.is_relative_to(root) for root in source_roots):
        raise ValueError('COURSE_PATH_INVALID: report cannot be written into a referenced source directory')
    return independent_output(path, inputs=inputs)


def preflight(spec, *, report_dir=None):
    """Return deterministic diagnostics; optionally persist a valid draft only.

    exit_code=0: explicit valid draft; 1: structurally valid formal but not
    admitted. Invalid specifications/paths/identity raise ValueError/OSError,
    mapped by the CLI to exit 2. No failure creates a report directory.
    """
    spec_path = _json_path(spec)
    spec_hash = sha256(spec_path)
    validated = course.validate_course_spec(_read_json(spec_path))
    expected_dreamer = PROJECT.parent / 'dreamerv3/dreamerv3'
    if Path(course.__file__).resolve().parent != expected_dreamer:
        raise ValueError('COURSE_FOREIGN_IMPORT: use the sibling Dreamer worktree')
    native = native_contracts(validated)
    sources, inputs = _source_metadata(validated, spec_path.parent)
    output = (_output_directory(report_dir, [spec_path, *inputs], [p.parent for p in inputs])
              if report_dir is not None else None)
    provenance = {}
    for name, root in (('gym', PROJECT), ('dreamer', PROJECT.parent / 'dreamerv3')):
        record = git_identity(root)
        provenance[name] = {k: v for k, v in record.items() if k != 'path'}
    if sha256(spec_path) != spec_hash:
        raise ValueError('COURSE_SOURCE_IDENTITY_MISMATCH: specification changed during inspection')
    readiness = course.inspect_course_readiness(validated)
    exit_code = 0 if validated['spec_mode'] == 'draft' else 1
    report = signed(dict(schema=REPORT_SCHEMA, course_schema=course.COURSE_SCHEMA,
        report_kind='NON_AUTHENTICATED_DIAGNOSTIC', spec_mode=validated['spec_mode'], exit_code=exit_code,
        spec_input_sha256=spec_hash, course_identity_hash=fingerprint(course.course_identity_payload(validated)),
        repositories=provenance,
        normalized_course={k: v for k, v in validated.items() if k != 'stage_contracts'}, native_contracts=native,
        source_readiness=sources, **readiness))
    if output is not None and exit_code == 0:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / RECEIPT_NAME, report)
    return report
