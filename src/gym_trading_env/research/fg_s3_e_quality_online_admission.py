"""F4 source bindings and external native audit attempts; no training authority."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import uuid

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import digest, signed, write_json
from .fg_s3_e_quality_course_preflight import _local_path
from .fg_s3_e_quality_supervision import (
    SupervisionSource, frozen_producer_audit, require_stage_producer_pin,
    require_prepared_verifier, verifier_binding)

ANCHOR = 'F3_ANCHOR'
PROTOCOL = 'FG_S3_E_QUALITY_ONLINE_SOURCE_BINDINGS_V1'
_AUDITS = {}


@dataclass(frozen=True)
class NativeSourceEvidence:
    report: dict


def require_native_audit(evidence, bindings):
    registered = _AUDITS.get(id(evidence))
    if (registered is None or registered[0] is not evidence
            or registered[1] != fingerprint(evidence.report)
            or evidence.report['source_bindings'] != bindings):
        raise ValueError('ONLINE_ACTUAL_NATIVE_SOURCE_AUDIT_REQUIRED')
    if digest(_local_path(evidence.report['path'])) != evidence.report['sha256']:
        raise ValueError('ONLINE_NATIVE_SOURCE_AUDIT_CHANGED')
    return copy.deepcopy(evidence.report)


def source_slots(settings):
    """Ordered roles, before opening any model or creating output directories."""
    slots = [ANCHOR]
    if settings['history_config'] is not None:
        from .fg_s3_e_quality_multi_checks import read_config
        config = read_config(settings['history_config'])
        if config['stage'] != 'ACTOR_STRICT':
            raise ValueError('ONLINE_HISTORY_F3_STAGE_REQUIRED')
        slots.extend(source['corpus_id'] for source in config['sources'])
    if (any(not isinstance(slot, str) or not slot for slot in slots)
            or len(set(slots)) != len(slots)):
        raise ValueError('ONLINE_UNIQUE_SOURCE_ROLE_REQUIRED')
    return slots


def resolve_verifiers(settings, *, producer_gym=None, producer_lock=None, producer_lock_sha256=None,
                      saved=None):
    """Explicit common pin or complete per-source map; never historical fallback."""
    slots = source_slots(settings)
    configured = settings.get('source_verifiers')
    common = dict(producer_gym=producer_gym, producer_lock=producer_lock,
                  producer_lock_sha256=producer_lock_sha256)
    if configured is not None:
        if any(value is not None for value in common.values()):
            raise ValueError('ONLINE_AMBIGUOUS_COMMON_AND_SOURCE_VERIFIERS')
        if type(configured) is not dict or set(configured) != set(slots):
            raise ValueError('ONLINE_COMPLETE_SOURCE_VERIFIER_MAP_REQUIRED')
        raw = configured
    else:
        raw = {slot: common for slot in slots}
    result = {}
    for slot in slots:
        options = raw[slot]
        if type(options) is not dict or set(options) != set(common):
            raise ValueError('ONLINE_EXPLICIT_SOURCE_VERIFIER_FIELDS_REQUIRED')
        require_stage_producer_pin('native-verification', options['producer_lock'],
                                   options['producer_lock_sha256'])
        if options['producer_gym'] is None:
            raise ValueError('SUPERVISION_NATIVE_EXECUTION_ARGUMENTS_REQUIRED')
        options = {**options, 'producer_gym': str(_local_path(options['producer_gym'])),
                   'producer_lock': str(_local_path(options['producer_lock']))}
        audit = frozen_producer_audit(**options)
        result[slot] = dict(options=options,
            verifier_binding=verifier_binding(dict(producer_audit=audit)))
    if saved is not None:
        if type(saved) is not dict or set(saved) != set(slots):
            raise ValueError('ONLINE_PREPARED_VERIFIER_ORDER_CHANGED')
        for slot in slots:
            # Paths may relocate; the externally reviewed bytes and code cannot.
            require_prepared_verifier(saved[slot], **result[slot]['options'])
    return result


def source_bindings(settings, proof, verifiers):
    from .fg_s3_e_quality_multi_checks import read_config
    roots = [(ANCHOR, proof['identity']['preparation']['settings']['corpus_root'])]
    if settings['history_config'] is not None:
        roots.extend((source['corpus_id'], source['corpus_root'])
            for source in read_config(settings['history_config'])['sources'])
    result = []
    for slot, root in roots:
        source = SupervisionSource.from_corpus(root, stage='ACTOR_SUPERVISED')
        result.append(dict(source_id=slot, source_role='anchor' if slot == ANCHOR else 'history',
            corpus_root=str(source.corpus_root), corpus_identity=source.corpus_content_hash,
            course_identity=source.course_identity_hash, teacher_identity=source.teacher_identity_hash,
            observation_contract=source.observation_contract_hash, source_lock=source.source_lock_hash,
            verifier_binding=copy.deepcopy(verifiers[slot]['verifier_binding'])))
    return result


class NativeAuditAttempt:
    """Unique external logs, with source identity checked before/after native calls."""

    def __init__(self, prepared_root, prepared_identity, run_root, bindings, verifiers):
        self.bindings = copy.deepcopy(bindings)
        self.verifiers = verifiers
        self.prepared_identity = prepared_identity
        prepared = _local_path(prepared_root)
        audit = _local_path(prepared.parent/'audit_stage')
        protected = [prepared, _local_path(run_root),
            *[_local_path(binding['corpus_root']) for binding in bindings]]
        for path in protected:
            if path.is_relative_to(audit) or audit.is_relative_to(path):
                raise ValueError('ONLINE_AUDIT_OVERLAPS_PROTECTED_OUTPUT_OR_SOURCE')
        self.root = audit/prepared_identity/uuid.uuid4().hex
        self.records = []

    def arguments(self, slot):
        matches = [b for b in self.bindings if b['source_id'] == slot]
        if len(matches) != 1:
            raise ValueError('ONLINE_UNKNOWN_NATIVE_SOURCE_ROLE')
        binding = matches[0]
        log = self.root/binding['source_role']/fingerprint(slot)/'native.log'
        return {**self.verifiers[slot]['options'], 'log_path': log}

    def record(self, slot, admission):
        native = admission.native_result
        binding = next(b for b in self.bindings if b['source_id'] == slot)
        actual = admission.source
        if (native is None or native['exit_code'] != 0
                or native['result'].get('G3') != 'GO'
                or actual.corpus_content_hash != binding['corpus_identity']
                or verifier_binding(native) != binding['verifier_binding']):
            raise ValueError('ONLINE_NATIVE_SOURCE_BINDING_MISMATCH')
        log = self.arguments(slot)['log_path']
        if digest(log) != native['log_sha256']:
            raise ValueError('ONLINE_NATIVE_AUDIT_LOG_CHANGED')
        self.records.append(dict(source=copy.deepcopy(binding), log=str(log),
            log_sha256=native['log_sha256'], native=copy.deepcopy(native)))

    def finish(self, error=None):
        expected = [b['source_id'] for b in self.bindings]
        actual = [r['source']['source_id'] for r in self.records]
        if error is None and (len(actual) != len(expected) or set(actual) != set(expected)):
            raise ValueError('ONLINE_NATIVE_AUDIT_INCOMPLETE')
        self.root.mkdir(parents=True, exist_ok=True)
        receipt = signed(dict(protocol=PROTOCOL, prepared_identity=self.prepared_identity,
            status='PASS' if error is None else 'FAIL', error=None if error is None else str(error),
            sources=self.records, training_admitted=False))
        with (self.root/'native_admission.json').open('x') as stream:
            import json
            json.dump(receipt, stream, sort_keys=True, indent=2)
            stream.write('\n')
        report = dict(path=str(self.root/'native_admission.json'),
            sha256=digest(self.root/'native_admission.json'), source_bindings=self.bindings)
        if error is not None:
            return report
        evidence = NativeSourceEvidence(report)
        _AUDITS[id(evidence)] = (evidence, fingerprint(report))
        return evidence
