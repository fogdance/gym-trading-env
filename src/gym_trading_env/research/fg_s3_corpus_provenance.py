"""Authenticate historical code as evidence, separately from executing code.

No historical checkout is opened or executed. Native G3 still reconstructs
every label and account with the current, independently pinned verifier.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import subprocess

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import check_files

PROJECT = Path(__file__).resolve().parents[3]


def code_location(name):
    path = Path(name)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Unsafe historical code path')
    for role, folder in (('gym', 'gym-trading-env'), ('dreamer', 'dreamerv3')):
        if folder in path.parts:
            position = path.parts.index(folder)
            relative = Path(*path.parts[position+1:]).as_posix()
            if runtime_file(role, relative):
                return role, relative
    raise ValueError('Unknown historical runtime code path: '+name)


def runtime_file(role, relative):
    parts = Path(relative).parts
    if not parts or '..' in parts or 'tests' in parts or not relative.endswith('.py'):
        return False
    if role == 'gym':
        return (relative.startswith(('src/gym_trading_env/', 'e_data_v1/')) or
                relative in ('tools/fg_s3_e_quality_corpus.py', 'tools/fg_s3_v2_experiment.py'))
    return role == 'dreamer' and relative.startswith(('dreamerv3/', 'embodied/'))


def authenticate_code(code, *, project=PROJECT):
    """Check the complete original inventory against immutable Git blobs."""
    if (set(code.get('repositories', {})) != {'gym', 'dreamer'}
            or not code.get('files') or fingerprint(code['files']) != code.get('execution_source_hash')):
        raise ValueError('Historical code inventory identity differs')
    inventories = {'gym': {}, 'dreamer': {}}
    for name, expected in code['files'].items():
        role, relative = code_location(name)
        if relative in inventories[role] or re.fullmatch('[0-9a-f]{64}', str(expected)) is None:
            raise ValueError('Ambiguous historical code inventory')
        inventories[role][relative] = expected
    counts = {}
    for role, root in (('gym', Path(project)), ('dreamer', Path(project).parent/'dreamerv3')):
        record = code['repositories'][role]
        sha = record.get('sha')
        if record.get('dirty') is not False or re.fullmatch('[0-9a-f]{40}', str(sha)) is None:
            raise ValueError('Historical Producer repository identity invalid')
        tree = subprocess.run(['git', '-C', str(root), 'ls-tree', '-rz', sha],
                              capture_output=True, check=True).stdout
        objects = {}
        for item in tree.split(b'\0'):
            if not item:
                continue
            metadata, raw_name = item.split(b'\t', 1)
            mode, kind, blob = metadata.decode().split()
            name = raw_name.decode()
            if runtime_file(role, name):
                if kind != 'blob' or mode not in ('100644', '100755'):
                    raise ValueError('Historical runtime must contain regular files')
                objects[name] = blob
        if set(objects) != set(inventories[role]):
            raise ValueError('Historical Git/runtime inventory differs: '+role)
        names = sorted(objects)
        payload = ''.join(objects[name]+'\n' for name in names).encode()
        blobs = subprocess.run(['git', '-C', str(root), 'cat-file', '--batch'], input=payload,
                               capture_output=True, check=True).stdout
        offset = 0
        for name in names:
            end = blobs.index(b'\n', offset)
            blob, kind, size = blobs[offset:end].split()
            if kind != b'blob' or blob.decode() != objects[name]:
                raise ValueError('Historical Git object differs')
            offset = end+1
            content = blobs[offset:offset+int(size)]
            if hashlib.sha256(content).hexdigest() != inventories[role][name]:
                raise ValueError('Historical code SHA256 differs: '+name)
            offset += int(size)+1
        counts[role] = len(names)
    return dict(protocol='FG_S3_CORPUS_CODE_PROVENANCE_V1',
                repositories=code['repositories'], execution_source_hash=code['execution_source_hash'],
                files_verified=counts, method='IMMUTABLE_GIT_BLOBS_NO_HISTORICAL_CHECKOUT')


@dataclass(frozen=True)
class SourceVerification:
    """Original identity metadata plus independently checked live guard files."""
    source: dict
    guard_files: dict
    historical_hashes: dict
    report: dict

    @classmethod
    def create(cls, manifest, current_code, *, project=PROJECT):
        report = authenticate_code(manifest['code'], project=project)
        historical = manifest['code']['files']
        source = manifest['source_binding']
        if any(source['files'].get(name) != sha for name, sha in historical.items()):
            raise ValueError('Historical code/source binding differs')
        data_files = {name: sha for name, sha in source['files'].items() if name not in historical}
        if set(data_files) & set(current_code['files']):
            raise ValueError('Market/source asset overlaps executable code')
        guards = {**data_files, **current_code['files']}
        check_files(guards)
        by_location = {code_location(name): sha for name, sha in historical.items()}
        replacements = {}
        for name, sha in current_code['files'].items():
            old = by_location.get(code_location(name))
            if old is not None:
                replacements.setdefault(sha, set()).add(old)
        report = dict(report, current_repositories=current_code['repositories'],
            current_execution_source_hash=current_code['execution_source_hash'],
            original_source_hash=fingerprint(source), runtime_guard_hash=fingerprint(guards),
            identity_policy='HISTORICAL_PROVENANCE_ONLY_ALL_NATIVE_FIELDS_RECOMPUTED')
        return cls(copy.deepcopy(source), guards, replacements, report)

    def descriptor(self, value):
        """Reconstruct provenance fields only; never replace a financial fact."""
        if isinstance(value, list):
            return [self.descriptor(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            if key.endswith('code_sha256') and isinstance(item, str) and item in self.historical_hashes:
                choices = self.historical_hashes[item]
                if len(choices) != 1:
                    raise ValueError('Ambiguous historical implementation descriptor')
                result[key] = next(iter(choices))
            else:
                result[key] = self.descriptor(item)
        return result

    def teacher(self, env, factory):
        from .fg_s3_quality_teacher_v2 import NativeQualityTeacher
        live_source = dict(self.source, files=self.guard_files)
        teacher = NativeQualityTeacher(env, factory, fold='TRAIN', source=live_source)
        # The original constructor establishes all native rules, market,
        # account isolation and live guards. Only provenance is represented in
        # its historical namespace so full signed labels can be compared.
        teacher.source = copy.deepcopy(self.source)
        teacher.identity = self.descriptor(dict(teacher.identity, source=self.source))
        teacher.identity_hash = fingerprint(teacher.identity)
        return teacher

    def strict(self, teacher):
        teacher.query.identity = self.descriptor(teacher.query.identity)
        teacher.query.identity_hash = fingerprint(teacher.query.identity)
        teacher.identity = self.descriptor(teacher.identity)
        teacher.identity['base_identity'] = teacher.query.identity_hash
        teacher.identity_hash = fingerprint(teacher.identity)
        return teacher
