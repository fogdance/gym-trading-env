"""Authenticate frozen provenance without importing a trainer or model."""
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from . import PROJECT, SCHEMA
from .io import LIMIT, TrackingError, digest, fingerprint, read_json, small_bytes
from .redaction import redact

FAMILIES = {'a2': 's3-a2', 'tutor': 's3-tutor-cd', 'b2': 's3-b2', 'e': 's3-e'}


def stable_id(*parts):
    return hashlib.sha256(('\0'.join((SCHEMA, *parts))).encode()).hexdigest()[:20]


def git_info(path):
    def run(*args):
        return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()
    try:
        return dict(sha=run('rev-parse', 'HEAD'), dirty=run('status', '--porcelain').splitlines())
    except (OSError, subprocess.CalledProcessError):
        return dict(sha=None, dirty=None)


class SourceRun:
    def __init__(self, root, prepared=None, *, family=None, limit=LIMIT):
        self.root = Path(root).resolve()
        self.limit = limit
        self.identity_document = read_json(self.root/'run_identity.json', signed=True, limit=limit)
        self.identity = identity = self.identity_document['identity']
        if identity.get('protocol') != 'FG_S3_V2_LIVE_LEARNING_V1':
            raise TrackingError('Unsupported trainer protocol; do not infer its log semantics')
        self.run_id = identity['run_id']
        self.family = family or FAMILIES.get(identity['mode'])
        if not self.family:
            raise TrackingError('Unknown experiment family; specify --family explicitly')
        expected = identity['preparation_sha256']
        if prepared:
            candidates = [Path(prepared).resolve()/'manifest.json']
        else:
            roots = [Path(p) for p in identity.get('versions', {}) if Path(p).name == 'gym-trading-env']
            candidates = [p for r in roots for p in (r/'artifacts/fg_s3_v2/prepared').glob('*/manifest.json')]
        matched = [p for p in candidates if p.is_file() and digest(p) == expected]
        if len(matched) != 1:
            raise TrackingError('Cannot identify exactly one authenticated preparation; use --prepared')
        self.prepared = matched[0].parent
        self.preparation = preparation = read_json(matched[0], signed=True, limit=limit)
        for name, sha in preparation['files'].items():
            target = (self.prepared/name).resolve()
            if not target.is_relative_to(self.prepared) or digest_bounded(target, limit) != sha:
                raise TrackingError(f'Frozen preparation file differs: {name}')
        self.settings = read_json(self.prepared/'settings.json', signed=True, limit=limit)['settings']
        if preparation.get('code') != identity.get('code') or preparation['versions'] != identity['versions']:
            raise TrackingError('Run and preparation source identities differ')
        for field in ('oracle_identity', 'training_identity'):
            if preparation.get(field) != identity.get(field):
                raise TrackingError(f'Run and preparation {field} differ')
        # Copies of a historical run are aliases, not new experiments.
        authoritative = Path(self.settings['run_dir']).resolve()
        self.requested_root = self.root
        if authoritative != self.root:
            original = read_json(authoritative/'run_identity.json', signed=True, limit=limit)
            if original != self.identity_document:
                raise TrackingError('Historical alias conflicts with its original Run identity')
            self.root = authoritative
        for path, sha in identity.get('immutable_files', {}).items():
            if digest_bounded(path, limit) != sha:
                raise TrackingError('Immutable source Run metadata differs')
        self.model = yaml.safe_load(small_bytes(self.root/'config.yaml', limit))
        if fingerprint(self.model) != identity['model_config_sha256']:
            raise TrackingError('Run model configuration differs from frozen identity')
        self.frozen_model = read_json(self.prepared/'model_config.json', signed=True, limit=limit)
        if self.frozen_model['config'] != self.model:
            raise TrackingError('Prepared model configuration differs from Run config')
        if identity.get('agent_extension') != self.frozen_model.get('agent_extension'):
            # Old A2 preparations may store an empty extension rather than omit it.
            if identity.get('agent_extension') or self.frozen_model.get('agent_extension'):
                raise TrackingError('Frozen agent extension differs')
        self.experiment = yaml.safe_load(small_bytes(self.prepared/'experiment.yaml', limit))
        self.clock = preparation.get('clock_protocol') or self.settings.get('clock_protocol')
        self.ids = {'train': stable_id('train', self.run_id), 'live_monitor': stable_id('live', self.run_id)}

    def trace(self):
        versions = {Path(k).name: v for k, v in self.identity['versions'].items()}
        dirty = {Path(k).name: v for k, v in self.preparation.get('dirty', {}).items()}
        receipt_path = self.root/'training_receipt.json'
        receipt = read_json(receipt_path, signed=True, limit=self.limit) if receipt_path.exists() else {}
        return dict(schema=SCHEMA, project=PROJECT, local_run_id=self.run_id,
            source_run_dir=str(self.root), source_prepared_dir=str(self.prepared),
            requested_source_alias=str(self.requested_root), family=self.family,
            mode=self.identity['mode'], scope=self.identity.get('scope'),
            clock_protocol=self.clock, versions=versions, dirty_at_prepare=dirty,
            source_identity_sha256=fingerprint(self.identity), wandb_ids=self.ids,
            preparation_sha256=self.identity['preparation_sha256'],
            run_identity_sha256=digest(self.root/'run_identity.json'),
            oracle_identity=self.identity['oracle_identity'],
            training_identity=self.identity['training_identity'],
            source_model_identity=self.identity['source_model_identity'],
            source_update_offset=self.identity['source_update_offset'],
            source_model_updates=self.preparation.get('source_model_updates'),
            source_training_status=receipt.get('status', 'NO_COMPLETION_RECEIPT_OBSERVED'),
            source_start_timestamp=None, source_start_timestamp_status='NOT_RECORDED_IN_FROZEN_IDENTITY',
            original_dirty_patch_status='NOT_REQUIRED_CLEAN' if all(not v for v in dirty.values())
                else 'UNAVAILABLE_FROM_FROZEN_PREPARATION_DO_NOT_SUBSTITUTE_CURRENT_DIFF',
            frozen_runtime_packages=self.identity.get('packages'),
            missing_metadata=[name for name, val in [('clock_protocol', self.clock),
                ('dirty_at_prepare', dirty or None)] if val is None],
            source_code_files=len(self.identity['code']))

    def configuration(self):
        return redact(dict(provenance=self.trace(), experiment=self.settings,
            frozen_experiment=self.experiment, model=self.model,
            agent_extension=self.identity.get('agent_extension', {}),
            gate_policy=self.identity.get('gate_policy')), excluded=False)

    def wandb_configuration(self):
        """Searchable allowlist; full frozen configuration is a small artifact."""
        trace = self.trace()
        values = {'s3/mode': self.identity['mode'], 's3/clock_protocol': self.clock,
            'source/gym_commit': trace['versions'].get('gym-trading-env'),
            'source/dreamer_commit': trace['versions'].get('dreamerv3'),
            'source/dirty_at_prepare': trace['dirty_at_prepare'],
            'source/model_identity': self.identity['source_model_identity'],
            'source/oracle_identity': self.identity['oracle_identity'],
            'source/training_identity': self.identity['training_identity'],
            'source/preparation_sha256': self.identity['preparation_sha256'],
            'source/model_update_offset': self.identity['source_update_offset'],
            'source/model_origin_updates': self.preparation.get('source_model_updates'),
            'source/clock_comparison_eligible': bool(self.clock),
            'source/start_timestamp_status': trace['source_start_timestamp_status'],
            'source/dirty_patch_status': trace['original_dirty_patch_status']}
        names = {'beta', 'beta_schedule', 'bc_lambda', 'collect_every', 'updates_per_collect',
            'batch_size', 'save_every', 'replay_capacity', 'replay_seed', 'env_slice_bars',
            'env_steps', 'device', 'contracts', 'scope'}
        for name, value in self.settings.items():
            if name in names or name.startswith(('tutor_', 'b2_')):
                values['s3/'+name] = value
        agent = self.model['agent']
        rssm = agent.get('dyn', {}).get('rssm', {})
        for name in ('deter', 'stoch', 'classes', 'hidden'):
            if name in rssm:
                values['model/rssm_'+name] = rssm[name]
        values['model/avail_threshold'] = agent.get('avail_threshold')
        values['model/optimizer'] = agent.get('opt')
        return redact(values)

    def checkpoints(self):
        result = []
        directory = self.root/'ckpt'
        for path in sorted(directory.glob('step_*')):
            if not (path/'done').is_file() or not (path/'manifest.json').is_file():
                continue
            meta = read_json(path/'manifest.json', limit=self.limit)
            step = meta.get('step')
            if (meta.get('protocol') != 'FG_FULL_V1' or meta.get('schema') != 1
                    or type(step) is not int or path.name != f'step_{step:012d}'
                    or meta.get('identity') != self.identity):
                raise TrackingError('Completed Checkpoint metadata identity differs')
            targets = [(path/name).resolve() for name in meta['files']]
            if any(not p.is_relative_to(path.resolve()) for p in targets):
                raise TrackingError('Checkpoint manifest file escaped its directory')
            result.append(dict(env_steps=step, checkpoint_manifest_sha256=digest(path/'manifest.json'),
                local_path=str(path), completed_marker=True, protected=meta.get('protected'),
                local_artifact_available=all(p.is_file() for p in targets),
                verification='METADATA_ONLY_NO_MODEL_OR_REPLAY_READ'))
        return result


def digest_bounded(path, limit):
    return hashlib.sha256(small_bytes(path, limit)).hexdigest()


def discover(roots):
    groups = {}
    for root in roots:
        for path in sorted(Path(root).glob('*/run_identity.json')):
            try:
                document = read_json(path, signed=True)
                identity = document['identity']
                if identity.get('mode') not in FAMILIES:
                    continue
                key = identity['run_id']
                row = groups.setdefault(key, dict(local_run_id=key, mode=identity['mode'], aliases=[]))
                row['aliases'].append(str(path.parent.resolve()))
            except (TrackingError, KeyError):
                continue
    return list(groups.values())
