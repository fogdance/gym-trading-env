"""Transactional independent copies; physical retention is separate from grade."""
from pathlib import Path
import shutil
import subprocess

from . import LEVELS, SCHEMA
from .io import (ArchiveError, atomic_json, check_inventory, digest, fingerprint,
                 independent, inventory, lock, now, read_json, relative, seal, stable_copy, exchange, prefix_hash)
from .source import snapshot


def selected(plan, level, preserve):
    kinds = {'evidence'}
    if preserve or level != 'EVIDENCE':
        kinds.add('checkpoint')
    if level != 'EVIDENCE':
        kinds.add('evaluation')
    if level == 'FULL':
        kinds.add('restore')
    return [row for row in plan['copy_plan'] if row['kind'] in kinds]


def space(plan, rows, reserve_gib):
    base = Path(plan['archive_root'])
    while not base.exists():
        base = base.parent
    available = shutil.disk_usage(base).free
    source_estimate = sum(len(subprocess.check_output(
        ['git', '-C', r['path'], 'archive', '--format=tar', r['sha']]))
        for r in plan['code_audit']['repositories'].values())
    estimate = sum(r['bytes'] for r in rows)+source_estimate
    reserve = int(reserve_gib*1024**3)
    return dict(estimated_bytes=estimate, available_bytes=available, reserve_bytes=reserve,
                sufficient=available >= estimate+reserve)


def read_plan(path):
    plan = read_json(path, limit=128*1024*1024)
    if (plan.get('schema') != SCHEMA or plan.get('identity_key') != fingerprint(plan['anchor'])
            or plan.get('plan_sha256') != fingerprint({k:v for k,v in plan.items() if k != 'plan_sha256'})):
        raise ArchiveError('Inspection plan identity or content hash differs')
    return plan


def write_readme(root, manifest):
    text = '# S3 research archive\n\n'
    text += 'Schema: `S3_ARCHIVE_V1`\n\n'
    text += 'Archive: `'+manifest['archive_id']+'`\n\n'
    text += 'Requested: **'+manifest['requested_level']+'**; achieved: **'+manifest['achieved_level']+'**.\n\n'
    text += 'State: `'+manifest['archive_state']+'`. '+manifest['verification_status']+'\n\n'
    text += 'Checkpoint bytes retained: '+str(manifest['checkpoint_preserved'])+'. This alone does not establish restoration.\n\n'
    text += 'Source/checkpoint/receipt bytes are original. Original absolute paths require the documented private namespace mappings.\n\n'
    text += 'See reports/inspect.json, reports/verify.json, reports/eval_report.json and reports/restore_report.json when present.\n\n'
    text += 'Missing / pending: '+str(manifest['verification']['missing_dependencies'])+'\n\n'
    text += 'W&B state is independent of archive grade. Original training artifacts were not written.\n'
    (root/'README.md').write_text(text)


def manifest_from(plan, level, preserve, root):
    audit = plan['code_audit']['repositories']
    identity = plan['identity']
    snapshots = root/'provenance/source_bundle'
    return dict(schema=SCHEMA, archive_id=plan['archive_id'], identity_key=plan['identity_key'],
        created_at_utc=now(), archive_state='PARTIAL' if level != 'EVIDENCE' else 'VERIFIED_EVIDENCE',
        requested_level=level, achieved_level='EVIDENCE', verification_status='L1_PENDING_HASH_VERIFICATION',
        checkpoint_preserved=preserve or level != 'EVIDENCE', source=plan['source'],
        identity=dict(gym_git_sha=audit['gym-trading-env']['sha'], dreamer_git_sha=audit['dreamerv3']['sha'],
            gym_dirty=bool(audit['gym-trading-env']['dirty_at_prepare']),
            dreamer_dirty=bool(audit['dreamerv3']['dirty_at_prepare']),
            code_snapshot_sha256=fingerprint(inventory(snapshots)),
            resolved_config_sha256=digest(root/'provenance/resolved_config.yaml'),
            reward_identity=identity['training_identity'].get('reward_params_hash'),
            oracle_identity=identity['oracle_identity'], train_manifest_identity=identity['training_identity'],
            clock_protocol=plan['clock_protocol'], additional_agent_identity=identity.get('agent_extension'),
            outcome_identity=identity.get('outcome_identity')),
        dependencies=[dict(archive_path=r['target'], original_path=r['source'], sha256=r['sha256'],
            bytes=r['bytes'], kind=r['kind'], prefix=r['prefix']) for r in selected(plan, level, preserve)],
        verification=dict(hashes_valid=None, source_identity_verified=None, eval_reproduced=None,
            resume_smoke_pass=None, exact_replay_verified=None, missing_dependencies=plan['missing']),
        wandb=dict(project='fg-single3-research', parent_train_run_id=None, eval_run_id=None,
            milestone_publish_state='PENDING'))


def execute(plan, *, level='EVIDENCE', preserve=False, execute=False, reserve_gib=10):
    if level not in LEVELS or reserve_gib < 0:
        raise ArchiveError('Invalid level or disk reserve')
    protected = [plan['source']['original_run_dir'], plan['source']['prepared_dir']]
    protected += [r['path'] for r in plan['code_audit']['repositories'].values()]
    base = independent(plan['archive_root'], protected)
    rows = selected(plan, level, preserve)
    budget = space(plan, rows, reserve_gib)
    result = dict(status='DRY_RUN', archive_id=plan['archive_id'], archive_dir=str(base/plan['archive_id']),
        requested_level=level, proposed_achieved_level='EVIDENCE', checkpoint_preserved=preserve or level != 'EVIDENCE',
        copy_files=len(rows), disk=budget, missing=plan['missing'], source_files_written=False)
    if not execute:
        return result
    if not plan['ready']:
        raise ArchiveError('Inspection cannot reconstruct actual source; refusing archive')
    if not budget['sufficient']:
        raise ArchiveError('Insufficient disk space including reserve; no archive created')
    base.mkdir(parents=True, exist_ok=True)
    final = base/plan['archive_id']
    with lock(base, plan['archive_id']):
        if final.exists():
            old = read_json(final/'archive_manifest.json', limit=128*1024*1024)
            if old['identity_key'] != plan['identity_key']:
                raise ArchiveError('Existing archive identity conflicts')
            check_inventory(final)
            required = {r['target'] for r in rows}
            if required <= {str(p.relative_to(final)) for p in final.rglob('*') if p.is_file()}:
                return dict(result, status='EXISTING_ARCHIVE_VERIFIED', achieved_level=old['achieved_level'])
            staging = base/'.staging'/(plan['archive_id']+'.upgrade.tmp')
            if not staging.exists():
                shutil.copytree(final, staging)
            version = staging/'verification'/('upgrade_'+str(len(list((staging/'verification').glob('upgrade_*')))+1))
            atomic_json(version/'previous_manifest.json', old)
        else:
            staging = base/'.staging'/(plan['archive_id']+'.tmp')
            staging.mkdir(parents=True, exist_ok=True)
        try:
            for row in rows:
                stable_copy(row, relative(staging, row['target']))
            snapshot(staging, plan['code_audit'])
            # Detect mutations to an early source file while later files were copied.
            for row in rows:
                if prefix_hash(row['source'],row['bytes']) != row['sha256']:
                    raise ArchiveError('Source changed before final archive promotion: '+row['source'])
            atomic_json(staging/'reports/inspect.json', plan)
            atomic_json(staging/'provenance/dependency_versions.json', plan['identity'].get('packages', {}))
            atomic_json(staging/'provenance/checkpoint_boundaries.json', plan['boundaries'])
            mappings = {r['source']:r['target'] for r in rows if r['kind'] in ('evaluation', 'restore')}
            mappings[plan['source']['prepared_dir']] = 'provenance/prepared'
            for name, repo in plan['code_audit']['repositories'].items():
                mappings[repo['path']] = 'provenance/source_bundle/'+name
            atomic_json(staging/'provenance/path_mapping.json', dict(read_only=mappings,
                writable_run_alias=plan['source']['original_run_dir']))
            manifest = manifest_from(plan, level, preserve, staging)
            if final.exists():
                manifest['created_at_utc'] = old['created_at_utc']
                manifest['achieved_level'] = old['achieved_level']
                manifest['verification'] = old['verification']
            atomic_json(staging/'archive_manifest.json', manifest)
            from .verification import evidence
            report = evidence(staging, check_hashes=False)
            manifest['verification'].update(hashes_valid=True, source_identity_verified=True)
            manifest['verification_status'] = 'L1_PASS_HIGHER_LEVEL_TESTS_NOT_RUN' if level != 'EVIDENCE' else 'L1_PASS'
            if level != 'EVIDENCE':
                manifest['verification']['missing_dependencies'] = plan['missing']+[
                    dict(kind='verification', reason='PENDING_EVALUATION_TEST'),
                    *([dict(kind='verification', reason='PENDING_RESUME_TEST')] if level == 'FULL' else [])]
            atomic_json(staging/'archive_manifest.json', manifest)
            atomic_json(staging/'reports/verify.json', report)
            write_readme(staging, manifest)
            seal(staging)
            check_inventory(staging)
            if final.exists():
                exchange(staging, final)
                previous = base/'.revisions'/plan['archive_id']
                previous.mkdir(parents=True, exist_ok=True)
                staging.rename(previous/('v'+str(len(list(previous.iterdir()))+1)))
            else:
                staging.rename(final)
            return dict(result, status='ARCHIVE_SEALED', achieved_level=manifest['achieved_level'],
                        archive_state=manifest['archive_state'])
        except Exception as exc:
            atomic_json(base/'.failures'/(plan['archive_id']+'.json'),
                dict(status='FAILED', error=str(exc), retained_staging=str(staging), timestamp=now()))
            raise
