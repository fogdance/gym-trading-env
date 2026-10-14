"""Evidence checks and proven grade promotion; file presence is not capability."""
from decimal import Decimal
from pathlib import Path
import shutil
from research_tracking.evals import entry_count
from research_tracking.eval_schema import unique_events

from . import LEVELS
from .io import (ArchiveError, atomic_json, check_inventory, digest, fingerprint,
                 inventory, lock, now, read_json, metadata_transaction, recover_metadata)
from .archive import write_readme


def evidence(root, *, check_hashes=True):
    root = Path(root).resolve()
    count = check_inventory(root) if check_hashes else None
    manifest = read_json(root/'archive_manifest.json', limit=128*1024*1024)
    source = manifest['source']
    run = read_json(root/'provenance/run_identity.json', signed=True)['identity']
    prep = read_json(root/'provenance/prepared/manifest.json', signed=True)
    if digest(root/'provenance/prepared/manifest.json') != run['preparation_sha256']:
        raise ArchiveError('Archived preparation and Run differ')
    import yaml
    config = yaml.safe_load((root/'provenance/resolved_config.yaml').read_text())
    model = read_json(root/'provenance/prepared/model_config.json', signed=True)
    if config != model['config'] or fingerprint(config) != run['model_config_sha256']:
        raise ArchiveError('Archived resolved configuration differs')
    if prep['versions'] != run['versions'] or prep['code'] != run['code']:
        raise ArchiveError('Archived historical source identities differ')
    for name, sha in prep['files'].items():
        if digest(root/'provenance/prepared'/name) != sha:
            raise ArchiveError('Archived preparation file differs: '+name)
    cp = root/'checkpoint'/Path(source['checkpoint_dir']).name
    meta = read_json(cp/'manifest.json')
    evaluation = root/'evaluations'/Path(source['eval_dir']).name
    receipt = read_json(evaluation/'receipt.json', signed=True)
    if (not (cp/'done').is_file() or meta['identity'] != run or meta['step'] != source['checkpoint_steps']
            or digest(cp/'manifest.json') != source['checkpoint_manifest_sha256']
            or digest(evaluation/'receipt.json') != source['eval_receipt_sha256']
            or receipt['checkpoint_manifest_sha256'] != source['checkpoint_manifest_sha256']
            or Path(receipt['checkpoint']) != Path(source['checkpoint_dir'])):
        raise ArchiveError('Archived Checkpoint/receipt identity differs')
    if manifest['checkpoint_preserved']:
        for name, sha in meta['files'].items():
            if digest(cp/name) != sha:
                raise ArchiveError('Archived Checkpoint member differs: '+name)
        if {str(p.relative_to(cp)) for p in cp.rglob('*') if p.is_file()} != set(meta['files'])|{'manifest.json','done'}:
            raise ArchiveError('Archived Checkpoint file set differs')
    audit = read_json(root/'provenance/code_identity.json', limit=128*1024*1024)
    for p, sha in run['code'].items():
        matches = [repo for repo in audit['repositories'].values() if Path(p).is_relative_to(repo['path'])]
        if len(matches) != 1:
            raise ArchiveError('Runtime source file cannot be assigned to a repository')
        repo = matches[0]; rel = Path(p).relative_to(repo['path'])
        if digest(root/'provenance/source_bundle'/Path(repo['path']).name/rel) != sha:
            raise ArchiveError('Archived runtime source differs: '+str(rel))
    if fingerprint(inventory(root/'provenance/source_bundle')) != manifest['identity']['code_snapshot_sha256']:
        raise ArchiveError('Complete source tree snapshot identity differs')
    derived = read_json(root/'reports/inspect.json', limit=128*1024*1024)['eval_derived']
    if (evaluation/'account_state.json').exists():
        account = read_json(evaluation/'account_state.json')
        attrs = unique_events(account.get('attributions', []))
        entries = entry_count(account)
        if entries != receipt['entries']:
            raise ArchiveError('Archived opening count differs from account evidence')
        net = sum((Decimal(str(a['net_trade_pnl'])) for a in attrs), Decimal(0))
        if (sum(a['reason']=='TAKE_PROFIT' for a in attrs) != receipt['tp_count']
                or sum(a['reason']=='STOP_LOSS' for a in attrs) != receipt['sl30']
                or account['total_reward'] != receipt['reward']
                or float(net) != derived['metrics']['eval/closed_trade_net_pnl']):
            raise ArchiveError('Archived historical evaluation does not reconcile')
    return dict(status='EVIDENCE_VERIFIED', files_verified=count, timestamp=now(),
        source_identity_verified=True, hashes_valid=check_hashes,
        checkpoint_manifest_sha256=source['checkpoint_manifest_sha256'],
        eval_receipt_sha256=source['eval_receipt_sha256'], historical_metrics=derived['metrics'],
        eval_reproduced=None, resume_smoke_pass=None, gpu_used=False)


def grade(manifest, report, mode):
    verification = dict(manifest['verification'])
    verification.update(hashes_valid=True, source_identity_verified=True)
    if mode == 'eval':
        verification['eval_reproduced'] = report.get('status') == 'ISOLATED_EVAL_PASS'
    elif mode == 'resume-smoke':
        verification['resume_smoke_pass'] = report.get('status') == 'ISOLATED_RESUME_SMOKE_PASS'
    achieved = 'EVIDENCE'
    if verification['eval_reproduced'] is True:
        achieved = 'EVAL'
    dependency_missing = [m for m in verification['missing_dependencies'] if m.get('kind') != 'verification']
    if achieved == 'EVAL' and verification['resume_smoke_pass'] is True and not dependency_missing:
        achieved = 'FULL'
    missing = list(dependency_missing)
    if LEVELS.index(manifest['requested_level']) >= 1 and verification['eval_reproduced'] is not True:
        missing.append(dict(kind='verification', reason='PENDING_OR_FAILED_EVALUATION_TEST'))
    if manifest['requested_level'] == 'FULL' and verification['resume_smoke_pass'] is not True:
        missing.append(dict(kind='verification', reason='PENDING_OR_FAILED_RESUME_TEST'))
    verification['missing_dependencies'] = missing
    return dict(manifest, achieved_level=achieved, verification=verification,
        archive_state='PARTIAL' if LEVELS.index(achieved)<LEVELS.index(manifest['requested_level']) else
            {'EVIDENCE':'VERIFIED_EVIDENCE','EVAL':'VERIFIED_EVAL','FULL':'VERIFIED_FULL'}[achieved],
        verification_status={'EVIDENCE':'L1_VERIFIED','EVAL':'L2_VERIFIED','FULL':'L3_VERIFIED'}[achieved],
        last_verification_status=report['status'])


def retain_execution(root, report, destination):
    """Bring executed scripts/logs and replayed evaluation receipts into the seal."""
    if not report.get('validation_dir'):
        return
    work = Path(report['validation_dir']).resolve()
    if not work.is_relative_to((root.parent/'.validation').resolve()):
        raise ArchiveError('Validation evidence escaped archive-owned scratch directory')
    for name, key in [('validation_worker.py','worker_sha256'),('stdout.log','log_sha256')]:
        if digest(work/name) != report[key]:
            raise ArchiveError('Executed validation evidence differs: '+name)
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('validation_worker.py','stdout.log','result.json','spec.json','isolation_command.json'):
        shutil.copy2(work/name,destination/name)
    if (work/'execution_environment.json').exists():
        shutil.copy2(work/'execution_environment.json',destination/'execution_environment.json')
    if report.get('reproduced_receipt'):
        eval_name = Path(report['reproduced_receipt']['output']).name
        evaluation = work/'run/evaluations'/eval_name
        if digest(evaluation/'receipt.json') != report['reproduced_receipt_sha256']:
            raise ArchiveError('Reproduced evaluation receipt bytes differ')
        for p in evaluation.iterdir():
            if p.is_file():
                target=destination/'reproduced_evaluation'/p.name
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(p,target)


def verify(archive_dir, *, mode='hashes', python=None, device=None):
    from .io import seal
    root = Path(archive_dir).resolve()
    with lock(root.parent, root.name):
        recover_metadata(root)
        result = evidence(root)
        manifest = read_json(root/'archive_manifest.json', limit=128*1024*1024)
        if mode != 'hashes':
            if device != 'gpu' or not python:
                raise ArchiveError('Explicit --device gpu and matching --python required for model validation')
            from .isolation import validate
            result = validate(root, mode, python)
        with metadata_transaction(root):
            version = root/'verification'/('v'+str(len(list((root/'verification').glob('v*')))+1))
            atomic_json(version/'previous_manifest.json', manifest)
            atomic_json(version/'report.json', result)
            if mode != 'hashes':
                retain_execution(root,result,version/'execution')
            # Also retain evidence from a verification completed before this
            # metadata-retention addition, without rerunning model computation.
            for filename in ('eval_report.json','restore_report.json'):
                p=root/'reports'/filename
                if p.exists():
                    prior=read_json(p)
                    destination=root/'verification/executions'/Path(prior.get('validation_dir','none')).name
                    if prior.get('validation_dir') and not destination.exists():
                        retain_execution(root,prior,destination)
            manifest = grade(manifest, result, mode)
            atomic_json(root/'archive_manifest.json', manifest)
            atomic_json(root/'reports/verify.json', result)
            if mode != 'hashes':
                atomic_json(root/'reports'/('eval_report.json' if mode == 'eval' else 'restore_report.json'), result)
            write_readme(root, manifest)
        return dict(result, achieved_level=manifest['achieved_level'], archive_state=manifest['archive_state'])
