"""Durable local outbox; cloud association never determines archive grade."""
import os
from pathlib import Path
import subprocess
import sys
import contextlib
import io

from research_tracking.auth import proxy_environment
from research_tracking.milestones import mapping, publish as upload
from research_tracking.redaction import redact, safe_error
from .io import (ArchiveError, atomic_json, fingerprint, lock, now, read_json,
                 metadata_transaction, recover_metadata)
from .verification import evidence


def publish(root, *, tracker_dir='/data/logdir/fg_single3/experiment_tracking',
            publication=None, mode='online', proxy=None, project='fg-single3-research'):
    if project != 'fg-single3-research':
        raise ArchiveError('W&B project is fixed to fg-single3-research')
    root = Path(root).resolve()
    with lock(root.parent,root.name), metadata_transaction(root):
        evidence(root)
        manifest = read_json(root/'archive_manifest.json',limit=128*1024*1024)
        plan = read_json(root/'reports/inspect.json',limit=128*1024*1024)
        source = manifest['source']
        key = fingerprint(dict(project=project,receipt_content_hash=source['eval_receipt_content_hash'],
            receipt_sha256=source['eval_receipt_sha256'],checkpoint_manifest_sha256=source['checkpoint_manifest_sha256']))
        step = source['checkpoint_steps']
        tags = ['archive:S3_ARCHIVE_V1',f'checkpoint:{step//1000}k' if step%1000==0 else f'checkpoint:{step}',
                'eval:actor-only','clock:'+(manifest['identity']['clock_protocol'] or 'unknown')]
        if plan['eval_derived']['metrics'].get('eval/add_executed',0)>0:
            # Caller explicitly selects significance; don't call every future ADD the first one.
            tags += plan.get('milestone_tags',[])
        record = dict(schema='S3_ARCHIVE_V1',idempotency_key=key,archive_id=manifest['archive_id'],
            achieved_level=manifest['achieved_level'],local_run_id=source['local_run_id'],
            receipt_sha256=source['eval_receipt_sha256'],receipt_content_hash=source['eval_receipt_content_hash'],
            checkpoint_manifest_sha256=source['checkpoint_manifest_sha256'],
            source_identity_sha256=source['source_identity_sha256'],
            gym_git_sha=manifest['identity']['gym_git_sha'],dreamer_git_sha=manifest['identity']['dreamer_git_sha'],
            metrics=plan['eval_derived']['metrics'],tags=tags)
        record = redact(record)
        atomic_json(root/'wandb/milestone.json',record)
        result = dict(status='PENDING',idempotency_key=key,timestamp=now(),reason='OFFLINE')
        if mode=='online':
            previous = dict(os.environ)
            try:
                os.environ.update(proxy_environment(proxy))
                try:
                    mapped = mapping(source,tracker_dir,publication or source.get('publication'))
                except Exception:
                    # Use the existing tracer, never synthesize Run IDs/display-name matches.
                    cli = Path(__file__).resolve().parents[1]/'tools/fg_s3_wandb.py'
                    command = [sys.executable,str(cli),'import-evals','--run-dir',source['original_run_dir'],
                               '--prepared',source['prepared_dir'],'--mode','online','--tracker-dir',tracker_dir]
                    if publication or source.get('publication'):
                        command += ['--publication',publication or source['publication']]
                    if proxy:
                        command += ['--proxy',proxy]
                    subprocess.run(command,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=600)
                    mapped = mapping(source,tracker_dir,publication or source.get('publication'))
                record.update(parent_train_run_id=mapped['train_id'],eval_run_id=mapped['eval_id'])
                atomic_json(root/'wandb/milestone.json',record)
                # Some installed SDK versions print API booleans to stdout.
                # Keep the CLI's stdout a single machine-readable JSON result.
                with contextlib.redirect_stdout(io.StringIO()):
                    result.update(upload(record,mapped))
                result.pop('reason',None)
                manifest['wandb'].update(parent_train_run_id=mapped['train_id'],eval_run_id=mapped['eval_id'])
            except Exception as exc:
                result['reason']=safe_error(exc)
            finally:
                os.environ.clear(); os.environ.update(previous)
        manifest['wandb']['milestone_publish_state']=result['status']
        atomic_json(root/'wandb/publish_state.json',result)
        atomic_json(root/'archive_manifest.json',manifest)
        return result
