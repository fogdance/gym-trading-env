"""Isolated W&B transport for tracker-generated, explicitly allowed files."""
import argparse
import json
import os
from pathlib import Path
import traceback
import time

from . import PROJECT
from .auth import credential, readiness
from .io import LIMIT, TrackingError, atomic_json, digest, read_json
from .redaction import safe_error
from .milestones import merge_tags


def remote_history(api, entity, run_id, first, last):
    from wandb.errors import CommError
    try:
        run = api.run(f'{entity}/{PROJECT}/{run_id}')
    except (ValueError, CommError) as exc:
        # SDK normalizes an absent-run ValueError into CommError. Readiness
        # already proved access; unrelated network/authentication errors fail.
        if 'Could not find run' in str(exc):
            return None, {}
        raise
    # SDK 0.19.1 routes scan_history(keys=[...]) through SampledHistoryScan;
    # acknowledgement needs complete HistoryScan, including nonnumeric hashes.
    rows = run.scan_history(min_step=first, max_step=last+1, page_size=1000)
    confirmed = {}
    for row in rows:
        if 'tracker/event_seq' not in row or 'tracker/event_sha256' not in row:
            raise TrackingError('Cloud history contains an unowned event without tracker markers')
        sequence = int(row['tracker/event_seq'])
        if sequence in confirmed or row.get('_step') != sequence:
            raise TrackingError('Cloud publishing sequence is duplicated or differs from SDK step')
        confirmed[sequence] = row['tracker/event_sha256']
    return run, confirmed


def export(spec):
    # These are this worker's environment only, never the trainer's.
    os.environ['WANDB_DISABLE_CODE'] = 'true'
    os.environ['WANDB_ERROR_REPORTING'] = 'false'
    os.environ['WANDB_CONSOLE'] = 'off'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    mode = spec['mode']
    os.environ['WANDB_MODE'] = mode
    if mode == 'offline':
        os.environ.pop('WANDB_API_KEY', None)
    else:
        readiness(spec['entity'])
    import wandb
    base = Path(spec['tracker_root']).resolve()
    batch = spec['batch']
    events = (base/batch['file']).resolve()
    if not events.is_relative_to(base/'outbox') or digest(events) != batch['sha256']:
        raise TrackingError('Transport event batch integrity/path differs')
    api, cloud_run, remote = None, None, {}
    if mode == 'online':
        api = wandb.Api(api_key=credential(), timeout=15)
        cloud_run, remote = remote_history(api, spec['entity'], batch['run_id'], batch['first_seq'], batch['last_seq'])
    settings = wandb.Settings(disable_git=True, disable_code=True, save_code=False,
        console='off', x_disable_stats=True, x_disable_machine_info=True,
        init_timeout=30, x_file_stream_max_bytes=256 * 1024, program='fg_s3_wandb.py')
    run = wandb.init(project=PROJECT, entity=spec.get('entity'), id=batch['run_id'],
        group=spec['family'], job_type=batch['job_type'], name=spec['display_name'],
        tags=merge_tags(cloud_run.tags if cloud_run else (), spec['tags']), mode=mode, dir=str(base/'sdk'),
        resume=('must' if cloud_run else 'never') if mode == 'online' else None,
        config=spec['configuration'], settings=settings)
    sent, skipped = 0, 0
    try:
        for axis in ('progress/env_steps', 'progress/local_update', 'progress/agent_update',
                     'provisional/progress/env_steps', 'eval/checkpoint_env_steps'):
            run.define_metric(axis)
        for prefix in ('train/*', 'raw_train/*', 'observations/*'):
            run.define_metric(prefix, step_metric='progress/env_steps', step_sync=False)
        run.define_metric('provisional/*', step_metric='provisional/progress/env_steps', step_sync=False)
        run.define_metric('eval/*', step_metric='eval/checkpoint_env_steps', step_sync=False)
        with events.open() as stream:
            for line in stream:
                event = json.loads(line)
                if event['seq'] in remote:
                    if remote[event['seq']] != event['sha256']:
                        raise TrackingError('Remote event identity conflicts with local durable history')
                    skipped += 1
                    continue
                metrics = dict(event['metrics'], **{'tracker/event_seq': event['seq'],
                    'tracker/event_sha256': event['sha256'], 'tracker/event_kind': event['kind']})
                # Publishing sequence is never an environment/update counter.
                # Explicit sequence also prevents offline segment _step resets.
                run.log(metrics, step=event['seq'])
                sent += 1
        for relative in batch['artifacts']:
            path = (base/relative).resolve()
            allowed = path == base/'config_preview.json' or path.is_relative_to(base/'artifacts')
            if not allowed or path.suffix != '.json' or path.stat().st_size > LIMIT:
                raise TrackingError('Attachment is outside generated small-file allowlist')
            artifact_type = 'configuration' if path.name == 'config_preview.json' else (
                'checkpoint-index' if path.name.startswith('checkpoint_') else 'evaluation-summary')
            artifact = wandb.Artifact('fg-s3-'+artifact_type+'-'+batch['run_id'], type=artifact_type,
                metadata={'source_payload_sha256': digest(path), 'contains_full_checkpoint': False})
            artifact.add_file(str(path), name=path.name)
            run.log_artifact(artifact)
            if artifact_type == 'evaluation-summary':
                report = read_json(path)
                columns = ['trade_index', 'exit_reason', 'held_market_bars', 'net_pnl', 'adverse_first_market_minute']
                # SDK 0.19.1 Table summary serialization probes the remote
                # server's client-ID capability even in offline mode. Offline
                # keeps this same small trade table in its JSON artifact.
                if mode == 'online' and report.get('trades'):
                    run.summary['eval/trades'] = wandb.Table(columns=columns,
                        data=[[r.get(k) for k in columns] for r in report['trades']])
        run.summary['tracker/export_last_seq'] = batch['last_seq']
        run.summary['tracker/export_status'] = 'OFFLINE_EXPORTED' if mode == 'offline' else 'ONLINE_UPLOAD_REQUESTED'
        directory = str(Path(run.dir).parent)
        url = run.url if mode == 'online' else None
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise
    result = dict(status='OFFLINE_EXPORTED' if mode == 'offline' else 'ONLINE_FINISHED',
        sdk_directory=directory, run_id=batch['run_id'], sent_events=sent, skipped_remote_events=skipped,
        last_seq=batch['last_seq'], url=url, wandb_version=wandb.__version__)
    if mode == 'online':
        with events.open() as stream:
            expected = {e['seq']:e['sha256'] for e in map(json.loads, stream)}
        deadline = time.monotonic() + 45
        while True:
            _, confirmed = remote_history(api, spec['entity'], batch['run_id'], batch['first_seq'], batch['last_seq'])
            if any(expected.get(seq) != sha for seq, sha in confirmed.items()):
                raise TrackingError('Cloud history acknowledgement conflicts with durable events')
            if confirmed == expected:
                break
            if time.monotonic() >= deadline:
                raise TrackingError(f'Cloud acknowledgement incomplete ({len(confirmed)}/{batch["count"]}); keep batch pending')
            time.sleep(3)
        result['status'] = 'ONLINE_VERIFIED'
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', required=True)
    args = p.parse_args()
    spec = read_json(args.spec)
    try:
        result = export(spec)
        atomic_json(spec['result_path'], result)
    except BaseException as exc:
        atomic_json(spec['result_path'], {'status': 'TRANSPORT_FAILED', 'error': safe_error(exc),
                    'traceback': safe_error(traceback.format_exc())})
        raise SystemExit(1)


if __name__ == '__main__':
    main()
