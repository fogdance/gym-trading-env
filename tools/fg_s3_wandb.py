#!/usr/bin/env python3
"""Track existing FG research Runs without importing or changing a trainer."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import yaml

from research_tracking import PROJECT, SCHEMA
from research_tracking.auth import credential, proxy_environment, readiness
from research_tracking.evals import parse_eval
from research_tracking.identity import SourceRun, discover, git_info
from research_tracking.importer import Importer
from research_tracking.io import TrackingError, atomic_json, small_bytes
from research_tracking.launcher import launch
from research_tracking.parsers import watermark
from research_tracking.redaction import redact, safe_error
from research_tracking.state import Store
from research_tracking.writer import Writer


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inspect', 'import', 'follow', 'import-evals', 'sync', 'auth-check', 'launch'])
    parser.add_argument('--run-dir')
    parser.add_argument('--prepared')
    parser.add_argument('--family')
    parser.add_argument('--config', default=str(ROOT/'configs/wandb/fg-single3-research.yaml'))
    parser.add_argument('--mode', choices=['offline', 'online', 'disabled'])
    parser.add_argument('--entity')
    parser.add_argument('--project')
    parser.add_argument('--proxy', help='Tracker-only HTTP/HTTPS proxy override (internal default: http://127.0.0.1:10808)')
    parser.add_argument('--publication', help='Explicit stable namespace when re-uploading after cloud deletion')
    parser.add_argument('--tracker-dir')
    parser.add_argument('--every', type=int)
    parser.add_argument('--poll-seconds', type=float)
    parser.add_argument('--polls', type=int, default=0, help='0 follows until interrupted; bounded values aid verification')
    parser.add_argument('--no-evals', action='store_true')
    parser.add_argument('--prediction-report', action='append', default=[],
                        help='Verified posthoc DEV receipt outside the source Run; may be repeated')
    parser.add_argument('--discover', action='store_true')
    parser.add_argument('--experiment-note', default='')
    parser.add_argument('--research-question', default='')
    parser.add_argument('--trainer-cwd', help='Original frozen worktree for the explicit command after --')
    argv = sys.argv[1:]
    separator = argv.index('--') if '--' in argv else len(argv)
    args = parser.parse_args(argv[:separator])
    args.trainer_command = argv[separator+1:]
    if args.trainer_command and args.command != 'launch':
        parser.error('Only launch accepts an original command after --')
    return args


def inspect(source):
    checkpoints = source.checkpoints()
    evaluations = [parse_eval(source, p) for p in sorted((source.root/'evaluations').glob('*/receipt.json'))]
    return dict(status='READ_ONLY_INSPECTION', trace_identity=source.trace(),
        checkpoints=checkpoints, committed_watermark=watermark(source, checkpoints),
        evaluations=evaluations, configuration_preview=source.configuration(),
        cloud_readiness=dict(credential_present=bool(credential()), network_checked=False),
        model_or_replay_read=False, source_files_written=False)


def track(args, cfg, mode, entity, stop=None):
    interval = args.poll_seconds if args.poll_seconds is not None else cfg['poll_seconds']
    if interval <= 0 or args.polls < 0:
        raise TrackingError('Polling interval/count is invalid')
    if stop is not None:
        root = Path(args.run_dir)
        def ready():
            return (root/'run_identity.json').exists() and any((root/'ckpt').glob('step_*/done'))
        while not ready():
            stop.wait(interval)
            if stop.is_set() and not ready():
                raise TrackingError('Original command ended without a completed source Checkpoint')
    source = SourceRun(args.run_dir, args.prepared, family=args.family, limit=cfg['max_small_file_bytes'],
                       publication=args.publication)
    if args.command == 'inspect':
        print(json.dumps(redact(inspect(source), excluded=False), indent=2, ensure_ascii=False))
        return dict(status='READ_ONLY_INSPECTION')
    if mode == 'disabled':
        return dict(status='TRACKING_DISABLED', local_run_id=source.run_id)
    every = args.every if args.every is not None else cfg['metrics_every_n_updates']
    with Store(source, args.tracker_dir or cfg['tracker_dir']) as store:
        tracker = dict(schema=SCHEMA, git=git_info(ROOT), python=platform.python_version(),
            wandb=importlib.metadata.version('wandb'), os=platform.system(),
            experiment_note=args.experiment_note, research_question=args.research_question)
        atomic_json(store.root/'tracker_provenance.json', redact(tracker))
        importer = Importer(store, every, prediction_reports=args.prediction_report)
        writer = Writer(mode, entity)
        online_status = 'NOT_CHECKED' if mode == 'online' else 'NOT_APPLICABLE'
        fallback_count = 0
        if args.command == 'sync':
            results = writer.sync(store)
            online_status = 'VERIFIED'
        else:
            results = []
            iteration = 0
            while True:
                final = stop is not None and stop.is_set()
                if args.command in ('import', 'follow', 'launch'):
                    mark = importer.canonical()
                    if args.command in ('follow', 'launch'):
                        importer.monitor(mark)
                if not args.no_evals:
                    importer.evaluations()
                try:
                    if mode == 'online' and args.command in ('follow', 'launch') and any(
                            b.get('export_mode') == 'offline' and not b['synced'] for b in store.state['batches']):
                        results.extend(writer.sync(store))
                    results.extend(writer.export(store))
                    if mode == 'online':
                        online_status = 'VERIFIED'
                except TrackingError as exc:
                    if args.command not in ('follow', 'launch') or mode != 'online':
                        raise
                    atomic_json(store.root/'last_online_fallback.json', dict(
                        status='OFFLINE_FALLBACK', error=safe_error(exc), trainer_affected=False))
                    online_status = 'OFFLINE_FALLBACK'
                    fallback_count += 1
                    print('Cloud unavailable; tracker keeps offline records: '+safe_error(exc), file=sys.stderr, flush=True)
                    results.extend(Writer('offline').export(store))
                results = results[-20:]
                iteration += 1
                if (args.command not in ('follow', 'launch') or final
                        or (stop is None and args.polls and iteration >= args.polls)):
                    break
                print(json.dumps(dict(status='FOLLOWING', **store.summary())), flush=True)
                if stop is not None:
                    stop.wait(interval)
                else:
                    time.sleep(interval)
        receipt = dict(status='TRACKER_OPERATION_COMPLETE', command=args.command, mode=mode,
            project=PROJECT, entity=entity, transport_results=results, **store.summary(),
            online_status=online_status, offline_fallback_count=fallback_count,
            original_files_written=False, gpu_training_started=False)
        atomic_json(store.root/'last_operation.json', receipt)
        return receipt


def main():
    args = arguments()
    trainer_environment = dict(os.environ)
    os.environ.update(proxy_environment(args.proxy))
    cfg = yaml.safe_load(small_bytes(args.config))
    if cfg['schema'] != SCHEMA:
        raise TrackingError('Unknown tracker configuration schema')
    project = args.project or os.getenv('WANDB_PROJECT') or cfg['project']
    if project != PROJECT or cfg['project'] != PROJECT:
        raise TrackingError('Project is fixed to fg-single3-research')
    mode = args.mode or os.getenv('WANDB_MODE') or cfg['mode']
    entity = args.entity or os.getenv('WANDB_ENTITY') or cfg.get('entity')
    if args.command == 'auth-check':
        print(json.dumps(readiness(entity), indent=2))
        return
    if args.discover:
        roots = [Path('/data/logdir/fg_single3/s3_v2')]
        roots.extend(ROOT.parents[1].glob('fg-s3-*/gym-trading-env/artifacts/fg_s3_v2/runs'))
        print(json.dumps(discover(roots), indent=2))
        return
    if not args.run_dir:
        raise TrackingError('Select --run-dir; tracking never creates or changes a trainer Run')
    if args.command == 'launch':
        if not args.trainer_cwd:
            raise TrackingError('launch requires the original --trainer-cwd')
        receipt = launch(args.trainer_command, cwd=args.trainer_cwd, environment=trainer_environment,
            track=(lambda stop: dict(status='TRACKING_DISABLED')) if mode == 'disabled'
                  else (lambda stop: track(args, cfg, mode, entity, stop)))
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        raise SystemExit(receipt['producer_returncode'])
    receipt = track(args, cfg, mode, entity)
    if args.command != 'inspect':
        print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Tracker stopped; source trainer is untouched.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr)
        raise SystemExit(2)
