#!/usr/bin/env python3
"""Manage separate S3-A/S3-B experiments through frozen YAML snapshots."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'prepare', 'start', 'resume', 'audit', '_execute'])
    parser.add_argument('--config', help='Experiment YAML path or name in configs/experiments/fg_single3/')
    parser.add_argument('--prepared', help='Use an existing immutable prepared snapshot')
    parser.add_argument('--phase', choices=['features', 'train'], help=argparse.SUPPRESS)
    parser.add_argument('--resume-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    from gym_trading_env.research.fg_s3_experiment import (
        activate, audit, execute_phase, prepare, read_experiment, start)
    if args.command == '_execute':
        if not args.prepared or not args.phase:
            parser.error('Internal worker requires --prepared and --phase')
        result = execute_phase(args.prepared, args.phase, args.resume_worker)
        print(json.dumps({'status': result['status']}, indent=2))
        return
    # Capture the caller's device selection before CPU preparation hides GPUs.
    os.environ['FG_S3_VISIBLE_GPUS'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    if args.command in ('plan', 'prepare'):
        if not args.config or args.prepared:
            parser.error('plan/prepare require --config and do not take --prepared')
        _, cfg = read_experiment(args.config)
        if args.command == 'plan':
            print(json.dumps(dict(status='CONFIG_ONLY_NOT_ADMITTED', config=cfg,
                                  formal_gpu_training='NOT_RUN'), ensure_ascii=False, indent=2))
            return
        result = prepare(args.config)
    else:
        if not args.config and not args.prepared:
            parser.error('Select an experiment with --config or --prepared')
        if args.config:
            _, cfg = read_experiment(args.config)
            prepared = args.prepared or cfg['prepared_dir']
        else:
            import yaml
            prepared = args.prepared
            cfg = yaml.safe_load((Path(prepared) / 'effective_config.yaml').read_text())
        activate(cfg)
        result = (audit(prepared, args.config) if args.command == 'audit'
                  else start(prepared, resume=args.command == 'resume', config=args.config))
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'packages')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
