#!/usr/bin/env python3
"""CPU-only C1_P4 labels entry point."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('labels-build')
    for key in ('dataset', 'oracle-receipt', 'output', 'run-id'):
        build.add_argument('--'+key, required=True)
    build.add_argument('--candidate', default='C1', choices=['C1'])
    build.add_argument('--seed', type=int, default=0)
    build.add_argument('--max-cpu-workers', type=int, default=1, choices=[1])
    build.add_argument('--dry-run', action='store_true')
    audit = sub.add_parser('labels-audit')
    audit.add_argument('--labels', required=True)
    args = parser.parse_args()
    from gym_trading_env.research.fg_s3a_labels import build_labels, load_labels, sources
    if args.command == 'labels-audit':
        result = load_labels(args.labels)
    elif args.dry_run:
        result = sources(args.dataset, args.oracle_receipt, args.candidate)
        print(json.dumps({k: v for k, v in result.items() if k != 'frozen_inputs'}, indent=2))
        return
    else:
        result = build_labels(args.dataset, args.oracle_receipt, args.output,
                              candidate=args.candidate, seed=args.seed, run_id=args.run_id)
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
