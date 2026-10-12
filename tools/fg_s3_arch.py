#!/usr/bin/env python3
"""Build/audit the shared complete TRAIN teacher catalog; no model training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build-catalog')
    for name in ('dataset', 'oracle-receipt', 's3a-labels', 'output'):
        build.add_argument('--'+name, required=True)
    build.add_argument('--diagnostic', action='store_true')
    audit = commands.add_parser('catalog-audit')
    audit.add_argument('--catalog', required=True)
    args = parser.parse_args()
    from gym_trading_env.research.fg_s3_teacher_catalog import build_catalog, load_catalog
    if args.command == 'build-catalog':
        build_catalog(args.dataset, args.oracle_receipt, args.s3a_labels, args.output,
                      diagnostic=args.diagnostic)
        result = load_catalog(args.output).summary()
    else:
        result = load_catalog(args.catalog).summary()
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
