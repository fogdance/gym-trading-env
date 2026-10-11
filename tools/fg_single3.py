#!/usr/bin/env python3
"""CPU Single3 research CLI. Subcommands are enabled as their Gates are built."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

from gym_trading_env.research.fg_single3 import env_audit, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    env = sub.add_parser('env-test', help='Real TRAIN original-Env and restoration reconciliation')
    env.add_argument('--config', default='configs/fg_single3_v1/cpu.yaml')
    env.add_argument('--output', required=True)
    env.add_argument('--contracts', nargs='+', default=['CZCE.FG809','CZCE.FG909'])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == 'env-test':
        result = env_audit(config, args.output, args.contracts)
    else:
        raise ValueError('Unimplemented command')
    print(f"{args.command}: {result['status']} → {Path(args.output).resolve()}", flush=True)


if __name__ == '__main__':
    main()
