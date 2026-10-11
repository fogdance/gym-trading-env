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
    reward=sub.add_parser('reward-study',help='TRAIN-only candidate Reward comparison with real Env replay')
    reward.add_argument('--config',default='configs/fg_single3_v1/cpu.yaml')
    reward.add_argument('--output',required=True)
    reward.add_argument('--env-receipt',required=True)
    reward.add_argument('--workers',type=int,default=12)
    reward.add_argument('--contracts',nargs='+',default=['CZCE.FG809','CZCE.FG909'])
    oracle=sub.add_parser('oracle',help='TRAIN candidates and exact optimality validation')
    oracle.add_argument('--config',default='configs/fg_single3_v1/cpu.yaml')
    oracle.add_argument('--output',required=True)
    oracle.add_argument('--env-receipt',required=True)
    oracle.add_argument('--reward-receipt',required=True)
    oracle.add_argument('--workers',type=int,default=4)
    oracle.add_argument('--contracts',nargs='+')
    oracle.add_argument('--resume',action='store_true')
    gen=sub.add_parser('replay-generate',help='Fresh complete single-position original-Driver Replay')
    gen.add_argument('--config',default='configs/fg_single3_v1/cpu.yaml')
    gen.add_argument('--output',required=True)
    gen.add_argument('--env-receipt',required=True)
    gen.add_argument('--reward-receipt',required=True)
    gen.add_argument('--oracle-receipt',required=True)
    gen.add_argument('--workers',type=int,default=12)
    gen.add_argument('--contracts',nargs='+')
    gen.add_argument('--resume',action='store_true')
    audit=sub.add_parser('replay-audit',help='Actual native Replay coverage and pool isolation')
    audit.add_argument('--config',default='configs/fg_single3_v1/cpu.yaml')
    audit.add_argument('--output',required=True)
    audit.add_argument('--dataset',required=True)
    audit.add_argument('--samples',type=int,default=1000)
    audit.add_argument('--seed',type=int,default=202)
    horizon=sub.add_parser('horizon-audit',help='Event discount and same-account causal forks')
    horizon.add_argument('--config',default='configs/fg_single3_v1/cpu.yaml')
    horizon.add_argument('--output',required=True)
    horizon.add_argument('--dataset',required=True)
    horizon.add_argument('--coverage',required=True)
    horizon.add_argument('--workers',type=int,default=12)
    horizon.add_argument('--snapshots-per-contract',type=int,default=8)
    pilot=sub.add_parser('prepare-pilot',help='CPU preparation only; never starts a GPU Agent')
    pilot.add_argument('--config',default='configs/fg_single3_v1/cpu.yaml')
    pilot.add_argument('--output',required=True)
    pilot.add_argument('--dataset',required=True)
    pilot.add_argument('--coverage',required=True)
    pilot.add_argument('--horizon',required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == 'env-test':
        result = env_audit(config, args.output, args.contracts)
    elif args.command=='reward-study':
        from gym_trading_env.research.fg_single3_reward import reward_study
        result=reward_study(config,args.output,args.env_receipt,workers=args.workers,contracts=args.contracts)
    elif args.command=='oracle':
        from gym_trading_env.research.fg_single3_oracle import oracle_study
        result=oracle_study(config,args.output,args.env_receipt,args.reward_receipt,
            workers=args.workers,contracts=args.contracts,resume=args.resume)
    elif args.command=='replay-generate':
        from gym_trading_env.research.fg_single3_dataset import generate_dataset
        result=generate_dataset(config,args.output,args.env_receipt,args.reward_receipt,args.oracle_receipt,
            workers=args.workers,contracts=args.contracts,resume=args.resume)
    elif args.command=='replay-audit':
        from gym_trading_env.research.fg_single3_coverage import audit_dataset
        result=audit_dataset(config,args.dataset,args.output,batches=args.samples,seed=args.seed)
    elif args.command=='horizon-audit':
        from gym_trading_env.research.fg_single3_horizon import horizon_audit
        result=horizon_audit(config,args.dataset,args.coverage,args.output,
            workers=args.workers,snapshots_per_contract=args.snapshots_per_contract)
    elif args.command=='prepare-pilot':
        from gym_trading_env.research.fg_single3_pilot import prepare_pilot
        result=prepare_pilot(config,args.dataset,args.coverage,args.horizon,args.output)
    else:
        raise ValueError('Unimplemented command')
    print(f"{args.command}: {result['status']} → {Path(args.output).resolve()}", flush=True)


if __name__ == '__main__':
    main()
