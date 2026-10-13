#!/usr/bin/env python3
"""CPU-only S3-V2 D0: bounded real TRAIN rollout, Oracle checks and model smoke."""
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import time
import uuid


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))
sys.path.insert(0, str(PROJECT.parent/'dreamerv3'))
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def read_config(path):
    from dreamerv3.configuration import read_config_yaml
    config = read_config_yaml(path)
    if config['schema'] != 'FG_S3_V2_D0_CONFIG_V1' or config['fold'] != 'TRAIN':
        raise ValueError('D0 accepts only explicit TRAIN CPU prototype configurations')
    if not 100 <= config['env_steps'] <= 1000:
        raise ValueError('D0 config environment budget must be 100..1000')
    for key in ('dreamer_root', 'env_receipt', 'oracle_receipt', 'output_root', 'catalog', 'source_checkpoint'):
        config[key] = str((PROJECT/config[key]).resolve())
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect', 'verify_oracle', 'model_smoke'])
    parser.add_argument('--config', default=str(PROJECT/'configs/experiments/fg_s3_v2/D0_FG201.yaml'))
    parser.add_argument('--mode', choices=['wait', 'scripted', 'mixed'], default='mixed')
    parser.add_argument('--env-steps', type=int)
    parser.add_argument('--output')
    parser.add_argument('--save-at', type=int)
    parser.add_argument('--resume-collection')
    args = parser.parse_args()
    config = read_config(args.config)
    steps = args.env_steps if args.env_steps is not None else config['env_steps']
    if not 100 <= steps <= 1000:
        parser.error('D0 environment budget must be 100..1000; formal training is not supported')
    output = Path(args.output).resolve() if args.output else Path(config['output_root'])/(
        time.strftime('%Y%m%dT%H%M%S')+'_'+args.command+'_'+args.mode+'_'+uuid.uuid4().hex[:8])
    if output.exists():
        raise FileExistsError('D0 output already exists; existing artifacts are immutable')
    if not output.is_relative_to(PROJECT/'artifacts/fg_s3_v2_d0'):
        raise ValueError('D0 CLI output must remain in this development worktree prototype directory')
    from gym_trading_env.research.fg_s3_oracle_state_query import load_train_query
    from gym_trading_env.research.fg_replay import signed, write_json
    query, env, row, frozen, candidates = load_train_query(
        config['env_receipt'], config['oracle_receipt'], config['contract'])
    if args.command == 'collect':
        from gym_trading_env.research.fg_s3_dagger_d0 import D0Collector
        collector = D0Collector(env, query, output, config['dreamer_root'], mode=args.mode,
            seed=config['seed'], beta=config['oracle_mix_probability'],
            actor_probability=config['scripted_add_probability'])
        try:
            write_json(output/'runtime_identity.json', signed(dict(config=config,
                command=sys.argv, code_versions={str(root):dict(
                    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).splitlines())
                    for root in (PROJECT,PROJECT.parent/'dreamerv3')})))
            if args.resume_collection:
                collector.load_collection(args.resume_collection)
            if args.save_at is not None:
                if not collector.counts['env_steps'] < args.save_at < steps:
                    raise ValueError('save-at must be strictly inside the remaining budget')
                collector.advance(args.save_at)
                collector.save_collection(output/'collection_checkpoints'/f'env_{args.save_at:06d}')
            collector.advance(steps)
            result = collector.finish()
            write_json(output/'oracle_identity.json', signed(dict(identity=query.identity,
                identity_hash=query.identity_hash, precompute_seconds=query.precompute_seconds,
                dp_seconds=query.dp_seconds)))
        except BaseException as error:
            write_json(output/'failure.json', signed(dict(status='FAIL', error_type=type(error).__name__,
                error=str(error), env_steps=collector.counts['env_steps'])))
            raise
        finally:
            collector.close()
    elif args.command == 'verify_oracle':
        from gym_trading_env.research.fg_s3_dagger_verify import verify_real_oracle
        output.mkdir(parents=True)
        result = verify_real_oracle(query, row, frozen, candidates)
        write_json(output/'receipt.json', signed(result))
    else:
        from gym_trading_env.research.fg_s3_dagger_model_smoke import model_smoke
        result = model_smoke(config, query, env, output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('identity','files','cases')}, indent=2))


if __name__ == '__main__':
    main()
