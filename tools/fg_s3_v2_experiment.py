#!/usr/bin/env python3
"""Prepare, run, restore and evaluate S3-V2; CPU is for isolated C2 diagnostics."""
import argparse
import json
import os
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
# The CUDA-enabled environment ships a coherent set of NVIDIA libraries.
# A host CUDA LD_LIBRARY_PATH overrides some of them and failed cuBLAS model
# initialization in the real GPU check. Restart only this process with its
# normal package library search; never mutate the shared environment/system.
if os.environ.get('LD_LIBRARY_PATH'):
    clean = dict(os.environ)
    clean.pop('LD_LIBRARY_PATH')
    clean['FG_S3_V2_CUDA_ENV_CLEANED'] = '1'
    os.execve(sys.executable, [sys.executable, *sys.argv], clean)
sys.path.insert(0, str(PROJECT/'src'))
sys.path.insert(0, str(PROJECT.parent/'dreamerv3'))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fg_s3_v2_mpl')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'start', 'resume', 'evaluate', 'report', 'gpu_check', 'c2_check'])
    p.add_argument('--config')
    p.add_argument('--prepared')
    p.add_argument('--run-dir')
    p.add_argument('--stop-at', type=int)
    p.add_argument('--eval-steps', type=int, default=1000)
    p.add_argument('--device', choices=['cpu', 'gpu'],
                   help='Only c2_check selects a backend here; runs use their immutable YAML device')
    p.add_argument('--authorize-training', action='store_true',
                   help='Explicit user authorization for formal 1M start/resume; diagnostics do not require it')
    args = p.parse_args()
    if args.command == 'c2_check':
        if args.config or args.prepared or args.stop_at or args.authorize_training:
            p.error('c2_check is a controlled fixture, not a prepared or formal training run')
        device = args.device or 'cpu'
    elif args.command == 'gpu_check':
        if args.device:
            p.error('gpu_check always requires GPU')
        device = 'gpu'
    else:
        if args.device:
            p.error('Run backend is selected by its immutable configuration')
        if not args.config and not args.prepared:
            p.error('Select --config or --prepared')
        # The canonical reader imports no JAX. Select the backend before any
        # model module can initialize CUDA in a CPU diagnostic process.
        from dreamerv3.configuration import read_config_yaml
        if args.config:
            path = Path(args.config)
            if not path.is_file():
                path = PROJECT/'configs/experiments/fg_s3_v2'/f'{path}.yaml'
            device = read_config_yaml(path)['device']
        else:
            device = json.loads((Path(args.prepared)/'settings.json').read_text())['settings']['device']
        if device not in ('cpu', 'gpu'):
            p.error('Unknown configured device')
    os.environ['JAX_PLATFORMS'] = 'cpu' if device == 'cpu' else 'cuda'
    os.environ.pop('JAX_PLATFORM_NAME', None)
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import gym_trading_env, dreamerv3, embodied
    if (Path(gym_trading_env.__file__).resolve().parent != PROJECT/'src/gym_trading_env'
            or Path(dreamerv3.__file__).resolve().parent != PROJECT.parent/'dreamerv3/dreamerv3'
            or Path(embodied.__file__).resolve().parent != PROJECT.parent/'dreamerv3/embodied'):
        raise ValueError('Foreign research Python imports')
    from gym_trading_env.research.fg_s3_v2_experiment import prepare, read_settings, settings_from_prepared
    if args.command == 'prepare':
        if not args.config or args.prepared:
            p.error('prepare requires a configuration')
        result = prepare(args.config)
    elif args.command == 'gpu_check':
        from gym_trading_env.research.fg_s3_v2_gpu_check import check_gpu_model
        result = check_gpu_model()
    elif args.command == 'c2_check':
        from gym_trading_env.research.fg_s3_v2_c2_check import check_c2_model
        result = check_c2_model(args.run_dir, device=device)
    else:
        if not args.config and not args.prepared:
            p.error('Select --config or --prepared')
        settings = read_settings(args.config) if args.config else settings_from_prepared(args.prepared)
        prepared = args.prepared or settings['prepared_dir']
        frozen = settings_from_prepared(prepared)
        if settings != frozen:
            raise ValueError('Selected configuration differs from the prepared snapshot')
        run_dir = args.run_dir or settings['run_dir']
        if args.command == 'report':
            from gym_trading_env.research.fg_replay import read_manifest
            result = read_manifest(Path(run_dir)/'training_receipt.json')
        elif args.command == 'evaluate':
            from gym_trading_env.research.fg_s3_v2_evaluation import evaluate
            result = evaluate(prepared, run_dir, args.eval_steps)
        else:
            from embodied.run.fg_s3_v2_train import LiveRun
            run = LiveRun(prepared, run_dir, resume=args.command == 'resume',
                          authorize_training=args.authorize_training)
            try:
                result = run.run_to(args.stop_at)
            except BaseException as e:
                from gym_trading_env.research.fg_replay import signed, write_json
                write_json(run.root/'failure.json', signed(dict(error_type=type(e).__name__, error=str(e),
                    env_steps=int(run.step), updates=run.counters.save())))
                raise
            finally:
                run.close()
    print(json.dumps({k: v for k, v in result.items() if k not in
                     ('identity', 'code', 'files', 'immutable_files', 'dirty')}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
