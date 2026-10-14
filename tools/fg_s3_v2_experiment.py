#!/usr/bin/env python3
"""Prepare, train, evaluate S3-V2 and simulate frozen Actors on live DB bars."""
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
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fg_s3_v2_mpl')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'start', 'resume', 'evaluate', 'report', 'gpu_check',
                                     'score-predictions', 'simulate-online', 'describe-composition'])
    p.add_argument('--config')
    p.add_argument('--prepared')
    p.add_argument('--run-dir')
    p.add_argument('--stop-at', type=int)
    p.add_argument('--eval-steps', type=int, default=1000)
    p.add_argument('--eval-dir', help='Existing sealed Actor-only evaluation for CPU post-processing')
    p.add_argument('--output-dir', help='New isolated prediction scoring directory')
    p.add_argument('--check-only', action='store_true', help='Online DB/model/RPC preflight; emits no orders')
    p.add_argument('--resume-online', action='store_true', help='Restore the same online account and policy carry')
    p.add_argument('--authorize-training', action='store_true',
                   help='Explicit user authorization for formal 1M start/resume; diagnostics do not require it')
    args = p.parse_args()
    if (args.check_only or args.resume_online) and args.command != 'simulate-online':
        p.error('--check-only and --resume-online require simulate-online')
    # Composition preflight precedes trainer imports and CUDA selection. The
    # shared pure resolver is also used by direct prepare/runtime/model callers.
    from dreamerv3.configuration import read_config_yaml
    from dreamerv3.fg_s3_composition import normalize_config, resolve_composition, settings_composition
    if args.command in ('prepare', 'start', 'resume', 'evaluate', 'report', 'describe-composition'):
        if args.config:
            path = Path(args.config)
            if not path.is_file():
                path = PROJECT/'configs/experiments/fg_s3_v2'/f'{args.config}.yaml'
            raw = read_config_yaml(path)
            normalize_config(raw, executable=args.command != 'describe-composition')
            if args.command == 'describe-composition':
                print(json.dumps(dict(resolved_composition=resolve_composition(raw),
                    legacy_c2=raw.get('mode') == 'c2'), indent=2))
                return
        elif args.command == 'describe-composition':
            p.error('describe-composition requires --config')
        elif args.prepared:
            settings_composition(json.loads((Path(args.prepared)/'settings.json').read_text())['settings'])
    os.environ['JAX_PLATFORMS'] = 'cpu' if args.command in ('prepare','report','score-predictions') else 'cuda'
    os.environ.pop('JAX_PLATFORM_NAME', None)
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    import gym_trading_env, dreamerv3, embodied
    if (Path(gym_trading_env.__file__).resolve().parent != PROJECT/'src/gym_trading_env'
            or Path(dreamerv3.__file__).resolve().parent != PROJECT.parent/'dreamerv3/dreamerv3'
            or Path(embodied.__file__).resolve().parent != PROJECT.parent/'dreamerv3/embodied'):
        raise ValueError('Foreign research Python imports')
    from gym_trading_env.research.fg_s3_v2_experiment import prepare, read_settings, settings_from_prepared
    if args.command == 'simulate-online':
        if not args.config:
            p.error('simulate-online requires --config')
        from gym_trading_env.research.fg_s3_online_simulation import simulate_online
        result = simulate_online(args.config, run_dir=args.run_dir,
                                 resume=args.resume_online, check_only=args.check_only)
    elif args.command == 'score-predictions':
        if not args.eval_dir or not args.output_dir:
            p.error('score-predictions requires --eval-dir and a new --output-dir')
        from gym_trading_env.research.fg_s3_prediction_evaluation import score_predictions
        result = score_predictions(args.eval_dir, args.output_dir)
    elif args.command == 'prepare':
        if not args.config or args.prepared:
            p.error('prepare requires a configuration')
        result = prepare(args.config)
    elif args.command == 'gpu_check':
        from gym_trading_env.research.fg_s3_v2_gpu_check import check_gpu_model
        result = check_gpu_model()
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
