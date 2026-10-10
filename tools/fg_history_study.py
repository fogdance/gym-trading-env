"""FG historical study CLI. Outputs are ignored research artifacts."""
import argparse
import os
import tempfile

# Set before numpy/pandas imports in both parent and spawned worker processes.
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'fg_history_study_mpl'))
from gym_trading_env.research.fg_history_study import audit_snapshot, audit_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'audit-features', 'events', 'statistics', 'window-coverage', 'market-profile', 'fold-plan', 'fold-events'])
    parser.add_argument('--config', default='configs/fg_entry_v1/history_study.yaml')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--roles', nargs='+', choices=['train', 'validation'], default=['train', 'validation'])
    parser.add_argument('--contracts', nargs='+')
    parser.add_argument('--run', help='Completed event run directory for statistics')
    args = parser.parse_args()
    if args.command == 'audit':
        result = audit_snapshot(args.config)
        print(f"contracts={result['contracts']} rows={result['rows']} trading_days={result['trading_days']}")
    elif args.command == 'audit-features':
        result = audit_features(args.config)
        print(f"fields={len(result['fields'])} checks={len(result['checks'])} passed")
    elif args.command == 'events':
        from gym_trading_env.research.fg_history_parallel import run_events
        run_events(args.config, workers=args.workers, roles=args.roles, contracts=args.contracts)
        print('Development events computed; blind labels remain sealed.')
        return
    elif args.command in {'fold-plan', 'fold-events'}:
        from gym_trading_env.research.fg_history_folds import run_folds
        run_folds(args.config, workers=args.workers, plan_only=args.command == 'fold-plan')
        return
    elif args.command == 'window-coverage':
        from gym_trading_env.research.fg_history_windows import window_coverage
        print(window_coverage(args.config))
        return
    elif args.command == 'market-profile':
        from gym_trading_env.research.fg_history_profile import market_profile
        print(market_profile(args.config))
        return
    else:
        if not args.run:
            parser.error('statistics requires --run')
        from gym_trading_env.research.fg_history_statistics import describe_events
        describe_events(args.run, args.config)
        return
    print('No event labels computed; blind outcomes remain sealed.')


if __name__ == '__main__':
    main()
