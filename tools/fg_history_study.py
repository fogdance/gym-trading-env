"""FG historical study CLI. Outputs are ignored research artifacts."""
import argparse
from gym_trading_env.research.fg_history_study import audit_snapshot, audit_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'audit-features'])
    parser.add_argument('--config', default='configs/fg_entry_v1/history_study.yaml')
    args = parser.parse_args()
    if args.command == 'audit':
        result = audit_snapshot(args.config)
        print(f"contracts={result['contracts']} rows={result['rows']} trading_days={result['trading_days']}")
    else:
        result = audit_features(args.config)
        print(f"fields={len(result['fields'])} checks={len(result['checks'])} passed")
    print('No event labels computed; blind outcomes remain sealed.')


if __name__ == '__main__':
    main()
