"""Build immutable FG ABC experiment inputs; does not train."""
import argparse
from gym_trading_env.research.fg_abcd import build_abcd

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/fg_entry_v1/abcd_comparison.yaml')
    args=parser.parse_args()
    result=build_abcd(args.config)
    print('ABC manifest:',result['content_hash'])
    print('Folds:',len(result['folds']),'method instances:',len(result['instances']))
