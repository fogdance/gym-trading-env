#!/usr/bin/env python3
"""Read one A2 run and invoke its original frozen Actor-only GPU evaluator."""
import argparse
import fcntl
from pathlib import Path
import sys


sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--eval-steps',type=int,default=5000)
    p.add_argument('--watch',action='store_true')
    p.add_argument('--no-evaluate',action='store_true')
    a=p.parse_args()
    from gym_trading_env.research.fg_s3_v2_observation import Observer
    observer=Observer(a.prepared,a.run_dir,a.eval_steps)
    with (observer.output/'observer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=observer.run(a.watch,not a.no_evaluate)
    print('Observed',result['live_train'].get('env_steps',0),'TRAIN steps; output=',observer.output)


if __name__=='__main__':
    main()
