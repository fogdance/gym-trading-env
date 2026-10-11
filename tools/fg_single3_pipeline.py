#!/usr/bin/env python3
"""CPU-only Gate-ordered continuation after the signed exact Oracle finishes.

Each command runs the ordinary reusable stage CLI and publishes its own receipt.
A failed command stops the chain, retains evidence and cannot publish SUCCESS.
No Agent/GPU training command exists in this pipeline.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env',required=True)
    parser.add_argument('--reward',required=True)
    parser.add_argument('--oracle',required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--coverage',required=True)
    parser.add_argument('--horizon',required=True)
    parser.add_argument('--pilot',required=True)
    parser.add_argument('--log',required=True)
    parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    cwd=Path(__file__).resolve().parents[1]
    log=Path(args.log).resolve();log.parent.mkdir(parents=True,exist_ok=True)
    if log.exists():raise FileExistsError('Pipeline log exists; preserve prior attempt')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
             MPLCONFIGDIR='/tmp/fg_single3_mpl')
    commands=[
        ['replay-generate','--env-receipt',args.env,'--reward-receipt',args.reward,'--oracle-receipt',args.oracle,
             '--output',args.dataset,'--workers',str(args.workers)],
        ['replay-audit','--dataset',args.dataset,'--output',args.coverage],
        ['horizon-audit','--dataset',args.dataset,'--coverage',args.coverage,'--output',args.horizon,'--workers',str(args.workers)],
        ['prepare-pilot','--dataset',args.dataset,'--coverage',args.coverage,'--horizon',args.horizon,'--output',args.pilot]]
    with log.open('x',buffering=1) as stream:
        stream.write('WAITING_FOR_EXACT_ORACLE\n')
        oracle=Path(args.oracle).resolve()
        while not (oracle/'manifest.json').exists():
            if (oracle/'failure.json').exists():raise RuntimeError('Oracle failed; formal downstream refused')
            time.sleep(30)
        # The stage CLI validates signature, source/artifact hashes and exact O
        # status before collecting any downstream data.
        for command in commands:
            started=time.time();stream.write(json.dumps(dict(status='START',time=started,command=command))+'\n')
            result=subprocess.run([sys.executable,str(cwd/'tools/fg_single3.py'),*command],cwd=cwd,env=env,
                stdout=stream,stderr=subprocess.STDOUT)
            stream.write(json.dumps(dict(status='PASS' if result.returncode==0 else 'FAIL',
                command=command,seconds=time.time()-started,returncode=result.returncode))+'\n')
            if result.returncode:raise RuntimeError('CPU pipeline stopped; inspect '+str(log))
        stream.write('CPU_PIPELINE_COMPLETED_GPU_NOT_RUN\n')
    print('CPU stages completed, GPU NOT_RUN. Evidence log: '+str(log),flush=True)


if __name__=='__main__':main()
