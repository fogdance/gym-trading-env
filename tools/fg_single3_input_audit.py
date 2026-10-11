#!/usr/bin/env python3
"""TRAIN-only real-prefix observation audit; no Oracle label becomes a tensor."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['JAX_PLATFORMS']='cpu'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
from pathlib import Path
import sys
import argparse
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
from gym_trading_env.research.fg_single3 import load_config,load_env,make_output,source_witness,execution_provenance
from gym_trading_env.research.fg_single3_replay import require_e
from gym_trading_env.research.fg_replay import FIELDS,check_files,digest,signed,write_json
from gym_trading_env.utils.market_features import build_market_features,FEATURES_MARKET_OBS_OUTPUT


def job(spec):
    began=time.monotonic();env=load_env(spec['row'],spec['frozen'])
    n=len(env._raw);rng=np.random.default_rng(np.random.SeedSequence([202,spec['index']]))
    boundaries=np.flatnonzero(env._raw.trading_date.to_numpy()[1:]!=env._raw.trading_date.to_numpy()[:-1])+1
    stops={1,min(60,n),min(1024,n),n}
    stops.update(map(int,rng.choice(np.arange(2,n),4,replace=False)))
    if len(boundaries):stops.update(int(boundaries[i])+1 for i in sorted({0,len(boundaries)//2,len(boundaries)-1}))
    for stop in sorted(stops):
        prefix=build_market_features(env._raw.iloc[:stop],is_future=True,real_futures_clock=True,
                                     futures_schedule=env.market_clock)
        values=prefix[FEATURES_MARKET_OBS_OUTPUT].to_numpy(dtype=np.float32)
        np.testing.assert_array_equal(values[max(0,stop-60):stop],env._features[max(0,stop-60):stop],
                                      err_msg=f'Future suffix affected observation: {env.contract}/{stop}')
    result=dict(status='PASS',contract=env.contract,price_sha256=spec['row']['sha256'],
        exact_float32_prefix_stops=sorted(stops),window=60,fields=FEATURES_MARKET_OBS_OUTPUT,
        minutes=n,seconds=time.monotonic()-began)
    check_files(spec['frozen']);env.close();return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-receipt',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();config=load_config('configs/fg_single3_v1/cpu.yaml')
    inputs,e=require_e(args.env_receipt);output=make_output(args.output,config)
    code=execution_provenance(config,'E');code['code_files'][str(Path(__file__).resolve())]=digest(__file__)
    snapshots=source_witness(output,code);live={**inputs['frozen_files'],**code['code_files']}
    specs=[dict(row=r,index=i,frozen=live) for i,r in enumerate(inputs['views']['C2']['rows'])]
    results=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(job,s) for s in specs]):
            r=future.result();results.append(r);print(f"Observation prefixes {r['contract']}: PASS {len(r['exact_float32_prefix_stops'])} cuts",flush=True)
    check_files(live)
    receipt=signed(dict(stage='INPUT_AUDIT',status='PASS',schema='FG_SINGLE3_CPU_V1',
        full_training_fold=True,contracts=sorted(results,key=lambda r:r['contract']),
        tensor_whitelist={k:[list(shape),str(dtype)] for k,(shape,dtype) in FIELDS.items()},
        hindsight_fields_in_tensors=False,provenance=code,source_snapshots=snapshots,
        frozen_inputs=inputs['frozen_files'],gradient_updates=0,
        files={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}))
    write_json(output/'manifest.pending.json',receipt);check_files(live)
    check_files({str(output/p):h for p,h in receipt['files'].items()})
    (output/'manifest.pending.json').replace(output/'manifest.json')
    print('INPUT_AUDIT PASS; only frozen TRAIN prices were opened',flush=True)

if __name__=='__main__':main()
