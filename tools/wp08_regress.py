"""Run original WP06/WP07 regressions concurrently in owned isolated cwd."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'artifacts/fg_s3_v2/e_quality_wp08'/sys.argv[1]
base.mkdir(parents=True,exist_ok=False)
source='/home/v/Documents/work/research/fg-s3-e-quality-wp07-release-20261019/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp07/synthetic_source_01'
groups=sys.argv[2:] or ['online_'+x for x in 'abcdefgh']+['wp06_eval']


def run(group):
    name=sys.argv[1]+'_'+group
    env={**os.environ,'FG_WP_REGRESSION_RECEIPTS':'wp08','FG_WP07_SOURCE_ROOT':source,
        'FG_WP07_B_ROOT':str(base/(group+'_B')),'FG_WP07_C_ROOT':str(base/(group+'_C')),
        'FG_WP07_G_ROOT':str(base/(group+'_G')),
        'FG_WP07_H_PROBE':str(ROOT/'artifacts/fg_s3_v2/e_quality_wp07'/('wp08_'+name+'_probe')),
        'FG_WP06_EVAL_ROOT':str(base/(group+'_eval')),
        'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','JAX_PLATFORMS':'cpu','CUDA_VISIBLE_DEVICES':''}
    argv=[sys.executable,'-B',str(ROOT/'tests/entry_v1/wp07_regression_worker.py'),group,name]
    began=time.monotonic()
    with (base/(group+'.log')).open('x') as stream:r=subprocess.run(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    record=dict(group=group,argv=argv,cwd=str(ROOT),environment={k:v for k,v in env.items() if k.startswith('FG_WP')},
        exit_code=r.returncode,seconds=time.monotonic()-began,log=str(base/(group+'.log')))
    (base/(group+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    return record


with concurrent.futures.ThreadPoolExecutor(min(len(groups),os.cpu_count() or 1)) as pool:results=list(pool.map(run,groups))
(ROOT/'docs/receipts/wp08'/('H_regression_index_'+sys.argv[1]+'.json')).write_text(json.dumps(results,indent=2)+'\n')
raise SystemExit(int(any(r['exit_code'] for r in results)))
