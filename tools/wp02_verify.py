"""Record exact CPU verification commands and exits; never launches training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
out = root/'artifacts/fg_s3_v2/e_quality_wp02/evidence'
out.mkdir(parents=True, exist_ok=True)
name, *tests = sys.argv[1:]
if not tests or not name.replace('_','').isalnum(): raise SystemExit('name and pytest paths required')
env = {**os.environ, 'PYTHONPATH':'.:src:../dreamerv3:tests/entry_v1',
       'JAX_PLATFORMS':'cpu', 'CUDA_VISIBLE_DEVICES':'', 'PYTHONDONTWRITEBYTECODE':'1',
       'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1'}
args = ['/home/v/miniconda3/envs/dreamerv3/bin/python','-B','-m','pytest',
        '-p','no:cacheprovider','-o','addopts=','-q', '--junitxml='+str(out/(name+'.xml')), *tests]
began = time.monotonic()
with (out/(name+'.log')).open('w') as f:
    result = subprocess.run(args,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
record = dict(argv=args,cwd=str(root),environment={k:env[k] for k in
    ['PYTHONPATH','JAX_PLATFORMS','CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},
    exit_code=result.returncode,seconds=time.monotonic()-began)
(out/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record)); print((out/(name+'.log')).read_text()[-10000:])
raise SystemExit(result.returncode)
