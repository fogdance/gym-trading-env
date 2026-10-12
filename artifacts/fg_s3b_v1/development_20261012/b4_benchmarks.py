"""Final code 512/4096/8192 CLI benchmarks, fixed sources and one CPU thread."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

GYM = Path(__file__).resolve().parents[3]
env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(GYM/'src'),
           OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
           MPLCONFIGDIR='/tmp/fg_s3b_mpl',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
results = []
for count in (512,4096,8192):
    root = GYM/'artifacts/fg_s3b_v1'/f'benchmark_final_{count}'
    command = [sys.executable,'tools/fg_s3b.py','values-build','--config',
        'configs/experiments/fg_s3b/certified_v1.yaml','--contract','CZCE.FG001',
        '--limit',str(count),'--output',str(root)]
    began = time.monotonic()
    with (Path(__file__).parent/f'B4_benchmark_{count}.stdout.log').open('x') as f:
        result = subprocess.run(command,cwd=GYM,env=env,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError('Benchmark failed: '+str(count))
    m = json.loads((root/'contracts/CZCE.FG001/manifest.json').read_text())
    record = dict(rows=count,total_cli_seconds=time.monotonic()-began,
        contract_seconds=m['elapsed_seconds'],dp_seconds=m['dp_seconds'],peak_rss_bytes=m['peak_rss_bytes'],
        summary=m['summary'],proof_code_sha256=m['identity']['code_sha256'])
    results.append(record)
    print(f"final benchmark {count}: {record['contract_seconds']:.2f}s, DP {record['dp_seconds']:.3f}s",flush=True)
(Path(__file__).parent/'B4_benchmarks.json').write_text(json.dumps(results,indent=2)+'\n')
