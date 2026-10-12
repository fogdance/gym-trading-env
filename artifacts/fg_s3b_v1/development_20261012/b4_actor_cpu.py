"""Few-update real-data CPU interface tests, not an S3 research training run."""
import json
import os
from pathlib import Path
import subprocess
import sys

GYM = Path(__file__).resolve().parents[3]
DREAMER = Path('/home/v/Documents/work/dreamerv3-dev')
OUT = Path(__file__).parent
BASE = GYM/'artifacts/fg_s3b_v1/actor_cpu_interface_v2'
env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(GYM/'src')+':'+str(DREAMER),
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/fg_s3b_mpl',
    JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
labels = GYM/'artifacts/fg_s3a_v1/labels_20261011_v2'
source = GYM/'artifacts/fg_s3a_v1/smoke_real_v3/source/ckpt/step_000000000000'
features = GYM/'artifacts/fg_s3a_v1/smoke_real_v3/features'
weights = GYM/'artifacts/fg_s3b_v1/weights_full_v1'
receipts = []
stages = [('certified_resumed','S3B_CERTIFIED',3,False),('certified_resumed','S3B_CERTIFIED',6,True),
          ('certified_continuous','S3B_CERTIFIED',6,False),('matched','S3A_MATCHED',6,False),
          ('full','S3A_FULL',6,False)]
for attempt,(run,mode,updates,resume) in enumerate(stages):
    command = [sys.executable,'scripts/fg_s3b.py','train','--diagnostic','--device','cpu',
        '--s3a-label-index',str(labels),'--weights',str(weights),'--feature-cache',str(features),
        '--source-checkpoint',str(source),'--output',str(BASE/run),'--mode',mode,
        '--seed','0','--run-id','B4_CPU_INTERFACE','--batch-size','128','--updates',str(updates),
        '--comparison-updates','6','--save-every','3']
    if resume: command.append('--resume')
    with (OUT/f'B4_actor_cli_v2_{attempt}.stdout.log').open('x') as f:
        result = subprocess.run(command,cwd=DREAMER,env=env,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError('Actor CPU interface CLI failed: '+str(attempt))
    receipt = json.loads((BASE/run/'training_receipt.json').read_text())
    receipts.append(dict(run=run,mode=mode,updates=updates,resume=resume,
        command=command,status=receipt['status'],checkpoint=receipt['checkpoint'],
        classification=receipt['classification']['status']))
    print(f'CPU interface {mode}: {updates}, resume={resume}, {receipt["classification"]["status"]}',flush=True)
(OUT/'B4_actor_cli_commands_v2.json').write_text(json.dumps(receipts,indent=2)+'\n')
