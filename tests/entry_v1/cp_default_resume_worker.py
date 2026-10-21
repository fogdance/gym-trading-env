"""A fresh interpreter restores the real CPU fixture, never a real TRAIN Run."""
import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('stage',choices=('f1','f3'))
parser.add_argument('settings')
parser.add_argument('index')
parser.add_argument('root')
parser.add_argument('mode',choices=('start','resume'))
parser.add_argument('stop',type=int)
parser.add_argument('--changed-code',action='store_true')
parser.add_argument('--changed-settings',action='store_true')
args=parser.parse_args()
random.seed(202);np.random.seed(202)

from gym_trading_env.research import fg_s3_e_quality_foundation as f1
from gym_trading_env.research import fg_s3_e_quality_actor as f3
from embodied.run import fg_s3_e_quality_foundation as f1run
from embodied.run import fg_s3_e_quality_actor as f3run
from embodied.run.fg_checkpoint import host
from batch2_cold_evidence import runtime_evidence,prove_graph_restore

if args.changed_code:
    changed=lambda:{'gym':{'head':'0'*40,'runtime_hash':'a'*64,'dirty':True},
                    'dreamer':{'head':'1'*40,'runtime_hash':'b'*64,'dirty':True}}
    for module in (f1,f3,f1run,f3run):module.execution_identity=changed

settings=json.loads(Path(args.settings).read_text())
root=Path(args.root)
if args.changed_settings:settings['report_every_updates']+=1
if args.stage=='f1':
    preparation=f1.preflight(settings)
    run=f1run.FoundationRun(preparation,args.index,root,resume=args.mode=='resume',
        _probe=f1run.synthetic_probe_permit(preparation))
else:
    preparation=f3.preflight(settings,_resuming=args.mode=='resume')
    run=f3run.ActorRun(preparation,args.index,root,resume=args.mode=='resume',
        _probe=f3run.engineering_permit(preparation))
if args.mode=='resume':
    prove_graph_restore(run)
    # Runtime intentionally defers starting Prefetch until all members load.
    # Enter the same running lifecycle before comparing a zero-update restore.
    if getattr(run.runtime,'resume_started',False):run.runtime.start()
result=run.run(args.stop)
with run.stream.pause():
    reader=run.reader.state_dict() if args.stage=='f1' else run.source.source.save()
    state=dict(agent=run.agent.save_exact(),reader=reader,runtime=runtime_evidence(run),
        counters=run.counters.save(),schedule=run.schedule.save(),next_batch=run.stream.save_full()['pending'][0][0])
(root/'restore_evidence.pkl').write_bytes(pickle.dumps(host(state),protocol=5))
if args.stage=='f3' and args.mode=='resume' and not (root/'evaluation_fixture_result.json').exists():
    from cp_default_resume_evaluation import evaluate
    evaluate(settings,preparation,root)
print(json.dumps(result))
