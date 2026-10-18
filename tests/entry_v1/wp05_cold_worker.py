"""New interpreter, original FullCheckpoint, no F1 handoff during F2 resume."""
import json,pickle,sys,random
from pathlib import Path
import numpy as np
random.seed(202);np.random.seed(202)
from gym_trading_env.research.fg_s3_e_quality_heads import preflight
from embodied.run.fg_s3_e_quality_heads import HeadsRun,engineering_permit
from embodied.run.fg_checkpoint import host
s=json.loads(Path(sys.argv[1]).read_text());root=Path(sys.argv[2]);index=Path(sys.argv[3])
p=preflight(s);run=HeadsRun(p,index,root,resume=sys.argv[4]=='resume',_probe=engineering_permit(p))
result=run.run(int(sys.argv[5]))
with run.stream.pause():
 state=dict(agent=run.agent.save_exact(),reader=run.reader.state_dict(),runtime=run.runtime.save(),
  counters=run.counters.save(),schedule=run.schedule.save(),next_batch=run.stream.save_full()['pending'][0][0],last_batch=run.last_batch)
(root/('evidence_'+sys.argv[5]+'.pkl')).write_bytes(pickle.dumps(host(state),protocol=5))
print(json.dumps(result))
