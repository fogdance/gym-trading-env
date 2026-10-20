"""Actual cold interpreter for the controlled CPU checkpoint contract."""
import json,pickle,sys
import random
import numpy as np
random.seed(202);np.random.seed(202)
from pathlib import Path
from gym_trading_env.research.fg_s3_e_quality_foundation import preflight
from embodied.run.fg_s3_e_quality_foundation import FoundationRun,synthetic_probe_permit
from embodied.run.fg_checkpoint import host
from batch2_cold_evidence import runtime_evidence,prove_graph_restore
s=json.loads(Path(sys.argv[1]).read_text());root=Path(sys.argv[2]);index=Path(sys.argv[3])
prep=preflight(s)
run=FoundationRun(prep,index,root,resume=sys.argv[4]=='resume',_probe=synthetic_probe_permit(prep))
if sys.argv[4]=='resume':prove_graph_restore(run)
result=run.run(int(sys.argv[5]))
# Snapshot with original Prefetch paused. Obtain the queued next batch without
# consuming it or advancing counters, exactly as FullCheckpoint does.
with run.stream.pause():
 state=dict(agent=run.agent.save_exact(),reader=run.reader.state_dict(),
   runtime=runtime_evidence(run),counters=run.counters.save(),schedule=run.schedule.save(),
   next_batch=run.stream.save_full()['pending'][0][0],last_batch=run.last_batch)
(root/('evidence_'+sys.argv[5]+'.pkl')).write_bytes(pickle.dumps(host(state),protocol=5))
print(json.dumps(result))
