"""New interpreter; original FullCheckpoint and original shared training loop."""
import json,pickle,random,sys
from pathlib import Path
import numpy as np
random.seed(202);np.random.seed(202)
from gym_trading_env.research.fg_s3_e_quality_actor import preflight
from embodied.run.fg_s3_e_quality_actor import ActorRun,engineering_permit
from embodied.run.fg_checkpoint import host
s=json.loads(Path(sys.argv[1]).read_text());root=Path(sys.argv[2]);index=Path(sys.argv[3])
p=preflight(s);run=ActorRun(p,index,root,resume=sys.argv[4]=='resume',_probe=engineering_permit(p))
if sys.argv[6]=='skip_first':
    import jax.numpy as jnp
    original=run.agent.train
    def once(carry,data):
        if int(run.step)==0:data={**data,'teacher_valid':jnp.zeros_like(data['teacher_valid'])}
        return original(carry,data)
    run.agent.train=once
stop=int(sys.argv[5])
if stop==-1:
    run.update();run.cp.save(final=True);result=run.result()
else:result=run.run(stop)
with run.stream.pause():
    evidence=dict(agent=run.agent.save_exact(),reader=run.source.save(),runtime=run.runtime.save(),
        counters=run.counters.save(),schedule=run.schedule.save(),
        next_batch=run.stream.save_full()['pending'][0][0],last_batch=run.last_batch)
(root/('evidence_'+str(stop)+'.pkl')).write_bytes(pickle.dumps(host(evidence),protocol=5))
print(json.dumps(result))
