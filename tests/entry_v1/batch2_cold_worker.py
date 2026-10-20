"""Fresh-interpreter data-only strict checkpoint worker; no optimizer or TEST."""
import argparse,json,pickle,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import embodied,elements
from embodied.run.fg_checkpoint import FullCheckpoint,RuntimeState,StepSchedule,host
from embodied.run.fg_s3_e_quality_online_state import JournalState
from gym_trading_env.research.fg_s3_reader_versions import (
    VersionedStream,CheckpointSource,VersionedRuntimeState,materialize_prefetch)
from batch2_reader_benchmark import open_case,wait_pending,tree_digest,NoModel


class History:
    def __init__(self):self.value=[]
    def save(self):return self.value
    def load(self,value):self.value=value


def make(config,mode,root,amount,seen=0):
    raw=open_case(config,mode)
    for r in raw.readers:
        r.seen_market={(r.contract,10_000_000+i) for i in range(seen)}
        r.seen_account={('DEV_ONLY_PRESSURE',i) for i in range(seen)}
    source=VersionedStream(raw);prefetch=embodied.streams.Prefetch(source,amount=amount)
    step=elements.Counter();history=History()
    runtime=RuntimeState(StepSchedule(1,2),SimpleNamespace(),[prefetch],lambda:None,lambda value:None,{},[0])
    runtime=VersionedRuntimeState.from_runtime(runtime,source)
    cp=FullCheckpoint(root/'ckpt',step,dict(protocol='BATCH2_EXACT_COLD_DATA_V2',source=source.binding,mode=mode,amount=amount),
        [prefetch],reserve_gib=0,keep_all=True,backend='external_immutable_course_reader',strict_latest=True)
    cp.course_reader=CheckpointSource(source);cp.agent=NoModel();cp.step=step;cp.runtime=runtime
    cp.logs=JournalState(root,('updates.jsonl',));cp.history=history
    return raw,source,prefetch,step,history,runtime,cp


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--root',required=True)
    p.add_argument('--mode',required=True);p.add_argument('--amount',type=int,default=4)
    p.add_argument('--stop',type=int,required=True);p.add_argument('--seen',type=int,default=0);p.add_argument('--resume',action='store_true');a=p.parse_args()
    random.seed(202);np.random.seed(202);root=Path(a.root)
    raw,source,prefetch,step,history,runtime,cp=make(json.loads(Path(a.config).read_text()),a.mode,root,a.amount,a.seen)
    restored=None
    if a.resume:
        expected=pickle.loads((cp.latest()/'runtime.pkl').read_bytes())['streams'][0]['source']['graph_sha256']
        cp.load();restored=prefetch.save_full()['source']['graph_sha256']
        assert restored==expected,'COLD_GRAPH_CHANGED_DURING_LOAD'
    else:cp.save()
    runtime.start()
    while int(step)<a.stop:
        batch=next(prefetch);step.increment();runtime.schedule.advance(int(step))
        history.value.append(tree_digest(batch))
        with (root/'updates.jsonl').open('a') as f:f.write(json.dumps(dict(consumed_batches=int(step)))+'\n')
    wait_pending(prefetch);cp.save(final=True)
    with prefetch.pause():
        state=runtime.save();state['streams']=[materialize_prefetch(source,state['streams'][0])]
        evidence=dict(reader=raw.save(),runtime=state,history=history.save(),step=int(step))
        (root/'evidence.pkl').write_bytes(pickle.dumps(host(evidence),protocol=5))
        result=dict(scope='DEV_ONLY_PRESSURE' if a.seen else 'LEGAL_SYNTHETIC_CPU_DATA',
            mode=a.mode,amount=a.amount,step=int(step),exact_graph_restored=restored is not None,
            restored_graph_sha256=restored,node_count=len(source.nodes),base_version=source.base_id,
            seen_keys=[len(r.seen_market) for r in source.readers],optimizer_updates=0)
        (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
