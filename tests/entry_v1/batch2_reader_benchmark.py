"""Reader/Adapter/Prefetch benchmark. Artificial history is DEV pressure only."""
import argparse,copy,hashlib,json,os,pickle,resource,sys,time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import embodied,elements
from embodied.run.fg_checkpoint import FullCheckpoint,RuntimeState,StepSchedule,host
from data_multi_fixture import build_sources,descriptors,open_stream


def open_case(config,mode):
    from embodied.run.fg_s3_e_quality_foundation import CourseStream
    from embodied.run.fg_s3_e_quality_actor import ActorStreams
    from gym_trading_env.research.fg_s3_e_quality_reader import EQualityCourseReader
    from gym_trading_env.research.fg_s3_e_quality_foundation import F1WindowAdapter
    from gym_trading_env.research.fg_s3_e_quality_heads import F2WindowAdapter
    from gym_trading_env.research.fg_s3_e_quality_actor import F3WindowAdapter
    from dreamerv3.fg_s3_e_quality_actor import LOSS_DEFAULTS
    from test_data_multi_f3 import open_f3
    s=dict(batch_size=4,target_steps=8,burnin_steps=4,sampling_spec=dict(mode='NATURAL'),actor_loss=dict(LOSS_DEFAULTS))
    if mode=='multi_f3':return open_f3(config,'MIXED',settings=s)
    if mode=='multi':return open_stream(config).bind(s,F1WindowAdapter)
    stage='WM_NATURAL' if mode=='f1' else 'HEADS_VALID' if mode=='f2' else 'ACTOR_STRICT'
    source=descriptors(config,stage)[0];records=[]
    if mode in ('f3','f3_aux'):
        s.update(model_init=dict(seed=202),aux_sampling_spec=None if mode=='f3' else
            dict(mode='MIXED',natural_weight=.5,strata_weights=dict(STRICT_ADD=1,STRICT_WAIT=1)))
        return ActorStreams(SimpleNamespace(settings=s,source=source['admission']),source['index_path'])
    roles=['NATURAL','AUXILIARY'] if mode=='f3_aux' else ['NATURAL']
    for i,role in enumerate(roles):
        spec=dict(mode='NATURAL') if i==0 else dict(mode='MIXED',natural_weight=.5,strata_weights=dict(STRICT_ADD=1,STRICT_WAIT=1))
        reader=EQualityCourseReader.open(source['admission'],stage=stage,sampling_spec=spec,seed=202+i,index_path=source['index_path'])
        adapter=F1WindowAdapter if mode=='f1' else F2WindowAdapter if mode=='f2' else F3WindowAdapter
        stream=CourseStream(reader,{**s,'sampling_spec':spec},adapter_cls=adapter)
        if hasattr(stream.adapter,'validate_support'):stream.adapter.validate_support()
        records.append((role,stream))
    if mode in ('f1','f2'):return records[0][1]
    return ActorStreams.from_streams(records,identity=dict(scope='BATCH2_SYNTHETIC',mode=mode))


def readers(source):return list(source.readers.values()) if isinstance(source.readers,dict) else list(source.readers)


def tree_digest(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,dict):
            for k in sorted(x):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(str(len(x)).encode())
            for item in x:visit(item)
        elif isinstance(x,np.ndarray):
            h.update(str((x.shape,x.dtype)).encode());h.update(x.tobytes())
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def wait_pending(prefetch):
    deadline=time.monotonic()+120
    with prefetch.queue.not_empty:
        while prefetch.queue._qsize()<prefetch.amount:
            if any(isinstance(x,str) for x in prefetch.queue.queue):raise RuntimeError('PREFETCH_FAILED')
            if time.monotonic()>deadline:raise TimeoutError('PREFETCH_FILL_TIMEOUT')
            prefetch.queue.not_empty.wait(.2)


class NoModel:
    params={}
    def save_exact(self):return dict(no_model=True)
    def load_exact(self,state):assert state==dict(no_model=True)
    save=save_exact
    load=load_exact


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=['f1','f2','f3','f3_aux','multi','multi_f3'],default='f1')
    p.add_argument('--seen',type=int,default=0);p.add_argument('--batches',type=int,default=8)
    p.add_argument('--amount',type=int,default=2);p.add_argument('--versioned',action='store_true');a=p.parse_args()
    config_path=Path(a.config)
    if not config_path.exists():
        built=build_sources(config_path.parent/'sources');config_path.write_bytes(built.read_bytes())
    config=json.loads(config_path.read_text());source=open_case(config,a.mode)
    for r in readers(source):
        r.seen_market={(r.contract,10_000_000+i) for i in range(a.seen)}
        r.seen_account={('DEV_ONLY_PRESSURE',i) for i in range(a.seen)}
    bound=source
    if a.versioned:
        from gym_trading_env.research.fg_s3_reader_versions import VersionedStream
        bound=VersionedStream(source)
    prefetch=embodied.streams.Prefetch(bound,amount=a.amount)
    runtime=RuntimeState(StepSchedule(1,3),SimpleNamespace(),[prefetch],lambda:None,lambda x:None,{},[0])
    runtime.start()
    for _ in range(2):next(prefetch)
    wait_pending(prefetch);batches=[];start=time.perf_counter()
    for _ in range(a.batches):batches.append(host(next(prefetch)))
    wait_pending(prefetch);elapsed=time.perf_counter()-start
    output=Path(a.out);output.mkdir(parents=True,exist_ok=False)
    with prefetch.pause():
        final_state=source.save();golden=tree_digest(dict(batches=batches,state=final_state))
        t=time.perf_counter();snapshot=prefetch.save_full();snapshot_seconds=time.perf_counter()-t
        snapshot_bytes=len(pickle.dumps(snapshot,protocol=5))
        step=elements.Counter();step.increment(a.batches)
        cp=FullCheckpoint(output/'ckpt',step,dict(protocol='BATCH2_READER_PERF_ONLY',mode=a.mode,seen=a.seen),[prefetch],reserve_gib=0,
            keep_all=True,backend='external_immutable_course_reader',strict_latest=True)
        if a.versioned:
            from gym_trading_env.research.fg_s3_reader_versions import CheckpointSource,VersionedRuntimeState
            cp.course_reader=CheckpointSource(bound)
            runtime=VersionedRuntimeState.from_runtime(runtime,bound)
        else:cp.course_reader=source
        cp.agent=NoModel();cp.step=step;cp.runtime=runtime
        t=time.perf_counter();cp.save(final=True);cp_seconds=time.perf_counter()-t
        cp_bytes=sum(p.stat().st_size for p in cp.latest().rglob('*') if p.is_file())
    result=dict(scope='DEV_ONLY_COVERAGE_PRESSURE' if a.seen else 'LEGAL_SYNTHETIC_CORPUS',
        mode=a.mode,seen_keys_per_reader=a.seen,reader_count=len(readers(source)),versioned=a.versioned,
        batches=a.batches,batch_seconds=elapsed,batches_per_second=a.batches/elapsed,
        seconds_per_100k_target_positions=elapsed/(sum(len(b['stepid'])*8 for b in batches)/100000),
        golden=golden,batch_digests=[tree_digest(b) for b in batches],state_digest=tree_digest(final_state),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        prefetch_amount=a.amount,pending_batches=len(snapshot['pending']),snapshot_seconds=snapshot_seconds,
        snapshot_bytes=snapshot_bytes,checkpoint_seconds=cp_seconds,checkpoint_bytes=cp_bytes,
        optimizer_updates=0,training_admitted=False,real_market_coverage='NOT_MEASURED_BY_PRESSURE_FIXTURE',argv=sys.argv,
        runtime_paths=dict(embodied=embodied.__file__,reader=__import__('gym_trading_env.research.fg_s3_e_quality_reader',fromlist=['x']).__file__))
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
