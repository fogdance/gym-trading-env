#!/usr/bin/env python3
"""Restore real Single3 full checkpoints on CPU without training or price evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['JAX_PLATFORMS']='cpu'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/fg_gate_matplotlib')

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DREAMER=ROOT.parent/'dreamerv3'
sys.path[:0]=[str(ROOT/'src'),str(DREAMER)]


def signature(value):
    """Semantic hash, independent of pickle encodings and array device placement."""
    import numpy as np
    h=hashlib.sha256()
    def walk(x):
        if isinstance(x,dict):
            h.update(b'dict')
            for k in sorted(x,key=repr):
                walk(k);walk(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode());h.update(str(len(x)).encode())
            for v in x:walk(v)
        elif hasattr(x,'dtype') and hasattr(x,'shape'):
            array=np.asarray(x)
            h.update(str(array.dtype).encode());h.update(str(array.shape).encode());h.update(array.tobytes())
        elif hasattr(x,'__dict__'):
            h.update((type(x).__module__+'.'+type(x).__qualname__).encode())
            walk(vars(x))
        elif callable(x):
            h.update((getattr(x,'__module__','')+'.'+getattr(x,'__qualname__',repr(x))).encode())
        else:
            h.update(type(x).__name__.encode());h.update(repr(x).encode())
    walk(value)
    return h.hexdigest()


def verify(paths,output):
    import elements
    import embodied
    import jax
    import numpy as np
    from dreamerv3.configuration import read_config_yaml
    from dreamerv3.main import make_agent,make_env,make_stream
    from embodied.run.fg_checkpoint import FullCheckpoint,RuntimeState,StepSchedule,validate_bundle,file_hash
    from embodied.run.action_mask_warmup import ActionMaskWarmup
    started=time.monotonic()
    output.mkdir(parents=True,exist_ok=False)
    metas=[validate_bundle(p) for p in paths]
    before={str(p):file_hash(p/'manifest.json') for p in paths}
    base=elements.Config(read_config_yaml(paths[0]/'config.yaml'))
    # Backend overrides exist only inside this CPU test. Frozen target config is untouched.
    cpu=base.update({'jax.platform':'cpu','jax.precompile':False,'jax.prealloc':False,
        'logdir':str(output/'neural_cpu')})
    agent=make_agent(cpu)
    assert all(d.platform=='cpu' for d in jax.devices())
    results=[]
    for path,meta in zip(paths,metas):
        cfg=elements.Config(read_config_yaml(path/'config.yaml'))
        saved={k:pickle.loads((path/(k+'.pkl')).read_bytes()) for k in ('agent','runtime','driver','action_mask_warmup','replay')}
        rs=saved['replay']
        replay=embodied.Replay(length=rs['length'],capacity=rs['capacity'],chunksize=rs['chunksize'],online=rs['online'],seed=202)
        train=agent.stream(make_stream(cfg,replay,'train'))
        report=agent.stream(make_stream(cfg,replay,'report'))
        schedule=StepSchedule(cfg.run.fg_report_every_steps,cfg.run.fg_save_every_steps,cfg.run.fg_key_steps)
        ratio=elements.when.Ratio(cfg.run.train_ratio/1024)
        carries=[None]
        def set_carry(x):
            carries[0]=dict(x)
            carries[0]['train']=agent.restore_carry(x['train'])
            carries[0]['report']=agent.restore_carry(x['report'])
        aggregates={k:elements.Agg() for k in saved['runtime']['aggregates']}
        samples=[0]
        runtime=RuntimeState(schedule,ratio,(train,report),lambda:carries[0],set_carry,aggregates,samples)
        gate=ActionMaskWarmup.for_single3_1m(cfg.action_mask_warmup)
        driver=embodied.Driver([lambda:make_env(cfg,0)],parallel=False)
        driver.reset(agent.init_policy)
        step=elements.Counter()
        cp=FullCheckpoint(path.parent,step,meta['identity'],(train,report),reserve_gib=0,keep_all=True)
        cp.step=step;cp.agent=agent;cp.replay=replay;cp.runtime=runtime;cp.driver=driver;cp.action_mask_warmup=gate
        try:
            assert cp.latest()==path
            cp.load()
            python_rng_equal=random.getstate()==saved['runtime']['python_rng']
            numpy_rng_equal=signature(np.random.get_state())==signature(saved['runtime']['numpy_rng'])
            restored_agent=agent.save_exact()
            assert signature(restored_agent)==signature(saved['agent'])
            assert signature(driver.save())==signature(saved['driver'])
            assert gate.save()==saved['action_mask_warmup'] and not agent.get_avail_actor_enabled()
            assert int(step)==meta['step']==180000
            # Resume both native prefetchers with their retained queued batches.
            runtime.start()
            assert signature(runtime.save())==signature(saved['runtime'])
            assert python_rng_equal and numpy_rng_equal
            for key in ('refs','items','fifo','itemid','current','streams','lengths','queue'):
                if key in saved['replay']:
                    assert signature(getattr(replay,key))==signature(saved['replay'][key]),key
            assert replay.sampler.rng.bit_generator.state==saved['replay']['sampler']['rng']
            for chunk in rs['chunks']:
                restored=replay.chunks[chunk['uuid']]
                with np.load(path/'replay'/chunk['name']) as z:
                    for key in z.files:
                        np.testing.assert_array_equal(restored.data[key][:chunk['length']],z[key])
            results.append(dict(path=str(path),identity=meta['identity'],step=int(step),
                agent_counters=restored_agent['counters'],agent_semantic_sha256=signature(restored_agent),
                parameter_groups={name:signature({k:v for k,v in restored_agent['params'].items() if k.startswith(prefix)})
                    for name,prefix in [('world_model',('enc/','dyn/','dec/','rew/','con/','avail/')),('actor','pol/'),('critic',('val/','slowval/')),('optimizer','opt/')]},
                financial_account_sha256=signature(driver.save()),runtime_sha256=signature(runtime.save()),
                replay_arrays_equal=True,replay_sampler_rng_equal=True,prefetch_restored=True,
                python_rng_equal=python_rng_equal,numpy_rng_equal=numpy_rng_equal,
                gate=gate.save(),actor_enabled=False,gradient_updates_performed=0))
            print('CPU_RESTORE_PASS',path,restored_agent['counters'],flush=True)
        finally:
            driver.close()
    for p in paths:
        assert file_hash(p/'manifest.json')==before[str(p)]
        validate_bundle(p)
    result=dict(status='REAL_NEURAL_FULL_CPU_RESTORE_PASS',results=results,
        elapsed_seconds=time.monotonic()-started,source_hashes_before_after=before,
        devices=[str(d) for d in jax.devices()],gradient_updates_performed=0,dev_test_accessed=False)
    (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps(dict(files={'analysis.json':file_hash(output/'analysis.json'),
        'script':file_hash(Path(__file__))}),indent=2)+'\n')
    print('CPU_RESTORE_COMPLETE',round(result['elapsed_seconds'],1),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path,nargs='+')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--prepared',type=Path,help='Certify the two migrated recovery paths')
    parser.add_argument('--source-proof',type=Path,help='Previously verified original 180k CPU restoration')
    args=parser.parse_args()
    existed=args.output.exists()
    try:
        verify(args.checkpoint,args.output)
        if args.prepared:
            if not args.source_proof:
                raise ValueError('--prepared requires the original --source-proof')
            from gym_trading_env.research.fg_single3_gate_calibration import certify_restore
            certify_restore(args.prepared,args.source_proof,args.output/'analysis.json')
    except BaseException as exc:
        if not existed and args.output.is_dir():
            (args.output/'failure.json').write_text(json.dumps(dict(status='FAILED',error_type=type(exc).__name__,error=str(exc)))+'\n')
        raise
