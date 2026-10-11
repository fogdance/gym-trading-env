#!/usr/bin/env python3
"""Full frozen neural architecture on a small real TRAIN batch, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','24')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
import copy
from pathlib import Path
import pickle
import resource
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def smoke(output):
    from gym_trading_env.research.fg_single3_experiment import resolved_config,DREAMER_ROOT
    from gym_trading_env.research.fg_replay import check_files,digest,signed,write_json
    sys.path.insert(0,str(DREAMER_ROOT))
    import embodied
    import jax
    import numpy as np
    from dreamerv3.main import make_agent,make_stream
    from embodied.jax import internal
    from embodied.run.fg_single3_audit import TrainingAudit
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    config=resolved_config().update({'logdir':str(output),'jax.platform':'cpu',
        'jax.precompile':False,'jax.profiler':False})
    neural_sources={str(DREAMER_ROOT/name):digest(DREAMER_ROOT/name) for name in (
        'dreamerv3/agent.py','dreamerv3/rssm.py','dreamerv3/main.py','dreamerv3/configuration.py',
        'dreamerv3/configs.yaml','embodied/jax/agent.py','embodied/jax/opt.py',
        'embodied/run/fg_single3_audit.py')}
    source=sorted(Path(config.run.replay_dir).glob('*.npz'))[0]
    with np.load(source,allow_pickle=False) as arrays:
        # Real contiguous windows from one native chunk; no fabricated fields.
        offsets=np.random.default_rng(202).choice(len(arrays['reward'])-64,16,replace=False)
        batch={k:np.stack([arrays[k][i:i+65] for i in offsets]) for k in arrays.files}
    batch=embodied.Replay._annotate_batch(None,batch,None,True)
    assert 'dyn/deter' not in batch and 'dyn/stoch' not in batch
    class SampleOnly:
        def sample(self,n,mode):
            assert n==16
            return copy.deepcopy(batch)
    stream=iter(make_stream(config,SampleOnly(),'train'))
    agent=make_agent(config)
    assert all(device.platform=='cpu' for device in jax.devices())
    assert set(agent.spaces)==set(batch)|{'consec'}
    audit=TrainingAudit(output,1024)
    carry=agent.init_train(16)
    observed=[]
    for update in (1,2):
        data=next(stream)
        assert data['reward'].shape==(16,65)
        data=internal.device_put(data,agent.train_sharded)
        data['seed']=agent._seeds(update-1,agent.train_mirrored)
        carry,outs,metrics=agent.train(carry,data)
        audit.observe(update,metrics)
        assert not outs  # Original Replay is never rewritten with latent caches.
        observed.append({k:float(v) for k,v in metrics.items() if np.asarray(v).size==1 and np.isfinite(v)})
        print('REAL_NEURAL_CPU_UPDATE',update,'loss',float(metrics['opt/loss']),
            'wm_changed',bool(metrics['opt/wm_changed']),'actor_changed',bool(metrics['opt/actor_changed']),flush=True)
    assert audit.state['wm_effective_updates']>0
    assert audit.state['actor_effective_updates']==0 and audit.state['actor_changes_gate_closed']==0
    data=internal.device_put(next(stream),agent.train_sharded)
    data['seed']=agent._seeds(2,agent.train_mirrored)
    report_carry,report=agent.report(agent.init_report(16),data)
    if any(not np.isfinite(np.asarray(v)).all() for k,v in report.items() if k.startswith('loss/')):
        raise FloatingPointError('Non-finite neural report')
    # Exact neural state round-trip; this is a smoke artifact, not a production checkpoint.
    state=agent.save_exact()
    saved=output/'neural_state.pkl'
    saved.write_bytes(pickle.dumps(state,protocol=5))
    agent.load_exact(pickle.loads(saved.read_bytes()))
    assert int(agent.n_updates)==2
    agent.sync_policy_from_train(reset_counter=True)
    for key in agent.policy_keys:
        np.testing.assert_array_equal(np.asarray(agent.policy_params[key]),np.asarray(agent.params[key]))
    assert not agent.pending_sync
    observation={k:batch[k][:1,0] for k in agent.obs_space}
    _,actions,diagnostics=agent.policy(agent.init_policy(1),observation,mode='eval')
    assert np.isfinite(diagnostics['p_add']).all()
    assert actions['action'].shape==(1,) and int(actions['action'][0]) in (0,1)
    check_files(neural_sources)
    parameter_shapes={key:list(value.shape) for key,value in agent.params.items() if key.startswith(('enc/','dyn/','dec/','pol/','val/','rew/'))}
    result=signed(dict(status='REAL_NEURAL_CPU_SMOKE_PASS',formal_training_status='NOT_RUN',
        formal_gradient_updates=0,smoke_gradient_updates=2,gpu_status='NOT_RUN',
        source_chunk=str(source),source_sha256=digest(source),batch_shape=[16,65],train_positions=2048,
        source_offsets=offsets.tolist(),context_mode='recompute_prefix',stop_gradient_context=True,
        architecture=parameter_shapes,audit=audit.save(),metrics=observed,
        exact_neural_restore=True,evaluation_policy_sync=True,report_executed=True,
        neural_policy_executed=True,policy_action=int(actions['action'][0]),
        policy_p_add=float(diagnostics['p_add'][0]),neural_source_sha256=neural_sources,
        elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        files={str(p.relative_to(output)):digest(p) for p in output.iterdir() if p.is_file()}))
    write_json(output/'manifest.json',result)
    print('CPU_NEURAL_SMOKE_PASS',result['elapsed_seconds'],'seconds',flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    smoke(args.output)
