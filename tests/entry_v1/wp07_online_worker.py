"""Native bounded synthetic execution in a NEW interpreter; no real TRAIN."""
import copy
import json
import pickle
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

random.seed(202);np.random.seed(202)
from embodied.run.fg_checkpoint import host
from embodied.run.fg_s3_e_quality_online_run import OnlineRun,engineering_permit
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv

config=json.loads(Path(sys.argv[1]).read_text());root=Path(sys.argv[2]);operation=sys.argv[3];target=int(sys.argv[4])
spec=config['spec'];profile=SimpleNamespace(**config['profile']);profile.contracts=set(profile.contracts)
factory=lambda:FixedDirectionEntryEnv(config_path=config['env_config'],e_profile=profile)
history=None
if spec['execution_mode']!='SHADOW':
    from data_multi_fixture import descriptors
    from gym_trading_env.research.fg_s3_e_quality_multi_actor import MultiCorpusF3Stream
    cfg=config['history'];names=[row['corpus_id'] for row in cfg['sources']]
    history=MultiCorpusF3Stream.open(descriptors(cfg,'ACTOR_STRICT'),settings=config['geometry'],
        per_source_sampling_spec=cfg['per_source_sampling_spec'],source_mix_weights=cfg['source_mix_weights'],seed=202,
        aux_per_source_sampling_spec={name:dict(mode='MIXED',natural_weight=.5,strata_weights=dict(STRICT_ADD=1,STRICT_WAIT=1)) for name in names})
permit=engineering_permit(spec,numeric_rl=config.get('numeric_probe',spec['execution_mode']=='ONLINE_RL_CONTROLLED'))
run=OnlineRun(spec,factory,config['source'],history,resume=operation=='resume',_probe=permit,_test_location=root,
    rl_policy=config.get('rl_policy'),wm_policy=config.get('wm_policy'))
if operation=='pause_continuous':
    run.run_to(16)
    result=run.run_to(target)
elif operation=='publication_continuous':
    run.run_to(16,_stop_after_publications=7)
    result=run.run_to(target)
elif operation=='publication_stop':result=run.run_to(16,_stop_after_publications=7)
else:result=run.run_to(target)
from gym_trading_env.research.fg_replay import reconcile_trace,E_QUALITY_SCHEMA
from gym_trading_env.research.fg_single3_evaluation import embodied_runtime
trace=root/('native_trace_'+str(int(run.step)))
run.collector.snapshot_trace(trace,dict(scope='SYNTHETIC_NATIVE_ACCOUNT_ONLY'))
replay_proof=reconcile_trace(trace,Path(__file__).resolve().parents[3]/'dreamerv3',
    env_factory=factory,schema=E_QUALITY_SCHEMA,runtime_factory=embodied_runtime)
pause=run.stream.pause() if run.stream is not None else __import__('contextlib').nullcontext()
with pause:
    from embodied.run.fg_checkpoint import save_replay
    replay_state=save_replay(run.collector.replay,root/('evidence_replay_'+str(target)+'_'+str(int(run.transaction))))
    evidence=dict(agent=run.agent.save_exact(),driver=run.collector.driver.save(),collector=run.collector.save(),
        publisher=None if run.publisher is None else run.publisher.save(),
        source=None if run.source is None else run.source.save(),runtime=run.runtime.save(),
        counters=run.counters.save(),phase=run.phase.save(),gates=run.gates.save(),source_exposures=run.source_exposures.save(),
        account_verification=run.account_verification.save(),
        rl_lifecycle=run.rl_lifecycle.save(),
        replay=replay_state,replay_arrays={str(chunk.uuid):{k:v[:chunk.length].copy() for k,v in chunk.data.items()}
            for chunk in run.collector.replay.chunks.values() if chunk.data is not None},native_replay_proof=replay_proof,
        transaction=int(run.transaction),market_step=int(run.step),
        prefetch=None if run.stream is None else run.stream.save_full(),
        updates=(root/'updates.jsonl').read_bytes() if (root/'updates.jsonl').exists() else b'',
        cycles=(root/'cycles.jsonl').read_bytes() if (root/'cycles.jsonl').exists() else b'')
    (root/'evidence.pkl').write_bytes(pickle.dumps(host(evidence),protocol=5))
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
run.close()
