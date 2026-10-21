"""Explicit RL warmup stops without silently exhausting the budget as BC."""
import copy
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

from wp07_fixture import execution
from test_wp07_checkpoint import compare


def test_native_warmup_blocks_safely_and_new_process_restores_exactly(tmp_path):
    from wp02_fixture import native_factory
    from data_multi_fixture import build_sources
    root=tmp_path/'rl_pause';root.mkdir()
    factory,path,profile=native_factory(root/'market',minutes=100,chunk=35)
    history=json.loads(build_sources(root/'history').read_text())
    spec=execution(root/'logical','ONLINE_RL_CONTROLLED')
    spec.update(env_steps=64,updates_per_collect=4,max_consumed_batches=32)
    config=dict(spec=spec,env_config=str(path),profile=dict(identity=profile.identity,
        contracts=sorted(profile.contracts),source_receipt=profile.source_receipt),history=history,
        source=dict(fold='TRAIN',scope='BUILTIN_SYNTHETIC_ONLY'),geometry=dict(batch_size=1,burnin_steps=4,target_steps=8),
        numeric_probe=False,rl_policy=dict(warmup_env_steps=32,calibration_batches=1,
            calibration_horizon=3,calibration_seed=202),wm_policy=dict(max_reward_mae=.01,max_continue_mae=.01))
    config_path=root/'config.json';config_path.write_text(json.dumps(config))
    records=[]
    def worker(name,operation,target):
        argv=[sys.executable,'-B',str(Path(__file__).with_name('wp07_online_worker.py')),
            str(config_path),str(root/name),operation,str(target)]
        log=root/(name+'_'+operation+'_'+str(target)+'.log')
        with log.open('x') as stream:r=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT)
        records.append(dict(argv=argv,exit_code=r.returncode,log=str(log)))
        assert r.returncode==0,log.read_text()[-6000:]
    worker('continuous','pause_continuous',64)
    worker('split','start',16);worker('split','resume',64)
    result=json.loads((root/'split/result.json').read_text())
    assert result['status']=='RL_BLOCKED' and result['env_steps']==32
    assert result['counters']['rl_updates']==0 and result['counters']['bc_updates']>0
    assert result['rl_lifecycle']['blocked_count']==1
    assert 'INSUFFICIENT' in result['rl_lifecycle']['reason']
    from embodied.run.fg_checkpoint import rehydrate_native_uuids
    a=rehydrate_native_uuids(pickle.loads((root/'continuous/evidence.pkl').read_bytes()))
    b=rehydrate_native_uuids(pickle.loads((root/'split/evidence.pkl').read_bytes()))
    compare(a,b)
    equivalence=list((root/'split/rl_admission').glob('attempt_*/calibration_live_equivalence.json'))
    assert equivalence and json.loads(equivalence[0].read_text())['equal']
    evidence=dict(scope='NATIVE_SYNTHETIC_RL_PAUSE_NOT_FORMAL',commands=records,result=result,
        exact_full_state=True,calibration_equivalence=[str(p) for p in equivalence])
    Path('docs/receipts/wp07/BATCH3_RL_PAUSE.json').write_text(json.dumps(evidence,indent=2)+'\n')
