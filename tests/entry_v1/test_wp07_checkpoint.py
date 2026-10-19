import concurrent.futures
import copy
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wp07_fixture import ARTIFACTS,execution


def compare(a,b,path='root'):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:compare(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'[{i}]')
    elif isinstance(a,np.ndarray):
        assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),path
    elif hasattr(a,'__dict__'):
        assert type(a)==type(b),path
        compare(vars(a),vars(b),path+'.__dict__')
    else:assert a==b,path


@pytest.fixture(scope='module')
def common():
    from wp02_fixture import native_factory
    from data_multi_fixture import build_sources
    root=Path(os.environ.get('FG_WP07_G_ROOT',ARTIFACTS/'G_cold_01')).resolve()
    factory,path,profile=native_factory(root/'common/market',minutes=100,chunk=35)
    history=json.loads(build_sources(root/'common/history').read_text())
    return root,dict(env_config=str(path),profile=dict(identity=profile.identity,
        contracts=sorted(profile.contracts),source_receipt=profile.source_receipt),history=history,
        source=dict(fold='TRAIN',scope='BUILTIN_SYNTHETIC_ONLY'),geometry=dict(batch_size=1,burnin_steps=4,target_steps=8))


def config_for(common,name):
    root,base=common;cfg=copy.deepcopy(base)
    mode='SHADOW' if name.startswith('SHADOW') else 'ONLINE_SUPERVISED' if name=='publication' else name
    s=execution(root/'logical'/name,mode)
    s.update(env_steps=64,collect_every=16,save_every=32,max_consumed_batches=0 if mode=='SHADOW' else 16,
        exploration_rate=.2 if mode!='SHADOW' else 0.)
    cfg['spec']=s;out=root/name;out.mkdir(parents=True,exist_ok=True)
    p=out/'config.json';p.write_text(json.dumps(cfg,indent=2)+'\n')
    return p,out


def worker(config,root,operation,target):
    root.parent.mkdir(parents=True,exist_ok=True)
    argv=[sys.executable,'-B',str(Path(__file__).with_name('wp07_online_worker.py')),str(config),str(root),operation,str(target)]
    log=root.parent/(root.name+'_'+operation+'_'+str(target)+'.log')
    with log.open('w') as stream:result=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT)
    (log.with_suffix('.command.json')).write_text(json.dumps(dict(argv=argv,exit_code=result.returncode),indent=2)+'\n')
    assert result.returncode==0,log.read_text()[-6000:]
    return root


def test_native_shadow_run_smoke(common):
    config,root=config_for(common,'SHADOW_smoke')
    cfg=json.loads(config.read_text());cfg['spec']['execution_mode']='SHADOW'
    config.write_text(json.dumps(cfg))
    worker(config,root/'run','start',16)
    result=json.loads((root/'run/result.json').read_text())
    assert result['env_steps']==16 and result['counters']['optimizer_updates']==0
    assert result['gate']['ready']['COLLECTION_VALID'] and not result['formal_rl_admitted']


def run_case(common,name):
    config,root=config_for(common,name)
    if name=='publication':
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            a=pool.submit(worker,config,root/'continuous','publication_continuous',64)
            worker(config,root/'split','publication_stop',16)
            interrupted=json.loads((root/'split/result.json').read_text())
            assert interrupted['phase']['stage']=='PUBLISH'
            worker(config,root/'split','resume',64);a.result()
    else:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            a=pool.submit(worker,config,root/'continuous','start',64)
            worker(config,root/'split','start',32)
            worker(config,root/'split','resume',64);a.result()
    from embodied.run.fg_checkpoint import rehydrate_native_uuids
    x=rehydrate_native_uuids(pickle.loads((root/'continuous/evidence.pkl').read_bytes()))
    y=rehydrate_native_uuids(pickle.loads((root/'split/evidence.pkl').read_bytes()))
    compare(x,y)
    result=json.loads((root/'split/result.json').read_text())
    if name=='ONLINE_RL_CONTROLLED':assert result['counters']['rl_updates']>0
    if name!='SHADOW':assert result['counters']['optimizer_updates']>0
    return dict(case=name,status='PASS_EXACT_NEW_PROCESS_STATE',result=result,
        evidence_paths=[str(root/'continuous/evidence.pkl'),str(root/'split/evidence.pkl')])


def test_four_native_online_cold_restore_scenarios(common):
    names=['SHADOW','ONLINE_SUPERVISED','publication','ONLINE_RL_CONTROLLED']
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        results=list(pool.map(lambda name:run_case(common,name),names))
    Path('docs/receipts/wp07/'+os.environ.get('FG_WP07_CASE_RECEIPT','G_cold_cases.json')).write_text(json.dumps(results,indent=2)+'\n')
