import concurrent.futures
import json
import os
import pickle
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from wp08_fixture import ARTIFACTS,native_source,controlled_actor
from test_wp08_contract import plan


def worker(base,root,operation,target,seed):
    source=ARTIFACTS/'C_native_04/source'
    argv=[sys.executable,'-B',str(Path(__file__).with_name('wp08_exam_worker.py')),
          str(base),str(root),operation,str(target),str(source)]
    env={**os.environ,'PYTHONHASHSEED':str(seed)}
    log=base/(root.name+'_'+operation+'_'+str(target)+'.log')
    with log.open('x') as stream:r=subprocess.run(argv,env=env,stdout=stream,stderr=subprocess.STDOUT)
    log.with_suffix('.command.json').write_text(json.dumps(dict(argv=argv,exit_code=r.returncode,hashseed=seed),indent=2)+'\n')
    assert r.returncode==0,log.read_text()[-6000:]


def test_1024_equals_333_new_process_691_exactly():
    from wp07_fixture import exact
    base=Path(os.environ.get('FG_WP08_G_ROOT',ARTIFACTS/'G_cold_01')).absolute();base.mkdir(parents=True)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        continuous=pool.submit(worker,base,base/'continuous','start',1024,11)
        worker(base,base/'split','start',333,11)
        worker(base,base/'split','resume',1024,22)
        continuous.result()
    a=pickle.loads((base/'continuous/evidence.pkl').read_bytes())
    b=pickle.loads((base/'split/evidence.pkl').read_bytes())
    exact(a,b)
    assert a['receipt']['reconciliation']['status']=='PASS'
    assert a['state']['valid_minutes']==1024
    Path('docs/receipts/wp08/G_exact_cold.json').write_text(json.dumps(dict(
        status='PASS_EXACT_NEW_PROCESS_1024_EQ_333_PLUS_691',evidence_paths=[str(base/'continuous/evidence.pkl'),str(base/'split/evidence.pkl')],
        receipt=a['receipt'],metrics=a['metrics']['summary'],same_stepids=True,same_account=True,
        same_rssm_and_inference_rng=True,same_model_and_optimizer_bytes=True,same_metrics=True),indent=2)+'\n')


def test_strict_latest_corrupt_chunk_and_source_change_refuse(tmp_path):
    from gym_trading_env.research.fg_s3_e_quality_exam_session import ExamSession
    from gym_trading_env.research.fg_replay import read_manifest
    source=native_source(ARTIFACTS/'C_native_04/source')
    p=plan(tmp_path);p['chunk_steps']=4
    actor=controlled_actor(tmp_path/'model')
    root=tmp_path/'session';s=ExamSession(p,source,actor,root,policy_id='F3_FIXED_SYNTHETIC')
    try:
        s.commit();s.advance(8);s.commit()
    finally:s.close()
    latest=root/'states/latest.json';saved=latest.read_bytes()
    latest.write_text('{}')
    t=ExamSession(p,source,actor,root,policy_id='F3_FIXED_SYNTHETIC',existing=True)
    try:
        with pytest.raises(ValueError):t.restore()
        latest.write_bytes(saved)
        node=read_manifest(root/'states'/read_manifest(latest)['node'])
        state=pickle.loads((root/'states'/node['payload']).read_bytes())
        chunk=root/state['chunks'][0]['path'];original=chunk.read_bytes();chunk.write_bytes(b'CORRUPT')
        with pytest.raises(ValueError,match='CHUNK_CHANGED'):t.restore()
        chunk.write_bytes(original)
        old=source.receipt['frozen_inputs'].copy();source.receipt['frozen_inputs']['/tmp/ABSENT_WP08_SOURCE']='0'*64
        with pytest.raises((ValueError,FileNotFoundError)):t.restore()
        source.receipt['frozen_inputs']=old
        # A crash-created tail is preserved, never silently discarded.
        orphan=root/'chunks/uncommitted_tail';orphan.write_text('keep')
        t.restore();assert list((root/'recovery_chunks').rglob('uncommitted_tail'))
    finally:t.close()
