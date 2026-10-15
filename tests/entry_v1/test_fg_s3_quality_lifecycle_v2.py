"""Shared CLI, real native Driver/Replay and complete CPU save/resume."""
from pathlib import Path
import json
import os
import subprocess
import sys
import uuid
import pickle

import numpy as np
import pandas as pd
import pytest
import yaml

from test_fg_s3_quality_teacher_v2 import quality_env
from gym_trading_env.research.fg_replay import read_manifest,signed,write_json

PROJECT=Path(__file__).resolve().parents[2]


def test_shared_cpu_prepare_start_resume_full_checkpoint():
    tag='QUALITY_CPU_ACCEPTANCE_'+uuid.uuid4().hex[:10]
    output=PROJECT/'artifacts/fg_s3_v2/quality_checks'/tag
    output.mkdir(parents=True)
    fixture=output/'fixture';fixture.mkdir()
    times=pd.date_range('2026-01-05 09:01',periods=75,freq='min').append([
        pd.date_range('2026-01-05 10:31',periods=60,freq='min'),
        pd.date_range('2026-01-05 13:31',periods=90,freq='min')])
    prices=([(1000,1000,1000,1000),(1000,1000,1000,1000),(1000,1007,1000,1007),
            (1000,1000,1000,1000)]*60)[:len(times)]
    env,_=quality_env(fixture,prices,times=times,
        calendar=[dict(start='2026-01-05 '+a,end='2026-01-05 '+b)
                  for a,b in (('09:00','10:15'),('10:30','11:30'),('13:30','15:00'))],liquidation='2026-01-05 15:00')
    market=fixture/'market.csv';env._raw.to_csv(market,index_label='Date');env.close()
    fixture_config=yaml.safe_load((fixture/'env.yaml').read_text())
    fixture_config['trading']['data_path']=str(market)
    (fixture/'env.yaml').write_text(yaml.safe_dump(fixture_config))
    cfg=yaml.safe_load((PROJECT/'configs/experiments/fg_s3_v2/QUALITY_V2_CPU_SMOKE.yaml').read_text())
    from test_fg_s3_quality_admission_v2 import fixture_policy
    policy=fixture_policy()
    policy['trained_contracts']=[fixture_config['entry']['contract']]
    policy_path=output/'synthetic_admission_policy.json'
    write_json(policy_path,signed({k:v for k,v in policy.items() if k!='content_hash'}))
    cfg.update(experiment=tag,quality_fixture_env=str(fixture/'env.yaml'),env_steps=200,
        quality_admission_policy=str(policy_path),
        prepared_dir='artifacts/fg_s3_v2/prepared/'+tag,
        run_dir='artifacts/fg_s3_v2/runs/'+tag,replay_capacity=256)
    path=output/'experiment.yaml';path.write_text(yaml.safe_dump(cfg))
    commands=[]
    def run(name,args):
        cmd=[sys.executable,'-B',str(PROJECT/'tools/fg_s3_v2_experiment.py'),*args]
        completed=subprocess.run(cmd,cwd=PROJECT,env={**os.environ,'JAX_PLATFORMS':'cpu',
            'PYTHONPATH':str(PROJECT/'src')+':'+str(PROJECT.parent/'dreamerv3'),
            'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'},
            text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (output/(name+'.log')).write_text(completed.stdout)
        commands.append(dict(command=cmd,exit_code=completed.returncode,log=str(output/(name+'.log'))))
        assert completed.returncode==0,completed.stdout[-12000:]
    run('prepare',['prepare','--config',str(path)])
    prepared=PROJECT/cfg['prepared_dir'];root=PROJECT/cfg['run_dir']
    run('start100',['start','--prepared',str(prepared),'--stop-at','100'])
    run('resume200',['resume','--prepared',str(prepared)])
    receipt=read_manifest(root/'training_receipt.json')
    assert receipt['env_steps']==200 and receipt['model_updates']==4
    assert receipt['counters']['wm_updates']==0 and not receipt['quality']['actor_influence']
    assert receipt['quality']['reference_actor_frozen'] and receipt['quality']['critic_rl_frozen']
    assert receipt['quality']['coverage']['counts']['score_valid']>0
    from gym_trading_env.research.fg_s3_quality_catalog_v2 import validate_catalog
    validated=validate_catalog(receipt['quality']['catalog'])
    assert validated['status']=='PASS_AUTHENTICATED_TRAIN_QUALITY_CATALOG'
    prefix=validate_catalog(root/'quality_catalog_step_000000000100.json')
    assert prefix['record_count']==100 and prefix['current_journal_extra_bytes']>0
    catalog=read_manifest(receipt['quality']['catalog'])
    corrupt={k:v for k,v in catalog.items() if k!='content_hash'}
    corrupt['identity']={**corrupt['identity'],'journal_prefix_sha256':'0'*64}
    tampered=output/'tampered_catalog_rejected.json';write_json(tampered,signed(corrupt))
    with pytest.raises(ValueError,match='journal bytes changed'):
        validate_catalog(tampered)
    node=Path(receipt['checkpoint'])
    from gym_trading_env.research.fg_s3_quality_admission import checkpoint_policy,_head_updates
    from embodied.run.fg_checkpoint import file_hash
    frozen,policy_hash,meta=checkpoint_policy(node)
    assert frozen==read_manifest(policy_path) and policy_hash==file_hash(policy_path)
    assert meta['identity']['immutable_files'][str(root/'quality_admission_policy.json')]==policy_hash
    _head_updates(node,meta['identity'])
    from gym_trading_env.research.fg_s3_quality_prediction_evaluation import frozen_q_budget
    assert frozen_q_budget(node,meta['identity'])=={
        name:cfg['quality_q_'+name] for name in ('max_states','max_seconds','max_suffix_bars')}
    assert all((node/p).is_file() for p in ('agent.pkl','collector.pkl','driver.pkl','runtime.pkl','replay.pkl','done'))
    updates=[json.loads(line) for line in (root/'updates.jsonl').read_text().splitlines()]
    assert all(r['metrics']['opt/quality_actor_delta_norm']==0 and r['metrics']['opt/wm_delta_norm']==0 for r in updates)
    assert any(r['metrics']['opt/quality_event_delta_norm']>0 for r in updates)
    assert all(r['metrics']['opt/actor_delta_norm']==0 and r['metrics']['opt/critic_delta_norm']==0 for r in updates)
    uninterrupted=root.with_name(tag+'_uninterrupted')
    run('uninterrupted200',['start','--prepared',str(prepared),'--run-dir',str(uninterrupted)])
    unbroken=read_manifest(uninterrupted/'training_receipt.json')
    from dreamerv3.fg_s3a_actor import parameter_digests
    resumed_state=pickle.loads((node/'agent.pkl').read_bytes())
    unbroken_state=pickle.loads((Path(unbroken['checkpoint'])/'agent.pkl').read_bytes())
    assert parameter_digests(resumed_state['params'])==parameter_digests(unbroken_state['params'])
    unbroken_updates=[json.loads(s) for s in (uninterrupted/'updates.jsonl').read_text().splitlines()]
    assert [r['metrics'] for r in updates]==[r['metrics'] for r in unbroken_updates]
    evidence=signed(dict(status='PASS_SHARED_CPU_PREPARE_START_RESUME',commands=commands,
        prepared=str(prepared),run_dir=str(root),checkpoint=str(node),env_steps=200,updates=4,
        coverage=receipt['quality']['coverage'],gpu='NOT_RUN',supervision='SYNTHETIC_NATIVE_ENV',
        initial_checkpoint_owns_immutable_admission_policy=True,admission_policy_sha256=policy_hash,
        posthoc_q_uses_frozen_training_proof_budget=True,
        unbroken_vs_resumed_gradient='PASS; all reported gradients/losses and final parameter digests equal'))
    write_json(output/'receipt.json',evidence)
    print('\nQUALITY_LIFECYCLE_RECEIPT='+str(output/'receipt.json'))
