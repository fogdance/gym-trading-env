import copy,json,pickle,os,shutil,subprocess,sys,uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import jax.numpy as jnp
import numpy as np
import pytest
from wp06_fixture import engineering_settings
from gym_trading_env.research.fg_s3_e_quality_actor import preflight,F3Preparation
from gym_trading_env.research.fg_s3_e_quality_reader import build_join_index
from embodied.run.fg_s3_e_quality_actor import ActorRun,engineering_permit
from embodied.run.fg_checkpoint import validate_bundle
from dreamerv3.fg_s3a_actor import parameter_digests

@pytest.fixture(scope='module')
def case(tmp_path_factory):
    root=Path(os.environ['FG_WP06_PERSIST_ROOT']) if 'FG_WP06_PERSIST_ROOT' in os.environ else tmp_path_factory.mktemp('wp06_cold')
    root.mkdir(parents=True,exist_ok=True);s=engineering_settings(root,target=64,burnin=32)
    s['aux_sampling_spec']=dict(mode='MIXED',natural_weight=.5,strata_weights={'STRICT_ADD':1,'STRICT_WAIT':1})
    s['actor_loss'].update(balanced=.2,strict_pair=.1)
    p=preflight(s);index=root/'join.sqlite';build_join_index(p.source,index)
    path=root/'settings.json';path.write_text(json.dumps(s))
    return root,s,path,index

def worker(case,name,mode,stop,scenario):
    root,_,path,index=case
    r=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('wp06_cold_worker.py')),str(path),str(root/name),str(index),mode,str(stop),scenario],capture_output=True,text=True)
    receipt=Path('docs/receipts/wp06')/('F_worker_'+name+'_'+mode+'_'+str(stop)+'_'+uuid.uuid4().hex)
    receipt.with_suffix('.log').write_text(r.stdout+r.stderr)
    receipt.with_suffix('.json').write_text(json.dumps(dict(argv=r.args,exit_code=r.returncode),indent=2))
    assert r.returncode==0,r.stdout[-1000:]+r.stderr[-5000:]
    return pickle.loads((root/name/('evidence_'+str(stop)+'.pkl')).read_bytes())

def exact(a,b,path=''):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:exact(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'/'+str(i))
    elif isinstance(a,np.ndarray):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),path
    else:assert a==b,path

@pytest.mark.parametrize('scenario',['ordinary','skip_first'])
def test_F01_exact_cold_all_state_readers_rng_prefetch_after_skip(case,scenario):
    a_name=scenario+'_continuous';b_name=scenario+'_split';first=2 if scenario=='ordinary' else -1
    with ThreadPoolExecutor(2) as pool:
        one=pool.submit(worker,case,a_name,'start',4,scenario)
        two=pool.submit(worker,case,b_name,'start',first,scenario)
        a=one.result();split=two.result()
    if scenario=='skip_first':assert split['counters']['optimizer_updates']==0 and split['counters']['consumed_batches']==1
    b=worker(case,b_name,'resume',4,scenario);exact(a,b)
    root,_,_,_=case
    assert (root/a_name/'updates.jsonl').read_bytes()==(root/b_name/'updates.jsonl').read_bytes()
    latest=sorted((root/b_name/'ckpt').glob('step_*'))[-1];m=validate_bundle(latest)
    assert b['counters']['optimizer_updates']==4 and m['step']==b['counters']['consumed_batches']
    assert len(b['reader']['readers'])==2 and not any(p.suffix=='.npz' or p.name=='replay' for p in latest.rglob('*'))
    Path('docs/receipts/wp06/F_cold_'+scenario+'.json').write_text(json.dumps(dict(status='PASS_EXACT_COLD_CPU',root=str(root),
        scenario=scenario,checkpoint=str(latest),counters=b['counters'],checkpoint_files=m['files'],
        all_state_and_next_batch_byte_equal=True,real_TRAIN_updates=0),indent=2))

@pytest.mark.parametrize('attack',['index','corrupt_latest'])
def test_F02_resume_rejects_before_model_or_run_writes(case,tmp_path,monkeypatch,attack):
    root,s,_,index=case;clone=tmp_path/'run';shutil.copytree(root/'ordinary_split',clone)
    p=preflight(s);idx=index
    if attack=='index':
        idx=tmp_path/'index.sqlite';shutil.copyfile(index,idx);shutil.copyfile(index.with_suffix('.json'),idx.with_suffix('.json'))
        with idx.open('ab') as f:f.write(b'changed')
    elif attack=='corrupt_latest':
        latest=sorted((clone/'ckpt').glob('step_*'))[-1];(latest/'agent.pkl').write_bytes(b'corrupt')
    before={str(x):x.read_bytes() for x in clone.rglob('*') if x.is_file()}
    def forbidden(*a,**k):raise AssertionError('bad resume constructed a model')
    monkeypatch.setattr('gym_trading_env.research.fg_s3_e_quality_actor.create_agent',forbidden)
    with pytest.raises(ValueError):ActorRun(p,idx,clone,resume=True,_probe=engineering_permit(p))
    assert before=={str(x):x.read_bytes() for x in clone.rglob('*') if x.is_file()}

def test_F02_code_change_restores_original_owner_and_trained_actor(case,tmp_path,monkeypatch):
    root,s,_,index=case;clone=tmp_path/'run';shutil.copytree(root/'ordinary_split',clone)
    before=(clone/'run_identity.json').read_bytes()
    monkeypatch.setattr('gym_trading_env.research.fg_s3_e_quality_actor.execution_identity',lambda:{'changed':True})
    p=preflight(s);run=ActorRun(p,index,clone,resume=True,_probe=engineering_permit(p))
    assert int(run.agent.n_updates)==run.counters.state['optimizer_updates']==4
    saved=pickle.loads((run.cp.latest()/'agent.pkl').read_bytes())
    assert parameter_digests(run.agent.params)==parameter_digests(saved['params'])
    assert (clone/'run_identity.json').read_bytes()==before
    run.source.close()

def test_F03_nonfinite_rejection_consumption_is_auditable_and_cannot_retry_forever(case,tmp_path):
    _,s,_,index=case;p=preflight(s);run=ActorRun(p,index,tmp_path/'nonfinite',_probe=engineering_permit(p))
    for key in run.agent.params:
        if key.startswith('quality_actor/'):run.agent.params[key]=jnp.full_like(run.agent.params[key],1e38)
    before=parameter_digests(run.agent.params)
    with pytest.raises(FloatingPointError,match='F3_NONFINITE'):run.update()
    assert before==parameter_digests(run.agent.params) and int(run.agent.n_updates)==0
    assert run.counters.state['consumed_batches']==run.counters.state['numerical_rejections']==1
    assert validate_bundle(run.cp.latest())['step']==1
    with pytest.raises(ValueError,match='REQUIRES_REVIEW'):run.update()

def test_F04_f1_checkpoint_is_not_f3_resume(case,tmp_path):
    _,s,_,index=case;p=preflight(s);clone=tmp_path/'f1';shutil.copytree(s['source_run'],clone)
    before={str(x):x.read_bytes() for x in clone.rglob('*') if x.is_file()}
    with pytest.raises(ValueError,match='RUN_STAGE_OR_BACKEND_MISMATCH'):ActorRun(p,index,clone,resume=True,_probe=engineering_permit(p))
    assert before=={str(x):x.read_bytes() for x in clone.rglob('*') if x.is_file()}
