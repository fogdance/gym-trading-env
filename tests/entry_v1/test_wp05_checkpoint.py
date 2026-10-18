import copy,json,pickle,subprocess,sys,shutil,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from wp05_fixture import engineering_settings
from gym_trading_env.research.fg_s3_e_quality_heads import preflight
from gym_trading_env.research.fg_s3_e_quality_reader import build_join_index
from embodied.run.fg_s3_e_quality_heads import HeadsRun,engineering_permit
from embodied.run.fg_checkpoint import validate_bundle
from dreamerv3.fg_s3a_actor import parameter_digests

@pytest.fixture(scope='module')
def case(tmp_path_factory):
 root=Path(os.environ['FG_WP05_PERSIST_ROOT']) if 'FG_WP05_PERSIST_ROOT' in os.environ else tmp_path_factory.mktemp('wp05_cold')
 root.mkdir(parents=True,exist_ok=True);s=engineering_settings(root,target=64,burnin=32)
 p=preflight(s);index=root/'join.sqlite';build_join_index(p.source,index)
 path=root/'settings.json';path.write_text(json.dumps(s))
 return root,s,path,index

def worker(case,name,mode,stop):
 root,_,path,index=case
 r=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('wp05_cold_worker.py')),str(path),str(root/name),str(index),mode,str(stop)],capture_output=True,text=True)
 out=Path('docs/receipts/wp05')/('E_worker_'+name+'_'+mode+'_'+str(stop)+'_'+__import__('uuid').uuid4().hex)
 out.with_suffix('.log').write_text(r.stdout+r.stderr);out.with_suffix('.json').write_text(json.dumps(dict(argv=r.args,exit_code=r.returncode),indent=2))
 assert r.returncode==0,r.stdout[-1000:]+r.stderr[-4000:]
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

def test_E01_four_vs_two_cold_two_full_exact(case):
 with ThreadPoolExecutor(2) as pool:
  continuous=pool.submit(worker,case,'continuous','start',4)
  split=pool.submit(worker,case,'split','start',2)
  a=continuous.result();split.result()
 b=worker(case,'split','resume',4);exact(a,b)
 root,s,_,_=case
 assert (root/'continuous/updates.jsonl').read_bytes()==(root/'split/updates.jsonl').read_bytes()
 latest=sorted((root/'split/ckpt').glob('step_*'))[-1];m=validate_bundle(latest)
 assert m['step']==b['counters']['consumed_batches'] and b['counters']['optimizer_updates']==4
 assert not any(p.suffix=='.npz' or p.name=='replay' for p in latest.rglob('*'))
 Path('docs/receipts/wp05/E_cold_evidence.json').write_text(json.dumps(dict(root=str(root),status='PASS_EXACT_COLD_CPU',updates=4,split=2,
  all_state_and_next_batch_byte_equal=True,counters=b['counters'],checkpoint=str(latest),checkpoint_files=m['files'],real_TRAIN_updates=0),indent=2))

@pytest.mark.parametrize('attack',['settings','code','index','source_f1','corrupt_latest'])
def test_E02_resume_failures_before_model_or_run_writes(case,tmp_path,monkeypatch,attack):
 root,s,_,index=case
 if not (root/'split/run_identity.json').exists():pytest.fail('E01 must generate checkpoint first')
 clone=tmp_path/'run';shutil.copytree(root/'split',clone)
 p=preflight(s);idx=index
 if attack=='settings':
  from gym_trading_env.research.fg_s3_e_quality_heads import F2Preparation
  changed=copy.deepcopy(p.receipt);changed['identity']['settings']['burnin_steps']+=1
  p=F2Preparation(p.settings,p.source,changed)
 elif attack=='code':monkeypatch.setattr('embodied.run.fg_s3_e_quality_heads.execution_identity',lambda:{'changed':True})
 elif attack=='index':
  idx=tmp_path/'join.sqlite';shutil.copyfile(index,idx);shutil.copyfile(index.with_suffix('.json'),idx.with_suffix('.json'))
  with idx.open('ab') as f:f.write(b'changed')
 elif attack=='source_f1':
  from gym_trading_env.research.fg_s3_e_quality_heads import F2Preparation
  changed=copy.deepcopy(p.receipt);changed['identity']['source_f1']['manifest_sha256']='0'*64
  p=F2Preparation(p.settings,p.source,changed)
 else:
  latest=sorted((clone/'ckpt').glob('step_*'))[-1];(latest/'agent.pkl').write_bytes(b'corrupt')
 before={str(x):x.stat().st_mtime_ns for x in clone.rglob('*') if x.is_file()}
 monkeypatch.setattr('embodied.run.fg_s3_e_quality_heads.create_agent',lambda *a,**k:(_ for _ in ()).throw(AssertionError('constructed model')))
 with pytest.raises(ValueError):HeadsRun(p,idx,clone,resume=True,_probe=engineering_permit(p))
 assert before=={str(x):x.stat().st_mtime_ns for x in clone.rglob('*') if x.is_file()}

def test_E03_skip_checkpoint_preserves_consumed_reader_and_does_not_handoff_again(case,tmp_path,monkeypatch):
 import jax.numpy as jnp
 root,s,_,index=case;p=preflight(s);run=HeadsRun(p,index,tmp_path/'skip',_probe=engineering_permit(p))
 initial=parameter_digests(run.agent.params);original=run.agent.train
 def invalid(carry,data):
  data={k:jnp.zeros_like(v) if k.startswith('quality_') and k.endswith('_valid') else v for k,v in data.items()}
  return original(carry,data)
 monkeypatch.setattr(run.agent,'train',invalid)
 m=run.update();assert m['f2/skip_no_valid'] and int(run.agent.n_updates)==0 and int(run.step)==1
 assert parameter_digests(run.agent.params)==initial
 saved=run.counters.save();reader=pickle.loads((run.cp.latest()/'course_reader.pkl').read_bytes())
 # Stop only this object's prefetch using its normal lifecycle; no host process termination.
 with run.stream.pause():
  def forbidden(*a,**k):raise AssertionError('resume called handoff')
  monkeypatch.setattr('embodied.run.fg_s3_e_quality_heads.transfer_parameters',forbidden)
  other=HeadsRun(p,index,tmp_path/'skip',resume=True,_probe=engineering_permit(p))
  assert other.counters.save()==saved and int(other.step)==1 and int(other.agent.n_updates)==0
  assert other.reader.state_dict()==reader
 Path('docs/receipts/wp05/E_skip_evidence.json').write_text(json.dumps(dict(counters=saved,checkpoint=str(run.cp.latest()),exact_reader_restored=True,optimizer_updates=0),indent=2))

def test_E04_no_authorization_no_run_side_effects(case,tmp_path):
 _,s,_,index=case;p=preflight(s);out=tmp_path/'forbidden'
 with pytest.raises(ValueError,match='AUTHORIZATION_REQUIRED'):HeadsRun(p,index,out)
 assert not out.exists()
