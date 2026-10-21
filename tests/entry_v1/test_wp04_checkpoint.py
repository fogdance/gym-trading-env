import copy,json,pickle,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from wp04_fixture import native_reader,settings
from wp03_fixture import native_corpus
from gym_trading_env.research.fg_s3_e_quality_foundation import preflight,F1Preparation,require_training_authority
from gym_trading_env.research.fg_s3_e_quality_supervision import Admission
from embodied.run.fg_checkpoint import validate_bundle

@pytest.fixture(scope='module')
def case(tmp_path_factory):
 root=tmp_path_factory.mktemp('wp04_cold');corpus=native_corpus(root/'corpus',minutes=100,chunk=120)
 s,r=native_reader(root,corpus=corpus,target=64,burnin=32);r.close()
 path=root/'settings.json';path.write_text(json.dumps(s))
 return root,path,s

def run_worker(case,name,mode,stop):
 root,path,_=case
 result=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('wp04_cold_worker.py')),
   str(path),str(root/name),str(root/'join.sqlite'),mode,str(stop)],capture_output=True,text=True)
 out=Path('docs/receipts/wp04')/('D_worker_'+name+'_'+mode+'_'+str(stop))
 if out.with_suffix('.json').exists():out=out.with_name(out.name+'_'+__import__('uuid').uuid4().hex)
 out.with_suffix('.log').write_text(result.stdout+result.stderr)
 out.with_suffix('.json').write_text(json.dumps(dict(argv=result.args,exit_code=result.returncode),indent=2))
 assert result.returncode==0,result.stdout[-2000:]+result.stderr[-3000:]
 return pickle.loads((root/name/('evidence_'+str(stop)+'.pkl')).read_bytes())

def exact(a,b,path=''):
 if isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for k in a:exact(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'/'+str(i))
 elif isinstance(a,np.ndarray):
  assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),path
 else:assert a==b,path

def test_D01_actual_cold_process_full_state_and_next_batch(case):
 continuous=run_worker(case,'continuous','start',4)
 run_worker(case,'split','start',2)
 restored=run_worker(case,'split','resume',4)
 # Python/NumPy ambient seeds are initialized explicitly by each worker below;
 # compare the entire saved runtime, including pending original JAX seeds.
 exact(continuous,restored)
 root,_,s=case
 a=(root/'continuous/updates.jsonl').read_bytes();b=(root/'split/updates.jsonl').read_bytes()
 assert a==b
 bundle=root/'split/ckpt/step_000000000004';meta=validate_bundle(bundle)
 assert meta['backend']=='external_immutable_course_reader'
 assert not any(p.suffix=='.npz' or p.name=='replay' for p in bundle.rglob('*'))
 report=dict(case_root=str(root),status='PASS_EXACT_COLD_CPU',updates=4,split=2,batch_size=1,target_steps=64,burnin_steps=32,
  all_agent_optimizer_runtime_reader_counter_queue_bytes_equal=True,update_metrics_equal=True,
  corpus_copy_files=0,checkpoint_files=meta['files'],corpus_bytes=sum(p.stat().st_size for p in Path(s['corpus_root']).rglob('*') if p.is_file()),
  checkpoint_bytes=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file()),real_TRAIN_updates=0)
 Path('docs/receipts/wp04/D_cold_restore_evidence.json').write_text(json.dumps(report,indent=2)+'\n')

def test_D02_resume_uses_original_identity_after_code_changes(case,monkeypatch):
 from embodied.run.fg_s3_e_quality_foundation import FoundationRun,synthetic_probe_permit
 root,_,s=case
 if not (root/'split/run_identity.json').exists():pytest.fail('D01 must create the actual checkpoint first')
 monkeypatch.setattr('gym_trading_env.research.fg_s3_e_quality_foundation.execution_identity',lambda:{'changed':True})
 prep=preflight(s)
 before=(root/'split/run_identity.json').read_bytes()
 run=FoundationRun(prep,root/'join.sqlite',root/'split',resume=True,_probe=synthetic_probe_permit(prep))
 assert int(run.step)==int(run.agent.n_updates)==4
 assert (root/'split/run_identity.json').read_bytes()==before
 run.source.close()

def test_D03_forged_native_dataclass_is_not_authority(case):
 _,_,s=case;p=preflight(s)
 receipt=copy.deepcopy(p.receipt);receipt['status']='F1_DATA_READY'
 source=Admission(p.source.source,p.source.receipt,dict(result=dict(G3='GO')))
 fake=F1Preparation(s,source,receipt)
 with pytest.raises(ValueError,match='EXECUTION_PROOF_REQUIRED'):require_training_authority(fake,authorized=True)

def test_D04_external_source_index_and_corrupt_bundle_fail_closed(case,tmp_path):
 import shutil
 from embodied.run.fg_s3_e_quality_foundation import CourseStream,FoundationRun,synthetic_probe_permit
 from gym_trading_env.research.fg_s3_e_quality_reader import EQualityCourseReader
 from embodied.run.fg_checkpoint import FullCheckpoint
 import elements
 root,_,s=case
 p=preflight(s);r=EQualityCourseReader.open(p.source,stage='WM_NATURAL',sampling_spec=s['sampling_spec'],seed=202,index_path=root/'join.sqlite')
 stream=CourseStream(r,s);saved=stream.save();stream.validate_snapshot(saved)
 wrong=copy.deepcopy(saved);wrong['stage']='ACTOR_STRICT'
 with pytest.raises(ValueError,match='EXTERNAL_READER_IDENTITY'):stream.validate_snapshot(wrong)
 r.close()
 # Preserve original evidence; attacks operate only on isolated copies.
 clone=tmp_path/'clone';shutil.copytree(root/'split',clone)
 latest=sorted((clone/'ckpt').glob('step_*'))[-1];(latest/'agent.pkl').write_bytes(b'corrupt')
 cp=FullCheckpoint(clone/'ckpt',elements.Counter(),json.loads((clone/'run_identity.json').read_text())['identity'],(),backend='external_immutable_course_reader')
 with pytest.raises(ValueError,match='corrupt file'):cp.latest()
 index=tmp_path/'changed.sqlite';shutil.copyfile(root/'join.sqlite',index)
 shutil.copyfile(root/'join.json',index.with_suffix('.json'))
 with index.open('ab') as f:f.write(b'changed')
 with pytest.raises(ValueError,match='INDEX_CHANGED'):
  FoundationRun(p,index,root/'split',resume=True,_probe=synthetic_probe_permit(p))
 source=tmp_path/'source';shutil.copytree(Path(s['corpus_root']),source)
 changed={**s,'corpus_root':str(source)}
 pp=preflight(changed)
 # Mutation after positive source validation, not a pre-invalid baseline.
 member=next((source/'student_replay').rglob('*.npz'))
 with member.open('ab') as f:f.write(b'changed')
 with pytest.raises(ValueError):preflight(changed)

def test_D05_initial_unknown_state_and_parameter_slots_fail_closed():
 from types import SimpleNamespace
 from dreamerv3.fg_s3_e_quality_foundation import register_state_schema
 layout=json.loads(Path('docs/receipts/wp04/B_initial_layout.json').read_text())
 params={k:np.zeros(v['shape'],v['dtype']) for k,v in layout.items()}
 model=SimpleNamespace(config=SimpleNamespace(opt=SimpleNamespace(wd=0)))
 register_state_schema(model,params)
 for key in ('alien/kernel','opt/state/1/1/alien/kernel'):
  with pytest.raises(ValueError,match='UNKNOWN'):
   register_state_schema(model,{**params,key:np.zeros(2,np.float32)})
