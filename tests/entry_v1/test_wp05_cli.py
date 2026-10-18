import copy,json,subprocess,sys
from pathlib import Path
import pytest,yaml
from wp05_fixture import engineering_settings
from gym_trading_env.research.fg_s3_e_quality_foundation import PROJECT
from gym_trading_env.research.fg_s3_e_quality_heads import preflight,prepare,require_training_authority,F2Preparation
from gym_trading_env.research.fg_s3_e_quality_supervision import Admission

@pytest.mark.parametrize('command',['f2-start','f2-resume'])
def test_F01_no_authorization_exit_one_no_output(tmp_path,command):
 out=tmp_path/'run'
 r=subprocess.run([sys.executable,'-B',str(PROJECT/'tools/fg_s3_v2_experiment.py'),command,'--prepared',str(tmp_path/'absent'),'--run-dir',str(out)],capture_output=True,text=True)
 assert r.returncode==1 and 'F2_EXTERNAL_TRAINING_AUTHORIZATION_REQUIRED' in r.stderr and not out.exists()

def test_F02_engineering_prepare_no_formal_side_effects(tmp_path):
 s=engineering_settings(tmp_path);config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(s))
 receipt=prepare(config);assert receipt['status']=='F2_ENGINEERING_READY' and not receipt['training_admitted']
 out=Path(s['prepared_dir']);assert {p.name for p in out.iterdir()}=={'settings.json','reader_binding.json','f2_preparation.json','join.sqlite','join.json'}
 r=subprocess.run([sys.executable,'-B',str(PROJECT/'tools/fg_s3_v2_experiment.py'),'f2-start','--prepared',str(out),'--authorize-training'],capture_output=True,text=True)
 assert r.returncode==1 and 'FOUNDATION_NOT_TRAINED' in r.stderr and not Path(s['run_dir']).exists()
 with pytest.raises(ValueError,match='OUTPUT_MUST_BE_NEW'):prepare(config)

def test_F03_loaded_or_forged_native_json_not_authority(tmp_path):
 p=preflight(engineering_settings(tmp_path));r=copy.deepcopy(p.receipt)
 r['status']='F2_DATA_READY';r['identity']['source_f1']['engineering_only']=False
 fake=F2Preparation(p.settings,Admission(p.source.source,p.source.receipt,{'result':{'G3':'GO'}}),r)
 with pytest.raises(ValueError,match='ACTUAL_NATIVE_EXECUTION_PROOF_REQUIRED'):require_training_authority(fake,authorized=True)

def test_F04_read_only_route_does_not_consume_an_unrelated_F1(tmp_path):
 from gym_trading_env.research.fg_s3_e_quality_heads_checks import read_check
 s=engineering_settings(tmp_path);config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(s))
 output=PROJECT/'artifacts/fg_s3_v2/e_quality_wp05'/('unit_read_'+__import__('uuid').uuid4().hex)
 r=read_check(config,output)
 assert r['source_f1_consumed'] is False and r['foundation_status']=='FOUNDATION_NOT_TRAINED'
 assert r['real_TRAIN_updates']==0 and r['before']==r['after']
 assert all(c['valid']>0 for c in r['coverage']['counts'].values())
 assert (output/'read_check.json').exists() and not (output/'run').exists() and not (output/'ckpt').exists()


def test_F05_real_read_smoke_cannot_prepare_with_synthetic_F1():
 from gym_trading_env.research.fg_s3_e_quality_heads import read_settings
 s=read_settings(PROJECT/'configs/experiments/fg_s3_v2/e_quality_f2_read_smoke.yaml')
 with pytest.raises(ValueError,match='F1_CORPUS_COURSE_MISMATCH'):preflight(s)
 assert not Path(s['run_dir']).exists() and not Path(s['prepared_dir']).exists()

def test_F06_formal_record_checker_rejects_missing_or_fake_updates(tmp_path):
 import shutil,pickle
 from embodied.run.fg_s3_e_quality_transfer import inspect_f1,training_evidence
 s=engineering_settings(tmp_path);proof,data=inspect_f1(s,allow_engineering=True)
 # Only the record-check function is exercised with a synthetic source;
 # this positive is never a FORMAL admission or a real trained F1 claim.
 evidence=training_evidence(Path(s['source_f1_run']),proof['identity'],data)
 assert evidence['verified_updates']==4 and any(evidence['changed_wm_keys'].values())
 root=tmp_path/'source';shutil.copytree(s['source_f1_run'],root)
 (root/'updates.jsonl').write_text('')
 with pytest.raises(ValueError,match='TRAINING_RECORDS_INCOMPLETE'):training_evidence(root,proof['identity'],data)
 (root/'updates.jsonl').write_bytes((Path(s['source_f1_run'])/'updates.jsonl').read_bytes())
 fresh=pickle.loads((root/'ckpt/step_000000000000/agent.pkl').read_bytes())
 with pytest.raises(ValueError,match='PARAMETERS_NOT_TRAINED'):training_evidence(root,proof['identity'],fresh)

 bad=copy.deepcopy(data);bad['params']['alien/kernel']=__import__('numpy').zeros(2)
 with pytest.raises(ValueError,match='FULL_STATE_SCHEMA'):training_evidence(root,proof['identity'],bad)

 rows=[json.loads(x) for x in (root/'updates.jsonl').read_text().splitlines()]
 rows[0]['metrics']['opt/loss']=float('nan')
 (root/'updates.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
 with pytest.raises(ValueError,match='TRAINING_RECORDS_INVALID'):training_evidence(root,proof['identity'],data)

def test_F07_burnin_only_support_creates_no_ready_receipt_or_model(tmp_path,monkeypatch):
 from gym_trading_env.research.fg_s3_e_quality_reader import build_join_index
 from embodied.run.fg_s3_e_quality_heads import HeadsRun,engineering_permit
 s=engineering_settings(tmp_path,target=8,burnin=90);s['sampling_spec']={'mode':'NATURAL','policies':['P1']}
 config=tmp_path/'bad_support.yaml';config.write_text(yaml.safe_dump(s))
 with pytest.raises(ValueError,match='F2_TARGET_LOSS'):prepare(config)
 assert not (Path(s['prepared_dir'])/'f2_preparation.json').exists()
 assert not (Path(s['prepared_dir'])/'settings.json').exists()
 p=preflight(s);index=tmp_path/'join.sqlite';build_join_index(p.source,index)
 def forbidden(*a,**k):raise AssertionError('No target support constructed a model')
 monkeypatch.setattr('embodied.run.fg_s3_e_quality_heads.create_agent',forbidden)
 with pytest.raises(ValueError,match='F2_TARGET_LOSS'):HeadsRun(p,index,Path(s['run_dir']),_probe=engineering_permit(p))
 assert not Path(s['run_dir']).exists()
