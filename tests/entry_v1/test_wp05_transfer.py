import copy,pickle,json,shutil
from pathlib import Path
import numpy as np
import pytest
from wp05_fixture import engineering_settings
from embodied.run.fg_s3_e_quality_transfer import inspect_f1,transfer_parameters
from gym_trading_env.research.fg_s3_e_quality_heads import create_agent,preflight,require_training_authority
from dreamerv3.fg_s3a_actor import parameter_digests
from gym_trading_env.research.fg_replay import signed,write_json,digest

@pytest.fixture(scope='module')
def source(tmp_path_factory):
 root=tmp_path_factory.mktemp('wp05_transfer');s=engineering_settings(root)
 proof,data=inspect_f1(s,allow_engineering=True)
 return root,s,proof,data

def test_B01_nonopt_complete_fresh_optimizer_and_mirror(source):
 root,s,proof,data=source;before=digest(Path(s['source_f1_checkpoint'])/'agent.pkl')
 a=create_agent(s,root/'model',controlled_small=True);fresh=parameter_digests(a.params)
 receipt=transfer_parameters(a,data,proof);after=parameter_digests(a.params)
 expected=parameter_digests(data['params'])
 assert all(after[k]==expected[k] for k in after if not k.startswith('opt/'))
 assert all(after[k]==fresh[k] for k in after if k.startswith('opt/'))
 assert int(a.n_updates)==int(a.n_batches)==int(a.n_actions)==0
 assert parameter_digests(a.policy_params)=={k:after[k] for k in a.policy_keys}
 assert a.pending_sync is a.pending_outs is a.pending_mets is None
 assert receipt['foundation_status']=='FOUNDATION_NOT_TRAINED' and not receipt['same_run_resume']
 assert digest(Path(s['source_f1_checkpoint'])/'agent.pkl')==before
 out=Path('docs/receipts/wp05/B_transfer_receipt.json');write_json(out,receipt)

@pytest.mark.parametrize('attack',['missing','shape','dtype','nonfinite'])
def test_B02_parameter_failure_before_mutation(source,tmp_path,attack):
 _,s,proof,data=source;a=create_agent(s,tmp_path/'model',controlled_small=True)
 altered=copy.deepcopy(data);key=next(k for k in altered['params'] if k.startswith('enc/'))
 if attack=='missing':altered['params'].pop(key)
 elif attack=='shape':altered['params'][key]=np.zeros(1,np.float32)
 elif attack=='dtype':altered['params'][key]=altered['params'][key].astype(np.float64)
 else:altered['params'][key][...]=np.nan
 before=parameter_digests(a.params)
 with pytest.raises(ValueError,match='HANDOFF'):transfer_parameters(a,altered,proof)
 assert parameter_digests(a.params)==before and int(a.n_updates)==0

@pytest.mark.parametrize('attack',['formal','unfinished','stage','reward','observation','model','corrupt'])
def test_B03_source_failures(source,tmp_path,attack):
 _,s,_,_=source
 if attack=='formal':
  with pytest.raises(ValueError,match='FOUNDATION_NOT_TRAINED'):inspect_f1(s)
  return
 src=tmp_path/'source';shutil.copytree(s['source_f1_run'],src)
 t={**s,'source_f1_run':str(src),'source_f1_checkpoint':str(src/'ckpt/step_000000000004')}
 if attack=='corrupt':
  (Path(t['source_f1_checkpoint'])/'agent.pkl').write_bytes(b'corrupt')
 else:
  r=json.loads((src/'run_identity.json').read_text());i=r['identity']
  if attack=='unfinished':i['preparation']['optimizer_updates']=5
  elif attack=='stage':i['preparation']['course_stage']='QUALITY_HEADS'
  elif attack=='reward':i['preparation']['reward_schema']='WRONG'
  elif attack=='observation':i['preparation']['observation']['macd_5m']['partial']='FUTURE'
  else:i['effective_model']['agent']['dyn']['rssm']['deter']+=1
  write_json(src/'run_identity.json',signed(dict(identity=i)))
  # Keep checksummed outer identity coherent to reach the intended validator.
  p=Path(t['source_f1_checkpoint'])/'manifest.json';m=json.loads(p.read_text());m['identity']=i;p.write_text(json.dumps(m))
 with pytest.raises(ValueError):inspect_f1(t,allow_engineering=True)

def test_B04_engineering_preflight_cannot_mint_training_authority(source):
 _,s,_,_=source;p=preflight(s)
 assert p.receipt['status']=='F2_ENGINEERING_READY' and 'FOUNDATION_NOT_TRAINED' in p.receipt['blockers']
 with pytest.raises(ValueError,match='AUTHORIZATION_REQUIRED'):require_training_authority(p)
 with pytest.raises(ValueError,match='FOUNDATION_NOT_TRAINED'):require_training_authority(p,authorized=True)
 assert not Path(s['run_dir']).exists()
