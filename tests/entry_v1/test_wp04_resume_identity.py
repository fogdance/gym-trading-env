"""Resume rejects valid changed contracts against a positively restored bundle."""
import copy
import json
from pathlib import Path
import pytest
from gym_trading_env.research import fg_s3_e_quality_foundation as contract
from embodied.run import fg_s3_e_quality_foundation as runmod

@pytest.fixture(scope='module')
def restored(tmp_path_factory):
 evidence=json.loads(Path('docs/receipts/wp04/D_cold_restore_evidence.json').read_text())
 case=Path(evidence['case_root']);s=json.loads((case/'settings.json').read_text())
 s.update(target_steps=8,burnin_steps=4)
 root=tmp_path_factory.mktemp('wp04_identity')/'run';p=contract.preflight(s)
 fresh=runmod.FoundationRun(p,case/'join.sqlite',root,_probe=runmod.synthetic_probe_permit(p))
 cold=runmod.FoundationRun(p,case/'join.sqlite',root,resume=True,_probe=runmod.synthetic_probe_permit(p))
 assert fresh.agent.save_exact()['counters']==cold.agent.save_exact()['counters']
 assert int(cold.step)==0
 try:yield s,case/'join.sqlite',root
 finally:fresh.reader.close();cold.reader.close()

def test_E01_current_contract_first_restores_positively(restored):
 assert (restored[2]/'ckpt/step_000000000000/done').exists()

@pytest.mark.parametrize('field',['burnin','seed','model','sampling','optimizer','loss','code'])
def test_E02_valid_changed_contract_refused_before_model_or_write(restored,field,monkeypatch):
 s,index,root=restored;s=copy.deepcopy(s)
 if field=='burnin':s['burnin_steps']+=1
 elif field=='seed':s['model_init']['seed']+=1
 elif field=='model':s['model_init']['presets']=['size12m']
 elif field=='sampling':s['sampling_spec']['policies']=['P0']
 elif field in ('optimizer','loss'):
  original=contract.model_config
  def changed(*args,**kwargs):
   cfg=original(*args,**kwargs)
   if field=='optimizer':cfg['agent']['opt']['lr']*=2
   else:cfg['agent']['loss_scales']['rew']*=2
   return cfg
  monkeypatch.setattr(contract,'model_config',changed)
 else:
  old=contract.execution_identity()
  def changed():
   data=copy.deepcopy(old);data['gym']['head']='0'*40;return data
  monkeypatch.setattr(contract,'execution_identity',changed)
  monkeypatch.setattr(runmod,'execution_identity',changed)
 # This is a valid newly resolved configuration, not an unknown key inserted
 # into an old receipt. Only the requested contract field differs.
 preparation=contract.preflight(s)
 def forbidden(*args,**kwargs):raise AssertionError('Rejected restore initialized model')
 monkeypatch.setattr(runmod,'create_agent',forbidden)
 before={str(p):p.stat().st_mtime_ns for p in root.rglob('*') if p.is_file()}
 with pytest.raises(ValueError,match='RESUME_IDENTITY_MISMATCH'):
  runmod.FoundationRun(preparation,index,root,resume=True,_probe=runmod.synthetic_probe_permit(preparation))
 assert before=={str(p):p.stat().st_mtime_ns for p in root.rglob('*') if p.is_file()}
