"""Resume attempts complete state loading despite current code/config changes."""
import copy
import json
from pathlib import Path
import pytest
from gym_trading_env.research import fg_s3_e_quality_foundation as contract
from embodied.run import fg_s3_e_quality_foundation as runmod

@pytest.fixture(scope='module')
def restored(tmp_path_factory):
 from wp04_fixture import native_reader
 case=tmp_path_factory.mktemp('wp04_identity');s,reader=native_reader(case)
 reader.close();root=case/'run';p=contract.preflight(s)
 fresh=runmod.FoundationRun(p,case/'join.sqlite',root,_probe=runmod.synthetic_probe_permit(p))
 cold=runmod.FoundationRun(p,case/'join.sqlite',root,resume=True,_probe=runmod.synthetic_probe_permit(p))
 assert fresh.agent.save_exact()['counters']==cold.agent.save_exact()['counters']
 assert int(cold.step)==0
 try:yield s,case/'join.sqlite',root
 finally:fresh.reader.close();cold.reader.close()

def test_E01_current_contract_first_restores_positively(restored):
 assert (restored[2]/'ckpt/step_000000000000/done').exists()

@pytest.mark.parametrize('field',['optimizer','loss','code'])
def test_E02_changed_definitions_attempt_full_restore_without_rewriting_owner(restored,field,monkeypatch):
 s,index,root=restored;s=copy.deepcopy(s)
 if field in ('optimizer','loss'):
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
 preparation=contract.preflight(s)
 before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file() and
         (p.name=='run_identity.json' or p.is_relative_to(root/'ckpt'))}
 run=runmod.FoundationRun(preparation,index,root,resume=True,_probe=runmod.synthetic_probe_permit(preparation))
 assert int(run.step)==int(run.agent.n_updates)==0
 assert before=={str(p):p.read_bytes() for p in root.rglob('*') if str(p) in before}
 run.source.close()
