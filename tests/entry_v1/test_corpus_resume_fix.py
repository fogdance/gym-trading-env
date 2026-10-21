"""Only the observed exit race and the public resume migration boundary."""
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import yaml

from gym_trading_env.research import fg_s3_corpus_production as production
from gym_trading_env.research import fg_s3_corpus_resources as resources
from gym_trading_env.research.fg_replay import digest, signed, write_json


class ExitController(production.Controller):
    def stage_command(self, directory):
        code=("import subprocess,sys,time,json;from pathlib import Path;"
              "subprocess.Popen([sys.executable,'-c','import time;time.sleep(%r)']);"
              "Path(%r).write_text(json.dumps(dict(status='PASS_EXACT_RATIONAL_RUNTIME',python_int_max_str_digits=0)));"
              "time.sleep(.2);sys.exit(%r)" %
              (self.config['helper_seconds'],str(directory/'result.json'),self.config['exit_code']))
        return [production.PYTHON,'-B','-c',code]


@pytest.mark.parametrize('duration,exit_code,success',[(.9,0,True),(60,0,False),(.1,7,False)])
def test_stage_exit_waits_for_helpers_but_preserves_failures(tmp_path,monkeypatch,duration,exit_code,success):
    monkeypatch.setattr(production,'PHASE_EXIT_GRACE',1.5 if success else .15)
    batch=dict(id='TRAIN01')
    config=dict(production_root=str(tmp_path/'production'),producer={},batches=[batch],
                helper_seconds=duration,exit_code=exit_code,
                resources=dict(affinity=sorted(os.sched_getaffinity(0))))
    controller=ExitController(config,'recipe');controller.initialize(False)
    controller.execution={}
    controller.capacity=12;controller.budgets={name:2 for name in resources.POLICIES}
    controller.ledger=resources.Ledger(12,1,0,0,check_memory=False)
    controller.broker=SimpleNamespace(setting='unused')
    directory=tmp_path/'attempt';directory.mkdir()
    record=dict(stage='runtime_limits',status='running',attempt=str(directory),started=time.time())
    controller.state['batches']['TRAIN01']['stages']['runtime_limits'].append(record)
    controller.running['TRAIN01']=record
    controller.run_stage(batch,'runtime_limits',record,{})
    assert (record['status']=='succeeded') is success
    assert record['exit_code']==exit_code
    report=json.loads((directory/'process_exit.json').read_text())
    assert report['exit_code']==exit_code
    assert (directory/'completion.json').exists() is success
    assert not any(state!='Z' for _,state in production.group_members(record['process']['pgid']))
    if success:
        assert report['initial_members'] and report['status']=='DRAINED'
        assert not controller.stopping.is_set()
    elif exit_code==0:
        assert report['status']=='TIMED_OUT' and report['remaining_members'][0]['command']
        assert record['cleanup']['remaining_members']==[]


def test_resume_migration_keeps_receipts_and_recovery_uses_original_pin(tmp_path,monkeypatch):
    monkeypatch.setattr(production,'authenticate',lambda producer:{})
    config=dict(protocol='FG_S3_CORPUS_PRODUCTION_V1',production_root=str(tmp_path/'production'),
        producer=dict(gym='unused',pin='old_pin.json',pin_sha256='a'*64),seed=202,reader_length=64,
        max_inflight_batches=2,batches=[dict(id='TRAIN01',contract='CZCE.FG809',start_day='2018-04-11',end_day='2018-08-08')],
        resources=dict(workers=12,affinity=list(range(24)),native_threads=1,
            policy_cpus={name:2 for name in resources.POLICIES},worker_memory_mib=1,
            coordinator_memory_mib=1,headroom_mib=1))
    old=tmp_path/'old.yaml';old.write_text(yaml.safe_dump(config))
    config['producer']=dict(gym='unused',pin='new_pin.json',pin_sha256='b'*64)
    current=tmp_path/'current.yaml';current.write_text(yaml.safe_dump(config))
    old_config,recipe=production.load_config(old)
    assert production.load_config(current)[1]==recipe
    controller=production.Controller(config,recipe)
    attempt=tmp_path/'generate/0001';attempt.mkdir(parents=True)
    corpus=attempt/'corpus';manifest=signed(dict(protocol='SYNTHETIC_NOT_TRAIN'))
    write_json(corpus/'corpus_manifest.json',manifest)
    result=dict(status='CORPUS_PRODUCED',corpus=str(corpus),corpus_identity=manifest['content_hash'])
    write_json(attempt/'generated.json',result);write_json(attempt/'result.json',result)
    write_json(attempt/'request.json',dict(config=old_config,batch=old_config['batches'][0]))
    failed=dict(stage='generate',status='failed',attempt=str(attempt),error='original exit audit failure')
    state=dict(protocol='FG_S3_CORPUS_PRODUCTION_STATE_V1',recipe_hash=production.legacy_recipe(old_config),
        batches={'TRAIN01':dict(stages={stage:([failed] if stage=='generate' else []) for stage in production.STAGES})})
    write_json(controller.state_path,state)
    # Migration's real-source validation is covered by the native provenance
    # test and actual production preparation; here verify state transitions.
    monkeypatch.setattr(production,'validate_source',lambda *a:None)
    monkeypatch.setattr(production,'validate_generated',lambda *a,**k:manifest)
    write_json(corpus/'source_lock.json',signed(dict(fold='SYNTHETIC_NOT_TRAIN')))
    before=digest(attempt/'result.json')
    receipt=production.prepare_resume(current,old,apply=True)
    assert receipt['status']=='RESUME_PREPARED' and receipt['stages_promoted'] is False
    saved=production.read(controller.state_path)
    assert saved['batches']==state['batches'] and digest(attempt/'result.json')==before
    assert production.prepare_resume(current,old,apply=True)['status']=='RESUME_ALREADY_PREPARED'
    controller.state=saved;controller.capacity=12;controller.budgets=resources.policy_cpus(config['resources']['policy_cpus'])
    request=controller.request(config['batches'][0],'generate',tmp_path/'generate/0002',{})
    assert request['recovered_generation']['origin_pin']['pin_sha256']=='a'*64
    assert request['recovered_generation']['corpus']==str(corpus)
    config['batches'][0]['end_day']='2018-08-07';current.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match='data recipe'):
        production.prepare_resume(current,old,apply=True)
