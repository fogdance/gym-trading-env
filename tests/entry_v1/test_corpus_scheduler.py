"""Scheduling contracts: finite subprocesses, complete receipts, no math mocks."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from gym_trading_env.research import fg_s3_corpus_resources as resources
from gym_trading_env.research import fg_s3_corpus_production as production
from gym_trading_env.research.fg_replay import signed, write_json


def test_weighted_allocation_and_minimum():
    expected = {24:[8,2,2,2,2,8],17:[5,2,2,2,2,4],18:[5,2,2,2,2,5],20:[6,2,2,2,2,6]}
    for count, budgets in expected.items():
        assert list(resources.allocate(resources.DEFAULT_CPUS,count).values())==budgets
    with pytest.raises(ValueError,match='at least 12'):
        resources.allocate(resources.DEFAULT_CPUS,11)
    with pytest.raises(ValueError,match='six policy'):
        resources.policy_cpus({'P0':8})
    with pytest.raises(ValueError,match='policy-cpus'):
        resources.policy_cpus('P0=8,P0=9')


def test_batch_manifest_uses_existing_published_reader_hash_contract(tmp_path):
    import hashlib
    value=dict(protocol='FG_S3_E_TRAIN_CORPUS_BATCH_RESULT_V1',files={'corpus/a':'123'})
    expected=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write_json(tmp_path/'BATCH_MANIFEST.json',production.batch_signed(value))
    assert production.read_batch(tmp_path/'BATCH_MANIFEST.json')['content_hash']==expected


def test_production_cli_failure_returns_no_go_json(tmp_path,capsys):
    from gym_trading_env.research.fg_s3_e_quality_corpus_cli import main
    assert main(['resume','--config',str(tmp_path/'missing.yaml')])==2
    assert json.loads(capsys.readouterr().out)['status']=='NO_GO'


def test_memory_reserves_unstarted_workers_and_backfills_fitting_task():
    memory=[1000]
    ledger=resources.Ledger(6,100,100,50,memory=lambda:memory[0],resident=lambda groups:0)
    first=ledger.acquire(4,'P0')
    memory[0]=650
    with ThreadPoolExecutor(2) as executor:
        large=executor.submit(ledger.acquire,4,'P4')
        small=executor.submit(ledger.acquire,1,'P1')
        token=small.result(timeout=3)
        assert ledger.used()==5 and not large.done()
        ledger.release(token)
        ledger.release(first)
        ledger.release(large.result(timeout=3))
    assert ledger.used()==0


def _account(job):
    start=time.monotonic()
    time.sleep(job['seconds'])
    return dict(name=job['name'],start=start,end=time.monotonic(),pid=os.getpid(),teachers=job['label_workers'])


def test_real_short_lived_accounts_share_broker_and_release_before_phase_end(monkeypatch):
    events=[]
    ledger=resources.Ledger(4,1,0,0,memory=lambda:100,resident=lambda groups:0,event=events.append)
    broker=resources.Broker(ledger)
    monkeypatch.setenv(resources.ENV,broker.setting)
    try:
        result=resources.run_policy_jobs(_account,
            [dict(name='P0',seconds=.2),dict(name='P4',seconds=1.5),dict(name='P1',seconds=.2)],
            dict(P0=2,P4=2,P1=2))
        resources.pause_phase()
    finally:
        ledger.stop();broker.close()
    by_name={item['name']:item for item in result}
    assert len({item['pid'] for item in result})==3
    assert all(item['teachers']==1 for item in result)
    assert by_name['P1']['start']<by_name['P4']['end']
    assert max(event['cpu_used'] for event in events)<=4
    assert ledger.used()==0
    assert all(resources.process_identity(item['pid']) is None for item in result)


def test_legacy_progress_only_releases_closed_teacher_pools_once(monkeypatch):
    ledger=resources.Ledger(24,1,0,0,memory=lambda:1000,resident=lambda groups:0)
    broker=resources.Broker(ledger)
    monkeypatch.setenv(resources.ENV,broker.setting)
    try:
        with resources.legacy_verification_lease(24) as progress:
            progress(json.dumps(dict(phase='verify',resource_budget=dict(workers=6,
                label_workers=[3]*6,total_worker_processes=24))))
            line=json.dumps(dict(behavior='P1',receipt={'status':'PASS'}))
            progress(line);progress(line)
            assert ledger.used()==21
        resources.pause_phase()
        assert ledger.used()==0
    finally:
        ledger.stop();broker.close()


def test_duplicate_controller_lock_and_pid_reuse_are_refused(tmp_path):
    with production.ProductionLock(tmp_path):
        with pytest.raises(ValueError,match='Another Corpus'):
            with production.ProductionLock(tmp_path):
                pass
    identity=resources.process_identity(os.getpid())
    identity['start_ticks']+=1
    with pytest.raises(ValueError,match='refusing'):
        production.terminate_owned(identity)


def test_phase_group_cleanup_includes_descendants(tmp_path):
    target=tmp_path/'child_pid'
    code="import subprocess,time; from pathlib import Path; p=subprocess.Popen(['sleep','60']); Path(%r).write_text(str(p.pid)); time.sleep(60)" % str(target)
    child=subprocess.Popen([production.PYTHON,'-c',code],start_new_session=True)
    try:
        for _ in range(100):
            if target.exists():break
            time.sleep(.02)
        grandchild=int(target.read_text())
        identity=resources.process_identity(child.pid)
        production.terminate_owned(identity,grace=.5)
        child.wait(timeout=3)
        assert all(state=='Z' for pid,state in production.group_members(identity['pgid']))
        assert grandchild!=child.pid
    finally:
        if child.poll() is None:
            production.terminate_owned(resources.process_identity(child.pid),grace=.1);child.wait()


def test_resume_can_clean_owned_orphans_after_group_leader_exits(tmp_path):
    target=tmp_path/'orphan'
    code="import subprocess,time; from pathlib import Path; p=subprocess.Popen(['sleep','60']); Path(%r).write_text(str(p.pid)); time.sleep(60)" % str(target)
    leader=subprocess.Popen([production.PYTHON,'-c',code],start_new_session=True)
    try:
        for _ in range(100):
            if target.exists():break
            time.sleep(.02)
        orphan=resources.process_identity(int(target.read_text()))
        identity=resources.process_identity(leader.pid)
        leader.terminate();leader.wait(timeout=3)
        production.terminate_owned(identity,grace=.5,known_members=[orphan])
        assert all(state=='Z' for pid,state in production.group_members(identity['pgid']))
    finally:
        if leader.poll() is None:leader.terminate();leader.wait()


def test_incomplete_or_signaled_zero_native_receipt_is_not_success(tmp_path):
    write_json(tmp_path/'result.json',dict(status='NATIVE_G3_VERIFIED',corpus_identity='a'))
    with pytest.raises(FileNotFoundError):
        production.stage_result(tmp_path,'native_g3')
    write_json(tmp_path/'native/native_verification.json',dict(exit_code=0,
        result=dict(status='PASS',G3='GO',verification_state='NATIVE_G3_VERIFIED',corpus_identity='a',native_replay=[])))
    with pytest.raises(ValueError,match='Incomplete native'):
        production.stage_result(tmp_path,'native_g3')


def test_completion_validates_hashes_and_actual_artifact(tmp_path):
    write_json(tmp_path/'result.json',dict(status='PASS_EXACT_RATIONAL_RUNTIME',python_int_max_str_digits=0))
    completion=signed(dict(status='succeeded',stage='runtime_limits',exit_code=0,
                          files={str(tmp_path/'result.json'):production.digest(tmp_path/'result.json')}))
    write_json(tmp_path/'completion.json',completion)
    record=dict(status='succeeded',stage='runtime_limits',attempt=str(tmp_path),completion_hash=completion['content_hash'])
    assert production.validate_completion(record)
    (tmp_path/'result.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        production.validate_completion(record)
    with pytest.raises(ValueError,match='changed'):
        production.publish_batch(dict(corpus='unused',out='unused',previous={'runtime_limits':str(tmp_path)},
            previous_receipts={'runtime_limits':completion['content_hash']}))


class FixtureController(production.Controller):
    def stage_command(self,directory):
        return [production.PYTHON,'-B',str(Path(__file__).with_name('corpus_scheduler_fixture.py')),str(directory/'request.json')]


def test_strategy_failure_aborts_other_accounts_without_waiting_for_workload(tmp_path,monkeypatch):
    monkeypatch.setattr(production,'authenticate',lambda producer:{})
    monkeypatch.setattr(production,'check_environment',lambda config:None)
    config=dict(production_root=str(tmp_path/'production'),producer={},max_inflight_batches=2,
        timeline=str(tmp_path/'timeline'),fail_policy=True,batches=[dict(id='TRAIN01')],
        resources=dict(workers=12,affinity=sorted(os.sched_getaffinity(0)),native_threads=1,
            policy_cpus={key:2 for key in resources.POLICIES},worker_memory_mib=1,
            coordinator_memory_mib=1,headroom_mib=1))
    controller=FixtureController(config,'recipe');start=time.monotonic()
    with pytest.raises(RuntimeError,match='SYNTHETIC_STRATEGY_FAILURE'):
        controller.run(False)
    assert time.monotonic()-start<20
    records=controller.state['batches']['TRAIN01']['stages']['generate']
    assert len(records)==1 and records[0]['status']!='succeeded'
    assert all(state=='Z' for pid,state in production.group_members(records[0]['process']['pgid']))


def test_sigint_stops_owned_phases_and_records_interrupted_state(tmp_path):
    import signal
    config=dict(production_root=str(tmp_path/'production'),producer={},max_inflight_batches=2,
        timeline=str(tmp_path/'timeline'),batches=[dict(id='TRAIN01'),dict(id='TRAIN02')],
        resources=dict(workers=12,affinity=sorted(os.sched_getaffinity(0)),native_threads=1,
            policy_cpus={key:2 for key in resources.POLICIES},worker_memory_mib=1,
            coordinator_memory_mib=1,headroom_mib=1))
    path=tmp_path/'config.json';write_json(path,config)
    script="import json,sys; from gym_trading_env.research import fg_s3_corpus_production as p; from test_corpus_scheduler import FixtureController; p.authenticate=lambda x:{}; p.check_environment=lambda x:None; FixtureController(json.load(open(sys.argv[1])),'recipe').run(False)"
    with (tmp_path/'controller.log').open('w') as log:
        child=subprocess.Popen([production.PYTHON,'-B','-c',script,str(path)],stdout=log,stderr=subprocess.STDOUT)
        try:
            state_path=Path(config['production_root'])/'control/state.json'
            for _ in range(400):
                if child.poll() is not None:pytest.fail((tmp_path/'controller.log').read_text())
                if state_path.exists():
                    state=json.loads(state_path.read_text())
                    g3=state['batches']['TRAIN01']['stages']['native_g3']
                    if g3 and len(g3[-1].get('members',[]))>2:break
                time.sleep(.05)
            else:pytest.fail('No active G3 descendants observed')
            child.send_signal(signal.SIGINT);assert child.wait(timeout=15)!=0
            state=json.loads(state_path.read_text())
            for batch in state['batches'].values():
                for records in batch['stages'].values():
                    for record in records:
                        assert record['status']!='running'
                        if record.get('process'):
                            assert all(status=='Z' for pid,status in production.group_members(record['process']['pgid']))
            assert state['batches']['TRAIN01']['stages']['native_g3'][-1]['status']=='interrupted'
        finally:
            if child.poll() is None:child.send_signal(signal.SIGINT);child.wait(timeout=15)


@pytest.mark.parametrize('failure',[False,True])
def test_real_pipeline_and_resume_skip_generated_batch(tmp_path,monkeypatch,failure):
    monkeypatch.setattr(production,'authenticate',lambda producer:{})
    monkeypatch.setattr(production,'check_environment',lambda config:None)
    cpus=sorted(os.sched_getaffinity(0))
    config=dict(production_root=str(tmp_path/'production'),producer={},max_inflight_batches=2,
        timeline=str(tmp_path/'timeline'),batches=[dict(id='TRAIN01'),dict(id='TRAIN02')],
        resources=dict(workers=12,affinity=cpus,native_threads=1,policy_cpus={key:2 for key in resources.POLICIES},
                       worker_memory_mib=1,coordinator_memory_mib=1,headroom_mib=1))
    if failure:config['fail_once']=str(tmp_path/'fail_once')
    controller=FixtureController(config,'recipe')
    if failure:
        with pytest.raises(RuntimeError,match='exited 9'):
            controller.run(False)
        generated=controller.successful(config['batches'][0])['generate']
        original=(Path(generated['attempt'])/'completion.json').read_bytes()
        # Simulate a controller crash after mkdir but before atomic state save.
        orphan=Path(controller.state['batches']['TRAIN01']['stages']['native_g3'][0]['attempt']).parent/'0002'
        orphan.mkdir();(orphan/'partial.log').write_text('preserve uncommitted attempt')
        controller=FixtureController(config,'recipe')
        assert controller.run(True)==0
        records=controller.state['batches']['TRAIN01']['stages']
        assert len(records['generate'])==1 and len(records['native_g3'])==2
        assert (Path(generated['attempt'])/'completion.json').read_bytes()==original
        assert records['native_g3'][0]['status']=='failed'
        assert Path(records['native_g3'][1]['attempt']).name=='0003'
        assert (orphan/'partial.log').read_text()=='preserve uncommitted attempt'
    else:
        assert controller.run(False)==0
        events=[json.loads(line) for line in Path(config['timeline']).read_text().splitlines()]
        starts={(v['batch'],v['stage']):v['time'] for v in events if v['event']=='start'}
        ends={(v['batch'],v['stage']):v['time'] for v in events if v['event']=='end'}
        assert starts['TRAIN02','generate']<ends['TRAIN01','native_g3']
    assert all('publish' in controller.successful(batch) for batch in config['batches'])
    events=[json.loads(line) for line in (Path(config['production_root'])/'control/resources.jsonl').read_text().splitlines()]
    assert max(value['cpu_used'] for value in events)<=12
    assert controller.ledger.used()==0
    # Idempotent resume performs no further stages.
    before=Path(config['timeline']).read_bytes()
    assert FixtureController(config,'recipe').run(True)==0
    assert Path(config['timeline']).read_bytes()==before


def test_only_resources_can_change_between_resume_attempts(tmp_path):
    import yaml
    config_path=Path(__file__).resolve().parents[2]/'configs/experiments/fg_s3_v2/corpus_production.yaml'
    config,identity=production.load_config(config_path)
    config['resources']['workers']=20
    config['resources']['policy_cpus']['P0']=6
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(config))
    assert production.load_config(path)[1]==identity
    config['seed']+=1;path.write_text(yaml.safe_dump(config))
    assert production.load_config(path)[1]!=identity
