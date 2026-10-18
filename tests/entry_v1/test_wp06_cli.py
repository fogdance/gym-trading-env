import copy,json,subprocess,sys,uuid
from pathlib import Path
import pytest,yaml
from wp06_fixture import engineering_settings
from gym_trading_env.research.fg_s3_e_quality_foundation import PROJECT
from gym_trading_env.research.fg_s3_e_quality_actor import prepare,preflight

@pytest.mark.parametrize('command',['f3-start','f3-resume'])
def test_F05_public_start_resume_no_authorization_no_output(tmp_path,command):
    out=tmp_path/'run'
    r=subprocess.run([sys.executable,'-B',str(PROJECT/'tools/fg_s3_v2_experiment.py'),command,'--prepared',str(tmp_path/'absent'),'--run-dir',str(out)],capture_output=True,text=True)
    assert r.returncode==1 and 'F3_EXTERNAL_TRAINING_AUTHORIZATION_REQUIRED' in r.stderr and not out.exists()

def test_F06_prepare_is_not_training_and_engineering_cannot_become_formal(tmp_path):
    s=engineering_settings(tmp_path);config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(s))
    receipt=prepare(config);assert receipt['status']=='F3_ENGINEERING_READY' and not receipt['training_admitted']
    out=Path(s['prepared_dir']);assert {p.name for p in out.iterdir()}=={'settings.json','reader_binding.json','f3_preparation.json','join.sqlite','join.json'}
    r=subprocess.run([sys.executable,'-B',str(PROJECT/'tools/fg_s3_v2_experiment.py'),'f3-start','--prepared',str(out),'--authorize-training'],capture_output=True,text=True)
    assert r.returncode==1 and 'FOUNDATION_NOT_TRAINED' in r.stderr and not Path(s['run_dir']).exists()
    with pytest.raises(ValueError,match='OUTPUT_MUST_BE_NEW'):prepare(config)

@pytest.mark.parametrize('command,args',[
    ('f3-probe',['--config','no-user-source.yaml']),
    ('f3-read-check',['--authorize-training']),
    ('f3-evaluate',['--diagnostic-components','outcome,score,q'])])
def test_F07_cli_forbids_user_source_probe_training_or_prediction_overrides(tmp_path,command,args):
    out=tmp_path/'forbidden'
    r=subprocess.run([sys.executable,'-B',str(PROJECT/'tools/fg_s3_v2_experiment.py'),command,'--output-dir',str(out),*args],capture_output=True,text=True)
    assert r.returncode==2 and not out.exists()

def test_F08_read_only_does_not_consume_source_model_or_touch_parameters(tmp_path):
    from gym_trading_env.research.fg_s3_e_quality_actor_checks import read_check
    s=engineering_settings(tmp_path);s['source_run']=str(tmp_path/'absent_f1');s['source_checkpoint']=str(tmp_path/'absent_f1/ckpt/step_000000000004')
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(s));out=PROJECT/'artifacts/fg_s3_v2/e_quality_wp06'/('unit_read_'+uuid.uuid4().hex)
    r=read_check(config,out)
    assert r['source_model_consumed'] is False and r['foundation_status']=='FOUNDATION_NOT_TRAINED'
    assert r['real_TRAIN_updates']==r['actual_optimizer_updates']==0 and r['before']==r['after']
    assert r['valid_l2_exposures']>0 and r['unique_target_market_cursors']<=r['target_account_rows']
    assert (out/'read_check.json').exists() and not (out/'run').exists() and not (out/'ckpt').exists()

def test_F09_different_corpus_and_empty_source_refused_without_prepared_or_run(tmp_path):
    from wp03_fixture import native_corpus
    s=engineering_settings(tmp_path);s['corpus_root']=str(native_corpus(tmp_path/'other_corpus',minutes=5))
    with pytest.raises(ValueError,match='SOURCE_CORPUS_COURSE_MISMATCH'):preflight(s)
    s['source_run']=''
    with pytest.raises(ValueError):preflight(s)
    assert not Path(s['run_dir']).exists() and not Path(s['prepared_dir']).exists()

def test_F10_no_target_support_creates_no_ready_receipt_or_model(tmp_path):
    s=engineering_settings(tmp_path,target=1,burnin=98);s['sampling_spec']={'mode':'NATURAL','policies':['P0']}
    config=tmp_path/'bad.yaml';config.write_text(yaml.safe_dump(s))
    with pytest.raises(ValueError,match='F3_TARGET_L2'):prepare(config)
    out=Path(s['prepared_dir'])
    assert not (out/'f3_preparation.json').exists() and not (out/'settings.json').exists()
    assert not Path(s['run_dir']).exists()


def test_F11_partial_engineering_budget_is_not_reported_complete():
    from types import SimpleNamespace
    from embodied.run.fg_s3_e_quality_actor import ActorRun,ActorCounters
    run=object.__new__(ActorRun);run.settings={'optimizer_updates':4};run.identity={'controlled_small':True,'training_admitted':False}
    run.preparation=SimpleNamespace(receipt={'identity':{'source_model':{'foundation_status':'FOUNDATION_NOT_TRAINED'}}})
    run.counters=ActorCounters();run.counters.state.update(optimizer_updates=2,consumed_batches=2)
    run.agent=SimpleNamespace(n_updates=2);run.cp=SimpleNamespace(latest=lambda:Path('/synthetic/checkpoint'))
    run.source=SimpleNamespace(coverage=lambda:{})
    assert run.result()['status']=='F3_ENGINEERING_PARTIAL'
    run.agent.n_updates=4;run.counters.state.update(optimizer_updates=4,consumed_batches=4)
    assert run.result()['status']=='F3_ENGINEERING_COMPLETE'
