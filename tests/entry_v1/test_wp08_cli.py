import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_wp08_contract import plan
from wp08_fixture import ARTIFACTS,f3_reference

ROOT=Path(__file__).resolve().parents[2]


def test_worker_refuses_undeclared_job_before_source_io(tmp_path,monkeypatch):
    from gym_trading_env.research import fg_s3_e_quality_exam as exam
    def forbidden(*args,**kwargs):raise AssertionError('source opened before job validation')
    monkeypatch.setattr(exam,'source_for',forbidden)
    p=plan(tmp_path/'out')
    with pytest.raises(ValueError,match='NOT_PREREGISTERED'):
        exam.execute_job(p,dict(contract=p['contracts'][0],policy='ALWAYS_WAIT',random_seed=None,label='../other'))
    assert not (tmp_path/'out').exists()


def test_existing_eval_path_keeps_default_data_overwrite_guard(tmp_path):
    from e_data_v1.io import independent_output
    existing=tmp_path/'owned';existing.mkdir()
    with pytest.raises(FileExistsError):independent_output(existing)
    assert independent_output(existing,allow_existing=True)==existing
    source=existing/'source.json';source.write_text('{}')
    with pytest.raises(ValueError):independent_output(existing,inputs=[source],allow_existing=True)
    with pytest.raises(ValueError):independent_output('/data/logdir/fg_single3/forbidden',allow_existing=True)


def test_formal_dev_without_trained_actor_nonzero_and_no_outputs(tmp_path):
    p=plan(tmp_path/'out');p.update(exam_mode='DEV_EXAM',contracts=['CZCE.FG409'],start_day='2024-04-11',end_day='2024-04-11')
    p['source']=dict(kind='VALIDATION',manifest='/home/v/Documents/work/gym-trading-env/artifacts/fg_abcd_v1/v3_shared_prefix/FG409/all/manifest.json',
        env_receipt='/data/logdir/fg_s3_market_minute/20261013_v2/env/v1')
    config=tmp_path/'dev.json';config.write_text(json.dumps(p))
    log=tmp_path/'formal_missing.log'
    assert cli(['exam-plan','--config',str(config)],log)!=0
    assert 'FORMAL_TRAINED_ACTOR_NOT_PROVIDED' in log.read_text()
    assert not (tmp_path/'out').exists()


def cli(args,log):
    argv=[sys.executable,'-B','tools/fg_s3_v2_experiment.py',*args]
    with Path(log).open('x') as stream:r=subprocess.run(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    Path(str(log)+'.command.json').write_text(json.dumps(dict(argv=argv,cwd=str(ROOT),exit_code=r.returncode),indent=2)+'\n')
    return r.returncode


@pytest.mark.parametrize('change', ['test','empty','dev_train','training_flag','gpu_flag'])
def test_cli_refusal_no_output_or_training_side_effect(tmp_path,change):
    p=plan(tmp_path/'out');flags=[]
    if change=='test':p['exam_mode']='SEALED_TEST_FINAL'
    if change=='empty':p['exam_mode']=''
    if change=='dev_train':p['exam_mode']='DEV_EXAM'
    if change=='training_flag':flags=['--authorize-training']
    if change=='gpu_flag':flags=['--eval-device','gpu']
    config=tmp_path/'plan.yaml';config.write_text(json.dumps(p))
    assert cli(['exam-run','--config',str(config),*flags],tmp_path/'negative.log')!=0
    assert not (tmp_path/'out').exists()
    assert not list(tmp_path.glob('**/agent.pkl')) and not list(tmp_path.glob('**/prepared.json'))


def test_synthetic_cli_plan_pause_resume_verify_compare_report():
    from gym_trading_env.research.fg_replay import read_manifest
    base=Path(os.environ.get('FG_WP08_H_ROOT',ARTIFACTS/'H_cli_01')).absolute();base.mkdir(parents=True)
    p=plan(base/'exam');p['end_day']='2026-01-08';p['baselines']=['ALWAYS_WAIT','ALWAYS_ADD','F3_FROZEN']
    f3=f3_reference();p['actors']={'F3_FROZEN':dict(kind='F3',run=f3['run_dir'],checkpoint=str(Path(f3['run_dir'])/'ckpt/step_000000000004'))}
    config=base/'plan.yaml';config.write_text(json.dumps(p,indent=2)+'\n')
    assert cli(['exam-plan','--config',str(config)],base/'plan.log')==0
    assert cli(['exam-run','--config',str(config),'--stop-at','33'],base/'pause.log')==0,(base/'pause.log').read_text()[-4000:]
    paused=read_manifest(base/'exam/EXAM_RECEIPT.json')
    assert paused['status']=='EXAM_PAUSED' and not (base/'exam/EXAM_REPORT.md').exists()
    assert cli(['exam-resume','--eval-dir',str(base/'exam')],base/'resume.log')==0,(base/'resume.log').read_text()[-4000:]
    complete=read_manifest(base/'exam/EXAM_RECEIPT.json')
    assert complete['status']=='EXAM_COMPLETE' and complete['workers']==3
    for command in ('exam-verify','exam-compare','exam-report'):
        out=base/command
        assert cli([command,'--eval-dir',str(base/'exam'),'--output-dir',str(out)],base/(command+'.log'))==0,(base/(command+'.log')).read_text()[-4000:]
    assert (base/'exam/EXAM_REPORT.md').exists()
    assert cli(['exam-run','--config',str(config)],base/'overwrite.log')!=0
    assert cli(['exam-report','--eval-dir',str(base/'exam'),'--output-dir',str(base/'exam-report')],base/'overwrite_report.log')!=0
    assert not list((base/'exam').glob('**/agent.pkl'))
    assert not list((base/'exam').glob('**/prepared.json'))
    import pickle
    latest=read_manifest(base/'exam/CZCE.FG2701/F3_FROZEN/states/latest.json')
    meta=read_manifest(base/'exam/CZCE.FG2701/F3_FROZEN/states'/latest['node'])
    state=pickle.loads((base/'exam/CZCE.FG2701/F3_FROZEN/states'/meta['payload']).read_bytes())
    assert not {'agent','optimizer','replay','teacher'}&state.keys()
    Path('docs/receipts/wp08/H_cli_summary.json').write_text(json.dumps(dict(
        root=str(base),complete=complete,eval_state_members=sorted(state),
        real_dev_exam='NOT_RUN',teacher_calls=0,training_updates=0),indent=2)+'\n')
