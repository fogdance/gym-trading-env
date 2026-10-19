import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from test_wp08_contract import plan
from wp08_fixture import ARTIFACTS,RELEASE,f3_reference,native_source


def test_baseline_rng_delay_visible_history_and_no_oracle():
    from gym_trading_env.research.fg_s3_e_quality_exam_compare import ExamBaseline
    obs=dict(action_mask=np.array([[1,1]],np.float32),is_last=np.array([False]))
    p=ExamBaseline('RANDOM_P10',seed=202,contract='CZCE.FG2701',random_seed=101)
    before=p.save();a=[p(None,obs)[1]['action'].tolist() for _ in range(32)]
    p.load(before);assert a==[p(None,obs)[1]['action'].tolist() for _ in range(32)]
    delay=ExamBaseline('DELAY_15',seed=202,contract='CZCE.FG2701')
    row=dict(cursor=0,advanced=True,fills=[],exits=[],lots_after=0,legal_add_before=True,valid_bar=True)
    for i in range(15):
        assert delay(None,obs)[1]['action'][0]==0
        delay.observe_row({**row,'cursor':i},legal_after=True)
    assert delay(None,obs)[1]['action'][0]==1
    with pytest.raises(ValueError,match='NO_ORACLE'):ExamBaseline('P4_ORACLE',seed=202,contract='CZCE.FG2701')


def test_same_source_full_interval_baselines_and_f3_f4():
    from gym_trading_env.research.fg_s3_e_quality_exam_compare import policy_jobs,compare_sessions
    base=Path(os.environ.get('FG_WP08_E_ROOT',ARTIFACTS/'E_compare_01')).absolute();base.mkdir(parents=True)
    p=plan(base);f3=f3_reference();p['actors']={'F3_FROZEN':dict(kind='F3',run=f3['run_dir'],checkpoint=str(Path(f3['run_dir'])/'ckpt/step_000000000004'))}
    for mode,label in [('SHADOW','F4_SHADOW'),('ONLINE_SUPERVISED','F4_BC'),('ONLINE_RL_CONTROLLED','F4_RL')]:
        root=RELEASE/'Release_G_native_01'/mode/'continuous'
        p['actors'][label]=dict(kind='F4',run=str(root),checkpoint=str(root/'ckpt/step_000000000003'))
    p['baselines']+=list(p['actors']);config=base/'plan.json';config.write_text(json.dumps(p,indent=2)+'\n')
    jobs=policy_jobs(p)
    def work(job):
        argv=[sys.executable,'-B',str(Path(__file__).with_name('wp08_compare_worker.py')),str(config),json.dumps(job)]
        log=base/(job['label']+'.log')
        with log.open('x') as stream:r=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT)
        log.with_suffix('.command.json').write_text(json.dumps(dict(argv=argv,exit_code=r.returncode),indent=2)+'\n')
        assert r.returncode==0,log.read_text()[-5000:]
        return base/job['contract']/job['label']
    with concurrent.futures.ThreadPoolExecutor(min(os.cpu_count() or 1,len(jobs))) as pool:roots=list(pool.map(work,jobs))
    source=native_source(ARTIFACTS/'C_native_04/source')
    report=compare_sessions([(r,source) for r in roots],base/'comparison')
    assert report['account_count']==10
    assert report['independent_contract_days']==4
    assert report['independent_contracts']==1
    wait=next(s for s in report['summaries'] if s['policy']=='ALWAYS_WAIT')
    assert wait['entries']==0 and wait['tp_rate'] is None and wait['entry_evidence']=='NO_ENTRY_SKILL_EVIDENCE'
    assert all(s['account_reconciled'] and s['illegal_add']==0 for s in report['summaries'])
    actor_pairs=[r for r in report['paired_deltas'] if r['policy']=='F4_RL']
    assert {r['reference'] for r in actor_pairs}=={'ALWAYS_WAIT','ALWAYS_ADD','DELAY_15',
        'RANDOM_P10_seed101','RANDOM_P10_seed202','RANDOM_P10_seed303','F3_FROZEN'}
    assert all('entry_density_delta' in r and 'mae_15_mean_delta' in r for r in actor_pairs)
    for root in roots:
        assert (root/'MODEL_READONLY_RECEIPT.json').exists()
    Path('docs/receipts/wp08/E_fair_comparison.json').write_text(json.dumps(dict(
        status=report['status'],independent_contract_days=report['independent_contract_days'],
        account_count=report['account_count'],summaries=report['summaries'],comparison_path=str(base/'comparison')),indent=2)+'\n')
    with pytest.raises(ValueError,match='NEW'):compare_sessions([(r,source) for r in roots],base/'comparison')


def test_shared_eval_seed_preserves_original_model_seed_and_parameters():
    from gym_trading_env.research.fg_replay import read_manifest
    root=ARTIFACTS/'E_compare_02/CZCE.FG2701'
    receipts=[read_manifest(root/name/'MODEL_READONLY_RECEIPT.json') for name in ('F3_FROZEN','F4_SHADOW','F4_BC','F4_RL')]
    assert {r['evaluation_seed'] for r in receipts}=={202}
    assert receipts[0]['model_config']['seed']==202
    assert all(r['model_config']['seed']==303 for r in receipts[1:])
    assert all(read_manifest(root/name/'SESSION_RECEIPT.json')['frozen']['all_parameter_bytes_equal']
        for name in ('F3_FROZEN','F4_SHADOW','F4_BC','F4_RL'))
