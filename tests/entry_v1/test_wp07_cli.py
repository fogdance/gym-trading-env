import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from wp07_fixture import ARTIFACTS,f3_source


ROOT=Path(__file__).resolve().parents[2]


def cli(*args):
    command=[sys.executable,'-B',str(ROOT/'tools/fg_s3_v2_experiment.py'),*map(str,args)]
    result=subprocess.run(command,cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True)
    return result,dict(argv=command,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)


def draft(tmp_path):
    from gym_trading_env.research.fg_s3_e_quality_online import read_settings
    s=read_settings(ROOT/'configs/experiments/fg_s3_v2/e_quality_wp07_online_draft.yaml')
    s['prepared_dir']=str(tmp_path/'prepared');s['execution']['run_dir']=str(tmp_path/'run')
    return s


def test_explicit_draft_exit_files_and_forbidden_side_effects(tmp_path):
    from gym_trading_env.research.fg_replay import read_manifest
    s=draft(tmp_path);config=tmp_path/'draft.json';config.write_text(json.dumps(s))
    result,raw=cli('online-prepare','--config',config,'--output-dir',tmp_path/'diagnostic')
    assert result.returncode==0,raw
    report=read_manifest(tmp_path/'diagnostic/contract_diagnostic.json')
    assert report['status']=='CONTRACT_ONLY' and report['admission']=='NOT_ADMITTED'
    assert not any(report[k] for k in ('authenticated','training_admitted','prepared_created','checkpoint_source_authenticated'))
    assert {'MISSING_SOURCE_RUN','MISSING_SOURCE_CHECKPOINT','LIVE_TRAIN_SOURCE_NOT_VERIFIED'}<=set(report['blockers'])
    assert list((tmp_path/'diagnostic').iterdir())==[tmp_path/'diagnostic/contract_diagnostic.json']
    assert not (tmp_path/'run').exists() and not (tmp_path/'prepared').exists()
    start,start_raw=cli('online-start','--prepared',tmp_path/'diagnostic','--authorize-training')
    assert start.returncode!=0 and not (tmp_path/'run').exists()
    Path('docs/receipts/wp07/H_draft_cli.json').write_text(json.dumps([raw,start_raw],indent=2)+'\n')


def test_missing_empty_path_test_and_synthetic_formal_sources_rejected(tmp_path):
    from gym_trading_env.research.fg_s3_e_quality_online import validate_settings
    s=draft(tmp_path)
    for mutate in (lambda x:x.pop('spec_mode'),lambda x:x['execution'].update(source_run=''),
                   lambda x:x['execution'].update(source_run='../unsafe'),lambda x:x['execution'].update(fold='TEST'),
                   lambda x:x['execution'].update(approved=True)):
        bad=copy.deepcopy(s);mutate(bad)
        with pytest.raises(ValueError):validate_settings(bad)
    s['spec_mode']='formal';config=tmp_path/'formal.json';config.write_text(json.dumps(s))
    result,missing=cli('online-prepare','--config',config)
    assert result.returncode!=0 and not Path(s['prepared_dir']).exists()
    source=f3_source();s['execution'].update(source_run=source['run_dir'],
        source_checkpoint=str(Path(source['run_dir'])/'ckpt/step_000000000004'))
    config.write_text(json.dumps(s));result,synthetic=cli('online-prepare','--config',config)
    assert result.returncode!=0 and 'SYNTHETIC_SOURCE_NOT_FORMAL' in result.stderr
    assert not Path(s['prepared_dir']).exists() and not Path(s['execution']['run_dir']).exists()
    Path('docs/receipts/wp07/H_formal_negative.json').write_text(json.dumps([missing,synthetic],indent=2)+'\n')


def test_independent_rl_authority_cannot_be_self_certified(tmp_path):
    from types import SimpleNamespace
    from embodied.run.fg_s3_e_quality_online_gate import OnlineGates
    from gym_trading_env.research.fg_s3_e_quality_online_checks import formal_mask_policy
    g=OnlineGates('synthetic',formal_mask_policy(f3_source()))
    forged=tmp_path/'forged.json';forged.write_text(json.dumps(dict(approved=True)))
    with pytest.raises(ValueError,match='SEPARATE_RL_AUTHORIZATION_REQUIRED'):
        g.review_rl(None,forged,'0'*64)
    with pytest.raises(ValueError,match='GATE_CLOSED'):g.require_rl(None,authorized=True,review=object())
    g.ready=dict.fromkeys(g.ready,True)
    fake=SimpleNamespace(receipt=dict(identity=dict(source_model=dict(engineering_only=True))))
    with pytest.raises(ValueError,match='FOUNDATION_NOT_TRAINED'):
        g.require_rl(fake,authorized=True,review=object())
    assert not g.formal_admitted


def test_unified_cli_native_probe_report_and_teacher_free_evaluation():
    from gym_trading_env.research.fg_replay import read_manifest
    root=Path(os.environ.get('FG_WP07_H_PROBE',ARTIFACTS/'H_cli_01')).resolve()
    result,raw=cli('online-probe','--output-dir',root,'--execution-mode','ONLINE_SUPERVISED')
    Path('docs/receipts/wp07/H_probe_cli.json').write_text(json.dumps(raw,indent=2)+'\n')
    assert result.returncode==0,raw['stderr'][-6000:]
    probe=read_manifest(root/'probe.json');assert probe['env_steps']==64 and probe['counters']['optimizer_updates']>0
    assert probe['complete_account_replay']['status']=='PASS'
    assert not probe['formal_rl_admitted'] and probe['real_TRAIN_updates']==0
    evaluation=probe['actor_only']
    assert not evaluation['teacher_used'] and evaluation['all_parameter_bytes_equal']
    assert evaluation['complete_account_replay']['status']=='PASS'
    assert evaluation['market_advance']==32
    assert probe['actor_only_deterministic'] and probe['actor_only_repeat_trace']==evaluation['trace_identity']
    result,raw=cli('online-report','--run-dir',root/'run','--output-dir',root/'report')
    Path('docs/receipts/wp07/H_report_cli.json').write_text(json.dumps(raw,indent=2)+'\n')
    assert result.returncode==0,raw['stderr'][-6000:]
    report=read_manifest(root/'report/online_report.json')
    assert report['market_steps']==64 and report['counters']['optimizer_updates']==probe['counters']['optimizer_updates']
    assert report['controlled_small'] and not report['research_admitted']
