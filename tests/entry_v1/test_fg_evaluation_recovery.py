import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research.entry_pipeline import file_hash
from gym_trading_env.research.fg_entry_v1 import write_json


def frozen_validation(tmp_path):
    market=tmp_path/'market.csv';market.write_text('Date,Close\n2024-01-01,1000\n')
    direction=tmp_path/'direction.yaml';direction.write_text('directions: []\n')
    contract=tmp_path/'contract.yaml';contract.write_text('contract: CZCE.FG409\n')
    env=tmp_path/'env.yaml'
    env.write_text(yaml.safe_dump(dict(strategy_mode='entry_only',entry=dict(contract='CZCE.FG409',
        direction_config=str(direction),contract_config=str(contract)),trading=dict(data_path=str(market)))))
    record=dict(contract='CZCE.FG409',role='validation',eligible_for_training=False,path=str(market),
        sha256=file_hash(market),env_config=str(env),env_sha256=file_hash(env),
        contract_config=str(contract),contract_sha256=file_hash(contract))
    data=dict(product='FG',files=[record],direction_path=str(direction),direction_sha256=file_hash(direction))
    data['content_hash']=fingerprint(data);dataset=tmp_path/'dataset.json';write_json(dataset,data)
    fold=dict(validation='CZCE.FG409',validation_env=str(env),datasets=dict(all=dict(manifest=str(dataset),content_hash=data['content_hash'])))
    manifest=dict(folds=[fold],instances=[dict(budget='formal',method='A',seed=0,fold='CZCE.FG409',
        phases=[dict(logdir=str(tmp_path/'train'),steps=100,full_checkpoint=True)])])
    manifest['content_hash']=fingerprint(manifest);path=tmp_path/'manifest.json';write_json(path,manifest)
    return path,env,market


def test_changed_validation_rejected_before_any_simulation(tmp_path,monkeypatch):
    import tools.evaluate_fg_abcd as tool
    manifest,env,market=frozen_validation(tmp_path)
    from gym_trading_env.research.fg_abcd_evaluation import validate_validation_inputs
    hashes=validate_validation_inputs(manifest,env)
    for filename in ('market.csv','env.yaml','direction.yaml','contract.yaml'):
        file=tmp_path/filename
        assert hashes[str(file.resolve())]==file_hash(file)
    market.write_text('changed future prices')
    monkeypatch.setattr(sys,'argv',['evaluate','--manifest',str(manifest),'--env-config',str(env),'--output',str(tmp_path/'output')])
    monkeypatch.setattr(tool,'replay_policy',lambda *a,**k:pytest.fail('Changed frozen input reached simulation'))
    with pytest.raises(ValueError,match='frozen validation input changed'):tool.main()
    assert not (tmp_path/'output').exists()


def test_failed_evaluation_can_retry_without_losing_outputs_or_retraining(tmp_path,monkeypatch):
    import tools.run_fg_abcd as tool
    manifest,env,_=frozen_validation(tmp_path)
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir();(checkpoint/'agent.pkl').write_bytes(b'agent')
    root=tmp_path/'formal_evaluation/CZCE.FG409/A/0'
    root.mkdir(parents=True);(root/'old_partial.txt').write_text('preserve')
    monkeypatch.setattr(sys,'argv',['run','--manifest',str(manifest),'--budget','formal','--execute'])
    monkeypatch.setattr(tool,'run_phase',lambda *a,**k:None)
    monkeypatch.setattr(tool,'phase_metadata',lambda *a:dict(learning_audit=dict(actual_learning=True)))
    monkeypatch.setattr(tool,'checkpoint_path',lambda *a,**k:checkpoint)
    attempts=[]
    def run(cmd,**kwargs):
        if 'evaluate_fg_abcd.py' not in str(cmd):return subprocess.CompletedProcess(cmd,0)
        destination=Path(cmd[cmd.index('--output')+1]);destination.mkdir(parents=True)
        attempts.append(destination)
        (destination/'partial.txt').write_text('attempt')
        if len(attempts)==1:raise subprocess.CalledProcessError(1,cmd)
        write_json(destination/'comparison.json',dict(model_status='EVALUATED',results=[dict(strategy=s) for s in ('always_add','random_add','rule_add','dreamerv3')]))
        for name in ('comparison.md','episode_metrics.csv','matched_random.csv','matched_random_summary.json'):
            (destination/name).write_text('complete')
        return subprocess.CompletedProcess(cmd,0)
    monkeypatch.setattr(tool.subprocess,'run',run)
    with pytest.raises(subprocess.CalledProcessError):tool.main()
    assert not root.exists()
    assert any(p.read_text()=='preserve' for p in root.parent.glob('0.incomplete_*/old_partial.txt'))
    assert (attempts[0]/'partial.txt').exists()
    tool.main()
    assert (root/'evaluation_receipt.json').exists()
    assert (attempts[0]/'partial.txt').exists()
    assert len(attempts)==2
    tool.main()
    assert len(attempts)==2  # Reuse completed evaluation, without running it again.

    (root/'matched_random.csv').write_text('corrupted completed output')
    with pytest.raises(ValueError,match='Completed evaluation output changed'):tool.main()
    assert len(attempts)==2
