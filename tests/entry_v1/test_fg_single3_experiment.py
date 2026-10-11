import json
from pathlib import Path

import numpy as np
import pytest

from gym_trading_env.research import fg_single3_evaluation as evaluation
from gym_trading_env.research.fg_single3_experiment import resolved_config,training_argv,DREAMER_ROOT


def test_prepared_configuration_matches_official_neural_cli_parser(tmp_path):
    import elements
    from dreamerv3.configuration import read_config_yaml
    from dreamerv3.main import _resolve_seed_protocol
    config=resolved_config()
    definitions=read_config_yaml(DREAMER_ROOT/'dreamerv3/configs.yaml')
    parsed,other=elements.Flags(configs=['defaults']).parse_known(training_argv('/tmp',config)[3:])
    restored=elements.Config(definitions['defaults'])
    for name in parsed.configs:
        restored=restored.update(definitions[name])
    restored=_resolve_seed_protocol(elements.Flags(restored).parse(other))
    assert dict(config)==dict(restored)
    assert isinstance(config.agent.valnorm.limit,float)
    assert config.agent.valnorm.limit==1e-8
    assert config.run.steps==62500
    example=tmp_path/'typed.yaml'
    example.write_text('limit: 1e-8\ncapacity: 1e6\ncontract: "0019"\nliteral: "1e-8"\n')
    assert read_config_yaml(example)==dict(limit=1e-8,capacity=1e6,contract='0019',literal='1e-8')


def test_actor_adapter_passes_only_observations_and_keeps_diagnostics_outside_replay():
    class Neural:
        def init_policy(self,n):
            return 'initial'
        def policy(self,carry,obs,mode):
            assert set(obs)=={'market_seq','position_state','action_mask','reward','is_first','is_last','is_terminal'}
            assert mode=='eval'
            return 'next',{'action':np.array([1],np.int32)},{'p_add':np.array([.8])}
    policy=evaluation.ActorPolicy(Neural())
    obs=dict(market_seq=np.zeros((1,60,18)),position_state=np.zeros((1,9)),action_mask=np.ones((1,2)),
        reward=np.zeros(1),is_first=np.ones(1,bool),is_last=np.zeros(1,bool),is_terminal=np.zeros(1,bool))
    carry,action,out=policy(None,obs)
    assert carry=='next' and action['action'][0]==1 and out=={}
    assert policy.last_diagnostics['p_add']==.8


def test_tp_classification_and_net_pnl_use_actual_exit_not_reward_event(tmp_path,monkeypatch):
    state=dict(attributions=[
        dict(entry_id='tp',exit_time='2024-04-09T09:05:00+08:00',exit_price='995',reason='TAKE_PROFIT',net_trade_pnl='290'),
        dict(entry_id='sl',exit_time='2024-04-09T10:00:00+08:00',exit_price='1030',reason='STOP_LOSS',net_trade_pnl='-1810')],
        fills=[dict(entry_id=k,time='2024-04-09T09:00:00+08:00',price='1000',side='SHORT',reason='ENTRY',committed=True)
            for k in ('tp','sl')],
        reward_events=[dict(entry_id='tp',reason='FIRST_ADVERSE10',reward=-.5),
            dict(entry_id='tp',reason='TAKE_PROFIT',reward=.9),dict(entry_id='sl',reason='STOP_LOSS',reward=-3)],
        total_reward=-2.6,entries={},unrealized_pnl='0',
        equity_history=[dict(equity=300000),dict(equity=298480)])
    (tmp_path/'account_state.json').write_text(json.dumps(state))
    monkeypatch.setattr(evaluation,'trace_manifest',lambda *a,**k:dict(contract='CZCE.FG409'))
    monkeypatch.setattr(evaluation,'iter_sidecar',lambda p:iter([
        dict(decision_executed=False),dict(decision_executed=True,cursor=1,time='t',direction='SHORT',
            action=0,legal_add=True)]))
    result=evaluation.analyze_trace(tmp_path,dict(dev_days=[20240409]),'always_wait')
    assert result['tp5']==1 and result['tp15']==1 and result['sl30']==1
    assert result['tp5_rate']==.5 and result['active_wait']==1
    assert result['net_pnl']==-1520
    assert result['total_reward']==pytest.approx(-2.6)


def test_final_verification_failure_retains_no_pass_report_or_completion(tmp_path,monkeypatch):
    check=evaluation.check_files
    calls=[]
    def fail_late(files):
        calls.append(files)
        check(files)
        if len(calls)==2:
            raise ValueError('late verification failure')
    monkeypatch.setattr(evaluation,'check_files',fail_late)
    with pytest.raises(ValueError,match='late verification'):
        evaluation.publish_report(tmp_path,'PASS report',{},dict(identity_hash='fixture'),{})
    assert not (tmp_path/'completion_manifest.json').exists()
    assert not (tmp_path/'C1_M1_1M_Report.md').exists()
    assert list(tmp_path.glob('C1_M1_1M_Report.failed_*.md'))


def test_training_summary_cannot_invent_effective_actor_updates(tmp_path,monkeypatch):
    import pickle
    from gym_trading_env.research import fg_single3_experiment as experiment
    from embodied.run import fg_checkpoint
    updates=experiment.UPDATES
    checkpoint=tmp_path/'ckpt'/f'step_{updates:012d}'
    checkpoint.mkdir(parents=True)
    (checkpoint/'manifest.json').write_text('{}')
    (checkpoint/'agent.pkl').write_bytes(pickle.dumps(dict(counters=dict(updates=updates))))
    state=dict(last_update=updates,training_positions=updates*1024,actor_effective_updates=1)
    (checkpoint/'training_audit.pkl').write_bytes(pickle.dumps(state))
    log=tmp_path/'train_updates.jsonl';log.write_text('evidence\n')
    summary=dict(state,status='TRAINING_COMPLETE',experiment='C1_M1_1M',identity_hash='fixture',
        checkpoint=str(checkpoint),checkpoint_manifest_sha256=evaluation.digest(checkpoint/'manifest.json'),
        log_sha256=evaluation.digest(log))
    filename=tmp_path/'training_summary.json'
    filename.write_text(json.dumps(summary))
    monkeypatch.setattr(experiment,'require_prepared',lambda _:dict(logdir=str(tmp_path),identity=dict(identity_hash='fixture')))
    monkeypatch.setattr(fg_checkpoint,'validate_bundle',lambda *a:dict(step=updates,protected=True))
    assert evaluation.require_training(tmp_path)[0]['actor_effective_updates']==1
    summary['actor_effective_updates']=2
    filename.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='checkpointed audit'):
        evaluation.require_training(tmp_path)


def test_incomplete_evaluation_is_preserved_and_retried_without_training(tmp_path,monkeypatch):
    output=tmp_path/'always_wait';output.mkdir()
    (output/'failure.log').write_text('previous attempt')
    (tmp_path/'dev_manifest.json').write_text('{}')
    dev=dict(rows=[dict(env_config='env.yaml')],frozen_inputs={},content_hash='fixture')
    monkeypatch.setattr(evaluation,'require_dev',lambda _:dev)
    calls=[]
    def collect(*args,**kwargs):
        calls.append(args)
        output.mkdir()
        (output/'trace.log').write_text('new attempt')
    monkeypatch.setattr(evaluation,'collect_trace',collect)
    monkeypatch.setattr(evaluation,'reconcile_trace',lambda *a,**k:dict(status='PASS'))
    def analyze(*a):
        result=dict(strategy='always_wait')
        (output/'summary.json').write_text(json.dumps(result))
        return result
    monkeypatch.setattr(evaluation,'analyze_trace',analyze)
    assert evaluation.evaluate_strategy(tmp_path,'always_wait',output)==dict(strategy='always_wait')
    assert evaluation.evaluate_strategy(tmp_path,'always_wait',output)==dict(strategy='always_wait')
    assert len(calls)==1
    assert next(tmp_path.glob('always_wait.incomplete_*/failure.log')).read_text()=='previous attempt'


def test_report_preverification_failure_invalidates_existing_completion(tmp_path,monkeypatch):
    from tools import fg_single3_experiment as launcher
    from gym_trading_env.research import fg_single3_experiment as experiment
    (tmp_path/'completion_manifest.json').write_text('old completion')
    (tmp_path/'C1_M1_1M_Report.md').write_text('old PASS')
    monkeypatch.setattr(experiment,'DEFAULT_LOGDIR',tmp_path)
    def reject(_):
        raise ValueError('input hash changed')
    monkeypatch.setattr(launcher,'_report',reject)
    with pytest.raises(ValueError,match='input hash'):
        launcher.report('prepared')
    assert not (tmp_path/'completion_manifest.json').exists()
    assert not (tmp_path/'C1_M1_1M_Report.md').exists()
    assert list(tmp_path.glob('completion_manifest.failed_*.json'))
