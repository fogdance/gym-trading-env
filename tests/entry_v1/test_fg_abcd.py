import copy
import json
import numpy as np
import pandas as pd
import pytest
import yaml

from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.envs.entry_dataset_env import FGEntryDatasetEnv
from gym_trading_env.research.fg_abcd import choose_contracts
from test_entry_contract import make_env


def test_recent_window_excludes_partly_overlapping_lifecycle():
    records=[dict(contract='old',first_day='2022-03-24',last_day='2022-08-10',complete_lifecycle=True),
             dict(contract='recent',first_day='2022-08-11',last_day='2022-12-09',complete_lifecycle=True),
             dict(contract='unfinished',first_day='2023-01-01',last_day='2024-01-01',complete_lifecycle=False)]
    assert [r['contract'] for r in choose_contracts(records,'2024-04-08',24)]==['recent']


def test_rotating_account_bank_preserves_loss_cursor_and_resume(tmp_path,monkeypatch):
    envs=[]
    paths=[]
    for i in range(2):
        directory=tmp_path/str(i);directory.mkdir()
        env=make_env(directory,[(1000,1000,1000,1000),(1000,1000,990,990),
                                (990,990,980,980),(980,980,970,970)],chunk=1)
        envs.append(env)
        paths.append(str(directory/'env.yaml'))
    # Exercise the real financial environments while supplying their in-memory bars.
    frames={p:e._raw.copy() for p,e in zip(paths,envs)}
    from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
    monkeypatch.setattr('gym_trading_env.envs.entry_dataset_env.FixedDirectionEntryEnv',
                        lambda config_path,**kwargs:FixedDirectionEntryEnv(config_path=config_path,df=frames[config_path],**kwargs))
    manifest=dict(product='FG',files=[dict(role='train',eligible_for_training=True,rows=4,env_config=p) for p in paths])
    manifest['content_hash']=fingerprint(manifest)
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(manifest))
    selector=tmp_path/'selector.yaml'
    selector.write_text(yaml.safe_dump(dict(strategy_mode='entry_dataset',role='train',manifest_path=str(mp),
                                           scheduler=dict(mode='least_progress',chunk_bars=1))))
    dataset=FGEntryDatasetEnv(selector)
    dataset.reset(seed=0)
    assert dataset.step(1)[3]
    loss=dataset.env._get_info()['account_net_pnl']
    assert loss<0 and dataset.env.net_lots==3
    dataset.reset()
    assert dataset.index==1 and dataset.env.net_lots==0
    dataset.step(0);dataset.reset()
    assert dataset.index==0 and dataset.env.net_lots==3
    assert dataset.env.current_step==1 and dataset.env._get_info()['account_net_pnl']==loss
    state=json.loads(json.dumps(dataset.export_state(),default=str))
    restored=FGEntryDatasetEnv(selector)
    restored.reset(options={'resume':state})
    a=dataset.step(0);b=restored.step(0)
    for key in a[0]:
        np.testing.assert_array_equal(a[0][key],b[0][key])
    assert a[1:4]==b[1:4]
    assert restored.env.net_lots==3
    assert restored.advanced==dataset.advanced
    changed=copy.deepcopy(state);changed['scheduler']['chunk_bars']=2
    with pytest.raises(ValueError,match='checkpoint mismatch'):
        restored.import_state(changed)
    dataset.close();restored.close()


def test_evaluation_reward_denominator_and_unfinished_account(tmp_path,monkeypatch):
    from gym_trading_env.research.fg_abcd_evaluation import replay_policy,matched_quotas
    directory=tmp_path/'fixture';directory.mkdir()
    prices=[(1000,1000,900,1000),(1000,1000,1000,1000),(1007,1007,1007,1007),
            (990,990,980,980),(980,980,970,970)]
    source=make_env(directory,prices,chunk=2)
    from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
    monkeypatch.setattr('gym_trading_env.research.fg_abcd_evaluation.FixedDirectionEntryEnv',
        lambda config_path:FixedDirectionEntryEnv(config_path=config_path,df=source._raw.copy()))
    config=directory/'env.yaml'
    a=replay_policy(config,tmp_path/'a',strategy='always_add')
    b=replay_policy(config,tmp_path/'b',strategy='always_add')
    assert a==b and a['reward_reconcile_error']==0
    assert a['total_reward']==pytest.approx(60/62)
    first_trade=pd.read_csv(tmp_path/'a/trades.csv').iloc[0]
    assert first_trade.MAE_points==0  # Signal bar's pre-entry low 900 is excluded.
    assert first_trade.duration_trading_bars==2
    assert a['open_lots_at_end']>0 and a['max_floating_loss']>0
    assert a['max_equity_drawdown']>0 and a['right_censored_batches']>0
    c=replay_policy(config,tmp_path/'c',strategy='rule_add')
    assert c['eligible_trading_days']==a['eligible_trading_days']==1
    assert pd.read_csv(tmp_path/'c/trades.csv').empty
    assert 'entry_id' in pd.read_csv(tmp_path/'c/trades.csv').columns
    quotas=matched_quotas(tmp_path/'a/policy_actions.csv')
    d=replay_policy(config,tmp_path/'d',quotas=quotas,seed=0)
    assert d['ADD_fill']<=sum(quotas.values())
    assert d['match_status'] in ('exact_match','partial_match')


def test_unfilled_main_exit_preserves_loss_and_is_terminal_reward_failure(tmp_path,monkeypatch):
    from gym_trading_env.research.fg_abcd_evaluation import replay_policy
    directory=tmp_path/'fixture';directory.mkdir()
    source=make_env(directory,[(1000,1000,1000,1000),(1000,1000,990,990),
                              (990,990,980,980)],main_exit='2026-01-05 09:33')
    source._raw.iloc[-1,source._raw.columns.get_loc('Volume')]=0
    from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
    monkeypatch.setattr('gym_trading_env.research.fg_abcd_evaluation.FixedDirectionEntryEnv',
        lambda config_path:FixedDirectionEntryEnv(config_path=config_path,df=source._raw.copy()))
    result=replay_policy(directory/'env.yaml',tmp_path/'result',strategy='always_add')
    assert result['open_lots_at_end']>0 and result['net_equity_change']<0
    assert result['total_reward']==0 and result['forced_exit_unfilled_batches']>0
    assert result['right_censored_batches']==0


def test_matched_workers_spawn_independent_real_accounts(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from tools.evaluate_fg_abcd import matched_worker
    from gym_trading_env.research.fg_abcd_evaluation import slot_key
    source=make_env(tmp_path,[(1000,1000,1000,1000),(1000,1000,990,990),(990,990,980,980)])
    market=tmp_path/'market.csv';source._raw.to_csv(market,index_label='Date')
    config=tmp_path/'env.yaml';raw=yaml.safe_load(config.read_text())
    raw['trading']['data_path']=str(market);config.write_text(yaml.safe_dump(raw))
    quotas={slot_key(source,0):2}
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(matched_worker,(str(config),str(tmp_path/f'worker_{seed}'),quotas,seed)) for seed in (0,1)]
        results=[f.result() for f in futures]
    assert [r['seed'] for r in results]==[0,1]
    assert all(r['ADD_fill']==2 and r['open_lots_at_end']==6 for r in results)
    assert all(r['total_reward']==0 and r['net_equity_change']<0 for r in results)


def test_sequential_dataset_refuses_to_discard_unfilled_terminal_loss(tmp_path,monkeypatch):
    source=make_env(tmp_path,[(1000,1000,1000,1000),(1000,1000,990,990),
                              (990,990,980,980)],main_exit='2026-01-05 09:33')
    source._raw.iloc[-1,source._raw.columns.get_loc('Volume')]=0
    from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
    monkeypatch.setattr('gym_trading_env.envs.entry_dataset_env.FixedDirectionEntryEnv',
        lambda config_path:FixedDirectionEntryEnv(config_path=config_path,df=source._raw.copy()))
    manifest=dict(product='FG',files=[dict(role='train',eligible_for_training=True,
                                         env_config=str(tmp_path/'env.yaml'))])
    manifest['content_hash']=fingerprint(manifest)
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    selector=tmp_path/'selector.yaml';selector.write_text(yaml.safe_dump(dict(
        strategy_mode='entry_dataset',role='train',manifest_path=str(path))))
    dataset=FGEntryDatasetEnv(selector);dataset.reset()
    dataset.step(1);dataset.step(0)
    assert dataset.env.status=='MAIN_ROLL' and dataset.env.net_lots==3
    assert dataset.step(0)[3] and dataset.env.status=='DATA_END'
    before=dataset.export_state()
    with pytest.raises(RuntimeError,match='unfinished account'):
        dataset.reset()
    assert dataset.export_state()==before
    assert dataset.env.total_reward==0 and dataset.env._get_info()['account_net_pnl']<0
    dataset.close()


def test_sequential_coverage_counts_early_exit_prefixes_without_claiming_full_passes():
    from tools.run_fg_abcd import account_progress
    records=[dict(contract='one',rows=101),dict(contract='two',rows=51)]
    state=dict(schema=1,index=0,completed=[
        dict(contract='one',advanced_bars=20),dict(contract='two',advanced_bars=50)],
        account=dict(contract='one',current_step=10))
    one,two=account_progress(state,records)
    assert one['advanced_bars']==30 and one['full_passes']==0
    assert one['prefix_counts']=={20:1,10:1} and one['saved_account']
    assert two['advanced_bars']==50 and two['full_passes']==1 and not two['saved_account']
    state['completed'][0].pop('advanced_bars')
    with pytest.raises(ValueError,match='actual coverage counters'):
        account_progress(state,records)


def shared_plans(tmp_path):
    from gym_trading_env.research.fg_abcd import plan_phases
    cfg=yaml.safe_load((__import__('pathlib').Path(__file__).resolve().parents[2]/'configs/fg_entry_v1/abcd_comparison.yaml').read_text())
    cfg['log_root']=str(tmp_path/'logs')
    datasets={k:dict(manifest=str(tmp_path/f'{k}.json'),selector=str(tmp_path/f'{k}.yaml')) for k in ('all','recent')}
    result={(budget,method):dict(fold='CZCE.FG409',budget=budget,method=method,seed=0,
        phases=plan_phases(cfg,'CZCE.FG409',datasets,budget,method,0)) for budget in ('pilot','formal') for method in 'ABC'}
    return result


def test_shared_graph_reuses_pilot_prefix_but_separates_c_branches(tmp_path):
    plans=shared_plans(tmp_path)
    a=plans['formal','A']['phases'][0];b=plans['formal','B']['phases'][0]
    source,recent=plans['formal','C']['phases']
    assert a['logdir']==source['logdir']==plans['pilot','A']['phases'][0]['logdir']
    assert b['logdir']==plans['pilot','B']['phases'][0]['logdir']
    assert source['shared_source'] and source['steps']==700000 and a['steps']==1000000
    assert recent['steps']==300000 and recent['total_step_offset']==700000
    assert recent['logdir']!=plans['pilot','C']['phases'][-1]['logdir']
    assert a['checkpoint']['key_steps']==[70000,100000,700000,1000000]


def test_dry_run_prints_three_unique_formal_lines_without_running_training(tmp_path,monkeypatch,capsys):
    from tools.run_fg_abcd import main
    plans=shared_plans(tmp_path)
    manifest=dict(instances=[plans['formal',m] for m in 'ABC'],folds=[],shared_prefix=True)
    manifest['content_hash']=fingerprint(manifest)
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    monkeypatch.setattr('sys.argv',['run_fg_abcd.py','--manifest',str(path),'--budget','formal'])
    monkeypatch.setattr('subprocess.run',lambda *a,**k:pytest.fail('Dry-run must not launch any subprocess'))
    main();output=capsys.readouterr().out
    assert output.count('-m dreamerv3.main')==3
    assert '# Reuse protected shared node 700000' in output
    assert '--run.fg_report_every_steps 5000' in output and '--run.fg_save_every_steps 10000' in output
    assert '--create-transfer' in output and 'C_recent_from_700000' in output


def test_phase_status_does_not_confuse_pilot_and_formal_shared_directory(tmp_path):
    from tools.run_fg_abcd import phase_record_path,phase_metadata
    plans=shared_plans(tmp_path)
    pilot=plans['pilot','A'];formal=plans['formal','A']
    path=phase_record_path(pilot,pilot['phases'][0]);path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(status='COMPLETED_BUDGET')))
    assert phase_metadata(pilot,pilot['phases'][0])['status']=='COMPLETED_BUDGET'
    assert phase_metadata(formal,formal['phases'][0])['status']=='NOT_RUN'
