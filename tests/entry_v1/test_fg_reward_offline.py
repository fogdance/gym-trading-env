"""Offline adapter contracts using the existing real Env and small causal bars."""
import json
from types import SimpleNamespace

import pytest

from tests.entry_v1.test_entry_contract import make_env
from tools import fg_reward_offline as tool
from gym_trading_env.research.fg_reward_study import candidate_grid, verify_direct, validate_saved_replay


def arguments(tmp_path, policy='always_add'):
    return SimpleNamespace(trace=None,manifest=None,env_config=tmp_path/'env.yaml',eval_dir=None,
        policy=policy,seed=0,out=tmp_path/'scored',beta=.2,rho=.3,lam=.0002,alpha=.05,gamma=332/333)


def test_frozen_reward_identity_cannot_be_replaced_with_same_sum(tmp_path, monkeypatch):
    factory=lambda **kwargs: make_env(tmp_path, prices=[(1000,1000,1000,1000),
        (1000,1000,1000,1000),(1000,1006,1000,1006),(1000,1000,1000,1000)],
        main_exit='2026-01-05 09:34:00')
    env=factory()
    actions=[]
    while not env.terminated:
        signal=str(env._raw.index[env.current_step])
        action=int(env._get_obs()['action_mask'][1])
        env.step(action)
        actions.append(dict(signal_timestamp=signal,action=action))
    frozen=tmp_path/'original';frozen.mkdir()
    tool._write_csv(frozen/'policy_actions.csv',actions)
    tool._write_csv(frozen/'fills.csv',env.fills)
    assert env.reward_events
    altered=[dict(r,entry_id='incorrect:identity') for r in env.reward_events]
    tool._write_csv(frozen/'reward_events.csv',altered)
    (frozen/'account_state.json').write_text(json.dumps(env.export_state()))
    (frozen/'summary.json').write_text(json.dumps(env._get_info()))
    monkeypatch.setattr('gym_trading_env.envs.entry_env.FixedDirectionEntryEnv',factory)
    args=SimpleNamespace(manifest=None,env_config=tmp_path/'env.yaml',eval_dir=frozen,
                         policy=None,seed=0,out=None)
    scorer=tool.lib.FGRewardScorer(tool.lib.CandidateConfig())
    with pytest.raises(RuntimeError,match='Reward.*entry_id'):
        tool._replay_existing_env(args,scorer)


def test_real_env_capacity_tp_chunk_restore_and_grid_agree(tmp_path,monkeypatch):
    factory=lambda **kw:make_env(tmp_path,prices=[(1000,1000,1000,1000),
        (1000,1000,1000,1000),(990,990,990,990),(990,996,990,996),
        (990,990,990,990),(990,990,990,990)],chunk=2,main_exit='2026-01-05 09:36:00')
    monkeypatch.setattr('gym_trading_env.envs.entry_env.FixedDirectionEntryEnv',factory)
    args=arguments(tmp_path)
    scorer=tool.lib.FGRewardScorer(tool.lib.CandidateConfig(beta=.2,alpha=.05,lam=.0002))
    rows,metadata=tool._replay_existing_env(args,scorer)
    assert [r['net_lots'] for r in rows]==[3,6,3,6,0]
    assert [r['capacity_basis'] for r in rows]==[.25,1,1,1,1]
    assert metadata['chunk_restores_verified']==2
    assert scorer.summary()['tp_entries']==1 and scorer.summary()['closed_entries']==3
    grid=candidate_grid(dict(beta=[0,.05,.1,.2],lam=[0,.00005,.0002,.001],
                             alpha=[0,.05],rho=.3,gamma=332/333))
    assert len(grid)==32
    for config in grid:
        assert verify_direct(metadata['_artifacts']['events'],rows,config)['max_step_error']<1e-12


def test_opportunity_uses_execution_clock_at_session_gap(tmp_path):
    env=make_env(tmp_path,times=['2026-01-05 09:31','2026-01-05 10:31'],
        prices=[(1000,1000,1000,1000)]*2,liquidation='2026-01-05 10:00:00',
        calendar=[dict(start='2026-01-05 09:30',end='2026-01-05 09:31'),
                  dict(start='2026-01-05 10:30',end='2026-01-05 10:31')])
    assert env._raw.index[0]<env.terms.last_open
    assert not tool.opportunity_clock(env,1)


def test_delay_waits_five_valid_minutes_and_skips_metadata(tmp_path,monkeypatch):
    factory=lambda **kw:make_env(tmp_path,prices=[(1000,1000,1000,1000)]*15,
        zero_duration_indices=(3,),main_exit='2026-01-05 09:45:00')
    monkeypatch.setattr('gym_trading_env.envs.entry_env.FixedDirectionEntryEnv',factory)
    scorer=tool.lib.FGRewardScorer(tool.lib.CandidateConfig())
    rows,metadata=tool._replay_existing_env(arguments(tmp_path,'delay5'),scorer)
    fills=[r for r in metadata['_artifacts']['fills'] if r['reason']=='ENTRY']
    assert len(fills)==2
    assert [r['time'] for r in fills]==['2026-01-05 09:37:00+08:00','2026-01-05 09:43:00+08:00']
    assert rows[2]['valid_bar'] is False and rows[2]['capacity_basis']==0
    assert tool.delay_counter(4,legal=False,eligible=True,opened=False)==0


@pytest.mark.parametrize('forced',[False,True])
def test_cutoff_and_unfilled_force_exit_remain_distinct(tmp_path,monkeypatch,forced):
    def factory(**kw):
        env=make_env(tmp_path,prices=[(1000,1000,1000,1000)]*4,
                     main_exit='2026-01-05 09:34:00' if forced else None)
        if forced:
            env._raw.loc[env._raw.index[-1],'Volume']=0
        return env
    monkeypatch.setattr('gym_trading_env.envs.entry_env.FixedDirectionEntryEnv',factory)
    summary=tool.run_single(arguments(tmp_path))
    rows=tool._read_csv(tmp_path/'scored/trade_components.csv')
    assert len(rows)==2 and summary['active_entries']==2
    assert {r['status'] for r in rows}==({'terminal_failure_unfilled'} if forced else {'right_censored'})
    state=json.loads((tmp_path/'scored/account_state.json').read_text())
    assert len(state['entries'])==2
    with pytest.raises(FileExistsError):tool.run_single(arguments(tmp_path))


def test_replay_reuse_rejects_changed_frozen_input(tmp_path,monkeypatch):
    monkeypatch.setattr('gym_trading_env.envs.entry_env.FixedDirectionEntryEnv',
        lambda **kw:make_env(tmp_path,main_exit='2026-01-05 09:38:00'))
    args=arguments(tmp_path)
    summary=tool.run_single(args)
    # Synthetic metadata isolates reuse validation; this is not a market audit.
    source=tmp_path/'frozen.txt';source.write_text('frozen')
    frozen={str(source):tool._hash(source)}
    meta=summary['input_provenance']
    meta.update(policy='logged_actions',frozen_manifest_verified=True,frozen_input_hashes=frozen)
    tool._write_json(args.out/'summary.json',summary)
    reference={k:getattr(args,k) for k in ('beta','rho','lam','alpha','gamma')}
    validate_saved_replay(args.out,reference,frozen)
    source.write_text('changed')
    with pytest.raises(ValueError,match='reuse source/input changed'):
        validate_saved_replay(args.out,reference,frozen)
