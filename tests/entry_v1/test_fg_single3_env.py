from dataclasses import asdict
from decimal import Decimal
import json

import numpy as np
import pytest
import yaml

from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from gym_trading_env.rewards.single3_reward import Single3RewardConfig
from test_entry_contract import make_env


def single_env(tmp_path, prices=None, side="LONG", candidate="C2", **kwargs):
    old = make_env(tmp_path, prices, side=side, **kwargs)
    config_path = tmp_path / "env.yaml"
    raw = yaml.safe_load(config_path.read_text())
    raw['entry'].update(protocol='fg_single3_v1', reward=asdict(Single3RewardConfig.for_candidate(candidate)))
    raw['trading'].update(max_long_position=3, max_short_position=3, stop_loss_enabled=True)
    config_path.write_text(yaml.safe_dump(raw))
    return FixedDirectionEntryEnv(df=old._raw.copy(), config_path=str(config_path))


@pytest.mark.parametrize('side', ['LONG', 'SHORT'])
def test_single_capacity_sl_double_touch_and_release(tmp_path, side):
    env = single_env(tmp_path, [(1000,1000,1000,1000), (1000,1008,965,1007),
                                (1000,1000,1000,1000)], side=side)
    obs, reward, terminal, _, _ = env.step(1)
    event = env.attributions[0]
    assert event['reason'] == 'STOP_LOSS' and event['ambiguous_tp_sl']
    assert event['same_entry_bar'] and event['closed_quantity'] == 3
    assert Decimal(event['exit_price']) == (970 if side == 'LONG' else 1030)
    assert reward == -4 and not terminal and env.status == 'ACTIVE'
    assert env.net_lots == 0 and obs['action_mask'][1] == 1
    assert [x['reason'] for x in env.reward_events] == ['FIRST_ADVERSE10', 'STOP_LOSS']
    env._assert_ledger_conservation()


def test_pending_no_liquidity_recovery_mask_and_resume(tmp_path):
    env = single_env(tmp_path, [(1000,1000,1000,1000), (1000,1000,965,980),
                                (1010,1010,1010,1010)], chunk=1)
    env._raw.loc[env._raw.index[1], 'Volume'] = 5  # three opened, two cannot close.
    obs, reward, _, truncated, _ = env.step(1)
    assert truncated and env.net_lots == 3 and not env.attributions
    assert reward == -1
    np.testing.assert_array_equal(obs['position_state'][[1,2,3]], [1,1,1])
    np.testing.assert_array_equal(obs['position_state'][6:], [0,0,0])
    assert obs['action_mask'][1] == 0
    state = json.loads(json.dumps(env.export_state()))
    env.reset(options={'resume': state})
    assert env.export_state() == state
    env.reset()  # continuation does not rescore the repeated observation.
    assert env.total_reward == -1
    obs, reward, terminal, _, info = env.step(1)
    assert info['reject_reason'] == 'MASKED_ADD' and not terminal
    assert env.net_lots == 0 and reward == -3
    assert env.attributions[0]['sl_pending_retry']
    assert Decimal(env.attributions[0]['exit_price']) == 1010
    assert len([e for e in env.reward_events if e['reason']=='FIRST_ADVERSE10']) == 1


@pytest.mark.parametrize('side', ['LONG', 'SHORT'])
def test_gap_uses_actual_quote_and_real_fees(tmp_path, side):
    env = single_env(tmp_path, [(1000,1000,1000,1000), (1000,1000,1000,1000),
                                (950,951,949,950)], side=side, fees=3, spread=1)
    env.step(1)
    before = env._get_obs()['position_state']
    assert before[1] == 1 and before[3] == 1
    signed_sl_distance = 35 - 5 * before[4]
    entry = next(iter(env.entries.values()))
    sign = 1 if side == 'LONG' else -1
    assert signed_sl_distance == pytest.approx(sign*(float(env.current_price)-float(entry['entry_price']))+30)
    _, reward, *_ = env.step(0)
    event = env.attributions[0]
    expected = 949 if side=='LONG' else 1051
    assert Decimal(event['exit_price']) == expected
    assert event['gap_slippage_points'] == 22
    assert event['open_fee'] == '9' and event['close_fee'] == '9'
    assert Decimal(event['gross_trade_pnl']) == -3120
    assert Decimal(event['net_trade_pnl']) == -3138
    assert reward == -4
    assert env._calculate_equity() == Decimal(300000)-3138


def test_tp_reward_and_other_exit_proportion(tmp_path):
    env = single_env(tmp_path, [(1000,1000,1000,1000), (1000,1000,1000,1000),
                                (1000,1007,1000,1007)], fees=3)
    env.step(1)
    _, reward, *_ = env.step(0)
    assert reward == pytest.approx(60/62)
    assert env.attributions[0]['net_trade_pnl'] == '282'
    other_path = tmp_path / 'other'
    other_path.mkdir()
    other = single_env(other_path, [(1000,1000,1000,1000), (1000,1000,1000,1000),
                                    (995,995,995,995)], fees=3, main_exit='2026-01-05 09:33')
    other.step(1)
    _, reward, terminal, *_ = other.step(0)
    assert terminal and other.attributions[0]['reason'] == 'MAIN_ROLL'
    assert reward == pytest.approx(-3*318/1800)
    assert other.attributions[0]['net_trade_pnl'] == '-318'


def test_partial_exit_rejected_before_account_mutation(tmp_path):
    env = single_env(tmp_path)
    env.step(1)
    before = env.export_state()
    with pytest.raises(ValueError, match='atomic full'):
        env.settle_exit(next(iter(env.entries)), 970, env._raw.index[1], 'STOP_LOSS', quantity=1)
    assert env.export_state() == before
    _, _, _, _, info = env.step(1)
    assert info['reject_reason'] == 'MASKED_ADD' and env.net_lots == 3


def test_metadata_and_right_censor_do_not_invent_exit(tmp_path):
    env = single_env(tmp_path, [(1000,1000,1000,1000), (1000,1000,1000,1000),
                                (1000,1000,960,970)], zero_duration_indices=(2,))
    env.step(1)
    _, reward, *_ = env.step(0)
    assert reward == 0 and env.net_lots == 3 and not env.attributions
    assert not next(iter(env.entries.values()))['sl_pending']
    _, reward, terminal, truncated, info = env.step(0)
    assert reward == 0 and not terminal and truncated and info['status']=='DATA_END'
    assert not env.attributions and env.net_lots == 3


def test_single_checkpoint_cannot_resume_legacy_or_another_reward(tmp_path):
    env = single_env(tmp_path)
    env.step(1)
    state = env.export_state()
    other_path = tmp_path/'other'
    other_path.mkdir()
    other = single_env(other_path, candidate='C3')
    with pytest.raises(ValueError, match='Incompatible'):
        other.import_state(state)
    legacy_path = tmp_path/'legacy'
    legacy_path.mkdir()
    legacy = make_env(legacy_path)
    with pytest.raises(ValueError, match='Incompatible'):
        legacy.import_state(state)


def test_margin_call_has_priority_over_sl_and_tp(tmp_path):
    env=single_env(tmp_path, [(50000,50000,50000,50000), (50000,50010,49960,50010)])
    _, reward, terminal, *_=env.step(1)
    assert terminal and env.attributions[0]['reason']=='MARGIN_CALL'
    assert not any(e['reason']=='STOP_LOSS' for e in env.reward_events)
    assert reward==-4 and env.net_lots==0


def test_atomic_sl_rejection_keeps_slot_and_retries_once(tmp_path, monkeypatch):
    from gym_trading_env.envs.action import ForexCode
    env=single_env(tmp_path, [(1000,1000,1000,1000), (1000,1000,965,980),
                            (1000,1000,1000,1000)])
    original=env._long_close
    monkeypatch.setattr(env,'_long_close',lambda *a,**k:ForexCode.ERROR_NO_ENOUGH_MONEY)
    _,reward,*_=env.step(1)
    assert reward==-1 and env.net_lots==3 and not env.attributions
    assert next(iter(env.entries.values()))['sl_pending']
    monkeypatch.setattr(env,'_long_close',original)
    _,reward,*_=env.step(0)
    assert reward==-3 and env.net_lots==0 and len(env.attributions)==1
    assert len(env.processed_fills)==1


def test_main_roll_overrides_pending_sl_without_stale_metadata(tmp_path):
    env=single_env(tmp_path,[(1000,1000,1000,1000),(1000,1000,965,980),
                            (1010,1010,990,990)],main_exit='2026-01-05 09:33')
    env._raw.loc[env._raw.index[1],'Volume']=5
    env.step(1)
    env.step(0)
    event=env.attributions[0]
    assert event['reason']=='MAIN_ROLL'
    assert event['price_reference_time']==str(env._raw.index[2])
    assert 'sl_pending_retry' not in event


def test_duplicate_fill_rejected_before_mutation(tmp_path):
    env=single_env(tmp_path)
    env.step(1)
    entry=next(iter(env.entries))
    env.settle_exit(entry,970,env._raw.index[1],'STOP_LOSS',fill_id='unique')
    before=env.export_state()
    with pytest.raises(ValueError, match='Duplicate'):
        env.settle_exit(entry,970,env._raw.index[1],'STOP_LOSS',fill_id='unique')
    assert env.export_state()==before


@pytest.mark.parametrize('other_kind',['legacy','C3'])
def test_dataset_rejects_same_shape_different_semantics(tmp_path, other_kind):
    from gym_trading_env.envs.entry_dataset_env import FGEntryDatasetEnv
    from gym_trading_env.envs.entry_contract import fingerprint
    first=tmp_path/'first';second=tmp_path/'second'
    first.mkdir();second.mkdir()
    source=single_env(first)
    source._raw.to_csv(first/'market.csv',index_label='Date')
    raw=yaml.safe_load((first/'env.yaml').read_text())
    raw['trading']['data_path']=str(first/'market.csv')
    (first/'env.yaml').write_text(yaml.safe_dump(raw))
    if other_kind=='legacy':
        make_env(second)
    else:
        single_env(second,candidate=other_kind)
    manifest=dict(product='FG',files=[dict(role='train',eligible_for_training=True,
        env_config=str(path/'env.yaml')) for path in (first,second)])
    (tmp_path/'manifest.json').write_text(json.dumps(dict(manifest,content_hash=fingerprint(manifest))))
    (tmp_path/'dataset.yaml').write_text(yaml.safe_dump(dict(strategy_mode='entry_dataset',
        role='train',manifest_path=str(tmp_path/'manifest.json'))))
    with pytest.raises(ValueError,match='mixes'):
        FGEntryDatasetEnv(str(tmp_path/'dataset.yaml'))
