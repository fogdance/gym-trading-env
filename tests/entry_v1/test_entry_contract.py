from decimal import Decimal
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from gym_trading_env.envs.entry_contract import DirectionSchedule, timestamp
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from gym_trading_env.rewards.fast_take_profit_reward import FastTakeProfitReward


def make_env(tmp_path, prices=None, side="LONG", chunk=0, intervals=None,
             margin_rate=.1, fees=0, spread=0, liquidation="2026-01-06 15:00:00",
             times=None, calendar=None):
    prices = prices or [(1000, 1000, 1000, 1000)]*8
    if side == "SHORT":
        prices = [(2000-o, 2000-l, 2000-h, 2000-c) for o,h,l,c in prices]
    idx = pd.date_range("2026-01-05 09:31", periods=len(prices), freq="min") if times is None else pd.DatetimeIndex(times)
    df = pd.DataFrame(prices, columns=["Open", "High", "Low", "Close"], index=idx)
    df["Volume"], df["OpenInterest"] = 100, 1000
    df["bob"] = df.index - pd.Timedelta(minutes=1)
    directions = dict(timezone="Asia/Shanghai", intervals=intervals if intervals is not None else [
        dict(contract="CZCE.FG2701", start="2026-01-05", end="2026-01-07", direction=side)])
    meta = dict(contract="CZCE.FG2701", tick_size=1, lot_size=20,
                last_open_at=liquidation, liquidate_at=liquidation,
                sessions=calendar or [dict(start="2026-01-05 09:30", end="2026-01-05 10:15")],
                terms=[dict(start="2026-01-01", end="2026-01-10", margin_rate=margin_rate,
                            open_fee=fees, close_fee=fees, spread=spread, slippage=0)])
    for name, data in (("directions.yaml", directions), ("contract.yaml", meta)):
        (tmp_path/name).write_text(yaml.safe_dump(data))
    config = dict(strategy_mode="entry_only", entry=dict(contract="CZCE.FG2701", direction_config="directions.yaml",
                    contract_config="contract.yaml", chunk_steps=chunk),
                  trading=dict(is_future=True, future_symbol="CZCE.FG2701", currency_pair="FG2701",
                               initial_balance=300000, trade_lot=3, max_long_position=6, max_short_position=6,
                               obs_feature_mode="obs", live_mode=False, data_tz="Asia/Shanghai"),
                  training=dict(window_size=60), debug=dict(log_level="CRITICAL"))
    path=tmp_path/"env.yaml"
    path.write_text(yaml.safe_dump(config))
    return FixedDirectionEntryEnv(df=df, config_path=str(path))


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_zero_three_six_three_six_independent_tp_and_reentry(tmp_path, side):
    env = make_env(tmp_path, [(1000,1000,1000,1000), (1000,1000,1000,1000),
        (990,990,990,990), (990,996,990,996), (990,990,990,990)], side)
    actual=[env.net_lots]
    rewards=[]
    for action in [1,1,0,1]:
        _, r, _, _, _ = env.step(action)
        actual.append(env.net_lots)
        rewards.append(r)
    assert actual == [0,3,6,3,6]
    assert rewards == [0,0,60/62,0]
    assert env.attributions[0]["entry_id"].endswith(":2")
    assert len(env.reward_events) == 1
    assert env.ledger.total_balance() == Decimal(300000)
    assert env._get_info()["account_net_pnl"] == -300 # +300 realized, -600 first-entry MTM


@pytest.mark.parametrize("minutes", [1,2,5])
@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_fast_tp_requires_after_entry_execution_and_correct_natural_time(tmp_path, minutes, side):
    prices=[(1000,1000,1000,1000)]*(minutes+2)
    prices[minutes]=(1000,1006,1000,1006)
    env=make_env(tmp_path, prices, side)
    rewards=[env.step(1)[1]]
    rewards.extend(env.step(0)[1] for _ in range(minutes-1))
    assert sum(rewards) == 60/(60+minutes)
    entry_fill, tp_fill=env.fills
    assert timestamp(entry_fill["time"]) == env.bobs[1]
    assert timestamp(tp_fill["time"]) == env._raw.index[minutes]
    assert env.reward_events[0]["elapsed_minutes"] == minutes
    assert env.net_lots == 0


def test_signal_bar_extreme_and_entry_bar_extreme_cannot_generate_reward(tmp_path):
    env=make_env(tmp_path, [(1000,1100,900,1000), (1000,1100,900,1000),
                            (1000,1005,1000,1000), (1000,1006,1000,1000)])
    assert env.step(1)[1] == 0
    assert env.net_lots == 3
    assert env.step(0)[1] == 0 # exact touch has no queue evidence
    assert env.step(0)[1] == 60/63 # pre-existing TP has a full-tick penetration


def test_direction_switch_and_invalidation_keep_old_tp_only(tmp_path):
    intervals=[dict(contract="CZCE.FG2701", start="2026-01-05 09:30", end="2026-01-05 09:32", direction="SHORT"),
               dict(contract="CZCE.FG2701", start="2026-01-05 09:32", end="2026-01-05 09:37", direction="LONG")]
    env=make_env(tmp_path, [(1000,1000,1000,1000), (1000,1000,1000,1000),
        (1000,1000,994,994), (1000,1000,1000,1000)], intervals=intervals)
    obs,r,*_=env.step(1)
    assert r==0 and obs["action_mask"].tolist()==[1,0]
    obs,r,*_=env.step(1)
    assert r==60/62 and env.net_lots==0 and obs["action_mask"].tolist()==[1,1]
    env.step(1)
    assert env.user_accounts.long_position==3 and env.user_accounts.short_position==0
    schedule=DirectionSchedule(dict(timezone="Asia/Shanghai", intervals=intervals))
    assert schedule.at("CZCE.FG2701", "2026-01-05 09:37") == "NO_TRADE"
    with pytest.raises(ValueError, match="Overlapping"):
        DirectionSchedule(dict(timezone="Asia/Shanghai", intervals=intervals+[intervals[0]]))


def test_fill_time_rechecks_expired_direction(tmp_path):
    intervals=[dict(contract="CZCE.FG2701", start="2026-01-05 09:30", end="2026-01-05 09:32", direction="LONG")]
    env=make_env(tmp_path, [(1000,1000,1000,1000)]*3, intervals=intervals)
    env.step(0)
    assert env.step(1)[1]==0 and env.net_lots==0


def test_loss_chunk_reset_and_json_restart_preserve_account(tmp_path):
    prices=[(1000,1000,1000,1000), (1000,1000,990,990), (990,990,990,990), (1000,1006,1000,1006)]
    env=make_env(tmp_path, prices, chunk=1)
    _,r,t,tr,_=env.step(1)
    assert r==0 and not t and tr
    cash=env.user_accounts.cash
    state=json.loads(json.dumps(env.export_state()))
    resumed=make_env(tmp_path, prices, chunk=1)
    resumed.import_state(state)
    for obj in (env,resumed):
        obj.reset()
        assert obj.net_lots==3 and obj.user_accounts.cash==cash
        assert obj.user_accounts.unrealized_pnl==Decimal(-600)
        obj.step(0)
        obj.reset()
        assert obj.step(0)[1]==60/63
    assert env.export_state()==resumed.export_state()


def test_data_end_and_gap_do_not_erase_losses_and_can_resume(tmp_path):
    prices=[(1000,1000,1000,1000), (1000,1000,990,990), (990,990,990,990)]
    env=make_env(tmp_path, prices)
    env.step(1);env.step(0)
    assert env.step(0)[3] and env.status=="DATA_END"
    with pytest.raises(RuntimeError, match="unfinished"):
        env.reset()
    state=env.export_state()
    assert Decimal(state["unrealized_pnl"])==Decimal(-600)
    full=make_env(tmp_path, prices+[(1000,1006,1000,1006)])
    env.continue_data(full._raw)
    assert env.step(0)[1]==60/63
    gap=make_env(tmp_path, [prices[0],prices[1],prices[2]], times=["2026-01-05 09:31", "2026-01-05 09:32", "2026-01-05 09:34"])
    gap.step(1)
    assert gap.step(0)[3] and gap.status=="DATA_GAP" and gap.net_lots==3


def test_expiry_and_margin_exit_zero_reward_including_insolvency(tmp_path):
    expiry=make_env(tmp_path, [(1000,1000,1000,1000)]*3, liquidation="2026-01-05 09:33", fees=2)
    expiry.step(1)
    _,r,t,tr,_=expiry.step(0)
    assert r==0 and t and not tr and expiry.attributions[0]["reason"]=="CONTRACT_EXPIRY"
    assert expiry._get_info()["account_net_pnl"] == -12
    margin=make_env(tmp_path, [(10000,10000,10000,10000), (10000,10006,1,10006)], margin_rate=.1)
    _,r,t,_,_=margin.step(1)
    assert r==0 and t and margin.attributions[0]["reason"]=="MARGIN_CALL"
    assert margin.user_accounts.cash < 0 and margin.net_lots==0
    assert margin.ledger.total_balance()==300000


def test_failed_close_rollback_and_partial_tp_do_not_reward_early(tmp_path, monkeypatch):
    env=make_env(tmp_path)
    env.step(1)
    entry_id=next(iter(env.entries))
    before=env.export_state()
    with monkeypatch.context() as ctx:
        ctx.setattr(env, "_post_atomic", lambda _: (_ for _ in ()).throw(ValueError("injected ledger failure")))
        code,event=env.settle_exit(entry_id,1005,"2026-01-05 09:33","TAKE_PROFIT")
    assert event is None and env.net_lots==3 and env.ledger.snapshot()=={k:Decimal(v) for k,v in before["balances"].items()}
    code,event=env.settle_exit(entry_id,1005,"2026-01-05 09:33","TAKE_PROFIT",quantity=1,fill_id="part-1")
    assert event is None and env.net_lots==2 and len(env.entries)==1
    with pytest.raises(ValueError,match="Duplicate"):
        env.settle_exit(entry_id,1005,"2026-01-05 09:33","TAKE_PROFIT",quantity=1,fill_id="part-1")
    code,event=env.settle_exit(entry_id,1005,"2026-01-05 09:34","TAKE_PROFIT",quantity=2,fill_id="part-2")
    assert env.reward_function([event])==60/63
    with pytest.raises(ValueError,match="Duplicate"):
        env.reward_function([event])
    assert env.net_lots==0 and env.ledger.total_balance()==300000


def test_causal_market_observation_does_not_change_with_future_prices(tmp_path):
    prices=[(1000,1000,1000,1000)]*5
    first=make_env(tmp_path,prices)
    prices[-1]=(2000,2000,2000,2000)
    second=make_env(tmp_path,prices)
    assert np.array_equal(first._get_obs()["market_seq"], second._get_obs()["market_seq"])
    assert first.observation_space.contains(first._get_obs())


def test_reward_has_no_account_inputs_and_rejects_uncommitted_tp():
    reward=FastTakeProfitReward()
    assert reward([])==0
    event=dict(entry_id="a",side="LONG",entry_time="2026-01-05 09:31",exit_time="2026-01-05 09:32",
               entry_price=1000,exit_price=1005,closed_quantity=3,reason="MARGIN_CALL",committed=True)
    assert reward([event])==0
    event.update(reason="TAKE_PROFIT",committed=False)
    with pytest.raises(ValueError,match="committed"):
        reward([event])


def test_two_simultaneous_tp_and_overnight_natural_time(tmp_path):
    both=make_env(tmp_path, [(1000,1000,1000,1000)]*3+[(1000,1006,1000,1006)])
    both.step(1);both.step(1)
    assert both.step(0)[1]==60/63+60/62 and both.net_lots==0
    assert len(both.reward_events)==2
    night=make_env(tmp_path, [(1000,1000,1000,1000)]*2+[(1000,1006,1000,1006)],
        times=["2026-01-05 09:31", "2026-01-05 09:32", "2026-01-05 21:31"],
        calendar=[dict(start="2026-01-05 09:30",end="2026-01-05 09:32"),
                  dict(start="2026-01-05 21:30",end="2026-01-05 21:31")])
    night.step(1)
    assert night.step(0)[1]==60/780


def test_mixed_partial_exit_is_not_a_tp_and_fill_volume_is_not_reused(tmp_path):
    env=make_env(tmp_path)
    env.step(1)
    entry_id=next(iter(env.entries))
    assert env.settle_exit(entry_id,990,"2026-01-05 09:33","MARGIN_CALL",quantity=1)[1] is None
    _,event=env.settle_exit(entry_id,1005,"2026-01-05 09:34","TAKE_PROFIT",quantity=2)
    assert event["reason"]=="MARGIN_CALL" and env.reward_function([event])==0
    scarce=make_env(tmp_path,[(1000,1000,1000,1000),(1000,1006,1000,1006)])
    scarce._raw.loc[scarce._raw.index[1],"Volume"]=3
    assert scarce.step(1)[1]==0 and scarce.net_lots==3 # entry consumes all simulated bar volume
