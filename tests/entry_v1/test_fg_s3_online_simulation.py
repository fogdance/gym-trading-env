import json
from pathlib import Path
import threading

import elements
import numpy as np
import pandas as pd
import pytest
import yaml

from gym_trading_env.research.fg_s3_online_simulation import OnlineEntryEnv, OnlineRunner
from test_fg_single3_env import single_env


CLOCK = dict(timezone='Asia/Shanghai', minutes_per_day=345,
    segments=[['21:00','23:00'], ['09:00','10:15'], ['10:30','11:30'], ['13:30','15:00']])


class Counter:
    def __init__(self):
        self.value, self.lock = 0, threading.Lock()
    def __int__(self):
        return self.value


class Agent:
    config = elements.Config(s3_v2_mode='a2')
    def __init__(self):
        self.n_updates, self.n_actions = Counter(), Counter()
    def init_policy(self, batch):
        return ({}, {}, {}, {'action': [np.int32(0)]})
    def policy(self, carry, obs, mode):
        assert mode == 'eval'
        self.seen_previous = int(np.asarray(carry[-1]['action']).reshape(-1)[0])
        self.n_actions.value += 1
        action = np.asarray([int(obs['action_mask'][0,1])], np.int32)
        return ({}, {}, {}, {'action': list(action)}), {'action':action}, {'p_add':np.array([.9])}


class Feed:
    def __init__(self, frames):
        self.frames = list(frames)
    def read(self, after):
        return self.frames.pop(0) if self.frames else pd.DataFrame()


class RPC:
    def __init__(self, success=True):
        self.signals, self.success = [], success
    def send(self, signal):
        self.signals.append(json.loads(signal.to_json()))
        return dict(success=self.success, msg='enqueued' if self.success else 'url_error')


def live_env(tmp_path, *, stale=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    original = single_env(tmp_path, [(1000,1000,1000,1000), (1000,1000,1000,1000),
        (1000,1000,1000,1000), (1000,1007,1000,1007)], side='SHORT', candidate='C1',
        times=pd.date_range('2026-01-05 09:01', periods=4, freq='min'),
        calendar=[dict(start='2026-01-05 09:00', end='2026-01-05 10:15')])
    cfg = yaml.safe_load((tmp_path/'env.yaml').read_text())
    cfg['entry']['market_clock'] = CLOCK
    (tmp_path/'env.yaml').write_text(yaml.safe_dump(cfg))
    frame = original._raw.copy()
    frame['trading_date'] = '2026-01-05'
    clock = [pd.Timestamp('2026-01-05 09:02', tz='Asia/Shanghai')]
    if stale:
        clock[0] += pd.Timedelta(hours=2)
    env = OnlineEntryEnv(frame.iloc[:2], str(tmp_path/'env.yaml'),
        Feed([frame.iloc[2:3], frame.iloc[3:4]]), threading.Event(),
        now=lambda:clock[0], poll_seconds=.001)
    return env, frame, clock


def test_online_short_db_append_emits_open_and_tp_once_with_driver_restore(tmp_path):
    env, frame, clock = live_env(tmp_path/'env')
    root = tmp_path/'run'; root.mkdir()
    rpc, agent = RPC(), Agent()
    runner = OnlineRunner(agent, env, root, dict(run_id='test'), rpc)
    other = None
    try:
        runner.driver(runner.policy, steps=1)
        assert env.current_step == 1 and not env.fills and not rpc.signals
        assert runner.env_steps == 0  # Initial history is feature context only.
        clock[0] += pd.Timedelta(minutes=1)
        runner.driver(runner.policy, steps=1)
        assert env.net_lots == 3 and runner.env_steps == 1
        assert rpc.signals[0]['action_index'] == 6
        assert rpc.signals[0]['action_name'] == 'SHORT_OPEN0'
        assert rpc.signals[0]['volume'] == 3 and not runner.pending
        assert json.loads((root/'status.json').read_text())['pending_rpc'] == 0
        state = env.export_state()
        other_env = OnlineEntryEnv(frame.iloc[:3], str(tmp_path/'env/env.yaml'),
            Feed([frame.iloc[3:]]), threading.Event(), now=lambda:clock[0])
        other = OnlineRunner(Agent(), other_env, root, dict(run_id='test'), rpc)
        other.restore()
        assert other_env.export_state() == state
        clock[0] += pd.Timedelta(minutes=1)
        other.driver(other.policy, steps=1)
        assert other.env_steps == 2 and other_env.net_lots == 0
        assert [x['action_name'] for x in rpc.signals] == ['SHORT_OPEN0','SHORT_CLOSE0']
        assert rpc.signals[1]['action_index'] == 7
        assert rpc.signals[0]['signal_id'] != rpc.signals[1]['signal_id']
        assert other_env.attributions[0]['reason'] == 'TAKE_PROFIT'
        assert int(agent.n_updates) == 0 and int(other.agent.n_updates) == 0
    finally:
        runner.close()
        if other:
            other.close()


def test_stale_startup_proposal_does_not_trade_and_carry_uses_wait(tmp_path):
    env, _, clock = live_env(tmp_path/'env', stale=True)
    root = tmp_path/'run'; root.mkdir()
    rpc, agent = RPC(), Agent()
    runner = OnlineRunner(agent, env, root, dict(run_id='test'), rpc)
    try:
        runner.driver(runner.policy, steps=1)
        assert runner.driver.acts['action'][0] == 0
        runner.driver(runner.policy, steps=1)
        assert not env.fills and not rpc.signals and agent.seen_previous == 0
        assert env.last_execution['stale_signal']
        assert env.current_step == 2 and runner.env_steps == 1
    finally:
        runner.close()


def test_proposal_that_becomes_stale_while_waiting_is_removed_from_real_carry(tmp_path):
    env, _, clock = live_env(tmp_path/'env')
    root = tmp_path/'run'; root.mkdir()
    rpc, agent = RPC(), Agent()
    runner = OnlineRunner(agent, env, root, dict(run_id='test'), rpc)
    try:
        runner.driver(runner.policy, steps=1)
        assert runner.driver.acts['action'][0] == 1
        clock[0] += pd.Timedelta(minutes=3)
        runner.driver(runner.policy, steps=1)
        assert env.last_execution['proposed_action'] == 1
        assert env.last_execution['applied_action'] == 0
        assert not env.fills and not rpc.signals and agent.seen_previous == 0
    finally:
        runner.close()


def test_rpc_failure_retains_committed_account_and_unsent_outbox(tmp_path):
    env, _, clock = live_env(tmp_path/'env')
    root = tmp_path/'run'; root.mkdir()
    runner = OnlineRunner(Agent(), env, root, dict(run_id='test'), RPC(success=False))
    try:
        runner.driver(runner.policy, steps=1)
        clock[0] += pd.Timedelta(minutes=1)
        with pytest.raises(RuntimeError, match='not acknowledged'):
            runner.driver(runner.policy, steps=1)
        import pickle
        state = pickle.loads((root/'state.pkl').read_bytes())
        assert len(state['pending']) == 1 and state['driver']['envs'][0]['env']['entries']
        assert env.net_lots == 3
        assert json.loads((root/'status.json').read_text())['status'] == 'RPC_RECONCILIATION_REQUIRED'
    finally:
        runner.close()


def test_online_configuration_rejects_changed_frozen_train_rules(tmp_path):
    import hashlib
    from gym_trading_env.research.fg_s3_online_simulation import read_settings
    rules = tmp_path/'rules.yaml'; rules.write_text('reward: C1\n')
    cfg = dict(schema='FG_S3_ONLINE_SIM_V1', clock_protocol='S3_MARKET_MINUTE',
        contract='CZCE.FG701', direction='SHORT', device='gpu', rpc_enabled=True,
        poll_seconds=2, max_signal_age_seconds=90, history_days=14,
        checkpoint=str(tmp_path/'checkpoint'), prepared=str(tmp_path/'prepared'),
        rules=str(rules), run_dir=str(tmp_path/'run'),
        rules_sha256=hashlib.sha256(rules.read_bytes()).hexdigest())
    path = tmp_path/'config.yaml'; path.write_text(yaml.safe_dump(cfg))
    assert read_settings(path)['rules'] == str(rules)
    rules.write_text('reward: C2\n')
    with pytest.raises(ValueError, match='pinned TRAIN'):
        read_settings(path)


def test_missing_minute_stops_without_rpc_or_discarding_account(tmp_path):
    env, frame, clock = live_env(tmp_path/'env')
    env.feed = Feed([frame.iloc[3:4]])  # 09:03 is absent.
    root = tmp_path/'run'; root.mkdir()
    rpc = RPC()
    runner = OnlineRunner(Agent(), env, root, dict(run_id='test'), rpc)
    try:
        runner.driver(runner.policy, steps=1)
        clock[0] += pd.Timedelta(minutes=2)
        runner.driver(runner.policy, steps=1)
        assert env.status == 'DATA_GAP' and env.current_step == 1
        assert not env.fills and not rpc.signals and runner.env_steps == 0
    finally:
        runner.close()
