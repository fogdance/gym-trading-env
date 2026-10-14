import copy
from datetime import datetime
from functools import partial
import json
from types import SimpleNamespace

import pytest

from gym_trading_env.research import fg_s3_online_monitor as module


@pytest.fixture
def watch(tmp_path):
    run, out = tmp_path/'run', tmp_path/'observations'
    run.mkdir()
    out.mkdir()
    for name in ('run_identity.json', 'status.json'):
        (run/name).write_text('{}')
    console = module.EventConsole(run)
    return SimpleNamespace(RUN=run, OUT=out, CONSOLE=console, module=module,
        save=partial(module.save, out=out, console=console),
        main=partial(module.monitor_online, run, output_dir=out))


def snapshot(sequence=60, minute='22:00', **overrides):
    bar = f'2026-10-14T{minute}:00+08:00'
    decision = dict(sequence=sequence, bar_end=bar, next_action=0, p_add=.2359,
        net_lots=0, account_net_pnl=0., execution=dict(advanced=True, fills=[]))
    result = dict(observed_at='2026-10-14T22:00:21+08:00', status=dict(
        contract='CZCE.FG701', direction='SHORT', last_bar=bar,
        env_steps=sequence-1, next_action=0, p_add=.2359, net_lots=0,
        account_net_pnl=0), rpc_health=dict(ok=True), warnings=[],
        latest_decision=decision, latest_rpc=None, executions=[], rpc_events=[],
        rpc_attempt_count=0, rpc_ack_count=0, unacknowledged_signal_ids=[])
    result.update(overrides)
    return result


def test_unchanged_healthy_heartbeat_is_silent(watch, capsys):
    first = snapshot()
    watch.save(first, compact=True)
    capsys.readouterr()
    second = copy.deepcopy(first)
    second['observed_at'] = '2026-10-14T22:00:52+08:00'
    watch.save(second, compact=True)
    assert capsys.readouterr().out == ''
    assert len((watch.OUT/'observations.jsonl').read_text().splitlines()) == 2
    assert json.loads((watch.OUT/'latest.json').read_text())['observed_at'] == second['observed_at']


def test_new_bar_is_one_line_and_mirrors_console_log(watch, capsys):
    watch.save(snapshot(), compact=True)
    capsys.readouterr()
    watch.save(snapshot(61, '22:01'), compact=True)
    text = capsys.readouterr().out
    assert len(text.splitlines()) == 1
    assert 'K线22:01' in text and 'WAIT' in text and '23.59%' in text
    assert 'RPC健康' not in text and '无异常' not in text
    assert (watch.RUN/'console.log').read_text().endswith(text)


def test_changed_alert_prints_once_and_recovery_prints_once(watch, capsys):
    watch.save(snapshot(), compact=True)
    capsys.readouterr()
    failed = snapshot(warnings=['RPC_HEALTH_UNAVAILABLE'], rpc_health=dict(ok=False))
    watch.save(failed, compact=True)
    assert '【告警】' in capsys.readouterr().out
    watch.save(failed, compact=True)
    assert capsys.readouterr().out == ''
    watch.save(snapshot(), compact=True)
    assert '【恢复】' in capsys.readouterr().out
    watch.save(snapshot(), compact=True)
    assert capsys.readouterr().out == ''


def test_every_new_fill_is_shown_once_including_between_polls(watch, capsys):
    watch.save(snapshot(), compact=True)
    capsys.readouterr()
    entries = [dict(signal_id='s1', action_name='SHORT_OPEN0', time='2026-10-14T22:01:00+08:00',
        price='856', quantity=3, committed=True, bar_end='2026-10-14T22:02:00+08:00',
        reason='ENTRY', entry_id='CZCE.FG701:1:1'),
        dict(signal_id='s2', action_name='SHORT_CLOSE0', time='2026-10-14T22:03:00+08:00',
        price='851', quantity='3', committed=True, bar_end='2026-10-14T22:03:00+08:00',
        reason='TAKE_PROFIT', entry_id='CZCE.FG701:1:1')]
    rpcs = [dict(signal=dict(signal_id=fill['signal_id'], action_name=fill['action_name'],
        symbol='CZCE.FG701', eob=fill['time'], volume=3, price=float(fill['price'])),
        response=dict(success=True, msg='enqueued', status=200, raw=dict(ok=True))) for fill in entries]
    result = snapshot(63, '22:03', executions=entries, rpc_events=rpcs,
        rpc_attempt_count=2, rpc_ack_count=2)
    watch.save(result, compact=True)
    text = capsys.readouterr().out
    assert text.count('【开仓】') == 1 and text.count('【止盈平仓】') == 1
    assert text.count('RPC 已入队') == 2
    assert '\n\n[' in text and '模拟价' in text
    watch.save(result, compact=True)
    assert capsys.readouterr().out == ''


def test_rpc_ack_arriving_later_is_distinct_from_broker_fill(watch, capsys):
    watch.save(snapshot(), compact=True)
    capsys.readouterr()
    fill = dict(signal_id='s1', action_name='SHORT_OPEN0', time='2026-10-14T22:01:00+08:00',
        price='856', quantity=3, committed=True, bar_end='2026-10-14T22:02:00+08:00',
        reason='ENTRY', entry_id='CZCE.FG701:1:1')
    watch.save(snapshot(62, '22:02', executions=[fill]), compact=True)
    assert 'RPC 待确认' in capsys.readouterr().out
    rpc = dict(signal=dict(signal_id='s1', action_name='SHORT_OPEN0', symbol='CZCE.FG701',
        eob='2026-10-14T22:02:00+08:00', volume=3, price=856.),
        response=dict(success=True, msg='enqueued', status=200, raw=dict(ok=True)))
    result = snapshot(62, '22:02', executions=[fill], rpc_events=[rpc], rpc_ack_count=1)
    watch.save(result, compact=True)
    text = capsys.readouterr().out
    assert '【RPC确认】' in text and '已入队' in text
    assert '成交确认' not in text
    watch.save(result, compact=True)
    assert capsys.readouterr().out == ''


def test_watcher_handover_does_not_skip_trade_after_previous_observation(watch, capsys):
    watch.CONSOLE.seed(snapshot())
    fill = dict(signal_id='s1', action_name='SHORT_OPEN0', time='2026-10-14T22:01:00+08:00',
        price='856', quantity=3, committed=True, bar_end='2026-10-14T22:01:00+08:00',
        reason='ENTRY', entry_id='CZCE.FG701:1:1')
    watch.save(snapshot(61, '22:01', executions=[fill]), compact=True)
    assert '【开仓】' in capsys.readouterr().out


def test_monitor_default_keeps_polling_after_old_deadline(watch, monkeypatch):
    class AfterOldDeadline(datetime):
        @classmethod
        def now(cls, tz):
            return cls(2026, 10, 14, 23, 6, tzinfo=tz)
    class PollObserved(Exception):
        pass
    def polling(delay):
        assert delay == 30
        raise PollObserved()
    monkeypatch.setattr(module, 'datetime', AfterOldDeadline)
    monkeypatch.setattr(module, 'snapshot', lambda *args, **kwargs:snapshot())
    monkeypatch.setattr(module.time, 'sleep', polling)
    with pytest.raises(PollObserved):
        watch.main()


def test_shared_entry_monitors_existing_run_without_rpc_or_model(watch):
    import os
    from pathlib import Path
    import subprocess
    import sys

    now = datetime.now(module.TZ).isoformat()
    identity = dict(run_id='test', rpc_endpoint=None)
    status = dict(status='WAITING_FOR_NEW_BAR', updated_at=now, last_bar=now,
                  contract='CZCE.FG701', direction='SHORT')
    for name, data in [('run_identity.json', identity), ('status.json', status)]:
        (watch.RUN/name).write_text(json.dumps(data))
    before = {p.name: p.read_bytes() for p in watch.RUN.iterdir()}
    project = Path(module.__file__).resolve().parents[3]
    result = subprocess.run([sys.executable, '-B', str(project/'tools/fg_s3_v2_experiment.py'),
        'monitor-online', '--run-dir', str(watch.RUN), '--output-dir', str(watch.OUT),
        '--once'], cwd=project, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
        text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '监控已连接' in result.stdout
    assert (watch.RUN/'console.log').read_text() == result.stdout
    sample = json.loads((watch.OUT/'latest.json').read_text())
    assert sample['warnings'] == []
    assert sample['rpc_health']['enabled'] is False
    assert sample['runner_alive'] is None
    assert all((watch.RUN/name).read_bytes() == data for name, data in before.items())


def test_negative_rpc_receipt_is_an_alert(watch):
    now = datetime.now(module.TZ).isoformat()
    (watch.RUN/'run_identity.json').write_text(json.dumps(dict(run_id='test', rpc_endpoint=None)))
    (watch.RUN/'status.json').write_text(json.dumps(dict(
        status='WAITING_FOR_NEW_BAR', updated_at=now, last_bar=now)))
    (watch.RUN/'rpc.jsonl').write_text(json.dumps(dict(
        signal=dict(signal_id='s1'), response=dict(success=True, raw=dict(ok=False))))+'\n')
    result = module.snapshot(watch.RUN)
    assert 'RPC_REJECTED_OR_FAILED' in result['warnings']
    assert result['rpc_ack_count'] == 0
