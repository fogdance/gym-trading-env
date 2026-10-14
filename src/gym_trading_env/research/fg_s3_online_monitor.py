"""Event console for live S3 runs; never loads a model or submits signals."""
import math
from collections import Counter
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import time
import urllib.request
from zoneinfo import ZoneInfo

TZ = ZoneInfo('Asia/Shanghai')


def json_lines(path):
    if not path.exists():
        return [], False
    content = path.read_bytes()
    partial = bool(content) and not content.endswith(b'\n')
    lines = content.splitlines()
    if partial:
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()], partial


def snapshot(run, *, runner_pid=None):
    now = datetime.now(TZ)
    identity = json.loads((run / 'run_identity.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    actions, action_partial = json_lines(run / 'actions.jsonl')
    rpcs, rpc_partial = json_lines(run / 'rpc.jsonl')
    keys = ('status', 'updated_at', 'contract', 'direction', 'last_bar', 'env_steps',
            'next_action', 'p_add', 'actor_proposed_action', 'stale_signal', 'net_lots',
            'account_net_pnl', 'pending_rpc', 'rpc_enabled', 'policy_mode',
            'training_updates_during_simulation', 'broker_fills_verified')
    result = dict(observed_at=now.isoformat(), watcher_pid=os.getpid(),
                  runner_pid=runner_pid, status={key: status.get(key) for key in keys})
    warnings = []
    result['runner_alive'] = None
    if runner_pid is not None:
        try:
            stat = Path(f'/proc/{runner_pid}/stat').read_text()
            result['runner_alive'] = stat.rsplit(')', 1)[1].split()[0] != 'Z'
        except FileNotFoundError:
            result['runner_alive'] = False
        if not result['runner_alive']:
            warnings.append('RUNNER_NOT_ALIVE')
    result['heartbeat_age_seconds'] = round((now - datetime.fromisoformat(status['updated_at'])).total_seconds(), 2)
    result['last_bar_age_seconds'] = round((now - datetime.fromisoformat(status['last_bar'])).total_seconds(), 2)
    if result['heartbeat_age_seconds'] > 65:
        warnings.append('HEARTBEAT_OVER_65_SECONDS')
    if now.hour in (21, 22) and result['last_bar_age_seconds'] > 150:
        warnings.append('NIGHT_BAR_OVER_150_SECONDS')
    if status['status'] not in ('ONLINE_RUNNING', 'WAITING_FOR_NEW_BAR'):
        warnings.append('RUNNER_STATUS_' + status['status'])
    if status.get('training_updates_during_simulation', 0) != 0:
        warnings.append('MODEL_UPDATED_DURING_LIVE')
    endpoint = identity.get('rpc_endpoint')
    result['rpc_health'] = dict(enabled=bool(endpoint), ok=None)
    if endpoint:
        try:
            with urllib.request.urlopen(endpoint.rstrip('/')+'/health', timeout=5) as response:
                health = json.loads(response.read())
                result['rpc_health'] = dict(enabled=True, http_status=response.status, ok=health.get('ok'))
                if health.get('ok') is not True:
                    warnings.append('RPC_HEALTH_NOT_OK')
        except Exception as error:
            result['rpc_health'] = dict(enabled=True, ok=False, error_type=type(error).__name__)
            warnings.append('RPC_HEALTH_UNAVAILABLE')
    fills = []
    for action in actions:
        for fill in (action.get('execution') or {}).get('fills', []):
            suffix = 'OPEN0' if fill['reason'] == 'ENTRY' else 'CLOSE0'
            name = fill['side'] + '_' + suffix
            signal_id = f"{identity['run_id']}|{fill['entry_id']}|{fill['time']}|{name}"
            fills.append(dict(signal_id=signal_id, action_name=name, time=fill['time'],
                              price=fill['price'], quantity=fill['quantity'],
                              committed=fill['committed'], bar_end=action['bar_end'],
                              reason=fill['reason'], entry_id=fill['entry_id']))
    acknowledgements = [row for row in rpcs if row['response'].get('success') is True
                        and row['response'].get('raw', {}).get('ok') is True]
    acknowledged_ids = {row['signal']['signal_id'] for row in acknowledgements}
    expected_ids = {fill['signal_id'] for fill in fills if fill['committed']}
    result['execution_count'] = len(fills)
    result['executions'] = fills
    result['rpc_events'] = rpcs
    result['rpc_attempt_count'] = len(rpcs)
    result['rpc_ack_count'] = len(acknowledgements)
    result['unacknowledged_signal_ids'] = sorted(expected_ids - acknowledged_ids)
    result['unexpected_rpc_signal_ids'] = sorted(acknowledged_ids - expected_ids)
    result['partial_log_line'] = dict(actions=action_partial, rpc=rpc_partial)
    if len(acknowledgements) != len(rpcs):
        warnings.append('RPC_REJECTED_OR_FAILED')
    if result['unexpected_rpc_signal_ids']:
        warnings.append('RPC_WITHOUT_MATCHING_EXECUTION')
    result['decision_counts'] = dict(Counter(str(row['next_action']) for row in actions))
    result['latest_decision'] = ({key: actions[-1].get(key) for key in
                                  ('sequence', 'bar_end', 'next_action', 'p_add', 'execution',
                                   'net_lots', 'account_net_pnl')}
                                 if actions else None)
    result['latest_execution'] = fills[-1] if fills else None
    result['latest_rpc'] = (dict(signal={key: rpcs[-1]['signal'].get(key) for key in
                                       ('signal_id', 'eob', 'action_name', 'volume', 'price')},
                               response={key: rpcs[-1]['response'].get(key) for key in
                                         ('success', 'msg', 'status')}) if rpcs else None)
    failure = run / 'failure.json'
    if failure.exists():
        result['failure'] = json.loads(failure.read_text())
        warnings.append('FAILURE_FILE_PRESENT')
    result['warnings'] = warnings
    return result


WARNING_TEXT = {
    'RUNNER_NOT_ALIVE': '交易进程已退出',
    'HEARTBEAT_OVER_65_SECONDS': '进程状态超过65秒未更新',
    'NIGHT_BAR_OVER_150_SECONDS': '夜盘K线超过150秒未更新',
    'MODEL_UPDATED_DURING_LIVE': '在线运行出现训练更新',
    'RPC_HEALTH_NOT_OK': '模拟账户RPC健康检查失败',
    'RPC_HEALTH_UNAVAILABLE': '无法连接模拟账户RPC',
    'RPC_REJECTED_OR_FAILED': 'RPC发送失败或被拒绝',
    'RPC_WITHOUT_MATCHING_EXECUTION': 'RPC缺少对应成交事件',
    'FAILURE_FILE_PRESENT': '交易进程记录了运行错误',
    'WATCHER_SAMPLE_FAILED': '监控采样失败',
}


class EventConsole:
    """Only changes reach the terminal; all samples remain in the audit file."""
    def __init__(self, run):
        self.run = Path(run)
        self.started = False
        self.decision_sequence = None
        self.seen_fills = set()
        self.rpc_count = 0
        self.warnings = ()
        self.last_alert_at = None
        self.baseline_bar = None

    def seed(self, previous):
        """Carry the previous observation boundary across a watcher restart."""
        decision = previous.get('latest_decision')
        if decision:
            self.baseline_bar = datetime.fromisoformat(decision['bar_end'])
            self.decision_sequence = decision['sequence']
            self.rpc_count = previous['rpc_attempt_count']

    @staticmethod
    def rpc_state(row):
        if row is None:
            return 'RPC 待确认'
        response = row['response']
        if response.get('success') is True and response.get('raw', {}).get('ok') is True:
            return 'RPC 已入队'
        return f"RPC 失败：{response.get('msg', '未知原因')}"

    def emit(self, kind, text, payload, *, compact):
        rendered = text if compact else json.dumps(
            dict(event=kind, **payload), ensure_ascii=False, default=str)
        with (self.run/'console.log').open('a') as output:
            output.write(rendered+'\n')
        print(rendered, flush=True)

    def update(self, result, *, compact):
        now = datetime.fromisoformat(result['observed_at'])
        clock = now.strftime('%H:%M:%S')
        executions = result.get('executions', [])
        rpcs = result.get('rpc_events', [])
        if not self.started:
            self.started = True
            # Existing history is a baseline, never announced as new trades.
            self.seen_fills = {fill['signal_id'] for fill in executions
                               if self.baseline_bar is None
                               or datetime.fromisoformat(fill['bar_end']) <= self.baseline_bar}
            if self.baseline_bar is None:
                self.rpc_count = len(rpcs)
            self.emit('MONITOR_STARTED', f'[{clock}] 监控已连接；新K线、交易和异常将显示在此。',
                      dict(observed_at=result['observed_at']), compact=compact)
        warnings = tuple(sorted(result.get('warnings', [])))
        reminder = (warnings and self.last_alert_at is not None
                    and (now-self.last_alert_at).total_seconds() >= 300)
        if warnings and (warnings != self.warnings or reminder):
            descriptions = [WARNING_TEXT.get(w, w.replace('RUNNER_STATUS_', '策略状态：')) for w in warnings]
            if result.get('error_type'):
                descriptions.append(result['error_type'])
            self.emit('ALERT', f"\n[{clock}] 【告警】\n  "+'；'.join(descriptions)+'\n',
                      dict(observed_at=result['observed_at'], warnings=warnings), compact=compact)
            self.last_alert_at = now
        elif self.warnings and not warnings:
            self.emit('RECOVERED', f'\n[{clock}] 【恢复】上一轮告警已解除。\n',
                      dict(observed_at=result['observed_at']), compact=compact)
            self.last_alert_at = None
        self.warnings = warnings
        by_signal = {row['signal']['signal_id']:row for row in rpcs}
        announced = set()
        for fill in executions:
            signal_id = fill['signal_id']
            if signal_id in self.seen_fills:
                continue
            reason = fill['reason']
            label = {'ENTRY':'开仓', 'TAKE_PROFIT':'止盈平仓', 'STOP_LOSS':'止损平仓'}.get(reason, '平仓')
            side = '做空' if fill['action_name'].startswith('SHORT') else '做多'
            contract = result['status']['contract']
            fill_time = datetime.fromisoformat(fill['time']).strftime('%H:%M:%S')
            ack = self.rpc_state(by_signal.get(signal_id))
            text = (f"\n[{clock}] 【{label}】{contract}  {side} {float(fill['quantity']):g}手\n"
                    f"  模拟成交 {fill_time}  模拟价 {float(fill['price']):.2f}\n"
                    f"  {ack}  |  条目 {fill['entry_id']}\n")
            self.emit('EXECUTION', text, dict(observed_at=result['observed_at'], fill=fill,
                      rpc=by_signal.get(signal_id)), compact=compact)
            self.seen_fills.add(signal_id)
            announced.add(signal_id)
        for row in rpcs[self.rpc_count:]:
            if row['signal']['signal_id'] in announced:
                continue
            accepted = self.rpc_state(row) == 'RPC 已入队'
            label = 'RPC确认' if accepted else 'RPC失败'
            text = (f"\n[{clock}] 【{label}】{row['signal']['action_name']}  {self.rpc_state(row)}\n"
                    f"  信号 {row['signal']['signal_id']}\n")
            self.emit('RPC_ACK' if accepted else 'RPC_FAILED', text,
                      dict(observed_at=result['observed_at'], rpc=row), compact=compact)
        self.rpc_count = len(rpcs)
        decision = result.get('latest_decision')
        status = result.get('status')
        if decision and status and decision['sequence'] != self.decision_sequence:
            bar = datetime.fromisoformat(decision['bar_end']).strftime('%H:%M')
            side = '空' if status['direction'] == 'SHORT' else '多'
            action = 'WAIT' if decision['next_action'] == 0 else 'ADD_3'
            text = (f"[{clock}] {status['contract']} {side}  K线{bar}  {action}  "
                    f"P(ADD)={decision['p_add']:.2%}  仓{decision['net_lots']:g}手  "
                    f"PnL{decision['account_net_pnl']:+.2f}")
            self.emit('DECISION', text, dict(observed_at=result['observed_at'], decision=decision),
                      compact=compact)
            self.decision_sequence = decision['sequence']


def save(result, *, out, console, compact=True):
    # Execution history stays in the original action/RPC files; samples are bounded.
    encoded = json.dumps({k: v for k, v in result.items() if k not in ('executions', 'rpc_events')},
                         ensure_ascii=False, default=str)
    temporary = out / f'latest.{os.getpid()}.tmp'
    temporary.write_text(encoded+'\n')
    temporary.replace(out/'latest.json')
    with (out/'observations.jsonl').open('a') as output:
        output.write(encoded+'\n')
    console.update(result, compact=compact)


def monitor_online(run_dir, *, output_dir=None, runner_pid=None, interval=30,
                   once=False, compact=True, until=None):
    """Follow an existing run; only console/monitor files are written."""
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError('Monitor interval must be positive and finite')
    if runner_pid is not None and runner_pid <= 0:
        raise ValueError('Runner PID must be positive')
    run = Path(run_dir).expanduser().resolve()
    for filename in ('run_identity.json', 'status.json'):
        if not (run/filename).is_file():
            raise FileNotFoundError(run/filename)
    until = datetime.fromisoformat(until) if until else None
    if until is not None and until.tzinfo is None:
        until = until.replace(tzinfo=TZ)
    out = Path(output_dir).expanduser().resolve() if output_dir else run/'monitor'
    out.mkdir(parents=True, exist_ok=True)
    console = EventConsole(run)
    with (out/'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = out/'latest.json'
        if previous.exists():
            console.seed(json.loads(previous.read_text()))
        while True:
            try:
                result = snapshot(run, runner_pid=runner_pid)
            except Exception as error:
                result = dict(observed_at=datetime.now(TZ).isoformat(), watcher_pid=os.getpid(),
                              warnings=['WATCHER_SAMPLE_FAILED'], error_type=type(error).__name__)
            save(result, out=out, console=console, compact=compact)
            if once or (until is not None and datetime.now(TZ) >= until):
                break
            delay = interval if until is None else min(
                interval, max(0, (until-datetime.now(TZ)).total_seconds()))
            time.sleep(delay)
