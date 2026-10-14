"""Forward S3 minute simulation over the existing DB and order RPC interfaces.

The original Single3 engine owns fills, TP5/SL30, rewards and accounting.
RPC acknowledgement means queued signal, not a confirmed broker fill.
"""
import copy
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path
import pickle
import signal
import threading
import time

import numpy as np
import pandas as pd
import yaml

from dreamerv3.configuration import read_config_yaml
from gym_trading_env.envs.action import Action, ForexCode
from gym_trading_env.envs.entry_contract import fingerprint, is_fg, timestamp
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from gym_trading_env.utils.bar_source import JUEJIN_FUT_BAR_TABLE, connect_juejin_db
from gym_trading_env.utils.rpc_protocol import TradeSignal
from gym_trading_env.utils.time_contract import from_db_naive_series, to_db_naive_dt
from .fg_replay import check_files, digest, read_manifest, signed, write_json


SCHEMA = 'FG_S3_ONLINE_SIM_V1'
PROJECT = Path(__file__).resolve().parents[3]


def read_settings(config):
    path = Path(config)
    if not path.is_file():
        path = PROJECT/'configs/experiments/fg_s3_v2'/f'{config}.yaml'
    s = read_config_yaml(path)
    if (s.get('schema') != SCHEMA or s.get('clock_protocol') != 'S3_MARKET_MINUTE'
            or not is_fg(s.get('contract', '')) or s.get('direction') not in ('LONG', 'SHORT')):
        raise ValueError('Explicit FG Single3 online configuration required')
    if type(s.get('rpc_enabled')) is not bool or s.get('device') not in ('cpu', 'gpu'):
        raise ValueError('Online RPC and device must be explicit')
    for key in ('poll_seconds', 'max_signal_age_seconds', 'history_days'):
        if (isinstance(s.get(key), bool) or not isinstance(s.get(key), (int, float))
                or not np.isfinite(s[key]) or s[key] <= 0):
            raise ValueError('Positive online setting required: '+key)
    for key in ('checkpoint', 'prepared', 'rules', 'run_dir'):
        value = Path(s[key]).expanduser()
        s[key] = str((PROJECT/value).resolve())
    if digest(s['rules']) != s['rules_sha256']:
        raise ValueError('Online economics/reward/clock differs from the pinned TRAIN rule snapshot')
    s['config_path'], s['config_sha256'] = str(path.resolve()), digest(path)
    return s


class JuejinSingle3Feed:
    """Read completed, real-contract bars; never substitute a continuous symbol."""
    def __init__(self, contract, history_days, *, now=None):
        self.contract, self.history_days = contract, history_days
        self.now = now or (lambda: pd.Timestamp.now(tz='Asia/Shanghai'))

    def _query(self, sql, args):
        conn = connect_juejin_db(connect_timeout=5, read_timeout=15, write_timeout=5)
        try:
            with conn.cursor() as c:
                c.execute(sql, args)
                return c.fetchall()
        finally:
            conn.close()

    def metadata(self):
        rows = self._query('SELECT symbol, symbol_kind, price_tick, multiplier, '
            'listed_date, delisted_date FROM fut_contract_v2 WHERE symbol=%s', (self.contract,))
        if len(rows) != 1 or rows[0]['symbol_kind'] != 'REAL':
            raise ValueError('Online simulation requires real contract metadata')
        return rows[0]

    def read(self, after=None):
        now = self.now()
        start = timestamp(after) if after is not None else now-pd.Timedelta(days=self.history_days)
        operator = '>' if after is not None else '>='
        rows = self._query(f'SELECT trading_date, symbol, bob, eob, `open`, `high`, '
            f'`low`, `close`, volume, `position` FROM {JUEJIN_FUT_BAR_TABLE} '
            f'WHERE symbol=%s AND eob {operator} %s AND eob <= %s AND bob < eob ORDER BY eob',
            (self.contract, to_db_naive_dt(start), to_db_naive_dt(now)))
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        for key in ('bob', 'eob'):
            frame[key] = from_db_naive_series(frame[key])
        frame = frame.set_index('eob').rename(columns={'open':'Open', 'high':'High',
            'low':'Low', 'close':'Close', 'volume':'Volume', 'position':'OpenInterest'})
        for key in ('Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest'):
            frame[key] = pd.to_numeric(frame[key], errors='raise').astype(float)
        frame['trading_date'] = pd.to_datetime(frame.trading_date).dt.strftime('%Y-%m-%d')
        if (frame.index.has_duplicates or not frame.index.is_monotonic_increasing
                or set(frame.symbol) != {self.contract}
                or not (frame.index-pd.DatetimeIndex(frame.bob) == pd.Timedelta(minutes=1)).all()):
            raise ValueError('Invalid online bar chronology or contract')
        return frame


def observed_sessions(frame, clock):
    """Expand observed sessions using the same ex-ante 345-minute schedule.

    Dates come from actual DB bars. Missing minutes within an observed session
    remain gaps; a holiday is never filled with fabricated prices.
    """
    sessions = set()
    for t in frame.index:
        t = timestamp(t)
        for start, end in clock['segments']:
            a, b = timestamp(f'{t.date()} {start}'), timestamp(f'{t.date()} {end}')
            if a < t <= b:
                sessions.add((a, b))
                break
        else:
            raise ValueError(f'Bar outside the frozen FG market schedule: {t}')
    return [dict(start=str(a), end=str(b)) for a,b in sorted(sessions)]


class OnlineEntryEnv(FixedDirectionEntryEnv):
    """A polling adapter; the frozen offline fill engine is reused unchanged."""
    def __init__(self, df, config_path, feed, stop, *, poll_seconds=2,
                 max_signal_age_seconds=90, now=None):
        self.feed, self.stop = feed, stop
        self.poll_seconds, self.max_signal_age = poll_seconds, max_signal_age_seconds
        self.now = now or (lambda: pd.Timestamp.now(tz='Asia/Shanghai'))
        self.on_wait = lambda: None
        self.last_execution = None
        super().__init__(df=df, config_path=config_path)
        # Move a fresh, empty account to the latest known quote. History only
        # supplies causal features, and generates no simulated trades or RPCs.
        self.current_step = len(self._raw)-1
        self._mark(self._raw.iloc[-1].Close, self._raw.index[-1])
        self.equity_history = []
        self._sample_equity()
        self._get_obs()
        self._initial_online_reset = True

    def _load_market(self, df):
        if df is None:
            raise ValueError('Online Entry requires an explicit DB frame')
        sessions = observed_sessions(df, self.market_clock)
        self.terms.calendar = pd.DatetimeIndex([t for s in sessions for t in
            pd.date_range(timestamp(s['start'])+pd.Timedelta(minutes=1), timestamp(s['end']), freq='min')])
        super()._load_market(df)

    def reset(self, seed=None, options=None):
        if getattr(self, '_initial_online_reset', False) and not options:
            self._initial_online_reset = False
            return self._get_obs(), self._get_info()
        return super().reset(seed=seed, options=options)

    def step(self, action):
        decision_time = self._raw.index[self.current_step]
        while self.current_step+1 == len(self._raw):
            if self.stop.is_set():
                raise InterruptedError('Online simulation stopped while waiting for a bar')
            incoming = self.feed.read(after=self._raw.index[-1])
            if incoming.empty:
                self.on_wait()
                self.stop.wait(self.poll_seconds)
                continue
            # Use the existing account-preserving continuation contract.
            self.status, self.truncated = 'DATA_END', True
            calendar = self.terms.calendar
            try:
                self.continue_data(pd.concat([self._raw, incoming]))
            except BaseException:
                self.terms.calendar = calendar
                raise
        age = (self.now()-decision_time).total_seconds()
        applied = 0 if age > self.max_signal_age else int(action)
        fills = len(self.fills)
        result = super().step(applied)
        self.last_execution = dict(signal_time=str(decision_time), proposed_action=int(action),
            applied_action=applied, stale_signal=age > self.max_signal_age,
            advanced=self._raw.index[self.current_step] != decision_time,
            fills=copy.deepcopy(self.fills[fills:]))
        return result


def fill_signal(fill, contract, run_id, bar_end):
    """RPC action_index is the execution enum, not WAIT/ADD_3's index."""
    if not fill['committed'] or Decimal(str(fill['quantity'])) != 3:
        raise ValueError('Only committed atomic Single3 fills may emit an RPC')
    suffix = 'OPEN0' if fill['reason'] == 'ENTRY' else 'CLOSE0'
    action = Action[f"{fill['side']}_{suffix}"]
    eob = timestamp(bar_end).tz_localize(None).strftime('%Y-%m-%d %H:%M:%S')
    return TradeSignal(signal_id=f"{run_id}|{fill['entry_id']}|{fill['time']}|{action.name}",
        symbol=contract, eob=eob, action_index=action.value, action_name=action.name,
        volume=3, price=float(fill['price']), action_result=ForexCode.SUCCESS.value,
        meta=dict(protocol=SCHEMA, simulated_fill=True, fill_reason=fill['reason'],
                  simulated_fill_time=fill['time'], entry_id=fill['entry_id']))


def atomic_bytes(path, data):
    tmp = Path(str(path)+'.tmp')
    with tmp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


class OnlineRunner:
    """Original Driver/ActorPolicy with durable account, carry and RPC outbox."""
    def __init__(self, agent, env, root, identity, rpc=None):
        import embodied
        from embodied.envs.from_gymnasium import FromGymnasium
        from .fg_single3_evaluation import ActorPolicy
        self.root, self.identity, self.rpc = Path(root), identity, rpc
        self.agent, self.env = agent, env
        self.actor = ActorPolicy(agent)
        wrapped = embodied.wrappers.CheckSpaces(embodied.wrappers.UnifyDtypes(FromGymnasium(env)))
        self.driver = embodied.Driver([lambda: wrapped], parallel=False)
        self.driver.reset(agent.init_policy)
        self.driver.on_step(self.capture)
        self.updates_at_start = int(agent.n_updates)
        self.pending, self.env_steps, self.records, self.heartbeat = [], 0, 0, 0
        self.state_status = 'INITIALIZING'
        env.on_wait = self.waiting

    def policy(self, carry, obs):
        from embodied.run.fg_s3_dagger_policy import replace_prevact
        if self.env.last_execution and not bool(obs['is_first'][0]):
            # A stale pending proposal can become WAIT while the DB is idle.
            carry = replace_prevact(carry, np.array([self.env.last_execution['applied_action']], np.int32))
        carry, acts, outs = self.actor(carry, obs)
        age = (self.env.now()-self.env._raw.index[self.env.current_step]).total_seconds()
        self.actor.last_diagnostics['stale_signal'] = age > self.env.max_signal_age
        self.actor.last_diagnostics['actor_proposed_action'] = int(acts['action'][0])
        if age > self.env.max_signal_age:
            acts = {'action': np.zeros(1, np.int32)}
            carry = replace_prevact(carry, acts['action'])
        return carry, acts, outs

    def status(self, state):
        payload = dict(status=state, updated_at=str(self.env.now()), contract=self.env.contract,
            direction=self.env.direction_at(self.env._raw.index[self.env.current_step]),
            last_bar=str(self.env._raw.index[self.env.current_step]), env_steps=self.env_steps,
            next_action=int(self.driver.acts['action'][0]), model_updates=int(self.agent.n_updates),
            training_updates_during_simulation=int(self.agent.n_updates)-self.updates_at_start,
            net_lots=float(self.env.net_lots), account_net_pnl=self.env._get_info()['account_net_pnl'],
            pending_rpc=len(self.pending), rpc_enabled=self.rpc is not None,
            policy_mode='eval_argmax', teacher_guidance=False, simulated_fills=True,
            broker_fills_verified=False, **self.actor.last_diagnostics)
        atomic_bytes(self.root/'status.json', json.dumps(payload, indent=2, default=str).encode())
        self.state_status = state
        return payload

    def waiting(self):
        if time.monotonic()-self.heartbeat >= 30:
            self.heartbeat = time.monotonic()
            print(json.dumps(self.status('WAITING_FOR_NEW_BAR'), default=str), flush=True)

    def save(self):
        if int(self.agent.n_updates) != self.updates_at_start:
            raise RuntimeError('Online simulation unexpectedly updated the model')
        log = self.root/'actions.jsonl'
        state = dict(schema=SCHEMA, identity=self.identity, driver=self.driver.save(),
            market=self.env._raw, pending=self.pending, env_steps=self.env_steps,
            records=self.records, diagnostics=self.actor.last_diagnostics,
            action_counter=int(self.agent.n_actions), log_bytes=log.stat().st_size if log.exists() else 0)
        atomic_bytes(self.root/'state.pkl', pickle.dumps(state))
        atomic_bytes(self.root/'account_state.json', json.dumps(self.env.export_state(), indent=2).encode())

    def restore(self):
        state = pickle.loads((self.root/'state.pkl').read_bytes())
        if state['schema'] != SCHEMA or state['identity'] != self.identity:
            raise ValueError('Online checkpoint/configuration/code identity changed')
        log = self.root/'actions.jsonl'
        if log.stat().st_size != state['log_bytes']:
            raise ValueError('Uncommitted online action log; preserve and reconcile it before resuming')
        self.env._load_market(state['market'])
        self.driver.load(state['driver'])
        self.env._initial_online_reset = False
        self.pending, self.env_steps, self.records = state['pending'], state['env_steps'], state['records']
        self.actor.last_diagnostics = state['diagnostics']
        with self.agent.n_actions.lock:
            self.agent.n_actions.value = state['action_counter']

    def send_pending(self):
        while self.pending:
            payload = self.pending[0]
            response = self.rpc.send(TradeSignal.from_dict(payload))
            with (self.root/'rpc.jsonl').open('a') as f:
                f.write(json.dumps(dict(signal=payload, response=response), default=str)+'\n')
            if not response.get('success'):
                # Preserve the outbox and stop: an HTTP error is not a broker
                # rejection, and blindly progressing would desync accounts.
                self.status('RPC_RECONCILIATION_REQUIRED')
                raise RuntimeError('RPC signal was not acknowledged: '+str(response.get('msg')))
            self.pending.pop(0)
            self.save()

    def capture(self, row, worker):
        execution = self.env.last_execution if not bool(row['is_first']) else None
        if execution and execution['advanced']:
            self.env_steps += 1
        self.records += 1
        record = dict(sequence=self.records, bar_end=str(self.env._raw.index[self.env.current_step]),
            next_action=int(row['action']), p_add=self.actor.last_diagnostics['p_add'],
            action_mask=np.asarray(row['action_mask']).tolist(), execution=execution,
            reward=float(row['reward']), net_lots=float(self.env.net_lots),
            account_net_pnl=self.env._get_info()['account_net_pnl'],
            diagnostics=self.actor.last_diagnostics)
        with (self.root/'actions.jsonl').open('a') as f:
            f.write(json.dumps(record, default=str)+'\n')
        if self.rpc is not None and execution:
            for fill in execution['fills']:
                sig = fill_signal(fill, self.env.contract, self.identity['run_id'], record['bar_end'])
                self.pending.append(json.loads(sig.to_json()))
        self.save()  # Persist the committed account and outbox before network I/O.
        self.send_pending()
        self.status('ONLINE_RUNNING')
        print(json.dumps(record, default=str), flush=True)

    def close(self):
        self.driver.close()


def prepare_env(s, meta, frame, root):
    rules = read_config_yaml(s['rules'])
    end = timestamp(meta['delisted_date'])
    start = timestamp(meta['listed_date'])
    costs = rules['costs']
    # Provider multiplier is not the research lot multiplier. Reuse the
    # checkpoint's user-configured economics (20 tonnes, fee/spread/margin).
    terms = dict(contract=s['contract'], tick_size=float(meta['price_tick']),
        lot_size=costs['lot_size'], last_open_at=str(end), liquidate_at=str(end),
        sessions=observed_sessions(frame, rules['market_clock']), terms=[dict(
            start=str(start), end=str(end+pd.Timedelta(days=1)),
            **{k:costs[k] for k in ('margin_rate','open_fee','close_fee','close_today_fee','spread','slippage')})])
    directions = dict(timezone='Asia/Shanghai', intervals=[dict(contract=s['contract'],
        start=str(start), end=str(end), direction=s['direction'])])
    cfg = copy.deepcopy(rules['environment'])
    cfg['entry'] = dict(contract=s['contract'], protocol='fg_single3_v1', chunk_steps=0,
        direction_config=str(root/'directions.yaml'), contract_config=str(root/'contract.yaml'),
        market_clock=rules['market_clock'], reward=rules['reward'])
    cfg['trading'].update(future_symbol=s['contract'], currency_pair=s['contract'], live_mode=False)
    for name, data in (('directions.yaml', directions), ('contract.yaml', terms), ('env.yaml', cfg)):
        path = root/name
        if path.exists():
            # Resume uses the original generated metadata snapshot.
            continue
        path.write_text(yaml.safe_dump(data, sort_keys=False))
    return root/'env.yaml'


def load_model(s, root):
    from embodied.run.fg_checkpoint import file_hash, validate_bundle
    from embodied.run.fg_s3_source_transfer import make_agent
    from dreamerv3.fg_s3_v2_agent import S3V2Agent
    from dreamerv3.fg_s3a_actor import parameter_digests
    node, prepared = Path(s['checkpoint']), Path(s['prepared'])
    if file_hash(node/'manifest.json') != s['checkpoint_manifest_sha256']:
        raise ValueError('Selected checkpoint is not the pinned 800K checkpoint')
    meta = validate_bundle(node)
    if meta['step'] != 800000 or meta['identity']['mode'] != 'tutor':
        raise ValueError('Expected Tutor-CD 800K, never the latest checkpoint')
    preparation = read_manifest(prepared/'manifest.json')
    check_files({str(prepared/k):v for k,v in preparation['files'].items()})
    if meta['identity']['preparation_sha256'] != file_hash(prepared/'manifest.json'):
        raise ValueError('Model preparation differs from the checkpoint')
    frozen = read_manifest(prepared/'model_config.json')
    settings = read_manifest(prepared/'settings.json')['settings']
    if (frozen['agent_extension'] != meta['identity']['agent_extension']
            or fingerprint(frozen['config']) != meta['identity']['model_config_sha256']):
        raise ValueError('Tutor architecture/settings identity differs')
    agent = make_agent(frozen['config'], root/'model', agent_cls=S3V2Agent,
        device=s['device'], bc_lambda=settings['bc_lambda'], extra_agent_config=frozen['agent_extension'])
    data = pickle.loads((node/'agent.pkl').read_bytes())
    expected = parameter_digests(data['params'])
    agent.load_exact(data)
    agent.sync_policy_from_train()
    actual = parameter_digests(agent.params)
    if expected != actual:
        raise ValueError('Loaded 800K parameters differ from the checkpoint')
    policy = parameter_digests(agent.policy_params)
    if any(value != actual[key] for key,value in policy.items()):
        raise ValueError('Online policy differs from the restored training weights')
    return agent, signed(dict(checkpoint=str(node), checkpoint_env_steps=meta['step'],
        checkpoint_manifest_sha256=file_hash(node/'manifest.json'),
        source_counters=data['counters'], exact_parameter_hashes_verified=True,
        policy_parameters_synchronized=True, model_config_sha256=file_hash(prepared/'model_config.json'),
        agent_extension=frozen['agent_extension'], training_started=False, teacher_guidance=False))


def simulate_online(config, *, run_dir=None, resume=False, check_only=False):
    s = read_settings(config)
    os.environ['JAX_PLATFORMS'] = 'cuda' if s['device'] == 'gpu' else 'cpu'
    root = Path(run_dir or s['run_dir']).resolve()
    if root.exists() and not resume and set(p.name for p in root.iterdir())-{'runner.log'}:
        raise FileExistsError('Existing online outputs retained; use --resume-online or a fresh run directory')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ['DREAMER_RUN_DIR'] = str(root)
    stop = threading.Event()
    previous_handlers = {}
    for number in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[number] = signal.signal(number, lambda *_: stop.set())
    runner = None
    try:
        feed = JuejinSingle3Feed(s['contract'], s['history_days'])
        meta, frame = feed.metadata(), feed.read()
        if len(frame) < 60:
            raise ValueError('Insufficient real DB history for the 60-bar observation')
        env_path = prepare_env(s, meta, frame, root)
        env = OnlineEntryEnv(frame, str(env_path), feed, stop, poll_seconds=s['poll_seconds'],
            max_signal_age_seconds=s['max_signal_age_seconds'])
        agent, source = load_model(s, root)
        rpc = None
        rpc_endpoint = None
        if s['rpc_enabled']:
            from gym_trading_env.config.settings import RPC_ORDER_CONFIG
            from gym_trading_env.utils.rpc_client import LanOrderClient
            import urllib.request
            if not RPC_ORDER_CONFIG.order_enabled:
                raise ValueError('Online configuration requires the existing RPC to be enabled')
            rpc_endpoint = RPC_ORDER_CONFIG.order_endpoint
            with urllib.request.urlopen(rpc_endpoint.rstrip('/')+'/health', timeout=5) as r:
                if not json.loads(r.read()).get('ok'):
                    raise ValueError('RPC health check failed')
            if not check_only:
                rpc = LanOrderClient(rpc_endpoint, RPC_ORDER_CONFIG.order_token,
                    RPC_ORDER_CONFIG.order_timeout_sec)
        from embodied.run.fg_s3_v2_train import execution_code
        identity = dict(run_id=s['experiment'], settings=s, source=source['checkpoint_manifest_sha256'],
            env_files={name:digest(root/name) for name in ('env.yaml','contract.yaml','directions.yaml')},
            code=execution_code(),
            rpc_endpoint=rpc_endpoint)
        write_json(root/'source_receipt.json', source)
        runner = OnlineRunner(agent, env, root, identity, rpc)
        if resume:
            runner.restore()
            if runner.pending:
                raise ValueError('Pending RPC needs reconciliation before resume; state/outbox retained')
            if env.terminated or env.truncated:
                raise ValueError('Stopped/gapped online account cannot be reset by resume; state retained')
        else:
            runner.driver(runner.policy, steps=1)  # Initial observation, no market advance.
        write_json(root/'run_identity.json', signed(identity))
        if check_only:
            result = signed(dict(status='ONLINE_PREFLIGHT_PASS', source=source,
                contract=s['contract'], direction=s['direction'], history_bars=len(frame),
                latest_db_bar=str(frame.index[-1]), observation_identity=env.observation_identity,
                p_add=runner.actor.last_diagnostics['p_add'],
                rpc_health='PASS' if s['rpc_enabled'] else 'DISABLED', rpc_orders_sent=0,
                model_updates=int(agent.n_updates), training_updates_during_simulation=0))
            write_json(root/'preflight_receipt.json', result)
            runner.status('PREFLIGHT_COMPLETE')
            return result
        runner.waiting()
        while not stop.is_set():
            runner.driver(runner.policy, steps=1)
            if env.terminated or env.truncated:
                runner.status('ENV_STOPPED_'+env.status)
                return signed(dict(status='ENV_STOPPED_'+env.status, output=str(root)))
        runner.status('STOPPED')
        return signed(dict(status='STOPPED', output=str(root)))
    except InterruptedError:
        if not stop.is_set():
            raise
        if runner:
            runner.status('STOPPED')
        return signed(dict(status='STOPPED', output=str(root)))
    except BaseException as error:
        write_json(root/'failure.json', dict(error_type=type(error).__name__, error=str(error)))
        if runner and runner.state_status != 'RPC_RECONCILIATION_REQUIRED':
            runner.status('FAILED')
        raise
    finally:
        if runner:
            runner.close()
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
        lock.close()
