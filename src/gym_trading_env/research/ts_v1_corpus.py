"""Thinking Student V1 teaching material for one TRAIN contract (C3 §5).

Everything that depends only on the market is computed once per contract and
indexed by the market cursor:

  market/   the causal E observation of every cursor (policy market input)
  facts/    the Native "ADD now, then hold" trade from a flat account (L_outcome)
  strict/   the strict TRAIN oracle's WAIT/ADD from a flat account (the P4 plan)
  graded/   the graded teacher's WAIT/ADD of that trade (L_policy_demo) and its
            MFE targets (L_outcome); thresholds given (ts_v1_grades)

A flat account whose equity is at least the funds floor gets exactly these
facts and choices at the same cursor: the Native trade depends only on the
market, and above the floor funds can neither block an ADD nor force a
margin call (`fg_single3_oracle.funds_certificate`; its TP-only optimum only
adds cash).

behaviors/<P>/ holds the account side of the behavior accounts P0-P4, one
row per cursor from the first minute to the true terminal (the TS env has no
CHUNK_END). A reader joins market rows by cursor and puts facts and graded
choices on the rows where that account is flat, may ADD and keeps the floor.

No manifests, hashes or reconcile passes: `source.json` records the TRAIN
pool row, inputs, label sources, counts and timings. Production replays a
few sampled cursors minute by minute through the Native env as the only
cross-check of the indexed facts (C3 5.9).

The market pool (TRAIN and VALIDATION rows, env files beside it) and the
grade thresholds are Thinking Student configs (dreamerv3/configs/
thinking_student/markets, entry_grades.yaml); they are always given, never
assumed by the code:

  python -m gym_trading_env.research.ts_v1_corpus --contract CZCE.FG809 \
      --pool ../dreamerv3/configs/thinking_student/markets/pool.json \
      --grades ../dreamerv3/configs/thinking_student/entry_grades.yaml \
      --out /data/logdir/fg_single3/thinking_student/corpus --workers 24

The Native TP / SL distances are those of the pool env (TP5 / SL30 unless
it sets entry.take_profit_points / stop_loss_points); --take-profit-points
and --stop-loss-points write other distances into inputs/env.yaml, so every
facts, grades, collection, CF and exam built from the corpus uses them.
Distances given are in ticks of the product (envs.entry_product, user
2026-11-08); facts and labels are in points, and source.json records the
product and its tick_points (dreamer labels.outcome_labels reads it).

V2 (V0.5, user 2026-10-31): --strategy adds a strategy overlay (dreamerv3/
configs/thinking_student/markets/strategy_v2_short_structural.yaml: the
structural stop, the 48-bucket 15m context, the net PnL reward) to the
inputs, and --teacher writes graded/ from the V2 positive material
(ts_v2_positive.graded_teacher) instead of the --grades thresholds.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from gym_trading_env.envs.entry_contract import read_yaml, resolve_path
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from gym_trading_env.rewards.single3_reward import QUALITY_TP_SCHEMA
from .fg_reward_candidates import quality_tp_reward
from .fg_s3_quality_teacher_v2 import official_day_index, trade_facts
from .fg_single3_oracle import GAMMA, CandidateIndex, optimize, reference_candidate
from .ts_v1_account import account_native

# Native endings of a trade; TP / SL at the env's distances
# (entry.take_profit_points / stop_loss_points, TP5 / SL30 by default).
KINDS = ('TP', 'SL', 'OTHER_EXIT', 'RIGHT_CENSORED')
MARKET_KEYS = ('market_seq_e_shared', 'market_context_15m', 'market_macd_5m')
BEHAVIORS = {
    'P0': dict(kind='always_wait'),
    'P1': dict(kind='always_add'),
    'P2_low': dict(kind='probabilistic', probability=0.1, seed=[202, 0, 2]),
    'P2_mid': dict(kind='probabilistic', probability=0.5, seed=[202, 0, 3]),
    'P3': dict(kind='single_delay', wait_minutes=15),
    'P4': dict(kind='strict_plan'),
}
# Per-cursor facts of the trade "ADD now, then hold" from a flat account.
FACTS = {
    'eligible': (bool, False), 'filled': (bool, False), 'kind': (np.int8, -1),
    'exit_reason': ('<U24', ''), 'entry_cursor': (np.int32, -1),
    'exit_cursor': (np.int32, -1), 'observed_until': (np.int32, -1),
    'held_minutes': (np.int32, 0), 'crossed_days': (np.int16, -1),
    'mae_15': (np.float32, 0.), 'mae_15_complete': (bool, False),
    'mae_60': (np.float32, 0.), 'mae_60_complete': (bool, False),
    'mae_full': (np.float32, 0.), 'path_complete': (bool, False),
    'early10_5': (bool, False), 'early10_5_valid': (bool, False),
    'early10_15': (bool, False), 'early10_15_valid': (bool, False),
    'reward': (np.float64, np.nan), 'net_pnl': (np.float64, np.nan),
    'score': (np.float32, np.nan),
}
PATH_KEYS = (('entry', 'contract_config'), ('entry', 'direction_config'),
             ('entry', 'strategy_config'), ('trading', 'data_path'))
# Where each table's supervision comes from and when it is known (C3 5.3);
# the material is OFFLINE_TRAIN, so a label is usable once TRAIN holds it.
LABELS = {
    'facts': dict(source='TEACHER_ADD_FACT', ready_at='observed_until: the exit, '
                  'or the end of observation of a censored trade'),
    'strict': dict(source='TRAIN_STRICT_ORACLE', ready_at='flat_end: the whole TRAIN suffix'),
    'graded': dict(source='GRADED_ENTRY_TEACHER', ready_at='ready_at: the exit, or the end of '
                   'the room window of a same-day TP; an MFE target at the end of its window'),
    'behaviors': dict(source='EXECUTED_NATIVE behavior accounts P0-P4', ready_at='the row'),
}
NATIVE_SAMPLE = 32  # cursors replayed minute by minute at production


# ---- inputs and env ----

def pool_row(contract, pool):
    """The contract's row of a market pool, its env file resolved against
    the pool file."""
    rows = [r for r in json.loads(Path(pool).read_text())['files']
            if r['contract'] == contract]
    if len(rows) != 1:
        raise ValueError(f'{contract}: {len(rows)} rows in pool {pool}')
    return dict(rows[0], env_config=str(resolve_path(pool, rows[0]['env_config'])))


def train_row(contract, pool):
    """The contract's TRAIN row of the pool; any other fold is refused."""
    row = pool_row(contract, pool)
    if row.get('role') != 'train' or not row.get('eligible_for_training'):
        raise ValueError(f'{contract} is not a TRAIN member of {pool}')
    return row


def dev_row(contract, pool):
    """The contract's VALIDATION row of the pool (not eligible for
    training): the market of a Teacher-free DEV exam. A TRAIN row is no DEV
    and TEST stays sealed, so any other role is refused."""
    row = pool_row(contract, pool)
    if row.get('role') != 'validation' or row.get('eligible_for_training'):
        raise ValueError(f'{contract} is not a VALIDATION row of {pool} (role {row.get("role")})')
    return row


STRATEGY_KEYS = {'version', 'direction', 'entry', 'reward'}


def load_strategy(path):
    """A V2 strategy overlay (V0.5, user 2026-10-31): `entry` keys of the
    Native env (stop_rule, context_15m_buckets, ...) and the `reward` that
    replaces the Quality TP reward, for one external `direction`: SHORT, or
    BOTH (V2.1, user 2026-11-04: each contract in the direction its env
    trades, with the two-sided stop rule prior_15m_extreme). V2.3 (user
    2026-11-05) may add `entry.rule_gate` ({SHORT, LONG}: V4 material rules
    files, relative to the strategy file, come back resolved) that opens an
    ADD only at their rule positions. Distances are in ticks and costs are
    the market's product profile's (envs.entry_product, user 2026-11-08):
    a strategy with `costs` is refused."""
    strategy = read_yaml(path)
    if 'costs' in strategy:
        raise ValueError(f'{path}: costs are the product profile\'s (envs.entry_product), not a strategy\'s')
    if set(strategy) != STRATEGY_KEYS or strategy['direction'] not in ('SHORT', 'BOTH'):
        raise ValueError(f'{path}: a SHORT or BOTH strategy of version, direction, entry, reward')
    if strategy['direction'] == 'BOTH' and strategy['entry'].get('stop_rule') != 'prior_15m_extreme':
        raise ValueError(f'{path}: a BOTH strategy needs entry.stop_rule prior_15m_extreme')
    gate = strategy['entry'].get('rule_gate')
    if gate is not None:
        if strategy['direction'] != 'BOTH' or not isinstance(gate, dict) or set(gate) != {'SHORT', 'LONG'}:
            raise ValueError(f'{path}: entry.rule_gate is {{SHORT, LONG}} rules files of a BOTH strategy')
        from .ts_v2_positive import load_config
        strategy['entry']['rule_gate'] = {side: str(resolve_path(path, f)) for side, f in gate.items()}
        for side, f in strategy['entry']['rule_gate'].items():
            if load_config(f)['direction'] != side:
                raise ValueError(f'{path}: entry.rule_gate {side} is the rules of another direction')
    return strategy


def write_inputs(row, folder, rules=None, strategy=None):
    """TS env inputs: the pool env with the Quality TP reward and without
    CHUNK_END (chunk_steps 0), so an account runs from the first minute to
    the true terminal and market memory never restarts artificially. The
    strategy, contract and direction files are copied next to env.yaml and
    referenced relatively: apart from the market data the folder is
    self-contained and can move (produce renames it into place). `rules`
    (take_profit_points / stop_loss_points, in ticks) replace the pool
    env's Native TP / SL distances. `strategy` (load_strategy) adds its
    entry keys and reward instead of the Quality TP reward; a contract the
    env trades in another direction is refused. The rules files of the
    strategy's entry.rule_gate are written next to env.yaml. The master's
    product profile (envs.entry_product, required) is written next to it
    as product.yaml, its cost preset inlined: the only costs of the inputs
    (user 2026-11-06: "应该只需要改1个地方才对")."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    source = Path(row['env_config'])
    env = read_yaml(source)
    for group, key in PATH_KEYS:
        if env.get(group, {}).get(key):
            env[group][key] = str(resolve_path(source, env[group][key]))
    if Path(env['trading']['data_path']) != Path(row['path']).resolve():
        raise ValueError(f'{row["contract"]}: env market differs from the pool row')
    env['entry']['chunk_steps'] = 0
    env['entry'].update(rules or {})
    if strategy is not None:
        from .ts_v2_positive import contract_direction
        if strategy['direction'] != 'BOTH' and contract_direction(row) != strategy['direction']:
            raise ValueError(f'{row["contract"]} is not traded {strategy["direction"]}')
        env['entry'].update(strategy['entry'])
    if env['entry'].get('rule_gate'):
        from .ts_v2_positive import load_config
        names = {}
        for side, f in env['entry']['rule_gate'].items():
            names[side] = f'rule_gate_{side.lower()}.yaml'
            (folder / names[side]).write_text(yaml.safe_dump(load_config(f), sort_keys=False, allow_unicode=True))
        env['entry']['rule_gate'] = names
    reward = dict(strategy['reward']) if strategy is not None else dict(schema=QUALITY_TP_SCHEMA)
    for key, name in (('contract_config', 'contract.yaml'), ('direction_config', 'directions.yaml')):
        if env['entry'].get(key):
            shutil.copyfile(env['entry'][key], folder / name)
            env['entry'][key] = name
    if env['entry'].get('strategy_config'):
        master = read_yaml(env['entry']['strategy_config'])
        master['reward'].update(reward)
        from gym_trading_env.envs.entry_product import master_profile, profile_copy
        profile = master_profile(master, env['entry']['strategy_config'])
        (folder / 'product.yaml').write_text(yaml.safe_dump(profile_copy(profile), sort_keys=False,
                                                            allow_unicode=True))
        master['product_profile'] = 'product.yaml'
        master['environment']['entry']['chunk_steps'] = 0
        strategy = folder / 'single3.yaml'
        strategy.write_text(yaml.safe_dump(master, sort_keys=False))
        env['entry']['strategy_config'] = strategy.name
    else:
        env['entry']['reward'].update(reward)
    config = folder / 'env.yaml'
    config.write_text(yaml.safe_dump(env, sort_keys=False))
    return config


def native_env(config, market=True):
    """The Native Single3 env of TS inputs. `market` adds the causal E market
    observation (policy input); the financial rules are the same either way."""
    profile = None
    if market:
        profile = SimpleNamespace(contracts=frozenset({read_yaml(config)['entry']['contract']}))
    return FixedDirectionEntryEnv(config_path=Path(config), e_profile=profile)


def corpus_env(corpus, market=True):
    return native_env(Path(corpus) / 'inputs' / 'env.yaml', market)


# ---- facts: one indexed Native trade per cursor ----

def quality_rewards(env, row, days, ordinal):
    """C1 feedback of the Quality TP reward for an indexed TP trade, as the
    Native Single3Reward pays it: early adverse penalty plus the TP score."""
    feedback = {}
    early = row.get('first_adverse_effective_bar')
    if early is not None and early <= 15:
        feedback[row['first_adverse_cursor']] = -env.reward_config.beta * (
            1 if early <= 5 else env.reward_config.rho)
    u, i = row['exit_cursor'], row['entry_cursor']
    feedback[u] = feedback.get(u, 0.) + quality_tp_reward(
        row['held_market_bars'], ordinal[days[u]] - ordinal[days[i]])
    return [[s, float(np.float32(v))] for s, v in sorted(feedback.items()) if v]


def candidate_facts(env, index, calendar, t):
    """The strict-oracle candidate (real entry, indexed exit, real settlement)
    with the env's feedback (Quality TP; under the V2 net PnL reward the
    candidate's own exit net PnL / scale), plus the trade facts the Outcome
    labels need."""
    row = index.candidate(t)
    out = dict(cursor=t, eligible=bool(row['eligible']), filled=bool(row.get('filled')),
               rewards={'C1': []}, exit=None, release_cursor=None, facts=None)
    if not out['filled']:
        return out
    rewards = row['rewards']['C1']
    if row['exit'] and row['exit']['reason'] == 'TAKE_PROFIT' and env.reward_config.schema == QUALITY_TP_SCHEMA:
        rewards = quality_rewards(env, row, calendar['days'], calendar['ordinal'])
    entry = dict(entry_cursor=row['entry_cursor'], price=row['entry_price'],
                 side=row['side'], time=row['entry_time'])
    facts = trade_facts(env, entry, row['exit'], row['observed_until_cursor'],
                        calendar_index=calendar)
    exit = row['exit'] and {k: row['exit'][k] for k in ('reason', 'net_trade_pnl')}
    out.update(rewards={'C1': rewards}, exit=exit, release_cursor=row['release_cursor'],
               facts=dict(
                   kind=KINDS.index(facts['exit_type']),
                   exit_reason=exit['reason'] if exit else '',
                   entry_cursor=row['entry_cursor'], exit_cursor=facts['exit_cursor'] if exit else -1,
                   observed_until=facts['observed_until_cursor'],
                   held_minutes=facts['held_market_minutes'],
                   crossed_days=facts['crossed_trading_days'] if exit else -1,
                   mae_15=facts['mae_15'], mae_15_complete=facts['mae_15_complete'],
                   mae_60=facts['mae_60'], mae_60_complete=facts['mae_window_complete'][1],
                   mae_full=facts['mae_full'], path_complete=facts['path_complete'],
                   **{k: facts[k] for k in ('early10_5', 'early10_5_valid',
                                            'early10_15', 'early10_15_valid')},
                   reward=sum(v for _, v in rewards) if exit else np.nan,
                   net_pnl=float(exit['net_trade_pnl']) if exit else np.nan,
                   score=facts['score'] if facts['score'] is not None else np.nan))
    return out


def _facts_job(job):
    config, cursors = job
    env = native_env(config, market=False)
    index = CandidateIndex(env)
    calendar = official_day_index(env)
    return [candidate_facts(env, index, calendar, t) for t in cursors]


def _native_job(job):
    config, cursors = job
    env = native_env(config, market=False)
    return [(t, reference_candidate(env, t)) for t in cursors]


def native_check(facts, t, native):
    """The first fact of cursor t that differs from its Native minute trace
    (`reference_candidate`), or None."""
    if facts['filled'][t] != native['filled']:
        return 'filled'
    if not native['filled']:
        return None
    events = facts['reward_events']
    if events[events[:, 0] == t][:, 1:].tolist() != native['rewards']:
        return 'rewards'
    kind = 'RIGHT_CENSORED' if native['exit'] is None else {
        'TAKE_PROFIT': 'TP', 'STOP_LOSS': 'SL'}.get(native['exit']['reason'], 'OTHER_EXIT')
    if KINDS[facts['kind'][t]] != kind:
        return 'kind'
    if native['exit'] is not None and facts['exit_cursor'][t] != native['exit_cursor']:
        return 'exit_cursor'
    return None


def fact_arrays(rows, n):
    arrays = {k: np.full(n, default, dtype) for k, (dtype, default) in FACTS.items()}
    events = []
    for row in rows:
        t = row['cursor']
        arrays['eligible'][t], arrays['filled'][t] = row['eligible'], row['filled']
        for key, value in (row['facts'] or {}).items():
            arrays[key][t] = value
        events += [(t, s, v) for s, v in row['rewards']['C1']]
    arrays['reward_events'] = np.array(events, np.float64).reshape(-1, 3)
    return arrays


# ---- strict TRAIN oracle choice from a flat account ----

def _zero_feedback(row, clock):
    """The exact discounted C1 feedback of a filled ADD is 0 (e.g. none)."""
    events = row['rewards']['C1']
    if all(v > 0 for _, v in events) != all(v < 0 for _, v in events):
        return False  # one sign only: nonzero (an empty list is both)
    t = row['cursor']
    return sum((GAMMA ** clock.reward_exponent(t, s) * Fraction.from_float(v)
                for s, v in events), Fraction(0)) == 0


def strict_arrays(env, rows, flat_end, n):
    """first[t] == t: the exact TRAIN optimum from a flat funded account at t
    starts with ADD now (the optimizer keeps only cash-positive TP trades;
    any other trade is never better than WAIT). A tie, which demonstrates
    nothing, is an ADD whose value equals WAIT exactly: a rejected ADD, an
    exact tie the optimizer resolved to WAIT, or a trade with zero discounted
    feedback when the optimum from t + 1 starts no trade before its release.
    A trade that loses, or blocks a better trade, is a strict WAIT."""
    plan = optimize(rows, flat_end, env.reward_config.candidate,
                    clock=env.market_clock_counter, return_suffix=True)
    suffix = plan.pop('suffix')
    out = dict(valid=np.zeros(n, bool), action=np.zeros(n, np.int8),
               tie=np.zeros(n, bool), value=np.full(n, np.nan))
    first, tied = suffix['first'], set(suffix['tied'])
    for row in rows:
        t = row['cursor']
        if not row['eligible']:
            continue
        out['valid'][t] = True
        out['action'][t] = int(first[t] == t)
        out['tie'][t] = not row['filled'] or t in tied or (
            row['release_cursor'] is not None and _zero_feedback(row, env.market_clock_counter)
            and (first[t + 1] is None or first[t + 1] >= row['release_cursor']))
    bounds = np.asarray(suffix['value_intervals'], np.float64)
    out['value'][:len(bounds)] = bounds.mean(-1)
    out['plan'] = np.asarray(plan['add_cursors'], np.int32)
    return out, plan


# ---- behavior accounts ----

class Behavior:
    """P0 never ADDs; P1 ADDs whenever legal; P2 ADDs a legal state with a
    fixed probability; P3 ADDs after 15 valid minutes flat and legal; P4
    executes the strict plan from the initial account."""

    def __init__(self, spec, plan=()):
        self.kind = spec['kind']
        self.rng = np.random.default_rng(spec.get('seed', 0))
        self.spec, self.plan, self.waited = spec, set(int(t) for t in plan), 0

    def __call__(self, env, obs):
        legal = obs['action_mask'][1] > 0.5
        flat = obs['position_state'][1] == 0
        if self.kind == 'always_wait':
            return 0
        if self.kind == 'always_add':
            return int(legal)
        if self.kind == 'probabilistic':
            return int(legal and self.rng.random() < self.spec['probability'])
        if self.kind == 'single_delay':
            if not (legal and flat):
                self.waited = 0
                return 0
            if self.waited >= self.spec['wait_minutes']:
                self.waited = 0
                return 1
            self.waited += int(env.valid_fill_clock[env.current_step])
            return 0
        if self.kind == 'strict_plan':
            if env.current_step in self.plan:
                if not legal:
                    raise ValueError(f'Strict plan ADD at {env.current_step} is not legal')
                return 1
            return 0
        raise ValueError(f'Unknown behavior {self.kind}')


def run_behavior(env, policy, market=False):
    """One account from reset to its terminal: per decision cursor the
    account state, the action and the reward that action produced."""
    obs, _ = env.reset(seed=0)
    rows, views, done = [], [], False
    while True:
        rows.append(dict(cursor=env.current_step, position_state=obs['position_state'],
                         action_mask=obs['action_mask'], account_native=account_native(env),
                         equity=float(env._calculate_equity()), status=env.status,
                         action=0, reward=0., is_last=done, is_terminal=bool(env.terminated)))
        if market:
            views.append({k: obs[k] for k in MARKET_KEYS})
        if done:
            break
        action = policy(env, obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        rows[-1].update(action=action, reward=float(np.float32(reward)))
        done = terminated or truncated
    arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    arrays['cursor'] = arrays['cursor'].astype(np.int32)
    arrays['action'] = arrays['action'].astype(np.int8)
    arrays['reward'] = arrays['reward'].astype(np.float32)
    arrays['is_first'] = np.arange(len(rows)) == 0
    market = {k: np.stack([v[k] for v in views]) for k in MARKET_KEYS} if market else None
    summary = dict(rows=len(rows), adds=int((arrays['action'] == 1).sum()),
                   trades=len(env.attributions), exits=_count(e['reason'] for e in env.attributions),
                   reward=float(arrays['reward'].astype(np.float64).sum()),
                   final_equity=float(env._calculate_equity()), status=env.status)
    return arrays, market, summary


def _count(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def _behavior_job(job):
    config, name, spec, plan, folder = job
    began = time.monotonic()
    market = name == 'P0'
    env = native_env(config, market=market)
    arrays, views, summary = run_behavior(env, Behavior(spec, plan), market)
    save_arrays(Path(folder) / 'behaviors' / name, arrays)
    if views is not None:
        save_arrays(Path(folder) / 'market', views)
    summary['seconds'] = round(time.monotonic() - began, 1)
    return name, summary


# ---- production ----

def save_arrays(folder, arrays):
    folder.mkdir(parents=True, exist_ok=True)
    for key, value in arrays.items():
        np.save(folder / f'{key}.npy', value)


def load_arrays(folder, mmap=None):
    return {p.stem: np.load(p, mmap_mode=mmap) for p in sorted(Path(folder).glob('*.npy'))}


def _repos():
    out = {}
    root = Path(__file__).resolve().parents[3]
    for name, path in (('gym', root), ('dreamerv3', root.parent / 'dreamerv3')):
        try:
            sha = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
            dirty = bool(subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True))
            out[name] = dict(sha=sha, dirty=dirty)
        except (OSError, subprocess.CalledProcessError):
            out[name] = None
    return out


def _pool(workers):
    # One native thread per worker; the parent keeps its own settings.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(key, '1')
    return ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'))


def produce(row, folder, grades, workers=None, behaviors=None, log=print, rules=None,
            strategy=None, teacher=None):
    """Write the teaching material of one TRAIN pool row into `folder`;
    `grades` are the thresholds of the graded teacher and `rules` the Native
    TP / SL distances when they differ from the pool env's. `strategy` (a
    load_strategy path) sets a V2 strategy; `teacher` (a V2 positive
    material folder) writes graded/ from it instead of the graded teacher."""
    from . import ts_v1_grades  # it reads this module
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(f'{folder} exists; remove it to regenerate')
    tmp = folder.with_name(folder.name + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    began, seconds = time.monotonic(), {}
    config = write_inputs(row, tmp / 'inputs', rules, strategy and load_strategy(strategy))
    env = native_env(config, market=False)
    index = CandidateIndex(env)
    n, flat_end = len(env._raw), index.flat_end
    workers = max(1, min(workers or os.cpu_count(), flat_end))
    with _pool(workers) as pool:
        shards = [(str(config), list(range(w, flat_end, workers))) for w in range(workers)]
        rows = sorted((r for part in pool.map(_facts_job, shards) for r in part),
                      key=lambda r: r['cursor'])
        seconds['facts'] = round(time.monotonic() - began, 1)
        facts = fact_arrays(rows, n)
        eligible = np.flatnonzero(facts['eligible'])
        sample = sorted(np.random.default_rng(202).choice(
            eligible, min(NATIVE_SAMPLE, len(eligible)), replace=False).tolist())
        jobs = [(str(config), part) for part in (sample[w::workers] for w in range(workers)) if part]
        traces = [x for part in pool.map(_native_job, jobs) for x in part]
    differ = {t: d for t, native in traces if (d := native_check(facts, t, native))}
    if differ:
        raise ValueError(f'Indexed facts differ from the Native minute trace: {differ}')
    seconds['native_check'] = round(time.monotonic() - began - seconds['facts'], 1)
    started = time.monotonic()
    strict, plan = strict_arrays(env, rows, flat_end, n)
    seconds['strict'] = round(time.monotonic() - started, 1)
    started = time.monotonic()
    if teacher is None:
        graded, summary = ts_v1_grades.grade(env, facts, grades, index=index, strict=strict)
        summary = ts_v1_grades.save(tmp / 'graded', row['contract'], graded, summary, started)
    else:
        from .ts_v2_positive import contract_direction, graded_teacher
        graded, summary = graded_teacher(env, facts, teacher, row['contract'], contract_direction(row),
                                         index=index)
        summary.update(seconds=round(time.monotonic() - started, 1),
                       created=time.strftime('%Y-%m-%d %H:%M:%S'), repos=_repos())
        save_arrays(tmp / 'graded', graded)
        (tmp / 'graded' / 'summary.json').write_text(json.dumps(summary, indent=1))
    seconds['graded'] = summary['seconds']
    save_arrays(tmp / 'facts', facts)
    save_arrays(tmp / 'strict', strict)
    save_arrays(tmp / 'market', dict(
        time=np.array([str(t) for t in env._raw.index]),
        valid_bar=np.asarray(env.valid_fill_clock, bool)))
    specs = {k: BEHAVIORS[k] for k in (behaviors or BEHAVIORS)}
    jobs = [(str(config), name, spec, strict['plan'].tolist(), str(tmp))
            for name, spec in specs.items()]
    started = time.monotonic()
    with _pool(len(jobs)) as pool:
        summaries = dict(pool.map(_behavior_job, jobs))
    seconds['behaviors'] = round(time.monotonic() - started, 1)
    for name in specs:
        cursor = np.load(tmp / 'behaviors' / name / 'cursor.npy')
        if not np.array_equal(cursor, np.arange(len(cursor))):
            raise ValueError(f'{name}: behavior rows are not one per consecutive cursor')
    kinds = facts['kind'][facts['filled']]
    proof = index.proof
    source = dict(
        # An env without a master strategy (a synthetic market) has no product; its tick is its contract's.
        contract=row['contract'], product=env.product and env.product['product'], tick_points=env.tick_points,
        fold='TRAIN', mode='OFFLINE_TRAIN', pool_row=row,
        inputs=dict(env=str(config.relative_to(tmp)), reward_schema=env.reward_config.schema,
                    chunk_steps=0, candidate=env.reward_config.candidate,
                    # The Native exit rule: the fixed distances, or the V2 structural stop.
                    rules=(dict(stop_rule=env.stop_rule) if env.stop_rule else
                           dict(take_profit_points=env.take_profit_points,
                                stop_loss_points=env.stop_loss_points)),
                    strategy=strategy and dict(load_strategy(strategy), path=str(Path(strategy).resolve()))),
        rows=n, flat_end=flat_end, first_time=str(env._raw.index[0]),
        last_time=str(env._raw.index[-1]),
        labels=dict(LABELS, graded=summary['label']) if teacher else LABELS,
        native_check=dict(cursors=sample, agree=len(sample)),
        funds_floor=float(proof['required_equity_upper_bound']),
        initial_balance=float(env.config.trading.initial_balance),
        gamma=[332, 333], clock='S3_MARKET_MINUTE',
        facts=dict(eligible=int(facts['eligible'].sum()), filled=int(facts['filled'].sum()),
                   kinds={k: int((kinds == i).sum()) for i, k in enumerate(KINDS)}),
        strict=dict(valid=int(strict['valid'].sum()),
                    add=int((strict['valid'] & ~strict['tie'] & (strict['action'] == 1)).sum()),
                    wait=int((strict['valid'] & ~strict['tie'] & (strict['action'] == 0)).sum()),
                    tie=int((strict['valid'] & strict['tie']).sum()),
                    plan_trades=len(strict['plan']), plan_cash_gain=plan['selected_cash_gain'],
                    exact_ties=plan['exact_ties']),
        graded={k: summary[k] for k in ('grades', 'graded', 'demo')},
        behaviors={name: {**BEHAVIORS[name], **summaries[name]} for name in specs},
        seconds=dict(seconds, total=round(time.monotonic() - began, 1)), workers=workers,
        created=time.strftime('%Y-%m-%d %H:%M:%S'), repos=_repos())
    (tmp / 'source.json').write_text(json.dumps(source, indent=1))
    tmp.rename(folder)
    log(json.dumps({k: source[k] for k in ('facts', 'strict', 'graded', 'seconds')}, indent=1))
    return source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--contract', required=True)
    parser.add_argument('--out', required=True, help='corpus root; writes <out>/<contract>')
    parser.add_argument('--pool', required=True,
                        help='market pool (dreamerv3/configs/thinking_student/markets/pool.json)')
    parser.add_argument('--grades',
                        help='graded teacher (dreamerv3/configs/thinking_student/entry_grades.yaml)')
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--take-profit-points', type=int,
                        help='Native TP distance in ticks (default: the pool env, TP5)')
    parser.add_argument('--stop-loss-points', type=int,
                        help='Native SL distance in ticks (default: the pool env, SL30)')
    parser.add_argument('--strategy', help='V2 strategy overlay (configs/thinking_student/markets/'
                        'strategy_v2_short_structural.yaml)')
    parser.add_argument('--teacher', help='V2 positive material folder: graded/ from it, not --grades')
    args = parser.parse_args(argv)
    from .ts_v1_grades import load_grades
    if not (args.grades or args.teacher):
        parser.error('--grades or --teacher is required')
    rules = {k: v for k, v in (('take_profit_points', args.take_profit_points),
                               ('stop_loss_points', args.stop_loss_points)) if v is not None}
    if args.strategy and rules:
        parser.error('--strategy sets the exit rule: no --take-profit-points / --stop-loss-points')
    if args.teacher and not args.strategy:
        parser.error('--teacher (the V2 positive material) needs --strategy (its structural stop)')
    if args.teacher and args.grades:
        parser.error('--teacher or --grades, not both')
    produce(train_row(args.contract, args.pool), Path(args.out) / args.contract,
            args.grades and load_grades(args.grades), args.workers, rules=rules, strategy=args.strategy,
            teacher=args.teacher)


if __name__ == '__main__':
    main()
