"""Thinking Student V1: native account snapshot for the Account Encoder (A2).

The Native Env stays the only financial authority. This module only reads the
actual account and reward-process state after a step and exposes it as a
fixed float vector; it never changes the ledger, mask or reward.
"""
from __future__ import annotations

import numpy as np

ACCOUNT_NATIVE_FIELDS = (
    'flat',                 # 1 when no lots are held
    'direction',            # -1 SHORT, +1 LONG, 0 NO_TRADE
    'held_ratio',           # held lots / max lots
    'tp_distance',          # favorable points still needed for TP / 5
    'sl_distance',          # adverse points left before SL / 30
    'unrealized_points',    # signed points from entry / 30
    'held_minutes',         # log1p(held market minutes) / log1p(43200)
    'reward_bars_seen',     # log1p(reward scorer bars) / log1p(43200)
    'adverse10_fired',      # FIRST_ADVERSE10 penalty already triggered
    'crossed_days',         # Quality TP crossed official trading days, capped at 5, / 5
)
ACCOUNT_NATIVE_KEY = 'account_native'
ACCOUNT_NATIVE_SHAPE = (len(ACCOUNT_NATIVE_FIELDS),)
_AGE_SCALE = np.log1p(43200)


def account_native(env) -> np.ndarray:
    """Return the decision-time account vector for the current env cursor."""
    out = np.zeros(ACCOUNT_NATIVE_SHAPE, np.float32)
    time = env._raw.index[env.current_step]
    out[1] = {'SHORT': -1., 'NO_TRADE': 0., 'LONG': 1.}[env.direction_at(time)]
    out[2] = float(env.net_lots) / float(env.max_lots)
    if not env.entries:
        out[0] = 1.
        return out
    if len(env.entries) != 1:
        raise ValueError('Thinking Student V1 expects the single3 single-position protocol')
    item = next(iter(env.entries.values()))
    sign = 1. if item['side'] == 'LONG' else -1.
    positions = (env.position_manager.long_positions if item['side'] == 'LONG'
                 else env.position_manager.short_positions)
    pos = positions[item['slot']]
    price = float(env.current_price)
    out[3] = sign * (float(pos.take_profit_price) - price) / 5.
    out[4] = sign * (price - float(pos.stop_loss_price)) / 30.
    out[5] = sign * (price - float(item['entry_price'])) / 30.
    held = env.market_clock_counter.held(item['entry_cursor'], env.current_step)
    out[6] = np.log1p(max(held, 0)) / _AGE_SCALE
    scorer = env.reward_function.scorer
    active = scorer.active.get(item['entry_id'])
    if active is not None:
        out[7] = np.log1p(active.bars_seen) / _AGE_SCALE
        out[8] = float(active.adverse10_fired)
    crossed = getattr(scorer, '_quality_crossed_days', {}).get(item['entry_id'], 0)
    out[9] = min(crossed, 5) / 5.
    if not np.isfinite(out).all():
        raise ValueError('Native account snapshot is not finite')
    return out


def replay_account_native(directory, env_factory):
    """(stepids, account_native) for every row of a recorded Native policy trace.

    Re-executes the recorded actions with the recorder's own step function
    (`fg_replay.execute_observed`) from the trace's initial account, requires
    every replayed observation to equal the recorded one, and reads the
    account at each row's decision state. Observation equality is the check;
    no file hash or identity gate is involved.
    """
    import json
    from pathlib import Path
    from .fg_replay import execute_observed, iter_replay_rows
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    initial = (json.loads((directory / 'initial_state.json').read_text())
               if manifest.get('initial_state') else None)
    env = env_factory()
    stepids, values, previous = [], [], None
    try:
        if initial is not None:
            env.import_state(initial)
        for ordinal, step in iter_replay_rows(directory, manifest):
            action = None if previous is None or previous['is_last'] else int(previous['action'])
            obs, _ = execute_observed(env, action, seed=0 if ordinal == 0 else None,
                                      resume_state=initial if ordinal == 0 else None)
            for key, value in obs.items():
                if not np.array_equal(step[key], value):
                    raise ValueError(f'{directory}: replayed {key} differs at row {ordinal}')
            stepids.append(np.asarray(step['stepid']))
            values.append(account_native(env))
            previous = step
    finally:
        env.close()
    return np.stack(stepids), np.stack(values)


class StudentEnv:
    """Native env plus the account snapshot key; step/reset semantics unchanged."""

    def __init__(self, env):
        self.env = env

    def _wrap(self, obs):
        return {**obs, ACCOUNT_NATIVE_KEY: account_native(self.env)}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._wrap(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._wrap(obs), reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)
