"""Thinking Student V1: native account snapshot for the Account Encoder (A2).

The Native Env stays the only financial authority. This module only reads the
actual account and reward-process state after a step and exposes it as a
fixed float vector; it never changes the ledger, mask or reward.

Besides the held position it carries what decides future ADD legality and
forced exits (Review R3): the exchange-calendar minutes left until ADD closes
and until a held position is force-closed (known in advance, causal), and
how far funds and equity cover one more ADD and the held margin. Both cover
ratios saturate at 4, so every well-funded account reads the same values.
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
    'to_add_deadline',      # log1p(calendar minutes until ADD closes) / log1p(43200)
    'to_forced_exit',       # log1p(calendar minutes until main roll / expiry exit) / log1p(43200)
    'funds_cover',          # min(available funds / one 3-lot ADD's margin and fees, 4) / 4
    'margin_cover',         # min(equity / margin of the held lots, 4) / 4; 1 when flat
)
ACCOUNT_NATIVE_KEY = 'account_native'
ACCOUNT_NATIVE_SHAPE = (len(ACCOUNT_NATIVE_FIELDS),)
_AGE_SCALE = np.log1p(43200)
# Fixed input scales of the position distances (the default TP5 / SL30), not
# the env's rule: a policy sees the same inputs under any TP / SL.
_TP_SCALE, _SL_SCALE = 5., 30.
_COVER_CAP = 4.


def _minutes(env, start, stop, side):
    """Exchange-calendar minutes between two times (0 when stop has passed)."""
    calendar = env.terms.calendar
    return max(0, int(calendar.searchsorted(stop, side=side))
               - int(calendar.searchsorted(start, side=side)))


def _cover(value, need):
    return min(max(value / need, 0.), _COVER_CAP) / _COVER_CAP


def account_native(env) -> np.ndarray:
    """Return the decision-time account vector for the current env cursor."""
    out = np.zeros(ACCOUNT_NATIVE_SHAPE, np.float32)
    time = env._raw.index[env.current_step]
    out[1] = {'SHORT': -1., 'NO_TRADE': 0., 'LONG': 1.}[env.direction_at(time)]
    out[2] = float(env.net_lots) / float(env.max_lots)
    # ADD is legal while the bar end is before last_open and the main exit;
    # a held position is closed at the first bar ending at the main exit or
    # the contract liquidation time.
    add_close = min(t for t in (env.terms.last_open, env.main_exit_at) if t is not None)
    forced = min(t for t in (env.main_exit_at, env.terms.liquidate) if t is not None)
    out[10] = np.log1p(_minutes(env, time, add_close, 'left')) / _AGE_SCALE
    out[11] = np.log1p(_minutes(env, time, forced, 'right')) / _AGE_SCALE
    terms = env._terms_at(time)
    price = float(env._market_row(env.current_step).Close)
    equity = float(env._calculate_equity())
    multiplier, rate = float(env.terms.multiplier), float(terms['margin_rate'])
    held_margin = float(env.net_lots) * multiplier * price * rate
    available = equity - max(float(env.user_accounts.margin), held_margin)
    out[12] = _cover(available, 3 * multiplier * price * rate + 3 * float(terms['open_fee']))
    out[13] = _cover(equity, held_margin) if env.net_lots else 1.
    if env.entries:
        _position(env, out)
    else:
        out[0] = 1.
    if not np.isfinite(out).all():
        raise ValueError('Native account snapshot is not finite')
    return out


def _position(env, out):
    if len(env.entries) != 1:
        raise ValueError('Thinking Student V1 expects the single3 single-position protocol')
    item = next(iter(env.entries.values()))
    sign = 1. if item['side'] == 'LONG' else -1.
    positions = (env.position_manager.long_positions if item['side'] == 'LONG'
                 else env.position_manager.short_positions)
    pos = positions[item['slot']]
    price = float(env.current_price)
    out[3] = sign * (float(pos.take_profit_price) - price) / _TP_SCALE
    out[4] = sign * (price - float(pos.stop_loss_price)) / _SL_SCALE
    out[5] = sign * (price - float(item['entry_price'])) / _SL_SCALE
    held = env.market_clock_counter.held(item['entry_cursor'], env.current_step)
    out[6] = np.log1p(max(held, 0)) / _AGE_SCALE
    scorer = env.reward_function.scorer
    active = scorer.active.get(item['entry_id'])
    if active is not None:
        out[7] = np.log1p(active.bars_seen) / _AGE_SCALE
        out[8] = float(active.adverse10_fired)
    crossed = getattr(scorer, '_quality_crossed_days', {}).get(item['entry_id'], 0)
    out[9] = min(crossed, 5) / 5.


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
