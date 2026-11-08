"""V3 long trend exit (user 2026-11-06, "这个是新的策略，做长线，而不是短线"; the entry is unchanged):
no TP; the stop of an open position moves with the troughs (a SHORT; a LONG mirrors with
the peaks) of the complete 15m buckets formed after the fill:

    0 troughs            the entry's own stop (the rule gate's: H + 5 or fill + 5)
    price 8 from fill    break-even (the fill)
    1 trough             break-even
    2 troughs            fill - 5 (5 points locked)
    >= 3 troughs         the last trough but one ("倒数第二个波谷")

Optional (user 2026-11-08: "0个波谷，止损5点 ... 价格离成交价 8 点，或 1 个波谷，成交价-1 ... 持仓直到被迫平仓或打止损"):
`breakeven_offset_points` (default 0) puts the break-even stop that many points past the fill (fill - 1
for a SHORT with 1); `hold_after_breakeven` (default false) stops there: no LOCK and no TROUGH stage,
the position is held until the stop or the forced close (MAIN_ROLL).

The stop only ever moves in the position's favour. Swings are those of
gym_trading_env.market_structure (the review-future rule, left / right 2 by default),
known at the close of the bucket that confirms them (confirm_index) and only once that
bucket is available; a swing counts after the fill when its bucket starts at or after
the fill time. Short space: a LONG's prices are negated (ts_v2_positive.sign)."""
from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd

from gym_trading_env.market_structure import analyze_market_structure

KEYS = ('kind', 'timeframe', 'pivot_left', 'pivot_right', 'breakeven_points', 'lock_points')
OPTIONAL = ('breakeven_offset_points', 'hold_after_breakeven')
STAGES = ('INITIAL', 'BREAKEVEN', 'LOCK', 'TROUGH')
# Its distances, in ticks (envs.entry_product: times the product's tick_points).
TICKS = (('breakeven_points',), ('lock_points',))


def check_config(cfg):
    """entry.exit_rule: exactly KEYS, kind swing_trail, timeframe 15m, positive integers."""
    if (not isinstance(cfg, dict) or not set(KEYS) <= set(cfg) <= set(KEYS + OPTIONAL) or cfg['kind'] != 'swing_trail'
            or cfg['timeframe'] != '15m'):
        raise ValueError(f'entry.exit_rule needs exactly {KEYS} (and optionally {OPTIONAL}), kind swing_trail, timeframe 15m')
    off = cfg.get('breakeven_offset_points', 0)
    if type(off) is not int or not 0 <= off < cfg.get('breakeven_points', 1) or type(cfg.get('hold_after_breakeven', False)) is not bool:
        raise ValueError('entry.exit_rule: breakeven_offset_points is an integer in [0, breakeven_points), '
                         'hold_after_breakeven true or false')
    if any(type(cfg[k]) is not int or cfg[k] < 1 for k in KEYS[2:]):
        raise ValueError('entry.exit_rule: pivot_left, pivot_right, breakeven_points, lock_points are positive integers')
    return dict(cfg)


class Swings:
    """The H / L swing chain of a market's complete 15m buckets as known at any time:
    one analysis of all buckets, replayed by confirm_index (the same as analysing each
    prefix, tests/entry_v1/test_swing_trail.py)."""

    def __init__(self, buckets, left, right):
        self.buckets = buckets
        bars = [dict(open=b['Open'], high=b['High'], low=b['Low'], close=b['Close']) for b in buckets]
        self.result = analyze_market_structure(bars, left=left, right=right)
        self.start_ns = np.array([pd.Timestamp(b['bucket_start']).value for b in buckets], np.int64)
        # A bucket is known from its arrival on; later buckets never earlier.
        avail = np.array([pd.Timestamp(b['available_at']).value for b in buckets], np.int64)
        self.known_ns = np.maximum.accumulate(avail) if len(avail) else avail
        self._chains = {}

    def chain(self, b):
        """The swing chain after the close of bucket b (b < 0: none)."""
        if b not in self._chains:
            chain = []
            for entry in self.result['swing_log']:
                p = entry['point']
                if p['confirm_index'] > b:
                    break
                if entry['action'] == 'added':
                    chain.append(p)
                elif entry['action'] == 'replaced':
                    chain[-1] = p
            self._chains[b] = chain
        return self._chains[b]

    def known(self, now_ns):
        """Index of the last bucket available at now_ns (-1: none)."""
        return int(np.searchsorted(self.known_ns, now_ns, side='right')) - 1

    def after(self, kind, fill_ns, now_ns):
        """Prices of the swings of `kind` (L troughs / H peaks) known at now_ns whose
        bucket starts at or after the fill, oldest first."""
        return [p['price'] for p in self.chain(self.known(now_ns))
                if p['kind'] == kind and self.start_ns[p['pivot_index']] >= fill_ns]


def trail_stop(fill, stop, stage, best, troughs, cfg):
    """Short space (a SHORT's prices; a LONG's negated): the stop after a closed bar.
    fill / stop: the fill and the current stop (above the fill at first); best: the
    lowest price reached since the fill; troughs: the prices of the troughs after the
    fill known now, oldest first. Returns (stop, stage); the stop never rises."""
    fill, stop, best = Decimal(str(fill)), Decimal(str(stop)), Decimal(str(best))
    k = len(troughs)
    candidates = []
    hold = cfg.get('hold_after_breakeven', False)
    if fill - best >= cfg['breakeven_points'] or k == 1 or (hold and k >= 1):
        candidates.append((fill - cfg.get('breakeven_offset_points', 0), 'BREAKEVEN'))
    if not hold and k == 2:
        candidates.append((fill - cfg['lock_points'], 'LOCK'))
    if not hold and k >= 3:
        candidates.append((Decimal(str(troughs[-2])), 'TROUGH'))
    for price, name in candidates:
        if price < stop:
            stop, stage = price, name
    return stop, stage
