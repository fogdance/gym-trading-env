"""FG S3 Quality Teacher V1: immutable, outcome-only TP5 grading.

Not an entry decision oracle.  This module has no dependency on account size,
Oracle certification, neural training or future-policy decisions.
"""
from __future__ import annotations

import math

PROTOCOL = 'FG_S3_QUALITY_TP_MARKET_DAY_V1'


def tp_base_points(market_minutes: int, later_trading_days: int) -> float:
    """Return the user's prescribed [0, 100] TP quality anchor (no MAE deduction).

    market_minutes includes every *valid* held bar; later_trading_days counts
    official trading-date transitions rather than natural midnights.
    """
    if (type(market_minutes) is not int or market_minutes < 1 or
            type(later_trading_days) is not int or later_trading_days < 0):
        raise ValueError('TP needs positive integral market minutes and nonnegative integral trading days')
    t, d = market_minutes, later_trading_days
    if d >= 1:
        return float(max(20, 45 - 5 * (d - 1)))
    if t <= 10:
        return 100.0
    if t <= 74:
        return 100.0 - 10.0 * (t - 10) / 64.0
    if t <= 120:
        return 90.0 - 5.0 * (t - 74) / 46.0
    if t <= 240:
        return 85.0 - 10.0 * (t - 120) / 120.0
    return 60.0 + 15.0 * math.exp(-(t - 240) / 120.0)


def tp_event_reward(market_minutes: int, later_trading_days: int) -> float:
    return tp_base_points(market_minutes, later_trading_days) / 100.0


def quality_for_exit(reason: str, market_minutes: int | None,
                     later_trading_days: int | None) -> tuple[float | None, str]:
    """Single-trade *outcome* score, not a WAIT/ADD instruction.

    STOP_LOSS is an observed failed entry. Other exit or right-censor has no
    certified comparable quality score; never encode unknown as zero or WAIT.
    """
    if reason == 'TAKE_PROFIT':
        if market_minutes is None or later_trading_days is None:
            raise ValueError('TP exit lacks authenticated time/day clocks')
        return tp_base_points(market_minutes, later_trading_days), 'OBSERVED_TP'
    if reason == 'STOP_LOSS':
        return 0.0, 'OBSERVED_SL'
    return None, 'OTHER_OR_CENSORED'
