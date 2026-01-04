# tests/oracles/agent_spec_oracle.py

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext, ROUND_HALF_UP
from typing import Any, Sequence, Optional, Dict, List

import numpy as np


getcontext().prec = 28
getcontext().rounding = ROUND_HALF_UP

D0 = Decimal("0")


def D(x: Any) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))


def _clip_dec(x: Decimal, lo: Decimal, hi: Decimal) -> Decimal:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _safe_div(a: Decimal, b: Decimal) -> Decimal:
    return a / b if b != D0 else D0


def _sum_sizes_vwap_age(positions: Sequence[Any], current_step: int):
    total = D0
    vwap_num = D0
    age_num = D0
    for p in positions:
        if p is None:
            continue
        size = D(getattr(p, "size"))
        entry = D(getattr(p, "entry_price"))
        open_step = int(getattr(p, "open_step"))
        total += size
        vwap_num += size * entry
        age_num += size * D(current_step - open_step)
    if total == D0:
        return D0, D0, D0
    return total, (vwap_num / total), (age_num / total)


def _active_side_snapshot(long_positions, short_positions, current_step: int):
    L, L_vwap, L_age = _sum_sizes_vwap_age(long_positions, current_step)
    S, S_vwap, S_age = _sum_sizes_vwap_age(short_positions, current_step)

    if L > D0 and S == D0:
        return {"side": "long", "size": L, "vwap": L_vwap, "age_min": L_age}
    if S > D0 and L == D0:
        return {"side": "short", "size": S, "vwap": S_vwap, "age_min": S_age}
    if L == D0 and S == D0:
        return {"side": None, "size": D0, "vwap": D0, "age_min": D0}

    if L > S:
        return {"side": "long", "size": L, "vwap": L_vwap, "age_min": L_age}
    if S > L:
        return {"side": "short", "size": S, "vwap": S_vwap, "age_min": S_age}

    latest_L = max((int(getattr(p, "open_step")) for p in long_positions if p is not None), default=-1)
    latest_S = max((int(getattr(p, "open_step")) for p in short_positions if p is not None), default=-1)
    return {"side": "long", "size": L, "vwap": L_vwap, "age_min": L_age} if latest_L >= latest_S else {
        "side": "short", "size": S, "vwap": S_vwap, "age_min": S_age
    }


def _calc_unrealized_pnl(current_price: Decimal, pos: Any, lot_size: Decimal, long: bool) -> Decimal:
    if pos is None:
        return D0
    entry = D(getattr(pos, "entry_price"))
    size = D(getattr(pos, "size"))
    if long:
        return (current_price - entry) * size * lot_size
    return (entry - current_price) * size * lot_size


def compute_unrealized_pnl_spec(long_positions, short_positions, current_price: Decimal, lot_size: Decimal) -> Decimal:
    upnl_long = sum((_calc_unrealized_pnl(current_price, p, lot_size, True) for p in long_positions if p is not None), D0)
    upnl_short = sum((_calc_unrealized_pnl(current_price, p, lot_size, False) for p in short_positions if p is not None), D0)
    return upnl_long + upnl_short


@dataclass(frozen=True)
class AgentSnapshot:
    long_positions: Sequence[Any]
    short_positions: Sequence[Any]

    current_step: int
    current_price: Decimal
    lot_size: Decimal

    realized_pnl_step: Decimal
    realized_pnl_cum: Decimal
    fee_step: Decimal
    fee_cum: Decimal

    cash_balance: Decimal
    used_margin: Decimal
    prev_max_equity: Decimal

    entries_used_today: int = 0
    max_entries_per_day: int = 1
    minutes_to_eod: int = 0
    day_len: int = 1

    initial_balance: Decimal = D0
    realized_today_cash: Decimal = D0
    R_cash: Decimal = D0

    market_open: int = 1
    can_long_open: int = 0
    can_short_open: int = 0
    can_long_close: int = 0
    can_short_close: int = 0


    action_result_code: int = 0
    action_result_max_code: int = 1  # ForexCode.max(value)


FEATURES_AGENT: List[str] = [
    "pos_t",
    "have_long_t",
    "have_short_t",
    "entry_price_t",
    "holding_minutes_t",
    "upnl_t",
    "realized_pnl_step_t",
    "realized_pnl_cum_t",
    "fee_step_t",
    "fee_cum_t",
    "equity_t",
    "max_equity_t",
    "drawdown_t",
    "sigma_entry_t",
    "sl_ticks_t",
    "tp_ticks_t",
    "sl_price_t",
    "tp_price_t",
    "minutes_to_timeout_t",
]

FEATURES_AGENT_OBS: List[str] = [
    "obs_market_open_t",
    "obs_can_long_open_t",
    "obs_can_short_open_t",
    "obs_can_long_close_t",
    "obs_can_short_close_t",

    "obs_pos_side_t",
    "obs_entries_left_frac_t",
    "obs_minutes_to_eod_frac_t",
    "obs_holding_frac_t",
    "obs_upnl_R_t",
    "obs_realized_today_R_t",
    "obs_equity_frac_t",
    "obs_drawdown_frac_t",
    "obs_action_result_t",
]


def compute_agent_features_raw_spec(s: AgentSnapshot) -> Dict[str, Decimal]:
    long_lots, _, _ = _sum_sizes_vwap_age(s.long_positions, s.current_step)
    short_lots, _, _ = _sum_sizes_vwap_age(s.short_positions, s.current_step)

    is_flat = (long_lots == D0) and (short_lots == D0)
    have_long = Decimal(1) if long_lots > D0 else Decimal(0)
    have_short = Decimal(1) if short_lots > D0 else Decimal(0)

    net_lots = (long_lots - short_lots)
    if is_flat:
        net_lots = D0
        have_long = Decimal(0)
        have_short = Decimal(0)

    snap = _active_side_snapshot(s.long_positions, s.short_positions, s.current_step)
    entry_price = D0 if snap["side"] is None else snap["vwap"]
    holding_minutes = D0 if snap["side"] is None else snap["age_min"]

    upnl = compute_unrealized_pnl_spec(s.long_positions, s.short_positions, s.current_price, s.lot_size)

    equity = s.cash_balance + s.used_margin + upnl
    max_equity = s.prev_max_equity if s.prev_max_equity is not None else equity
    if equity > max_equity:
        max_equity = equity
    drawdown = max_equity - equity
    if drawdown < D0:
        drawdown = D0

    out = {
        "pos_t": net_lots,
        "have_long_t": have_long,
        "have_short_t": have_short,
        "entry_price_t": entry_price,
        "holding_minutes_t": holding_minutes,
        "upnl_t": upnl,
        "realized_pnl_step_t": s.realized_pnl_step,
        "realized_pnl_cum_t": s.realized_pnl_cum,
        "fee_step_t": s.fee_step,
        "fee_cum_t": s.fee_cum,
        "equity_t": equity,
        "max_equity_t": max_equity,
        "drawdown_t": drawdown,
        # optional passthroughs：Spec 固定为 0（由 env 未来扩展）
        "sigma_entry_t": D0,
        "sl_ticks_t": D0,
        "tp_ticks_t": D0,
        "sl_price_t": D0,
        "tp_price_t": D0,
        "minutes_to_timeout_t": D0,
    }
    return out


def compute_agent_features_obs_spec(s: AgentSnapshot, raw: Dict[str, Decimal]) -> Dict[str, Decimal]:
    obs_market_open = Decimal(1) if int(s.market_open) == 1 else Decimal(0)

    obs_can_long_open  = Decimal(1) if int(s.can_long_open) == 1 else Decimal(0)
    obs_can_short_open = Decimal(1) if int(s.can_short_open) == 1 else Decimal(0)
    obs_can_long_close = Decimal(1) if int(s.can_long_close) == 1 else Decimal(0)
    obs_can_short_close= Decimal(1) if int(s.can_short_close) == 1 else Decimal(0)


    max_code = int(s.action_result_max_code)
    code = int(s.action_result_code)
    if code < 0:
        code = 0
    if max_code <= 0:
        obs_action_result = D0
    else:
        if code > max_code:
            code = max_code
        obs_action_result = Decimal(code) / Decimal(max_code)

    have_long = raw.get("have_long_t", D0)
    have_short = raw.get("have_short_t", D0)

    if have_long > D0 and have_short == D0:
        pos_side = Decimal(1)
    elif have_short > D0 and have_long == D0:
        pos_side = Decimal(-1)
    else:
        pos_side = Decimal(0)

    max_e = int(s.max_entries_per_day) if int(s.max_entries_per_day) > 0 else 1
    used = int(s.entries_used_today) if int(s.entries_used_today) >= 0 else 0
    left = max(0, max_e - used)
    entries_left_frac = Decimal(left) / Decimal(max_e)

    day_len = int(s.day_len) if int(s.day_len) > 0 else 1
    m2e = max(0, int(s.minutes_to_eod))
    minutes_to_eod_frac = Decimal(min(m2e, day_len)) / Decimal(day_len)

    holding_minutes = raw.get("holding_minutes_t", D0)
    holding_frac = _safe_div(holding_minutes, Decimal(day_len))
    holding_frac = _clip_dec(holding_frac, D0, Decimal(1))

    R_cash = s.R_cash if s.R_cash is not None else D0
    if R_cash <= D0:
        upnl_R = D0
        realized_today_R = D0
    else:
        upnl_R = _clip_dec(_safe_div(raw.get("upnl_t", D0), R_cash), Decimal("-5"), Decimal("5"))
        realized_today_R = _clip_dec(_safe_div(s.realized_today_cash, R_cash), Decimal("-10"), Decimal("10"))

    B0 = s.initial_balance if s.initial_balance is not None else D0
    if B0 <= D0:
        equity_frac = D0
        drawdown_frac = D0
    else:
        equity = raw.get("equity_t", D0)
        drawdown = raw.get("drawdown_t", D0)
        equity_frac = _clip_dec((equity - B0) / B0, Decimal("-1"), Decimal("1"))
        drawdown_frac = _clip_dec(drawdown / B0, Decimal("0"), Decimal("1"))

    return {
        "obs_market_open_t": obs_market_open,
        "obs_can_long_open_t": obs_can_long_open,
        "obs_can_short_open_t": obs_can_short_open,
        "obs_can_long_close_t": obs_can_long_close,
        "obs_can_short_close_t": obs_can_short_close,
        "obs_pos_side_t": pos_side,
        "obs_entries_left_frac_t": entries_left_frac,
        "obs_minutes_to_eod_frac_t": minutes_to_eod_frac,
        "obs_holding_frac_t": holding_frac,
        "obs_upnl_R_t": upnl_R,
        "obs_realized_today_R_t": realized_today_R,
        "obs_equity_frac_t": equity_frac,
        "obs_drawdown_frac_t": drawdown_frac,
        "obs_action_result_t": obs_action_result,
    }


def to_float32_vector(feat: Dict[str, Decimal], keys: List[str]) -> np.ndarray:
    # 与你们 decimal_to_float(precision=5, ROUND_HALF_UP) 对齐
    q = Decimal("1." + "0" * 5)

    def _f(x: Decimal) -> float:
        if not isinstance(x, Decimal):
            x = D(x)
        return float(x.quantize(q, rounding=ROUND_HALF_UP))

    return np.array([_f(feat.get(k, D0)) for k in keys], dtype=np.float32)
