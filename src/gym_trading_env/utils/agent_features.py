# src/gym_trading_env/utils/agent_features.py
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional, Sequence
import numpy as np
from typing import List, Optional, Dict

from gym_trading_env.envs.position import Position
from gym_trading_env.utils.decimal_util import D0, D, decimal_to_float
from gym_trading_env.utils.trade_util import calc_unrealized_pnl

# Agent-side features (single vector, emitted by env at runtime)
FEATURES_AGENT_OBS: List[str] = [
    "obs_pos_t",                  # Current position size (positive = long, negative = short)
    "obs_have_long_t",            # 1 if holding long position, 0 otherwise
    "obs_have_short_t",           # 1 if holding short position, 0 otherwise
    "obs_entry_price_t",          # Entry price of the current position
    "obs_holding_minutes_t",      # Number of minutes the position has been held
    "obs_upnl_t",                 # Unrealized PnL at time t
    "obs_realized_pnl_step_t",    # Realized PnL in the current step
    "obs_realized_pnl_cum_t",     # Cumulative realized PnL
    "obs_fee_step_t",             # Trading fee in the current step
    "obs_fee_cum_t",              # Cumulative trading fees
    "obs_equity_t",               # Current equity (cash + unrealized PnL)
    "obs_max_equity_t",           # Historical maximum equity (for drawdown calc)
    "obs_drawdown_t",             # Current drawdown from peak equity
    "obs_sigma_entry_t",          # Volatility estimate at entry time
    "obs_sl_ticks_t",             # Stop-loss distance in ticks
    "obs_tp_ticks_t",             # Take-profit distance in ticks
    "obs_sl_price_t",             # Stop-loss price level
    "obs_tp_price_t",             # Take-profit price level
    "obs_minutes_to_timeout_t",   # Minutes remaining until position timeout
]


# Agent-side features (single vector, emitted by env at runtime)
FEATURES_AGENT: List[str] = [
    "pos_t",                  # Current position size (positive = long, negative = short)
    "have_long_t",            # 1 if holding long position, 0 otherwise
    "have_short_t",           # 1 if holding short position, 0 otherwise
    "entry_price_t",          # Entry price of the current position
    "holding_minutes_t",      # Number of minutes the position has been held
    "upnl_t",                 # Unrealized PnL at time t
    "realized_pnl_step_t",    # Realized PnL in the current step
    "realized_pnl_cum_t",     # Cumulative realized PnL
    "fee_step_t",             # Trading fee in the current step
    "fee_cum_t",              # Cumulative trading fees
    "equity_t",               # Current equity (cash + unrealized PnL)
    "max_equity_t",           # Historical maximum equity (for drawdown calc)
    "drawdown_t",             # Current drawdown from peak equity
    "sigma_entry_t",          # Volatility estimate at entry time
    "sl_ticks_t",             # Stop-loss distance in ticks
    "tp_ticks_t",             # Take-profit distance in ticks
    "sl_price_t",             # Stop-loss price level
    "tp_price_t",             # Take-profit price level
    "minutes_to_timeout_t",   # Minutes remaining until position timeout
]

@dataclass(frozen=True)
class AgentFeatureInput:
    # positions
    long_positions: Sequence[Optional[Position]]
    short_positions: Sequence[Optional[Position]]

    # time/price
    current_step: int                 # step index (minute granularity)
    current_price: Decimal
    lot_size: Decimal

    # accounting (cumulative & step deltas)
    realized_pnl_step: Decimal
    realized_pnl_cum: Decimal
    fee_step: Decimal
    fee_cum: Decimal

    # balances used for equity
    cash_balance: Decimal
    used_margin: Decimal

    # drawdown tracking
    prev_max_equity: Decimal

    # optional risk features (未实现也能先接 0)
    sigma_entry: Decimal = D0
    sl_ticks: Decimal = D0
    tp_ticks: Decimal = D0
    sl_price: Decimal = D0
    tp_price: Decimal = D0
    minutes_to_timeout: Decimal = D0


def _sum_sizes_vwap_age(positions: Sequence[Optional[Position]], current_step: int):
    total = D0
    vwap_num = D0
    age_num = D0
    for p in positions:
        if p is None:
            continue
        total += p.size
        vwap_num += p.size * p.entry_price
        age_num += p.size * D(current_step - int(p.open_step))
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

    # both present: choose larger exposure; tie -> most recent open_step
    if L > S:
        return {"side": "long", "size": L, "vwap": L_vwap, "age_min": L_age}
    if S > L:
        return {"side": "short", "size": S, "vwap": S_vwap, "age_min": S_age}

    latest_L = max((p.open_step for p in long_positions if p is not None), default=-1)
    latest_S = max((p.open_step for p in short_positions if p is not None), default=-1)
    return {"side": "long", "size": L, "vwap": L_vwap, "age_min": L_age} if latest_L >= latest_S \
        else {"side": "short", "size": S, "vwap": S_vwap, "age_min": S_age}


def compute_unrealized_pnl(long_positions, short_positions, current_price: Decimal, lot_size: Decimal) -> Decimal:
    upnl_long = sum(
        (calc_unrealized_pnl(current_price, pos=p, lot_size=lot_size, long=True) for p in long_positions if p is not None),
        Decimal("0")
    )
    upnl_short = sum(
        (calc_unrealized_pnl(current_price, pos=p, lot_size=lot_size, long=False) for p in short_positions if p is not None),
        Decimal("0")
    )
    return upnl_long + upnl_short


def compute_agent_features(inp: AgentFeatureInput) -> Dict[str, Decimal]:
    long_lots, _, _ = _sum_sizes_vwap_age(inp.long_positions, inp.current_step)
    short_lots, _, _ = _sum_sizes_vwap_age(inp.short_positions, inp.current_step)

    is_flat = (long_lots == D0) and (short_lots == D0)
    have_long = Decimal(1) if long_lots > D0 else Decimal(0)
    have_short = Decimal(1) if short_lots > D0 else Decimal(0)

    net_lots = (long_lots - short_lots)
    if is_flat:
        net_lots = D0
        have_long = Decimal(0)
        have_short = Decimal(0)

    snap = _active_side_snapshot(inp.long_positions, inp.short_positions, inp.current_step)
    entry_price = D0 if snap["side"] is None else snap["vwap"]
    holding_minutes = D0 if snap["side"] is None else snap["age_min"]

    upnl = compute_unrealized_pnl(inp.long_positions, inp.short_positions, inp.current_price, inp.lot_size)

    equity = inp.cash_balance + inp.used_margin + upnl
    max_equity = inp.prev_max_equity if inp.prev_max_equity is not None else equity
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
        "realized_pnl_step_t": inp.realized_pnl_step,
        "realized_pnl_cum_t": inp.realized_pnl_cum,
        "fee_step_t": inp.fee_step,
        "fee_cum_t": inp.fee_cum,
        "equity_t": equity,
        "max_equity_t": max_equity,
        "drawdown_t": drawdown,
        "sigma_entry_t": inp.sigma_entry,
        "sl_ticks_t": inp.sl_ticks,
        "tp_ticks_t": inp.tp_ticks,
        "sl_price_t": inp.sl_price,
        "tp_price_t": inp.tp_price,
        "minutes_to_timeout_t": inp.minutes_to_timeout,
    }

    # flat 强约束（可选）
    if is_flat:
        assert out["pos_t"] == D0 and out["have_long_t"] == 0 and out["have_short_t"] == 0

    return out


def agent_feature_vector(feat: Dict[str, Decimal]) -> np.ndarray:
    vec = np.array([decimal_to_float(feat[k]) for k in FEATURES_AGENT], dtype=np.float32)
    return vec
