# src/gym_trading_env/utils/agent_features.py
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional, Sequence, List

import numpy as np

from gym_trading_env.envs.position import Position
from gym_trading_env.utils.decimal_util import D0, D, decimal_to_float
from gym_trading_env.utils.trade_util import calc_unrealized_pnl


# -----------------------------
# RAW features (debug/log)
# -----------------------------
FEATURES_AGENT: List[str] = [
    "pos_t",                  # Net lots (+long, -short). In intraday single-position mode, magnitude is fixed trade_lot.
    "have_long_t",            # 1 if holding any long position else 0
    "have_short_t",           # 1 if holding any short position else 0
    "entry_price_t",          # VWAP entry price of active side (0 if flat)
    "holding_minutes_t",      # Holding age in minutes (0 if flat)
    "upnl_t",                 # Unrealized PnL in cash currency
    "realized_pnl_step_t",    # Realized PnL delta this step (cash)
    "realized_pnl_cum_t",     # Cumulative realized PnL (cash)
    "fee_step_t",             # Fee delta this step (cash)
    "fee_cum_t",              # Cumulative fee (cash)
    "equity_t",               # cash + used_margin + upnl (cash)
    "max_equity_t",           # max equity seen so far (cash)
    "drawdown_t",             # max_equity - equity (cash)
    "sigma_entry_t",          # (optional) volatility at entry (cash or unitless), default 0
    "sl_ticks_t",             # (optional) SL dist in ticks, default 0
    "tp_ticks_t",             # (optional) TP dist in ticks, default 0
    "sl_price_t",             # (optional) SL price, default 0
    "tp_price_t",             # (optional) TP price, default 0
    "minutes_to_timeout_t",   # (optional) timeout mins, default 0
]


# -----------------------------
# OBS features (train-time, 8D)
# -----------------------------
FEATURES_AGENT_OBS: List[str] = [
    "obs_pos_side_t",            # {-1,0,+1} position side: short=-1, flat=0, long=+1 (no long+short simultaneously)
    "obs_entries_left_frac_t",   # [0,1] remaining entries fraction for today
    "obs_minutes_to_eod_frac_t", # [0,1] minutes to end-of-day liquidation fraction
    "obs_holding_frac_t",        # [0,1] holding age fraction of DAY_LEN (flat=0)
    "obs_upnl_R_t",              # clipped cash PnL in R units (R_cash defined by stop-loss distance)
    "obs_realized_today_R_t",    # clipped realized-today cash PnL in R units
    "obs_equity_frac_t",         # clipped (equity - B0)/B0
    "obs_drawdown_frac_t",       # clipped drawdown/B0
]


@dataclass(frozen=True)
class AgentFeatureInput:
    # positions
    long_positions: Sequence[Optional[Position]]
    short_positions: Sequence[Optional[Position]]

    # time/price
    current_step: int
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

    # ---- intraday / normalization extras (required for 8D OBS) ----
    entries_used_today: int = 0
    max_entries_per_day: int = 1
    minutes_to_eod: int = 0
    day_len: int = 1

    initial_balance: Decimal = D0             # B0
    realized_today_cash: Decimal = D0         # realized_pnl_cum - day_start_realized_cum
    R_cash: Decimal = D0                      # 1R cash scale (derived from SL distance, passed from env)

    # optional (kept for backward compat / future use)
    sigma_entry: Decimal = D0
    sl_ticks: Decimal = D0
    tp_ticks: Decimal = D0
    sl_price: Decimal = D0
    tp_price: Decimal = D0
    minutes_to_timeout: Decimal = D0


def _clip_dec(x: Decimal, lo: Decimal, hi: Decimal) -> Decimal:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _safe_div(a: Decimal, b: Decimal) -> Decimal:
    return a / b if b != D0 else D0


def _sum_sizes_vwap_age(positions: Sequence[Optional[Position]], current_step: int):
    """
    Weighted sums across multiple slots.
    Returns: (total_size, vwap_entry, vwap_age_minutes)
    """
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
    """
    Choose a single 'active' side snapshot for RAW display.
    In intraday single-position mode you should never have both sides simultaneously,
    but we keep deterministic behavior anyway.
    """
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
        (calc_unrealized_pnl(current_price, pos=p, lot_size=lot_size, long=True)
         for p in long_positions if p is not None),
        Decimal("0")
    )
    upnl_short = sum(
        (calc_unrealized_pnl(current_price, pos=p, lot_size=lot_size, long=False)
         for p in short_positions if p is not None),
        Decimal("0")
    )
    return upnl_long + upnl_short


def compute_agent_features_raw(inp: AgentFeatureInput) -> Dict[str, Decimal]:
    """
    RAW features for debug/log. Values are in cash units where applicable.
    """
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

        # optional passthroughs (keep stable keys)
        "sigma_entry_t": inp.sigma_entry,
        "sl_ticks_t": inp.sl_ticks,
        "tp_ticks_t": inp.tp_ticks,
        "sl_price_t": inp.sl_price,
        "tp_price_t": inp.tp_price,
        "minutes_to_timeout_t": inp.minutes_to_timeout,
    }

    # strict flat constraint (optional)
    if is_flat:
        assert out["pos_t"] == D0 and out["have_long_t"] == 0 and out["have_short_t"] == 0

    return out


def compute_agent_features_obs(inp: AgentFeatureInput, raw: Dict[str, Decimal]) -> Dict[str, Decimal]:
    """
    8D OBS features for training, already normalized/clipped.

    Normalization rules:
    - pos_side: {-1,0,+1}
    - entries_left_frac: [0,1]
    - minutes_to_eod_frac: [0,1]
    - holding_frac: [0,1]
    - upnl_R: clip(upnl_cash/R_cash, -5, 5)
    - realized_today_R: clip(realized_today_cash/R_cash, -10, 10)
    - equity_frac: clip((equity-B0)/B0, -1, 1)
    - drawdown_frac: clip(drawdown/B0, 0, 1)
    """
    have_long = raw.get("have_long_t", D0)
    have_short = raw.get("have_short_t", D0)

    # pos_side
    if have_long > D0 and have_short == D0:
        pos_side = Decimal(1)
    elif have_short > D0 and have_long == D0:
        pos_side = Decimal(-1)
    else:
        pos_side = Decimal(0)

    # entries_left_frac
    max_e = int(inp.max_entries_per_day) if int(inp.max_entries_per_day) > 0 else 1
    used = int(inp.entries_used_today) if int(inp.entries_used_today) >= 0 else 0
    left = max(0, max_e - used)
    entries_left_frac = Decimal(left) / Decimal(max_e)

    # minutes_to_eod_frac
    day_len = int(inp.day_len) if int(inp.day_len) > 0 else 1
    m2e = max(0, int(inp.minutes_to_eod))
    minutes_to_eod_frac = Decimal(min(m2e, day_len)) / Decimal(day_len)

    # holding_frac
    holding_minutes = raw.get("holding_minutes_t", D0)
    holding_frac = _safe_div(holding_minutes, Decimal(day_len))
    if holding_frac < D0:
        holding_frac = D0
    if holding_frac > Decimal(1):
        holding_frac = Decimal(1)

    # R normalization
    R_cash = inp.R_cash if inp.R_cash is not None else D0
    if R_cash <= D0:
        R_cash = Decimal("1")  # last-ditch fallback

    upnl_R = _clip_dec(_safe_div(raw.get("upnl_t", D0), R_cash), Decimal("-5"), Decimal("5"))
    realized_today_R = _clip_dec(_safe_div(inp.realized_today_cash, R_cash), Decimal("-10"), Decimal("10"))

    # equity / drawdown normalized by initial balance
    B0 = inp.initial_balance if inp.initial_balance is not None else D0
    if B0 <= D0:
        equity_frac = D0
        drawdown_frac = D0
    else:
        equity = raw.get("equity_t", D0)
        drawdown = raw.get("drawdown_t", D0)
        equity_frac = _clip_dec((equity - B0) / B0, Decimal("-1"), Decimal("1"))
        drawdown_frac = _clip_dec(drawdown / B0, Decimal("0"), Decimal("1"))

    return {
        "obs_pos_side_t": pos_side,
        "obs_entries_left_frac_t": entries_left_frac,
        "obs_minutes_to_eod_frac_t": minutes_to_eod_frac,
        "obs_holding_frac_t": holding_frac,
        "obs_upnl_R_t": upnl_R,
        "obs_realized_today_R_t": realized_today_R,
        "obs_equity_frac_t": equity_frac,
        "obs_drawdown_frac_t": drawdown_frac,
    }


def agent_feature_vector(feat: Dict[str, Decimal], feature_list: Optional[List[str]] = None) -> np.ndarray:
    """
    Convert {name: Decimal} dict into float32 vector in the specified order.
    Missing keys are treated as 0.
    """
    if feature_list is None:
        feature_list = FEATURES_AGENT
    vec = np.array([decimal_to_float(feat.get(k, D0)) for k in feature_list], dtype=np.float32)
    return vec
