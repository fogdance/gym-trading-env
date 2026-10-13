"""FG V0/V2/V3 offline, event-sourced reward scorer (stdlib only).

Consumes *committed* entry fills, full-position settlement attributions and
observed bars from the EXISTING FG Env. It never invents fills or modifies the
Env account / business reward. Parameters are research candidates, not defaults
for production training.

Contract: one entry_id represents exactly three filled lots; at most two active
entry_ids; steps are in causal market order. See tools/fg_reward_offline.py.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from hashlib import sha256
from math import isfinite
from typing import Any, Mapping
import copy
import json


class RewardInputError(ValueError):
    """Missing, contradictory, noncausal or duplicate event in the offline trace."""


def _clock(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise RewardInputError(f"Invalid timestamp: {value!r}") from error


def _number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as error:
        raise RewardInputError(f"Invalid {label}: {value!r}") from error
    if not isfinite(number):
        raise RewardInputError(f"Nonfinite {label}")
    return number


def _true_flag(value: Any) -> bool:
    # An absent committed field is NOT proof of a real fill.
    return value is True or value == 1 or value == "True" or value == "true"


def _side(value: Any) -> int:
    if value == "LONG":
        return 1
    if value == "SHORT":
        return -1
    raise RewardInputError(f"Invalid side: {value!r}")


def legacy_tp(minutes: float) -> float:
    minutes = _number(minutes, "TP natural minutes")
    if minutes < 0:
        raise RewardInputError("TP precedes entry")
    return 60 / (60 + minutes)


def timeboxed_tp(minutes: float) -> float:
    """V2/V3 candidate; TP must be an actual committed complete settlement."""
    minutes = _number(minutes, "TP natural minutes")
    if minutes < 0:
        raise RewardInputError("TP precedes entry")
    if minutes <= 60:
        return legacy_tp(minutes)
    if minutes < 480:
        return legacy_tp(minutes) * (480 - minutes) / 420
    return 0.0


def capacity_units(active_slots: int) -> float:
    if not isinstance(active_slots, int) or active_slots not in (0, 1, 2):
        raise RewardInputError("FG requires 0/1/2 occupied entry slots")
    return (active_slots / 2) ** 2


@dataclass(frozen=True)
class CandidateConfig:
    beta: float = 0.0
    rho: float = 0.3
    lam: float = 0.0
    alpha: float = 0.0
    gamma: float = 332 / 333
    risk_points: float = 10.0
    take_profit_points: float = 5.0
    clock_protocol: str = 'LEGACY_WALL_MINUTE'

    def __post_init__(self):
        if self.clock_protocol not in ('LEGACY_WALL_MINUTE', 'S3_MARKET_MINUTE'):
            raise RewardInputError('Unknown reward time protocol')
        for key in ("beta", "rho", "lam", "alpha", "gamma", "risk_points", "take_profit_points"):
            _number(getattr(self, key), key)
        if self.beta < 0 or self.lam < 0 or self.alpha < 0 or not (0 < self.rho < 1):
            raise RewardInputError("Expected beta/lam/alpha>=0 and 0<rho<1")
        if not (0 < self.gamma < 1) or self.risk_points <= 0 or self.take_profit_points != 5:
            raise RewardInputError("Expected 0<gamma<1, positive risk_points, TP=5")


@dataclass
class _Position:
    entry_id: str
    side: str
    entry_time: str
    entry_price: float
    bars_seen: int = 0
    adverse10_fired: bool = False
    adverse_first_bar: int | None = None
    early_penalty: float = 0.0
    capacity_basis: float = 0.0
    exit_time: str | None = None
    exit_reason: str | None = None
    tp_minutes: float | None = None
    tp_market_minutes: int | None = None
    original_tp: float = 0.0
    timeboxed_tp: float = 0.0
    intrabar_ambiguous: bool = False


def _potential(active: Mapping[str, _Position], mark: float | None) -> float:
    if mark is None or not active:
        return 0.0
    return 0.5 * sum(max(-2.0, min(1.0, _side(p.side) * (mark - p.entry_price) / 5.0))
                     for p in active.values())


class FGRewardScorer:
    """Deterministic stateful offline score; never submits orders.

    Input per step:
      bar_open, bar_end, open, high, low, close (OHLC of OBSERVED bar)
      entries: committed ENTRY fills with entry_id,time,side,quantity=3,price
      exits: completed attribution events with the FG Env's entry_id,
             entry_time,exit_time,entry_price,exit_price,side,
             closed_quantity=3,reason,committed
      valid_bar: recorded duration one minute (not zero-duration metadata)
      eligible_market_minute: exogenous known opportunity clock, independent
             of position availability / future price or TP. Applied only to V3.
      chunk_end: nonterminal pause (DOES NOT reset scorer)
      true_terminal: terminal after all positions closed, not a data cutoff.

    All timestamps must use the same timezone convention. Same-bar OHLC does
    not identify touch ordering; if both adverse10 and TP occur, risk wins for
    this diagnostic and intrabar_ambiguous is reported.
    """

    SCHEMA = 1

    def __init__(self, config: CandidateConfig):
        self.config = config
        self.SCHEMA = 2 if config.clock_protocol == 'S3_MARKET_MINUTE' else 1
        self.active: dict[str, _Position] = {}
        self.closed: dict[str, _Position] = {}
        self.seen: set[str] = set()
        self.last_mark: float | None = None
        self.last_end: datetime | None = None
        self.step_number = 0
        self.totals = {k: 0.0 for k in ("v0", "tp_timeboxed", "early", "potential", "capacity", "v2", "v3", "capacity_basis")}
        self.discounted = {k: 0.0 for k in ("v0", "v2", "v3")}
        self.discount_weight = 1.0
        self.ambiguities = 0

    def step(self, bar: Mapping[str, Any]) -> dict[str, Any]:
        # Fail before committing *any* mutations, including partial exits.
        # Atomic validation without deep-copying all historically closed trades
        # on every minute (would become quadratic in a long FG replay).
        clone = copy.copy(self)
        clone.active = copy.deepcopy(self.active)
        clone.closed = self.closed.copy()   # finished positions are immutable here
        clone.seen = self.seen.copy()
        clone.totals = self.totals.copy()
        clone.discounted = self.discounted.copy()
        result = clone._apply(bar)
        self.__dict__.update(clone.__dict__)
        return result

    def _apply(self, bar: Mapping[str, Any]) -> dict[str, Any]:
        try:
            start, stop = _clock(bar["bar_open"]), _clock(bar["bar_end"])
            op, hi, lo, close = (_number(bar[k], k) for k in ("open", "high", "low", "close"))
        except KeyError as error:
            raise RewardInputError(f"Missing bar field: {error}") from error
        if start > stop or (self.last_end is not None and
                            (start < self.last_end or stop <= self.last_end)):
            raise RewardInputError("Noncausal or duplicate bar timestamps")
        if min(op, hi, lo, close) <= 0 or hi < max(op, lo, close) or lo > min(op, hi, close):
            raise RewardInputError("Invalid OHLC")
        if "valid_bar" not in bar or "eligible_market_minute" not in bar:
            raise RewardInputError("Bar must declare its validity and exogenous opportunity clock")
        valid_bar = bar["valid_bar"]
        eligible = bar["eligible_market_minute"]
        if not isinstance(valid_bar, bool) or not isinstance(eligible, bool):
            raise RewardInputError("valid_bar and eligible_market_minute must be booleans")
        if valid_bar and (stop-start).total_seconds() != 60:
            raise RewardInputError("A valid observed FG bar must last exactly 60 seconds")
        if not valid_bar and stop != start:
            raise RewardInputError("Invalid metadata row should have zero duration")
        entries, exits = bar.get("entries", []), bar.get("exits", [])
        if not isinstance(entries, list) or not isinstance(exits, list):
            raise RewardInputError("entries/exits must be lists")
        if not valid_bar and (entries or exits):
            raise RewardInputError("Zero-duration metadata row cannot execute fills")
        phi_before = _potential(self.active, self.last_mark)
        newly_opened: list[str] = []
        for fill in entries:
            if fill.get("reason") != "ENTRY" or not _true_flag(fill.get("committed")):
                raise RewardInputError("Entry must be a committed actual ENTRY fill")
            entry_id = str(fill.get("entry_id", ""))
            if not entry_id or entry_id in self.seen or len(self.active) >= 2 or not valid_bar:
                raise RewardInputError("Duplicate/invalid entry, capacity exceeded or non-executable bar")
            if _number(fill.get("quantity"), "ENTRY quantity") != 3:
                raise RewardInputError("Only complete three-lot batches are supported")
            entry_time = _clock(fill.get("time"))
            if entry_time != start:
                raise RewardInputError("ENTRY must have executed at actual bar opening clock")
            price = _number(fill.get("price"), "ENTRY price")
            if price <= 0:
                raise RewardInputError("Invalid ENTRY price")
            p = _Position(entry_id, str(fill["side"]), str(entry_time), price)
            _side(p.side)
            self.active[entry_id] = p
            self.seen.add(entry_id)
            newly_opened.append(entry_id)
        n_slots = len(self.active)
        # Both existing positions and newly committed entries occupied the bar
        # interval. Full settlement occurs at eob. No cost during closed hours.
        unit = capacity_units(n_slots) if valid_bar and eligible else 0.0
        if n_slots:
            for p in self.active.values():
                p.capacity_basis += unit / n_slots
        early = 0.0
        penalized: list[str] = []
        early_first5 = early_next10 = 0
        risk_windows: dict[str, int] = {}
        for p in self.active.values():
            if valid_bar:
                p.bars_seen += 1
                adverse = p.entry_price-lo if p.side == "LONG" else hi-p.entry_price
                if adverse >= self.config.risk_points-1e-10 and not p.adverse10_fired:
                    p.adverse10_fired = True
                    p.adverse_first_bar = p.bars_seen
                    weight = (1.0 if p.bars_seen <= 5 else
                              self.config.rho if p.bars_seen <= 15 else 0.0)
                    p.early_penalty = -self.config.beta * weight
                    early += p.early_penalty
                    penalized.append(p.entry_id)
                    risk_windows[p.entry_id] = p.bars_seen
                    early_first5 += int(p.bars_seen <= 5)
                    early_next10 += int(6 <= p.bars_seen <= 15)
        v0 = tp = 0.0
        settled: list[str] = []
        ambiguous: list[str] = []
        for event in exits:
            entry_id = str(event.get("entry_id", ""))
            if entry_id not in self.active or entry_id in settled or not _true_flag(event.get("committed")):
                raise RewardInputError("Unmatched, duplicate or uncommitted exit")
            p = self.active[entry_id]
            if _number(event.get("closed_quantity"), "closed quantity") != 3:
                raise RewardInputError("Expected completed three-lot attribution; not a partial fill")
            if (str(event.get("side")) != p.side or
                    _number(event.get("entry_price"), "recorded entry_price") != p.entry_price or
                    _clock(event.get("entry_time")) != _clock(p.entry_time) or
                    _clock(event.get("exit_time")) != stop):
                raise RewardInputError("Exit identity, clocks or entry-price mismatch")
            reason = str(event.get("reason"))
            exit_price = _number(event.get("exit_price"), "exit price")
            if reason == "TAKE_PROFIT":
                if _side(p.side)*(exit_price-p.entry_price) < 5-1e-10:
                    raise RewardInputError("TP exit did not fill at +5")
                if ((p.side == "LONG" and hi < exit_price - 1e-10) or
                        (p.side == "SHORT" and lo > exit_price + 1e-10)):
                    raise RewardInputError("TP claim contradicts observed OHLC")
                # Existing Env prohibits same-bar TP from using an intrabar
                # high/low that could precede the opening fill.
                if _clock(p.entry_time) == start and (
                        (p.side == "LONG" and close < exit_price - 1e-10) or
                        (p.side == "SHORT" and close > exit_price + 1e-10)):
                    raise RewardInputError("Same-bar TP must be supported by close")
                if self.config.clock_protocol == 'S3_MARKET_MINUTE':
                    minutes = p.bars_seen
                    p.tp_market_minutes = minutes
                    if event.get('held_market_bars', minutes) != minutes:
                        raise RewardInputError('Env/reward holding market clocks differ')
                else:
                    minutes = (stop-_clock(p.entry_time)).total_seconds()/60
                    p.tp_minutes = minutes
                p.original_tp = legacy_tp(minutes)
                p.timeboxed_tp = timeboxed_tp(minutes)
                v0 += p.original_tp
                tp += p.timeboxed_tp
                # Only a recorded completed TP can make this bar ambiguous.
                adverse = p.entry_price-lo if p.side == "LONG" else hi-p.entry_price
                if valid_bar and adverse >= self.config.risk_points-1e-10:
                    p.intrabar_ambiguous = True
                    ambiguous.append(entry_id)
            p.exit_time, p.exit_reason = str(stop), reason
            self.closed[entry_id] = self.active.pop(entry_id)
            settled.append(entry_id)
        if bar.get("true_terminal", False) and self.active:
            raise RewardInputError("True terminal cannot silently discard open entries")
        # A nonterminal 1024-step chunk does not reset the mark, active slots,
        # first-adverse flags or potential. Right-censoring is NOT a terminal.
        phi_after = _potential(self.active, close)
        advance = int(valid_bar) if self.config.clock_protocol == 'S3_MARKET_MINUTE' else 1
        discount = self.config.gamma ** advance
        potential_basis = discount*phi_after-phi_before
        potential = self.config.alpha*potential_basis
        capacity = -self.config.lam*unit
        v2 = tp+early+potential
        v3 = v2+capacity
        values = dict(v0=v0,tp_timeboxed=tp,early=early,potential=potential,capacity=capacity,
                      capacity_basis=unit,v2=v2,v3=v3)
        for key, value in values.items():
            self.totals[key] += value
        for key in self.discounted:
            self.discounted[key] += self.discount_weight*values[key]
        self.discount_weight *= discount
        self.last_end, self.last_mark = stop, close
        self.step_number += 1
        self.ambiguities += len(ambiguous)
        return dict(bar_end=str(stop),step=self.step_number,active_slots_for_interval=n_slots,
                    eligible_market_minute=eligible,valid_bar=valid_bar,opened=newly_opened,
                    penalized=penalized,settled=settled,intrabar_ambiguous=ambiguous,
                    early_first5=early_first5,early_next10=early_next10,
                    ambiguous_early_first5=sum(risk_windows.get(k, 16) <= 5 for k in ambiguous),
                    ambiguous_early_next10=sum(6 <= risk_windows.get(k, 16) <= 15 for k in ambiguous),
                    potential_basis=potential_basis,
                    phi_before=phi_before,phi_after=phi_after,chunk_end=bool(bar.get("chunk_end",False)),
                    **values)

    def snapshot(self) -> dict[str, Any]:
        """JSON-serializable state for nonterminal chunk / process recovery."""
        payload = dict(schema=self.SCHEMA, config=asdict(self.config), active={k:asdict(v) for k,v in self.active.items()},
                       closed={k:asdict(v) for k,v in self.closed.items()}, seen=sorted(self.seen),
                       last_mark=self.last_mark,last_end=str(self.last_end) if self.last_end else None,
                       step_number=self.step_number, totals=self.totals, discounted=self.discounted,
                       discount_weight=self.discount_weight, ambiguities=self.ambiguities)
        payload["sha256"] = sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        return payload

    @classmethod
    def restore(cls, state: Mapping[str, Any], config: CandidateConfig) -> "FGRewardScorer":
        payload = copy.deepcopy(dict(state))
        check = payload.pop("sha256", None)
        if check != sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest():
            raise RewardInputError("Offline scorer checkpoint hash mismatch")
        if payload.get("schema") != cls(config).SCHEMA or payload.get("config") != asdict(config):
            raise RewardInputError("Reward protocol/configuration mismatch")
        obj = cls(config)
        obj.active = {k:_Position(**v) for k,v in payload["active"].items()}
        obj.closed = {k:_Position(**v) for k,v in payload["closed"].items()}
        obj.seen = set(payload["seen"])
        if obj.seen != set(obj.active)|set(obj.closed) or len(obj.active) > 2:
            raise RewardInputError("Corrupted entry identity or capacity")
        obj.last_mark = payload["last_mark"]
        obj.last_end = _clock(payload["last_end"]) if payload["last_end"] else None
        obj.step_number = int(payload["step_number"])
        obj.totals = dict(payload["totals"])
        obj.discounted = dict(payload["discounted"])
        obj.discount_weight = float(payload["discount_weight"])
        obj.ambiguities = int(payload["ambiguities"])
        return obj

    def trade_rows(self, *, censored: bool = True) -> list[dict[str, Any]]:
        rows=[]
        for p in [*self.closed.values(), *self.active.values()]:
            row=asdict(p)
            row["status"] = ("right_censored" if censored else "open") if p.entry_id in self.active else (
                "success" if p.exit_reason == "TAKE_PROFIT" else "terminal_failure")
            row["capacity_cost"] = -self.config.lam*p.capacity_basis
            rows.append(row)
        return rows

    def summary(self) -> dict[str, Any]:
        return dict(config=asdict(self.config), steps=self.step_number, totals=dict(self.totals),
                    discounted=dict(self.discounted), active_entries=len(self.active),
                    closed_entries=len(self.closed), tp_entries=sum(p.exit_reason=="TAKE_PROFIT" for p in self.closed.values()),
                    intrabar_ambiguous=self.ambiguities)
