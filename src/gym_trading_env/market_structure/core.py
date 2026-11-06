"""Port of packages/market-structure/src/{index,pivots}.ts, including plateau fixes.

No calendar, database, chart, numpy, or pandas dependencies. The caller supplies
chronological closed bars from one actual contract and one timeframe.
"""

from collections.abc import Iterable, Mapping
import math
from numbers import Integral, Real
from typing import Literal

from .types import (
    AmbiguousPolicy, EventType, OhlcBar, PlateauPolicy, StructureConfig,
    StructureEvent, StructureResult, StructureState, SwingAction, SwingLogEntry,
    SwingPoint, Trend,
)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _normalize_bars(bars: Iterable[Mapping[str, object]]) -> list[OhlcBar]:
    normalized: list[OhlcBar] = []
    for index, bar in enumerate(bars):
        if not isinstance(bar, Mapping):
            raise TypeError(f"Bar {index} must be a mapping with open/high/low/close")
        values = {}
        for key in ("open", "high", "low", "close"):
            if key not in bar:
                raise ValueError(f"Bar {index} is missing {key}")
            values[key] = _finite_number(bar[key], f"Bar {index} {key}")
        if (values["high"] < max(values["open"], values["low"], values["close"])
                or values["low"] > min(values["open"], values["high"], values["close"])):
            raise ValueError(f"Invalid OHLC relationship at bar {index}")
        normalized.append({"open": values["open"], "high": values["high"],
                           "low": values["low"], "close": values["close"]})
    return normalized


def _is_extreme(bars: list[OhlcBar], pivot: int, confirmation: int,
                config: StructureConfig, side: Literal["high", "low"]) -> bool:
    value = bars[pivot][side]

    def more_extreme(other: float) -> bool:
        return other > value if side == "high" else other < value

    # A repeated touch on the right postpones confirmation to the later touch.
    for neighbor in range(pivot + 1, confirmation + 1):
        other = bars[neighbor][side]
        if other == value or more_extreme(other):
            return False
    remaining = config["left"]
    for neighbor in range(pivot - 1, -1, -1):
        other = bars[neighbor][side]
        if more_extreme(other):
            return False
        if other == value:
            if config["plateau_policy"] != "last":
                return False
            continue
        remaining -= 1
        if remaining == 0:
            return True
    # Equal-only history must not manufacture a pivot at the left boundary.
    return False


def _confirmed_pivot(bars: list[OhlcBar], index: int,
                     config: StructureConfig) -> tuple[SwingPoint | None, bool]:
    pivot = index - config["right"]
    if pivot < config["left"]:
        return None, False
    high = _is_extreme(bars, pivot, index, config, "high")
    low = _is_extreme(bars, pivot, index, config, "low")
    if high and low:
        if config["ambiguous_policy"] == "raise":
            raise ValueError(f"Bar {pivot} is both a swing high and swing low")
        return None, True
    if not high and not low:
        return None, False
    return {
        "kind": "H" if high else "L",
        "price": bars[pivot]["high" if high else "low"],
        "pivot_index": pivot,
        "confirm_index": index,
    }, False


def _append_swing(swings: list[SwingPoint], point: SwingPoint) -> SwingAction:
    if not swings or swings[-1]["kind"] != point["kind"]:
        swings.append(point)
        return "added"
    last = swings[-1]
    more_extreme = (point["price"] >= last["price"] if point["kind"] == "H"
                    else point["price"] <= last["price"])
    if more_extreme:
        swings[-1] = point
        return "replaced"
    return "ignored"


def analyze_market_structure(
    bars: Iterable[Mapping[str, object]],
    *,
    left: int = 2,
    right: int = 2,
    plateau_policy: PlateauPolicy = "last",
    ambiguous_policy: AmbiguousPolicy = "skip",
    structure_tolerance_pct: float = 0.0,
    break_buffer_pct: float = 0.0,
) -> StructureResult:
    """Analyze a closed OHLC sequence without consuming future data early.

    Each bar is a mapping containing lowercase open/high/low/close numbers.
    Extra fields (e.g. eob) are ignored; output indices refer to input order.
    No sorting or timeframe aggregation is performed. Empty input is valid.

    Returns JSON-serializable dictionaries with config, swings, swing_log,
    history, events, current, and ambiguous_pivot_count (see README.md).
    The last normalized swing may move until an opposite pivot confirms.
    Treat the returned points/states as read-only.

    Raises TypeError/ValueError on invalid prices, OHLC, or parameters.
    """
    for name, value in (("left", left), ("right", right)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer >= 1")
        if not 1 <= value <= 2**53 - 1:
            raise ValueError(f"{name} must be an integer between 1 and 2**53 - 1")
    if plateau_policy not in ("last", "strict"):
        raise ValueError("plateau_policy must be 'last' or 'strict'")
    if ambiguous_policy not in ("skip", "raise"):
        raise ValueError("ambiguous_policy must be 'skip' or 'raise'")
    tolerance = _finite_number(structure_tolerance_pct, "structure_tolerance_pct")
    buffer = _finite_number(break_buffer_pct, "break_buffer_pct")
    if tolerance < 0 or buffer < 0:
        raise ValueError("Tolerance and break buffer must be non-negative")
    config: StructureConfig = {
        "left": int(left), "right": int(right), "plateau_policy": plateau_policy,
        "ambiguous_policy": ambiguous_policy,
        "structure_tolerance_pct": tolerance, "break_buffer_pct": buffer,
    }
    data = _normalize_bars(bars)

    def lt(a: float, b: float, margin: float = tolerance) -> bool:
        return a < b * (1 - margin)

    def gt(a: float, b: float, margin: float = tolerance) -> bool:
        return a > b * (1 + margin)

    swings: list[SwingPoint] = []
    highs: list[SwingPoint] = []
    lows: list[SwingPoint] = []
    swing_log: list[SwingLogEntry] = []
    events: list[StructureEvent] = []
    history: list[StructureState] = []
    trend: Trend = "neutral"
    trend_since_index = None
    down_setup = up_setup = None
    protected_high = external_low = candidate_lh = None
    protected_low = external_high = candidate_hl = None
    latest_high = latest_low = None
    ambiguous_pivot_count = 0

    for index, bar in enumerate(data):
        close = bar["close"]
        event: EventType | None = None

        def emit(kind: EventType, level: float, protected: SwingPoint | None) -> None:
            nonlocal event
            event = kind
            events.append({"index": index, "type": kind, "break_level": level,
                           "protected_point": protected})

        point, ambiguous = _confirmed_pivot(data, index, config)
        ambiguous_pivot_count += int(ambiguous)
        accepted = None
        # 1. Consume pivots that have become knowable at this close.
        if point is not None:
            action = _append_swing(swings, point)
            swing_log.append({"point": point, "action": action})
            if action != "ignored":
                accepted = point
                same_kind = highs if point["kind"] == "H" else lows
                if action == "replaced" and same_kind:
                    same_kind[-1] = point
                else:
                    same_kind.append(point)
                if point["kind"] == "H":
                    latest_high = point
                else:
                    latest_low = point
                if len(swings) >= 3:
                    first, middle, last = swings[-3:]
                    if first["kind"] == "H":
                        down_setup = (first, middle, last) if lt(last["price"], first["price"]) else None
                    else:
                        up_setup = (first, middle, last) if gt(last["price"], first["price"]) else None
                if (trend == "downtrend" and point["kind"] == "H" and protected_high and external_low
                        and point["pivot_index"] > external_low["pivot_index"]):
                    candidate_lh = point if lt(point["price"], protected_high["price"]) else None
                elif (trend == "uptrend" and point["kind"] == "L" and protected_low and external_high
                        and point["pivot_index"] > external_high["pivot_index"]):
                    candidate_hl = point if gt(point["price"], protected_low["price"]) else None

        # 2. Invalidate before considering continuation or a new trend.
        if trend == "downtrend" and protected_high and gt(close, protected_high["price"], buffer):
            emit("downtrend_invalidated", protected_high["price"], protected_high)
            trend = "neutral"
            trend_since_index = None
            protected_high = external_low = candidate_lh = None
        elif trend == "uptrend" and protected_low and lt(close, protected_low["price"], buffer):
            emit("uptrend_invalidated", protected_low["price"], protected_low)
            trend = "neutral"
            trend_since_index = None
            protected_low = external_high = candidate_hl = None

        # 3. Continuation protects the newly confirmed pullback extreme.
        if trend == "downtrend" and external_low and candidate_lh and lt(close, external_low["price"], buffer):
            protected_high = candidate_lh
            candidate_lh = None
            emit("downtrend_continuation", external_low["price"], protected_high)
        elif trend == "uptrend" and external_high and candidate_hl and gt(close, external_high["price"], buffer):
            protected_low = candidate_hl
            candidate_hl = None
            emit("uptrend_continuation", external_high["price"], protected_low)

        # 4. A setup can trigger only after its last pivot confirms.
        if trend == "neutral":
            down_trigger = down_setup is not None and lt(close, down_setup[1]["price"], buffer)
            up_trigger = up_setup is not None and gt(close, up_setup[1]["price"], buffer)
            if down_trigger and up_trigger and down_setup and up_setup:
                down_trigger = down_setup[2]["pivot_index"] >= up_setup[2]["pivot_index"]
                up_trigger = not down_trigger
            if down_trigger and down_setup:
                trend = "downtrend"
                trend_since_index = index
                protected_high, external_low = down_setup[2], down_setup[1]
                candidate_lh = protected_low = external_high = candidate_hl = None
                emit("downtrend_confirmed", external_low["price"], protected_high)
            elif up_trigger and up_setup:
                trend = "uptrend"
                trend_since_index = index
                protected_low, external_high = up_setup[2], up_setup[1]
                candidate_hl = protected_high = external_low = candidate_lh = None
                emit("uptrend_confirmed", external_high["price"], protected_low)

        # 5. Promote external extremes AFTER testing against the old break level.
        if accepted is not None:
            if (trend == "downtrend" and accepted["kind"] == "L" and protected_high and external_low
                    and accepted["pivot_index"] > protected_high["pivot_index"]
                    and lt(accepted["price"], external_low["price"])):
                external_low = accepted
                candidate_lh = None
            elif (trend == "uptrend" and accepted["kind"] == "H" and protected_low and external_high
                    and accepted["pivot_index"] > protected_low["pivot_index"]
                    and gt(accepted["price"], external_high["price"])):
                external_high = accepted
                candidate_hl = None

        high_relation = ("NA" if len(highs) < 2 else "HH" if gt(highs[-1]["price"], highs[-2]["price"])
                         else "LH" if lt(highs[-1]["price"], highs[-2]["price"]) else "EH")
        low_relation = ("NA" if len(lows) < 2 else "HL" if gt(lows[-1]["price"], lows[-2]["price"])
                        else "LL" if lt(lows[-1]["price"], lows[-2]["price"]) else "EL")
        local_structure = {
            ("LH", "LL"): "down_structure", ("HH", "HL"): "up_structure",
            ("LH", "HL"): "contraction_or_transition", ("HH", "LL"): "expansion_or_transition",
        }.get((high_relation, low_relation), "incomplete_or_equal")
        previous_high = protected_high if trend == "downtrend" else external_high if trend == "uptrend" else latest_high
        previous_low = external_low if trend == "downtrend" else protected_low if trend == "uptrend" else latest_low
        if trend == "downtrend":
            phase = ("downtrend_lh_confirmed" if candidate_lh else "downtrend_impulse"
                     if previous_low and lt(close, previous_low["price"], buffer) else "downtrend_pullback_or_range")
        elif trend == "uptrend":
            phase = ("uptrend_hl_confirmed" if candidate_hl else "uptrend_impulse"
                     if previous_high and gt(close, previous_high["price"], buffer) else "uptrend_pullback_or_range")
        else:
            phase = "transition_or_range"
        history.append({
            "index": index, "trend": trend, "trend_since_index": trend_since_index, "phase": phase,
            "high_relation": high_relation, "low_relation": low_relation, "local_structure": local_structure,
            "previous_high": previous_high, "previous_low": previous_low,
            "protected_high": protected_high, "protected_low": protected_low,
            "external_high": external_high, "external_low": external_low,
            "candidate_lh": candidate_lh, "candidate_hl": candidate_hl, "event": event,
        })
    return {
        "config": config, "swings": swings, "swing_log": swing_log,
        "history": history, "events": events,
        "current": history[-1] if history else None,
        "ambiguous_pivot_count": ambiguous_pivot_count,
    }
