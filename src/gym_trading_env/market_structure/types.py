"""Public, JSON-serializable return types. All indices are zero-based."""

from typing import Literal, TypedDict

Trend = Literal["uptrend", "downtrend", "neutral"]
SwingKind = Literal["H", "L"]
SwingAction = Literal["added", "replaced", "ignored"]
PlateauPolicy = Literal["last", "strict"]
AmbiguousPolicy = Literal["skip", "raise"]
EventType = Literal[
    "uptrend_confirmed", "uptrend_continuation", "uptrend_invalidated",
    "downtrend_confirmed", "downtrend_continuation", "downtrend_invalidated",
]


class OhlcBar(TypedDict):
    open: float
    high: float
    low: float
    close: float


class StructureConfig(TypedDict):
    left: int
    right: int
    plateau_policy: PlateauPolicy
    ambiguous_policy: AmbiguousPolicy
    structure_tolerance_pct: float
    break_buffer_pct: float


class SwingPoint(TypedDict):
    kind: SwingKind
    price: float
    pivot_index: int
    confirm_index: int


class SwingLogEntry(TypedDict):
    point: SwingPoint
    action: SwingAction


class StructureEvent(TypedDict):
    index: int
    type: EventType
    break_level: float
    protected_point: SwingPoint | None


class StructureState(TypedDict):
    index: int
    trend: Trend
    trend_since_index: int | None
    phase: str
    high_relation: str
    low_relation: str
    local_structure: str
    previous_high: SwingPoint | None
    previous_low: SwingPoint | None
    protected_high: SwingPoint | None
    protected_low: SwingPoint | None
    external_high: SwingPoint | None
    external_low: SwingPoint | None
    candidate_lh: SwingPoint | None
    candidate_hl: SwingPoint | None
    event: EventType | None


class StructureResult(TypedDict):
    config: StructureConfig
    swings: list[SwingPoint]
    swing_log: list[SwingLogEntry]
    history: list[StructureState]
    events: list[StructureEvent]
    current: StructureState | None
    ambiguous_pivot_count: int
