"""Standalone swing/trend analysis matching the review-future TypeScript core.

Copied unchanged (user 2026-11-06: "趋势识别的代码在，可以拷贝过来") from
~/Documents/work/review_future/python/market_structure, not tracked by git there; sha256 of the sources:
core.py 3ff5e868834316a9808da5147119348668906871d617c3f256d4acfbc9d41f82,
types.py 779d98f9ddd93d74cacdd3b71c96a5d2192c96722cf90ed5f1f49dfe3aad9617,
__init__.py 92233c37cfa57bc14b1ce760b13508f249257d7a82d58f8562b2090248b03b1a (this docstring added),
__main__.py 3e7cc362a168e1372d7beed447942bffd397f1b75a0c03ffe3d1034d40e916b2.
"""

from .core import analyze_market_structure
from .types import (
    AmbiguousPolicy, EventType, OhlcBar, PlateauPolicy, StructureConfig,
    StructureEvent, StructureResult, StructureState, SwingAction, SwingKind,
    SwingLogEntry, SwingPoint, Trend,
)

__all__ = [
    "analyze_market_structure", "OhlcBar", "StructureConfig", "StructureResult",
    "StructureState", "StructureEvent", "SwingPoint", "SwingLogEntry", "Trend",
    "SwingKind", "SwingAction", "PlateauPolicy", "AmbiguousPolicy", "EventType",
]
