"""Copy native JSON account snapshots without re-encoding equity history."""
from __future__ import annotations

import json


_HISTORY_FIELDS = frozenset(('time', 'equity', 'unrealized_pnl', 'net_lots', 'intrabar'))
_NUMBERS = (int, float)


def copy_account_snapshot(state):
    """Exactly retain the public JSON round-trip and independent ownership.

    Native equity samples have five primitive JSON fields. Copy those rows
    directly instead of decimal-formatting and reparsing every historical
    float on every teacher restore. Other values retain the original JSON
    normalization, including noncanonical externally supplied history rows.
    No cache, shortened history, or mutable source reference enters a copy.
    """
    history = state.get('equity_history')
    from .native_account import NativeHistory
    if isinstance(history, NativeHistory):
        result = json.loads(json.dumps({k: v for k, v in state.items() if k != 'equity_history'}, default=str))
        result['equity_history'] = history.to_public()
        return result
    if type(history) is not list:
        return json.loads(json.dumps(state, default=str))
    result = json.loads(json.dumps({k: v for k, v in state.items() if k != 'equity_history'}, default=str))
    copied = []
    for row in history:
        if (type(row) is dict and row.keys() == _HISTORY_FIELDS
                and type(row['time']) is str and type(row['intrabar']) is bool
                and type(row['equity']) in _NUMBERS
                and type(row['unrealized_pnl']) in _NUMBERS
                and type(row['net_lots']) in _NUMBERS):
            copied.append(row.copy())
        else:
            copied.append(json.loads(json.dumps(row, default=str)))
    result['equity_history'] = copied
    return result
