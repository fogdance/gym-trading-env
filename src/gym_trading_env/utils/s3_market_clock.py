"""The sole S3 economic clock; raw cursors remain physical Replay addresses."""
from __future__ import annotations

import numpy as np

PROTOCOL = 'S3_MARKET_MINUTE'
OBS_VERSION = 'FG_SINGLE3_MARKET_MINUTE_60x18_9_2'


def market_advance(valid_bar):
    """One complete observed 1m bar counts even when its volume is zero."""
    if not isinstance(valid_bar, (bool, np.bool_)):
        raise ValueError('Market clock requires an explicit valid-bar boolean')
    return int(valid_bar)


class MarketClock:
    """prefix[k] counts valid rows strictly before raw row k.

    observation a -> b consumes (a, b]; a fill on i held through j consumes
    [i, j]. First execution reward has discount exponent zero.
    """
    def __init__(self, valid):
        valid = np.asarray(valid)
        if valid.ndim != 1 or valid.dtype != np.bool_:
            raise ValueError('Market clock requires a 1D boolean validity array')
        self.valid = valid.copy()
        self.prefix = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
        self.rows = np.flatnonzero(valid)

    def _cursor(self, cursor):
        if not isinstance(cursor, (int, np.integer)) or not 0 <= cursor < len(self.valid):
            raise ValueError('Market clock cursor out of bounds')

    def advance(self, before, after):
        self._cursor(before); self._cursor(after)
        if after < before:
            raise ValueError('Market clock cannot move backwards')
        return int(self.prefix[after + 1] - self.prefix[before + 1])

    def held(self, entry, through):
        self._cursor(entry); self._cursor(through)
        if through < entry or not self.valid[entry]:
            raise ValueError('Holding clock needs a real fill bar and causal endpoint')
        return int(self.prefix[through + 1] - self.prefix[entry])

    def reward_exponent(self, decision, feedback):
        self._cursor(decision); self._cursor(feedback)
        if feedback <= decision:
            raise ValueError('Reward must follow its decision')
        return int(self.prefix[feedback] - self.prefix[decision + 1])

    def window(self, through, length):
        self._cursor(through)
        end = int(self.prefix[through + 1])
        return self.rows[max(0, end - length):end]
