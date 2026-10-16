"""Reuse canonical market JSON without changing Entry checkpoint hashes."""
import hashlib
import json


class MarketPrefix:
    """Process-local serialization cache, revalidated against the current frame.

    Hash exactly the original ``fingerprint(frame.iloc[:n].to_dict('split'))``
    bytes. Cached suffix bytes never enter a prefix hash. The independent
    frame copy detects in-place edits as well as replaced/continued markets;
    no cache or authentication shortcut is persisted in checkpoints.
    """

    def __init__(self, frame):
        self.frame = frame.copy(deep=True)
        split = frame.to_dict(orient='split')
        encode = lambda value: json.dumps(value, sort_keys=True, default=str).encode()
        self.header = b'{"columns": ' + encode(split['columns']) + b', "data": ['
        self.data, self.data_ends = self._join(encode(row) for row in split['data'])
        self.index, self.index_ends = self._join(encode(value) for value in split['index'])

    @staticmethod
    def _join(values):
        parts, ends = [], [0]
        for value in values:
            part = (b', ' if parts else b'') + value
            parts.append(part)
            ends.append(ends[-1] + len(part))
        return b''.join(parts), ends

    def matches(self, frame):
        return self.frame.equals(frame)

    def digest(self, count):
        result = hashlib.sha256(self.header)
        result.update(memoryview(self.data)[:self.data_ends[count]])
        result.update(b'], "index": [')
        result.update(memoryview(self.index)[:self.index_ends[count]])
        result.update(b']}')
        return result.hexdigest()
