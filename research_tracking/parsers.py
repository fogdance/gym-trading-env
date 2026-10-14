"""Incremental complete-line readers and checkpoint-committed watermarks."""
import hashlib
import json
from pathlib import Path

from .io import TrackingError, prefix_digest

MAX_LINE = 2 * 1024 * 1024


class JsonLines:
    def __init__(self, path, cursor=None):
        self.path = Path(path)
        self.cursor = dict(cursor or {'offset': 0})
        self.offset = self.cursor.get('offset', 0)
        self.line_start = self.cursor.get('line_start', 0)
        self.line_sha256 = self.cursor.get('line_sha256')
        self.inode = None

    def changed_prefix(self):
        if not self.offset:
            return False
        if not self.path.exists() or self.path.stat().st_size < self.offset:
            return True
        with self.path.open('rb') as stream:
            stream.seek(self.line_start)
            line = stream.read(self.offset - self.line_start)
        return hashlib.sha256(line).hexdigest() != self.line_sha256

    def verify_prefix(self):
        if self.changed_prefix():
            raise TrackingError('Source log rolled back or changed at its saved boundary')
        if self.cursor.get('prefix_sha256') and prefix_digest(self.path, self.offset) != self.cursor['prefix_sha256']:
            raise TrackingError('Previously confirmed source log prefix changed')

    def rows(self):
        if not self.path.exists():
            return
        self.inode = self.path.stat().st_ino
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            while True:
                start = stream.tell()
                line = stream.readline(MAX_LINE + 1)
                if len(line) > MAX_LINE:
                    raise TrackingError('Source JSONL line exceeds bounded record limit')
                if not line or not line.endswith(b'\n'):
                    return
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TrackingError('Invalid complete JSONL line in source log') from exc
                if not isinstance(row, dict):
                    raise TrackingError('Source JSONL record is not an object')
                # Cursor advances only when the consumer accepts this row.
                yield row, dict(offset=stream.tell(), line_start=start,
                                line_sha256=hashlib.sha256(line).hexdigest(), inode=self.inode)

    def accept(self, cursor):
        self.offset = cursor['offset']
        self.line_start = cursor['line_start']
        self.line_sha256 = cursor['line_sha256']

    def saved_cursor(self, *, hash_prefix=False):
        result = dict(offset=self.offset, line_start=self.line_start,
                      line_sha256=self.line_sha256, inode=self.inode)
        if hash_prefix and self.offset:
            result['prefix_sha256'] = prefix_digest(self.path, self.offset)
        return result


def watermark(source, checkpoints=None):
    checkpoints = source.checkpoints() if checkpoints is None else checkpoints
    candidates = [c for c in checkpoints if c['local_artifact_available']]
    if not candidates:
        raise TrackingError('No complete and locally available Checkpoint metadata')
    checkpoint = max(candidates, key=lambda c: c['env_steps'])
    target = checkpoint['env_steps']
    offset = source.identity['source_update_offset']
    if target == 0:
        return dict(checkpoint, local_update=0, agent_update=offset, cycle_count=0)
    reader = JsonLines(source.root/'cycles.jsonl')
    previous_env, previous_update, count = 0, offset, 0
    for row, cursor in reader.rows():
        env, update = row['env_steps'], row['agent_updates']
        if env > target:
            break
        if (type(env) is not int or type(update) is not int
                or env <= previous_env or update <= previous_update
                or row['counts']['env_steps'] != env):
            raise TrackingError('Checkpoint cycle prefix chronology differs')
        previous_env, previous_update = env, update
        count += 1
        if env == target:
            return dict(checkpoint, local_update=update-offset, agent_update=update,
                        cycle_count=count)
    raise TrackingError('Completed Checkpoint has no matching complete cycle boundary')
