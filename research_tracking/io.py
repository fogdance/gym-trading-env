"""Bounded source reads and atomic writes to tracker-owned directories."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

LIMIT = 16 * 1024 * 1024


class TrackingError(ValueError):
    pass


def fingerprint(value):
    # This is the existing FG signed-manifest convention, including spaces.
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def small_bytes(path, limit=LIMIT):
    path = Path(path)
    if path.stat().st_size > limit:
        raise TrackingError(f'Source file exceeds bounded metadata limit: {path.name}')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise TrackingError(f'Source file grew beyond metadata limit: {path.name}')
    return data


def read_json(path, *, signed=False, limit=LIMIT):
    try:
        data = json.loads(small_bytes(path, limit))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackingError(f'Cannot read source metadata: {Path(path).name}') from exc
    if not isinstance(data, dict):
        raise TrackingError(f'Expected metadata object: {Path(path).name}')
    if signed and data.get('content_hash') != fingerprint(
            {k: v for k, v in data.items() if k != 'content_hash'}):
        raise TrackingError(f'Signed source metadata differs: {Path(path).name}')
    return data


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def prefix_digest(path, end):
    result = hashlib.sha256()
    remaining = end
    with Path(path).open('rb') as stream:
        while remaining:
            data = stream.read(min(1024 * 1024, remaining))
            if not data:
                raise TrackingError('Source log is shorter than its confirmed prefix')
            result.update(data)
            remaining -= len(data)
    return result.hexdigest()
