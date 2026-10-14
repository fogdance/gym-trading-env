"""Archive-owned I/O; no helper writes a source artifact."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from research_tracking.io import atomic_json, digest, fingerprint, read_json


class ArchiveError(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE,
                                   env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'))


def relative(root, name):
    p = root/name
    if Path(name).is_absolute() or not p.resolve().is_relative_to(root.resolve()):
        raise ArchiveError('Artifact path escapes its owned directory: '+str(name))
    return p


def independent(root, protected):
    root = Path(root).resolve()
    for source in protected:
        source = Path(source).resolve()
        if root.is_relative_to(source) or source.is_relative_to(root):
            raise ArchiveError('Output overlaps a protected source: '+str(source))
    return root


def prefix_hash(path, size):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        left = size
        while left:
            data = stream.read(min(left, 4*1024*1024))
            if not data:
                raise ArchiveError('Source is shorter than its committed boundary')
            h.update(data)
            left -= len(data)
    return h.hexdigest()


def files(root):
    for p in sorted(Path(root).rglob('*')):
        if p.is_symlink():
            if not p.resolve().is_relative_to(Path(root).resolve()):
                raise ArchiveError('External archive symlink: '+str(p))
        if p.is_file():
            yield p


def inventory(root):
    selected = [p for p in files(root) if p.name != 'SHA256SUMS' and not p.name.startswith('.pending')]
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        hashes = list(pool.map(digest, selected))
    return {str(p.relative_to(root)): h for p, h in zip(selected, hashes)}


def seal(root):
    rows = inventory(root)
    temp = root/'.pending.SHA256SUMS'
    with temp.open('w') as out:
        for name, sha in rows.items():
            if '\n' in name or '\r' in name:
                raise ArchiveError('Newlines in artifact filenames are unsupported')
            out.write(sha+'  '+name+'\n')
        out.flush(); os.fsync(out.fileno())
    os.replace(temp, root/'SHA256SUMS')
    return rows


def check_inventory(root):
    expected = {}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        sha, name = line.split('  ', 1)
        relative(root, name)
        if name in expected:
            raise ArchiveError('Duplicate inventory entry')
        expected[name] = sha
    actual = inventory(root)
    if actual != expected:
        bad = sorted(k for k in set(actual)|set(expected) if actual.get(k) != expected.get(k))
        raise ArchiveError('Archive inventory differs: '+', '.join(bad[:8]))
    return len(actual)


def exchange(staging, final):
    """Atomically swap complete directories; an interrupted upgrade retains both."""
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, 'renameat2'):
        raise ArchiveError('Atomic directory exchange unavailable; upgrade stays staged')
    result = libc.renameat2(-100, os.fsencode(staging), -100, os.fsencode(final), 2)
    if result:
        raise OSError(ctypes.get_errno(), 'Atomic archive directory exchange failed')


METADATA_FILES = ('archive_manifest.json','README.md','SHA256SUMS','reports/verify.json',
                  'reports/eval_report.json','reports/restore_report.json',
                  'wandb/milestone.json','wandb/publish_state.json')


def recover_metadata(root):
    """Recover an interrupted owned metadata transaction, preserving diagnostics."""
    base = root.parent/'.transactions'
    pending = base/(root.name+'.pending.json')
    if not pending.exists():
        return
    journal = read_json(pending)
    backup = Path(journal['backup'])
    known = set(journal['known_files'])
    for name in METADATA_FILES:
        p = root/name
        saved = backup/name
        if saved.exists():
            if digest(saved) != journal['backup_hashes'][name]:
                raise ArchiveError('Metadata recovery backup differs')
            p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(saved,p)
        elif p.exists():
            target = backup/'abandoned'/name;target.parent.mkdir(parents=True,exist_ok=True)
            p.rename(target)
    for p in list(files(root)):
        name = str(p.relative_to(root))
        if name not in known and name.startswith('verification/'):
            target = backup/'abandoned'/name;target.parent.mkdir(parents=True,exist_ok=True)
            p.rename(target)
    pending.rename(backup/'recovered.json')


@contextmanager
def metadata_transaction(root):
    """Caller holds archive lock; a durable journal makes sealing retryable."""
    recover_metadata(root)
    base = root.parent/'.transactions';base.mkdir(parents=True,exist_ok=True)
    backup = base/(root.name+'_'+uuid.uuid4().hex)
    backup.mkdir()
    hashes = {}
    for name in METADATA_FILES:
        p = root/name
        if p.exists():
            target=backup/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target);hashes[name]=digest(target)
    journal = dict(backup=str(backup),backup_hashes=hashes,
                   known_files=[str(p.relative_to(root)) for p in files(root)])
    pending=base/(root.name+'.pending.json');atomic_json(pending,journal)
    try:
        yield
        seal(root)
    except BaseException:
        recover_metadata(root)
        raise
    else:
        pending.rename(backup/'completed.json')


@contextmanager
def lock(base, key):
    directory = base/'.locks'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/(key+'.lock')).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArchiveError('Another archive operation owns this identity') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def stable_copy(row, target):
    """Bounded prefixes may grow at the tail; committed bytes must stay equal."""
    source = Path(row['source'])
    size, expected = row['bytes'], row['sha256']
    before = source.stat()
    if prefix_hash(source, size) != expected:
        raise ArchiveError('Source bytes changed since inspection: '+str(source))
    if target.exists():
        if target.is_symlink() or target.stat().st_ino == before.st_ino and target.stat().st_dev == before.st_dev:
            raise ArchiveError('Archive file is bound to its source')
        if target.stat().st_size == size and digest(target) == expected:
            return
        raise ArchiveError('Existing staged artifact conflicts; retain staging and inspect again')
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name('.pending.'+target.name)
    with source.open('rb') as inp, temp.open('wb') as out:
        left = size
        while left:
            data = inp.read(min(left, 4*1024*1024))
            if not data:
                raise ArchiveError('Source shortened while copying')
            out.write(data); left -= len(data)
        out.flush(); os.fsync(out.fileno())
    after = source.stat()
    if (before.st_ino != after.st_ino or before.st_dev != after.st_dev
            or prefix_hash(source, size) != expected or digest(temp) != expected
            or (not row.get('prefix') and (after.st_size != size or before.st_mtime_ns != after.st_mtime_ns))):
        raise ArchiveError('SNAPSHOT_UNSAFE: source changed while copying: '+str(source))
    os.replace(temp, target)
    shutil.copymode(source, target)
