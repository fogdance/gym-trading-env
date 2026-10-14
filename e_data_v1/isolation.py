"""Hash evidence for protected source/config/archive files; never write them."""
from __future__ import annotations

from pathlib import Path
import subprocess

from .io import git_identity, stable_identity


def capture_protected(repositories, files):
    roots, members = {}, {}
    for root in repositories:
        root = Path(root).resolve()
        roots[str(root)] = git_identity(root)
        names = subprocess.check_output(['git','-C',str(root),'ls-files','-z']).decode().split('\0')
        for name in names:
            if name and (root/name).is_file():
                record = stable_identity(root/name)
                members[record['path']] = record
    for path in files:
        record = stable_identity(path)
        members[record['path']] = record
    return dict(repositories=roots, files=members,
                scope='tracked source files and specified immutable formal configs/archive evidence; no mutable live Replay/log hashes')


def verify_protected(before):
    differences = []
    for root, record in before['repositories'].items():
        after = git_identity(root)
        if after != record:
            differences.append(dict(path=root,before=record,after=after))
    for path, record in before['files'].items():
        after = stable_identity(path)
        if after != record:
            differences.append(dict(path=path,before=record,after=after))
    return dict(status='PASS' if not differences else 'FAIL',
                checked_repositories=len(before['repositories']),checked_files=len(before['files']),
                differences=differences,scope=before['scope'])
