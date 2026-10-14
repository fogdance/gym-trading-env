"""Historical Git identity and standalone trees, without source Git mutations."""
import hashlib
import io
from pathlib import Path
import tarfile

from .io import ArchiveError, atomic_json, digest, fingerprint, git, relative


def audit(source):
    repositories, missing = {}, []
    for path, sha in source.identity['versions'].items():
        root = Path(path); name = root.name
        if name not in ('gym-trading-env', 'dreamerv3'):
            continue
        try:
            head = git(root, 'rev-parse', 'HEAD').decode().strip()
            git(root, 'cat-file', '-e', sha+'^{commit}')
            tree = git(root, 'rev-parse', sha+'^{tree}').decode().strip()
            recorded_dirty = source.preparation.get('dirty', {}).get(path)
            rows = {}
            for p, expected in source.identity['code'].items():
                p = Path(p)
                if not p.is_relative_to(root):
                    continue
                rel = str(p.relative_to(root))
                try:
                    blob = git(root, 'show', sha+':'+rel)
                except Exception:
                    blob = None
                historical = hashlib.sha256(blob).hexdigest() if blob is not None else None
                actual = digest(p) if p.is_file() else None
                if historical == expected:
                    origin = 'COMMITTED_BLOB'
                elif actual == expected and recorded_dirty and head == sha:
                    origin = 'VERIFIED_DIRTY_EXECUTION_BYTES'
                else:
                    origin = 'UNAVAILABLE'
                    missing.append('historical execution source: '+str(p))
                rows[rel] = dict(expected_sha256=expected, historical_sha256=historical,
                                 current_sha256=actual, snapshot_origin=origin)
            modules = git(root, 'ls-tree', '-r', sha).decode().splitlines()
            if any(line.startswith('160000 ') for line in modules):
                missing.append(name+': submodule snapshot adapter required')
            attributes = git(root, 'ls-tree', '-r', '--name-only', sha).decode().splitlines()
            if '.gitattributes' in attributes and b'filter=lfs' in git(root, 'show', sha+':.gitattributes'):
                missing.append(name+': Git LFS content snapshot adapter required')
            if recorded_dirty is None:
                missing.append(name+': original dirty state not recorded')
            repositories[name] = dict(path=str(root), sha=sha, historical_tree=tree, current_head=head,
                dirty_at_prepare=recorded_dirty,
                current_status=git(root, 'status', '--porcelain=v1', '--untracked-files=all').decode().splitlines(),
                code_files=rows)
        except Exception as exc:
            missing.append(name+': cannot reconstruct recorded Git commit: '+type(exc).__name__)
    if set(repositories) != {'gym-trading-env', 'dreamerv3'}:
        missing.append('both authenticated Git repositories required')
    return dict(repositories=repositories, missing=missing, verified=not missing)


def snapshot(root, report):
    for name, repo in report['repositories'].items():
        dest = root/'provenance/source_bundle'/name
        # Re-extraction is confined to owned staging and completes interrupted
        # source exports; an existing directory alone is not a completion marker.
        dest.mkdir(parents=True,exist_ok=True)
        content = git(repo['path'], 'archive', '--format=tar', repo['sha'])
        with tarfile.open(fileobj=io.BytesIO(content)) as archive:
            for member in archive.getmembers():
                relative(dest, member.name)
                if member.issym() or member.islnk():
                    relative(dest, str(Path(member.name).parent/member.linkname))
            archive.extractall(dest, filter='data')
        for rel, row in repo['code_files'].items():
            target = relative(dest, rel)
            if row['snapshot_origin'] == 'VERIFIED_DIRTY_EXECUTION_BYTES':
                original = Path(repo['path'])/rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(original.read_bytes())
            if not target.is_file() or digest(target) != row['expected_sha256']:
                raise ArchiveError('Snapshot does not reconstruct runtime source: '+rel)
        dirty_execution = [rel for rel,row in repo['code_files'].items()
                           if row['snapshot_origin']=='VERIFIED_DIRTY_EXECUTION_BYTES']
        # A newer HEAD/current diff is observation, not the historical runtime patch.
        patch = git(repo['path'], 'diff', '--binary', repo['sha'], '--', *dirty_execution) if dirty_execution else b''
        (root/'provenance/source_bundle'/(name+'.execution.binary.patch')).write_bytes(patch)
    atomic_json(root/'provenance/code_identity.json', report)
