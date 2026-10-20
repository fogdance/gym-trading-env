"""Per-query frozen-file checks with shared directory traversal.

No checks are deferred: every file is stat'ed on every query. Plain absolute
paths share ancestor lstat checks before/after the file pass. A changed namespace,
symlink or relative path uses the original strict resolve+stat comparison.
"""
from pathlib import Path
import stat


def _identity(value):
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)


class FrozenSourceStats:
    def __init__(self, records):
        self.records=tuple((Path(name),resolved,expected) for name,(resolved,expected) in records.items())
        self._plain_absolute=all(p.is_absolute() and '..' not in p.parts for p,_,_ in self.records)
        self.parents=tuple(sorted({p for path,_,_ in self.records for p in path.parents},key=str))
        self._parents=self._directory_state() if self._plain_absolute else None
        self.statistics=dict(queries=0,files_checked=0,fast_queries=0,full_resolutions=0)

    def _directory_state(self):
        return tuple((_identity(s),s.st_mode) for path in self.parents for s in (path.lstat(),))

    def _exact(self,path,resolved,expected):
        self.statistics['full_resolutions']+=1
        if str(path.resolve(strict=True))!=resolved or _identity(path.stat())!=expected:
            raise ValueError('Frozen quality source changed during collection: '+str(path))

    def _fast_files(self):
        for path,resolved,expected in self.records:
            actual=path.lstat()
            if not stat.S_ISREG(actual.st_mode):
                self._exact(path,resolved,expected)
            elif _identity(actual)!=expected:
                raise ValueError('Frozen quality source changed during collection: '+str(path))

    def check(self):
        self.statistics['queries']+=1
        self.statistics['files_checked']+=len(self.records)
        if not self._plain_absolute:
            for record in self.records:self._exact(*record)
            return
        before=self._directory_state()
        if before==self._parents and all(stat.S_ISDIR(mode) for _,mode in before):
            self._fast_files()
            # A namespace change during the pass cannot defer detection to a
            # later query, even if the final source inode stayed unchanged.
            after=self._directory_state()
            if before==after:
                self.statistics['fast_queries']+=1
                return
            before=after
        for record in self.records:self._exact(*record)
        after=self._directory_state()
        # Directory mtime changes from unrelated output creation are allowed
        # by the old guard. Reuse a new traversal only after exact verification
        # under a stable namespace; symlink ancestors always take the slow path.
        self._parents=after if before==after else None
