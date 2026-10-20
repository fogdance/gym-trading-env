"""The faster traversal retains immediate strict path and stat identity checks."""
import os
from pathlib import Path

import pytest

from gym_trading_env.envs.entry_market_cache import _stat_identity
from gym_trading_env.utils.source_guard import FrozenSourceStats


def guard(path):
    return FrozenSourceStats({str(path):(str(path.resolve(strict=True)),_stat_identity(path))})


def test_every_query_checks_file_and_shared_ancestors_unrelated_output_is_allowed(tmp_path):
    path=tmp_path/'source';path.write_text('original')
    g=guard(path)
    for _ in range(3):g.check()
    assert g.statistics['files_checked']==3 and g.statistics['fast_queries']==3
    (tmp_path/'unrelated-output').write_text('allowed')
    parent=tmp_path.stat()
    os.utime(tmp_path,ns=(parent.st_atime_ns,parent.st_mtime_ns+1_000_000_000))
    g.check();g.check()
    assert g.statistics['full_resolutions']>=1 and g.statistics['files_checked']==5
    path.write_text('changed!')
    # This filesystem can report identical timestamps for immediate same-size
    # writes; exercise a real metadata change deterministically, without sleep.
    current=path.stat()
    os.utime(path,ns=(current.st_atime_ns,g.records[0][2][3]+1_000_000_000))
    assert _stat_identity(path)!=g.records[0][2]
    with pytest.raises(ValueError,match='Frozen quality source changed'):g.check()


def test_symlink_retarget_same_inode_is_rejected(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.write_text('same inode')
    os.link(a,b)
    path=tmp_path/'source';path.symlink_to(a)
    g=guard(path);g.check();expected=_stat_identity(path)
    path.unlink();path.symlink_to(b)
    assert _stat_identity(path)==expected  # stat alone is insufficient
    with pytest.raises(ValueError,match='Frozen quality source changed'):g.check()


def test_parent_retarget_during_pass_rechecks_original_resolved_identity(tmp_path,monkeypatch):
    folder=tmp_path/'original';folder.mkdir()
    path=folder/'source';path.write_text('same file')
    g=guard(path);g.check();expected=_stat_identity(path)
    other=tmp_path/'moved'
    original=g._fast_files
    def switch_after_files():
        original()
        folder.rename(other)
        folder.symlink_to(other,target_is_directory=True)
    monkeypatch.setattr(g,'_fast_files',switch_after_files)
    with pytest.raises(ValueError,match='Frozen quality source changed'):g.check()
    assert _stat_identity(path)==expected


def test_relative_path_retains_original_cwd_resolution_guard(tmp_path,monkeypatch):
    first=tmp_path/'one';second=tmp_path/'two';first.mkdir();second.mkdir()
    (first/'source').write_text('same inode');os.link(first/'source',second/'source')
    monkeypatch.chdir(first);g=guard(Path('source'));g.check()
    monkeypatch.chdir(second)
    with pytest.raises(ValueError,match='Frozen quality source changed'):g.check()
