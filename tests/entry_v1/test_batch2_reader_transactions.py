"""Exact local undo through the existing native Reader and role/source adapters."""
import pytest

from data_multi_fixture import build_sources,open_stream,exact


@pytest.fixture(scope='module')
def config(tmp_path_factory):
    import json
    return json.loads(build_sources(tmp_path_factory.mktemp('batch2_transactions')).read_text())


def test_nested_reader_undo_preserves_preexisting_coverage_and_rng(config):
    stream=open_stream(config);reader=stream.readers['alpha']
    try:
        reader.sample(2,8);before=reader.state_dict()
        outer=reader.begin_batch();reader.sample(3,8)
        middle=reader.state_dict();inner=reader.begin_batch();reader.sample(4,8)
        with pytest.raises(ValueError,match='OWNER_OR_ORDER'):reader.commit_batch(outer)
        reader.rollback_batch(inner);exact(middle,reader.state_dict())
        inner=reader.begin_batch();reader.sample(2,8);reader.commit_batch(inner)
        reader.rollback_batch(outer);exact(before,reader.state_dict())
        expected=reader.sample(4,8);reader.load_state_dict(before);exact(expected,reader.sample(4,8))
        with pytest.raises(ValueError,match='OWNER_OR_ORDER'):reader.rollback_batch(outer)
        with pytest.raises(ValueError,match='OWNER_OR_ORDER'):stream.readers['beta'].commit_batch(outer)
    finally:stream.close()


def test_reader_chunk_failure_restores_rng_before_any_delivery(config,monkeypatch):
    stream=open_stream(config);reader=stream.readers['alpha']
    try:
        before=reader.state_dict()
        monkeypatch.setattr(reader,'_chunk',lambda *a:(_ for _ in ()).throw(OSError('chunk fault')))
        with pytest.raises(OSError,match='chunk fault'):reader.sample(3,8)
        exact(before,reader.state_dict())
    finally:stream.close()


def test_cross_source_outer_merge_failure_undoes_committed_children(config,monkeypatch):
    import gym_trading_env.research.fg_s3_e_quality_multi as module
    stream=open_stream(config)
    try:
        stream.sample(8,8);before=stream.state_dict()
        monkeypatch.setattr(module,'_merge',lambda *a:(_ for _ in ()).throw(ValueError('outer merge fault')))
        with pytest.raises(ValueError,match='outer merge fault'):stream.sample(8,8)
        exact(before,stream.state_dict())
    finally:stream.close()


def test_multi_f3_statistics_failure_undoes_both_roles_and_sources(config,monkeypatch):
    from test_data_multi_f3 import open_f3
    stream=open_f3(config,'MIXED')
    try:
        stream.sample();before=stream.save()
        original=stream.group.__class__.__next__
        def fail_after_roles(group):
            original(group);raise ValueError('after both roles')
        monkeypatch.setattr(stream.group.__class__,'__next__',fail_after_roles)
        with pytest.raises(ValueError,match='after both roles'):stream.sample()
        exact(before,stream.save())
    finally:stream.close()
