import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from gym_trading_env.research.fg_s3_e_quality_reader import EQualityCourseReader,build_join_index,batch_pair_counts
from gym_trading_env.research.fg_s3_e_quality_supervision import SupervisionSource,QualitySupervisionAdmission
from gym_trading_env.research.fg_replay import replay_fields,E_QUALITY_SCHEMA
from wp03_fixture import native_corpus


@pytest.fixture(scope='module')
def indexed(tmp_path_factory):
    root=tmp_path_factory.mktemp('wp03_reader');corpus=native_corpus(root/'corpus')
    a=QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(corpus))
    index=root/'index.sqlite';build_join_index(a,index)
    return a,index


def reader(indexed,stage='WM_NATURAL',spec=None,seed=202,**kwargs):
    a,index=indexed
    return EQualityCourseReader.open(a,stage=stage,sampling_spec=spec or dict(mode='NATURAL'),seed=seed,
                                     index_path=index,**kwargs)


def same(a,b):
    assert set(a)==set(b)
    for k in a:np.testing.assert_array_equal(a[k],b[k],err_msg=k)


@pytest.mark.parametrize('stage',['WM_NATURAL','HEADS_VALID','ACTOR_STRICT','ACTOR_QUALITY_PAIR'])
def test_C01_native_contiguous_episode_safe_windows_all_stages(indexed,stage):
    r=reader(indexed,stage)
    try:
        b=r.sample(3,8)
        assert b['stepid'].shape==(3,9,20) and b['stepid'].dtype==np.uint8
        assert set(replay_fields(E_QUALITY_SCHEMA))<=b.keys()
        assert not b['is_first'][:,1:].any()
        assert not b['teacher_valid'][:,0].any() and not b['quality_event_valid'][:,0].any()
        for ids in b['stepid']:
            records=[r.db.execute('SELECT policy,segment,ordinal FROM rows WHERE stepid=?',(i.tobytes(),)).fetchone() for i in ids]
            assert len({row[:2] for row in records})==1
            assert [row[2] for row in records]==list(range(records[0][2],records[0][2]+9))
        if stage=='ACTOR_QUALITY_PAIR':assert batch_pair_counts(b)['quality_pair']>0
        assert r.coverage()['actual_optimizer_exposures']==0
    finally:r.close()


@pytest.mark.parametrize('mode',['NATURAL','STRATIFIED','MIXED'])
def test_C02_seed_and_complete_rng_state_exact_next_batch(indexed,mode):
    spec=dict(mode=mode)
    if mode!='NATURAL':spec['strata_weights']={'STRICT_ADD':1,'STRICT_WAIT':1,'SL30':1,'TP_FAST':1}
    if mode=='MIXED':spec['natural_weight']=.5
    a=reader(indexed,'ACTOR_STRICT',spec);b=reader(indexed,'ACTOR_STRICT',spec)
    try:
        same(a.sample(2,8),b.sample(2,8))
        saved=json.loads(json.dumps(a.state_dict()))
        expected=a.sample(2,8)
        b.sample(2,8);b.load_state_dict(saved)
        same(expected,b.sample(2,8))
        assert a.state_dict()==b.state_dict()
    finally:a.close();b.close()


def test_C03_missing_class_is_support_failure_without_exposure_or_rng_change(indexed):
    r=reader(indexed,'HEADS_VALID',dict(mode='STRATIFIED',strata_weights={'NOT_A_REAL_CLASS':1}))
    try:
        before=r.state_dict()
        with pytest.raises(ValueError,match='INSUFFICIENT_CURRICULUM_SUPPORT'):r.sample(1,8)
        assert r.state_dict()==before
        with pytest.raises(ValueError,match='INSUFFICIENT_CURRICULUM_SUPPORT'):r.sample(1,1000)
    finally:r.close()


def test_C04_restore_rejects_source_index_stage_or_sampling_change(indexed):
    r=reader(indexed)
    try:
        state=r.state_dict()
        for key in ('source','index','stage','sampling_spec'):
            bad=copy.deepcopy(state);bad[key]='changed'
            with pytest.raises(ValueError,match='RESTORE_IDENTITY_MISMATCH'):r.load_state_dict(bad)
        assert r.state_dict()==state
    finally:r.close()


def test_C05_same_price_different_account_and_reset_never_form_pair(indexed):
    r=reader(indexed,'ACTOR_QUALITY_PAIR')
    try:
        batch=r.sample(1,8)
        assert batch_pair_counts(batch)['quality_pair']>0
        changed=copy.deepcopy(batch)
        # Each effective state deliberately assigned another account domain.
        changed['quality_owner'][0,:,0]=np.arange(9,dtype=np.uint32)+1
        assert batch_pair_counts(changed)==dict(strict_pair=0,quality_pair=0)
        changed=copy.deepcopy(batch);changed['is_first'][...]=True
        assert batch_pair_counts(changed)==dict(strict_pair=0,quality_pair=0)
        changed=copy.deepcopy(batch);changed['quality_owner'][...]=0
        assert batch_pair_counts(changed)==dict(strict_pair=0,quality_pair=0)
    finally:r.close()


def test_C06_original_macd_bytes_and_bounded_cache_no_parallel_replay(indexed):
    r=reader(indexed,cache_bytes=1024)
    try:
        batch=r.sample(2,8)
        for b in range(2):
            for t,stepid in enumerate(batch['stepid'][b]):
                chunk,offset=r.db.execute('SELECT chunk,offset FROM rows WHERE stepid=?',(stepid.tobytes(),)).fetchone()
                with np.load(r.root/chunk,allow_pickle=False) as f:
                    assert f['market_macd_5m'][offset].tobytes()==batch['market_macd_5m'][b,t].tobytes()
        c=r.coverage();assert c['peak_cached_bytes']<=1024
        assert c['unique_market_cursors']==30 and c['distinct_account_decisions']==6*29
        assert c['delivered_counts']['delivered_candidate_exposures']==16
        assert c['actual_optimizer_exposures']==0
    finally:r.close()


def test_C07_relocation_preserves_reader_restore_without_old_absolute_root(indexed,tmp_path,monkeypatch):
    a,index=indexed;r=reader(indexed)
    try:
        r.sample(1,8);state=r.state_dict();expected=r.sample(2,8)
        root=tmp_path/'relocated';shutil.copytree(a.source.corpus_root,root)
        original_open=Path.open
        def without_origin(path,*args,**kwargs):
            if path.absolute().is_relative_to(a.source.corpus_root):
                raise AssertionError('Relocated Reader accessed original Corpus location')
            return original_open(path,*args,**kwargs)
        monkeypatch.setattr(Path,'open',without_origin)
        current=QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(root))
        other=EQualityCourseReader.open(current,stage='WM_NATURAL',sampling_spec=dict(mode='NATURAL'),
            seed=0,index_path=index)
        try:other.load_state_dict(state);same(expected,other.sample(2,8))
        finally:other.close()
    finally:r.close()


def test_C08_open_reader_detects_changed_sidecar_or_chunk(indexed,tmp_path):
    a,index=indexed;root=tmp_path/'copy';shutil.copytree(a.source.corpus_root,root)
    current=QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(root))
    r=EQualityCourseReader.open(current,stage='WM_NATURAL',sampling_spec=dict(mode='NATURAL'),seed=0,index_path=index)
    try:
        p=root/'teacher_sidecar/P0/steps.jsonl.gz'
        p.write_bytes(p.read_bytes()+b'changed')
        with pytest.raises(ValueError,match='CONTENT_CHANGED'):r.sample(1,8)
    finally:r.close()


def test_C09_mixed_requires_explicit_weights_and_never_interprets_empty_mode(indexed):
    for spec in [dict(mode=''),dict(mode='MIXED'),dict(mode='STRATIFIED'),dict(mode='MIXED',natural_weight=.5,strata_weights={'SL30':0})]:
        with pytest.raises(ValueError):reader(indexed,spec=spec)


def test_C10_window_crosses_original_native_chunk_files_without_crossing_episode(indexed,tmp_path):
    import embodied
    from gym_trading_env.research.fg_replay import (iter_replay_rows,iter_sidecar,trace_manifest,
        read_manifest,write_json,signed,digest)
    from wp03_fixture import inventory,rewrite_rows
    admission,_=indexed;root=tmp_path/'corpus';shutil.copytree(admission.source.corpus_root,root)
    original=root/'policies/P0';student=root/'student_replay/P0'
    m=trace_manifest(original,schema=E_QUALITY_SCHEMA)
    rows=list(iter_sidecar(original))
    replay=embodied.Replay(length=3,chunksize=8,capacity=None,directory=str(tmp_path/'native_chunks'),save_wait=True)
    try:
        for (ordinal,step),row in zip(iter_replay_rows(original,m),rows):
            replay.add({k:v for k,v in step.items() if k!='stepid'},worker=0)
            uuid,offset=replay.streams[0][-1];row['stepid']=(bytes(uuid)+offset.to_bytes(4,'big')).hex()
        replay.save()
        chunks=[dict(path='replay/'+c.filename,uuid=str(c.uuid),succ=str(c.succ),length=c.length)
                for c in replay.chunks.values() if c.length]
    finally:replay.workers.shutdown(wait=True)
    for chunk in chunks:
        for dest in (original,student):
            target=dest/chunk['path'];target.parent.mkdir(exist_ok=True)
            shutil.copyfile(tmp_path/'native_chunks'/Path(chunk['path']).name,target)
    m.pop('content_hash');m['chunks']=chunks
    m['files'].update({c['path']:digest(original/c['path']) for c in chunks})
    write_json(original/'manifest.json',signed(m))
    published=read_manifest(student/'manifest.json');published.pop('content_hash')
    published['chunks']=chunks;published['files']={c['path']:digest(student/c['path']) for c in chunks}
    write_json(student/'manifest.json',signed(published))
    rewrite_rows(root,'P0',rows)
    inventory(root,read_manifest(root/'corpus_manifest.json'))
    current=QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(root))
    index=tmp_path/'index.sqlite';build_join_index(current,index)
    r=EQualityCourseReader.open(current,stage='WM_NATURAL',sampling_spec=dict(mode='NATURAL',policies=['P0']),seed=202,index_path=index)
    try:
        batch=r.sample(1,8)
        files={r.db.execute('SELECT chunk FROM rows WHERE stepid=?',(i.tobytes(),)).fetchone()[0] for i in batch['stepid'][0]}
        assert len(files)>1 and not batch['is_first'][:,1:].any()
    finally:r.close()
