"""Original collector fixtures, explicitly synthetic; real TRAIN audit is CLI."""
from pathlib import Path
import copy

import numpy as np
import pytest

from test_fg_s3a_labels import native_dataset
from gym_trading_env.research.fg_s3a_labels import build_labels
from gym_trading_env.research.fg_s3_teacher_catalog import (
    build_catalog, load_catalog, DatasetIdentityError, ROW_DTYPE)
from gym_trading_env.research.fg_replay import (digest, iter_replay_rows, read_manifest, signed, write_json,
    collect_trace,reconcile_trace,SINGLE3_SCHEMA)


@pytest.fixture(scope='module')
def catalog_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp('shared_teacher')
    dataset, oracle, trace, tm, factory = native_dataset(root/'native',return_factory=True)
    # Natural uses the signed Replay identity, whereas P4 uses Oracle identity.
    # Collect a separate ORIGINAL account; do not relabel/copy a P4 chunk.
    from gym_trading_env.research.fg_single3_replay import Single3BehaviorPolicy
    dreamer=str(Path(__file__).resolve().parents[3]/'dreamerv3')
    nat=dataset/'traces/C1/P1'/tm['contract']
    ident=dict(schema=SINGLE3_SCHEMA,original_training_hash='synthetic_TRAIN',fold='TRAIN',
               conditional_direction_only=True,selected_contracts=[tm['contract']])
    nt=collect_trace(root/'native/env/env.yaml',dict(id='P1',kind='always_add',seed_components=[0]),
                     nat,ident,dreamer,protocol='fg_single3_v1',policy_factory=Single3BehaviorPolicy,
                     env_factory=factory)
    nr=reconcile_trace(nat,dreamer,schema=SINGLE3_SCHEMA,env_factory=factory)
    write_json(nat/'reconciliation.json',signed(dict(nr,trace_manifest_sha256=digest(nat/'manifest.json'))))
    dm=read_manifest(dataset/'manifest.json');dm.pop('content_hash')
    dm['traces'].append(dict(candidate='C1',policy_id='P1',contract=nt['contract'],
        directory=str(nat.relative_to(dataset)),manifest_sha256=digest(nat/'manifest.json'),
        receipt_sha256=digest(nat/'reconciliation.json'),trace_identity=nt['identity'],records=nt['records'],
        replay_trace_hash=nt['replay_trace_hash'],reward_params_hash=nt['reward_params_hash'],
        observation_identity=nt['observation_identity']))
    dm['pool_members']['C1_natural_oracle']=[0,1]
    write_json(dataset/'manifest.json',signed(dm))
    labels = root/'labels'
    build_labels(dataset, oracle, labels)
    out = root/'catalog'
    build_catalog(dataset, oracle, labels, out, diagnostic=True)
    return root, dataset, oracle, labels, trace, tm, out


def test_complete_view_exact_sparse_and_pure_join(catalog_bundle):
    from embodied.run.fg_s3_teacher_adapter import attach_teacher
    root, dataset, oracle, labels, trace, tm, out = catalog_bundle
    c = load_catalog(out)
    lm = read_manifest(labels/'manifest.json')
    sparse = np.load(labels/lm['traces'][0]['labels_file'])
    for row in sparse:
        got = c.lookup(row['stepid'])
        assert got.status == ('VALID_ADD' if row['label'] else 'VALID_WAIT')
        for k in sparse.dtype.names:
            np.testing.assert_array_equal(got.record[k], row[k])
    steps = [s for _,s in iter_replay_rows(trace,tm)]
    held = c.lookup(steps[1]['stepid'])
    assert held.status == 'EXCLUDED_held_forced_wait' and held.record['label'] == 0
    nt=dataset/'traces/C1/P1'/tm['contract']
    natural=next(iter_replay_rows(nt,read_manifest(nt/'manifest.json')))[1]
    assert c.lookup(natural['stepid']).status=='NATURAL_NONTEACHER'
    batch = {k:np.stack([s[k] for s in steps[:65]])[None] for k in steps[0]}
    before = {k:v.copy() for k,v in batch.items()}
    joined = attach_teacher(batch,c,mode='train')
    assert joined['teacher_action'].dtype == np.int32
    assert joined['teacher_valid'].dtype == bool
    assert joined['teacher_action'][0,0] == 1 and not joined['teacher_valid'][0,1]
    for k in batch: np.testing.assert_array_equal(batch[k],before[k])
    class NoLookup:
        def lookup(self,*args): raise AssertionError('report queried teacher')
    report = attach_teacher(batch,NoLookup(),mode='report')
    assert not report['teacher_valid'].any() and not report['teacher_action'].any()
    shifted = copy.deepcopy(batch);shifted['stepid'] = np.roll(shifted['stepid'],1,axis=1)
    with pytest.raises(DatasetIdentityError,match='MISMATCH'):
        attach_teacher(shifted,c,mode='train')
    with pytest.raises(DatasetIdentityError,match='UNKNOWN_SOURCE'):c.lookup(bytes(20))
    uid = steps[0]['stepid'].tobytes()[:16]
    with pytest.raises(DatasetIdentityError,match='OFFSET_OUT_OF_RANGE'):
        c.lookup(uid+(2**32-1).to_bytes(4,'big'))
    assert len(c.index) == tm['records'] and c.valid_count == len(sparse)


@pytest.mark.parametrize('mutation,error',[('missing','ELIGIBLE_LABEL_MISSING'),
    ('invalid','INVALID_LABEL_PRESENT'),('label','LABEL_FIELD_MISMATCH'),
    ('duplicate','DUPLICATE_STEPID')])
def test_sparse_corruption_fails_closed(catalog_bundle,tmp_path,mutation,error):
    import shutil
    _,dataset,oracle,labels,trace,tm,_ = catalog_bundle
    dest = tmp_path/'labels';shutil.copytree(labels,dest)
    m = read_manifest(dest/'manifest.json');m.pop('content_hash')
    filename = m['traces'][0]['labels_file'];a = np.load(dest/filename)
    if mutation=='missing': a=a[1:]
    if mutation=='duplicate': a=np.concatenate([a,a[:1]])
    if mutation=='label': a['label'][0]=0
    if mutation=='invalid':
        x=a[:1].copy();x['stepid'][0]=next(s for n,s in iter_replay_rows(trace,tm) if n==1)['stepid']
        a=np.concatenate([a,x])
    np.save(dest/filename,a,allow_pickle=False)
    m['traces'][0]['samples']=len(a);m['files'][filename]=digest(dest/filename)
    write_json(dest/'manifest.json',signed(m))
    with pytest.raises(DatasetIdentityError,match=error):
        build_catalog(dataset,oracle,dest,tmp_path/'catalog',diagnostic=True)


def test_catalog_tamper_output_and_diagnostic_boundary(catalog_bundle,tmp_path):
    import shutil
    _,dataset,oracle,labels,_,_,out=catalog_bundle
    with pytest.raises(DatasetIdentityError,match='SYNTHETIC'):
        build_catalog(dataset,oracle,labels,tmp_path/'formal')
    with pytest.raises(FileExistsError):build_catalog(dataset,oracle,labels,out,diagnostic=True)
    dest=tmp_path/'catalog';shutil.copytree(out,dest)
    m=read_manifest(dest/'manifest.json');p=dest/m['traces'][0]['file']
    a=np.load(p);a['label'][0]=0;np.save(p,a)
    with pytest.raises(ValueError,match='changed'):load_catalog(dest)
