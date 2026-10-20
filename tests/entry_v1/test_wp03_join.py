import copy
import json
import shutil
import sqlite3

import numpy as np
import pytest

from gym_trading_env.research.fg_replay import iter_sidecar, read_manifest, signed, write_json, digest
from gym_trading_env.research.fg_s3_e_quality_supervision import (
    SupervisionSource, QualitySupervisionAdmission, project_label, supervision_settings)
from gym_trading_env.research.fg_s3_e_quality_reader import build_join_index
from wp03_fixture import native_corpus, inventory, rewrite_rows


@pytest.fixture(scope='module')
def native_base(tmp_path_factory):
    return native_corpus(tmp_path_factory.mktemp('wp03_native')/'corpus')


@pytest.fixture
def corpus(native_base,tmp_path):
    root=tmp_path/'corpus';shutil.copytree(native_base,root);return root


def joined(root,index):
    a=QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(root))
    return build_join_index(a,index)


def test_B01_six_native_accounts_stream_to_metadata_index(corpus,tmp_path):
    receipt=joined(corpus,tmp_path/'index.sqlite')
    assert set(receipt['policies'])=={'P0','P1','P2_low','P2_mid','P3','P4'}
    assert all(p['rows']>30 for p in receipt['policies'].values())
    assert receipt['actual_optimizer_exposures']==0
    db=sqlite3.connect(tmp_path/'index.sqlite')
    rows=db.execute('SELECT policy,ordinal,stepid,segment,labels,audit FROM rows').fetchall()
    assert all(len(r[2])==20 for r in rows)
    owners={}
    for policy,ordinal,stepid,segment,labels,audit in rows:
        labels=json.loads(labels);audit=json.loads(audit)
        if labels['teacher_valid'] or labels['quality_outcome_valid']:
            assert any(labels['quality_owner'])
            owners.setdefault((policy,segment),set()).add(tuple(labels['quality_owner']))
        for k in ('quality_q_c1_valid','quality_q_v2_valid','quality_soft_valid'):assert labels[k] is False
        assert labels['quality_soft_source']==0
    assert all(len(values)==1 for values in owners.values())
    assert len(set(next(iter(v)) for v in owners.values()))==len(owners)
    db.close()


@pytest.mark.parametrize('attack',['ordinal','stepid','missing','duplicate','shift','account','policy','executed','incoming_action'])
def test_B02_resigned_three_way_or_time_attacks_fail_closed(corpus,tmp_path,attack):
    rows=list(iter_sidecar(corpus/'policies/P0'))
    if attack=='ordinal':rows[2]['ordinal']+=1
    elif attack=='stepid':rows[2]['stepid']=rows[1]['stepid']
    elif attack=='missing':rows.pop(2)
    elif attack=='duplicate':rows.insert(2,copy.deepcopy(rows[2]))
    elif attack=='shift':rows[2]['teacher']=copy.deepcopy(rows[3]['teacher'])
    elif attack=='account':rows[2]['account_snapshot_hash']='d'*64
    elif attack=='policy':rows=list(iter_sidecar(corpus/'policies/P1'))
    elif attack=='executed':rows[2]['decision_executed']=False
    elif attack=='incoming_action':rows[2]['incoming_action']=1
    rewrite_rows(corpus,'P0',rows)
    with pytest.raises((ValueError,KeyError,sqlite3.IntegrityError)):
        joined(corpus,tmp_path/'index.sqlite')


def test_B03_sidecar_or_student_chunk_exchange_is_rejected(corpus,tmp_path):
    shutil.copyfile(corpus/'teacher_sidecar/P1/steps.jsonl.gz',corpus/'teacher_sidecar/P0/steps.jsonl.gz')
    inventory(corpus,read_manifest(corpus/'corpus_manifest.json'))
    with pytest.raises(ValueError,match='PUBLICATION_TRACE_OWNER'):
        joined(corpus,tmp_path/'index.sqlite')


def test_B04_l2_invalid_does_not_erase_l1_and_forced_wait_is_not_strict(corpus,tmp_path):
    joined(corpus,tmp_path/'index.sqlite')
    db=sqlite3.connect(tmp_path/'index.sqlite')
    labels=[json.loads(r[0]) for r in db.execute('SELECT labels FROM rows')]
    assert any(l['quality_outcome_valid'] and not l['teacher_valid'] for l in labels)
    assert any(not l['teacher_valid'] and not l['quality_outcome_valid'] for l in labels)
    db.close()


def test_B05_current_observation_teacher_not_behavior_action(corpus,tmp_path):
    joined(corpus,tmp_path/'index.sqlite')
    db=sqlite3.connect(tmp_path/'index.sqlite')
    labels=[json.loads(r[0]) for r in db.execute("SELECT labels FROM rows WHERE policy='P0'")]
    assert any(l['teacher_valid'] and l['teacher_action']==1 for l in labels)
    assert all(row['action']==0 for row in iter_sidecar(corpus/'policies/P0'))
    for l,first in db.execute('SELECT labels,first FROM rows'):
        label=json.loads(l)
        if first:assert not label['teacher_valid'] and not any(label['quality_event_valid'])
    db.close()


def test_B06_each_window_mask_is_independent_and_invalid_huge_target_zeroed(corpus):
    rows=list(iter_sidecar(corpus/'policies/P0'))
    row=next(r for r in rows if not r['reset'] and r['teacher']['L1']['score_valid'])
    label=copy.deepcopy(row['teacher']);facts=label['L1'];facts.pop('content_hash')
    facts['event_valid'][0]=False;facts['event'][0]=999999
    facts['touch_valid'][0][0]=False;facts['touch'][0][0]=1e30
    facts['adverse_valid'][0]=False;facts['adverse'][0]=1e30
    label['L1']=signed(facts);label.pop('content_hash');label=signed(label)
    projected=project_label(label,row,{'action_mask':np.ones(2,np.float32)},'1'*64)
    assert projected['quality_score_valid']
    assert projected['quality_event'][0]==projected['quality_touch'][0,0]==projected['quality_adverse'][0]==0
    assert projected['quality_event_valid'][1:].any()


def test_B07_fake_strict_wait_and_rational_witness_rejected(corpus):
    from gym_trading_env.research.fg_s3_quality_catalog_v2 import (
        validate_corpus_label,_check_numeric_pair_cached)
    row=next(r for r in iter_sidecar(corpus/'policies/P0') if r['teacher']['L2']['teacher_valid'])
    hits=_check_numeric_pair_cached.cache_info().hits
    validate_corpus_label(row['teacher']);validate_corpus_label(row['teacher'])
    assert _check_numeric_pair_cached.cache_info().hits>hits
    for attack in ('forced','witness','margin','actual_mask'):
        label=copy.deepcopy(row['teacher']);l2=label['L2'];l2.pop('content_hash')
        if attack=='forced':l2.update(teacher_valid=False,strict_action=0)
        elif attack=='witness':l2['value_rational_witness']=['0','0']
        elif attack=='margin':l2['margin']+=1
        else:
            facts=label['L1'];facts.pop('content_hash');facts['actual_legal_add']=False
            label['L1']=signed(facts)
        label['L2']=signed(l2);label.pop('content_hash');label=signed(label)
        with pytest.raises(ValueError):project_label(label,row,{'action_mask':np.ones(2,np.float32)},'1'*64)


def test_B08_pair_identity_requires_nonzero_same_owner_same_reset():
    import jax.numpy as jnp
    from dreamerv3.fg_s3_quality_v2 import pair_identity
    owners=jnp.ones((1,4,8),jnp.uint32)
    assert bool(pair_identity({'quality_owner':owners},jnp.zeros((1,4),bool))[0,0,1])
    zeros=jnp.zeros_like(owners)
    assert not np.asarray(pair_identity({'quality_owner':zeros},jnp.zeros((1,4),bool))).any()
    different=owners.at[:,1,:].set(2)
    assert not bool(pair_identity({'quality_owner':different},jnp.zeros((1,4),bool))[0,0,1])
    assert not bool(pair_identity({'quality_owner':owners},jnp.array([[1,0,1,0]],bool))[0,0,2])


def test_B09_new_protocol_rejects_old_q_soft_weights_and_preserves_legacy():
    from dreamerv3.fg_s3_quality_v2 import quality_extensions
    settings=supervision_settings();cfg=quality_extensions(settings)
    assert all(cfg[k]==0 for k in ('quality_q_c1_lambda','quality_q_v2_lambda','quality_soft_lambda'))
    for key in ('quality_q_c1_lambda','quality_q_v2_lambda','quality_soft_lambda'):
        with pytest.raises(ValueError,match='Q_SOFT_FORBIDDEN'):quality_extensions({**settings,key:.1})
    assert 'quality_supervision_protocol' not in quality_extensions({})


def test_B10_json_claim_is_not_an_admission_object(tmp_path):
    with pytest.raises(ValueError,match='PREFLIGHT_OBJECT_REQUIRED'):
        build_join_index(dict(status='SUPERVISION_DATA_READY'),tmp_path/'index.sqlite')


def test_B11_resigned_student_chunk_exchange_or_macd_change_rejected(corpus,tmp_path):
    path=corpus/'student_replay/P0'
    m=read_manifest(path/'manifest.json');chunk=path/m['chunks'][0]['path']
    with np.load(chunk,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    arrays['market_macd_5m'][1,0]+=1
    np.savez_compressed(chunk,**arrays)
    m.pop('content_hash');m['files'][m['chunks'][0]['path']]=digest(chunk)
    write_json(path/'manifest.json',signed(m));inventory(corpus,read_manifest(corpus/'corpus_manifest.json'))
    with pytest.raises(ValueError,match='PUBLICATION_TRACE_OWNER'):
        joined(corpus,tmp_path/'index.sqlite')


def test_B12_mae_does_not_change_score_and_nan_is_rejected_before_loss(corpus):
    row=next(r for r in iter_sidecar(corpus/'policies/P0') if not r['reset'] and r['teacher']['L1']['score_valid'])
    original=project_label(row['teacher'],row,{'action_mask':np.ones(2,np.float32)},'1'*64)
    label=copy.deepcopy(row['teacher']);facts=label['L1'];facts.pop('content_hash');facts['mae_full']=99999.
    label['L1']=signed(facts);label.pop('content_hash');label=signed(label)
    changed=project_label(label,row,{'action_mask':np.ones(2,np.float32)},'1'*64)
    assert changed['quality_score']==original['quality_score']
    label['L1']['score']=float('nan')
    with pytest.raises(ValueError):project_label(label,row,{'action_mask':np.ones(2,np.float32)},'1'*64)


@pytest.mark.parametrize('reason',['FORCED_WAIT_POSITION','FUNDS_OUTSIDE_ORIGINAL_CERTIFICATE',
                                  'ADD_RIGHT_CENSORED','NO_ACTIVE_DECISION'])
def test_B13_invalid_strict_reasons_never_become_valid_wait(corpus,reason):
    row=next(r for r in iter_sidecar(corpus/'policies/P0') if not r['reset'] and r['teacher']['L1']['score_valid'])
    label=copy.deepcopy(row['teacher']);strict=label['L2'];strict.pop('content_hash')
    strict.update(teacher_valid=False,strict_action=None,tie=None,action_value_add=None,
                  action_value_wait=None,margin=None,reason=reason)
    label['L2']=signed(strict);label.pop('content_hash');label=signed(label)
    out=project_label(label,row,{'action_mask':np.ones(2,np.float32)},'1'*64)
    assert not out['teacher_valid'] and out['quality_score_valid']


def test_B14_real_native_quality_suffix_tie_certifies_wait_only_with_two_actions(tmp_path):
    import pandas as pd
    from wp02_fixture import native_factory
    from gym_trading_env.research.fg_s3_e_quality_strict import QualityStrictTeacher
    from gym_trading_env.research.fg_s3_quality_teacher_v2 import NativeQualityTeacher
    from gym_trading_env.research.fg_s3_e_quality_corpus import l1_labeler
    factory,_,profile=native_factory(tmp_path/'market',minutes=8)
    market=tmp_path/'market/market.csv';data=pd.read_csv(market)
    for key in ('Open','High','Low','Close'):data[key]=1000.
    # Flat prices still incur the native MAIN_ROLL reward when ADD fills.
    # Insufficient next-bar volume makes ADD genuinely unfilled and identical
    # to WAIT, while both current published actions remain legal.
    data['Volume']=1.
    data.to_csv(market,index=False)
    profile.source_receipt['sources']['CZCE.FG2701']['market']['sha256']=digest(market)
    env=factory();source=dict(fold='TRAIN',calendar_complete=True)
    strict=QualityStrictTeacher(env,source=source);facts=NativeQualityTeacher(env,factory,fold='TRAIN',source=source)
    try:
        label=l1_labeler(facts,strict)(env,None,None)
        assert label['L2']['teacher_valid'] and label['L2']['tie'] and label['L2']['strict_action']==0
        row=dict(decision_executed=True,decision_valid_minute=True,reset=False,is_first=False,is_last=False,is_terminal=False)
        out=project_label(label,row,{'action_mask':env._published_mask},'1'*64)
        assert out['teacher_valid'] and out['teacher_action']==0
        assert not out['quality_outcome_valid'] and not out['quality_score_valid']
        out=project_label(label,row,{'action_mask':np.array([1,0],np.float32)},'1'*64)
        assert not out['teacher_valid']
        from gym_trading_env.research.fg_s3_e_quality_corpus import generate_behavior,behavior_specs
        trace,receipt=generate_behavior(factory,behavior_specs()[0],tmp_path/'tie_trace',
            __import__('pathlib').Path(__file__).resolve().parents[3]/'dreamerv3',source=source,strict=strict)
        assert receipt['status']=='PASS'
        assert any(r['teacher']['L2']['teacher_valid'] and r['teacher']['L2']['tie']
                   and not r['teacher']['L1']['immediate_outcome_verified']
                   for r in iter_sidecar(tmp_path/'tie_trace'))
    finally:facts.close();env.close()
