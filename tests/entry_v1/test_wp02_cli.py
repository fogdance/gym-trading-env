import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

from gym_trading_env.research.fg_replay import signed,write_json,read_manifest,digest,iter_sidecar
from gym_trading_env.research.fg_s3_e_quality_corpus_cli import source_inputs,generate_small,ARTIFACTS

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('role,eligible',[('test',True),('validation',True),('train','true'),('train',False)])
def test_E01_nontrain_rejected_before_price_or_env_receipt_open(tmp_path,monkeypatch,role,eligible):
    source=tmp_path/'pool.json'
    write_json(source,signed(dict(product='FG',files=[dict(contract='CZCE.FG201',role=role,
        eligible_for_training=eligible,complete_lifecycle=True,calendar_complete=True,
        path=str(tmp_path/'NEVER_OPEN_TEST.csv'))])))
    def forbidden(*a,**k):pytest.fail('opened held-out or unauthenticated prices')
    monkeypatch.setattr(__import__('pandas'),'read_csv',forbidden)
    with pytest.raises(ValueError,match='TRAIN'):
        source_inputs(source,tmp_path/'missing_env_receipt','CZCE.FG201')


def test_E02_fake_approval_and_invalid_output_write_nothing(tmp_path):
    pool=tmp_path/'pool.json';write_json(pool,signed(dict(product='FG',files=[],certified=True)))
    with pytest.raises(ValueError,match='forged'):source_inputs(pool,tmp_path,'CZCE.FG201')
    out=tmp_path/'old_run'
    with pytest.raises(ValueError,match='forged'):
        generate_small(pool,tmp_path,'CZCE.FG201',out)
    assert not out.exists()
    with pytest.raises(ValueError,match='traversal'):
        generate_small(pool,tmp_path,'CZCE.FG201',ARTIFACTS/'../escape')


@pytest.mark.parametrize('mode,expected',[('draft',0),('formal',2),('',2)])
def test_E03_draft_is_only_diagnostic_real_cli(tmp_path,mode,expected):
    spec=json.loads((ROOT/'docs/fixtures/fg_s3_e_quality_course_v2.example.json').read_text());spec['spec_mode']=mode
    p=tmp_path/'course.json';p.write_text(json.dumps(spec))
    before=set(tmp_path.iterdir())
    run=subprocess.run([sys.executable,'-B',str(ROOT/'tools/fg_s3_e_quality_corpus.py'),'validate','--course',str(p)],
        capture_output=True,text=True,env={**os.environ,'JAX_PLATFORMS':'cpu','CUDA_VISIBLE_DEVICES':''})
    assert run.returncode==expected,run.stderr
    body=json.loads(run.stdout.splitlines()[-1]);assert body.get('training_admitted',False) is False
    if mode=='draft':assert body['status']=='CONTRACT_ONLY' and body['admission_status']=='NOT_ADMITTED'
    assert set(tmp_path.iterdir())==before


@pytest.mark.parametrize('parallel',[False,True])
def test_E04_resigned_teacher_forgery_is_rejected_by_native_facts(tmp_path,parallel):
    from wp02_fixture import native_factory
    from gym_trading_env.research.fg_s3_e_quality_corpus import generate_behavior,behavior_specs,l1_labeler
    from gym_trading_env.research.fg_s3_quality_teacher_v2 import NativeQualityTeacher
    from gym_trading_env.research.fg_replay import reconcile_trace,E_QUALITY_SCHEMA
    factory,_,_=native_factory(tmp_path/'market',minutes=8)
    source=dict(fold='TRAIN',state_cache_entries=0)
    out=tmp_path/'p0';generate_behavior(factory,behavior_specs()[0],out,ROOT.parent/'dreamerv3',source=source)
    rows=list(iter_sidecar(out))
    wrong=rows[0]['teacher'];facts={k:v for k,v in wrong['L1'].items() if k!='content_hash'}
    facts['score']=99.;wrong['L1']=signed(facts)
    rows[0]['teacher']=signed({k:v for k,v in wrong.items() if k!='content_hash'})
    h=hashlib.sha256()
    with gzip.open(out/'steps.jsonl.gz','wt') as stream:
        for row in rows:
            canonical={k:v for k,v in row.items() if k not in ('stepid','ordinal')}
            h.update(json.dumps(canonical,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
            stream.write(json.dumps(row)+'\n')
    m=read_manifest(out/'manifest.json');m.pop('content_hash');m['files']['steps.jsonl.gz']=digest(out/'steps.jsonl.gz');m['event_hash']=h.hexdigest()
    m['replay_trace_hash']=__import__('gym_trading_env.envs.entry_contract',fromlist=['fingerprint']).fingerprint({k:m[k] for k in ('identity','policy','env_config_hash','reward_version','array_hash','event_hash')})
    write_json(out/'manifest.json',signed(m))
    env=factory();env.reset(seed=0);teacher=NativeQualityTeacher(env,factory,fold='TRAIN',source=source)
    verifier=l1_labeler(teacher)
    if parallel:
        from gym_trading_env.research.fg_s3_quality_label_pool import NativeLabelPool
        verifier=NativeLabelPool(dict(env_config=str(tmp_path/'market/env.yaml')),env._e_profile,
            env.export_state(),source,None,teacher.identity_hash,workers=2)
    try:
        with pytest.raises(ValueError,match='posterior facts'):
            reconcile_trace(out,ROOT.parent/'dreamerv3',env_factory=factory,schema=E_QUALITY_SCHEMA,label_verifier=verifier)
    finally:
        if parallel:verifier.close()
        teacher.close();env.close()


def test_E05_self_signed_env_receipt_does_not_create_a_train_authority(tmp_path,monkeypatch):
    pool=tmp_path/'pool.json'
    write_json(pool,signed(dict(product='FG',source_training_hash='f'*64,files=[dict(
        contract='CZCE.FG201',role='train',eligible_for_training=True,complete_lifecycle=True,calendar_complete=True)])))
    receipt=tmp_path/'env';receipt.mkdir()
    write_json(receipt/'manifest.json',signed(dict(stage='E',status='PASS',clock_protocol='S3_MARKET_MINUTE')))
    def forbidden(*a,**k):pytest.fail('opened prices behind a forged certificate')
    monkeypatch.setattr(__import__('pandas'),'read_csv',forbidden)
    with pytest.raises(ValueError,match='trust root'):source_inputs(pool,receipt,'CZCE.FG201')


def test_E06_manifest_traversal_rejected_before_hashing_any_price(tmp_path,monkeypatch):
    from gym_trading_env.research import fg_s3_e_quality_corpus_cli as cli
    def forbidden(*a,**k):pytest.fail('hashed attacker-controlled outside file')
    monkeypatch.setattr(cli,'check_files',forbidden)
    with pytest.raises(ValueError,match='Unsafe'):
        cli._inventory(tmp_path,dict(files={'../NEVER_OPEN_TEST.csv':'a'*64},behaviors=[]))


def test_E07_student_and_posterior_publications_are_separate(tmp_path):
    from wp02_fixture import native_factory
    from gym_trading_env.research.fg_s3_e_quality_corpus import generate_behavior,behavior_specs
    from gym_trading_env.research.fg_s3_e_quality_corpus_cli import _publish_streams,_verify_publications
    factory,_,_=native_factory(tmp_path/'market',minutes=8)
    generate_behavior(factory,behavior_specs()[0],tmp_path/'policies/P0',ROOT.parent/'dreamerv3',source=dict(fold='TRAIN'))
    _publish_streams(tmp_path,['P0']);_verify_publications(tmp_path,['P0'])
    m=read_manifest(tmp_path/'student_replay/P0/manifest.json')
    assert not (tmp_path/'student_replay/P0/steps.jsonl.gz').exists()
    assert (tmp_path/'teacher_sidecar/P0/steps.jsonl.gz').is_file()
    m.pop('content_hash');m['quality_score']=99
    write_json(tmp_path/'student_replay/P0/manifest.json',signed(m))
    with pytest.raises(ValueError,match='posterior metadata'):_verify_publications(tmp_path,['P0'])


def test_E08_corpus_catalog_keeps_fact_action_and_permissions_separate(tmp_path):
    from wp02_fixture import native_factory
    from gym_trading_env.research.fg_s3_e_quality_strict import QualityStrictTeacher
    from gym_trading_env.research.fg_s3_e_quality_corpus import l1_labeler
    from gym_trading_env.research.fg_s3_quality_teacher_v2 import NativeQualityTeacher
    from gym_trading_env.research.fg_s3_quality_catalog_v2 import validate_corpus_label,validate_label
    factory,_,_=native_factory(tmp_path/'market',minutes=20)
    env=factory();strict=QualityStrictTeacher(env,source=dict(fold='TRAIN'))
    facts=NativeQualityTeacher(env,factory,fold='TRAIN',source={})
    try:
        label=l1_labeler(facts,strict)(env,None,None)
        assert validate_corpus_label(label)==label and label['L2']['teacher_valid']
        with pytest.raises(ValueError,match='protocol'):validate_label(label)
        bad={**label,'decision_admitted':True};bad=signed({k:v for k,v in bad.items() if k!='content_hash'})
        with pytest.raises(ValueError,match='admission'):validate_corpus_label(bad)
        l2={**label['L2'],'teacher_valid':False,'strict_action':0}
        bad=signed({**{k:v for k,v in label.items() if k!='content_hash'},'L2':l2})
        with pytest.raises(ValueError,match='active WAIT'):validate_corpus_label(bad)
    finally:facts.close();env.close()


def test_E09_nested_replay_manifest_cannot_read_outside_market_files(tmp_path,monkeypatch):
    from gym_trading_env.research import fg_replay
    from gym_trading_env.research.fg_s3_e_quality_corpus_cli import _inventory
    write_json(tmp_path/'manifest.json',signed(dict(schema=fg_replay.E_QUALITY_SCHEMA,
        reward_version='FG_SINGLE3_REWARD_MARKET_MINUTE_TP_QUALITY_V1',
        files={'../NEVER_OPEN_TEST.csv':'a'*64},chunks=[])))
    def forbidden(*a,**k):pytest.fail('read an external file through a nested Replay manifest')
    monkeypatch.setattr(fg_replay,'check_files',forbidden)
    with pytest.raises(ValueError,match='Unsafe E Replay'):
        fg_replay.trace_manifest(tmp_path,schema=fg_replay.E_QUALITY_SCHEMA)
    link=tmp_path/'source.csv';link.symlink_to(tmp_path.parent/'NEVER_OPEN_TEST.csv')
    with pytest.raises(ValueError,match='symlink'):
        _inventory(tmp_path,dict(files={'source.csv':'a'*64},behaviors=[]))


@pytest.mark.parametrize('attack',['ledger_rewrite','different_account'])
def test_E10_resigned_account_tampering_fails_native_replay(tmp_path,attack):
    from decimal import Decimal
    from wp02_fixture import native_factory
    from gym_trading_env.research.fg_s3_e_quality_corpus import generate_behavior,behavior_specs
    from gym_trading_env.research.fg_replay import reconcile_trace,E_QUALITY_SCHEMA
    factory,_,_=native_factory(tmp_path/'market',minutes=8)
    out=tmp_path/'P0'
    generate_behavior(factory,behavior_specs()[0],out,ROOT.parent/'dreamerv3',source=dict(fold='TRAIN'))
    state=json.loads((out/'account_state.json').read_text())
    if attack=='ledger_rewrite':
        key=next(iter(state['balances']));state['balances'][key]=str(Decimal(state['balances'][key])+1)
    else:
        other=tmp_path/'P1'
        generate_behavior(factory,behavior_specs()[1],other,ROOT.parent/'dreamerv3',source=dict(fold='TRAIN'))
        state=json.loads((other/'account_state.json').read_text())
    write_json(out/'account_state.json',state)
    m=read_manifest(out/'manifest.json');m.pop('content_hash')
    m['files']['account_state.json']=digest(out/'account_state.json');write_json(out/'manifest.json',signed(m))
    with pytest.raises(ValueError,match='Final account/ledger'):
        reconcile_trace(out,ROOT.parent/'dreamerv3',env_factory=factory,schema=E_QUALITY_SCHEMA)
