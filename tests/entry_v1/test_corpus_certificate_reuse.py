"""Synthetic certificate/API tests, never real production certification."""
import importlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from gym_trading_env.research.fg_replay import digest, read_manifest, signed, trace_manifest, write_json, E_QUALITY_SCHEMA
from gym_trading_env.research.fg_s3_corpus_production import batch_signed
from gym_trading_env.research import fg_s3_e_quality_supervision as supervision
from wp03_fixture import native_corpus, inventory, ROOT
from wp04_fixture import settings as f1_settings


def seal(corpus, receipts):
    manifest=read_manifest(corpus/'corpus_manifest.json')
    reader=read_manifest(receipts/'reader/WP03_COURSE_READER_COVERAGE.json')
    files={'corpus/'+str(p.relative_to(corpus)):digest(p) for p in corpus.rglob('*') if p.is_file()}
    files.update({'receipts/'+str(p.relative_to(receipts)):digest(p) for p in receipts.rglob('*')
        if p.is_file() and p.name not in ('BATCH_MANIFEST.json','IDENTITY_LOCK.json')})
    fields=dict(batch_id='SYNTHETIC_ONLY',corpus_identity=manifest['content_hash'],
        source_lock=manifest['source_lock_hash'],course_identity=manifest['course_identity_hash'],
        teacher_identity=manifest['teacher_identity'],producer_pin_sha256='f'*64,
        native_verifier_pin_sha256=digest(receipts/'native/VERIFIER_PIN.json'),index_sha256=reader['index_sha256'])
    path=receipts/'BATCH_MANIFEST.json'
    write_json(path,batch_signed(dict(protocol='FG_S3_E_TRAIN_CORPUS_BATCH_RESULT_V1',
        status='NATIVE_G3_AND_INDEPENDENT_READER_COMPLETED',training_admitted=False,optimizer_updates=0,
        producer_repositories=manifest['code']['repositories'],files=files,**fields)))
    write_json(receipts/'IDENTITY_LOCK.json',signed(dict(protocol='FG_S3_E_TRAIN_CORPUS_IDENTITY_LOCK_V1',
        files=dict(files,**{'receipts/BATCH_MANIFEST.json':digest(path)}),**fields)))
    return dict(batch_manifest=str(path),sha256=digest(path))


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    from gym_trading_env.research.fg_s3_e_quality_reader import build_join_index
    root=tmp_path_factory.mktemp('synthetic_certificate')
    corpus=native_corpus(root/'corpus',minutes=20,chunk=25)
    manifest=read_manifest(corpus/'corpus_manifest.json')
    lock=read_manifest(corpus/'source_lock.json');lock.pop('content_hash')
    lock.update(source_pool=dict(path='SYNTHETIC_POOL'),env_receipt=dict(path='SYNTHETIC_ENV'),selection={})
    write_json(corpus/'source_lock.json',signed(lock))
    teacher=read_manifest(corpus/'teacher_manifest.json');teacher.pop('content_hash')
    teacher['source_lock_hash']=read_manifest(corpus/'source_lock.json')['content_hash']
    write_json(corpus/'teacher_manifest.json',signed(teacher))
    repositories={key:dict(sha=letter*40,dirty=False) for key,letter in [('gym','a'),('dreamer','b')]}
    manifest.update(source_lock_hash=teacher['source_lock_hash'],code=dict(repositories=repositories))
    inventory(corpus,manifest)
    admission=supervision.QualitySupervisionAdmission.preflight(supervision.SupervisionSource.from_corpus(corpus))
    receipts=root/'receipts'
    build_join_index(admission,receipts/'reader/join.sqlite')
    pin=receipts/'native/VERIFIER_PIN.json'
    write_json(pin,signed(dict(protocol='FG_S3_E_QUALITY_PRODUCER_PIN_V1',repositories=repositories)))
    log=receipts/'native/native_verify.log';log.write_text('SYNTHETIC_ROUTING_ONLY_NOT_REAL_G3\n')
    traces=[trace_manifest(corpus/'policies'/p,schema=E_QUALITY_SCHEMA) for p in supervision.POLICIES]
    audit={role:dict(head=row['sha'],runtime_diff_exit_code=0,untracked_runtime=[])
           for role,row in repositories.items()}
    audit['operator_producer_pin_sha256']=digest(pin)
    write_json(receipts/'native/native_verification.json',dict(exit_code=0,producer_audit=audit,
        log_sha256=digest(log),result=dict(status='PASS',G3='GO',verification_state='NATIVE_G3_VERIFIED',
        training_admitted=False,corpus_identity=admission.source.corpus_content_hash,
        native_replay=[dict(status='PASS',records=t['records'],teacher_labels_verified=t['records'],
                           replay_trace_hash=t['replay_trace_hash']) for t in traces])))
    write_json(receipts/'reader/WP03_COURSE_READER_COVERAGE.json',signed(dict(status='PASS_READER_ONLY',
        corpus_identity=admission.source.corpus_content_hash,source_identity=admission.receipt['content_identity'],
        index_sha256=digest(receipts/'reader/join.sqlite'),training_admitted=False,actual_optimizer_exposures=0)))
    return corpus,receipts,seal(corpus,receipts)


@pytest.fixture
def no_reexecution(monkeypatch,bundle):
    from gym_trading_env.research import fg_s3_corpus_provenance as provenance
    from gym_trading_env.research import fg_s3_e_quality_corpus_cli as cli
    from gym_trading_env.research import fg_s3_e_quality_reader as reader
    def forbidden(*a,**k):raise AssertionError('Teacher/G3/index rebuild must not execute')
    monkeypatch.setattr(supervision,'native_verification',forbidden)
    monkeypatch.setattr(reader,'build_join_index',forbidden)
    # Synthetic data has no real Git/TRAIN authority. The real FG809 check
    # uses both actual validators; these stubs prove API routing only.
    monkeypatch.setattr(provenance,'authenticate_code',lambda code:None)
    monkeypatch.setattr(cli,'source_inputs',lambda *a,**k:(read_manifest(bundle[0]/'source_lock.json'),None,None))


@pytest.mark.parametrize('stage,name,run_class',[
    ('f1','foundation','FoundationRun'),('f2','heads','HeadsRun'),('f3','actor','ActorRun')])
def test_prepare_start_resume_reuse_certificate_without_g3_or_index_rebuild(
        tmp_path,monkeypatch,bundle,no_reexecution,stage,name,run_class):
    corpus,receipts,reference=bundle
    owner=importlib.import_module('gym_trading_env.research.fg_s3_e_quality_'+name)
    runtime=importlib.import_module('embodied.run.fg_s3_e_quality_'+name)
    s=f1_settings(corpus);s.update(corpus_certification=reference,prepared_dir=str(tmp_path/'prepared'),
        run_dir=str(ROOT/'artifacts/fg_s3_v2/certificate_reuse'/tmp_path.name))
    admission=supervision.QualitySupervisionAdmission.preflight(supervision.SupervisionSource.from_corpus(corpus))
    proof=dict(identity=dict(preparation=dict(course=admission.source.course_identity_hash,
        source=admission.receipt['content_identity'])),engineering_only=False)
    transfer=importlib.import_module('embodied.run.fg_s3_e_quality_transfer')
    if stage=='f2':
        s.update(schema='FG_S3_E_QUALITY_F2_CONFIG_V1',stage='QUALITY_HEADS',sampling_stage='HEADS_VALID',
            source_f1_run=str(tmp_path/'source'),source_f1_checkpoint=str(tmp_path/'source/ckpt/step_000000000004'))
        monkeypatch.setattr(transfer,'inspect_f1',lambda *a,**k:(proof,None))
        monkeypatch.setattr(owner.F2WindowAdapter,'validate_support',lambda self:{'synthetic':True})
    elif stage=='f3':
        from test_wp06_contract import settings
        s.update({k:v for k,v in settings(corpus).items() if k not in ('run_dir','prepared_dir')})
        monkeypatch.setattr(transfer,'inspect_stage',lambda *a,**k:(proof,None))
        class Streams:
            roles=('natural',)
            streams=(SimpleNamespace(adapter=SimpleNamespace(validate_support=lambda:{'synthetic':True})),)
            def __init__(self,*a,**k):pass
            def close(self):pass
        monkeypatch.setattr(runtime,'ActorStreams',Streams)
    config=tmp_path/'config.json';config.write_text(json.dumps(s))
    result=owner.prepare(config,purpose='certified-corpus')
    assert result['status']==stage.upper()+'_DATA_READY'
    prepared=Path(s['prepared_dir'])
    assert digest(prepared/'join.sqlite')==digest(receipts/'reader/join.sqlite')
    assert digest(prepared/'join.json')==digest(receipts/'reader/join.json')
    before={p.name:p.read_bytes() for p in prepared.iterdir() if p.is_file()}
    calls=[]
    class Run:
        def __init__(self,p,index,root,**flags):
            owner.require_training_authority(p,authorized=flags['authorize_training'])
            assert p.source.native_result['verification_mode']=='PUBLISHED_NATIVE_G3_REUSED'
            assert p.source.native_result['fresh_native_execution'] is False
            calls.append(flags)
        def run(self,stop):return dict(status='SYNTHETIC_ROUTING_ONLY')
    monkeypatch.setattr(runtime,run_class,Run)
    for resume in (False,True):
        assert owner.start_prepared(prepared,authorized=True,resume=resume)['status']=='SYNTHETIC_ROUTING_ONLY'
    assert [c['resume'] for c in calls]==[False,True]
    assert before=={p.name:p.read_bytes() for p in prepared.iterdir() if p.is_file()}
    with pytest.raises(ValueError,match='AUTHORIZATION_REQUIRED'):owner.start_prepared(prepared)


@pytest.mark.parametrize('attack',['batch_digest','file','incomplete_g3'])
def test_changed_or_incomplete_production_certificate_is_rejected(tmp_path,bundle,no_reexecution,attack):
    from gym_trading_env.research.fg_s3_e_quality_foundation import preflight
    corpus,original,reference=bundle
    receipts=tmp_path/'receipts';shutil.copytree(original,receipts)
    reference=dict(batch_manifest=str(receipts/'BATCH_MANIFEST.json'),sha256=reference['sha256'])
    if attack=='batch_digest':reference['sha256']='0'*64
    elif attack=='file':(receipts/'native/native_verify.log').write_text('CHANGED\n')
    else:
        path=receipts/'native/native_verification.json';value=json.loads(path.read_text())
        value['result']['native_replay'].pop();write_json(path,value)
        reference=seal(corpus,receipts)
    with pytest.raises(ValueError,match='CERTIFICATION|INCOMPLETE_LABEL_VERIFICATION'):
        preflight({**f1_settings(corpus),'corpus_certification':reference},purpose='certified-corpus')
