"""Synthetic API wiring evidence only; stubs never count as real native G3."""
import importlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from gym_trading_env.research.fg_replay import signed, write_json, read_manifest, digest
from gym_trading_env.research import fg_s3_e_quality_supervision as supervision
from wp03_fixture import ROOT
from wp04_fixture import settings as f1_settings


@pytest.fixture
def frozen_pair(tmp_path):
    parent=tmp_path/'frozen';heads={}
    for role,folder,runtime in [('gym','gym-trading-env','src/gym_trading_env/native.py'),
                                ('dreamer','dreamerv3','dreamerv3/native.py')]:
        root=parent/folder;root.mkdir(parents=True)
        p=root/runtime;p.parent.mkdir(parents=True);p.write_text('NATIVE_TEST_ONLY = True\n')
        def git(*args):return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.PIPE).strip()
        git('init','-q');git('add','.')
        git('-c','user.name=Corpus test','-c','user.email=corpus-test@invalid','-c','commit.gpgsign=false','commit','-qm','synthetic frozen code')
        heads[role]=dict(sha=git('rev-parse','HEAD'),dirty=False)
    pin=tmp_path/'pin.json';write_json(pin,signed(dict(protocol='FG_S3_E_QUALITY_PRODUCER_PIN_V1',repositories=heads)))
    return parent/'gym-trading-env',pin,digest(pin)


@pytest.mark.parametrize('attack',['digest','gym_head','dreamer_head','dirty','untracked'])
def test_frozen_pair_audit_rejects_wrong_or_changed_code(frozen_pair,attack):
    gym,pin,sha=frozen_pair
    if attack=='digest':sha='f'*64
    elif attack in ('gym_head','dreamer_head'):
        obj=read_manifest(pin);obj.pop('content_hash')
        obj['repositories']['gym' if attack=='gym_head' else 'dreamer']['sha']='f'*40
        write_json(pin,signed(obj));sha=digest(pin)
    elif attack=='dirty':(gym/'src/gym_trading_env/native.py').write_text('CHANGED = True\n')
    else:(gym/'untracked_runtime.py').write_text('CHANGED = True\n')
    with pytest.raises(ValueError,match='PIN_SHA256_MISMATCH|PRODUCER_MISMATCH|FROZEN_PRODUCER_CODE_CHANGED'):
        supervision.frozen_producer_audit(gym,producer_lock=pin,producer_lock_sha256=sha)


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
    from wp04_fixture import multiscale_reader
    settings,reader=multiscale_reader(tmp_path_factory.mktemp('pin_corpus'),target=8,burnin=4)
    reader.close()
    return Path(settings['corpus_root'])


@pytest.mark.parametrize('stage,name,run_module,run_class',[
    ('f1','foundation','fg_s3_e_quality_foundation','FoundationRun'),
    ('f2','heads','fg_s3_e_quality_heads','HeadsRun'),
    ('f3','actor','fg_s3_e_quality_actor','ActorRun')])
def test_prepare_start_resume_bind_same_pin_and_reexecute_verifier(
        tmp_path,monkeypatch,frozen_pair,corpus,stage,name,run_module,run_class):
    owner=importlib.import_module('gym_trading_env.research.fg_s3_e_quality_'+name)
    runtime=importlib.import_module('embodied.run.'+run_module)
    s=f1_settings(corpus);s.update(target_steps=8,burnin_steps=4,prepared_dir=str(tmp_path/'prepared'),
        run_dir=str(ROOT/'artifacts/fg_s3_v2/semver_binding'/tmp_path.name))
    a=supervision.QualitySupervisionAdmission.preflight(supervision.SupervisionSource.from_corpus(corpus))
    proof=dict(identity=dict(preparation=dict(course=a.source.course_identity_hash,source=a.receipt['content_identity'])),engineering_only=False)
    transfer=importlib.import_module('embodied.run.fg_s3_e_quality_transfer')
    if stage=='f2':
        s.update(schema='FG_S3_E_QUALITY_F2_CONFIG_V1',stage='QUALITY_HEADS',sampling_stage='HEADS_VALID',
            source_f1_run=str(tmp_path/'source'),source_f1_checkpoint=str(tmp_path/'source/ckpt/step_000000000004'))
        monkeypatch.setattr(transfer,'inspect_f1',lambda *a,**k:(proof,None))
    elif stage=='f3':
        from test_wp06_contract import settings
        s=settings(corpus);s.update(target_steps=8,burnin_steps=4,prepared_dir=str(tmp_path/'prepared'),
            run_dir=str(ROOT/'artifacts/fg_s3_v2/semver_binding'/tmp_path.name))
        monkeypatch.setattr(transfer,'inspect_stage',lambda *a,**k:(proof,None))
    config=tmp_path/'config.json';config.write_text(json.dumps(s))
    gym,pin,sha=frozen_pair;calls=[]
    def native(source,*,producer_gym,log_path,producer_lock,producer_lock_sha256,**unused):
        audit=supervision.frozen_producer_audit(producer_gym,producer_lock=producer_lock,producer_lock_sha256=producer_lock_sha256)
        calls.append((str(producer_lock),producer_lock_sha256))
        Path(log_path).parent.mkdir(parents=True,exist_ok=True)
        Path(log_path).write_text('STUB_ROUTING_ONLY_NOT_CERTIFIED\n')
        return dict(producer_audit=audit,log_sha256=digest(Path(log_path)),result=dict(G3='GO'),
                    scope='SYNTHETIC_ROUTING_STUB_NOT_REAL_G3')
    monkeypatch.setattr(supervision,'native_verification',native)
    kwargs=dict(producer_gym=gym,producer_lock=pin,producer_lock_sha256=sha)
    receipt=owner.prepare(config,purpose='native-verification',**kwargs)
    saved=read_manifest(Path(s['prepared_dir'])/(stage+'_preparation.json'))
    assert saved['verifier_binding']==receipt['verifier_binding']
    assert saved['verifier_binding']['operator_producer_pin_sha256']==sha
    assert saved['compatibility']['course_abi_version']=='3.1.0'
    assert read_manifest(Path(s['prepared_dir'])/'reader_binding.json')['source_identity']==a.receipt['content_identity']
    before={str(p):p.read_bytes() for p in Path(s['prepared_dir']).rglob('*') if p.is_file()}
    constructed=[]
    class RoutingRun:
        def __init__(self,prep,index,root,**flags):
            assert prep.receipt['verifier_binding']==saved['verifier_binding']
            assert prep.source.native_result['scope']=='SYNTHETIC_ROUTING_STUB_NOT_REAL_G3'
            assert not Path(root).exists()
            constructed.append(flags)
        def run(self,stop):return dict(status='STUB_ROUTING_ONLY_NOT_CERTIFIED')
    monkeypatch.setattr(runtime,run_class,RoutingRun)
    for resume in (False,True):
        assert owner.start_prepared(s['prepared_dir'],authorized=True,resume=resume,**kwargs)['status']=='STUB_ROUTING_ONLY_NOT_CERTIFIED'
    assert [c['resume'] for c in constructed]==[False,True]
    assert calls==[(str(pin),sha)]*3
    # Even another valid pin for the same pair cannot replace this Prepared pin.
    alternate=tmp_path/'alternate.json';obj=read_manifest(pin);obj.pop('content_hash');obj['review']='different external review'
    write_json(alternate,signed(obj))
    with pytest.raises(ValueError,match='PREPARED_VERIFIER_CHANGED'):
        owner.start_prepared(s['prepared_dir'],authorized=True,producer_gym=gym,
            producer_lock=alternate,producer_lock_sha256=digest(alternate))
    with pytest.raises(ValueError,match='AUTHORIZATION_REQUIRED'):
        owner.start_prepared(s['prepared_dir'],**kwargs)
    assert len(calls)==3 and len(constructed)==2
    assert before=={str(p):p.read_bytes() for p in Path(s['prepared_dir']).rglob('*') if p.is_file()}
    # Resigning a changed ABI claim cannot replace the actual compatibility.
    changed=tmp_path/'changed_prepared';shutil.copytree(s['prepared_dir'],changed)
    path=changed/(stage+'_preparation.json');obj=read_manifest(path);obj.pop('content_hash')
    obj['compatibility']['course_abi_version']='3.9.0';write_json(path,signed(obj))
    with pytest.raises(ValueError,match='PREPARED_SOURCE_OR_ABI_CHANGED'):
        owner.start_prepared(changed,authorized=True,**kwargs)
    assert len(calls)==4 and len(constructed)==2
