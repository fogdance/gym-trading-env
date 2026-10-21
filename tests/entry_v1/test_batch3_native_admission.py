"""Real F4 pin/CLI/audit contracts. Native subprocesses are not production G3."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gym_trading_env.research import fg_s3_e_quality_online as online
from gym_trading_env.research import fg_s3_e_quality_online_admission as admission
from gym_trading_env.research.fg_replay import digest, signed, write_json

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def verifier(tmp_path):
    repositories = {}
    for role, name in (('gym','gym-trading-env'), ('dreamer','dreamerv3')):
        root = tmp_path/'verifier'/name
        root.mkdir(parents=True)
        subprocess.run(['git','init','-q',str(root)],check=True)
        folder = root/('src/gym_trading_env' if role == 'gym' else 'dreamerv3')
        folder.mkdir(parents=True)
        (folder/'runtime.py').write_text('VALUE = 1\n')
        subprocess.run(['git','-C',str(root),'add','.'],check=True)
        subprocess.run(['git','-C',str(root),'-c','user.name=CPU Fixture',
            '-c','user.email=fixture@invalid','commit','-qm','bounded verifier fixture'],check=True)
        repositories[role] = dict(sha=subprocess.check_output(
            ['git','-C',str(root),'rev-parse','HEAD'],text=True).strip())
    pin = tmp_path/'reviewed_pin.json'
    write_json(pin,signed(dict(protocol='FG_S3_E_QUALITY_PRODUCER_PIN_V1',repositories=repositories)))
    return dict(producer_gym=str(tmp_path/'verifier/gym-trading-env'),
        producer_lock=str(pin),producer_lock_sha256=digest(pin))


def settings(tmp_path):
    config = dict(schema='FG_S3_E_MULTICORPUS_CONFIG_V1',stage='ACTOR_STRICT',
        sources=[dict(corpus_id=name,corpus_root=str(tmp_path/name),index_path=str(tmp_path/(name+'.sqlite')))
            for name in ('beta','alpha')],per_source_sampling_spec={},source_mix_weights={},seed=202)
    path=tmp_path/'history.json';path.write_text(json.dumps(config))
    return dict(history_config=str(path),prepared_dir=str(tmp_path/'prepared'),execution=dict(run_dir=str(tmp_path/'run')))


@pytest.mark.parametrize('action',['prepare','start','resume'])
def test_online_cli_forwards_all_pin_arguments(tmp_path,monkeypatch,action):
    monkeypatch.delenv('LD_LIBRARY_PATH',raising=False)
    spec=importlib.util.spec_from_file_location('batch3_cli',ROOT/'tools/fg_s3_v2_experiment.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    calls=[]
    name='prepare' if action=='prepare' else 'start_prepared'
    monkeypatch.setattr(online,name,lambda *args,**kwargs: calls.append((args,kwargs)) or dict(status='STUB_ROUTING_ONLY'))
    argv=['cli','online-'+action,'--producer-gym',str(tmp_path/'verifier'),
        '--producer-lock',str(tmp_path/'pin.json'),'--producer-lock-sha256','a'*64,
        '--config' if action=='prepare' else '--prepared',str(tmp_path/'input')]
    monkeypatch.setattr(sys,'argv',argv);cli.main()
    assert len(calls)==1
    kwargs=calls[0][1]
    assert kwargs['producer_gym']==str(tmp_path/'verifier')
    assert kwargs['producer_lock']==str(tmp_path/'pin.json') and kwargs['producer_lock_sha256']=='a'*64
    if action!='prepare':assert kwargs['resume']==(action=='resume')


def test_all_source_bindings_allow_shared_pin_and_distinct_pins(tmp_path,verifier):
    s=settings(tmp_path)
    shared=admission.resolve_verifiers(s,**verifier)
    assert list(shared)==['F3_ANCHOR','beta','alpha']
    assert len({json.dumps(v['verifier_binding'],sort_keys=True) for v in shared.values()})==1
    # JSON key sorting must not spuriously change the source order contract.
    saved=json.loads(json.dumps(shared,sort_keys=True))
    assert admission.resolve_verifiers(s,saved=saved,**verifier)==shared
    s['source_verifiers']={slot:dict(verifier) for slot in shared}
    assert admission.resolve_verifiers(s)==shared
    with pytest.raises(ValueError,match='AMBIGUOUS'):admission.resolve_verifiers(s,**verifier)
    s['source_verifiers'].pop('alpha')
    with pytest.raises(ValueError,match='COMPLETE_SOURCE_VERIFIER'):admission.resolve_verifiers(s)


def test_bad_verifier_fails_before_source_or_output(tmp_path,verifier):
    s=settings(tmp_path)
    for bad in ({}, {**verifier,'producer_gym':None},
        {**verifier,'producer_lock_sha256':'0'*64},
        {**verifier,'producer_lock':str(tmp_path/'absent.json')}):
        with pytest.raises((ValueError,FileNotFoundError)):admission.resolve_verifiers(s,**bad)
    shared=admission.resolve_verifiers(s,**verifier)
    forged=copy.deepcopy(shared);forged['alpha']['verifier_binding']['operator_producer_pin_sha256']='f'*64
    with pytest.raises(ValueError,match='PREPARED_VERIFIER_CHANGED'):
        admission.resolve_verifiers(s,**verifier,saved=forged)
    runtime=Path(verifier['producer_gym'])/'src/gym_trading_env/runtime.py'
    runtime.write_text('VALUE = 2\n')
    with pytest.raises(ValueError,match='FROZEN_PRODUCER_CODE_CHANGED'):
        admission.resolve_verifiers(s,**verifier)
    assert not Path(s['prepared_dir']).exists() and not Path(s['execution']['run_dir']).exists()


def test_second_source_wrong_pin_rejects_whole_registry(tmp_path,verifier):
    s=settings(tmp_path)
    s['source_verifiers']={slot:dict(verifier) for slot in admission.source_slots(s)}
    s['source_verifiers']['alpha']['producer_lock_sha256']='0'*64
    with pytest.raises(ValueError,match='PIN_SHA256_MISMATCH'):admission.resolve_verifiers(s)
    assert not (tmp_path/'audit_stage').exists() and not (tmp_path/'run').exists()


def test_unique_native_logs_all_sources_and_unforgeable_public_receipt(tmp_path,verifier):
    s=settings(tmp_path); verifiers=admission.resolve_verifiers(s,**verifier)
    bindings=[dict(source_id=slot,source_role='anchor' if slot==admission.ANCHOR else 'history',
        corpus_root=str(tmp_path/('corpus_'+slot)),corpus_identity=str(i)*64,
        verifier_binding=verifiers[slot]['verifier_binding']) for i,slot in enumerate(verifiers,1)]
    attempts=[admission.NativeAuditAttempt(tmp_path/'prepared','a'*64,tmp_path/'run',bindings,verifiers) for _ in range(2)]
    paths=[]
    for attempt in attempts:
        for binding in bindings:
            slot=binding['source_id'];args=attempt.arguments(slot);log=args['log_path'];paths.append(log)
            assert all(args[k]==verifier[k] for k in verifier)
            log.parent.mkdir(parents=True);log.write_text('STUB_NATIVE_ROUTING_ONLY_NOT_G3\n')
            audit=admission.frozen_producer_audit(**verifier)
            native=dict(exit_code=0,result=dict(G3='GO'),producer_audit=audit,log_sha256=digest(log))
            attempt.record(slot,SimpleNamespace(source=SimpleNamespace(corpus_content_hash=binding['corpus_identity']),native_result=native))
        evidence=attempt.finish()
        assert admission.require_native_audit(evidence,bindings)==evidence.report
        with pytest.raises(ValueError,match='ACTUAL_NATIVE_SOURCE_AUDIT'):
            admission.require_native_audit(admission.NativeSourceEvidence(copy.deepcopy(evidence.report)),bindings)
    assert len(set(paths))==6 and not (tmp_path/'run').exists()
    with pytest.raises(FileExistsError):
        with paths[0].open('x'):pass


def test_native_history_requires_audit_and_forwards_four_inputs(tmp_path,monkeypatch):
    from gym_trading_env.research import fg_s3_e_quality_multi_checks as checks
    from gym_trading_env.research.fg_s3_e_quality_supervision import QualitySupervisionAdmission,SupervisionSource
    s=settings(tmp_path)
    monkeypatch.setattr(checks,'published_sources',lambda config:[])
    monkeypatch.setattr(checks,'_sources',lambda config:[dict(corpus_id='alpha',admission=SimpleNamespace(
        source=SimpleNamespace(corpus_root=tmp_path/'alpha')))])
    monkeypatch.setattr(SupervisionSource,'from_corpus',lambda *args,**kwargs:'SOURCE')
    with pytest.raises(ValueError,match='NATIVE_AUDIT_REQUIRED'):online._history(s,{},native=True)
    class Reached(Exception):pass
    inputs=dict(producer_gym='GYM',producer_lock='PIN',producer_lock_sha256='a'*64,log_path='LOG')
    def receive(source,**kwargs):
        assert source=='SOURCE' and kwargs==dict(purpose='native-verification',**inputs)
        raise Reached
    monkeypatch.setattr(QualitySupervisionAdmission,'preflight',receive)
    with pytest.raises(Reached):online._history(s,{},native=True,audit=SimpleNamespace(arguments=lambda slot:inputs))


def test_audit_root_overlap_and_incomplete_source_set_reject(tmp_path,verifier):
    s=settings(tmp_path); v=admission.resolve_verifiers(s,**verifier)
    b=[dict(source_id=admission.ANCHOR,source_role='anchor',corpus_root=str(tmp_path/'data'))]
    with pytest.raises(ValueError,match='OVERLAPS'):
        admission.NativeAuditAttempt(tmp_path/'prepared','a'*64,tmp_path/'audit_stage/run',b,v)
    a=admission.NativeAuditAttempt(tmp_path/'prepared','a'*64,tmp_path/'run',b,v)
    with pytest.raises(ValueError,match='INCOMPLETE'):a.finish()
    failure=a.finish(ValueError('STUB_NATIVE_FAILED'))
    assert json.loads(Path(failure['path']).read_text())['status']=='FAIL'


def test_anchor_and_ordered_corpus_contents_are_frozen_separately(tmp_path,verifier):
    s=settings(tmp_path);v=admission.resolve_verifiers(s,**verifier)
    for i,slot in enumerate((admission.ANCHOR,'beta','alpha'),1):
        corpus=tmp_path/slot;corpus.mkdir()
        write_json(corpus/'corpus_manifest.json',signed(dict(course_identity_hash='a'*64,
            teacher_identity='b'*64,observation=dict(contract_hash='c'*64),source_lock_hash=str(i)*64)))
    proof=dict(identity=dict(preparation=dict(settings=dict(corpus_root=str(tmp_path/admission.ANCHOR)))))
    first=admission.source_bindings(s,proof,v)
    assert [b['source_id'] for b in first]==[admission.ANCHOR,'beta','alpha']
    assert len({b['corpus_identity'] for b in first})==3
    path=tmp_path/admission.ANCHOR/'corpus_manifest.json'
    body=json.loads(path.read_text());body.pop('content_hash');body['teacher_identity']='f'*64
    write_json(path,signed(body))
    changed=admission.source_bindings(s,proof,v)
    assert first[0]!=changed[0] and first[1:]==changed[1:]
    config=json.loads(Path(s['history_config']).read_text());config['sources'].reverse()
    Path(s['history_config']).write_text(json.dumps(config))
    assert admission.source_bindings(s,proof,v)[1:]==list(reversed(changed[1:]))
