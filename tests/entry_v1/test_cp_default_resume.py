"""Default complete restore uses Run ownership, not current source identity."""
import copy
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gym_trading_env.research.fg_replay import digest, signed, write_json


@pytest.mark.parametrize('stage', ['f1', 'f3'])
@pytest.mark.parametrize('attack', [None, 'source', 'index'])
def test_prepared_start_ignores_code_and_configuration_but_checks_bytes(tmp_path, monkeypatch, stage, attack):
    from gym_trading_env.research import fg_s3_e_quality_foundation as f1
    from gym_trading_env.research import fg_s3_e_quality_actor as f3
    from embodied.run import fg_s3_e_quality_foundation as f1run
    from embodied.run import fg_s3_e_quality_actor as f3run
    module, runner = (f1, f1run) if stage == 'f1' else (f3, f3run)
    directory = tmp_path/'prepared';directory.mkdir()
    index = directory/'join.sqlite';index.write_bytes(b'fixture-index')
    settings = dict(run_dir=str(module.PROJECT/'artifacts/fg_s3_v2'/tmp_path.name)) if stage == 'f1' else dict(
        run_dir=str(f1.PROJECT/'artifacts/fg_s3_v2'/tmp_path.name))
    identity = dict(code={'old': True}, source='original', source_model={'engineering_only': False}, model={'loss': 1})
    saved = dict(status=stage.upper()+'_DATA_READY', identity=identity,
                 verifier_binding={'pin': 'original'}, compatibility={'corpus': 'original'})
    current = copy.deepcopy(saved)
    current['identity'].update(code={'new': True}, model={'loss': 2})
    source_identity = 'changed' if attack == 'source' else 'original'
    preparation = SimpleNamespace(settings=settings, receipt=current,
        source=SimpleNamespace(receipt={'content_identity': source_identity}))
    write_json(directory/'settings.json', signed(dict(settings=settings)))
    write_json(directory/(stage+'_preparation.json'), signed(saved))
    write_json(directory/'reader_binding.json', signed(dict(index_sha256=digest(index),source_identity='original')))
    if attack == 'index':index.write_bytes(b'changed')
    events = []
    monkeypatch.setattr(module, 'validate_settings', lambda value: value)
    monkeypatch.setattr(module, 'require_prepared_verifier', lambda *a, **k: events.append('verifier'))
    monkeypatch.setattr(module, 'preflight', lambda *a, **k: preparation)
    monkeypatch.setattr(module, 'execution_identity', lambda: {'new': True})
    if stage == 'f3':
        monkeypatch.setattr(module, 'require_training_authority', lambda *a, **k: events.append('authority'))

    class Run:
        def __init__(self, *args, **kwargs):events.append('run')
        def run(self, stop):return 'started'

    monkeypatch.setattr(runner, 'FoundationRun' if stage == 'f1' else 'ActorRun', Run)
    before = {str(p):p.read_bytes() for p in directory.iterdir()}
    if attack:
        with pytest.raises(ValueError, match='INDEX_CHANGED|SOURCE'):
            module.start_prepared(directory, authorized=True)
        assert 'run' not in events
    else:
        assert module.start_prepared(directory, authorized=True) == 'started'
        assert 'run' in events
    assert 'verifier' in events
    assert before == {str(p):p.read_bytes() for p in directory.iterdir()}


def worker(settings,index,root,stage,mode,stop,*,changed=False,changed_settings=False,success=True):
    command=[sys.executable,'-B',str(Path(__file__).with_name('cp_default_resume_worker.py')),
             stage,str(settings),str(index),str(root),mode,str(stop)]
    if changed:command.append('--changed-code')
    if changed_settings:command.append('--changed-settings')
    result=subprocess.run(command,capture_output=True,text=True,timeout=300)
    log=root.parent/(root.name+'_'+mode+'_'+str(stop)+('_changed' if changed else '')+'.log')
    log.write_text(result.stdout+result.stderr)
    log.with_suffix('.command.json').write_text(json.dumps(dict(argv=command,exit_code=result.returncode)))
    if success:
        assert result.returncode==0,result.stdout[-1500:]+result.stderr[-4000:]
        return pickle.loads((root/'restore_evidence.pkl').read_bytes())
    assert result.returncode!=0
    return result.stdout+result.stderr


def immutable_files(root):
    return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()
            and (p.name=='run_identity.json' or p.is_relative_to(root/'ckpt') and p.name!='latest')}


@pytest.fixture(scope='module')
def cold_cases(tmp_path_factory):
    from wp03_fixture import native_corpus
    from wp04_fixture import native_reader
    from test_wp06_contract import settings as f3_settings
    from gym_trading_env.research.fg_s3_e_quality_supervision import QualitySupervisionAdmission,SupervisionSource
    from gym_trading_env.research.fg_s3_e_quality_reader import build_join_index
    from test_wp04_checkpoint import exact
    root=tmp_path_factory.mktemp('default_resume')
    corpus=native_corpus(root/'corpus',minutes=100,chunk=120)
    s,reader=native_reader(root,corpus=corpus,target=8,burnin=4);reader.close()
    s.update(optimizer_updates=2,save_every_updates=1)
    f1_settings=root/'f1.json';f1_settings.write_text(json.dumps(s))
    cases={}
    for stage in ('f1','f3'):
        if stage=='f1':path,index=f1_settings,root/'join.sqlite'
        else:
            settings=f3_settings(corpus)
            settings.update(optimizer_updates=2,save_every_updates=1,
                source_run=str(cases['f1']['split']),
                source_checkpoint=str(cases['f1']['split']/'ckpt/step_000000000002'))
            path=root/'f3.json';path.write_text(json.dumps(settings))
            index=root/'f3_join.sqlite'
            source=SupervisionSource.from_corpus(corpus,stage='ACTOR_SUPERVISED')
            build_join_index(QualitySupervisionAdmission.preflight(source),index)
        continuous=root/(stage+'_continuous');split=root/(stage+'_split')
        expected=worker(path,index,continuous,stage,'start',2)
        worker(path,index,split,stage,'start',1)
        original=immutable_files(split)
        restored=worker(path,index,split,stage,'resume',2,changed=True)
        exact(expected,restored)
        assert (continuous/'updates.jsonl').read_bytes()==(split/'updates.jsonl').read_bytes()
        assert all(immutable_files(split)[name]==sha for name,sha in original.items())
        # A second cold load must accept the newly saved CP with its original owner.
        again=worker(path,index,split,stage,'resume',2,changed=True,changed_settings=True)
        exact(restored,again)
        identity=json.loads((split/'run_identity.json').read_text())['identity']
        assert identity['code']['gym']['head']!='0'*40
        assert all(json.loads(p.read_text())['identity']==identity for p in (split/'ckpt').glob('*/manifest.json'))
        cases[stage]=dict(settings=path,index=index,split=split,identity=identity)
    return root,cases


def test_f1_f3_cold_restore_keeps_all_state_and_original_checkpoint_owner(cold_cases):
    _,cases=cold_cases
    assert set(cases)=={'f1','f3'}


@pytest.mark.parametrize('stage,attack',[('f1','model_dtype'),('f3','optimizer_member')])
def test_actual_state_layout_failure_does_not_publish_checkpoint(cold_cases,tmp_path,stage,attack):
    _,cases=cold_cases;case=cases[stage];root=tmp_path/'run'
    shutil.copytree(case['split'],root)
    node=root/'ckpt/step_000000000002';path=node/'agent.pkl'
    data=pickle.loads(path.read_bytes())
    if attack=='model_dtype':
        key=next(k for k in data['params'] if not k.startswith('opt/'))
        data['params'][key]=data['params'][key].astype(np.float64)
    else:
        key=next(k for k in data['params'] if k.startswith('opt/state/'))
        data['params'].pop(key)
    path.write_bytes(pickle.dumps(data,protocol=5))
    manifest_path=node/'manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest['files']['agent.pkl']=digest(path);manifest_path.write_text(json.dumps(manifest))
    before=immutable_files(root)
    error=worker(case['settings'],case['index'],root,stage,'resume',2,changed=True,success=False)
    assert 'OFFLINE_RESTORE_PARAMETER_' in error
    assert immutable_files(root)==before


@pytest.mark.parametrize('stage,attack',[('f1','corrupt'),('f3','missing'),('f3','latest')])
def test_strict_latest_still_rejects_damage(cold_cases,tmp_path,stage,attack):
    import elements
    from embodied.run.fg_checkpoint import FullCheckpoint
    _,cases=cold_cases;case=cases[stage];root=tmp_path/'run'
    shutil.copytree(case['split'],root);node=root/'ckpt/step_000000000002'
    if attack=='corrupt':(node/'agent.pkl').write_bytes(b'corrupt')
    elif attack=='missing':(node/'counters.pkl').unlink()
    else:(root/'ckpt/latest').write_text('step_000000000001')
    before=immutable_files(root)
    cp=FullCheckpoint(root/'ckpt',elements.Counter(),case['identity'],(),
        backend='external_immutable_course_reader',strict_latest=True)
    with pytest.raises((ValueError,OSError)):cp.latest()
    assert immutable_files(root)==before


def test_source_and_index_byte_checks_still_reject(cold_cases,tmp_path):
    from gym_trading_env.research.fg_s3_e_quality_foundation import preflight
    from embodied.run.fg_s3_e_quality_foundation import FoundationRun,synthetic_probe_permit
    _,cases=cold_cases;case=cases['f1'];settings=json.loads(case['settings'].read_text())
    preparation=preflight(settings)
    index=tmp_path/'join.sqlite';shutil.copyfile(case['index'],index)
    shutil.copyfile(case['index'].with_suffix('.json'),index.with_suffix('.json'))
    with index.open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='INDEX_CHANGED'):
        FoundationRun(preparation,index,case['split'],resume=True,_probe=synthetic_probe_permit(preparation))
    corpus=tmp_path/'corpus';shutil.copytree(settings['corpus_root'],corpus)
    member=next((corpus/'student_replay').rglob('*.npz'))
    with member.open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='CONTENT_CHANGED'):
        preflight({**settings,'corpus_root':str(corpus)})
