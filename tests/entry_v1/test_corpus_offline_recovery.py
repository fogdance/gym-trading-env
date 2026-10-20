"""Actual offline Run assembly with a non-neural Agent; no optimizer updates."""
import importlib
import json
import hashlib
import pickle
import shutil
from pathlib import Path

import embodied
import pytest

from gym_trading_env.research.fg_s3_e_quality_supervision import QualitySupervisionAdmission, SupervisionSource
from gym_trading_env.research.fg_s3_e_quality_reader import build_join_index
from embodied.run.fg_checkpoint import validate_bundle
from gym_trading_env.research.fg_replay import read_manifest, signed, write_json


class Agent:
    params={}
    def __init__(self):self.n_updates=0
    def stream(self,source):return embodied.streams.Prefetch(source)
    def save_exact(self):return self.n_updates
    def load_exact(self,state):self.n_updates=state
    save=save_exact
    load=load_exact


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
    from wp04_fixture import multiscale_reader
    settings,reader=multiscale_reader(tmp_path_factory.mktemp('strict_course'),target=8,burnin=4)
    reader.close();return Path(settings['corpus_root'])


@pytest.mark.parametrize('stage,name,class_name',[
    ('f1','foundation','FoundationRun'),('f2','heads','HeadsRun'),('f3','actor','ActorRun')])
def test_all_offline_runtime_and_probe_use_strict_latest(corpus,tmp_path,monkeypatch,stage,name,class_name):
    from wp04_fixture import settings
    s=settings(corpus);s.update(run_dir=str(tmp_path/'run'),prepared_dir=str(tmp_path/'prepared'))
    source=QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(corpus))
    proof=dict(identity=dict(preparation=dict(course=source.source.course_identity_hash,
        source=source.receipt['content_identity'])),engineering_only=True)
    owner=importlib.import_module('gym_trading_env.research.fg_s3_e_quality_'+name)
    runtime=importlib.import_module('embodied.run.fg_s3_e_quality_'+name)
    transfer=importlib.import_module('embodied.run.fg_s3_e_quality_transfer')
    if stage=='f2':
        s.update(schema='FG_S3_E_QUALITY_F2_CONFIG_V1',stage='QUALITY_HEADS',sampling_stage='HEADS_VALID',
            source_f1_run=str(tmp_path/'f1'),source_f1_checkpoint=str(tmp_path/'f1/ckpt/step_000000000004'))
        monkeypatch.setattr(transfer,'inspect_f1',lambda *a,**k:(proof,None))
    elif stage=='f3':
        from test_wp06_contract import settings as actor_settings
        s=actor_settings(corpus);s.update(run_dir=str(tmp_path/'run'),prepared_dir=str(tmp_path/'prepared'))
        monkeypatch.setattr(transfer,'inspect_stage',lambda *a,**k:(proof,None))
    monkeypatch.setattr(transfer,'transfer_parameters',lambda *a,**k:dict(content_hash='STUB_TRANSFER_ONLY'))
    if stage=='f2':
        # F2 binds these helpers at module import; patch its lookup sites too.
        monkeypatch.setattr(runtime,'inspect_f1',transfer.inspect_f1)
        monkeypatch.setattr(runtime,'transfer_parameters',transfer.transfer_parameters)
    monkeypatch.setattr(owner,'create_agent',lambda *a,**k:Agent())
    if stage!='f3':monkeypatch.setattr(runtime,'create_agent',lambda *a,**k:Agent())
    prep=owner.preflight(s);index=tmp_path/'join.sqlite';build_join_index(prep.source,index)
    permit=(runtime.synthetic_probe_permit(prep) if stage=='f1' else runtime.engineering_permit(prep))
    cls=getattr(runtime,class_name);run=cls(prep,index,tmp_path/'run',_probe=permit)
    assert run.identity['recovery_policy']=='STRICT_PUBLISHED_LATEST_V2'
    assert validate_bundle(run.cp.latest())['strict_latest'] is True
    run.step.increment();run.agent.n_updates=1
    run.counters.state.update(consumed_batches=1,optimizer_updates=1)
    row=dict(local_update=1,agent_update=1)
    if stage!='f1':row['consumed_batches']=1
    (run.root/'updates.jsonl').write_text(json.dumps(row)+'\n');run.cp.save()
    cold=tmp_path/'cold';shutil.copytree(run.root,cold)
    with (cold/'updates.jsonl').open('ab') as f:f.write(b'uncommitted crash tail\n')
    restored=cls(prep,index,cold,resume=True,_probe=permit)
    assert int(restored.step)==restored.agent.n_updates==1
    assert (cold/'updates.jsonl').read_bytes()==(run.root/'updates.jsonl').read_bytes()
    assert any(p.read_bytes().endswith(b'uncommitted crash tail\n') for p in (cold/'recovery_logs').iterdir())
    restored.source.close()
    def forbidden(*a,**k):raise AssertionError('invalid latest constructed an Agent')
    monkeypatch.setattr(owner,'create_agent',forbidden)
    if stage!='f3':monkeypatch.setattr(runtime,'create_agent',forbidden)
    try:
        for attack in ('missing_done','backend','strict_flag','log_prefix','log_position','legacy_identity'):
            clone=tmp_path/attack;shutil.copytree(run.root,clone)
            node=clone/'ckpt/step_000000000001';meta_path=node/'manifest.json'
            if attack=='missing_done':(node/'done').unlink()
            elif attack in ('backend','strict_flag'):
                meta=json.loads(meta_path.read_text())
                if attack=='backend':meta['backend']='native_replay'
                else:meta['strict_latest']=False
                meta_path.write_text(json.dumps(meta))
            elif attack=='log_prefix':(clone/'updates.jsonl').write_bytes(b'CHANGED\n')
            elif attack=='log_position':
                state=pickle.loads((node/'logs.pkl').read_bytes())
                state['updates.jsonl']=dict(bytes=0,lines=0,sha256=hashlib.sha256(b'').hexdigest())
                raw=pickle.dumps(state);(node/'logs.pkl').write_bytes(raw)
                meta=json.loads(meta_path.read_text());meta['files']['logs.pkl']=hashlib.sha256(raw).hexdigest()
                meta_path.write_text(json.dumps(meta))
            else:
                path=clone/'run_identity.json';obj=read_manifest(path);obj.pop('content_hash')
                obj['identity']['recovery_policy']='LEGACY_UNREVIEWED';write_json(path,signed(obj))
            before={str(p):p.read_bytes() for p in clone.rglob('*') if p.is_file()}
            with pytest.raises(ValueError):cls(prep,index,clone,resume=True,_probe=permit)
            assert before=={str(p):p.read_bytes() for p in clone.rglob('*') if p.is_file()}
    finally:run.source.close()
