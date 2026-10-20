"""Actual offline Run assembly with a non-neural Agent; no optimizer updates."""
import importlib
import json
import shutil
from pathlib import Path

import embodied
import pytest

from gym_trading_env.research.fg_s3_e_quality_supervision import QualitySupervisionAdmission, SupervisionSource
from gym_trading_env.research.fg_s3_e_quality_reader import build_join_index
from embodied.run.fg_checkpoint import validate_bundle


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
    owner=importlib.import_module('gym_trading_env.research.fg_s3_e_quality_'+name)
    runtime=importlib.import_module('embodied.run.fg_s3_e_quality_'+name)
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
    clone=tmp_path/'broken';shutil.copytree(run.root,clone)
    (clone/'ckpt/step_000000000001/done').unlink()
    before={str(p):p.read_bytes() for p in clone.rglob('*') if p.is_file()}
    def forbidden(*a,**k):raise AssertionError('invalid latest constructed an Agent')
    monkeypatch.setattr(owner,'create_agent',forbidden)
    if stage!='f3':monkeypatch.setattr(runtime,'create_agent',forbidden)
    try:
        with pytest.raises(ValueError,match='Incomplete FG full checkpoint'):
            cls(prep,index,clone,resume=True,_probe=permit)
        assert before=={str(p):p.read_bytes() for p in clone.rglob('*') if p.is_file()}
    finally:run.source.close()
