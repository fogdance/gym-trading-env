"""CPU management of the actual joint runner, not an alternative trainer."""
import copy
from pathlib import Path
import sys

import pytest
import yaml

from gym_trading_env.research import fg_s3_experiment as manager
from gym_trading_env.research.fg_replay import digest,read_manifest

DREAMER=Path(__file__).resolve().parents[3]/'dreamerv3'
sys.path.insert(0,str(DREAMER/'embodied/tests'))
from test_fg_s3_source_transfer import source_bundle


def test_managed_joint_prepare_start_resume_and_immutable_budget(source_bundle,tmp_path,monkeypatch):
    b=source_bundle
    template=tmp_path/'model.yaml'
    defaults=copy.deepcopy(b['config'])
    defaults['agent']['avail_actor_enabled']=True  # Controlled diagnostic fixture.
    template.write_text(yaml.safe_dump({'defaults':defaults}))
    cfg=dict(schema=manager.SCHEMA,experiment='joint_cpu_fixture',trainer='s3c',diagnostic=True,
        prepared_dir=str(tmp_path/'prepared'),dreamer_root=str(DREAMER),
        model=dict(template=str(template),presets=['defaults'],overrides={'report_length':16}),
        paths=dict(catalog=str(b['root']/'catalog'),replay_directory=str(b['trace']/'replay')),
        source=dict(kind='direct_init',checkpoint=None,base_checkpoint=None,features=None),
        train=dict(mode='S3C1',seed=0,replay_seed=202,bc_lambda=1.,batch_size=2,updates=2,
            report_every=1,save_every=1,consec=2,capacity=2048,device='cpu',logdir=str(tmp_path/'run')))
    requested=tmp_path/'experiment.yaml';requested.write_text(yaml.safe_dump(cfg))
    receipt=manager.prepare(requested)
    assert receipt['protocol']==manager.JOINT_PROTOCOL and receipt['status']=='READY_FOR_JOINT'
    assert not (tmp_path/'run').exists()
    assert manager.audit(tmp_path/'prepared')['status']=='READY_FOR_JOINT'
    result=manager.start(tmp_path/'prepared',config=requested)
    assert result['local_updates']==2 and result['counters']['bc_effective_updates']==2
    checkpoint=Path(result['checkpoint']);before=digest(checkpoint/'agent.pkl')
    binding=read_manifest(tmp_path/'run/managed_snapshot/manifest.json')['binding']
    assert binding['prepared_manifest_sha256']==digest(tmp_path/'prepared/manifest.json')
    assert not (tmp_path/'prepared/features_binding.json').exists()
    monkeypatch.setattr(manager,'_joint_admission',lambda *a,**k:pytest.fail('resume re-admitted source'))
    assert manager.start(tmp_path/'prepared',resume=True,config=requested)['local_updates']==2
    assert digest(checkpoint/'agent.pkl')==before
    cfg['train']['updates']=3;requested.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='YAML differs'):
        manager.start(tmp_path/'prepared',resume=True,config=requested)
    assert manager.require_prepared(tmp_path/'prepared')[1]['train']['updates']==2


def test_joint_yaml_uses_canonical_types_and_rejects_foreign_source(tmp_path):
    _,cfg=manager.read_experiment('S3C1_CPU_FLOAT32')
    cfg['train']['bc_lambda']=0.00004
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg))
    assert isinstance(manager.read_experiment(path)[1]['train']['bc_lambda'],float)
    cfg['source']['kind']='after_bc_s3b';path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='NOT_SUPPORTED'):manager.read_experiment(path)
