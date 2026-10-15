"""Released E fresh source/signature contracts; formal data stays strict."""
import ast
import copy
from pathlib import Path
import subprocess

import pytest

from test_fg_s3_teacher_catalog import catalog_bundle
from gym_trading_env.research.fg_s3_teacher_catalog import load_catalog
from gym_trading_env.research.fg_replay import read_manifest, signed, write_json
from dreamerv3.fg_s3_composition import normalize_config, resolve_composition
from dreamerv3.fg_s3_e_config import e_target_config, resolve_model_init
from dreamerv3.fg_s3_tutor_loss import agent_extensions
from dreamerv3.fg_s3_v2_agent import S3V2Agent
from dreamerv3.fg_s3a_actor import parameter_digests
from embodied.run.fg_s3_source_admission import _inspect_source, inspect_source, _prepare_transfer
from embodied.run.fg_s3_source_transfer import make_agent, apply_and_audit_transfer
from embodied.run.fg_s3_e_source import make_binding, validate_e_records
from gym_trading_env.research.fg_s3_v2_gpu_check import small_config
from gym_trading_env.research.fg_s3_v2_experiment import _validate_settings, read_settings
from e_model_fixture import SyntheticEProvider

ROOT=Path(__file__).resolve().parents[2]
DREAMER=ROOT.parent/'dreamerv3'


def test_EM14_size50m_canonical_recipe_and_fresh_settings_without_checkpoint(tmp_path):
    cfg=resolve_model_init(dict(seed=0,presets=['size50m']))
    e=cfg['agent']['enc']['e_multiscale_1m15m']
    assert e['units']==512 and e['context_units']==128 and e['layers']==3
    assert cfg['agent']['dyn']['rssm']['deter']==4096
    for mode in ('a2','tutor_cd','b2'):
        from dreamerv3.configuration import read_config_yaml
        raw=read_config_yaml(ROOT/'configs/experiments/fg_s3_v2/F0_A2_LEGACY_1K.yaml')
        raw.pop('source_checkpoint');raw['source_kind']='direct_init'
        raw['composition']=dict(encoder='e_multiscale_1m15m',teaching=mode)
        raw['model_init']=dict(seed=0,presets=['size50m'])
        raw['e_data_manifest']='artifacts/fg_s3_v2/e_data_pending/manifest.json'
        raw['e_data_source_pool']='artifacts/fg_s3_v2/e_data_pending/train_pool.json'
        path=tmp_path/(mode+'.yaml')
        from ruamel.yaml import YAML
        YAML().dump(raw,path)
        s=_validate_settings(normalize_config(raw,executable=False),path)
        assert 'source_checkpoint' not in s and s['resolved_composition']['support_state']=='ready'
        assert read_settings(path)==s
        wrong=copy.deepcopy(raw);wrong['source_checkpoint']='old/checkpoint'
        with pytest.raises(ValueError,match='without source_checkpoint'):
            _validate_settings(normalize_config(wrong,executable=False),path)


@pytest.mark.parametrize('mode',['a2','tutor','b2'])
def test_EM14_authenticated_direct_init_model_and_signed_records(catalog_bundle,tmp_path,mode):
    catalog=load_catalog(catalog_bundle[-1])
    cfg=e_target_config(small_config())
    provider=SyntheticEProvider(tmp_path)
    teaching={'a2':'a2','tutor':'tutor_cd','b2':'b2'}[mode]
    comp=resolve_composition({'composition':dict(encoder='e_multiscale_1m15m',teaching=teaching)})
    ext=agent_extensions(dict(mode=mode,b2_mix_warmup=2,b2_mix_ramp=2,b2_min_event_updates=2))
    binding=make_binding(cfg,comp,provider.manifest)
    kwargs=dict(catalog=catalog,seed=0,config=cfg,composition=comp,diagnostic=True,e_binding=binding)
    if mode=='b2':kwargs['agent_extension']=ext
    spec=_inspect_source('direct_init',**kwargs)
    assert spec['origin']['step']==spec['target_update_offset']==0
    assert _prepare_transfer(spec) is None and 'source_param_digests' not in spec
    agent=make_agent(cfg,tmp_path/'model',agent_cls=S3V2Agent,device='cpu',bc_lambda=1.,
        extra_agent_config=ext,composition=comp)
    if mode=='b2':
        from dreamerv3.agent import Agent
        with pytest.raises(ValueError,match='shared S3V2'):
            make_agent(cfg,tmp_path/'wrong_model',agent_cls=Agent,device='cpu',
                extra_agent_config=ext,composition=comp)
    receipt=apply_and_audit_transfer(agent,spec)
    assert agent.save_exact()['counters']==dict(updates=0,batches=0,actions=0)
    assert int(agent.params['opt/step/value'])==0
    assert receipt['params_before']==receipt['params_after']
    assert spec['gate_transfer']['new_gate_state']['actor_enabled'] is False
    assert spec['gate_transfer']['new_gate_state']['samples']==0
    if mode=='a2':
        peer=make_agent(cfg,tmp_path/'same_seed',agent_cls=S3V2Agent,device='cpu',bc_lambda=1.,
            extra_agent_config=ext,composition=comp)
        assert parameter_digests(peer.params)==parameter_digests(agent.params)
        with pytest.raises(ValueError,match='seed/Reward'):
            _inspect_source('direct_init',**{**kwargs,'seed':1})
        with pytest.raises(ValueError,match='integer seed'):
            _inspect_source('direct_init',**{**kwargs,'seed':False})
    if mode=='b2':
        assert all(any(k.startswith(p) for k in agent.params) for p in
                   ('b2_event/','b2_early/','b2_adapter/','b2_schedule/'))
    frozen=signed(dict(config=cfg,e_binding=binding,resolved_composition=comp,agent_extension=ext))
    prepared=signed(dict(protocol='FG_S3_E_CPU_PREPARED_CONTRACT_V1',
        status='CPU_CONTRACT_ONLY_NOT_PRODUCTION_PREPARED',e_binding=binding,resolved_composition=comp))
    assert validate_e_records(spec,spec['target_model_identity'],frozen,prepared)==binding
    write_json(tmp_path/'source_admission.json',spec)
    write_json(tmp_path/'transfer.json',receipt)
    write_json(tmp_path/'model_config.json',frozen)
    write_json(tmp_path/'prepared_contract.json',prepared)
    wrong={**frozen,'e_binding':{**binding,'seed':1}}
    with pytest.raises(ValueError,match='identity mismatch'):validate_e_records(spec,wrong)
    corrupt=copy.deepcopy(spec);corrupt.pop('content_hash')
    corrupt['target_model_identity']['e_binding']=copy.deepcopy(corrupt['target_model_identity']['e_binding'])
    corrupt['target_model_identity']['e_binding']['data_identity']='foreign'
    with pytest.raises(ValueError,match='identity mismatch'):_prepare_transfer(signed(corrupt))
    assert inspect_source('direct_init',**kwargs)==spec
    for key,value in (('seed',1),('agent_config_hash','foreign'),('data_manifest_sha256','foreign')):
        bad={**kwargs,'e_binding':{**binding,key:value}}
        with pytest.raises(ValueError,match='identity mismatch'):_inspect_source('direct_init',**bad)
    with pytest.raises(ValueError,match='old Encoder'):_inspect_source('warm_start',**kwargs)
    bad={**kwargs,'composition':resolve_composition({'mode':mode})}
    with pytest.raises(ValueError,match='E fresh-init'):_inspect_source('direct_init',**bad)


def test_formal_e_binding_still_refuses_synthetic_training_data(tmp_path):
    cfg=e_target_config(small_config())
    provider=SyntheticEProvider(tmp_path)
    comp=resolve_composition({'composition':dict(encoder='e_multiscale_1m15m',teaching='a2')})
    with pytest.raises(ValueError,match='authenticated real TRAIN data'):
        make_binding(cfg,comp,provider.manifest,diagnostic=False)


def test_EM16_original_teaching_method_bodies_unchanged():
    original=ast.parse(subprocess.check_output(['git','show','eb63de7:dreamerv3/fg_s3_v2_agent.py'],cwd=DREAMER))
    current=ast.parse((DREAMER/'dreamerv3/fg_s3_v2_agent.py').read_text())
    old=next(n for n in original.body if isinstance(n,ast.ClassDef) and n.name=='S3V2Agent')
    new=next(n for n in current.body if isinstance(n,ast.ClassDef) and n.name=='_S3V2Model')
    methods=lambda n:{f.name:ast.dump(f,include_attributes=False) for f in n.body
                     if isinstance(f,ast.FunctionDef) and f.name!='__init__'}
    assert methods(old)==methods(new)
