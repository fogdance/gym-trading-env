"""Checksummed synthetic source states; not formal training/admission evidence."""
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from gym_trading_env.research.fg_replay import digest, signed, write_json
from gym_trading_env.research.fg_s3_e_observation import observation_contract
from dreamerv3.fg_s3_e_quality_course_contract import ALL_GROUPS, REWARD_SCHEMA
from embodied.run.fg_s3_e_quality_transfer import inspect_f1, inspect_stage


def bundle(root, identity, step, params):
    node=root/'ckpt'/f'step_{step:012d}';node.mkdir(parents=True,exist_ok=True)
    values=dict(agent=dict(params=params,counters=dict(updates=step)),
                counters=dict(optimizer_updates=step,consumed_batches=step),step=step)
    for key,value in values.items():(node/(key+'.pkl')).write_bytes(pickle.dumps(value))
    manifest=dict(schema=1,protocol='FG_FULL_V1',step=step,identity=identity,
                  protected=True,backend='external_immutable_course_reader',strict_latest=True,
                  files={p.name:digest(p) for p in node.iterdir() if p.suffix=='.pkl'})
    (node/'manifest.json').write_text(json.dumps(manifest));(node/'done').touch()
    (root/'ckpt/latest').write_text(node.name)
    return node


@pytest.fixture
def sources(tmp_path,monkeypatch):
    from gym_trading_env.research import fg_s3_e_quality_heads as heads,fg_s3_e_quality_actor as actor
    from dreamerv3.fg_s3_e_quality_heads import execution_contract
    cfg=dict(agent=dict(opt=dict(wd=0.)))
    for module in (heads,actor):monkeypatch.setattr(module,'model_config',lambda *a,**k:copy.deepcopy(cfg))
    params={p+'w':np.array([0.],np.float32) for p in (*ALL_GROUPS,'retnorm/')}
    for name in list(params):
        for slot in (1,2):params[f'opt/state/{slot}/1/'+name.replace('_','')]=np.zeros(1,np.float32)
    for key in ('opt/step/value','opt/state/1/0','opt/state/2/0','opt/state/3/count'):params[key]=np.array(0,np.int32)
    root=tmp_path/'f1'
    identity=dict(protocol='FG_S3_E_QUALITY_F1_RUN_V1',backend='external_immutable_course_reader',
        recovery_policy='STRICT_PUBLISHED_LATEST_V2',controlled_small=True,training_admitted=False,
        effective_model=cfg,preparation=dict(course_stage='FOUNDATION_WM',optimizer_updates=1,
            observation=observation_contract(),reward_schema=REWARD_SCHEMA))
    write_json(root/'run_identity.json',signed(dict(identity=identity)))
    bundle(root,identity,0,params);final=copy.deepcopy(params);final['enc/w']+=1
    final['opt/step/value'][...]=1;bundle(root,identity,1,final)
    settings=dict(source_f1_run=str(root),source_f1_checkpoint=str(root/'ckpt/step_000000000001'))
    proof,_=inspect_f1(settings,allow_engineering=True)
    f2=tmp_path/'f2';s={**settings,'stage':'QUALITY_HEADS','optimizer_updates':1}
    i2=dict(protocol='FG_S3_E_QUALITY_F2_RUN_V1',backend='external_immutable_course_reader',
        controlled_small=True,training_admitted=False,preparation=dict(protocol='FG_S3_E_QUALITY_HEADS_V1',
            settings=s,execution_contract=execution_contract(),observation=observation_contract(),
            effective_model=cfg,source_f1=proof))
    write_json(f2/'run_identity.json',signed(dict(identity=i2)))
    initial=copy.deepcopy(final);initial['opt/step/value'][...]=0
    bundle(f2,i2,0,initial);f2final=copy.deepcopy(initial);f2final['quality_event/w']+=1
    f2final['opt/step/value'][...]=1;bundle(f2,i2,1,f2final)
    metrics={'f2/updated':True,'opt/updates':1,'f2/finite':True,'opt/loss':1.,'opt/grad_norm':1.}
    (f2/'updates.jsonl').write_text(json.dumps(dict(consumed_batches=1,local_update=1,agent_update=1,metrics=metrics))+'\n')
    return settings,root,identity,params,final,f2,i2


def inspect_route(sources,route):
    settings,root,_,_,_,f2,_=sources
    if route=='F1_F2':return inspect_f1(settings,allow_engineering=True)
    stage='F1' if route=='F1_F3' else 'F2';source=root if stage=='F1' else f2
    return inspect_stage(dict(source_stage=stage,source_run=str(source),
        source_checkpoint=str(source/'ckpt/step_000000000001')),allow_engineering=True)


@pytest.mark.parametrize('route',['F1_F2','F1_F3','F1_F2_F3'])
def test_all_routes_include_the_same_complete_f1_proof(sources,route):
    proof,_=inspect_route(sources,route)
    f1=proof.get('upstream_proof',proof)
    assert f1['protocol']=='FG_S3_F1_SOURCE_V2'
    assert f1['source_freeze']['all_frozen_bytes_equal']
    assert f1['source_freeze']['changed_groups']['enc/']==1
    assert sum(f1['source_freeze']['changed_groups'].values())==1
    assert f1['engineering_only'] and not f1['identity']['training_admitted']


@pytest.mark.parametrize('route',['F1_F2','F1_F3','F1_F2_F3'])
@pytest.mark.parametrize('attack',['actor','head','normalizer','momentum','unknown_slot','fake_clock','keys','shape','dtype','finite'])
def test_every_f1_successor_rejects_frozen_or_schema_drift(sources,route,attack):
    _,root,identity,initial,final,_,_=sources
    a,b=copy.deepcopy(initial),copy.deepcopy(final)
    key={'actor':'quality_actor/w','head':'quality_event/w','normalizer':'retnorm/w',
         'momentum':'opt/state/1/1/qualityactor/w'}.get(attack,'enc/w')
    if attack in ('actor','head','normalizer','momentum'):b[key]+=1
    elif attack in ('unknown_slot','fake_clock'):
        unknown='opt/unknown' if attack=='unknown_slot' else 'opt/state/3/alien/count'
        a[unknown]=b[unknown]=np.zeros((),np.int32)
    elif attack=='keys':b.pop('pol/w')
    elif attack=='shape':b[key]=np.zeros(2,np.float32)
    elif attack=='dtype':b[key]=b[key].astype(np.float64)
    else:b[key][...]=np.nan
    bundle(root,identity,0,a);bundle(root,identity,1,b)
    with pytest.raises(ValueError):inspect_route(sources,route)


def test_f2_lineage_requires_the_saved_freeze_evidence(sources):
    _,_,_,_,_,f2,identity=sources
    identity['preparation']['source_f1'].pop('source_freeze',None)
    write_json(f2/'run_identity.json',signed(dict(identity=identity)))
    for n in (0,1):
        node=f2/'ckpt'/f'step_{n:012d}';meta=json.loads((node/'manifest.json').read_text())
        meta['identity']=identity;(node/'manifest.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='LINEAGE'):inspect_route(sources,'F1_F2_F3')
