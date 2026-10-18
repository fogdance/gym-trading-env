import copy,json
import numpy as np
import pytest
from dreamerv3.fg_s3_e_quality_actor import execution_contract,model_extensions,LOSS_DEFAULTS
from dreamerv3.fg_s3_e_quality_course_contract import expected_stage_contract
from gym_trading_env.research.fg_s3_e_quality_actor import validate_settings,create_agent,CONFIG_SCHEMA
from wp04_fixture import settings as original

def settings(root):
    s=original(root)
    s.update(schema=CONFIG_SCHEMA,stage='ACTOR_SUPERVISED',sampling_stage='ACTOR_STRICT',source_stage='F1',
        source_run=str(root.parent/'f1'),source_checkpoint=str(root.parent/'f1/ckpt/step_000000000004'),
        max_consumed_batches=40,actor_loss=dict(LOSS_DEFAULTS),aux_sampling_spec=None)
    return s

def test_A01_execution_subset_preserves_course():
    frozen=json.dumps(expected_stage_contract('ACTOR_SUPERVISED'),sort_keys=True)
    c=execution_contract();assert c['effective_trainable_groups']==['quality_actor/']
    assert all(v=='FROZEN' for k,v in c['parameter_group_policy'].items() if k!='quality_actor/')
    assert frozen==json.dumps(expected_stage_contract('ACTOR_SUPERVISED'),sort_keys=True)

@pytest.mark.parametrize('key,value',[('quality_stage','CONTROLLED_FULL'),('quality_f3_protocol','FAKE'),
    ('quality_f2_protocol','FAKE'),('quality_q_v2_lambda',1.),('quality_soft_lambda',1.),
    ('quality_outcome_lambda',1.),('quality_decision_admitted',True),('quality_fuse_score',True)])
def test_A02_guard_rejects_bypass(key,value):
    from dreamerv3.fg_s3_quality_v2 import quality_extensions
    with pytest.raises(ValueError):quality_extensions({**model_extensions(8,4),key:value})

@pytest.mark.parametrize('key,value',[('source_stage','OLD'),('source_run',''),('burnin_steps',0),
    ('sampling_spec',{'mode':'STRATIFIED'}),('actor_loss',{'natural':0}),('actor_loss',{'quality_pair':float('nan')}),
    ('max_consumed_batches',1),('actor_loss',{'soft':1})])
def test_A03_invalid_settings(tmp_path,key,value):
    with pytest.raises(ValueError):validate_settings({**settings(tmp_path/'corpus'),key:value})

def test_A04_complete_model_no_teacher_policy_inputs(tmp_path):
    s=validate_settings(settings(tmp_path/'corpus'));a=create_agent(s,tmp_path/'model',controlled_small=True)
    assert a.model.f3 and not a.model.f1 and not a.model.f2 and int(a.n_updates)==0
    assert not any(k.startswith(('teacher_','quality_','f3_')) for k in a.obs_space)
    actor=[k for k in a.params if k.startswith('quality_actor/')]
    assert actor and all(a.model.f3_state_owner[k]=='quality_actor/' for k in actor)
    assert all(np.isfinite(np.asarray(v)).all() for v in a.params.values())
    assert all(any(k.startswith(p) for k in a.params) for p in expected_stage_contract('ACTOR_SUPERVISED')['parameter_group_policy'])
