import copy,json
from pathlib import Path
import numpy as np
import pytest
from dreamerv3.fg_s3_e_quality_heads import execution_contract,model_extensions,GROUPS,LABELS
from dreamerv3.fg_s3_e_quality_course_contract import expected_stage_contract
from gym_trading_env.research.fg_s3_e_quality_heads import create_agent,validate_settings
from wp04_fixture import settings as f1_settings

def settings(root):
 s=f1_settings(root)
 s.update(schema='FG_S3_E_QUALITY_F2_CONFIG_V1',stage='QUALITY_HEADS',sampling_stage='HEADS_VALID',source_f1_run=str(root.parent/'f1'),source_f1_checkpoint=str(root.parent/'f1/ckpt/step_000000000004'))
 return s

def test_A01_subset_preserves_original_course():
 upper=expected_stage_contract('QUALITY_HEADS');frozen=json.dumps(upper,sort_keys=True)
 c=execution_contract();assert tuple(c['effective_trainable_groups'])==GROUPS
 assert all(upper['parameter_group_policy'][p]=='TRAIN_WITH_VALID_LABEL' for p in GROUPS)
 assert c['parameter_group_policy']['quality_q_v2/']=='FROZEN'
 assert len(upper['parameter_groups_trainable'])==6
 assert json.dumps(expected_stage_contract('QUALITY_HEADS'),sort_keys=True)==frozen

@pytest.mark.parametrize('key,value',[('quality_stage','HEADS_ONLY'),('quality_f2_protocol','FAKE'),('quality_q_c1_lambda',.1),('quality_soft_lambda',.1),('quality_decision_admitted',True),('quality_f1_protocol','FAKE')])
def test_A02_protocol_and_guard_negative(key,value):
 from dreamerv3.fg_s3_quality_v2 import quality_extensions
 with pytest.raises(ValueError):quality_extensions({**model_extensions(8,4),key:value})

@pytest.mark.parametrize('key,value',[('source_f1_run',''),('stage','FOUNDATION_WM'),('sampling_stage','WM_NATURAL'),('burnin_steps',0),('sampling_spec',{'mode':'MIXED'}),('prediction_weights',{'quality_q_v2_lambda':1})])
def test_A03_settings_negatives(tmp_path,key,value):
 with pytest.raises(ValueError):validate_settings({**settings(tmp_path/'corpus'),key:value})

def test_A04_complete_native_model(tmp_path):
 a=create_agent(validate_settings(settings(tmp_path/'corpus')),tmp_path/'model',controlled_small=True)
 assert a.model.f2 and not a.model.f1 and int(a.n_updates)==0
 assert set(k for k in a.spaces if k.startswith('quality_'))==set(LABELS)
 assert not any(k.startswith(('quality_','teacher_')) for k in a.obs_space)
 assert not any(k.startswith('teacher_') for k in a.spaces)
 assert all(any(k.startswith(p) for k in a.params) for p in expected_stage_contract('QUALITY_HEADS')['parameter_group_policy'])
 assert any(k.startswith('enc/macd5/') for k in a.params)
 assert all(np.isfinite(np.asarray(v)).all() for v in a.params.values())
