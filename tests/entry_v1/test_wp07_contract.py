import copy
import json
from pathlib import Path

import pytest

from dreamerv3.fg_s3_e_quality_online import PROTOCOL, execution_contract, validate_execution
from wp07_fixture import execution, f3_source


def test_modes_preserve_course_and_owner_policy():
    from dreamerv3.fg_s3_e_quality_course_contract import expected_stage_contract
    before=json.dumps(expected_stage_contract('ONLINE_ADAPT'),sort_keys=True)
    assert execution_contract('SHADOW')['effective_trainable_groups']==[]
    assert execution_contract('ONLINE_SUPERVISED')['effective_trainable_groups']==['quality_actor/']
    assert execution_contract('ONLINE_RL_CONTROLLED')['effective_trainable_groups']==['quality_actor/','val/','slowval/']
    assert before==json.dumps(expected_stage_contract('ONLINE_ADAPT'),sort_keys=True)
    assert not execution_contract('ONLINE_SUPERVISED')['formal_online_adapt_admitted']


@pytest.mark.parametrize('field,value', [('fold','TEST'),('source_stage','F1'),('execution_mode','ONLINE_ADAPT'),
    ('source_run',''),('model_seed',True),('exploration_rate',float('nan')),('updates_per_collect',1)])
def test_explicit_spec_rejects_illegal_fields(tmp_path,field,value):
    # Pure structure test; no source artifact construction necessary.
    s=dict(protocol=PROTOCOL,source_stage='ACTOR_SUPERVISED',execution_mode='SHADOW',fold='TRAIN',
        source_run=str(tmp_path/'src'),source_checkpoint=str(tmp_path/'src/ckpt/x'),run_dir=str(tmp_path/'out'),
        model_seed=1,env_seed=2,replay_seed=3,env_steps=100,collect_every=10,updates_per_collect=0,
        max_consumed_batches=0,save_every=20,replay_capacity=512,exploration_rate=0.,history_ratio=.5)
    with pytest.raises(ValueError):validate_execution({**s,field:value})
    assert not (tmp_path/'out').exists()


def test_actual_f3_source_and_fresh_all_parameter_handoff(tmp_path):
    from embodied.run.fg_s3_e_quality_online import inspect_f3
    from embodied.run.fg_s3_e_quality_transfer import transfer_parameters
    from gym_trading_env.research.fg_s3_e_quality_actor import create_agent
    from dreamerv3.fg_s3a_actor import parameter_digests
    s=execution(tmp_path/'out');proof,state=inspect_f3(s,allow_engineering=True)
    settings=copy.deepcopy(f3_source());settings['model_init']['seed']=s['model_seed']
    a=create_agent(settings,tmp_path/'model',controlled_small=True)
    before=parameter_digests(a.params)
    receipt=transfer_parameters(a,state,proof,target_stage='ACTOR_ONLINE')
    after=parameter_digests(a.params)
    assert receipt['nonopt_keys'] and all(after[k]==receipt['source_nonopt'][k] for k in receipt['nonopt_keys'])
    assert all(after[k]==before[k] for k in before if k.startswith('opt/'))
    assert int(a.n_updates)==0 and proof['source_freeze']['changed_parameters']
    assert receipt['protocol']=='F3_TO_ONLINE_TRANSFER_RECEIPT' and not receipt['same_run_resume']
    Path('docs/receipts/wp07/A_handoff.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with pytest.raises(ValueError,match='NOT_FORMAL'):inspect_f3(s)
    assert not (tmp_path/'out').exists()


def test_source_tamper_and_wrong_stage_refuse_before_run(tmp_path):
    from gym_trading_env.research.fg_s3_e_quality_online import preflight
    from gym_trading_env.research.fg_replay import read_manifest,signed,write_json
    import shutil
    s=execution(tmp_path/'out');clone=tmp_path/'src';shutil.copytree(s['source_run'],clone)
    s.update(source_run=str(clone),source_checkpoint=str(clone/'ckpt/step_000000000004'))
    p=clone/'run_identity.json';value=read_manifest(p);value.pop('content_hash')
    value['identity']['protocol']='FG_S3_E_QUALITY_F2_RUN_V1';write_json(p,signed(value))
    with pytest.raises(ValueError):preflight(s,allow_engineering=True)
    assert not (tmp_path/'out').exists()
