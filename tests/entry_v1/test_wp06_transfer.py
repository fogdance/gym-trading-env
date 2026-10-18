import copy,json,pickle,shutil
from pathlib import Path
import numpy as np
import pytest
from wp06_fixture import engineering_settings
from gym_trading_env.research.fg_s3_e_quality_actor import create_agent,preflight,require_training_authority
from embodied.run.fg_s3_e_quality_transfer import inspect_stage,transfer_parameters
from dreamerv3.fg_s3a_actor import parameter_digests

@pytest.fixture(scope='module',params=['F1','F2'])
def case(tmp_path_factory,request):
    root=tmp_path_factory.mktemp('wp06_transfer_'+request.param);s=engineering_settings(root,stage=request.param)
    proof,data=inspect_stage(s,allow_engineering=True)
    return root,s,proof,data

def test_B01_full_parameter_handoff_fresh_optimizer_and_mirror(case):
    root,s,proof,data=case
    from gym_trading_env.research.fg_replay import digest,write_json
    before_source=digest(Path(s['source_checkpoint'])/'agent.pkl')
    a=create_agent(s,root/'model',controlled_small=True);fresh=a.save_exact();before=parameter_digests(a.params)
    receipt=transfer_parameters(a,data,proof,target_stage='ACTOR_SUPERVISED');after=parameter_digests(a.params)
    expected=parameter_digests(data['params'])
    assert all(after[k]==expected[k] for k in after if not k.startswith('opt/'))
    assert all(after[k]==before[k] for k in after if k.startswith('opt/'))
    assert a.save_exact()['counters']==fresh['counters']
    assert parameter_digests(a.policy_params)=={k:after[k] for k in a.policy_keys}
    assert a.pending_sync is a.pending_outs is a.pending_mets is None
    assert receipt['protocol']=='F1_OR_F2_TO_F3_TRANSFER_RECEIPT' and receipt['reader_stage']=='ACTOR_STRICT'
    assert not receipt['training_admitted'] and not receipt['same_run_resume']
    assert digest(Path(s['source_checkpoint'])/'agent.pkl')==before_source
    write_json(Path('docs/receipts/wp06')/('B_'+s['source_stage']+'_handoff.json'),receipt)

@pytest.mark.parametrize('attack',['missing','shape','dtype','nonfinite'])
def test_B02_parameter_rejection_before_mutation(case,tmp_path,attack):
    _,s,proof,data=case;a=create_agent(s,tmp_path/'model',controlled_small=True)
    changed=copy.deepcopy(data);key=next(k for k in changed['params'] if k.startswith('quality_actor/'))
    if attack=='missing':changed['params'].pop(key)
    elif attack=='shape':changed['params'][key]=np.zeros(1,np.float32)
    elif attack=='dtype':changed['params'][key]=changed['params'][key].astype(np.float64)
    else:changed['params'][key][...]=np.nan
    before=parameter_digests(a.params)
    with pytest.raises(ValueError,match='HANDOFF'):transfer_parameters(a,changed,proof,target_stage='ACTOR_SUPERVISED')
    assert before==parameter_digests(a.params) and int(a.n_updates)==0

def test_B03_engineering_is_not_formal_and_wrong_stage_rejected(case):
    _,s,_,_=case
    with pytest.raises(ValueError,match='NOT_TRAINED'):inspect_stage(s)
    p=preflight(s);assert p.receipt['status']=='F3_ENGINEERING_READY' and not p.receipt['training_admitted']
    assert 'FOUNDATION_NOT_TRAINED' in p.receipt['blockers']
    with pytest.raises(ValueError,match='AUTHORIZATION_REQUIRED'):require_training_authority(p)
    with pytest.raises(ValueError,match='NOT_TRAINED'):require_training_authority(p,authorized=True)
    with pytest.raises(ValueError):inspect_stage({**s,'source_stage':'F2' if s['source_stage']=='F1' else 'F1'},allow_engineering=True)
    assert not Path(s['run_dir']).exists()

def test_B04_source_freeze_and_ledger_attacks(case,tmp_path):
    from embodied.run.fg_s3_e_quality_transfer import source_freeze_evidence
    from dreamerv3.fg_s3_e_quality_course_contract import WM_GROUPS
    from dreamerv3.fg_s3_e_quality_heads import GROUPS
    _,s,proof,data=case;changed=copy.deepcopy(data)
    key=next(k for k in changed['params'] if k.startswith('quality_actor/'));changed['params'][key]+=1
    with pytest.raises(ValueError,match='FROZEN_STATE_CHANGED'):
        source_freeze_evidence(s['source_run'],proof['identity'],changed,WM_GROUPS if s['source_stage']=='F1' else GROUPS)
    if s['source_stage']=='F2':
        root=tmp_path/'source';shutil.copytree(s['source_run'],root)
        (root/'updates.jsonl').write_text('')
        with pytest.raises(ValueError,match='LEDGER_INCOMPLETE'):
            inspect_stage({**s,'source_run':str(root),'source_checkpoint':str(root/'ckpt/step_000000000004')},allow_engineering=True)


def test_B05_editable_receipt_cannot_grant_authority(case):
    from gym_trading_env.research.fg_s3_e_quality_actor import F3Preparation
    from gym_trading_env.research.fg_s3_e_quality_supervision import Admission
    _,s,_,_=case;p=preflight(s);receipt=copy.deepcopy(p.receipt)
    receipt['status']='F3_DATA_READY';receipt['identity']['source_model']['engineering_only']=False
    fake=F3Preparation(p.settings,Admission(p.source.source,p.source.receipt,{'result':{'G3':'GO'}}),receipt)
    with pytest.raises(ValueError,match='ACTUAL_NATIVE_EXECUTION_PROOF_REQUIRED'):require_training_authority(fake,authorized=True)


@pytest.mark.parametrize('attack',['observation','stage','unfinished','corrupt'])
def test_B06_source_identity_negatives(case,tmp_path,attack):
    from gym_trading_env.research.fg_replay import write_json,signed
    _,s,_,_=case;root=tmp_path/'source';shutil.copytree(s['source_run'],root)
    node=root/'ckpt/step_000000000004';current={**s,'source_run':str(root),'source_checkpoint':str(node)}
    if attack=='corrupt':(node/'agent.pkl').write_bytes(b'corrupt')
    else:
        identity=json.loads((root/'run_identity.json').read_text())['identity'];p=identity['preparation']
        if attack=='observation':p['observation']['macd_5m']['schema']='FG_S3_E_MACD5M_CAUSAL_V2'
        elif attack=='stage':
            if s['source_stage']=='F1':p['course_stage']='ACTOR_SUPERVISED'
            else:p['settings']['stage']='ACTOR_SUPERVISED'
        else:
            if s['source_stage']=='F1':p['optimizer_updates']+=1
            else:p['settings']['optimizer_updates']+=1
        write_json(root/'run_identity.json',signed(dict(identity=identity)))
        m=json.loads((node/'manifest.json').read_text());m['identity']=identity;(node/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):inspect_stage(current,allow_engineering=True)
