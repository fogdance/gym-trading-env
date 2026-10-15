"""Posthoc metrics and research authorization boundaries."""
import copy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import yaml

from gym_trading_env.research.fg_s3_quality_prediction_evaluation import quality_scores
from gym_trading_env.research.fg_s3_v2_experiment import require_training_authorization
from gym_trading_env.research.fg_s3_quality_catalog_v2 import soft_teacher
from gym_trading_env.research.fg_replay import signed,write_json,digest


def test_prediction_metrics_compare_to_train_prior_and_keep_censor_missing():
    rows=[]
    for c,score in ((0,100),(1,0)):
        label=dict(event=[c]*7,event_valid=[True]*7,score_valid=True,score=score,exit_type='TP5' if c==0 else 'SL30',
            touch=[[1,0]]*4,touch_valid=[[True,True]]*4,adverse=[c]*4,adverse_valid=[True]*4)
        pred={'quality/outcome_probs':[[.7,.1,.1,.1]]*7,'quality/score_prediction':90 if score else 20,
            'quality/touch_probability':[[.8,.2]]*4,'quality/adverse_probability':[.1]*4,
            'quality/delta_O':-.2,'quality/delta_S':.1,'quality/delta_Q':0.,'quality/z_final':-.1}
        rows.append(dict(prediction=pred,label=label,action=0,equity=298182))
    prior=dict(event_counts=[[8,1,1,0]]*7,touch_counts=[[10,10]]*4,touch_positive=[[8,2]]*4,
               adverse_counts=[10]*4,adverse_positive=[1]*4)
    m=quality_scores(rows,prior)
    assert m['horizons']['0']['train_prior_nll']==pytest.approx(-.5*np.log(.8*.1))
    assert m['score_mae']==15 and m['p80_precision']==m['p80_recall']==1
    assert m['score_by_funds']['below_300000']['samples']==2
    assert m['score_by_exit_type']['TP5']['mae']==10 and m['score_by_exit_type']['SL30']['mae']==20
    assert m['horizons']['0']['by_funds']['below_300000']['samples']==2
    assert m['binary_windows']['touch_0_0']['samples']==2
    censored=copy.deepcopy(rows)
    for r in censored:
        r['label']['event_valid']=[False]*7;r['label']['score_valid']=False
        r['label']['touch_valid']=[[False,False]]*4;r['label']['adverse_valid']=[False]*4
    missing=quality_scores(censored)
    assert missing['score_mae'] is None and missing['horizons']['0']['nll'] is None


def test_soft_teacher_source_and_low_funds_do_not_create_strict_labels():
    high=dict(immediate_outcome_verified=True,exit_type='TP5',score_valid=True,score=100)
    assert soft_teacher(high,dict(valid=False))['soft_target_source']=='INVALID'
    assert soft_teacher(high,dict(valid=True,action=0))['soft_target_pADD']==.25
    q={**high,'q':dict(q_v2_pair_valid=True,q_v2=[1.,2.])}
    assert soft_teacher(q,dict(valid=False))['soft_target_source']=='Q_V2_VERIFIED'


def test_first_quality_gpu_smoke_requires_explicit_authorization():
    settings=dict(mode='quality_v2',device='gpu',scope='diagnostic')
    with pytest.raises(PermissionError,match='separate explicit authorization'):
        require_training_authorization(settings,False)
    require_training_authorization(settings,True)
    require_training_authorization({**settings,'device':'cpu'},False)


def test_same_state_reference_keeps_own_carry_and_actual_previous_action():
    from gym_trading_env.research.fg_s3_quality_prediction_evaluation import same_state_inference
    class Model:
        def __init__(self,bias):self.bias=bias;self.seen=[]
        def init_policy(self,batch):return (0,None,None,{'action':np.zeros(batch,np.int32)})
        def policy(self,carry,obs,mode):
            self.seen.append((carry[0],int(carry[-1]['action'][0])))
            carry=(carry[0]+self.bias,None,None,{'action':np.ones(1,np.int32)})
            return carry,{'action':np.ones(1,np.int32)},{'quality/z_final':np.asarray([carry[0]])}
    a,b=Model(1),Model(10)
    same_state_inference({'a':a,'b':b},[{}, {}, {}],[0,0,1])
    assert a.seen==[(0,0),(1,0),(2,0)]
    assert b.seen==[(0,0),(10,0),(20,0)]


def test_holdout_snapshot_uses_exact_disjoint_train_members(tmp_path):
    from test_fg_s3_quality_teacher_v2 import quality_env
    from gym_trading_env.rewards.single3_reward import Single3RewardConfig
    from gym_trading_env.research.fg_s3_v2_evaluation import prepare_holdout,require_quality_evaluation_snapshot
    from gym_trading_env.research.fg_single3 import load_env
    from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
    rows=[];files={}
    for contract in ('CZCE.FG201','CZCE.FG401'):
        folder=tmp_path/contract;folder.mkdir()
        env,_=quality_env(folder,[(1000,1000,1000,1000)]*3)
        raw=env._raw.copy();env.close()
        config=yaml.safe_load((folder/'env.yaml').read_text())
        config['entry'].update(contract=contract,reward=asdict(Single3RewardConfig.for_candidate('C1')))
        config['trading'].update(future_symbol=contract,currency_pair=contract.split('.')[1],data_path=str(folder/'market.csv'))
        for name in ('contract.yaml','directions.yaml'):
            data=yaml.safe_load((folder/name).read_text())
            if name=='contract.yaml':data['contract']=contract
            else:
                for item in data['intervals']:item['contract']=contract
            (folder/name).write_text(yaml.safe_dump(data))
        raw.to_csv(folder/'market.csv',index_label='Date')
        (folder/'env.yaml').write_text(yaml.safe_dump(config))
        checked=FixedDirectionEntryEnv(config_path=folder/'env.yaml');checked.close()
        row=dict(contract=contract,path=str(folder/'market.csv'),sha256=digest(folder/'market.csv'),
            env_config=str(folder/'env.yaml'),env_sha256=digest(folder/'env.yaml'),
            contract_config=str(folder/'contract.yaml'),contract_sha256=digest(folder/'contract.yaml'),
            role='train',eligible_for_training=True,calendar_complete=False,rows=3)
        rows.append(row)
        files.update({str(p):digest(p) for p in folder.iterdir() if p.is_file()})
    root=tmp_path/'source';(root/'inputs').mkdir(parents=True)
    inputs=signed(dict(original_content_hash='test-fixed-train-whitelist',
        views={'C1':dict(rows=rows)},frozen_files=files))
    write_json(root/'inputs/inputs.json',inputs)
    write_json(root/'manifest.json',signed(dict(status='PASS',clock_protocol='S3_MARKET_MINUTE',
        files={'inputs/inputs.json':digest(root/'inputs/inputs.json')})))
    settings=dict(mode='quality_v2',env_receipt=str(root),contracts=['CZCE.FG201'],
                  reward=asdict(Single3RewardConfig.for_candidate('C1')))
    identity=dict(quality_training_contracts=['CZCE.FG201'],
                  training_identity=dict(training_hash='test-fixed-train-whitelist'))
    output=tmp_path/'holdout';output.mkdir()
    snapshot=prepare_holdout(settings,output,'CZCE.FG401',identity)
    assert snapshot['role']=='TRAIN_HOLDOUT' and not snapshot['train_join_allowed']
    native=load_env(snapshot['rows'][0],snapshot['frozen_inputs'])
    assert native.contract=='CZCE.FG401';native.close()
    with pytest.raises(ValueError,match='cannot be its holdout'):
        prepare_holdout(settings,output,'CZCE.FG201',identity)
    with pytest.raises(ValueError,match='whitelist'):
        prepare_holdout(settings,output,'CZCE.FG999',identity)
    changed={**identity,'quality_training_contracts':['CZCE.FG201','CZCE.FG401']}
    with pytest.raises(ValueError,match='Checkpoint source'):
        require_quality_evaluation_snapshot(output/'holdout_manifest.json',identity=changed)
