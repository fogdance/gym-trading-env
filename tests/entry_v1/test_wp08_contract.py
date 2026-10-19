import copy
import json
from pathlib import Path

import pytest

from gym_trading_env.research.fg_s3_e_quality_exam_contract import PROTOCOL,validate_plan,plan_hash


def plan(root):
    return dict(protocol=PROTOCOL,exam_mode='SYNTHETIC_DEV',contracts=['CZCE.FG2701'],
        source=dict(kind='BUILTIN_SYNTHETIC',manifest=None,env_receipt=None),actors={},
        baselines=['ALWAYS_WAIT','ALWAYS_ADD','RANDOM_P10','DELAY_15'],
        start_day='2026-01-08',end_day='2026-01-13',chunk_steps=256,seed=202,
        random_seeds=[101,202,303],output_dir=str(root),evidence_policy=None)


def test_pure_contract_stable_hash_and_no_io(tmp_path,monkeypatch):
    p=plan(tmp_path/'out')
    def forbidden(*a,**k):raise AssertionError('PURE_CONTRACT_OPENED_FILE')
    monkeypatch.setattr(Path,'open',forbidden)
    assert validate_plan(p)==p
    assert plan_hash(p)==plan_hash(json.loads(json.dumps(p,sort_keys=True)))
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('field,value',[
    ('exam_mode',''),('exam_mode','SEALED_TEST_FINAL'),('contracts',['CZCE.FG2701']*2),
    ('seed',True),('random_seeds',[1]),('baselines',['P4_ORACLE']),('chunk_steps',0),
    ('output_dir','/tmp/../run'),('end_day','2026-01-01')])
def test_invalid_contract_no_output(tmp_path,field,value):
    p=plan(tmp_path/'out');p[field]=value
    with pytest.raises(ValueError):validate_plan(p)
    assert not (tmp_path/'out').exists()


def test_train_corpus_cannot_be_dev_and_empty_is_not_draft(tmp_path):
    p=plan(tmp_path/'out');p['exam_mode']='DEV_EXAM'
    with pytest.raises(ValueError,match='ROLE_MISMATCH'):validate_plan(p)
    p['source']=dict(kind='VALIDATION',manifest='',env_receipt='/tmp/receipt')
    with pytest.raises(ValueError,match='EXPLICIT_ABSOLUTE'):validate_plan(p)
    assert not (tmp_path/'out').exists()


def test_role_refused_before_selected_price_open(tmp_path,monkeypatch):
    from gym_trading_env.research import fg_s3_e_quality_exam_source as source
    p=plan(tmp_path/'out');p['exam_mode']='DEV_EXAM'
    metadata=tmp_path/'manifest.json'
    metadata.write_text(json.dumps(dict(product='FG',files=[dict(contract='CZCE.FG2701',role='test',
        eligible_for_training=False,path=str(tmp_path/'DO_NOT_OPEN_TEST.csv'))])))
    p['source']=dict(kind='VALIDATION',manifest=str(metadata),env_receipt=str(tmp_path/'receipt'))
    def forbidden(*a,**k):raise AssertionError('OPENED_PRICE_BEFORE_ROLE_CHECK')
    monkeypatch.setattr(source,'digest',forbidden)
    with pytest.raises(ValueError,match='ROLE_REQUIRED_PRICE_NOT_OPENED'):source.validation_members(p)
    assert not (tmp_path/'out').exists()


def test_symlink_and_forged_validation_receipt_refused(tmp_path):
    from gym_trading_env.research.fg_s3_e_quality_exam_source import validation_members
    p=plan(tmp_path/'out');p['exam_mode']='DEV_EXAM'
    metadata=tmp_path/'manifest.json';link=tmp_path/'link.json'
    metadata.write_text('{}');link.symlink_to(metadata)
    p['source']=dict(kind='VALIDATION',manifest=str(link),env_receipt=str(tmp_path/'receipt'))
    with pytest.raises(ValueError,match='symlink'):validation_members(p)
    p['source']['manifest']=str(metadata)
    with pytest.raises(ValueError,match='DEV_SOURCE_NOT_AVAILABLE'):validation_members(p)
    assert not (tmp_path/'out').exists()


def test_json_cannot_mint_native_validation_admission(tmp_path):
    from gym_trading_env.research.fg_s3_e_quality_exam_source import ExamSource
    fake=ExamSource(lambda:None,{},1,dict(role='validation',eligible_for_training=False,
        start_day='2026-01-08',end_day='2026-01-13',frozen_inputs={}))
    p=plan(tmp_path/'out');p['exam_mode']='DEV_EXAM'
    with pytest.raises(ValueError,match='NOT_JSON'):fake.require_mode(p)


def test_actual_training_corpus_manifest_is_not_a_dev_market_source(tmp_path):
    from wp08_fixture import f3_reference
    from gym_trading_env.research.fg_s3_e_quality_exam_source import validation_members
    p=plan(tmp_path/'out');p['exam_mode']='DEV_EXAM'
    p['source']=dict(kind='VALIDATION',manifest=str(Path(f3_reference()['corpus_root'])/'corpus_manifest.json'),
        env_receipt=str(tmp_path/'missing'))
    with pytest.raises(ValueError,match='DEV_SOURCE_NOT_AVAILABLE'):validation_members(p)
    assert not (tmp_path/'out').exists()


def test_disjoint_contract_windows_are_explicit_and_inside_study(tmp_path):
    from gym_trading_env.research.fg_s3_e_quality_exam_contract import contract_window
    p=plan(tmp_path/'out');p['contracts']=['CZCE.FG2701','CZCE.FG2705']
    p['contract_windows']={'CZCE.FG2701':dict(start_day='2026-01-08',end_day='2026-01-09'),
        'CZCE.FG2705':dict(start_day='2026-01-12',end_day='2026-01-13')}
    assert contract_window(validate_plan(p),'CZCE.FG2705')['start_day']=='2026-01-12'
    bad=copy.deepcopy(p);bad['contract_windows'].pop('CZCE.FG2705')
    with pytest.raises(ValueError,match='ALL_CONTRACT_WINDOWS'):validate_plan(bad)
    bad=copy.deepcopy(p);bad['contract_windows']['CZCE.FG2705']['end_day']='2026-01-14'
    with pytest.raises(ValueError,match='OUTSIDE_STUDY_ENVELOPE'):validate_plan(bad)
