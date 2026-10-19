import json
import os
from pathlib import Path

import pytest

from wp08_fixture import ARTIFACTS,native_source,controlled_actor
from test_wp08_contract import plan


def test_native_main_roll_and_unfinished_trade_are_distinct(tmp_path):
    from gym_trading_env.research.fg_s3_e_quality_exam_session import ExamSession
    from gym_trading_env.research.fg_s3_e_quality_exam_compare import ExamBaseline
    from gym_trading_env.research.fg_s3_e_quality_exam_metrics import analyze_trace
    base=Path(os.environ.get('FG_WP08_ROLL_ROOT',ARTIFACTS/'H_roll_01')).absolute()
    source=native_source(base/'source',end_day='2026-01-14')
    p=plan(base);p['end_day']='2026-01-14'
    policy=ExamBaseline('ALWAYS_ADD',seed=202,contract='CZCE.FG2701')
    session=ExamSession(p,source,policy,base/'session',policy_id='ALWAYS_ADD')
    try:
        session.advance();receipt=session.finish()
        data=analyze_trace(base/'session/trace',source,base/'metrics')
        assert receipt['complete_interval'] and receipt['reconciliation']['status']=='PASS'
        assert session.raw.terminated and not session.raw.entries
        assert data['summary']['main_roll']==1 and data['summary']['right_censored']==0
        assert data['trades'][-1]['exit_reason']=='MAIN_ROLL'
        assert data['trades'][-1]['mae_full_valid']
        Path('docs/receipts/wp08/H_native_roll.json').write_text(json.dumps(dict(
            receipt=receipt,summary=data['summary'],last_trade=data['trades'][-1]),indent=2)+'\n')
    finally:session.close()


def test_contracts_start_separate_accounts_carries_and_rng(tmp_path):
    from gym_trading_env.research.fg_s3_e_quality_exam_session import ExamSession
    from gym_trading_env.research.fg_replay import iter_sidecar
    base=Path(os.environ.get('FG_WP08_CONTRACT_ROOT',ARTIFACTS/'H_contracts_01')).absolute()
    p=plan(base);p['contracts']=['CZCE.FG2701','CZCE.FG2705']
    p['contract_windows']={'CZCE.FG2701':dict(start_day='2026-01-08',end_day='2026-01-08'),
        'CZCE.FG2705':dict(start_day='2026-01-09',end_day='2026-01-09')}
    evidence=[]
    for contract in p['contracts']:
        source=native_source(base/contract/'source',contract=contract,**p['contract_windows'][contract])
        actor=controlled_actor(tmp_path/contract)
        assert actor.save()['inference']['actions']==0
        s=ExamSession(p,source,actor,base/contract/'session',policy_id='F3_FIXED_SYNTHETIC')
        try:
            assert s.driver.carry is None and not s.raw.entries
            assert s.raw.config.trading.currency_pair==contract.split('.')[1]
            s.advance(8);receipt=s.finish();rows=list(iter_sidecar(s.root/'trace'))
            assert rows[0]['is_first'] and rows[0]['lots_before']==0
            assert all(r['e_snapshot']['source_contract']==contract for r in rows)
            evidence.append(dict(contract=contract,owner=rows[0]['account_owner'],
                inference_actions=actor.save()['inference']['actions'],receipt=receipt))
        finally:s.close()
    assert evidence[0]['owner']!=evidence[1]['owner']
    assert evidence[0]['inference_actions']==evidence[1]['inference_actions']==9
    Path('docs/receipts/wp08/H_contract_isolation.json').write_text(json.dumps(evidence,indent=2)+'\n')


def test_declared_actor_actions_reproduce_and_legal_action_tamper_is_rejected(tmp_path,monkeypatch):
    import numpy as np
    from gym_trading_env.research import fg_s3_e_quality_exam_verify as verify
    from gym_trading_env.research.fg_s3_e_quality_exam_compare import policy_jobs
    root=ARTIFACTS/'H_cli_02/exam';p=json.loads((root/'PLAN.json').read_text())
    job=next(j for j in policy_jobs(p) if j['policy']=='F3_FROZEN')
    case=root/job['contract']/job['label']
    assert verify.verify_policy(p,job,case,tmp_path/'model')['records']==346
    original=verify.iter_replay_rows
    def mutated(*args,**kwargs):
        changed=False
        for i,step in original(*args,**kwargs):
            if not changed and step['action_mask'][1] and not step['is_last']:
                step=dict(step);step['action']=np.int32(1-int(step['action']));changed=True
            yield i,step
    monkeypatch.setattr(verify,'iter_replay_rows',mutated)
    with pytest.raises(ValueError,match='ACTION_TRACE_MISMATCH'):verify.verify_policy(p,job,case,tmp_path/'mutant_model')
