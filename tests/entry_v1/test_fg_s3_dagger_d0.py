from fractions import Fraction
import copy
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from gym_trading_env.research.fg_single3_oracle import CandidateIndex, empty_at, GAMMA, optimize
from gym_trading_env.research.fg_s3_oracle_state_query import OracleStateQuery
from test_fg_single3_env import single_env


DREAMER = Path(__file__).resolve().parents[2].parent/'dreamerv3'


def query_for(env, config_path):
    teacher = FixedDirectionEntryEnv(df=env._raw.copy(), config_path=str(config_path))
    index = CandidateIndex(teacher)
    rows = [index.candidate(t) for t in range(index.flat_end)]
    return OracleStateQuery(env, rows, index.flat_end, fold='TRAIN',
                           source_identity={'kind':'original_env_test_fixture'}), rows


@pytest.mark.parametrize('side', ['LONG', 'SHORT'])
def test_suffix_query_all_cursors_matches_original_env_exhaustive_actions(tmp_path, side):
    prices=[(1000,1000,1000,1000),(1000,1007,1000,1006),
            (1002,1003,991,1001),(1000,1007,995,1007),
            (1001,1003,970,995),(995,1003,994,1000)]
    env=single_env(tmp_path, prices, side=side, candidate='C1', fees=3, chunk=2)
    query, rows=query_for(env,tmp_path/'env.yaml')
    legacy=optimize(rows,query.end,'C1')
    extended=optimize(rows,query.end,'C1',return_suffix=True)
    assert {k:v for k,v in extended.items() if k!='suffix'}==legacy
    for start in range(query.end):
        optimum=Fraction(0)
        for actions in itertools.product((0,1),repeat=query.end-start):
            empty_at(env,start); total=Fraction(0)
            for action in actions:
                if env.truncated:env.reset()
                if env.terminated:break
                _,reward,*_=env.step(action)
                total+=GAMMA**(env.current_step-start-1)*Fraction.from_float(float(np.float32(reward)))
            optimum=max(optimum,total)
        obs=empty_at(env,start); total=Fraction(0)
        while env.current_step<query.end:
            if env.truncated:obs,_=env.reset()
            answer=query.query(env,obs['action_mask'])
            action=answer['action'] if answer['valid'] else 0
            obs,reward,*_=env.step(action)
            total+=GAMMA**(env.current_step-start-1)*Fraction.from_float(float(np.float32(reward)))
        assert total==optimum


def test_query_funding_forced_wait_tie_and_state_identity(tmp_path):
    prices=[(1000,1000,1000,1000),(1000,1000,965,980),
            (1000,1000,1000,1000),(1000,1000,1000,1000)]
    env=single_env(tmp_path,prices,candidate='C1',fees=3)
    query,_=query_for(env,tmp_path/'env.yaml')
    with pytest.raises(ValueError,match='TRAIN'):
        OracleStateQuery(env,[],3,fold='DEV',source_identity={})
    obs=empty_at(env,0)
    before=env.export_state(); answer=query.query(env,obs['action_mask'])
    assert answer['valid'] and answer['action']==0 and env.export_state()==before
    stale=env.export_state(); env.step(1)
    answer=query.query(env,env._published_mask)
    assert not answer['valid'] and answer['reason']=='FUNDS_OUTSIDE_ORIGINAL_CERTIFICATE'
    assert query.query(env,env._published_mask,snapshot=stale)['reason']=='STALE_ACCOUNT_SNAPSHOT'
    assert query.query(env,[1,0])['reason']=='REAL_ACTION_MASK_MISMATCH'
    obs=empty_at(env,len(prices)-1)
    assert query.query(env,obs['action_mask'])['reason']=='CONTRACT_OR_GAP_BOUNDARY'
    obs=empty_at(env,2); env.step(1)
    assert query.query(env,env._published_mask)['reason']=='CONTRACT_OR_GAP_BOUNDARY'


def test_driver_mixing_native_replay_restore_and_no_teacher_leak(tmp_path):
    from gym_trading_env.research.fg_s3_dagger_d0 import D0Collector,join_d0_teacher
    from embodied.run.fg_s3_dagger_policy import replace_prevact
    prices=[(1000,1000,1000,1000),(1000,1000,1000,1000),
            (1000,1007,1000,1007)]*30
    env=single_env(tmp_path,prices,candidate='C1',fees=3,chunk=7,
                   times=pd.date_range('2026-01-05 13:31',periods=90,freq='min'),
                   calendar=[dict(start='2026-01-05 13:30',end='2026-01-05 15:30')],
                   liquidation='2026-01-05 15:30')
    query,_=query_for(env,tmp_path/'env.yaml')
    config=str(tmp_path/'env.yaml')
    def factory():return FixedDirectionEntryEnv(df=env._raw.copy(),config_path=config)
    first=D0Collector(factory(),query,tmp_path/'full',DREAMER,seed=202,beta=1.)
    resumed=None
    try:
        first.advance(40)
        checkpoint=tmp_path/'collection40';first.save_collection(checkpoint)
        state=first.env.export_state()
        resumed=D0Collector(factory(),query,tmp_path/'resumed',DREAMER,seed=202,beta=1.)
        resumed.load_collection(checkpoint)
        assert resumed.env.export_state()==state
        first.advance(75); resumed.advance(75)
        a,b=first.finish(),resumed.finish()
        assert a['semantic_trace_hash']==b['semantic_trace_hash']
        assert a['final_state_id']==b['final_state_id']
        assert [r['stepid'] for r in first.records]==[r['stepid'] for r in resumed.records]
        assert a['counts']['executed_ADD']>0 and a['native_sample']=='PASS'
        assert a['env_steps']==75 and a['counts']['resets']>1
        assert any(r['actor_proposal']!=r['executed_action'] for r in first.records)
        with np.load(first.root/'native_sample.npz') as f:sample={k:f[k] for k in f.files}
        assert not any(k.startswith(('teacher','oracle','q_')) for k in sample)
        joined=join_d0_teacher(sample,first.records)
        assert joined['teacher_valid'].sum()>0
        key=sample['stepid'][0,np.flatnonzero(joined['teacher_valid'][0])[0]].tobytes().hex()
        corrupted=copy.deepcopy(first.records)
        row=next(r for r in corrupted if r['stepid']==key)
        row['teacher_action']=1-row['teacher_action']
        with pytest.raises(ValueError,match='Uncertified'):
            join_d0_teacher(sample,corrupted)
        row['oracle_action']=row['teacher_action']
        with pytest.raises(ValueError,match='Oracle proof'):
            join_d0_teacher(sample,corrupted)
        bad={**sample,'action':1-sample['action']}
        with pytest.raises(ValueError,match='executed action'):
            join_d0_teacher(bad,first.records)
        with pytest.raises(ValueError,match='TRAIN'):
            join_d0_teacher(sample,first.records,fold='DEV')
        original=({}, {}, {}, {'action':[np.array(1,np.int32)]})
        patched=replace_prevact(original,np.array([0],np.int32))
        assert patched[0] is original[0] and patched[-1]['action'][0]==0
        assert original[-1]['action'][0]==1
    finally:
        first.close()
        if resumed:resumed.close()


def test_original_env_rejections_forced_wait_and_gap(tmp_path):
    root=tmp_path/'volume';root.mkdir()
    prices=[(1000,1000,1000,1000)]*4
    env=single_env(root,prices,candidate='C1',fees=3)
    env._raw.loc[env._raw.index[1],'Volume']=2
    query,_=query_for(env,root/'env.yaml')
    obs=empty_at(env,0); before=env.export_state()
    answer=query.query(env,obs['action_mask'])
    assert answer['valid'] and answer['action']==0  # Both requests have zero reward: WAIT tie.
    _,r,*_,info=env.step(1)
    assert r==0 and info['reject_reason']=='INSUFFICIENT_BAR_VOLUME' and not env.fills
    env.import_state(before)
    _,r,*_=env.step(0)
    assert r==0 and not env.fills
    env._raw.loc[env._raw.index[1],'Volume']=100
    assert query.query(env,env._published_mask)['reason']=='RULE_OR_MARKET_IDENTITY_MISMATCH'

    root=tmp_path/'held';root.mkdir()
    env=single_env(root,prices,candidate='C1')
    query,_=query_for(env,root/'env.yaml')
    empty_at(env,0);obs,*_=env.step(1)
    answer=query.query(env,obs['action_mask'])
    assert not answer['valid'] and answer['reason']=='NO_LEARNABLE_DECISION'
    _,_,*_,info=env.step(1)
    assert info['reject_reason']=='MASKED_ADD'

    root=tmp_path/'deadline';root.mkdir()
    env=single_env(root,prices,candidate='C1',liquidation='2026-01-05 09:32',zero_duration_indices=(1,))
    query,_=query_for(env,root/'env.yaml')
    obs=empty_at(env,0)
    assert obs['action_mask'][1] and query.query(env,obs['action_mask'])['action']==0
    _,_,*_,info=env.step(1)
    assert info['reject_reason']=='DIRECTION_OR_DEADLINE_CHANGED'

    root=tmp_path/'gap';root.mkdir()
    env=single_env(root,prices[:3],candidate='C1',times=pd.to_datetime(
        ['2026-01-05 09:31','2026-01-05 09:33','2026-01-05 09:34']))
    query,_=query_for(env,root/'env.yaml')
    obs=empty_at(env,0)
    assert query.end==0 and not query.query(env,obs['action_mask'])['valid']
    _,_,_,truncated,info=env.step(0)
    assert truncated and info['status']=='DATA_GAP' and env.current_step==0


def test_rejected_add_keeps_executed_prevact_and_independent_wait_label(tmp_path):
    from gym_trading_env.research.fg_s3_dagger_d0 import D0Collector,join_d0_teacher
    env=single_env(tmp_path,[(1000,1000,1000,1000)]*4,candidate='C1',
                   liquidation='2026-01-05 09:32',zero_duration_indices=(1,))
    query,_=query_for(env,tmp_path/'env.yaml')
    collector=D0Collector(env,query,tmp_path/'out',DREAMER,mode='scripted',actor_probability=1.)
    try:
        collector.advance(1); receipt=collector.finish()
        row=collector.records[0]
        assert row['executed_action']==1 and row['teacher_valid'] and row['teacher_action']==0
        assert row['execution_result']['reject_reason']=='DIRECTION_OR_DEADLINE_CHANGED'
        assert collector.records[1]['actor_prev_action']==1
        assert receipt['counts']['entries']==0 and receipt['env_steps']==1
        path=next((collector.root/'replay').glob('*.npz'))
        with np.load(path) as f:batch={k:f[k][None] for k in f.files}
        joined=join_d0_teacher(batch,collector.records)
        assert joined['action'][0,0]==1 and joined['teacher_action'][0,0]==0
        assert joined['teacher_valid'][0,0]
    finally:collector.close()
