from pathlib import Path

import pandas as pd
import pytest

from gym_trading_env.research.fg_replay import (SINGLE3_SCHEMA,collect_trace,reconcile_trace,iter_sidecar,
    write_json,signed,digest)
from gym_trading_env.research import fg_replay_coverage as base
from gym_trading_env.research import fg_single3_coverage as coverage
from gym_trading_env.research.fg_single3_dataset import pool_members
from gym_trading_env.research.fg_single3_replay import Single3BehaviorPolicy
from test_fg_single3_env import single_env


def test_real_native_sample_new_labels_and_single_reward_pool(tmp_path):
    root=Path(__file__).resolve().parents[3]/'dreamerv3'
    prices=[(1000,1000,1000,1000)]*80
    prices[10]=(1000,1000,960,980)
    prices[20]=(1000,1007,1000,1007)
    times=pd.date_range('2026-01-05 21:01',periods=80,freq='min')
    records=[]
    for candidate in ('C0','C2'):
        def factory(candidate=candidate):
            folder=tmp_path/('env_'+candidate);folder.mkdir(exist_ok=True)
            return single_env(folder,prices,candidate=candidate,chunk=7,times=times,
                calendar=[dict(start='2026-01-05 21:00',end='2026-01-05 23:00')])
        spec=dict(id='P1',kind='always_add',seed_components=[0,0,1])
        directory=tmp_path/candidate
        m=collect_trace(tmp_path/'unused.yaml',spec,directory,dict(test='coverage'),root,
            env_factory=factory,protocol='fg_single3_v1',policy_factory=Single3BehaviorPolicy)
        receipt=reconcile_trace(directory,root,env_factory=factory,schema=SINGLE3_SCHEMA)
        write_json(directory/'reconciliation.json',signed(dict(receipt,trace_manifest_sha256=digest(directory/'manifest.json'))))
        records.append(dict(directory=candidate,candidate=candidate,policy_id='P1',contract=m['contract'],
            sequence_starts=m['sequence_starts'],reward_params_hash=m['reward_params_hash']))
    dataset=dict(schema=SINGLE3_SCHEMA,traces=records,views={'C2_natural':'C2/replay'},pool_members={'C2_natural':[1]})
    corpus=base.corpus_statistics(tmp_path,dataset,columns=coverage.COLUMNS,label_fn=coverage.row_labels,
        trace_schema=SINGLE3_SCHEMA,group_fn=coverage.group,row_hook=coverage.row_hook,trade_finalize=coverage.finalize_trade)
    sample=base.sample_statistics(tmp_path,dataset,corpus,'C2_natural',batches=3,batch_size=16,
                                  columns=coverage.COLUMNS,group_fn=coverage.group)
    assert sample['first_batch_reproduced'] and sample['discontinuous_sequences']==0
    assert set(sample['sampled_policy_sequences'])=={'C2_P1'}
    assert sample['counts'].shape==(3,2,len(coverage.COLUMNS))
    assert corpus['summaries']['C2_P1']['sl']==1
    assert corpus['summaries']['C2_P1']['tp']==1
    assert {t['status'] for t in corpus['trades']} >= {'stop_loss','success','right_censored'}
    assert sum(t['first_adverse_effective_bar']==10 for t in corpus['trades'])==2
    sl=next(r for r in iter_sidecar(tmp_path/'C2') if r['exits'] and r['exits'][0]['reason']=='STOP_LOSS')
    labels=coverage.row_labels(sl)
    assert labels[coverage.INDEX['sl']]==1 and labels[coverage.INDEX['tp']]==0


def test_pool_members_exclude_wait_and_cross_reward():
    records=[dict(candidate=c,policy_id=p) for c in ('C0','C1','C2','C3') for p in ('P1','P2a','P2b','P3','P4')]
    records.append(dict(candidate='C2',policy_id='P0'))
    pools=pool_members(records)
    assert len(pools)==29
    assert all(records[i]['policy_id']!='P0' for i in pools['C2_natural_oracle'])
    assert {records[i]['candidate'] for i in pools['C1_natural']}=={'C1'}
    assert {records[i]['policy_id'] for i in pools['C2_natural']}=={'P1','P2a','P2b','P3'}
