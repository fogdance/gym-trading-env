import numpy as np
import pytest

from gym_trading_env.research.fg_replay import (SINGLE3_SCHEMA,collect_trace,
    reconcile_trace,trace_manifest,iter_sidecar,iter_replay_rows)
from gym_trading_env.research.fg_single3_replay import Single3BehaviorPolicy
from test_fg_single3_env import single_env


def test_native_single_delay_and_derived_reward_view(tmp_path):
    prices=[(1000,1000,1000,1000)]*44
    prices[25]=(1000,1000,960,980)
    def factory(candidate):
        folder=tmp_path/candidate
        folder.mkdir(exist_ok=True)
        return single_env(folder,prices,candidate=candidate,chunk=7)
    root='/home/v/Documents/work/dreamerv3'
    source=tmp_path/'source'
    spec=dict(id='P3',kind='single_delay',wait_minutes=15,seed_components=[0,0,4])
    manifest=collect_trace(tmp_path/'C2/env.yaml',spec,source,dict(test='single3'),root,
        env_factory=lambda:factory('C2'),protocol='fg_single3_v1',policy_factory=Single3BehaviorPolicy)
    receipt=reconcile_trace(source,root,env_factory=lambda:factory('C2'),schema=SINGLE3_SCHEMA)
    assert receipt['status']=='PASS' and manifest['tp_events']==0 and manifest['sl_events']==1
    assert manifest['open_lots']==3 and manifest['chunk_resets']>0
    rows=list(iter_sidecar(source))
    assert [r['cursor'] for r in rows if any(f['reason']=='ENTRY' for f in r['fills'])]==[16,41]
    with pytest.raises(ValueError,match='Incompatible'):
        trace_manifest(source)  # Legacy consumers must refuse new same-shape data.
    derived=tmp_path/'derived'
    derived_spec=dict(spec,source_directory=str(source),source_trace_hash=manifest['replay_trace_hash'])
    collect_trace(tmp_path/'C3/env.yaml',derived_spec,derived,dict(test='single3_C3'),root,
        env_factory=lambda:factory('C3'),protocol='fg_single3_v1',policy_factory=Single3BehaviorPolicy)
    assert reconcile_trace(derived,root,env_factory=lambda:factory('C3'),schema=SINGLE3_SCHEMA)['status']=='PASS'
    original=list(iter_replay_rows(source,manifest))
    target=list(iter_replay_rows(derived,trace_manifest(derived,schema=SINGLE3_SCHEMA)))
    assert len(original)==len(target)
    for (_,left),(_,right) in zip(original,target):
        for key in left.keys()-{'reward','stepid'}:
            np.testing.assert_array_equal(left[key],right[key])
    assert sum(float(s['reward']) for _,s in target)<sum(float(s['reward']) for _,s in original)


def test_p4_refuses_unproven_schedule_before_execution(tmp_path):
    from gym_trading_env.research.fg_replay import write_json,signed,digest
    schedule=tmp_path/'plan.json'
    write_json(schedule,signed(dict(oracle_optimality_status='heuristic_only',
        candidate='C2',gamma=332/333,add_cursors=[0])))
    with pytest.raises(ValueError,match='exact'):
        Single3BehaviorPolicy(dict(id='P4',kind='oracle_schedule',seed_components=[0],
            schedule_path=str(schedule),schedule_sha256=digest(schedule),candidate='C2'))
