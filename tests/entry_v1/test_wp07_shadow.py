import json
import os
from pathlib import Path

import numpy as np
import pytest

from wp07_fixture import ARTIFACTS, f3_source


@pytest.fixture(scope='module')
def actor():
    import jax.numpy as jnp
    from gym_trading_env.research.fg_s3_e_quality_actor import create_agent
    a=create_agent(f3_source(),ARTIFACTS/'B_model',controlled_small=True)
    # Declared synthetic Actor initialization, to exercise both native exits.
    # It is not a trained policy result and never changes after audit begins.
    for k in a.params:
        if k.startswith('quality_actor/') and k.endswith('/kernel') and a.params[k].shape[-1]==2:
            a.params[k]=jnp.zeros_like(a.params[k])
        if k.startswith('quality_actor/') and k.endswith('/bias') and a.params[k].shape==(2,):
            a.params[k]=jnp.array([0.,2.],dtype=a.params[k].dtype)
    a.sync_policy_from_train()
    return a


@pytest.mark.parametrize('start',[0,330])
def test_native_shadow_teacher_raises_full_replay_and_frozen_model(actor,monkeypatch,start):
    from wp02_fixture import native_factory
    from gym_trading_env.research.fg_s3_e_quality_online_collector import OnlineCollector
    from gym_trading_env.research.fg_replay import reconcile_trace,E_QUALITY_SCHEMA,digest,write_json
    from gym_trading_env.research.fg_single3_evaluation import embodied_runtime
    from dreamerv3.fg_s3a_actor import parameter_digests
    import pandas as pd
    root=Path(os.environ.get('FG_WP07_B_ROOT',ARTIFACTS/'B_shadow_01'))/str(start)
    factory,_,profile=native_factory(root/'market',minutes=600,chunk=19)
    import yaml
    directions=root/'market/directions.yaml'
    d=yaml.safe_load(directions.read_text())
    for row in d['intervals']:row['direction']='SHORT'
    directions.write_text(yaml.safe_dump(d))
    path=root/'market/market.csv';frame=pd.read_csv(path)
    frame['High']=[1035 if i%17==3 else 1002 for i in range(len(frame))]
    frame['Low']=[990 if i%13==4 and i%17!=3 else 999 for i in range(len(frame))]
    frame.to_csv(path,index=False)
    profile.source_receipt['sources']['CZCE.FG2701']['market']['sha256']=digest(path)
    def forbidden(*args,**kwargs):raise AssertionError('Teacher consulted by live execution')
    monkeypatch.setattr('gym_trading_env.research.fg_s3_e_quality_strict.QualityStrictTeacher',forbidden)
    monkeypatch.setattr('gym_trading_env.research.fg_s3_quality_teacher_v2.NativeQualityTeacher',forbidden)
    env=factory();initial=None
    if start:
        for _ in range(start):
            env.step(0)
            if env.truncated:env.reset()
        initial=env.export_state()
    before=parameter_digests(actor.params)
    c=OnlineCollector(env,dict(fold='TRAIN',scope='BUILTIN_SYNTHETIC_ONLY'),root/'rollout',
        Path(__file__).resolve().parents[3]/'dreamerv3',agent=actor,seed=202,budget=128,
        capacity=512,initial_state=initial,replay_length=13,chunksize=32)
    # Fail on *any* attempt to use the old Oracle state/query route.
    c.oracle.decision_state=forbidden;c.oracle.query=forbidden
    try:
        c.advance(128)
        manifest=c.snapshot_trace(root/'trace',dict(scope='SYNTHETIC_ACTOR_BIAS_FIXTURE_NO_RESEARCH'))
        rows=c.rows
        assert all(not any(k.startswith(('teacher','oracle')) for k in row.get('policy_diagnostics',{})) for row in rows)
        assert all(not row['action'] or row['legal_add'] for row in rows)
        assert c.counts['tp_events'] and c.counts['sl_events']
        assert any(r['action']==0 for r in rows) and any(r['action']==1 for r in rows)
        assert start==0 or len({r['trading_date'] for r in rows})==2
        result=reconcile_trace(root/'trace',Path(__file__).resolve().parents[3]/'dreamerv3',
            env_factory=factory,schema=E_QUALITY_SCHEMA,runtime_factory=embodied_runtime)
        assert result['status']=='PASS' and result['account_restores']>0
        assert before==parameter_digests(actor.params)
        write_json(root/'evidence.json',dict(replay=result,counts=dict(c.counts),manifest=manifest,
            all_model_and_optimizer_bytes_equal=True,teacher_calls=0,scope='SYNTHETIC_ONLY',real_TRAIN_updates=0))
        Path(f'docs/receipts/wp07/B_shadow_{start}.json').write_text(json.dumps(dict(
            evidence=str(root/'evidence.json'),replay=result,counts=dict(c.counts)),indent=2)+'\n')
    finally:c.close()
