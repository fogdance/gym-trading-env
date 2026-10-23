from pathlib import Path
import numpy as np
import pytest
from wp02_fixture import native_factory
from gym_trading_env.research.fg_replay import (collect_trace,reconcile_trace,iter_replay_rows,
    trace_manifest,E_QUALITY_PROTOCOL,E_QUALITY_SCHEMA)
from gym_trading_env.research.fg_single3_replay import Single3BehaviorPolicy


def test_B01_native_macd_replay_every_byte_cold_restore(tmp_path):
    factory,path,profile=native_factory(tmp_path/'market')
    root=tmp_path/'trace';dreamer=Path(__file__).resolve().parents[3]/'dreamerv3'
    spec=dict(id='P0',kind='always_wait',seed_components=[7,0,0])
    collect_trace(path,spec,root,dict(fold='CPU_SYNTHETIC'),dreamer,env_factory=factory,
        protocol=E_QUALITY_PROTOCOL,policy_factory=Single3BehaviorPolicy)
    receipt=reconcile_trace(root,dreamer,env_factory=factory,schema=E_QUALITY_SCHEMA)
    assert receipt['status']=='PASS' and receipt['account_restores']>=3
    manifest=trace_manifest(root,schema=E_QUALITY_SCHEMA)
    rows=list(iter_replay_rows(root,manifest))
    assert any(step['market_macd_5m'][3]==1 for _,step in rows)
    assert not any('teacher' in k for _,step in rows for k in step)
    with pytest.raises(ValueError,match='Incompatible'):trace_manifest(root)
    # An honestly hashed manifest cannot conceal changed replay MACD bytes.
    p=root/manifest['chunks'][0]['path']
    with np.load(p) as f:data={k:f[k] for k in f.files}
    data['market_macd_5m'][0,0]=3
    np.savez_compressed(p,**data)
    with pytest.raises(ValueError,match='Frozen'):reconcile_trace(root,dreamer,env_factory=factory,schema=E_QUALITY_SCHEMA)
