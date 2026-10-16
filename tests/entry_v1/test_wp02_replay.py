from pathlib import Path
import json
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
    # Fresh real Agent consumes the persisted native bytes, not fabricated shapes.
    import sys
    sys.path.insert(0,str(dreamer/'embodied/tests'))
    from test_fg_s3_e_model import model
    agent=model(tmp_path/'agent')
    agent.sync_policy_from_train()
    step=next(step for _,step in rows if step['market_macd_5m'][3]==1)
    obs={k:np.stack([step[k]]) for k in agent.obs_space}
    saved=agent.save_exact()
    carry,_,out=agent.policy(agent.init_policy(1),obs,mode='eval')
    agent.load_exact(saved)
    changed={**obs,'market_macd_5m':np.array([[5.,2.,3.,1.]],np.float32)}
    other,_,out2=agent.policy(agent.init_policy(1),changed,mode='eval')
    # A categorical sample can stay in the same bin. Prove sensitivity in
    # the actual Agent's continuous Encoder and RSSM posterior logits.
    import ninjax as nj
    import jax.numpy as jnp
    def forward(o):
        reset=jnp.zeros((1,),bool)
        _,_,tokens=agent.model.enc({},o,reset,False,True)
        _,_,feat=agent.model.dyn.observe(agent.model.dyn.initial(1),tokens,
            {'action':jnp.zeros((1,),jnp.int32)},reset,False,True)
        return tokens,feat['logit']
    call=nj.pure(forward)
    _,a=call(agent.params,{k:jnp.asarray(v) for k,v in obs.items()},seed=11)
    _,b=call(agent.params,{k:jnp.asarray(v) for k,v in changed.items()},seed=11)
    assert not np.array_equal(a[0],b[0])
    assert not np.array_equal(a[1],b[1])
    assert np.isfinite(out['p_add']).all() and np.isfinite(out2['p_add']).all()
    invalid={**obs,'market_macd_5m':np.zeros((1,4),np.float32)}
    polluted={**invalid,'market_macd_5m':np.array([[999.,-999.,100.,0.]],np.float32)}
    _,zero=call(agent.params,{k:jnp.asarray(v) for k,v in invalid.items()},seed=11)
    _,ignored=call(agent.params,{k:jnp.asarray(v) for k,v in polluted.items()},seed=11)
    for x,y in zip(zero,ignored):np.testing.assert_array_equal(x,y)
    agent.load_exact(saved)
    _,_,invalid_out=agent.policy(agent.init_policy(1),invalid,mode='eval')
    assert np.isfinite(invalid_out['p_add']).all()
    with pytest.raises(AssertionError):
        agent.policy(agent.init_policy(1),{**obs,'teacher_score':np.array([100.],np.float32)},mode='eval')
    with pytest.raises(ValueError,match='Incompatible'):trace_manifest(root)
    # An honestly hashed manifest cannot conceal changed replay MACD bytes.
    p=root/manifest['chunks'][0]['path']
    with np.load(p) as f:data={k:f[k] for k in f.files}
    data['market_macd_5m'][0,0]=3
    np.savez_compressed(p,**data)
    with pytest.raises(ValueError,match='Frozen'):reconcile_trace(root,dreamer,env_factory=factory,schema=E_QUALITY_SCHEMA)
