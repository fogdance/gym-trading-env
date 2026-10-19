import copy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from test_wp07_data import live
from wp07_fixture import f3_source,execution


@pytest.fixture(scope='module')
def rl(live):
    from gym_trading_env.research.fg_s3_e_quality_actor import create_agent
    from embodied.run.fg_s3_e_quality_online import inspect_f3
    from embodied.run.fg_s3_e_quality_transfer import transfer_parameters
    root,c,p,r=live;p.publish();s=copy.deepcopy(f3_source());s['batch_size']=4
    a=create_agent(s,root/'rl_model',controlled_small=True,execution_mode='ONLINE_RL_CONTROLLED')
    proof,state=inspect_f3(execution(root/'unused'),allow_engineering=True)
    transfer_parameters(a,state,proof,target_stage='ACTOR_ONLINE')
    data,_=r.sample()
    return root,a,data


def test_real_imaginary_same_actor_and_legacy_pol_independent(rl):
    from dreamerv3.fg_s3_e_quality_foundation import causal_context
    _,a,data=rl
    def fn(data):
        carry,obs,prev,ids=causal_context(a.model,data,4,8)
        wm=a.model.world_model(carry,obs,prev,False,diagnostics=True)
        rep=wm[7];mask=obs['action_mask']
        real,diagnostics=a.model.policy_distribution(rep,mask,2)
        imag=a.model._imagination_distribution(rep,mask,2)
        return dict(real=real['action'].logits,imag=imag['action'].logits,
            con_target=wm[-1]['f1_raw/continue_target'],discount=a.model._feedback_discount(obs))
    def forward():
        _,out=nj.pure(fn)(a.params,{k:jnp.asarray(v) for k,v in data.items()},seed=202,create=False,modify=True)
        return jax.tree.map(np.asarray,out)
    first=forward();np.testing.assert_array_equal(first['real'],first['imag'])
    # Changing old pol cannot affect either actual policy route.
    old={k:v for k,v in a.params.items() if k.startswith('pol/')}
    for k in old:a.params[k]=jnp.full_like(a.params[k],100.)
    other=forward();np.testing.assert_array_equal(first['real'],other['real'])
    a.params.update(old)
    gamma=np.float32(332/333)
    np.testing.assert_allclose(first['discount'],gamma**data['market_advance'][:,5:],rtol=1e-6)
    np.testing.assert_allclose(first['con_target'],(~data['is_terminal'][:,5:])*first['discount'],rtol=1e-6)


def test_native_rl_updates_quality_actor_value_not_old_pol_or_wm(rl):
    from test_wp06_updates import train
    from dreamerv3.fg_s3a_actor import parameter_digests
    root,a,data=rl
    before=parameter_digests(a.params)
    _,_,off=train(a,data)
    assert not off['f4/rl_active'] and off['f4/value_loss']==0
    after_off=parameter_digests(a.params)
    protected=('val/','slowval/','slowval_count/','retnorm/','valnorm/','advnorm/')
    assert all(before[k]==after_off[k] for k in before if k.startswith(protected))
    # Explicit SYNTHETIC numeric branch probe, not any formal gate admission.
    a.set_avail_actor_enabled(True)
    before=parameter_digests(a.params)
    rows=[]
    for i in range(2):
        _,_,m=train(a,data,seed=202+i);rows.append({k:np.asarray(v).tolist() for k,v in m.items()})
        assert m['f4/rl_active'] and m['f3/finite'] and m['f3/updated']
        assert np.isfinite(m['f4/value_loss']) and np.isfinite(m['opt/grad_norm'])
        assert m['f4/imag_invalid']==0
    after=parameter_digests(a.params);changed=[k for k in before if before[k]!=after[k]]
    assert any(k.startswith('quality_actor/') for k in changed)
    assert any(k.startswith('val/') for k in changed)
    assert not any(k.startswith(('pol/','enc/','dyn/','dec/','rew/','con/','avail/','quality_event/',
        'quality_touch/','quality_adverse/','quality_score/','quality_q_')) for k in changed)
    assert all(before[k]==after[k] for k,owner in a.model.f3_state_owner.items() if owner is None
        and not k.startswith(protected) and 'val/' not in k)
    Path('docs/receipts/wp07/E_rl_delta.json').write_text(json.dumps(dict(
        scope='SYNTHETIC_EXPLICIT_NUMERIC_PROBE_NOT_FORMAL',before=before,after=after,changed=changed,
        actual_rl_updates=2,metrics=rows,formal_rl_gate=False,wm_research_validated=False),indent=2)+'\n')
