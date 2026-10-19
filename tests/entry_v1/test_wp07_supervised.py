import copy
import json
from pathlib import Path

import numpy as np
import pytest

from test_wp07_data import live
from wp07_fixture import f3_source,execution


@pytest.fixture(scope='module')
def online(live):
    from gym_trading_env.research.fg_s3_e_quality_actor import create_agent
    from embodied.run.fg_s3_e_quality_online import inspect_f3
    from embodied.run.fg_s3_e_quality_transfer import transfer_parameters
    root,c,p,r=live;p.publish()
    settings=copy.deepcopy(f3_source());settings['batch_size']=r.settings['batch_size']
    settings['model_init']['seed']=303
    a=create_agent(settings,root/'supervised_model',controlled_small=True,execution_mode='ONLINE_SUPERVISED')
    proof,state=inspect_f3(execution(root/'unused'),allow_engineering=True)
    transfer_parameters(a,state,proof,target_stage='ACTOR_ONLINE')
    return root,a,r


def test_real_online_l2_updates_only_original_actor_and_slots(online):
    from test_wp06_updates import train
    from dreamerv3.fg_s3a_actor import parameter_digests
    root,a,r=online;before=parameter_digests(a.params);exposure=dict(add=0,wait=0)
    for i in range(4):
        data,_=r.sample();_,_,m=train(a,data,seed=202+i)
        assert m['f3/updated'] and m['f3/finite']
        exposure['add']+=int(m['f3/natural_add']);exposure['wait']+=int(m['f3/natural_wait'])
    after=parameter_digests(a.params);changed=[k for k in before if before[k]!=after[k]]
    assert all(exposure.values())
    assert any(k.startswith('quality_actor/') for k in changed)
    assert any(k.startswith('opt/state/') and 'qualityactor' in k for k in changed)
    assert all(before[k]==after[k] for k,owner in a.model.f3_state_owner.items() if owner is None)
    assert not any(k.startswith(('pol/','val/','slowval/','enc/','dyn/','quality_q_')) for k in changed)
    Path('docs/receipts/wp07/D_actor_delta.json').write_text(json.dumps(dict(
        scope='SYNTHETIC_NATIVE_POST_STEP_L2',updates=4,exposure=exposure,changed=changed,
        before=before,after=after,all_non_actor_owners_and_slots_frozen=True),indent=2)+'\n')


def test_no_label_and_nonfinite_freeze_inherited_optimizer_state(online):
    from test_wp06_updates import train
    from dreamerv3.fg_s3a_actor import parameter_digests
    _,a,r=online;data,_=r.sample();before=parameter_digests(a.params)
    invalid={**data,'teacher_valid':np.zeros_like(data['teacher_valid'])}
    _,_,m=train(a,invalid);assert not m['f3/updated'] and before==parameter_digests(a.params)
    # Input NaNs are already rejected by the native Encoder. Exercise a
    # finite-input, overflowing optimizer candidate and its complete rollback.
    import jax.numpy as jnp
    healthy=a.save_exact()
    for k in a.params:
        if k.startswith('quality_actor/'):a.params[k]=jnp.full_like(a.params[k],1e38)
    overflowing=parameter_digests(a.params)
    try:
        with pytest.raises(FloatingPointError,match='F3_NONFINITE'):train(a,data)
        assert overflowing==parameter_digests(a.params)
    finally:a.load_exact(healthy)


def test_future_targets_never_change_actor_features_and_shadow_rejects_training(online):
    from dreamerv3.fg_s3_e_quality_actor import read_only_forward
    from gym_trading_env.research.fg_s3_e_quality_actor import create_agent
    from test_wp06_updates import train
    root,a,r=online;data,_=r.sample();out=read_only_forward(a,data)
    changed=copy.deepcopy(data);changed['teacher_action']=1-changed['teacher_action']
    changed['quality_score'][:]=99
    other=read_only_forward(a,changed)
    np.testing.assert_array_equal(out['features'],other['features'])
    np.testing.assert_array_equal(out['logits'],other['logits'])
    shadow=create_agent(f3_source(),root/'shadow_model',controlled_small=True,execution_mode='SHADOW')
    with pytest.raises(ValueError,match='SHADOW_OPTIMIZER'):train(shadow,data)
