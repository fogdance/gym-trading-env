import copy,json
from pathlib import Path
import jax.numpy as jnp
import pytest
from wp06_fixture import engineering_settings
from gym_trading_env.research.fg_s3_e_quality_actor import preflight
from gym_trading_env.research.fg_s3_e_quality_reader import build_join_index
from embodied.run.fg_s3_e_quality_actor import ActorRun,engineering_permit
from dreamerv3.fg_s3a_actor import parameter_digests

@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    root=tmp_path_factory.mktemp('wp06_run');s=engineering_settings(root)
    s['aux_sampling_spec']=dict(mode='MIXED',natural_weight=.5,strata_weights={'STRICT_ADD':1,'STRICT_WAIT':1})
    s['actor_loss'].update(balanced=.2,strict_pair=.1,quality_pair=.1)
    p=preflight(s);index=root/'join.sqlite';build_join_index(p.source,index)
    return root,p,index

def test_D11_real_original_loop_two_readers_and_actor_only_updates(setup):
    root,p,index=setup;run=ActorRun(p,index,root/'active',_probe=engineering_permit(p))
    before=parameter_digests(run.agent.params);result=run.run(2);after=parameter_digests(run.agent.params)
    assert result['counters']['optimizer_updates']==2 and result['counters']['consumed_batches']>=2
    assert result['counters']['exposures']['natural_count']>0 and result['counters']['exposures']['auxiliary_count']>0
    assert all(before[k]==after[k] for k,v in run.agent.model.f3_state_owner.items() if v is None)
    assert any(before[k]!=after[k] for k in before if k.startswith('quality_actor/'))
    assert not result['training_admitted'] and not result['model_research_go']
    from gym_trading_env.research.fg_replay import write_json,signed
    write_json(Path('docs/receipts/wp06/D_actor_run.json'),signed(result))

def test_D12_consumption_bound_and_all_invalid_do_not_loop_or_update(setup,tmp_path,monkeypatch):
    _,p,index=setup;s=copy.deepcopy(p.settings);s['max_consumed_batches']=4;p=preflight(s)
    run=ActorRun(p,index,tmp_path/'skip',_probe=engineering_permit(p))
    before=parameter_digests(run.agent.params);original=run.agent.train
    def invalid(carry,data):
        data={**data,'teacher_valid':jnp.zeros_like(data['teacher_valid'])}
        return original(carry,data)
    monkeypatch.setattr(run.agent,'train',invalid)
    with pytest.raises(ValueError,match='CONSUMPTION_BUDGET_EXHAUSTED'):run.run()
    assert run.counters.state['consumed_batches']==4 and int(run.agent.n_updates)==0
    assert run.counters.state['skip_no_valid']==4 and before==parameter_digests(run.agent.params)
    assert run.cp.latest().name=='step_000000000004'

def test_D13_no_permission_no_run_side_effect(setup,tmp_path):
    _,p,index=setup;out=tmp_path/'forbidden'
    with pytest.raises(ValueError,match='AUTHORIZATION_REQUIRED'):ActorRun(p,index,out)
    assert not out.exists()
