import copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from wp03_fixture import native_corpus
from test_wp06_contract import settings
from gym_trading_env.research.fg_s3_e_quality_actor import validate_settings
from gym_trading_env.research.fg_s3_e_quality_supervision import SupervisionSource,QualitySupervisionAdmission
from gym_trading_env.research.fg_s3_e_quality_reader import build_join_index
from embodied.run.fg_s3_e_quality_actor import ActorStreams

@pytest.fixture(scope='module')
def indexed(tmp_path_factory):
    root=tmp_path_factory.mktemp('wp06_sampling');corpus=native_corpus(root/'corpus',chunk=40)
    admission=QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(corpus,stage='ACTOR_SUPERVISED'))
    index=root/'join.sqlite';build_join_index(admission,index)
    return admission,index,settings(corpus)

def stream(indexed,mode):
    admission,index,s=indexed;s=copy.deepcopy(s)
    s.update(target_steps=8,burnin_steps=4,batch_size=2)
    if mode:
        spec=dict(mode=mode)
        if mode!='NATURAL':spec['strata_weights']={'STRICT_ADD':1,'STRICT_WAIT':1}
        if mode=='MIXED':spec['natural_weight']=.5
        s['aux_sampling_spec']=spec
    return ActorStreams(SimpleNamespace(settings=validate_settings(s),source=admission),index)

@pytest.mark.parametrize('mode',[None,'NATURAL','STRATIFIED','MIXED'])
def test_D08_original_readers_roles_rng_restore_and_bounded_cache(indexed,mode):
    a=stream(indexed,mode);b=stream(indexed,mode)
    try:
        before=a.save();state=copy.deepcopy(before);x=next(a);a.load(state);y=next(a);z=next(b)
        for key in x:np.testing.assert_array_equal(x[key],y[key]);np.testing.assert_array_equal(x[key],z[key])
        assert a.save()==b.save() and state==before
        assert x['stepid'].shape==((4 if mode else 2),13,20)
        assert x['f3_natural'][:2].all()
        if mode:assert not x['f3_natural'][2:].any()
        assert len({r.report['source_identity'] for r in a.readers})==1
        coverage=a.coverage()
        assert all(c['actual_optimizer_exposures']==0 and c['peak_cached_bytes']<=c['cache_budget_bytes'] for c in coverage.values())
        assert all(c['delivered_unique_market_cursors']<=c['delivered_unique_account_decisions']+4 for c in coverage.values())
        Path('docs/receipts/wp06/D_sampling_'+str(mode)+'.json').write_text(json.dumps(dict(roles=a.roles,coverage=coverage,state=a.save(),shape=list(x['stepid'].shape),updates=0),indent=2))
    finally:a.close();b.close()

def test_D09_auxiliary_failure_restores_both_consumption_states(indexed,monkeypatch):
    a=stream(indexed,'MIXED')
    try:
        before=a.save()
        def fail():raise ValueError('injected auxiliary failure')
        monkeypatch.setattr(a.streams[1].adapter,'sample',fail)
        with pytest.raises(ValueError,match='auxiliary failure'):next(a)
        assert a.save()==before
        bad=copy.deepcopy(before);bad['roles'].reverse()
        with pytest.raises(ValueError,match='GROUP_IDENTITY_MISMATCH'):a.load(bad)
        assert a.save()==before
    finally:a.close()

def test_D10_missing_auxiliary_support_is_no_go_not_resampled(indexed):
    admission,index,s=indexed;s=copy.deepcopy(s)
    s['aux_sampling_spec']=dict(mode='STRATIFIED',strata_weights={'NOT_A_CLASS':1})
    with pytest.raises(ValueError,match='INSUFFICIENT_CURRICULUM_SUPPORT'):
        ActorStreams(SimpleNamespace(settings=validate_settings(s),source=admission),index)


def test_D14_labels_only_in_burnin_do_not_count_as_target_support(indexed):
    admission,index,s=indexed;s=copy.deepcopy(s)
    s.update(target_steps=1,burnin_steps=29,sampling_spec=dict(mode='NATURAL',policies=['P0']))
    with pytest.raises(ValueError,match='F3_TARGET_L2'):
        ActorStreams(SimpleNamespace(settings=validate_settings(s),source=admission),index)
