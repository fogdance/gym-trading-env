"""Internal spawn cache transport retains source/financial/causal ownership."""
from dataclasses import replace, FrozenInstanceError

import numpy as np
import pytest

from gym_trading_env.envs.entry_market_cache import EntryMarketCache
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from gym_trading_env.research.fg_s3_e_quality_corpus_cli import _factory
from wp02_fixture import native_factory


def test_sealed_seed_avoids_feature_rebuild_and_retains_cold_restore_and_independent_owners(tmp_path,monkeypatch):
    plain_factory,config,profile=native_factory(tmp_path/'market',minutes=190,chunk=7)
    factory=_factory(dict(env_config=str(config)),profile)
    owner=factory();owner.reset(seed=0)
    seed=factory._market_cache.export_seed()
    assert type(seed.payload) is bytes
    with pytest.raises(FrozenInstanceError):seed.payload=b'changed'
    with pytest.raises(ValueError,match='seed content changed'):
        EntryMarketCache(1,seed=replace(seed,payload=seed.payload+b'changed'))
    expected=plain_factory();expected.reset(seed=0)
    # Seed came from pristine cache storage, not this mutable source account.
    owner.step(1)
    owner._raw.iloc[-1,owner._raw.columns.get_loc('Volume')]+=1
    owner._features[-1,0]+=100
    def forbidden(*a,**kw):raise AssertionError('Spawn seed rebuilt pure market features')
    monkeypatch.setattr(FixedDirectionEntryEnv,'_load_market',forbidden)
    seeded=_factory(dict(env_config=str(config)),profile,market_seed=seed)
    a,b=seeded(),seeded();a.reset(seed=0);b.reset(seed=0)
    try:
        assert seeded._market_cache.hits==2 and seeded._market_cache.misses==0
        assert not np.shares_memory(a._features,b._features)
        assert not np.shares_memory(a._e_market.obs_array,b._e_market.obs_array)
        assert a._e_market is not b._e_market
        state=None
        for cursor in range(181):
            if cursor in (0,80,174,175,180):
                assert a.export_state()==expected.export_state()
                for key,value in expected._get_obs().items():
                    np.testing.assert_array_equal(a._get_obs()[key],value)
                if cursor==80:state=a.export_state()
            if cursor<180:
                if a.truncated:a.reset();expected.reset()
                action=int(bool(a._published_mask[1]) and cursor%11==0)
                x,y=a.step(action),expected.step(action)
                assert x[1:4]==y[1:4]
        assert a.fills and a.reward_events
        b.import_state(state);a.import_state(state);expected.import_state(state)
        assert b.export_state()==a.export_state()==expected.export_state()
        snapshot=b.export_state();a.step(0)
        assert b.export_state()==snapshot
    finally:owner.close();a.close();b.close();expected.close()


def test_seed_hit_still_hashes_and_authenticates_current_csv(tmp_path):
    _,config,profile=native_factory(tmp_path/'market',minutes=20)
    factory=_factory(dict(env_config=str(config)),profile)
    env=factory();seed=factory._market_cache.export_seed()
    path=config.parent/'market.csv';path.write_text(path.read_text()+'\n')
    try:
        with pytest.raises(ValueError,match='E admitted market SHA256 mismatch'):
            _factory(dict(env_config=str(config)),profile,market_seed=seed)()
    finally:env.close()


def test_seed_preserves_visible_prefix_feature_builder_for_delayed_arrivals(tmp_path):
    import hashlib
    import pandas as pd
    from e_data_v1.schema import Decision
    plain,config,profile=native_factory(tmp_path/'market',minutes=31)
    path=config.parent/'market.csv';frame=pd.read_csv(path)
    frame['available_at']=frame.Date
    frame.loc[14,'available_at']=frame.Date.iloc[16]
    frame.to_csv(path,index=False)
    profile.source_receipt['sources']['CZCE.FG2701']['market']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    factory=_factory(dict(env_config=str(config)),profile)
    a=factory()
    seeded=_factory(dict(env_config=str(config)),profile,market_seed=factory._market_cache.export_seed())
    b=seeded();expected=plain()
    try:
        counts=[]
        assert b._e_market.delayed
        for cursor in (14,15,16):
            decision=Decision(b.contract,b._raw.index[cursor])
            actual=b._e_market.snapshot(decision)
            original=expected._e_market.snapshot(decision)
            for left,right in zip(actual[:5],original[:5],strict=True):
                np.testing.assert_array_equal(left,right)
            assert actual[5]==original[5]
            counts.append(len(actual[5]['context_bucket_ids']))
        assert counts==[0,0,1]
        assert seeded._market_cache.hits==1 and seeded._market_cache.misses==0
    finally:a.close();b.close();expected.close()


def test_seeded_spawn_workers_match_original_complete_teacher_and_full_account_sha(tmp_path):
    from gym_trading_env.research.fg_s3_quality_teacher_v2 import NativeQualityTeacher
    from gym_trading_env.research.fg_s3_e_quality_corpus import l1_labeler
    from gym_trading_env.research.fg_s3_quality_label_pool import NativeLabelPool
    _,config,profile=native_factory(tmp_path/'market',minutes=25,chunk=7)
    factory=_factory(dict(env_config=str(config)),profile)
    env=factory();env.reset(seed=0)
    source=dict(fold='TRAIN',state_cache_entries=0)
    teacher=NativeQualityTeacher(env,factory,fold='TRAIN',source=source)
    expected=l1_labeler(teacher)
    pool=NativeLabelPool(dict(env_config=str(config)),profile,env.export_state(),source,None,
        teacher.identity_hash,workers=2,market_seed=factory._market_cache.export_seed())
    try:
        for _ in range(6):
            before=env.export_state()
            assert pool(env,None,None).result()==expected(env,None,None)
            assert env.export_state()==before
            env.step(0)
        stats=pool.report_statistics()
        assert stats['market_seed_bytes']>0
        assert all(w['market_cache']['hits']>=2 and w['market_cache']['misses']==0 for w in stats['native_workers'])
    finally:pool.close();teacher.close();env.close()
