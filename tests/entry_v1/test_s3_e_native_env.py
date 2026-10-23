"""Actual Env integration contract; must run on the user's real Gym venv."""
import numpy as np
import pytest

pytest.importorskip('gymnasium')

from types import SimpleNamespace
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from gym_trading_env.envs.entry_market_cache import EntryMarketCache
from test_entry_market_cache import csv_env_config


def _profile(market, identity='c'*64):
    return SimpleNamespace(identity=identity, contracts={'CZCE.FG2701'})


def test_native_e_reuses_normalized_market_feature_build_and_clock(tmp_path,monkeypatch):
    from e_data_v1 import core
    from gym_trading_env.utils import market_features
    config,market=csv_env_config(tmp_path/'single_source')
    profile=_profile(market)
    original_csv=__import__('pandas').read_csv
    original_features=market_features.build_market_features
    reads=[];builds=[]
    def counted_csv(path,*args,**kwargs):
        reads.append(str(path));return original_csv(path,*args,**kwargs)
    def counted_features(*args,**kwargs):
        builds.append(1);return original_features(*args,**kwargs)
    def duplicate_normalization(*args,**kwargs):
        raise AssertionError('E must reuse the already normalized Env market')
    monkeypatch.setattr(__import__('pandas'),'read_csv',counted_csv)
    monkeypatch.setattr(market_features,'build_market_features',counted_features)
    monkeypatch.setattr(core,'normalize_bars',duplicate_normalization)
    env=FixedDirectionEntryEnv(config_path=config,e_profile=profile)
    try:
        assert len(reads)==1 and builds==[1]
        assert env._e_market.market_clock_counter is env.market_clock_counter
    finally:env.close()


def test_native_profile_preserves_original_oracle_financial_rules(tmp_path):
    from test_fg_s3_dagger_d0 import query_for
    config,market=csv_env_config(tmp_path/'oracle')
    profile=_profile(market)
    legacy=FixedDirectionEntryEnv(config_path=config)
    native=FixedDirectionEntryEnv(config_path=config,e_profile=profile)
    try:
        query,_=query_for(legacy,config)
        legacy.reset(seed=202);native.reset(seed=202)
        answer=query.query(native,native._published_mask)
        assert answer['certified'],answer['reason']
        assert legacy.export_state()=={k:v for k,v in native.export_state().items() if k!='e_state'}
    finally:legacy.close();native.close()


def test_native_env_rejects_unavailable_current_minute(tmp_path):
    import pandas as pd
    from e_data_v1 import EDataError
    config,path=csv_env_config(tmp_path/'late')
    frame=pd.read_csv(path);frame['available_at']=frame.Date
    frame.loc[2,'available_at']=frame.Date.iloc[5]
    frame.to_csv(path,index=False)
    env=FixedDirectionEntryEnv(config_path=config,
        e_profile=_profile(path))
    try:
        env.step(0)
        with pytest.raises(EDataError,match='market_seq'):env.step(0)
    finally:env.close()


def test_e_obs_is_emitted_by_the_original_env_and_exact_legacy_18(tmp_path):
    config, market = csv_env_config(tmp_path/'e_input')
    profile = _profile(market, 'b'*64)
    legacy = FixedDirectionEntryEnv(config_path=config)
    extended = FixedDirectionEntryEnv(config_path=config, e_profile=profile)
    try:
        a, _ = legacy.reset()
        b, _ = extended.reset()
        assert set(a) == {'market_seq', 'position_state', 'action_mask'}
        assert set(b) == set(a) | {'market_seq_e_shared', 'market_context_15m', 'market_macd_5m'}
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(legacy._features, extended._features)
        assert extended._e_market is not None
        assert b['market_seq_e_shared'].shape == (60, 18)
        assert b['market_context_15m'].shape == (24, 7)
        assert np.isin(b['market_context_15m'][:, 6], (0., 1.)).all()
        saved = extended.export_state()
        before = extended._get_obs()
        extended.step(0)
        extended.import_state(saved)
        after = extended._get_obs()
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
    finally:
        legacy.close()
        extended.close()


def test_market_cache_does_not_cross_legacy_and_e_profile(tmp_path):
    config, market = csv_env_config(tmp_path/'source')
    cache = EntryMarketCache(capacity=2)
    profile = _profile(market, 'b'*64)
    old = FixedDirectionEntryEnv(config_path=config, market_cache=cache)
    new = FixedDirectionEntryEnv(config_path=config, market_cache=cache,
                                 e_profile=profile)
    again = FixedDirectionEntryEnv(config_path=config, market_cache=cache,
                                   e_profile=profile)
    try:
        assert len(cache) == 2
        assert new._e_market is not again._e_market
        assert new._e_market is not None and again._e_market is not None
        assert old._e_market is None
        assert old.config_hash==new.config_hash and old.observation_identity==new.observation_identity
        for key in new._get_obs():
            np.testing.assert_array_equal(new._get_obs()[key], again._get_obs()[key])
    finally:
        old.close(); new.close(); again.close()
