"""CPU regressions for immutable market reuse across real account rotations."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.envs.entry_dataset_env import FGEntryDatasetEnv
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from test_fg_single3_env import single_env


def csv_env_config(directory, contract='CZCE.FG2701'):
    directory.mkdir()
    source = single_env(directory, [
        (1000, 1000, 1000, 1000), (1000, 1007, 1000, 1007),
        (1000, 1000, 1000, 1000), (1000, 1000, 969, 970),
        (1000, 1000, 1000, 1000), (1000, 1007, 1000, 1007),
        (1000, 1000, 1000, 1000), (1000, 1000, 1000, 1000),
    ], candidate='C1', fees=3)
    frame = source._raw.copy()
    frame['trading_date'] = '20260105'
    frame['symbol'] = contract
    market = directory/'market.csv'
    frame.to_csv(market, index_label='Date')
    config = directory/'env.yaml'
    raw = yaml.safe_load(config.read_text())
    raw['entry'].update(contract=contract, market_clock=dict(
        timezone='Asia/Shanghai', minutes_per_day=345,
        segments=[['21:00', '23:00'], ['09:00', '10:15'],
                  ['10:30', '11:30'], ['13:30', '15:00']]))
    raw['trading'].update(data_path=str(market), future_symbol=contract)
    config.write_text(yaml.safe_dump(raw))
    terms = directory/'contract.yaml'
    item = yaml.safe_load(terms.read_text()); item['contract'] = contract
    terms.write_text(yaml.safe_dump(item))
    directions = directory/'directions.yaml'
    item = yaml.safe_load(directions.read_text())
    for row in item['intervals']: row['contract'] = contract
    directions.write_text(yaml.safe_dump(item))
    source.close()
    return config, market


def test_real_clock_does_not_build_discarded_legacy_features(tmp_path, monkeypatch):
    from gym_trading_env.utils import bar_source, market_features
    config, _ = csv_env_config(tmp_path/'real')
    original = market_features.build_market_features
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs.get('real_futures_clock', False))
        return original(*args, **kwargs)

    monkeypatch.setattr(bar_source, 'build_market_features', counted)
    monkeypatch.setattr(market_features, 'build_market_features', counted)
    env = FixedDirectionEntryEnv(config_path=config)
    try:
        assert calls == [True]
        assert env._features.shape == (8, 18)
    finally:
        env.close()


def test_market_cache_reuses_only_market_and_owns_independent_copies(tmp_path, monkeypatch):
    from gym_trading_env.envs.entry_market_cache import EntryMarketCache
    config, _ = csv_env_config(tmp_path/'real')
    cache = EntryMarketCache(capacity=1)
    original = FixedDirectionEntryEnv._load_market
    calls = []

    def counted(self, df):
        calls.append(self.contract)
        return original(self, df)

    monkeypatch.setattr(FixedDirectionEntryEnv, '_load_market', counted)
    a = FixedDirectionEntryEnv(config_path=config, market_cache=cache)
    original_raw = a._raw.copy()
    original_features = a._features.copy()
    a.step(1)
    assert a.net_lots == 0 and a.attributions[0]['reason'] == 'TAKE_PROFIT'
    a._raw.loc[a._raw.index[-1], 'Volume'] = 0
    a._features[-1, 0] += 10
    b = FixedDirectionEntryEnv(config_path=config, market_cache=cache)
    try:
        assert calls == ['CZCE.FG2701']
        pd.testing.assert_frame_equal(b._raw, original_raw)
        np.testing.assert_array_equal(b._features, original_features)
        assert not np.shares_memory(a._features, b._features)
        assert b.current_step == 0 and b.total_reward == 0 and not b.fills
        assert b.config_hash == a.config_hash and b.observation_identity == a.observation_identity
    finally:
        a.close(); b.close()


def test_cache_rechecks_csv_bytes_and_config_identity(tmp_path, monkeypatch):
    from gym_trading_env.envs.entry_market_cache import EntryMarketCache
    config, market = csv_env_config(tmp_path/'real')
    cache = EntryMarketCache(capacity=1)
    original = FixedDirectionEntryEnv._load_market
    calls = []

    def counted(self, df):
        calls.append(self.config_hash)
        return original(self, df)

    monkeypatch.setattr(FixedDirectionEntryEnv, '_load_market', counted)
    a = FixedDirectionEntryEnv(config_path=config, market_cache=cache)
    frame = pd.read_csv(market)
    frame.loc[len(frame)-1, ['High', 'Close']] = 1009
    frame.to_csv(market, index=False)
    b = FixedDirectionEntryEnv(config_path=config, market_cache=cache)
    assert b._raw.iloc[-1].Close == 1009 and a._raw.iloc[-1].Close == 1000
    terms = config.parent/'contract.yaml'
    raw = yaml.safe_load(terms.read_text())
    raw['terms'][0]['open_fee'] = 4
    terms.write_text(yaml.safe_dump(raw))
    c = FixedDirectionEntryEnv(config_path=config, market_cache=cache)
    try:
        assert len(calls) == 3 and calls[0] != calls[2]
        assert len(cache) == 1
    finally:
        a.close(); b.close(); c.close()


def test_changed_file_during_cache_build_is_not_published(tmp_path, monkeypatch):
    from gym_trading_env.envs.entry_market_cache import EntryMarketCache
    config, market = csv_env_config(tmp_path/'real')
    cache = EntryMarketCache(capacity=1)
    original = FixedDirectionEntryEnv._load_market

    def changed(self, df):
        original(self, df)
        market.write_text(market.read_text()+'\n')

    monkeypatch.setattr(FixedDirectionEntryEnv, '_load_market', changed)
    with pytest.raises(ValueError, match='Market input changed'):
        FixedDirectionEntryEnv(config_path=config, market_cache=cache)
    assert len(cache) == 0


def test_injected_frames_bypass_csv_cache_and_capacity_is_bounded(tmp_path):
    from gym_trading_env.envs.entry_market_cache import EntryMarketCache
    first, market = csv_env_config(tmp_path/'a')
    second, _ = csv_env_config(tmp_path/'b', contract='CZCE.FG2702')
    cache = EntryMarketCache(capacity=1)
    a = FixedDirectionEntryEnv(config_path=first, market_cache=cache)
    b = FixedDirectionEntryEnv(config_path=second, market_cache=cache)
    injected = pd.read_csv(market)
    injected.loc[len(injected)-1, 'Volume'] = 0
    c = FixedDirectionEntryEnv(config_path=first, df=injected, market_cache=cache)
    try:
        assert len(cache) == 1
        assert c._raw.iloc[-1].Volume == 0 and a._raw.iloc[-1].Volume == 100
        cache.clear()
        assert len(cache) == 0
    finally:
        a.close(); b.close(); c.close()


def test_cached_pool_matches_uncached_tp_sl_and_cold_restore(tmp_path):
    paths = [csv_env_config(tmp_path/'a')[0],
             csv_env_config(tmp_path/'b', contract='CZCE.FG2702')[0]]
    manifest = dict(product='FG', files=[dict(role='train', eligible_for_training=True,
                    rows=8, env_config=str(p)) for p in paths])
    manifest['content_hash'] = fingerprint(manifest)
    path = tmp_path/'pool.json'; path.write_text(json.dumps(manifest))
    selector = tmp_path/'selector.yaml'
    selector.write_text(yaml.safe_dump(dict(strategy_mode='entry_dataset', role='train',
        manifest_path=str(path), scheduler=dict(mode='least_progress', chunk_bars=2))))
    cached = FGEntryDatasetEnv(selector)
    original = FGEntryDatasetEnv(selector, cache_market=False)
    restored = None
    try:
        cached.reset(seed=0); original.reset(seed=0)
        for _ in range(8):
            a = cached.step(1); b = original.step(1)
            for key in a[0]: np.testing.assert_array_equal(a[0][key], b[0][key])
            assert a[1:] == b[1:]
            assert cached.env.export_state() == original.env.export_state()
            if a[3]:
                a = cached.reset(); b = original.reset()
                for key in a[0]: np.testing.assert_array_equal(a[0][key], b[0][key])
                assert a[1] == b[1]
        state = json.loads(json.dumps(cached.export_state()))
        assert state == json.loads(json.dumps(original.export_state()))
        assert len(cached._market_cache) == 2
        restored = FGEntryDatasetEnv(selector)
        restored.reset(options={'resume': copy.deepcopy(state)})
        assert json.loads(json.dumps(restored.export_state())) == state
        for _ in range(2):
            a = cached.step(0); b = restored.step(0)
            for key in a[0]: np.testing.assert_array_equal(a[0][key], b[0][key])
            assert a[1:] == b[1:]
            if a[3]: cached.reset(); restored.reset()
        assert cached.export_state() == restored.export_state()
        assert {x['reason'] for r in cached.accounts.values() for x in r['attributions']} >= {
            'TAKE_PROFIT', 'STOP_LOSS'}
        assert 'market_cache' not in state and state['schema'] == 2
    finally:
        cached.close(); original.close()
        if restored: restored.close()
    assert len(cached._market_cache) == 0
