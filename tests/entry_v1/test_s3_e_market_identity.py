"""Env must consume the exact CSV bytes recorded by E admission."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from e_data_v1 import Decision, MarketData
from e_data_v1.io import (sha256, stable_identity, write_bundle,
                         write_collection_index, write_json)
from e_data_v1.provider import ETrainAdmission
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from gym_trading_env.envs.entry_market_cache import EntryMarketCache
from gym_trading_env.research.fg_replay import signed
from test_entry_market_cache import csv_env_config


@pytest.fixture
def admitted(tmp_path):
    # Real admission code, isolated synthetic market/config bytes. No TRAIN
    # input is modified and this fixture does not authorize model training.
    config, market = csv_env_config(tmp_path/'source')
    raw = yaml.safe_load(config.read_text())
    terms = config.parent/'contract.yaml'
    schedule = dict(raw['entry']['market_clock'],
                    dated_sessions=yaml.safe_load(terms.read_text())['sessions'])
    data = MarketData(pd.read_csv(market), schedule)
    sources = dict(market=stable_identity(market), env=stable_identity(config),
                   contract=stable_identity(terms), master=None)
    row = dict(contract=data.contract, role='train', eligible_for_training=True,
        path=str(market), sha256=sha256(market), env_config=str(config),
        env_sha256=sha256(config), contract_config=str(terms), contract_sha256=sha256(terms))
    pool = tmp_path/'pool.json'
    write_json(pool, signed(dict(product='FG', files=[row])))
    bundle = data.build([Decision(data.contract, data.frame.index[-1])])
    bundle.provenance.update(sources)
    collection = tmp_path/'collection'
    collection.mkdir()
    dest = write_bundle(bundle, collection/data.contract)
    report = dict(schema_sha256=bundle.schema['schema_sha256'],
                  source_pool=dict(sha256=sha256(pool)), bundles=[dict(path=str(dest))])
    write_json(collection/'diagnostics.json', report)
    write_collection_index(collection, report)
    profile = ETrainAdmission.from_train_pool(pool, collection, pool)
    return config, market, profile


@pytest.mark.parametrize('route', ['uncached', 'cache_miss', 'primed_cache'])
def test_env_refuses_csv_replaced_after_e_admission(admitted, route):
    config, market, profile = admitted
    cache = None if route == 'uncached' else EntryMarketCache(capacity=1)
    if route == 'primed_cache':
        original = FixedDirectionEntryEnv(config_path=config, e_profile=profile, market_cache=cache)
        original.close()
        assert len(cache) == 1
    frame = pd.read_csv(market)
    frame.loc[len(frame)-1, 'Volume'] += 1
    frame.to_csv(market, index=False)
    with pytest.raises(ValueError, match='E admitted market SHA256 mismatch'):
        env = FixedDirectionEntryEnv(config_path=config, e_profile=profile, market_cache=cache)
        env.close()


def test_e_parser_consumes_the_same_bytes_that_were_authenticated(admitted, monkeypatch):
    config, market, profile = admitted
    original = pd.read_csv
    expected = original(market)['Volume'].to_numpy()
    calls = []

    def replace_before_parse(source, *args, **kwargs):
        frame = original(market)
        frame.loc[len(frame)-1, 'Volume'] += 1
        frame.to_csv(market, index=False)
        calls.append(source)
        return original(source, *args, **kwargs)

    monkeypatch.setattr(pd, 'read_csv', replace_before_parse)
    env = FixedDirectionEntryEnv(config_path=config, e_profile=profile)
    try:
        np.testing.assert_array_equal(env._raw.Volume.to_numpy(), expected)
        assert len(calls) == 1
        assert env.e_market_source['sha256'] == profile.source_receipt['sources'][env.contract]['market']['sha256']
        assert env.e_market_source['sha256'] != sha256(market)
    finally:
        env.close()


def test_e_admission_cannot_be_bypassed_with_injected_frame(admitted):
    config, market, profile = admitted
    injected = pd.read_csv(market)
    injected.loc[len(injected)-1, 'Volume'] += 1
    with pytest.raises(ValueError, match='E admitted market requires CSV bytes'):
        env = FixedDirectionEntryEnv(config_path=config, e_profile=profile, df=injected)
        env.close()


def test_e_market_rejects_missing_source_receipt(admitted):
    config, _, profile = admitted
    unsigned = SimpleNamespace(identity=profile.identity, contracts=profile.contracts)
    with pytest.raises(ValueError, match='E admitted market source receipt missing'):
        env = FixedDirectionEntryEnv(config_path=config, e_profile=unsigned)
        env.close()


def _market_content_hash(market):
    digest = hashlib.sha256()
    for frame in (market.frame, market.features):
        digest.update(str((list(frame.columns), list(frame.dtypes))).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    for array in (market.obs_array, market.aux_array, market.price_array, market.good,
                  market.bucket_end_ns, market.bucket_available_ns, market.ref_source,
                  market.market_clock_counter.valid, market.market_clock_counter.prefix,
                  market.market_clock_counter.rows):
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    digest.update(json.dumps(dict(buckets=market.buckets, events=market.events,
        schedule=market.schedule), sort_keys=True, default=str).encode())
    return digest.hexdigest()


def test_shared_e_market_content_is_stable_across_accounts_and_restore(admitted):
    config, _, profile = admitted
    cache = EntryMarketCache(capacity=1)
    a = FixedDirectionEntryEnv(config_path=config, e_profile=profile, market_cache=cache)
    b = FixedDirectionEntryEnv(config_path=config, e_profile=profile, market_cache=cache)
    try:
        assert a._e_market is b._e_market
        assert a.e_market_source == b.e_market_source
        before = _market_content_hash(a._e_market)
        state = a.export_state()
        expected = a._get_obs()
        b.reset()
        a.step(1)
        b.step(0)
        a.reset(options={'resume': state})
        actual = a._get_obs()
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])
        assert _market_content_hash(a._e_market) == before == _market_content_hash(b._e_market)
        assert cache.hits == 1
    finally:
        a.close()
        b.close()
