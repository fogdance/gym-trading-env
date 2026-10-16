import copy

import numpy as np
import pytest

from e_model_fixture import collector_fixture
from gym_trading_env.research.fg_replay import FIELDS
from gym_trading_env.research.fg_s3_e_observation import (
    E_FIELDS_V2, OBS_SCHEMA, observation_contract, validate_e_observation)


def test_EM10_original_driver_dagger_native_replay_context_and_executed_action(tmp_path):
    collector,provider=collector_fixture(tmp_path)
    try:
        collector.advance(75)
        raw=collector.replay.sample(1)
        assert set(E_FIELDS_V2).issubset(raw)
        assert 'market_context_15m' not in FIELDS and 'market_seq_e_shared' not in FIELDS
        joined=collector.join(raw)
        assert joined['teacher_valid'][:,1:].any()
        assert all(r['e_snapshot']['data_identity']==provider.identity for r in collector.records)
        assert collector.identity['e_observation']==observation_contract()
        for pos in np.ndindex(raw['action'].shape):
            row=collector.index[raw['stepid'][pos].tobytes().hex()]
            assert raw['action'][pos]==row['executed_action']
        wrong=copy.deepcopy(raw);wrong['market_context_15m'][0,10,23,0]+=1
        with pytest.raises(ValueError,match='observation alignment'):collector.join(wrong)
        wrong=copy.deepcopy(raw);wrong['action'][0,10]=1-wrong['action'][0,10]
        with pytest.raises(ValueError,match='executed action'):collector.join(wrong)
        wrong=copy.deepcopy(raw);wrong['market_seq_e_shared'][0,10,0,0]+=1
        with pytest.raises(ValueError,match='observation alignment'):collector.join(wrong)
    finally:
        collector.close()


def test_EM05_schema_and_future_outcome_actor_fields_rejected(tmp_path):
    collector,_=collector_fixture(tmp_path)
    try:
        collector.advance(65)
        sample=collector.replay.sample(1)
        obs={k:v[:,0] for k,v in sample.items() if k in E_FIELDS_V2 and k!='action'}
        validate_e_observation(obs,batched=True)
        for key in ('teacher_action','oracle_future_close','outcome_event','future_ohlc'):
            with pytest.raises(ValueError):collector.policy(None,{**obs,key:np.ones(1)})
        bad={**obs,'market_context_15m':obs['market_context_15m'].astype(np.float64)}
        with pytest.raises(ValueError,match='dtype'):validate_e_observation(bad,batched=True)
    finally:
        collector.close()
