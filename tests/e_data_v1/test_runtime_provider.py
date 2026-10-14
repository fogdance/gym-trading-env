"""M4 adapter tests: the audited formulas must run on every Env decision."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from e_data_v1 import Decision
from e_data_v1.provider import ETrainProvider, verify_training_members


@pytest.mark.parametrize('minute', [0, 14, 15, 40, 359, 600])
def test_provider_consumes_current_cutoff_and_exact_market_seq(prepared, minute):
    contract = prepared.contract
    row = prepared.frame.iloc[minute]
    cutoff = prepared.frame.index[minute]
    source_obs = prepared.snapshot(Decision(contract, cutoff))[1]
    env = SimpleNamespace(contract=contract, _raw=prepared.frame, current_step=minute)
    provider = ETrainProvider({contract: prepared}, identity='1'*64,
                               source_receipt={'scope': 'synthetic test only'})
    arrays, meta = provider.snapshot(env, {'market_seq':source_obs})
    assert arrays['market_seq_e_shared'].shape == (60,18)
    assert arrays['market_context_15m'].shape == (24,7)
    assert arrays['market_context_15m'].dtype == np.float32
    assert meta['visible_cutoff'] == cutoff.isoformat()
    assert all(pd.Timestamp(x) <= cutoff for x in meta['context_available_at'])
    assert meta['data_identity'] == '1'*64
    if minute < 14:
        assert not arrays['market_context_15m'][:, 6].any()


def test_provider_refuses_foreign_contract_and_drift(prepared):
    provider=ETrainProvider({prepared.contract:prepared},identity='2'*64,
                            source_receipt={'scope': 'synthetic test only'})
    i=50; at=prepared.frame.index[i]
    obs=prepared.snapshot(Decision(prepared.contract, at))[1]
    e=SimpleNamespace(contract='CZCE.FG999',_raw=prepared.frame,current_step=i)
    with pytest.raises(ValueError,match='contract'):
        provider.snapshot(e,{'market_seq':obs})
    e.contract=prepared.contract
    tampered=obs.copy();tampered[-1,0]+=.01
    with pytest.raises(ValueError,match='market_seq|match'):
        provider.snapshot(e,{'market_seq':tampered})


def test_prepared_train_source_has_exact_underlying_market_identity():
    sample=dict(contract='CZCE.FG809', role='train', eligible_for_training=True,
        path='/readonly/FG809.csv', sha256='a'*64, env_config='/readonly/env.yaml',
        env_sha256='b'*64, contract_config='/readonly/contract.yaml',contract_sha256='c'*64)
    assert verify_training_members([sample], [sample]) == ('CZCE.FG809',)
    wrong={**sample,'sha256':'d'*64}
    with pytest.raises(ValueError,match='differs'):
        verify_training_members([wrong], [sample])
    with pytest.raises(ValueError,match='duplicate'):
        verify_training_members([sample], [sample,sample])
    with pytest.raises(ValueError,match='TRAIN'):
        verify_training_members([{**sample,'role':'dev'}], [sample])
