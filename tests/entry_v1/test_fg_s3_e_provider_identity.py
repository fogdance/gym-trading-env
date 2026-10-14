"""Regression: provider metadata must describe the actual Env decision."""
import pytest
from e_model_fixture import SyntheticEProvider, collector_fixture


@pytest.mark.parametrize('fault',['contract','time'])
def test_EM10_provider_contract_and_decision_time_must_match_env(tmp_path,fault):
    class WrongProvider(SyntheticEProvider):
        def snapshot(self,env,obs):
            arrays,meta=super().snapshot(env,obs)
            if fault=='contract':meta['source_contract']='FOREIGN.CONTRACT'
            else:meta['decision_time']=meta['visible_cutoff']='2100-01-01T00:00:00+08:00'
            return arrays,meta
    collector,_=collector_fixture(tmp_path,provider=WrongProvider(tmp_path))
    try:
        with pytest.raises(ValueError,match='E provider.*Env'):
            collector.advance(1)
    finally:collector.close()
