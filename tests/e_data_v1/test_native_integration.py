"""Cross-check production function injection against the unchanged E math."""
import numpy as np
import pytest

from e_data_v1 import Decision, MarketData
from e_data_v1 import core


def test_e_market_from_existing_feature_clock_is_identical(market, prepared, monkeypatch):
    frame, schedule = market
    calls = []

    def reuse_existing(frame, clock):
        calls.append((len(frame), clock))
        # The actual Env passes its existing build_market_features here.
        # This fixture returns the same feature frame without another build.
        return prepared.features

    def reject_duplicate_legacy_loading(*args, **kwargs):
        raise AssertionError('Runtime E must use the existing Env feature builder')

    from e_data_v1.reference import legacy_helper
    tz_normalizer = legacy_helper('time_contract').ensure_feature_tz_index
    ohlc_normalizer = legacy_helper('ohlcvi_contract').normalize_ohlcvi
    monkeypatch.setattr(core, 'legacy_features', reject_duplicate_legacy_loading)
    monkeypatch.setattr(core, 'legacy_helper', reject_duplicate_legacy_loading)
    model = MarketData(frame, schedule, feature_function=reuse_existing,
        market_clock_class=type(prepared.market_clock_counter),
        tz_normalizer=tz_normalizer, ohlc_normalizer=ohlc_normalizer)
    assert len(calls) == 1
    assert calls[0][0] == len(prepared.features)
    assert np.array_equal(model.obs_array, prepared.obs_array)
    for i in (0, 14, 15, 39, 360, 400):
        decision = Decision(model.contract, model.frame.index[i])
        x = model.snapshot(decision)
        y = prepared.snapshot(decision)
        for a, b in zip(x[:3], y[:3]):
            assert np.array_equal(a, b)
        assert x[5] == y[5]


def test_original_delay_causality_survives_feature_injection(market, prepared):
    frame, schedule = market
    frame = frame.copy()
    frame['available_at'] = frame['Date']
    # Existing Data-V2 fixes must continue to reject retroactive baselines.
    frame.loc[14, 'available_at'] = frame.loc[14, 'Date'] + np.timedelta64(45, 'm')
    source = MarketData(frame, schedule)
    t = frame.loc[29, 'Date']
    from e_data_v1 import EDataError
    result = source.snapshot(Decision(source.contract, t))
    assert result[2].shape == (60, 18)
