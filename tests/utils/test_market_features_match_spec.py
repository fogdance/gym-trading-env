# tests/utils/test_market_features_match_spec.py

import numpy as np
import pandas as pd

from tests.factories.market_data_factory import MarketDataFactory
from tests.oracles.market_spec_oracle import (
    build_market_features_spec,
    REQUIRED_MARKET_COLS,
)
from tests.utils.assertions import assert_frame_allclose

from gym_trading_env.utils.market_features import build_market_features


def test_market_features_match_spec_futures_full_table():
    bundle = MarketDataFactory.make_futures_bundle(num_days=3)
    df = bundle.df_1m
    df_prev = bundle.df_prev_session

    got = build_market_features(
        df,
        tz=bundle.tz,
        is_future=True,
        df_prev_session=df_prev,
        limit_up_pct=None,
        limit_down_pct=None,
    )

    exp = build_market_features_spec(
        df,
        tz=bundle.tz,
        df_prev_session=df_prev,
        limit_up_pct=None,
        limit_down_pct=None,
    )

    # 1) 必须包含所有 REQUIRED_MARKET_COLS
    assert list(got.columns) == list(REQUIRED_MARKET_COLS)
    assert list(exp.columns) == list(REQUIRED_MARKET_COLS)

    # 2) 逐字段逐行 allclose
    assert_frame_allclose(got, exp, cols=list(REQUIRED_MARKET_COLS), rtol=1e-6, atol=1e-6)
