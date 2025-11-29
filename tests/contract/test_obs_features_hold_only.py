# tests/contract/test_obs_features_hold_only.py
import numpy as np
import pandas as pd
import pytest

from gym_trading_env.envs.trading_env import CustomTradingEnv, Action
from gym_trading_env.utils.market_features import FEATURES_MARKET, build_market_features
from gym_trading_env.utils.agent_features import FEATURES_AGENT
from gym_trading_env.utils.trade_util import step_wrapper
from ..support.feature_oracle import FeatureOracle

pytestmark = pytest.mark.contract


def make_one_day_df(start="2020-01-01 21:01:00", periods=1440):
    idx = pd.date_range(start=start, periods=periods, freq="min")
    close = 1.1000 + 0.0001 * np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "Date": idx,
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": np.ones(periods, dtype=float),
        }
    )


@pytest.fixture
def env():
    df = make_one_day_df()
    e = CustomTradingEnv(df=df, config_path="tests/test.yaml")
    e.config.training.randomize_start = False
    e.config.training.episode_length = 1440  # 固定 1 天
    obs, info = e.reset()
    yield e
    e.close()


def _assert_shapes_and_dtypes(obs):
    assert "market_seq" in obs, "obs must contain 'market_seq'"
    assert "agent_state" in obs, "obs must contain 'agent_state'"

    m = obs["market_seq"]
    a = obs["agent_state"]

    assert m.shape == (1440, len(FEATURES_MARKET)), f"market_seq shape mismatch: {m.shape}"
    assert a.shape == (len(FEATURES_AGENT),), f"agent_state shape mismatch: {a.shape}"

    assert m.dtype == np.float32, f"market_seq dtype should be float32, got {m.dtype}"
    assert a.dtype == np.float32, f"agent_state dtype should be float32, got {a.dtype}"


def test_hold_only_three_steps_feature_integrity(env):
    """
    不交易：reset + HOLD 3 步
    每一步逐列核对：
    - market_seq（build_xt 真值 env.df_market）+ temporal reveal（未来全 0）
    - agent_state（无持仓契约）
    """
    # t0
    obs0, info0 = env.reset()
    _assert_shapes_and_dtypes(obs0)

    f0 = FeatureOracle.frontier_from_obs_market(obs0["market_seq"])
    # 在 episode_length=1440 且不允许非 0 起步的前提下，t0 一般应该只 reveal 到 0
    assert f0 == 0, f"frontier at reset must be 0, got {f0}"

    exp_m0 = FeatureOracle.expected_market_seq(env, frontier=f0)
    FeatureOracle.assert_market_columnwise(obs0["market_seq"], exp_m0, frontier=f0)

    exp_a0 = FeatureOracle.expected_agent_state_no_position(env)
    FeatureOracle.assert_agent_columnwise(obs0["agent_state"], exp_a0)

    # t1~t3
    for t in [1, 2, 3]:
        obs, reward, terminated, truncated, info = step_wrapper(env, Action.HOLD)
        _assert_shapes_and_dtypes(obs)

        f = FeatureOracle.frontier_from_obs_market(obs["market_seq"])
        assert f == t, f"after {t} HOLD steps, frontier must be {t}, got {f}"

        exp_m = FeatureOracle.expected_market_seq(env, frontier=f)
        FeatureOracle.assert_market_columnwise(obs["market_seq"], exp_m, frontier=f)

        exp_a = FeatureOracle.expected_agent_state_no_position(env)
        FeatureOracle.assert_agent_columnwise(obs["agent_state"], exp_a)

        assert terminated is False
        assert truncated is False
