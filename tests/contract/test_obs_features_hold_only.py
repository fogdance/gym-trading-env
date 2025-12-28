import numpy as np
import pandas as pd
import pytest

from gym_trading_env.envs.trading_env import CustomTradingEnv, Action
from gym_trading_env.utils.trade_util import step_wrapper
from ..support.feature_oracle import FeatureOracle

pytestmark = pytest.mark.unit


def make_one_day_df(start="2020-01-01 21:01:00", periods=345):
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
    e.config.training.episode_length = 345  # 固定 1 天

    # 强制从 21:01 锚点起步（你的 make_one_day_df 正好从 21:01 开始）
    e.config.training.start_clock = "future_night"

    obs, info = e.reset()
    yield e
    e.close()


def _assert_shapes_and_dtypes(env, obs):
    assert "market_seq" in obs
    assert "agent_state" in obs

    m = obs["market_seq"]
    a = obs["agent_state"]

    assert m.shape == (env.window_size, len(env._OBS_FEATURES_MARKET)), f"market_seq shape mismatch: {m.shape}"
    assert a.shape == (len(env._OBS_FEATURES_AGENT),), f"agent_state shape mismatch: {a.shape}"

    assert m.dtype == np.float32
    assert a.dtype == np.float32
    assert np.isfinite(m).all()
    assert np.isfinite(a).all()


def test_hold_only_three_steps_feature_integrity(env):
    """
    不交易：reset + HOLD 3 步
    每一步逐列核对：
    - market_seq：对齐现有 _get_obs()（全局滑窗 + 左 padding）
    - agent_state：对齐 agent_features 计算结果（raw/obs 由 config 决定）
    """
    # t0
    obs0, info0 = env.reset()
    _assert_shapes_and_dtypes(env, obs0)

    end0 = int(env.current_step)
    exp_m0 = FeatureOracle.expected_market_seq(env, end_i=end0)
    FeatureOracle.assert_market_columnwise(obs0["market_seq"], exp_m0, end_i=end0)

    # 左 padding 检查：current_step=0 时应为 ws-1（如果真的从 0 起步）
    pad0 = FeatureOracle.pad_len_from_obs_market(env, obs0["market_seq"])
    exp_pad0 = max(0, int(env.window_size) - (end0 + 1))
    assert pad0 == exp_pad0, f"pad_len at reset mismatch: got {pad0}, expected {exp_pad0}"

    exp_a0 = FeatureOracle.expected_agent_state(env)
    FeatureOracle.assert_agent_columnwise(env, obs0["agent_state"], exp_a0)

    # t1~t3
    for _ in [1, 2, 3]:
        obs, reward, terminated, truncated, info = step_wrapper(env, Action.HOLD)
        _assert_shapes_and_dtypes(env, obs)

        end_i = int(env.current_step)

        exp_m = FeatureOracle.expected_market_seq(env, end_i=end_i)
        FeatureOracle.assert_market_columnwise(obs["market_seq"], exp_m, end_i=end_i)

        pad = FeatureOracle.pad_len_from_obs_market(env, obs["market_seq"])
        exp_pad = max(0, int(env.window_size) - (end_i + 1))
        assert pad == exp_pad, f"pad_len mismatch at step={end_i}: got {pad}, expected {exp_pad}"

        exp_a = FeatureOracle.expected_agent_state(env)
        FeatureOracle.assert_agent_columnwise(env, obs["agent_state"], exp_a)

        assert terminated is False
        assert truncated is False
