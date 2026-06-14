import numpy as np
import pandas as pd
import pytest

from gym_trading_env.envs.action import TargetPos
from gym_trading_env.envs.trading_env import CustomTradingEnv


pytestmark = pytest.mark.unit


def make_one_day_df(start="2020-01-01 21:01:00", periods=345):
    idx = pd.date_range(start=start, periods=periods, freq="min")
    close = 1.1000 + 0.0001 * np.arange(periods, dtype=float)
    return pd.DataFrame(
        {"Date": idx, "Open": close, "High": close, "Low": close, "Close": close, "Volume": 1.0}
    )


def test_rgb_array_render_returns_equity_curve_image():
    env = CustomTradingEnv(df=make_one_day_df(), config_path="tests/test.yaml", render_mode="rgb_array")
    try:
        obs, _ = env.reset(seed=0)
        assert "action_mask" in obs

        image = env.render()
        assert image is not None
        assert image.dtype == np.uint8
        assert image.shape == (96, 96, 3)
        assert image.sum() > 0

        history_len = len(env._render_equity_history)
        image_again = env.render()
        assert image_again.shape == image.shape
        assert len(env._render_equity_history) == history_len

        env.step(int(TargetPos.FLAT))
        image_after_step = env.render()
        assert image_after_step.shape == image.shape
        assert len(env._render_equity_history) == history_len + 1
    finally:
        env.close()
