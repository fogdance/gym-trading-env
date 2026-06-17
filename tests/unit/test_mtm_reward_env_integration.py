import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from gym_trading_env.envs.trading_env import CustomTradingEnv
from gym_trading_env.rewards.reward_audit import RewardAuditMixin


pytestmark = pytest.mark.unit


ROOT = Path(__file__).resolve().parents[2]
DREAMER_ROOT = Path("/home/v/Documents/work/dreamerv3")
BASE_CONFIG = ROOT / "configs" / "env_trading_stage1_jm_walk_forward_train_20240603_20250731.yaml"


V1_DISABLED = [
    "fee",
    "dd",
    "eod",
    "close",
    "sl",
    "mkt_closed",
    "invalid_time",
    "invalid_streak",
    "invalid_total",
    "r_atr_close",
]

V2_DISABLED = [
    "fee",
    "eod",
    "close",
    "sl",
    "mkt_closed",
    "invalid_time",
    "invalid_streak",
    "invalid_total",
    "r_atr_close",
]

V2_EXTRA_KEYS = [
    "risk_dd",
    "risk_adverse",
    "risk_loss_time",
    "drawdown_cash",
    "drawdown_inc_cash",
    "adverse_cash",
    "adverse_inc_cash",
    "loss_steps",
    "w_dd",
    "w_adverse",
    "w_loss_time",
]


def _config_with_reward(tmp_path, reward_name):
    data = yaml.safe_load(BASE_CONFIG.read_text())
    data["training"]["reward_function"] = reward_name
    data["training"]["randomize_start"] = False
    path = tmp_path / f"{reward_name}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _valid_action(obs):
    valid = np.flatnonzero(np.asarray(obs["action_mask"]) > 0.5)
    assert valid.size > 0
    return int(1 if 1 in valid else valid[0])


@pytest.fixture
def env_cwd():
    previous = Path.cwd()
    os.chdir(DREAMER_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def _step_n(env, obs, n=3):
    info = None
    reward = None
    for _ in range(n):
        obs, reward, terminated, truncated, info = env.step(_valid_action(obs))
        if terminated or truncated:
            break
    assert info is not None
    return obs, reward, info


def test_env_info_logs_v1_clean_reward_components(tmp_path, env_cwd):
    config = _config_with_reward(
        tmp_path,
        "futures_intraday_mtm_clean_reward_function",
    )
    env = CustomTradingEnv(config_path=str(config))
    try:
        obs, _ = env.reset(seed=20260617)
        _, reward, info = _step_n(env, obs)

        assert isinstance(env.reward_function, RewardAuditMixin)
        for key in [
            "total",
            "pnl",
            "raw_total",
            "mtm_equity",
            "prev_mtm_equity",
            "delta_equity",
            "scale_cash",
            "close",
            "r_atr_close",
            "dd",
            "invalid_total",
        ]:
            assert f"log/env/reward/{key}" in info
        assert float(info["log/env/reward/total"]) == pytest.approx(float(reward))
        for key in V1_DISABLED:
            assert float(info[f"log/env/reward/{key}"]) == pytest.approx(0.0), key
    finally:
        env.close()


def test_env_info_logs_v2_risk_reward_components_and_extra_keys(tmp_path, env_cwd):
    config = _config_with_reward(
        tmp_path,
        "futures_intraday_mtm_risk_reward_function",
    )
    env = CustomTradingEnv(config_path=str(config))
    try:
        obs, reset_info = env.reset(seed=20260617)
        for key in V2_EXTRA_KEYS:
            assert f"log/env/reward/{key}" in reset_info
        _, reward, info = _step_n(env, obs)

        assert isinstance(env.reward_function, RewardAuditMixin)
        for key in [
            "total",
            "pnl",
            "raw_total",
            "mtm_equity",
            "delta_equity",
            "scale_cash",
            "dd",
            "risk_dd",
            "risk_adverse",
            "risk_loss_time",
            "drawdown_cash",
            "adverse_cash",
            "loss_steps",
        ]:
            assert f"log/env/reward/{key}" in info
        reset_reward_keys = {
            key for key in reset_info if key.startswith("log/env/reward/")
        }
        step_reward_keys = {
            key for key in info if key.startswith("log/env/reward/")
        }
        assert step_reward_keys == reset_reward_keys
        assert float(info["log/env/reward/total"]) == pytest.approx(float(reward))
        for key in V2_DISABLED:
            assert float(info[f"log/env/reward/{key}"]) == pytest.approx(0.0), key
    finally:
        env.close()
