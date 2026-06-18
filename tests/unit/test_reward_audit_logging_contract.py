import os
from pathlib import Path

import numpy as np
import pytest

from gym_trading_env.envs.trading_env import (
    CustomTradingEnv,
    REWARD_DEBUG_KEYS,
    REWARD_EPISODE_AUDIT_KEYS,
)
from gym_trading_env.rewards.reward_audit import RewardAuditMixin


pytestmark = pytest.mark.unit


ROOT = Path(__file__).resolve().parents[2]
DREAMER_ROOT = Path("/home/v/Documents/work/dreamerv3")
CONFIG = Path(
    os.environ.get(
        "GYM_TRADING_TEST_ENV_CONFIG",
        "/data/logdir/trading_contracts/jm_walk_forward_20240603_20251202/"
        "configs/env/jm_walk_forward_20240603_20251202_train.yaml",
    )
)


@pytest.fixture
def reward_env():
    if not CONFIG.exists():
        pytest.skip(f"external test env config is unavailable: {CONFIG}")
    previous_cwd = Path.cwd()
    env = None
    try:
        # The CSV loader resolves data/... relative to cwd; Dreamer owns the
        # shared walk-forward data files used by formal training.
        os.chdir(DREAMER_ROOT)
        env = CustomTradingEnv(config_path=str(CONFIG))
        yield env
    finally:
        if env is not None:
            env.close()
        os.chdir(previous_cwd)


def _reward_key(name: str) -> str:
    return f"log/env/reward/{name}"


def _episode_key(name: str) -> str:
    return f"log/env/reward_episode/{name}"


def _valid_action(obs) -> int:
    valid = np.flatnonzero(np.asarray(obs["action_mask"]) > 0.5)
    assert valid.size > 0
    # Prefer FLAT if legal to avoid unnecessary trading in the aggregation test.
    if 1 in valid:
        return 1
    return int(valid[0])


def test_reward_debug_keys_exist_from_reset(reward_env):
    _, info = reward_env.reset(seed=20260617)

    assert isinstance(reward_env.reward_function, RewardAuditMixin)
    missing = [key for key in REWARD_DEBUG_KEYS if _reward_key(key) not in info]
    assert not missing

    missing_episode = [
        key for key in REWARD_EPISODE_AUDIT_KEYS
        if _episode_key(key) not in info
    ]
    assert not missing_episode

    for key in REWARD_EPISODE_AUDIT_KEYS:
        assert float(info[_episode_key(key)]) == pytest.approx(0.0)


def test_step_info_contains_current_reward_debug(reward_env):
    obs, _ = reward_env.reset(seed=20260617)

    obs, reward, terminated, truncated, info = reward_env.step(_valid_action(obs))

    assert not (terminated or truncated)
    for key in REWARD_DEBUG_KEYS:
        assert _reward_key(key) in info
    assert float(info[_reward_key("total")]) == pytest.approx(float(reward))

    for key in REWARD_EPISODE_AUDIT_KEYS:
        assert _episode_key(key) in info
        assert float(info[_episode_key(key)]) == pytest.approx(0.0)


def test_terminal_info_emits_episode_reward_aggregation(reward_env):
    obs, _ = reward_env.reset(seed=20260617)

    reward_total = 0.0
    terminal_info = None
    for _ in range(500):
        obs, reward, terminated, truncated, info = reward_env.step(_valid_action(obs))
        reward_total += float(reward)
        if terminated or truncated:
            terminal_info = info
            break

    assert terminal_info is not None
    assert float(terminal_info[_episode_key("reward_total")]) == pytest.approx(
        reward_total, abs=1e-6)
    assert float(terminal_info[_episode_key("final_equity")]) == pytest.approx(
        float(reward_env._calculate_equity()), abs=1e-6)
    assert float(terminal_info[_episode_key("trade_count")]) == pytest.approx(0.0)
