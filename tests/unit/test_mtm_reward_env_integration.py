import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from gym_trading_env.envs.trading_env import CustomTradingEnv
from gym_trading_env.rewards.reward_audit import RewardAuditMixin
from gym_trading_env.rewards.reward_functions import reward_classes


pytestmark = pytest.mark.unit


ROOT = Path(__file__).resolve().parents[2]
DREAMER_ROOT = Path("/home/v/Documents/work/dreamerv3")
BASE_CONFIG = Path(
    os.environ.get(
        "GYM_TRADING_TEST_ENV_CONFIG",
        "/data/logdir/trading_contracts/jm_walk_forward_20240603_20251202/"
        "configs/env/jm_walk_forward_20240603_20251202_train.yaml",
    )
)


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


def _config_with_reward(
    tmp_path,
    reward_name,
    *,
    reward_scale_cash=None,
    episode_length=None,
    max_episode_steps=None,
):
    if not BASE_CONFIG.exists():
        pytest.skip(f"external test env config is unavailable: {BASE_CONFIG}")
    data = yaml.safe_load(BASE_CONFIG.read_text())
    data["training"]["reward_function"] = reward_name
    data["training"]["randomize_start"] = False
    if episode_length is not None:
        data["training"]["episode_length"] = episode_length
    if max_episode_steps is not None:
        data["training"]["max_episode_steps"] = max_episode_steps
    if reward_scale_cash is not None:
        data["trading"]["reward_scale_cash"] = reward_scale_cash
    path = tmp_path / f"{reward_name}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _valid_action(obs):
    valid = np.flatnonzero(np.asarray(obs["action_mask"]) > 0.5)
    assert valid.size > 0
    return int(1 if 1 in valid else valid[0])


class _EpisodeEndProbeReward(RewardAuditMixin):
    def __init__(self, env):
        self.env = env
        self.start_calls = 0
        self.end_calls = 0
        self.last_end_context = None
        self._set_reward_debug(self.default_reward_debug())

    def on_episode_start(self, context):
        self.start_calls += 1

    def __call__(self, obs):
        debug = self.default_reward_debug()
        debug["total"] = 0.0
        debug["raw_total"] = 0.0
        self._set_reward_debug(debug)
        return 0.0

    def on_episode_end(self, context):
        self.end_calls += 1
        self.last_end_context = dict(context)
        debug = self.default_reward_debug()
        debug["pnl"] = 123.0
        debug["total"] = float(context["reward"])
        debug["raw_total"] = float(context["reward"])
        self._set_reward_debug(debug)


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


def test_env_primes_pnl_aligned_reward_after_account_reset(tmp_path, env_cwd):
    config = _config_with_reward(
        tmp_path,
        "futures_intraday_pnl_aligned_reward_function",
        reward_scale_cash=100,
    )
    env = CustomTradingEnv(config_path=str(config))
    try:
        obs, reset_info = env.reset(seed=20260617)

        assert isinstance(env.reward_function, RewardAuditMixin)
        assert env.reward_function.previous_equity is not None
        assert env.reward_function.initial_equity == env.reward_function.previous_equity
        assert "log/env/reward/total" in reset_info
        assert float(reset_info["log/env/reward/total"]) == pytest.approx(0.0)
        assert float(reset_info["log/env/reward/scale_cash"]) == pytest.approx(100.0)

        obs, reward, terminated, truncated, info = env.step(_valid_action(obs))

        assert np.isfinite(float(reward))
        assert float(info["log/env/reward/total"]) == pytest.approx(float(reward))
    finally:
        env.close()


def test_env_calls_reward_episode_end_once_on_terminal_step(tmp_path, env_cwd):
    reward_name = "episode_end_probe_reward_function"
    previous = reward_classes.get(reward_name)
    reward_classes[reward_name] = _EpisodeEndProbeReward
    try:
        config = _config_with_reward(
            tmp_path,
            reward_name,
            episode_length=2,
            max_episode_steps=10,
        )
        env = CustomTradingEnv(config_path=str(config))
        try:
            obs, _ = env.reset(seed=20260617)
            reward_fn = env.reward_function

            terminal_reward = None
            terminal_info = None
            terminal_terminated = False
            terminal_truncated = False
            for _ in range(5):
                obs, reward, terminated, truncated, info = env.step(_valid_action(obs))
                if terminated or truncated:
                    terminal_reward = reward
                    terminal_info = info
                    terminal_terminated = terminated
                    terminal_truncated = truncated
                    break

            assert terminal_info is not None
            assert reward_fn.start_calls == 1
            assert reward_fn.end_calls == 1
            assert reward_fn.last_end_context["reward"] == pytest.approx(float(terminal_reward))
            assert reward_fn.last_end_context["terminated"] is bool(terminal_terminated)
            assert reward_fn.last_end_context["truncated"] is bool(terminal_truncated)
            assert "reward_debug" in reward_fn.last_end_context
            assert "reward_episode_sums" in reward_fn.last_end_context
            assert reward_fn.last_end_context["reward_debug"]["pnl"] == pytest.approx(0.0)
            assert "log/env/reward_episode/reward_total" in terminal_info
            assert float(terminal_info["log/env/reward/pnl"]) == pytest.approx(0.0)
            assert float(reward_fn.reward_audit_debug()["pnl"]) == pytest.approx(0.0)

            env.step(0)
            assert reward_fn.end_calls == 1
        finally:
            env.close()
    finally:
        if previous is None:
            reward_classes.pop(reward_name, None)
        else:
            reward_classes[reward_name] = previous
