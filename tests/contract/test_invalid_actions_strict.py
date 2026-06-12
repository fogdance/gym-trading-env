import numpy as np
import pandas as pd
import pytest

from gym_trading_env.envs.action import TargetPos
from gym_trading_env.envs.trading_env import CustomTradingEnv
from gym_trading_env.utils.decimal_util import D, D0


pytestmark = pytest.mark.unit


def make_one_day_df(start="2020-01-01 21:01:00", periods=345):
    idx = pd.date_range(start=start, periods=periods, freq="min")
    close = 1.1000 + 0.0001 * np.arange(periods, dtype=float)
    return pd.DataFrame(
        {"Date": idx, "Open": close, "High": close, "Low": close, "Close": close, "Volume": 1.0}
    )


@pytest.fixture
def env():
    result = CustomTradingEnv(df=make_one_day_df(), config_path="tests/test.yaml")
    result.config.training.randomize_start = False
    result.config.training.start_clock = "future_night"
    result.config.trading.session_policy.force_flatten_eod = False
    result.reset(seed=0)
    yield result
    result.close()


def assert_rejected_strict_noop(env, info_before, info_after, ledger_entries_before):
    assert int(info_after["log/env/invalid_action"]) == 1
    assert int(info_after["log/env/action_rejected"]) == 1
    assert int(info_after["log/env/execution_failed"]) == 0
    assert len(env.ledger.entries) == ledger_entries_before
    assert D(info_after["fees_collected"]) == D(info_before["fees_collected"])
    assert D(info_after["long_position"]) == D(info_before["long_position"])
    assert D(info_after["short_position"]) == D(info_before["short_position"])


def test_max_entries_rejects_new_target_as_strict_noop(env):
    env._entries_used_today = int(env.config.trading.max_entries_per_day)
    obs = env._get_obs()
    info_before = env._get_info()
    ledger_entries_before = len(env.ledger.entries)

    assert obs["action_mask"].tolist() == [0.0, 1.0, 0.0]
    _, _, _, _, info_after = env.step(int(TargetPos.LONG))

    assert_rejected_strict_noop(env, info_before, info_after, ledger_entries_before)


def test_market_closed_rejects_target_change_as_strict_noop(env):
    env.bar_source.store.row_mask[env.current_step] = 0.0
    obs = env._get_obs()
    info_before = env._get_info()
    ledger_entries_before = len(env.ledger.entries)

    assert obs["action_mask"].tolist() == [0.0, 1.0, 0.0]
    _, _, _, _, info_after = env.step(int(TargetPos.SHORT))

    assert_rejected_strict_noop(env, info_before, info_after, ledger_entries_before)


def test_near_eod_rejects_open_as_strict_noop(env):
    env.current_step = int(env.bar_source.store.day_ranges[env._day_i][1]) - 1
    env.current_minute = int(env.bar_source.store.row_minute[env.current_step])
    env._last_valid_price = D(env.bar_source.store.row_C[env.current_step])
    obs = env._get_obs()
    info_before = env._get_info()
    ledger_entries_before = len(env.ledger.entries)

    assert obs["action_mask"].tolist() == [0.0, 1.0, 0.0]
    _, _, _, _, info_after = env.step(int(TargetPos.LONG))

    assert_rejected_strict_noop(env, info_before, info_after, ledger_entries_before)


def test_entries_increment_only_for_successful_target_change(env):
    assert env._entries_used_today == 0

    _, _, _, _, opened = env.step(int(TargetPos.LONG))
    assert int(opened["log/env/action_rejected"]) == 0
    assert env.user_accounts.long_position > D0
    assert env._entries_used_today == 1

    env._entries_used_today = int(env.config.trading.max_entries_per_day)
    env._get_obs()
    _, _, _, _, rejected = env.step(int(TargetPos.SHORT))

    assert int(rejected["log/env/action_rejected"]) == 1
    assert env._entries_used_today == int(env.config.trading.max_entries_per_day)
