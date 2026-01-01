# tests/contract/test_invalid_actions_strict.py
import pytest
import numpy as np
import pandas as pd

from gym_trading_env.envs.trading_env import CustomTradingEnv, Action
from gym_trading_env.utils.decimal_util import D, D0
from gym_trading_env.utils.trade_util import step_wrapper

pytestmark = pytest.mark.unit


def make_one_day_df(start="2020-01-01 21:01:00", periods=200):
    idx = pd.date_range(start=start, periods=periods, freq="min")
    close = 1.1000 + 0.0001 * np.arange(periods, dtype=float)
    return pd.DataFrame(
        {"Date": idx, "Open": close, "High": close, "Low": close, "Close": close, "Volume": 1.0}
    )


def pick_action(env: CustomTradingEnv, preferred: str, fallback: str):
    if hasattr(Action, preferred):
        return getattr(Action, preferred)
    return getattr(Action, fallback)


@pytest.fixture
def env():
    df = make_one_day_df()
    e = CustomTradingEnv(df=df, config_path="tests/test.yaml")
    e.config.training.randomize_start = False
    # 固定锚点，保证 current_step=0 可控
    if hasattr(e.config.training, "start_clock"):
        e.config.training.start_clock = "future_night"
    e.reset(seed=0)
    yield e
    e.close()


def _assert_invalid_contract(env, info_before, info_after, ledger_n_before):
    # 1) 必须打出 invalid 标记（否则无法诊断“agent vs env”）
    assert "log/env/invalid_action" in info_after, "env must emit log/env/invalid_action"
    assert int(info_after["log/env/invalid_action"]) == 1

    # 2) 无效动作必须严格 no-op：不扣费、不写账、不改变仓位
    assert len(env.ledger.entries) == ledger_n_before, "invalid action must not write ledger"

    assert D(info_after["fees_collected"]) == D(info_before["fees_collected"]), "invalid action must not change fees"
    assert D(info_after["long_position"]) == D(info_before["long_position"]), "invalid action must not change long"
    assert D(info_after["short_position"]) == D(info_before["short_position"]), "invalid action must not change short"


def _assert_valid_contract(info_after):
    assert "log/env/invalid_action" in info_after
    assert int(info_after["log/env/invalid_action"]) == 0


def test_close_long_while_flat_is_invalid_noop(env):
    # flat
    obs0, info0 = env.reset(seed=0)
    assert D(info0["long_position"]) == D0
    assert D(info0["short_position"]) == D0

    A_LONG_CLOSE = pick_action(env, "LONG_CLOSE0", "LONG_CLOSE")

    ledger_n0 = len(env.ledger.entries)
    _, r, term, trunc, info1 = step_wrapper(env, A_LONG_CLOSE)

    _assert_invalid_contract(env, info0, info1, ledger_n0)


def test_close_short_while_flat_is_invalid_noop(env):
    obs0, info0 = env.reset(seed=0)
    assert D(info0["long_position"]) == D0
    assert D(info0["short_position"]) == D0

    A_SHORT_CLOSE = pick_action(env, "SHORT_CLOSE0", "SHORT_CLOSE")

    ledger_n0 = len(env.ledger.entries)
    _, r, term, trunc, info1 = step_wrapper(env, A_SHORT_CLOSE)

    _assert_invalid_contract(env, info0, info1, ledger_n0)


def test_open_short_while_long_is_invalid_noop(env):
    env.reset(seed=0)

    A_LONG_OPEN = pick_action(env, "LONG_OPEN0", "LONG_OPEN")
    A_SHORT_OPEN = pick_action(env, "SHORT_OPEN0", "SHORT_OPEN")

    # open long (valid)
    _, _, _, _, info1 = step_wrapper(env, A_LONG_OPEN)
    _assert_valid_contract(info1)
    assert D(info1["long_position"]) > D0
    assert D(info1["short_position"]) == D0

    # then attempt open short while holding long => invalid
    ledger_n1 = len(env.ledger.entries)
    _, _, _, _, info2 = step_wrapper(env, A_SHORT_OPEN)

    _assert_invalid_contract(env, info1, info2, ledger_n1)


def test_open_long_while_short_is_invalid_noop(env):
    env.reset(seed=0)

    A_SHORT_OPEN = pick_action(env, "SHORT_OPEN0", "SHORT_OPEN")
    A_LONG_OPEN = pick_action(env, "LONG_OPEN0", "LONG_OPEN")

    # open short (valid)
    _, _, _, _, info1 = step_wrapper(env, A_SHORT_OPEN)
    _assert_valid_contract(info1)
    assert D(info1["short_position"]) > D0
    assert D(info1["long_position"]) == D0

    # then attempt open long while holding short => invalid
    ledger_n1 = len(env.ledger.entries)
    _, _, _, _, info2 = step_wrapper(env, A_LONG_OPEN)

    _assert_invalid_contract(env, info1, info2, ledger_n1)


def test_close_wrong_side_is_invalid_noop(env):
    env.reset(seed=0)

    A_LONG_OPEN = pick_action(env, "LONG_OPEN0", "LONG_OPEN")
    A_SHORT_CLOSE = pick_action(env, "SHORT_CLOSE0", "SHORT_CLOSE")

    # open long (valid)
    _, _, _, _, info1 = step_wrapper(env, A_LONG_OPEN)
    _assert_valid_contract(info1)
    assert D(info1["long_position"]) > D0
    assert D(info1["short_position"]) == D0

    # close_short while long => invalid
    ledger_n1 = len(env.ledger.entries)
    _, _, _, _, info2 = step_wrapper(env, A_SHORT_CLOSE)
    _assert_invalid_contract(env, info1, info2, ledger_n1)


def test_entries_used_today_increments_only_on_successful_open(env):
    env.reset(seed=0)

    # 这个字段你在 FeatureOracle 里已经用到了，所以建议当作强契约
    assert hasattr(env, "_entries_used_today"), "env must track _entries_used_today"

    A_LONG_OPEN = pick_action(env, "LONG_OPEN0", "LONG_OPEN")
    A_SHORT_OPEN = pick_action(env, "SHORT_OPEN0", "SHORT_OPEN")

    # reset 后应为 0
    assert int(getattr(env, "_entries_used_today")) == 0

    # 成功开仓后应递增
    _, _, _, _, info1 = step_wrapper(env, A_LONG_OPEN)
    _assert_valid_contract(info1)
    assert int(getattr(env, "_entries_used_today")) == 1

    # 持仓下尝试反手 open（应 invalid/no-op），entries_used_today 不应变化
    _, _, _, _, info2 = step_wrapper(env, A_SHORT_OPEN)
    assert int(info2["log/env/invalid_action"]) == 1
    assert int(getattr(env, "_entries_used_today")) == 1
