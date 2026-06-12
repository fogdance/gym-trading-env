import numpy as np
import pandas as pd
import pytest

from gym_trading_env.envs.action import Action, ForexCode, TargetPos
from gym_trading_env.envs.trading_env import CustomTradingEnv
from gym_trading_env.utils.decimal_util import D


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


def assert_mask_matches_table(env, expected):
    obs = env._get_obs()
    table = env._get_target_transition_table()
    actual = obs["action_mask"]
    assert actual.dtype == np.float32
    assert actual.tolist() == expected
    assert actual.tolist() == [float(decision.allowed) for decision in table.decisions]
    assert env.observation_space["action_mask"].contains(actual)
    gates = env.get_oracle_snapshot()["agent_input"]
    current = env._current_target()
    assert gates["can_long_open"] == int(current == TargetPos.FLAT and actual[TargetPos.LONG])
    assert gates["can_short_open"] == int(current == TargetPos.FLAT and actual[TargetPos.SHORT])
    assert gates["can_long_close"] == int(current == TargetPos.LONG and actual[TargetPos.FLAT])
    assert gates["can_short_close"] == int(current == TargetPos.SHORT and actual[TargetPos.FLAT])


def test_flat_normal_state_all_targets_are_legal(env):
    assert_mask_matches_table(env, [1.0, 1.0, 1.0])


def test_max_entries_masks_new_entries_but_keeps_hold_and_close(env):
    env._entries_used_today = int(env.config.trading.max_entries_per_day)
    assert_mask_matches_table(env, [0.0, 1.0, 0.0])

    env._entries_used_today = 0
    env._get_obs()
    env.step(int(TargetPos.LONG))
    env._entries_used_today = int(env.config.trading.max_entries_per_day)
    assert_mask_matches_table(env, [0.0, 1.0, 1.0])


def test_near_eod_masks_open_and_flip_but_keeps_close(env):
    env.current_step = int(env.bar_source.store.day_ranges[env._day_i][1]) - 1
    env.bar_source.store.row_mask[env.current_step] = 1.0
    env.current_minute = int(env.bar_source.store.row_minute[env.current_step])
    env._last_valid_price = D(env.bar_source.store.row_C[env.current_step])
    assert env._near_eod()
    assert_mask_matches_table(env, [0.0, 1.0, 0.0])

    env.current_step = 0
    env.current_minute = int(env.bar_source.store.row_minute[env.current_step])
    env._last_valid_price = D(env.bar_source.store.row_C[env.current_step])
    env._get_obs()
    env.step(int(TargetPos.LONG))
    env.current_step = int(env.bar_source.store.day_ranges[env._day_i][1]) - 1
    env.bar_source.store.row_mask[env.current_step] = 1.0
    env.current_minute = int(env.bar_source.store.row_minute[env.current_step])
    env._last_valid_price = D(env.bar_source.store.row_C[env.current_step])
    assert_mask_matches_table(env, [0.0, 1.0, 1.0])


def test_market_closed_only_current_target_is_legal(env):
    env.bar_source.store.row_mask[env.current_step] = 0.0
    assert_mask_matches_table(env, [0.0, 1.0, 0.0])


def test_insufficient_money_masks_new_entries(env):
    balances = env.ledger.snapshot()
    balances["user_cash"] = D("0")
    env.ledger.restore(balances)
    assert_mask_matches_table(env, [0.0, 1.0, 0.0])


def test_masked_request_is_strict_noop_and_reports_decision(env):
    env._entries_used_today = int(env.config.trading.max_entries_per_day)
    env._get_obs()
    ledger_entries = len(env.ledger.entries)
    long_before = env.user_accounts.long_position
    short_before = env.user_accounts.short_position

    _, _, _, _, info = env.step(int(TargetPos.LONG))

    assert len(env.ledger.entries) == ledger_entries
    assert env.user_accounts.long_position == long_before
    assert env.user_accounts.short_position == short_before
    assert env.requested_target == TargetPos.LONG
    assert env.planned_action == Action.LONG_OPEN0
    assert env.executed_action == Action.HOLD
    assert env.action_result == ForexCode.ERROR_HIT_DAY_MAX_OPEN
    assert info["reject_code"] == ForexCode.ERROR_HIT_DAY_MAX_OPEN.value
    assert int(info["log/env/invalid_action"]) == 1
    assert int(info["log/env/valid_action_cardinality"]) == 1
    assert int(info["log/env/chosen_action_index"]) == int(TargetPos.LONG)
    assert int(info["log/env/chosen_action_was_valid"]) == 0
    assert int(info["log/env/action_rejected"]) == 1
    assert int(info["log/env/execution_failed"]) == 0
    assert int(info["log/env/executed_action"]) == Action.HOLD.value


def test_legal_request_executes_planned_action(env):
    _, _, _, _, info = env.step(int(TargetPos.LONG))

    assert env.requested_target == TargetPos.LONG
    assert env.planned_action == Action.LONG_OPEN0
    assert env.executed_action == Action.LONG_OPEN0
    assert env.action_result == ForexCode.SUCCESS
    assert info["reject_code"] == ForexCode.SUCCESS.value
    assert env.user_accounts.long_position > D("0")
    assert int(info["log/env/invalid_action"]) == 0
    assert int(info["log/env/chosen_action_was_valid"]) == 1
    assert int(info["log/env/action_rejected"]) == 0
    assert int(info["log/env/execution_failed"]) == 0


def test_execution_failure_is_separate_from_action_rejection(env, monkeypatch):
    monkeypatch.setattr(
        env, "_commit_target_execution_quote",
        lambda quote: ForexCode.ERROR_NO_ENOUGH_MONEY)

    _, _, _, _, info = env.step(int(TargetPos.LONG))

    assert int(info["log/env/action_rejected"]) == 0
    assert int(info["log/env/execution_failed"]) == 1
    assert int(info["log/env/invalid_action"]) == 0
    assert info["reject_code"] == ForexCode.ERROR_NO_ENOUGH_MONEY.value
    assert env.planned_action == Action.LONG_OPEN0
    assert env.executed_action == Action.HOLD


def test_transition_quotes_are_execution_ready_and_side_effect_free(env):
    before = env._snapshot_execution_state()

    env._get_obs()
    table = env._pending_transition_table

    assert_execution_state_equal(env, before)
    assert table.action_price == env._last_valid_price
    for decision in table.decisions:
        if decision.allowed and decision.planned_action != Action.HOLD:
            assert decision.quote is not None
            assert decision.quote.action == decision.planned_action
            assert decision.quote.action_price == table.action_price
            assert decision.quote.ledger_entry.postings
        else:
            assert decision.quote is None


def test_step_commits_published_open_quote_without_requoting(env, monkeypatch):
    env._get_obs()
    quoted_step = env.current_step
    decision = env._pending_transition_table.for_index(int(TargetPos.LONG))
    quote = decision.quote
    original_quote_target_execution = env._quote_target_execution

    def reject_same_step_requote(action, price):
        if env.current_step == quoted_step:
            raise AssertionError("step() requoted the published transition")
        return original_quote_target_execution(action, price)

    monkeypatch.setattr(env, "_quote_target_execution", reject_same_step_requote)
    monkeypatch.setattr(
        env, "_long_open",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("step() called legacy open path")))

    _, _, _, _, info = env.step(int(TargetPos.LONG))

    assert int(info["log/env/execution_failed"]) == 0
    assert env.position_manager.long_positions[0] is quote.open_position
    assert env.ledger.entries[-1] is quote.ledger_entry


def test_step_commits_published_flip_quote_without_requoting(env, monkeypatch):
    env.step(int(TargetPos.LONG))
    quoted_step = env.current_step
    decision = env._pending_transition_table.for_index(int(TargetPos.SHORT))
    quote = decision.quote
    original_quote_close_long = env.position_manager.quote_close_long

    def reject_same_step_requote(*args, **kwargs):
        if env.current_step == quoted_step:
            raise AssertionError("step() requoted the published close")
        return original_quote_close_long(*args, **kwargs)

    monkeypatch.setattr(env.position_manager, "quote_close_long", reject_same_step_requote)
    monkeypatch.setattr(
        env, "_flip_long_to_short",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("step() called legacy flip path")))

    _, _, _, _, info = env.step(int(TargetPos.SHORT))

    assert int(info["log/env/execution_failed"]) == 0
    assert env.position_manager.short_positions[0] is quote.open_position
    assert env.ledger.entries[-1] is quote.ledger_entry


def test_step_fails_fast_when_observation_transition_table_is_stale(env):
    obs = env._get_obs()
    assert obs["action_mask"][TargetPos.LONG] == 1.0
    state_before = env._snapshot_execution_state()

    env._entries_used_today = int(env.config.trading.max_entries_per_day)

    with pytest.raises(RuntimeError, match="Stale target transition table"):
        env.step(int(TargetPos.LONG))

    assert env.user_accounts.long_position == D("0")
    assert env.user_accounts.short_position == D("0")
    assert env.ledger.snapshot() == state_before["ledger_balances"]
    assert len(env.ledger.entries) == state_before["ledger_entries_len"]

    refreshed = env._get_obs()
    assert refreshed["action_mask"].tolist() == [0.0, 1.0, 0.0]
    _, _, _, _, info = env.step(int(TargetPos.LONG))
    assert int(info["log/env/action_rejected"]) == 1


def assert_execution_state_equal(env, expected):
    actual = env._snapshot_execution_state()
    assert actual["ledger_balances"] == expected["ledger_balances"]
    assert actual["ledger_entries_len"] == expected["ledger_entries_len"]
    assert actual["long_positions"] == expected["long_positions"]
    assert actual["short_positions"] == expected["short_positions"]
    assert actual["closed_trade_profits"] == expected["closed_trade_profits"]
    assert actual["realized_pnl"] == expected["realized_pnl"]
    assert actual["unrealized_pnl"] == expected["unrealized_pnl"]
    assert actual["trade_history_len"] == expected["trade_history_len"]
    assert actual["entries_used_today"] == expected["entries_used_today"]
    assert actual["last_close_position"] == expected["last_close_position"]


def test_open_commit_failure_restores_complete_execution_state(env, monkeypatch):
    before = env._snapshot_execution_state()
    monkeypatch.setattr(
        env.position_manager, "add_long_position",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("forced open failure")))

    code = env._long_open(env._last_valid_price, env.config.trading.spread, slot=0)

    assert code == ForexCode.ERROR_OPEN_POSITION
    assert_execution_state_equal(env, before)


def test_close_commit_failure_restores_complete_execution_state(env, monkeypatch):
    assert env._long_open(env._last_valid_price, env.config.trading.spread, slot=0) == ForexCode.SUCCESS
    before = env._snapshot_execution_state()
    monkeypatch.setattr(
        env.user_accounts, "realize_pnl",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("forced pnl failure")))

    code = env._long_close(env._last_valid_price, env.config.trading.spread, slot=0)

    assert code == ForexCode.ERROR_NO_ENOUGH_MONEY
    assert_execution_state_equal(env, before)


def test_flip_commit_failure_restores_complete_execution_state(env, monkeypatch):
    assert env._long_open(env._last_valid_price, env.config.trading.spread, slot=0) == ForexCode.SUCCESS
    before = env._snapshot_execution_state()
    monkeypatch.setattr(
        env.position_manager, "add_short_position",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("forced flip failure")))

    code = env._flip_long_to_short(env._last_valid_price, env.config.trading.spread)

    assert code == ForexCode.ERROR_NO_ENOUGH_MONEY
    assert_execution_state_equal(env, before)


def test_target_open_quote_commit_failure_restores_complete_execution_state(env, monkeypatch):
    quote = env._pending_transition_table.for_index(int(TargetPos.LONG)).quote
    before = env._snapshot_execution_state()
    monkeypatch.setattr(
        env.position_manager, "add_long_position",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("forced open failure")))

    code = env._commit_target_execution_quote(quote)

    assert code == ForexCode.ERROR_OPEN_POSITION
    assert_execution_state_equal(env, before)


def test_target_close_quote_commit_failure_restores_complete_execution_state(env, monkeypatch):
    env.step(int(TargetPos.LONG))
    quote = env._pending_transition_table.for_index(int(TargetPos.FLAT)).quote
    before = env._snapshot_execution_state()
    monkeypatch.setattr(
        env.user_accounts, "realize_pnl",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("forced pnl failure")))

    code = env._commit_target_execution_quote(quote)

    assert code == ForexCode.ERROR_NO_ENOUGH_MONEY
    assert_execution_state_equal(env, before)


def test_target_flip_quote_commit_failure_restores_complete_execution_state(env, monkeypatch):
    env.step(int(TargetPos.LONG))
    quote = env._pending_transition_table.for_index(int(TargetPos.SHORT)).quote
    before = env._snapshot_execution_state()
    monkeypatch.setattr(
        env.position_manager, "add_short_position",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("forced flip failure")))

    code = env._commit_target_execution_quote(quote)

    assert code == ForexCode.ERROR_NO_ENOUGH_MONEY
    assert_execution_state_equal(env, before)


def test_mask_respecting_rollout_has_no_rejections(env):
    rng = np.random.default_rng(7)
    env.config.trading.session_policy.force_flatten_eod = False
    obs = env._get_obs()

    for _ in range(300):
        legal = np.flatnonzero(obs["action_mask"] > 0.5)
        action = int(rng.choice(legal))
        obs, _, terminated, truncated, info = env.step(action)
        assert int(info["log/env/chosen_action_was_valid"]) == 1
        assert int(info["log/env/action_rejected"]) == 0
        assert int(info["log/env/execution_failed"]) == 0
        assert int(info["reject_code"]) == ForexCode.SUCCESS.value
        assert int(info["log/env/invalid_action"]) == 0
        if terminated or truncated:
            break
