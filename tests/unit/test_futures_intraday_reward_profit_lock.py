import math
from decimal import Decimal
from types import SimpleNamespace

import pytest

from gym_trading_env.envs.action import ForexCode
from gym_trading_env.rewards.reward_audit import (
    REWARD_DEBUG_KEYS,
    validate_reward_audit,
)
from gym_trading_env.rewards.reward_functions import (
    FuturesIntradayReward,
    FuturesIntradayRewardProfitLock,
    PROFIT_LOCK_DEBUG_KEYS,
    reward_classes,
)


pytestmark = pytest.mark.unit


class _Ledger:
    def __init__(self):
        self.cash = Decimal("10000")
        self.margin = Decimal("0")

    def balances(self):
        return {
            "user_cash": self.cash,
            "user_margin": self.margin,
        }


class _FakeEnv:
    def __init__(self, *, profit_lock_enabled=True):
        self.config = SimpleNamespace(
            trading=SimpleNamespace(
                alpha_unrealized=0.5,
                invalid_action_punish=0.02,
                invalid_time_cost_total=0.2,
                invalid_streak_cap=10,
                invalid_streak_mode="linear",
                atr_takeprofit_ratio=0.70,
                w_atr_close=0.10,
                lot_size=Decimal("60"),
                trade_lot=Decimal("1"),
                tick_size=Decimal("0.5"),
                spread=Decimal("0.5"),
                stop_loss_mode="abs",
                stop_loss_value=Decimal("5"),
                reward_profit_lock_enabled=profit_lock_enabled,
                reward_profit_lock_trigger_R=1.0,
                reward_profit_lock_scale_mode="daily_atr_ratio",
                reward_profit_lock_atr_ratio=0.55,
                reward_profit_lock_allowed_giveback_ratio=0.50,
                reward_profit_lock_min_capture_ratio=0.50,
                reward_profit_lock_w_giveback=0.05,
                reward_profit_lock_w_lost_mfe=0.10,
                reward_profit_lock_w_lock_close=0.05,
                reward_profit_lock_max_overlay_abs=0.20,
                reward_profit_lock_incremental_giveback=True,
                reward_profit_lock_lost_mfe_one_shot=True,
            ),
            training=SimpleNamespace(max_episode_steps=100, episode_length=None),
        )
        self.ledger = _Ledger()
        self.user_accounts = SimpleNamespace(
            unrealized_pnl=Decimal("0"),
            long_position=Decimal("0"),
            short_position=Decimal("0"),
        )
        self.bar_source = SimpleNamespace(
            store=SimpleNamespace(daily_atr_price=[100.0])
        )
        self.fee_step = Decimal("0")
        self.stop_loss_fired = 0
        self.last_close_position = None
        self.current_step = 0
        self.end_idx = 0
        self._eod_idx = 100
        self._day_i = 0
        self._minutes_to_eod_last = 100
        self._R_cash_last = Decimal("100")
        self.action_result = ForexCode.SUCCESS
        self._last_action_rejected = False
        self._reward_debug = {}


def _apply_state(
    env: _FakeEnv,
    *,
    cash="10000",
    margin="0",
    unrealized="0",
    long_pos="0",
    short_pos="0",
    close_pnl=None,
):
    env.ledger.cash = Decimal(str(cash))
    env.ledger.margin = Decimal(str(margin))
    env.user_accounts.unrealized_pnl = Decimal(str(unrealized))
    env.user_accounts.long_position = Decimal(str(long_pos))
    env.user_accounts.short_position = Decimal(str(short_pos))
    env.last_close_position = (
        {"pnl": Decimal(str(close_pnl))}
        if close_pnl is not None
        else None
    )


def _init_reward(reward):
    out = reward()
    assert out == pytest.approx(0.0)
    return reward


def _validate(reward_fn, reward):
    return validate_reward_audit(
        reward_fn,
        reward_name=reward_fn.__class__.__name__,
        returned_reward=reward,
    )


def test_profit_lock_reward_is_registered():
    assert (
        reward_classes["futures_intraday_reward_profit_lock_function"]
        is FuturesIntradayRewardProfitLock
    )


def test_profit_lock_disabled_matches_old_reward_with_zero_extra_fields():
    old_env = _FakeEnv()
    new_env = _FakeEnv(profit_lock_enabled=False)
    old_reward = _init_reward(FuturesIntradayReward(old_env))
    new_reward = _init_reward(FuturesIntradayRewardProfitLock(new_env))

    steps = [
        dict(unrealized="40", long_pos="1"),
        dict(unrealized="120", long_pos="1"),
        dict(unrealized="0", close_pnl="80"),
    ]
    for state in steps:
        _apply_state(old_env, **state)
        _apply_state(new_env, **state)
        old_out = old_reward()
        new_out = new_reward()
        assert new_out == pytest.approx(old_out, rel=0, abs=1e-12)

        old_debug = dict(old_env._reward_debug)
        new_debug = _validate(new_reward, new_out)
        for key in REWARD_DEBUG_KEYS:
            assert new_debug[key] == pytest.approx(old_debug[key], rel=0, abs=1e-12)
        for key in PROFIT_LOCK_DEBUG_KEYS:
            assert new_debug[key] == pytest.approx(0.0, rel=0, abs=1e-12)


def test_profit_lock_daily_atr_scale_keeps_one_tick_below_trigger():
    env = _FakeEnv()
    reward = _init_reward(FuturesIntradayRewardProfitLock(env))

    _apply_state(env, unrealized="30", long_pos="1")
    out = reward()
    debug = _validate(reward, out)

    assert debug["profit_lock_scale_cash"] == pytest.approx(3300.0)
    assert debug["profit_lock_scale_source_code"] == pytest.approx(1.0)
    assert debug["profit_lock_atr_price"] == pytest.approx(100.0)
    assert debug["profit_lock_atr_ratio"] == pytest.approx(0.55)
    assert debug["profit_lock_tick_cash"] == pytest.approx(30.0)
    assert debug["profit_lock_mfe_cash"] == pytest.approx(30.0)
    assert debug["profit_lock_mfe_R"] == pytest.approx(30.0 / 3300.0)
    assert debug["profit_lock_active"] == pytest.approx(0.0)
    assert debug["r_profit_lock_total"] == pytest.approx(0.0)


def test_profit_lock_daily_atr_scale_triggers_at_one_R():
    env = _FakeEnv()
    reward = _init_reward(FuturesIntradayRewardProfitLock(env))

    _apply_state(env, unrealized="3300", long_pos="1")
    out = reward()
    debug = _validate(reward, out)

    assert debug["profit_lock_scale_cash"] == pytest.approx(3300.0)
    assert debug["profit_lock_mfe_R"] == pytest.approx(1.0)
    assert debug["profit_lock_active"] == pytest.approx(1.0)


def test_profit_lock_scale_falls_back_to_stop_distance_when_atr_missing():
    env = _FakeEnv()
    env.bar_source.store.daily_atr_price = [0.0]
    reward = _init_reward(FuturesIntradayRewardProfitLock(env))

    _apply_state(env, unrealized="30", long_pos="1")
    out = reward()
    debug = _validate(reward, out)
    assert debug["profit_lock_scale_cash"] == pytest.approx(300.0)
    assert debug["profit_lock_scale_source_code"] == pytest.approx(2.0)
    assert debug["profit_lock_active"] == pytest.approx(0.0)

    _apply_state(env, unrealized="300", long_pos="1")
    out = reward()
    debug = _validate(reward, out)
    assert debug["profit_lock_mfe_R"] == pytest.approx(1.0)
    assert debug["profit_lock_active"] == pytest.approx(1.0)


def test_profit_lock_scale_final_fallback_uses_legacy_R_cash():
    env = _FakeEnv()
    env.bar_source.store.daily_atr_price = [0.0]
    env.config.trading.stop_loss_value = Decimal("0")
    reward = _init_reward(FuturesIntradayRewardProfitLock(env))

    _apply_state(env, unrealized="100", long_pos="1")
    out = reward()
    debug = _validate(reward, out)

    assert debug["profit_lock_scale_cash"] == pytest.approx(100.0)
    assert debug["profit_lock_scale_source_code"] == pytest.approx(3.0)
    assert debug["profit_lock_active"] == pytest.approx(1.0)


def test_profit_lock_giveback_penalty_is_incremental():
    env = _FakeEnv()
    reward = _init_reward(FuturesIntradayRewardProfitLock(env))

    _apply_state(env, unrealized="3600", long_pos="1")
    reward()
    assert env._reward_debug["profit_lock_active"] == pytest.approx(1.0)

    _apply_state(env, unrealized="1200", long_pos="1")
    out = reward()
    debug = _validate(reward, out)
    assert debug["profit_lock_giveback_cash"] == pytest.approx(2400.0)
    assert debug["profit_lock_allowed_giveback_cash"] == pytest.approx(1800.0)
    assert debug["profit_lock_excess_giveback_cash"] == pytest.approx(600.0)
    assert debug["profit_lock_excess_giveback_inc_cash"] == pytest.approx(600.0)
    assert debug["r_profit_giveback"] < 0.0

    _apply_state(env, unrealized="1200", long_pos="1")
    out = reward()
    debug = _validate(reward, out)
    assert debug["profit_lock_excess_giveback_cash"] == pytest.approx(600.0)
    assert debug["profit_lock_excess_giveback_inc_cash"] == pytest.approx(0.0)
    assert debug["r_profit_giveback"] == pytest.approx(0.0)


def test_profit_lock_lost_mfe_penalty_is_one_shot():
    env = _FakeEnv()
    reward = _init_reward(FuturesIntradayRewardProfitLock(env))

    _apply_state(env, unrealized="3300", long_pos="1")
    reward()

    _apply_state(env, unrealized="-1", long_pos="1")
    out = reward()
    debug = _validate(reward, out)
    assert debug["r_lost_mfe"] == pytest.approx(-0.10)
    assert debug["profit_lock_lost_mfe_fired"] == pytest.approx(1.0)

    _apply_state(env, unrealized="-20", long_pos="1")
    out = reward()
    debug = _validate(reward, out)
    assert debug["r_lost_mfe"] == pytest.approx(0.0)
    assert debug["profit_lock_lost_mfe_fired"] == pytest.approx(1.0)


def test_profit_lock_close_reward_for_good_capture():
    env = _FakeEnv()
    reward = _init_reward(FuturesIntradayRewardProfitLock(env))

    _apply_state(env, unrealized="3600", long_pos="1")
    reward()

    _apply_state(env, unrealized="0", close_pnl="2400")
    out = reward()
    debug = _validate(reward, out)

    assert debug["profit_lock_active"] == pytest.approx(1.0)
    assert debug["profit_lock_mfe_cash"] == pytest.approx(3600.0)
    assert debug["profit_lock_capture_ratio"] == pytest.approx(2400.0 / 3600.0)
    assert debug["r_lock_close"] > 0.0
    assert debug["r_profit_lock_total"] > 0.0


def test_profit_lock_close_at_loss_after_active_fires_lost_mfe():
    env = _FakeEnv()
    reward = _init_reward(FuturesIntradayRewardProfitLock(env))

    _apply_state(env, unrealized="3300", long_pos="1")
    reward()

    _apply_state(env, unrealized="0", close_pnl="-10")
    out = reward()
    debug = _validate(reward, out)

    assert debug["profit_lock_active"] == pytest.approx(1.0)
    assert debug["r_lock_close"] == pytest.approx(0.0)
    assert debug["r_lost_mfe"] == pytest.approx(-0.10)
    assert debug["profit_lock_lost_mfe_fired"] == pytest.approx(1.0)


def test_profit_lock_audit_fields_are_numeric_and_finite():
    env = _FakeEnv()
    reward = _init_reward(FuturesIntradayRewardProfitLock(env))

    _apply_state(env, unrealized="3300", long_pos="1")
    out = reward()
    debug = _validate(reward, out)

    for key in REWARD_DEBUG_KEYS:
        assert key in debug
    for key in PROFIT_LOCK_DEBUG_KEYS:
        assert key in debug
        assert isinstance(debug[key], (int, float)), key
        assert math.isfinite(float(debug[key])), key
