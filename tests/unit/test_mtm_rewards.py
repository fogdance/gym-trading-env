from decimal import Decimal
from types import SimpleNamespace

import math
import pytest

from gym_trading_env.rewards.reward_audit import (
    REWARD_DEBUG_KEYS,
    RewardAuditMixin,
    validate_reward_audit,
)
from gym_trading_env.rewards.reward_functions import (
    FuturesIntradayMTMCleanReward,
    FuturesIntradayMTMRiskReward,
    reward_classes,
)


pytestmark = pytest.mark.unit


V1_DISABLED = {
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
}

V2_DISABLED = {
    "fee",
    "eod",
    "close",
    "sl",
    "mkt_closed",
    "invalid_time",
    "invalid_streak",
    "invalid_total",
    "r_atr_close",
}


class _Ledger:
    def __init__(self, cash="10000", margin="0"):
        self.cash = Decimal(str(cash))
        self.margin = Decimal(str(margin))

    def balances(self):
        return {
            "user_cash": self.cash,
            "user_margin": self.margin,
        }


class _FakeEnv:
    def __init__(self):
        self.ledger = _Ledger()
        self.user_accounts = SimpleNamespace(
            unrealized_pnl=Decimal("0"),
            long_position=Decimal("0"),
            short_position=Decimal("0"),
        )
        self.config = SimpleNamespace(
            trading=SimpleNamespace(
                reward_scale_cash_fallback="10",
                reward_v2_w_dd=0.05,
                reward_v2_w_adverse=0.05,
                reward_v2_w_loss_time=0.001,
            )
        )
        self._R_cash_last = Decimal("10")
        self.fee_step = Decimal("0")
        self._last_action_rejected = False

    def mtm_equity(self):
        return (
            self.ledger.cash
            + self.ledger.margin
            + self.user_accounts.unrealized_pnl
        )


def _validate(reward_fn, reward):
    return validate_reward_audit(
        reward_fn,
        reward_name=reward_fn.__class__.__name__,
        returned_reward=reward,
    )


def _assert_schema(debug):
    assert set(REWARD_DEBUG_KEYS).issubset(debug)
    for key, value in debug.items():
        assert isinstance(value, (int, float)), key
        assert math.isfinite(float(value)), key


def test_mtm_reward_imports_and_registration():
    assert issubclass(FuturesIntradayMTMCleanReward, RewardAuditMixin)
    assert issubclass(FuturesIntradayMTMRiskReward, RewardAuditMixin)
    assert (
        reward_classes["futures_intraday_mtm_clean_reward_function"]
        is FuturesIntradayMTMCleanReward
    )
    assert (
        reward_classes["futures_intraday_mtm_risk_reward_function"]
        is FuturesIntradayMTMRiskReward
    )


@pytest.mark.parametrize(
    "reward_cls,disabled",
    [
        (FuturesIntradayMTMCleanReward, V1_DISABLED),
        (FuturesIntradayMTMRiskReward, V2_DISABLED),
    ],
)
def test_mtm_reward_audit_schema_after_init_and_first_call(reward_cls, disabled):
    env = _FakeEnv()
    reward_fn = reward_cls(env)

    init_debug = validate_reward_audit(
        reward_fn,
        reward_name=reward_cls.__name__,
        returned_reward=None,
    )
    _assert_schema(init_debug)

    reward = reward_fn()
    debug = _validate(reward_fn, reward)
    _assert_schema(debug)
    assert reward == pytest.approx(0.0)
    assert debug["total"] == pytest.approx(0.0)
    assert debug["delta_equity"] == pytest.approx(0.0)
    assert debug["mtm_equity"] == pytest.approx(float(env.mtm_equity()))
    assert debug["prev_mtm_equity"] == pytest.approx(float(env.mtm_equity()))
    assert debug["scale_cash"] == pytest.approx(10.0)
    for key in disabled:
        assert debug[key] == pytest.approx(0.0), key


def test_v1_mtm_equity_formula_and_favorable_adverse_moves():
    env = _FakeEnv()
    reward_fn = FuturesIntradayMTMCleanReward(env)
    assert reward_fn() == pytest.approx(0.0)

    env.ledger.margin = Decimal("100")
    env.ledger.cash = Decimal("9920")
    env.user_accounts.unrealized_pnl = Decimal("30")
    reward = reward_fn()
    debug = _validate(reward_fn, reward)
    assert debug["mtm_equity"] == pytest.approx(10050.0)
    assert debug["alpha_unrealized"] == pytest.approx(1.0)
    assert reward > 0
    assert debug["pnl"] == pytest.approx(debug["total"])
    assert debug["delta_equity"] == pytest.approx(50.0)

    env.user_accounts.unrealized_pnl = Decimal("-20")
    reward = reward_fn()
    debug = _validate(reward_fn, reward)
    assert reward < 0
    assert debug["pnl"] == pytest.approx(debug["total"])
    assert debug["delta_equity"] == pytest.approx(-50.0)


def test_v1_fee_is_reflected_in_equity_delta_without_double_penalty():
    env = _FakeEnv()
    reward_fn = FuturesIntradayMTMCleanReward(env)
    reward_fn()

    env.ledger.cash -= Decimal("3")
    env.fee_step = Decimal("3")
    reward = reward_fn()
    debug = _validate(reward_fn, reward)
    assert reward < 0
    assert debug["delta_equity"] == pytest.approx(-3.0)
    assert debug["raw_total"] == pytest.approx(-0.3)
    assert debug["fee"] == pytest.approx(0.0)
    assert debug["fee_cash_debug"] == pytest.approx(3.0)


def test_v1_close_and_invalid_action_are_debug_only():
    env = _FakeEnv()
    env.last_close_position = {"pnl": Decimal("500")}
    env._last_action_rejected = True
    reward_fn = FuturesIntradayMTMCleanReward(env)
    reward_fn()
    reward = reward_fn()
    debug = _validate(reward_fn, reward)
    assert debug["close"] == pytest.approx(0.0)
    assert debug["r_atr_close"] == pytest.approx(0.0)
    assert debug["invalid_action_debug"] == pytest.approx(1.0)
    assert debug["invalid_total"] == pytest.approx(0.0)


def test_v1_no_double_tanh_contract():
    env = _FakeEnv()
    reward_fn = FuturesIntradayMTMCleanReward(env, clip=1.0)
    reward_fn()

    env.ledger.cash += Decimal("5")
    reward = reward_fn()
    debug = _validate(reward_fn, reward)
    assert debug["raw_total"] == pytest.approx(0.5)
    assert debug["total"] == pytest.approx(math.tanh(0.5))
    assert reward == pytest.approx(math.tanh(0.5))


def test_v2_first_call_and_favorable_movement_have_no_risk_penalty():
    env = _FakeEnv()
    reward_fn = FuturesIntradayMTMRiskReward(env)
    reward = reward_fn()
    debug = _validate(reward_fn, reward)
    assert debug["dd"] == pytest.approx(0.0)
    assert debug["risk_dd"] == pytest.approx(0.0)
    assert debug["risk_adverse"] == pytest.approx(0.0)
    assert debug["risk_loss_time"] == pytest.approx(0.0)

    env.ledger.cash += Decimal("5")
    reward = reward_fn()
    debug = _validate(reward_fn, reward)
    assert reward > 0
    assert debug["dd"] == pytest.approx(0.0)
    assert debug["risk_adverse"] == pytest.approx(0.0)
    assert debug["risk_loss_time"] == pytest.approx(0.0)


def test_v2_drawdown_increment_creates_negative_risk_dd():
    env = _FakeEnv()
    reward_fn = FuturesIntradayMTMRiskReward(env)
    reward_fn()

    env.ledger.cash += Decimal("20")
    reward_fn()
    env.ledger.cash -= Decimal("10")
    reward = reward_fn()
    debug = _validate(reward_fn, reward)
    assert debug["drawdown_cash"] > 0
    assert debug["drawdown_inc_cash"] > 0
    assert debug["risk_dd"] < 0
    assert debug["dd"] < 0


def test_v2_adverse_loss_and_underwater_time_then_recovery():
    env = _FakeEnv()
    env.user_accounts.long_position = Decimal("1")
    reward_fn = FuturesIntradayMTMRiskReward(env)
    reward_fn()

    env.user_accounts.unrealized_pnl = Decimal("-20")
    reward = reward_fn()
    debug = _validate(reward_fn, reward)
    assert debug["adverse_cash"] == pytest.approx(20.0)
    assert debug["adverse_inc_cash"] == pytest.approx(20.0)
    assert debug["risk_adverse"] < 0
    assert debug["risk_loss_time"] < 0
    assert debug["loss_steps"] == pytest.approx(1.0)

    reward = reward_fn()
    debug = _validate(reward_fn, reward)
    assert debug["loss_steps"] == pytest.approx(2.0)
    assert debug["risk_loss_time"] < 0

    env.user_accounts.unrealized_pnl = Decimal("5")
    reward = reward_fn()
    debug = _validate(reward_fn, reward)
    assert debug["loss_steps"] == pytest.approx(0.0)
    assert debug["risk_adverse"] == pytest.approx(0.0)
    assert debug["risk_loss_time"] == pytest.approx(0.0)
    assert debug["drawdown_inc_cash"] >= 0
    assert debug["adverse_inc_cash"] >= 0


def test_v2_fee_and_disabled_shaping_components():
    env = _FakeEnv()
    reward_fn = FuturesIntradayMTMRiskReward(env)
    reward_fn()

    env.fee_step = Decimal("3")
    env.ledger.cash -= Decimal("3")
    reward = reward_fn()
    debug = _validate(reward_fn, reward)
    assert debug["fee"] == pytest.approx(0.0)
    assert debug["fee_cash_debug"] == pytest.approx(3.0)
    for key in V2_DISABLED:
        assert debug[key] == pytest.approx(0.0), key
