import pytest
from decimal import Decimal

pytestmark = pytest.mark.unit

from gym_trading_env.rewards.reward_functions import FuturesIntradayReward
from gym_trading_env.envs.action import ForexCode


class _CfgTrading:
    invalid_action_punish = 0.02
    invalid_time_cost_total = 0.2
    invalid_streak_cap = 10


class _CfgTraining:
    # 用 max_episode_steps 来固定 A 的 per-step
    max_episode_steps = 100
    episode_length = None


class _Cfg:
    trading = _CfgTrading()
    training = _CfgTraining()


class _UserAccounts:
    def equity(self) -> Decimal:
        return Decimal("10000")


class _FakeEnv:
    def __init__(self):
        self.config = _Cfg()
        self.user_accounts = _UserAccounts()

        # reward 里会访问这些字段（给默认即可）
        self.max_equity = Decimal("10000")
        self.fee_step = Decimal("0")
        self.stop_loss_fired = 0
        self.last_close_position = None

        self.current_step = 0
        self.end_idx = 0
        self._eod_idx = 0

        self._minutes_to_eod_last = 100
        self._R_cash_last = Decimal("100")  # scale 不为 0

        self.action_result = ForexCode.SUCCESS  # 默认


def _init_reward(env: _FakeEnv):
    r = FuturesIntradayReward(env)
    # 首次调用会初始化内部 prev_equity，不产生奖惩
    out0 = r()
    assert out0 == 0.0
    return r


def test_invalid_time_cost_per_step_is_total_over_episode_steps():
    env = _FakeEnv()
    r = _init_reward(env)

    # per-step = total / max_episode_steps = 0.2 / 100 = 0.002
    assert abs(r._invalid_time_cost_per_step - 0.002) < 1e-12


def test_invalid_first_time_only_pays_time_cost_not_streak():
    env = _FakeEnv()
    r = _init_reward(env)

    env.action_result = ForexCode.ERROR_NO_POSITION_TO_CLOSE
    out = r()

    dbg = env._reward_debug
    assert dbg["invalid_streak_len"] == 1
    # 第一次：streak 惩罚应为 0（只剩 time cost）
    assert dbg["invalid_streak"] == 0.0
    assert dbg["invalid_time"] < 0.0
    assert dbg["invalid_total"] == pytest.approx(dbg["invalid_time"], rel=0, abs=1e-12)


def test_invalid_repeated_increases_streak_penalty_linearly():
    env = _FakeEnv()
    r = _init_reward(env)

    env.action_result = ForexCode.ERROR_NO_POSITION_TO_CLOSE
    r()  # streak=1
    r()  # streak=2
    dbg2 = env._reward_debug
    assert dbg2["invalid_streak_len"] == 2
    assert dbg2["invalid_streak"] == pytest.approx(-0.02 * 1, abs=1e-12)

    r()  # streak=3
    dbg3 = env._reward_debug
    assert dbg3["invalid_streak_len"] == 3
    assert dbg3["invalid_streak"] == pytest.approx(-0.02 * 2, abs=1e-12)


def test_success_resets_invalid_streak():
    env = _FakeEnv()
    r = _init_reward(env)

    env.action_result = ForexCode.ERROR_NO_POSITION_TO_CLOSE
    r()
    r()
    assert env._reward_debug["invalid_streak_len"] == 2

    env.action_result = ForexCode.SUCCESS
    r()
    assert env._reward_debug["invalid_streak_len"] == 0
    assert env._reward_debug["invalid_total"] == 0.0


def test_market_closed_not_counted_as_invalid_and_not_in_streak():
    env = _FakeEnv()
    r = _init_reward(env)

    # 先让 streak=2
    env.action_result = ForexCode.ERROR_NO_POSITION_TO_CLOSE
    r()
    r()
    assert env._reward_debug["invalid_streak_len"] == 2

    # market closed：应该走 mkt_closed，不走 invalid，也不改变 streak（实现里 invalid=False -> streak reset）
    env.action_result = ForexCode.ERROR_MARKET_CLOSED
    r()
    dbg = env._reward_debug
    assert dbg["invalid_total"] == 0.0
    assert dbg["mkt_closed"] < 0.0
    # 注意：我们实现里 invalid=False 会把 streak reset 为 0
    assert dbg["invalid_streak_len"] == 0


def test_streak_capped():
    env = _FakeEnv()
    env.config.trading.invalid_streak_cap = 3
    r = _init_reward(env)

    env.action_result = ForexCode.ERROR_NO_POSITION_TO_CLOSE
    r()  # s=1
    r()  # s=2 -> -0.02*1
    r()  # s=3 -> -0.02*2
    r()  # s=4 but capped to 3 -> still -0.02*2

    dbg = env._reward_debug
    assert dbg["invalid_streak_len"] >= 4  # 记录的原始 streak 还在增长（如果你保留原值）
    # 但惩罚部分使用 min(streak, cap)
    assert dbg["invalid_streak"] == pytest.approx(-0.02 * 2, abs=1e-12)
