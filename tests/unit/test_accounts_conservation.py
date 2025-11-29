# tests/unit/test_accounts_conservation.py

from decimal import Decimal
from gym_trading_env.envs.action import ForexCode
from gym_trading_env.utils.decimal_util import D, D0


def _snapshot(env):
    return {
        "user_balance": env.user_accounts.balance.get_balance(),
        "user_margin": env.user_accounts.margin.get_balance(),
        "broker_balance": env.broker_accounts.balance.get_balance(),
        "broker_fees": env.broker_accounts.fees.get_balance(),
        "long_pos": env.user_accounts.long_position,
        "short_pos": env.user_accounts.short_position,
    }


def _delta(s1, s0, k):
    return s1[k] - s0[k]


def _consolidated_total(env):
    """
    Route A: margin 是 memo（占用/风控口径），不是真实现金腿，不参与“总资金守恒”。
    真实资金守恒口径 = user_balance + broker_balance + broker_fees
    """
    return (
        env.user_accounts.balance.get_balance()
        + env.broker_accounts.balance.get_balance()
        + env.broker_accounts.fees.get_balance()
    )


def _assert_double_entry_closed(s0, s1):
    """
    Route A 的“复式闭合”只检查真实资金腿闭合：
      Δuser_balance + Δbroker_balance + Δbroker_fees == 0

    注意：user_margin 不在此闭合检查范围内（它不是资金转移，只是占用指标）。
    """
    d_user = _delta(s1, s0, "user_balance")
    d_broker = _delta(s1, s0, "broker_balance")
    d_fee = _delta(s1, s0, "broker_fees")
    assert d_user + d_broker + d_fee == D0


def test_total_funds_conserved_success_long_open_close(env):
    """
    成功开/平仓（Route A）：
    - 真实资金口径(user_balance + broker_balance + broker_fees) 守恒
    - 开仓：margin 增加（memo 占用），broker_balance 不变
    - 平仓：回到 flat，margin 归零
    """
    spread = env.config.trading.spread

    total_before = _consolidated_total(env)
    s0 = _snapshot(env)

    # --- OPEN ---
    rc = env._long_open(price=D("1.1000"), spread=spread, slot=0)
    assert rc == ForexCode.SUCCESS

    s1 = _snapshot(env)

    # 真实资金守恒（不含 margin）
    assert _consolidated_total(env) == total_before

    # 真实资金腿复式闭合
    _assert_double_entry_closed(s0, s1)

    # 开仓：slot 0 有仓位，margin 增加 = initial_margin；broker_balance 不变
    pos0 = env.position_manager.get_position(0, is_long=True)
    assert pos0 is not None

    assert _delta(s1, s0, "broker_balance") == D0
    assert _delta(s1, s0, "user_margin") == pos0.initial_margin
    assert s1["long_pos"] == pos0.size
    assert s1["short_pos"] == D0

    # 手续费：user_balance 减少、broker_fees 增加（闭合已由 _assert_double_entry_closed 保证）
    open_fee = _delta(s1, s0, "broker_fees")
    assert open_fee >= D0
    assert _delta(s1, s0, "user_balance") == -open_fee

    # --- CLOSE ---
    rc = env._long_close(price=D("1.1000"), spread=spread, slot=0)
    assert rc == ForexCode.SUCCESS

    s2 = _snapshot(env)

    # 平仓后：回到 flat，margin 归零
    assert env.position_manager.no_position() is True
    assert s2["long_pos"] == D0
    assert s2["short_pos"] == D0
    assert s2["user_margin"] == D0

    # 真实资金口径依然守恒
    assert _consolidated_total(env) == total_before

    # 平仓这一步的复式闭合（真实资金腿）
    _assert_double_entry_closed(s1, s2)
