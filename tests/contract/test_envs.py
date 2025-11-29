# tests/test_envs.py
import unittest
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from gym_trading_env.envs.trading_env import CustomTradingEnv, Action
from gym_trading_env.utils.decimal_util import D, D0, decimal_to_float

pytestmark = pytest.mark.unit


def make_one_day_minute_df(start="2020-01-01 21:01:00", minutes=1440):
    """
    当前系统固定 1 天分时图（1440），episode_length=1440 且必须从 minute=0 起步。
    """
    idx = pd.date_range(start=start, periods=minutes, freq="min")
    base = 1.1000
    drift = np.linspace(0, 0.0020, minutes)  # ~20 pips/天
    c = base + drift
    return pd.DataFrame(
        {
            "Date": idx,
            "Open": c,
            "High": c,
            "Low": c,
            "Close": c,
            "Volume": np.ones(minutes, dtype=float),
        }
    )


def total_funds(info) -> Decimal:
    """
    真实资产口径（用于“看起来不丢钱”的守恒检查）：
    - balance: 可用现金（free cash）
    - used_margin: 被占用/冻结的保证金（escrow腿）
    - broker_balance + fees_collected: 经纪商相关腿
    """
    return (
        D(info["balance"])
        + D(info["used_margin"])
        + D(info["broker_balance"])
        + D(info["fees_collected"])
    )



def total_pnl(info) -> Decimal:
    return D(info["realized_pnl"]) + D(info["unrealized_pnl"])


def fee_per_side(env) -> Decimal:
    return D(env.config.trading.trade_lot) * D(env.config.trading.trading_fee_per_lot)


def action_idx(env, act: Action) -> int:
    """
    Dreamer 离散动作契约：step() 喂的是 0..N-1 的类别 id（也就是 valid_actions 的 index）
    """
    return int(env.valid_actions.index(act))


def step_by_action(env, act: Action):
    return env.step(action_idx(env, act))


def pick_action(env, preferred: str, fallback: str) -> Action:
    """
    兼容 Action 是否带 0 slot 的命名。
    """
    if hasattr(Action, preferred):
        return getattr(Action, preferred)
    return getattr(Action, fallback)


class TestCustomTradingEnv(unittest.TestCase):
    def setUp(self):
        df = make_one_day_minute_df()
        self.env = CustomTradingEnv(df=df, config_path="tests/test.yaml")
        self.env.config.training.randomize_start = False
        self.env.reset()

        # 常用动作（兼容有没有 *0）
        self.A_HOLD = Action.HOLD
        self.A_LONG_OPEN = pick_action(self.env, "LONG_OPEN0", "LONG_OPEN")
        self.A_LONG_CLOSE = pick_action(self.env, "LONG_CLOSE0", "LONG_CLOSE")
        self.A_SHORT_OPEN = pick_action(self.env, "SHORT_OPEN0", "SHORT_OPEN")
        self.A_SHORT_CLOSE = pick_action(self.env, "SHORT_CLOSE0", "SHORT_CLOSE")

    def tearDown(self):
        self.env.close()

    # -------------------- reset 契约（对外 info + 基础资金守恒） --------------------

    def test_reset_contract(self):
        obs, info = self.env.reset()

        # 基础字段存在（对外契约）
        for k in [
            "balance",
            "equity",
            "used_margin",
            "free_margin",
            "long_position",
            "short_position",
            "unrealized_pnl",
            "realized_pnl",
            "fees_collected",
            "broker_balance",
        ]:
            self.assertIn(k, info)

        # reset 后应该 flat
        self.assertEqual(D(info["long_position"]), D0)
        self.assertEqual(D(info["short_position"]), D0)
        self.assertEqual(D(info["unrealized_pnl"]), D0)
        self.assertEqual(D(info["realized_pnl"]), D0)
        self.assertEqual(D(info["fees_collected"]), D0)

        # equity/balance 初始一致（浮动容忍，但这里用 Decimal 直接等）
        self.assertEqual(D(info["equity"]), D(info["balance"]))
        self.assertEqual(D(info["used_margin"]), D0)
        self.assertEqual(D(info["free_margin"]), D(info["equity"]))

        # 真实资金口径：reset 后 total_funds == 初始 balance(+broker+fees)
        self.assertEqual(total_funds(info), D(info["balance"]) + D(info["broker_balance"]) + D(info["fees_collected"]))

    # -------------------- roundtrip 回归（long/short 各一条） --------------------

    def test_long_roundtrip_conservation_fee_reward(self):
        _, info0 = self.env.reset()
        tf0 = total_funds(info0)
        fp = fee_per_side(self.env)
        is_round_turn = bool(self.env.config.trading.is_round_turn)

        # --- OPEN ---
        _, r1, term1, trunc1, info1 = step_by_action(self.env, self.A_LONG_OPEN)

        # 资金守恒
        self.assertEqual(total_funds(info1), tf0)

        # 开仓后应持有 long（short 为 0）
        self.assertGreater(D(info1["long_position"]), D0)
        self.assertEqual(D(info1["short_position"]), D0)

        # 开仓手续费（成功开仓才应收）
        fee_open = D(info1["fees_collected"]) - D(info0["fees_collected"])
        self.assertEqual(fee_open, fp)

        pnl_before_close = total_pnl(info1)

        # --- CLOSE ---
        _, r2, term2, trunc2, info2 = step_by_action(self.env, self.A_LONG_CLOSE)

        # 平仓回到 flat
        self.assertEqual(D(info2["long_position"]), D0)
        self.assertEqual(D(info2["short_position"]), D0)
        self.assertEqual(D(info2["unrealized_pnl"]), D0)

        # 资金守恒仍成立
        self.assertEqual(total_funds(info2), tf0)

        # 平仓手续费：round-turn 才收
        fee_close = D(info2["fees_collected"]) - D(info1["fees_collected"])
        self.assertEqual(fee_close, fp if is_round_turn else D0)

        # reward 契约：Δtotal_pnl（沿用你旧测试口径）
        expected_reward = total_pnl(info2) - pnl_before_close
        self.assertAlmostEqual(
            r2,
            float(decimal_to_float(expected_reward, precision=2)),
            places=2,
        )

        self.assertFalse(trunc1)
        self.assertFalse(trunc2)

    def test_short_roundtrip_conservation_fee_reward(self):
        _, info0 = self.env.reset()
        tf0 = total_funds(info0)
        fp = fee_per_side(self.env)
        is_round_turn = bool(self.env.config.trading.is_round_turn)

        # --- OPEN ---
        _, r1, term1, trunc1, info1 = step_by_action(self.env, self.A_SHORT_OPEN)

        self.assertEqual(total_funds(info1), tf0)
        self.assertGreater(D(info1["short_position"]), D0)
        self.assertEqual(D(info1["long_position"]), D0)

        fee_open = D(info1["fees_collected"]) - D(info0["fees_collected"])
        self.assertEqual(fee_open, fp)

        pnl_before_close = total_pnl(info1)

        # --- CLOSE ---
        _, r2, term2, trunc2, info2 = step_by_action(self.env, self.A_SHORT_CLOSE)

        self.assertEqual(D(info2["short_position"]), D0)
        self.assertEqual(D(info2["long_position"]), D0)
        self.assertEqual(D(info2["unrealized_pnl"]), D0)

        self.assertEqual(total_funds(info2), tf0)

        fee_close = D(info2["fees_collected"]) - D(info1["fees_collected"])
        self.assertEqual(fee_close, fp if is_round_turn else D0)

        expected_reward = total_pnl(info2) - pnl_before_close
        self.assertAlmostEqual(
            r2,
            float(decimal_to_float(expected_reward, precision=2)),
            places=2,
        )

        self.assertFalse(trunc1)
        self.assertFalse(trunc2)

    # -------------------- fee 累积（不手算价格，只验证“成功交易才计费”） --------------------

    def test_fee_accumulates_on_multiple_opens(self):
        _, info0 = self.env.reset()
        tf0 = total_funds(info0)
        fp = fee_per_side(self.env)

        # 第一次 open
        _, _, _, _, info1 = step_by_action(self.env, self.A_LONG_OPEN)
        self.assertEqual(total_funds(info1), tf0)
        self.assertEqual(D(info1["fees_collected"]) - D(info0["fees_collected"]), fp)

        # 第二次 open：可能成功加仓，也可能因上限被拒绝
        prev_pos = D(info1["long_position"])
        prev_fee = D(info1["fees_collected"])
        _, _, _, _, info2 = step_by_action(self.env, self.A_LONG_OPEN)
        self.assertEqual(total_funds(info2), tf0)

        if D(info2["long_position"]) > prev_pos:
            # 成功加仓 -> 费用再加一笔
            self.assertEqual(D(info2["fees_collected"]) - prev_fee, fp)
        else:
            # 未加仓 -> 不应多扣费（否则训练会被“假交易”污染）
            self.assertEqual(D(info2["fees_collected"]) - prev_fee, D0)

    # -------------------- position limit（只测“不越界 + 守恒 + 成功才计费”） --------------------

    def test_position_limit_long(self):
        _, info0 = self.env.reset()
        tf0 = total_funds(info0)
        fp = fee_per_side(self.env)

        max_pos = D(self.env.config.trading.max_long_position)

        info = info0
        for _ in range(10):  # 多开几次，确保触发上限逻辑
            prev_pos = D(info["long_position"])
            prev_fee = D(info["fees_collected"])

            _, _, _, _, info = step_by_action(self.env, self.A_LONG_OPEN)

            self.assertEqual(total_funds(info), tf0)
            self.assertEqual(D(info["short_position"]), D0)
            self.assertLessEqual(D(info["long_position"]), max_pos)

            if D(info["long_position"]) > prev_pos:
                self.assertEqual(D(info["fees_collected"]) - prev_fee, fp)
            else:
                self.assertEqual(D(info["fees_collected"]) - prev_fee, D0)

    def test_position_limit_short(self):
        _, info0 = self.env.reset()
        tf0 = total_funds(info0)
        fp = fee_per_side(self.env)

        max_pos = D(self.env.config.trading.max_short_position)

        info = info0
        for _ in range(10):
            prev_pos = D(info["short_position"])
            prev_fee = D(info["fees_collected"])

            _, _, _, _, info = step_by_action(self.env, self.A_SHORT_OPEN)

            self.assertEqual(total_funds(info), tf0)
            self.assertEqual(D(info["long_position"]), D0)
            self.assertLessEqual(D(info["short_position"]), max_pos)

            if D(info["short_position"]) > prev_pos:
                self.assertEqual(D(info["fees_collected"]) - prev_fee, fp)
            else:
                self.assertEqual(D(info["fees_collected"]) - prev_fee, D0)


if __name__ == "__main__":
    unittest.main()
