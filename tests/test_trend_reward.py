# tests/test_trend_reward.py

import unittest
from decimal import Decimal
from gym_trading_env.rewards.reward_functions import TrendReward
import numpy as np
from gym_trading_env.envs.action import Action
import pandas as pd

class MockEnv:
    def __init__(self, equity, long_pos, short_pos, df_window, action=Action.LONG_OPEN0, last_close=None):
        self.config = type('Config', (), {'trading': type('Trading', (), {'initial_balance': Decimal('1000')})})()
        self.user_accounts = type('Accounts', (), {'equity': lambda self: Decimal(str(equity))})()
        self.position_manager = type('Manager', (), {
            'total_long_position': lambda self: Decimal(str(long_pos)),
            'total_short_position': lambda self: Decimal(str(short_pos))
        })()
        self.df_window = df_window
        self.action = action
        self.last_close_position = last_close

class TestTrendReward(unittest.TestCase):
    def setUp(self):
        # 模拟 df_window，20 周期 ATR ≈ 0.0012 (EUR/USD)
        self.df_window = pd.DataFrame({
            'High': np.array([1.1] * 200, dtype=float),
            'Low': np.array([1.0] * 200, dtype=float),
            'Close': np.array([1.05] * 200, dtype=float)
        })

    # 测试顺 1h 趋势在接近前低时开多仓的奖励，预期 0.02
    # 当前价 1.0512 在 l1_1h (1.05) 的 2*ATR (0.0024) 范围内
    def test_open_reward_long_near_low(self):
        obs = {'indicators': [1.05, 1.0, 1.04, 0.99,  # 5m
                              1.05, 1.0, 1.04, 0.99,  # 15m 上涨
                              1.1, 1.05, 1.09, 1.04,  # 1h 上涨
                              1.0512]}  # 在 l1_1h (1.05) + 2*ATR (0.0024) 内
        env = MockEnv(1000, 0.01, 0, self.df_window, action=Action(Action.LONG_OPEN))
        reward = TrendReward(env)
        result = reward(obs)
        expected = 0.2 * 1 * 1 * 0.1  # 0.02
        self.assertAlmostEqual(result, expected, places=5)

    # 测试逆 1h 趋势开空仓的奖励，预期 0（不鼓励）
    # 当前价 1.1 未在 h1_1h (1.1) 的 2*ATR 范围内，但逆势无奖励
    def test_open_reward_short_reverse_trend(self):
        obs = {'indicators': [1.05, 1.0, 1.04, 0.99,  # 5m
                              1.05, 1.0, 1.04, 0.99,  # 15m 上涨
                              1.1, 1.05, 1.09, 1.04,  # 1h 上涨
                              1.1]}  # 未在 h1_1h 附近
        env = MockEnv(1000, 0, 0.01, self.df_window, action=Action(Action.SHORT_OPEN))
        reward = TrendReward(env)
        result = reward(obs)
        expected = 0.0  # 逆 1h 不鼓励
        self.assertAlmostEqual(result, expected, places=5)

    # # 测试顺 1h 趋势持多仓并突破前高的奖励，预期 0.7
    # # 当前价 1.11 突破 h1_1h (1.1)，趋势和突破奖励叠加
    # def test_hold_reward_long_breakout(self):
    #     obs = {'indicators': [1.05, 1.0, 1.04, 0.99,  # 5m
    #                           1.05, 1.0, 1.04, 0.99,  # 15m 上涨
    #                           1.1, 1.05, 1.09, 1.04,  # 1h 上涨
    #                           1.11]}  # 突破 h1_1h
    #     env = MockEnv(1001.2, 0.01, 0, self.df_window, action=Action(Action.HOLD))
    #     reward = TrendReward(env)
    #     result = reward(obs)
    #     trend = 0.2 * (0.7 * 1 + 0.3 * 1)  # 0.2
    #     breakout = 0.5 * 1  # 0.5
    #     expected = trend + breakout  # 0.7
    #     self.assertAlmostEqual(result, expected, places=2)

    # # 测试顺 1h 上涨持空仓并突破前高的惩罚，预期 -0.25
    # def test_hold_reward_short_reverse_breakout(self):
    #     obs = {'indicators': [1.05, 1.0, 1.04, 0.99,  # 5m
    #                           1.05, 1.0, 1.04, 0.99,  # 15m 上涨
    #                           1.1, 1.05, 1.09, 1.04,  # 1h 上涨
    #                           1.11]}  # 突破 h1_1h
    #     env = MockEnv(998.8, 0, 0.01, self.df_window, action=Action(Action.HOLD))
    #     reward = TrendReward(env)
    #     result = reward(obs)
    #     trend = 0
    #     breakout = 0.5*-0.5  # 空仓突破 h1 小惩罚
    #     expected = breakout + trend
    #     self.assertAlmostEqual(result, expected, places=2)


    # 测试多仓止盈接近 1h 前高的奖励，预期 0.5
    # 当前价 1.0976 在 h1_1h (1.1) 的 2*ATR (0.0024) 范围内
    def test_close_reward_take_profit(self):
        obs = {'indicators': [0, 0, 0, 0,  # 5m
                              0, 0, 0, 0,  # 15m
                              1.1, 1.05, 1.09, 1.04,  # 1h
                              1.0976]}  # 在 h1_1h (1.1) - 2*ATR (0.0024) 内
        last_close = {'type': 'LONG', 'pnl': 5.0}
        env = MockEnv(1005, 0, 0, self.df_window, action=Action(Action.LONG_CLOSE), last_close=last_close)
        reward = TrendReward(env)
        result = reward(obs)
        expected = 0.5 * 1  # 0.5
        self.assertAlmostEqual(result, expected, places=2)

    # 测试多仓止损跌破 1h 前低的奖励，预期 0.5
    # 当前价 1.04 < l1_1h (1.05)，触发止损奖励
    def test_close_reward_stop_loss(self):
        obs = {'indicators': [0, 0, 0, 0,  # 5m
                              0, 0, 0, 0,  # 15m
                              1.1, 1.05, 1.09, 1.04,  # 1h
                              1.04]}  # 跌破 l1_1h
        last_close = {'type': 'LONG', 'pnl': -3.0}
        env = MockEnv(997, 0, 0, self.df_window, action=Action(Action.LONG_CLOSE), last_close=last_close)
        reward = TrendReward(env)
        result = reward(obs)
        expected = 0.5 * 1.5
        self.assertAlmostEqual(result, expected, places=2)

if __name__ == '__main__':
    unittest.main()