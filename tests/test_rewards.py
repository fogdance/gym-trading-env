# tests/test_rewards.py

import unittest
from decimal import Decimal
from math import log1p, copysign
from gym_trading_env.envs.trading_env import ForexCode
from gym_trading_env.rewards.reward_functions import StepReward, CloseReward, EventReward, TerminationReward,FastCarRacingReward

# Mock 必要的环境类
class MockConfig:
    class Trading:
        initial_balance = Decimal('1000')
    class Risk:
        daily_lost_ratio = 0.05  # 5%
        max_drawdown_ratio = 0.1  # 10%
    trading = Trading()
    risk = Risk()

class MockPositionManager:
    def __init__(self, long_position):
        self.long_position = long_position
    def total_long_position(self):
        return self.long_position
    def total_short_position(self):
        return Decimal('0.0')

class MockUserAccounts:
    def __init__(self, equity_value):
        self._equity = Decimal(str(equity_value))
    def equity(self):
        return self._equity

class MockMetrics:
    def __init__(self, daily_lost_pct, drawdown_pct, max_profit=0, max_loss=0, calmar=None, sharpe=None):
        self.metrics = {
            'current_day_lost_pct': Decimal(str(daily_lost_pct * 100)),
            'current_drawdown_pct': Decimal(str(drawdown_pct * 100)),
            'max_profit': max_profit,
            'max_loss': max_loss,
            'calmar_ratio': calmar,
            'sharpe_ratio': sharpe
        }
    def get_metrics(self):
        return self.metrics

class MockEnv:
    def __init__(self, equity_value, daily_lost_pct=0.01, drawdown_pct=0.05, action_result=None, long_position=Decimal("0.01")):
        self.config = MockConfig()
        self.position_manager = MockPositionManager(long_position)
        self.user_accounts = MockUserAccounts(equity_value)
        self.action_result = action_result
        self.last_close_position = None
        self.metrics = MockMetrics(daily_lost_pct, drawdown_pct)

class TestRewards(unittest.TestCase):
    def setUp(self):
        # 初始化默认环境
        self.env_normal = MockEnv(equity_value=1000, daily_lost_pct=0.01, drawdown_pct=0.05)


    def test_step_reward_equity_decrease_with_no_position(self):
        """测试净值下降 + 空仓的边界情况"""
        env = MockEnv(equity_value=990, long_position=Decimal(0))  # 净值跌到 990
        step_reward = StepReward(env)
        step_reward.previous_equity = Decimal('1000')
        reward = step_reward()
        self.assertAlmostEqual(reward, -0.04, places=5)  # -0.03 (净值) - 0.01 (空仓)

    def test_step_reward_invalid_action(self):
        """测试无效动作的情况"""
        env = MockEnv(equity_value=1000, action_result=ForexCode.ERROR_OPEN_POSITION)
        step_reward = StepReward(env)
        step_reward.previous_equity = Decimal('1000')
        reward = step_reward()
        self.assertAlmostEqual(reward, -0.01, places=5)  # -0.01 (无效动作)

    # 测试 CloseReward
    def test_close_reward_positive_pnl(self):
        """测试盈利平仓，边界值"""
        env = MockEnv(equity_value=1000)
        env.last_close_position = {'pnl': 5.0, 'margin': 1.0}  # 高盈亏比
        close_reward = CloseReward(env)
        reward = close_reward()
        self.assertAlmostEqual(reward, 0.75, places=5)  # log1p(5.0) > 0.75，限制为 0.75

    def test_close_reward_negative_pnl(self):
        """测试亏损平仓，边界值"""
        env = MockEnv(equity_value=1000)
        env.last_close_position = {'pnl': -2.0, 'margin': 1.0}
        close_reward = CloseReward(env)
        reward = close_reward()
        self.assertAlmostEqual(reward, -0.5, places=5)  # log1p(2.0) > 0.5，限制为 -0.5

    def test_close_reward_no_position(self):
        """测试无平仓情况"""
        env = MockEnv(equity_value=1000)
        close_reward = CloseReward(env)
        reward = close_reward()
        self.assertEqual(reward, 0.0)

    # 测试 EventReward
    def test_event_reward_max_profit(self):
        """测试突破最大盈利"""
        env = MockEnv(equity_value=1000)
        env.last_close_position = {'pnl': 5.0}
        env.metrics = MockMetrics(0.01, 0.05, max_profit=4.0, max_loss=-2.0)
        event_reward = EventReward(env, once=0, repeated=0.5)
        reward = event_reward()
        self.assertAlmostEqual(reward, 1.0, places=5)  # 仅最大盈利触发

    def test_event_reward_max_loss(self):
        """测试突破最大亏损"""
        env = MockEnv(equity_value=1000)
        env.last_close_position = {'pnl': -3.0}
        env.metrics = MockMetrics(0.01, 0.05, max_profit=4.0, max_loss=-2.0)
        event_reward = EventReward(env, once=0, repeated=0.5)
        reward = event_reward()
        self.assertAlmostEqual(reward, -0.5, places=5)  # 仅最大亏损触发

    def test_event_reward_calmar_achieved(self):
        """测试 Calmar 比率达成"""
        env = MockEnv(equity_value=1000)
        env.metrics = MockMetrics(0.01, 0.05, calmar=1.5)
        event_reward = EventReward(env, once=0, repeated=0.5)
        reward = event_reward()
        self.assertAlmostEqual(reward, 0.5, places=5)  # 首次达成 0.5

    # 测试 TerminationReward
    def test_termination_reward_daily_loss_exceeded(self):
        """测试超日亏损终止"""
        env = MockEnv(equity_value=1000, daily_lost_pct=0.06)  # 6% > 5%
        termination_reward = TerminationReward(env)
        reward = termination_reward()
        self.assertEqual(reward, -2.0)

    def test_termination_reward_drawdown_exceeded(self):
        """测试超最大回撤终止"""
        env = MockEnv(equity_value=1000, drawdown_pct=0.15)  # 15% > 10%
        termination_reward = TerminationReward(env)
        reward = termination_reward()
        self.assertEqual(reward, -2.0)

    def test_termination_reward_normal(self):
        """测试正常情况无终止"""
        env = MockEnv(equity_value=1000, daily_lost_pct=0.01, drawdown_pct=0.05)
        termination_reward = TerminationReward(env)
        reward = termination_reward()
        self.assertEqual(reward, 0.0)

        # 更新 StepReward 测试
    def test_step_reward_equity_increase(self):
        env = MockEnv(equity_value=1010)
        step_reward = StepReward(env, equity_coeff=3.0, penalty=0.01)
        step_reward.previous_equity = Decimal('1000')
        reward = step_reward()
        self.assertAlmostEqual(reward, 0.03, places=5)

    # 测试 FastCarRacingReward
    def test_fast_car_racing_reward_range(self):
        """测试总奖励范围"""
        env = MockEnv(equity_value=990, daily_lost_pct=0.06)  # 触发终止
        env.last_close_position = {'pnl': -2.0, 'margin': 1.0}  # 负平仓
        config = {
            'step': {'equity_coeff': 3.0, 'penalty': 0.01},
            'close': {'min_reward': -0.5, 'max_reward': 0.75},
            'event': {'once': 0, 'repeated': 0.5, 'max_profit_reward': 1.0, 'max_loss_penalty': -0.5},
            'termination': {'limit': -2.0}
        }
        fast_reward = FastCarRacingReward(env, config)
        reward = fast_reward()
        # 预期：-2.0 (终止) + -0.5 (平仓) + -0.03 (净值) 被限制为 -2.0
        self.assertGreaterEqual(reward, -2.0)
        self.assertLessEqual(reward, 1.0)

    def test_fast_car_racing_reward_positive(self):
        """测试正向奖励范围"""
        env = MockEnv(equity_value=1010)
        env.last_close_position = {'pnl': 5.0, 'margin': 1.0}  # 正平仓
        env.metrics = MockMetrics(0.01, 0.05, max_profit=4.0, calmar=1.5)
        fast_reward = FastCarRacingReward(env)
        reward = fast_reward()
        # 预期：0.03 (净值) + 0.75 (平仓) + 1.0 (max_profit) + 0.5 (calmar) 被限制为 1.0
        self.assertGreaterEqual(reward, -2.0)
        self.assertLessEqual(reward, 1.0)

if __name__ == '__main__':
    unittest.main()