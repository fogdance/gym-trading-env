# src/gym_trading_env/rewards/reward_functions.py

from decimal import Decimal
from gym_trading_env.utils.conversion import decimal_to_float
from math import log1p, copysign
from collections import deque
from gym_trading_env.envs.action import ForexCode


class TotalPnlReward:
    """基于总盈亏（已实现 + 未实现）的奖励类"""
    def __init__(self, env):
        self.env = env
        self.previous_total_pnl = Decimal(0)

    def __call__(self):
        """计算奖励并更新状态"""
        # 计算当前总盈亏
        total_pnl = self.env.user_accounts.realized_pnl + self.env.user_accounts.unrealized_pnl
        
        # 计算奖励：当前总盈亏与上一步的差值
        reward = total_pnl - self.previous_total_pnl
        
        # 日志记录（调试用）
        self.env.logger.debug(f"Previous Total P&L: {self.previous_total_pnl}")
        self.env.logger.debug(f"Current Total P&L: {total_pnl}")
        self.env.logger.debug(f"Reward: {reward}")
        
        # 更新状态
        self.previous_total_pnl = total_pnl
        
        # 转换为 float，保留两位小数
        return float(decimal_to_float(reward, precision=2))


class CurrentBalanceReward:
    """基于当前余额的奖励类"""
    def __init__(self, env):
        self.env = env

    def __call__(self):
        """直接返回当前余额作为奖励"""
        balance = self.env.user_accounts.balance.get_balance()
        return float(decimal_to_float(balance, precision=2))

class FastCarRacingReward():
    def __init__(self, env):
        self.env = env

        self.rewards = [
            StepReward(env),
            CloseReward(env),
            EventReward(env, once=1, repeated=0.5),
            TerminationReward(env),
        ]

    def __call__(self):
        reward = sum([fn() for fn in self.rewards])
        self.env.last_close_position = None

        return reward

class StepReward:
    """每步奖励"""
    def __init__(self, env):
        self.env = env

    def __call__(self):
        reward = Decimal('0.0')
        
        # 不鼓励空仓
        if self.env.position_manager.total_long_position() == Decimal('0.0') and \
           self.env.position_manager.total_short_position() == Decimal('0.0'):
            reward -= Decimal('0.1')
        
        # 不鼓励对冲（多空持仓相等）
        if self.env.position_manager.total_long_position() == self.env.position_manager.total_short_position() and \
           self.env.position_manager.total_long_position() > Decimal('0.0'):
            reward -= Decimal('0.1')
        
        # 无效动作
        if self.env.action_result == ForexCode.ERROR_HIT_MAX_POSITION or self.env.action_result == ForexCode.ERROR_NO_POSITION_TO_CLOSE or self.env.action_result == ForexCode.ERROR_OPEN_POSITION:
            reward -= Decimal('0.1')
        
        # 净值增长奖励
        equity = self.env.user_accounts.equity()
        equity_change = (equity - self.env.previous_equity) / self.env.initial_balance
        reward += Decimal('0.05') * equity_change
        
        return float(reward)

class CloseReward:
    """平仓奖励"""
    def __init__(self, env):
        self.env = env

    def __call__(self):
        if self.env.last_close_position is None:
            return 0.0
        
        pnl = self.env.last_close_position['pnl']
        margin = self.env.last_close_position['margin']
        sign = copysign(1, float(pnl))  # 盈亏方向
        reward = sign * log1p(abs(float(pnl) / float(margin)))  # log1p(abs(pnl / margin))
        return min(reward, 2.0)  # 限制最大值为 2

class EventReward:
    """事件奖励"""
    def __init__(self, env, once=0, repeated=1):
        self.env = env
        self.once = once
        self.repeated = repeated
        self.calmar_ratio_goal = True
        self.sharpe_ratio_goal = True

    def __call__(self):
        reward = 0.0
        if self.env.last_close_position is not None:
            pnl = self.env.last_close_position['pnl']
            
            metrics = self.env.metrics.get_metrics()

            # 突破单笔最大盈利
            if pnl > metrics['max_profit']:
                reward += 2.0
                self.max_profit = pnl
            
            # 突破单笔最大亏损
            if pnl < metrics['max_loss']:
                reward -= 1.0
                self.max_loss = pnl

            if metrics['calmar_ratio'] is not None and metrics['calmar_ratio'] >= 1.0:
                if self.calmar_ratio_goal:
                    self.calmar_ratio_goal = False
                    reward += self.once
                reward += 2.0 * self.repeated
        
            if metrics['sharpe_ratio'] is not None and metrics['sharpe_ratio'] >= 1.0:
                if self.sharpe_ratio_goal:
                    self.sharpe_ratio_goal = False
                    reward += self.once
                reward += 2.0 * self.repeated
        
        return float(reward)

class TerminationReward:
    """终止条件奖励"""
    def __init__(self, env):
        self.env =env

    def __call__(self):
        reward = 0.0
        metrics = self.env.metrics.get_metrics()
        daily_lost_pct = decimal_to_float(metrics['current_day_lost_pct'] / Decimal('100.0'))
        drawdown_pct = decimal_to_float(metrics['current_drawdown_pct'] / Decimal('100.0')) 
        if daily_lost_pct > self.env.daily_lost_ratio:
            reward = -5.0
        if drawdown_pct > self.env.max_drawdown_ratio:
            reward = -5.0
        return float(reward)


reward_classes = {
    'current_balance_reward_function': CurrentBalanceReward,
    'total_pnl_reward_function': TotalPnlReward,
    'fast_car_racing_likely_reward_function': FastCarRacingReward,
}
