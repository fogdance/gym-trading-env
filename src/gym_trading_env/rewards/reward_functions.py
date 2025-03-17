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

class StepReward:
    """每步奖励"""
    def __init__(self, env, equity_coeff=3.0, penalty=0.01):
        self.env = env
        self.equity_coeff = Decimal(str(equity_coeff))
        self.penalty = Decimal(str(penalty))
        self.previous_equity = env.config.trading.initial_balance

    def __call__(self):
        reward = Decimal('0.0')
        
        # 不鼓励空仓
        if self.env.position_manager.total_long_position() == Decimal('0.0') and \
           self.env.position_manager.total_short_position() == Decimal('0.0'):
            reward -= self.penalty
        # 不鼓励对冲（多空持仓相等）
        if self.env.position_manager.total_long_position() == self.env.position_manager.total_short_position() and \
           self.env.position_manager.total_long_position() > Decimal('0.0'):
            reward -= self.penalty
        # 无效动作
        if self.env.action_result in [ForexCode.ERROR_HIT_MAX_POSITION, 
                                      ForexCode.ERROR_NO_POSITION_TO_CLOSE, 
                                      ForexCode.ERROR_OPEN_POSITION]:
            reward -= self.penalty
        # 净值增长奖励
        equity = self.env.user_accounts.equity()
        equity_change = (equity - self.previous_equity) / self.env.config.trading.initial_balance
        reward += self.equity_coeff * equity_change
        self.previous_equity = equity
        return float(reward)

class CloseReward:
    """平仓奖励"""
    def __init__(self, env, min_reward=-0.5, max_reward=0.75):
        self.env = env
        self.min_reward = min_reward
        self.max_reward = max_reward

    def __call__(self):
        if self.env.last_close_position is None:
            return 0.0
        pnl = self.env.last_close_position['pnl']
        margin = self.env.last_close_position['margin']
        sign = copysign(1, float(pnl))
        reward = sign * log1p(abs(float(pnl) / float(margin)))  # log1p(abs(pnl / margin))
        return min(max(reward, self.min_reward), self.max_reward)

class FastCarRacingReward:
    def __init__(self, env, config=None):
        self.env = env
        self.lower_limit = -2.0
        self.upper_limit = 1.0
        # 默认配置
        default_config = {
            'step': {'equity_coeff': 3.0, 'penalty': 0.01},
            'close': {'min_reward': -0.5, 'max_reward': 0.75},
            'event': {'once': 0, 'repeated': 0.5, 'max_profit_reward': 1.0, 'max_loss_penalty': -0.5},
            'termination': {'limit': -2.0}
        }
        self.config = config if config else default_config

        self.rewards = [
            StepReward(env, **self.config['step']),
            CloseReward(env, **self.config['close']),
            EventReward(env, once=self.config['event']['once'], repeated=self.config['event']['repeated']),
            TerminationReward(env, self.config['termination']['limit']),
        ]

    def __call__(self):
        reward = sum([fn() for fn in self.rewards])
        self.env.last_close_position = None
        return max(self.lower_limit, min(self.upper_limit, reward))


class EventReward:
    """事件奖励"""
    def __init__(self, env, once=0, repeated=1):
        self.env = env
        self.once = once  # 调整为 0，避免重复奖励
        self.repeated = repeated  # 保持 0.5，累计奖励 1.0
        self.calmar_ratio_goal = True
        self.sharpe_ratio_goal = True

    def __call__(self):
        reward = 0.0
        metrics = self.env.metrics.get_metrics()

        if self.env.last_close_position is not None:
            pnl = self.env.last_close_position['pnl']  # float

            # 突破单笔最大盈利
            if pnl > metrics['max_profit']:
                reward += 1.0  # 调整为 1.0
            
            # 突破单笔最大亏损
            if pnl < metrics['max_loss']:
                reward -= 0.5  # 调整为 -0.5

        # Calmar 和 Sharpe 比率
        if metrics['calmar_ratio'] is not None and metrics['calmar_ratio'] >= 1.0:
            if self.calmar_ratio_goal:
                self.calmar_ratio_goal = False
                reward += self.once  # 0
            reward += 1.0 * self.repeated  # 0.5
        
        if metrics['sharpe_ratio'] is not None and metrics['sharpe_ratio'] >= 1.0:
            if self.sharpe_ratio_goal:
                self.sharpe_ratio_goal = False
                reward += self.once  # 0
            reward += 1.0 * self.repeated  # 0.5
        
        return float(reward)

class TerminationReward:
    """终止条件奖励"""
    def __init__(self, env, limit=-2):
        self.env = env
        self.limit = limit

    def __call__(self):
        reward = 0.0
        metrics = self.env.metrics.get_metrics()
        daily_lost_pct = decimal_to_float(metrics['current_day_lost_pct'] / Decimal('100.0'))
        drawdown_pct = decimal_to_float(metrics['current_drawdown_pct'] / Decimal('100.0')) 
        if daily_lost_pct > self.env.config.risk.daily_lost_ratio:
            reward = self.limit
        if drawdown_pct > self.env.config.risk.max_drawdown_ratio:
            reward = self.limit
        return float(reward)

reward_classes = {
    'current_balance_reward_function': CurrentBalanceReward,
    'total_pnl_reward_function': TotalPnlReward,
    'fast_car_racing_likely_reward_function': FastCarRacingReward,
}
