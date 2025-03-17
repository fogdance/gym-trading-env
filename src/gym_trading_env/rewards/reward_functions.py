# src/gym_trading_env/rewards/reward_functions.py

from decimal import Decimal
from gym_trading_env.utils.conversion import decimal_to_float
from math import log1p, copysign
from collections import deque
from gym_trading_env.envs.action import ForexCode
import numpy as np
import talib


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
    def __init__(self, env, profit_coeff=2, penalty=0.01):
        self.env = env
        self.penalty = Decimal(str(penalty))
        self.previous_equity = env.config.trading.initial_balance
        self.initial_balance = env.config.trading.initial_balance
        self.empty_position_count = 0
        self.max_empty_position_count = 10
        self.hedge_count = 0
        self.max_hedge_count = 10
        self.invalid_action = 0
        self.max_invalid_action = 5

        self.profit_coeff = Decimal(str(profit_coeff))  # 每美元收益系数
        self.short_atr_period = 25  # 短期 ATR 周期
        self.long_atr_period = 100  # 长期 ATR 周期

    def __call__(self):
        reward = Decimal('0.0')
        
        # 不鼓励空仓
        if self.env.position_manager.total_long_position() == Decimal('0.0') and \
           self.env.position_manager.total_short_position() == Decimal('0.0'):
            self.empty_position_count += 1
            if self.empty_position_count >= self.max_empty_position_count:
                self.empty_position_count = 0
                reward -= self.penalty
        else:
            self.empty_position_count = 0

        # 不鼓励对冲（多空持仓相等）
        if self.env.position_manager.total_long_position() == self.env.position_manager.total_short_position() and \
           self.env.position_manager.total_long_position() > Decimal('0.0'):
            self.hedge_count += 1
            if self.hedge_count >= self.max_hedge_count:
                self.hedge_count = 0
                reward -= self.penalty
        else:
            self.hedge_count = 0

        # 无效动作
        if self.env.action_result in [ForexCode.ERROR_HIT_MAX_POSITION, 
                                      ForexCode.ERROR_NO_POSITION_TO_CLOSE, 
                                      ForexCode.ERROR_OPEN_POSITION]:
            self.invalid_action += 1
            if self.invalid_action >= self.max_invalid_action:
                self.invalid_action = 0
                reward -= self.penalty
        else:
            self.invalid_action = 0


        # 净值增长奖励
        equity = self.env.user_accounts.equity()
        equity_change = equity - self.previous_equity
        if equity_change != Decimal('0.0'):
            current_price = self.env.current_price
            
            # 计算短期和长期波动率
            df_short = self.env.df_window.tail(self.short_atr_period*2)
            df_long = self.env.df_window.tail(self.long_atr_period*2)
            high_short = np.array(df_short['High'], dtype=float)
            low_short = np.array(df_short['Low'], dtype=float)
            close_short = np.array(df_short['Close'], dtype=float)
            high_long = np.array(df_long['High'], dtype=float)
            low_long = np.array(df_long['Low'], dtype=float)
            close_long = np.array(df_long['Close'], dtype=float)
            
            short_atr = talib.ATR(high_short, low_short, close_short, timeperiod=self.short_atr_period)[-1]
            long_atr = talib.ATR(high_long, low_long, close_long, timeperiod=self.long_atr_period)[-1]
            short_vol = Decimal(str(short_atr / float(current_price)))
            long_vol = Decimal(str(long_atr / float(current_price)))
            
            # 调整系数
            adjusted_coeff = self.profit_coeff * (long_vol / short_vol)
            
            # 标准化收益
            profit_ratio = abs(equity_change) / self.initial_balance * 100
            sign = Decimal('1.0') if equity_change > 0 else Decimal('-1.0')
            reward += adjusted_coeff * Decimal(str(log1p(float(profit_ratio)))) * sign

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
            'step': {'profit_coeff': 2, 'penalty': 0.005},
            'close': {'min_reward': -0.5, 'max_reward': 1.0},
            'event': {'repeated': 0.5, 'max_profit_reward': 1.0, 'max_loss_penalty': -0.5},
            'termination': {'limit': -2.0}
        }
        self.config = config if config else default_config

        self.rewards = [
            StepReward(env, **self.config['step']),
            CloseReward(env, **self.config['close']),
            EventReward(env, repeated=self.config['event']['repeated'], max_profit_reward=self.config['event']['max_profit_reward'], max_loss_penalty=self.config['event']['max_loss_penalty']),
            TerminationReward(env, self.config['termination']['limit']),
        ]

    def __call__(self):
        reward = sum([fn() for fn in self.rewards])
        self.env.last_close_position = None
        return max(self.lower_limit, min(self.upper_limit, reward))


class EventReward:
    """事件奖励"""
    def __init__(self, env, repeated=1, max_profit_reward=1, max_loss_penalty=-0.5):
        self.env = env
        self.max_profit_reward = max_profit_reward
        self.max_loss_penalty = max_loss_penalty
        self.repeated = repeated  # 保持 0.5，累计奖励 1.0
        # 默认阈值和奖励
        self.thresholds = {
            'calmar': {0.5: 0.2, 0.6: 0.5, 0.8: 0.8},
            'sharpe': {0.5: 0.2, 0.6: 0.5, 0.8: 0.8}
        }

    def __call__(self):
        reward = 0.0
        metrics = self.env.metrics.get_metrics()

        if self.env.last_close_position is not None:
            pnl = self.env.last_close_position['pnl']  # float

            # 突破单笔最大盈利
            if pnl > metrics['max_profit']:
                reward += self.max_profit_reward  # 调整为 1.0
            
            # 突破单笔最大亏损
            if pnl < metrics['max_loss']:
                reward += self.max_loss_penalty  # 调整为 -0.5

        # Calmar 分级奖励
        if metrics['calmar_ratio'] is not None:
            calmar = metrics['calmar_ratio']
            max_calmar_reward = 0.0
            for threshold, value in sorted(self.thresholds['calmar'].items()):
                if calmar >= threshold:
                    max_calmar_reward = value  # 取当前最高阈值的奖励
            reward += max_calmar_reward

        # Sharpe 分级奖励
        if metrics['sharpe_ratio'] is not None:
            sharpe = metrics['sharpe_ratio']
            max_sharpe_reward = 0.0
            for threshold, value in sorted(self.thresholds['sharpe'].items()):
                if sharpe >= threshold:
                    max_sharpe_reward = value  # 取当前最高阈值的奖励
            reward += max_sharpe_reward

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
