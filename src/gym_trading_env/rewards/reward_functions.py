# src/gym_trading_env/rewards/reward_functions.py

from decimal import Decimal
from gym_trading_env.utils.conversion import decimal_to_float, float_to_decimal
from math import log1p, copysign
from collections import deque
from gym_trading_env.envs.action import ForexCode
import numpy as np
import talib

from gym_trading_env.utils.trade_util import calc_unrealized_pnl

class TotalPnlReward:
    """基于总盈亏（已实现 + 未实现）的奖励类"""
    def __init__(self, env):
        self.env = env
        self.previous_total_pnl = Decimal(0)

    def __call__(self, obs=None):
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

    def __call__(self, obs=None):
        """直接返回当前余额作为奖励"""
        balance = self.env.user_accounts.balance.get_balance()
        return float(decimal_to_float(balance, precision=2))

class StepReward:
    """每步奖励"""
    def __init__(self, env, profit_coeff=2, penalty=0.01, min_reward=-0.5, max_reward=0.75):
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
        self.min_reward = min_reward
        self.max_reward = max_reward
        
        self.profit_coeff = Decimal(str(profit_coeff))  # 每美元收益系数
        self.short_atr_period = 25  # 短期 ATR 周期
        self.long_atr_period = 100  # 长期 ATR 周期

    def __call__(self, obs=None):
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

        # 持仓收益奖励
        def pos_reward(pos, long: bool):
            if pos is None:
                return Decimal('0')

            pnl = calc_unrealized_pnl(self.env.current_price, pos, self.env.config.trading.lot_size, long)

            sign = copysign(1, float(pnl))
            reward = sign * log1p(abs(float(pnl) / float(pos.initial_margin)))  # log1p(abs(pnl / margin))
            reward = min(max(reward, self.min_reward), self.max_reward)
            return float_to_decimal(reward)

        for pos in self.env.position_manager.long_positions:
            reward += pos_reward(pos, True)
        for pos in self.env.position_manager.short_positions:
            reward += pos_reward(pos, False)

        return float(reward)

class CloseReward:
    """平仓奖励"""
    def __init__(self, env, min_reward=-0.5, max_reward=0.75):
        self.env = env
        self.min_reward = min_reward
        self.max_reward = max_reward

    def __call__(self, obs=None):
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
            TrendReward(env),
        ]

    def __call__(self, obs=None):
        reward = sum([fn(obs) for fn in self.rewards])
        self.env.last_close_position = None
        return max(self.lower_limit, min(self.upper_limit, reward))



class TrendReward:
    """趋势奖励类，支持多时间框架趋势跟踪和多交易对标准化"""
    def __init__(self, env, open_coeff=0.2, profit_coeff=0.2, trend_coeff=0.2, breakout_coeff=0.5, close_coeff=0.5):
        self.env = env
        self.open_coeff = Decimal(str(open_coeff))
        self.profit_coeff = Decimal(str(profit_coeff))
        self.trend_coeff = Decimal(str(trend_coeff))
        self.breakout_coeff = Decimal(str(breakout_coeff))
        self.close_coeff = Decimal(str(close_coeff))
        self.w1 = Decimal('0.7')  # 1h 趋势权重
        self.w2 = Decimal('0.3')  # 15m 趋势权重
        self.max_position = Decimal('0.1')  # 最大仓位标准化
        self.initial_balance = env.config.trading.initial_balance
        self.long_atr_period = 100
        self.short_atr_period = 20  # 新增短期 ATR 周期
        self.last_action = None

    def __call__(self, obs=None):
        reward = Decimal('0.0')
        indicators = obs['indicators']
        try:
            current_price = Decimal(str(indicators[-1]))  # 当前价格 (Close)
            long_pos = self.env.position_manager.total_long_position()
            short_pos = self.env.position_manager.total_short_position()
            net_position = long_pos - short_pos
            position_factor = min(abs(net_position) / self.max_position, Decimal('1.0'))

            # 提取趋势指标
            h1_1h, l1_1h, h2_1h, l2_1h = map(Decimal, map(str, indicators[8:12]))  # 1h
            h1_15m, l1_15m, h2_15m, l2_15m = map(Decimal, map(str, indicators[4:8]))  # 15m
            h1_5m, l1_5m, h2_5m, l2_5m = map(Decimal, map(str, indicators[0:4]))  # 5m

            # 计算 20 周期 ATR
            df_short = self.env.df_window.tail(self.short_atr_period * 2)
            high_short = np.array(df_short['High'], dtype=float)
            low_short = np.array(df_short['Low'], dtype=float)
            close_short = np.array(df_short['Close'], dtype=float)
            short_atr = Decimal(str(talib.ATR(high_short, low_short, close_short, timeperiod=self.short_atr_period)[-1]))
            atr_threshold = short_atr * Decimal('2.0')  # 2 个 ATR

            # 趋势一致性
            trend_align_1h = self._get_trend_alignment(h1_1h, l1_1h, h2_1h, l2_1h, net_position)
            trend_align_15m = self._get_trend_alignment(h1_15m, l1_15m, h2_15m, l2_15m, net_position)

            # 1. 开仓奖励
            if 'OPEN' in self.env.action.name:
                open_trend_factor = self._get_open_trend_factor(h1_1h, l1_1h, h2_1h, l2_1h, self.env.action)
                entry_proximity_factor = self._get_entry_proximity_factor(
                    current_price, h1_5m, l1_5m, h1_15m, l1_15m, h1_1h, l1_1h, atr_threshold, self.env.action
                )
                reward += self.open_coeff * open_trend_factor * entry_proximity_factor * position_factor

            # # 2. 持仓奖励
            # if net_position != Decimal('0.0'):
            #     # 趋势跟随奖励
            #     trend_factor = self.w1 * trend_align_1h + self.w2 * trend_align_15m
            #     trend_reward = self.trend_coeff * trend_factor

            #     # 突破奖励
            #     breakout_factor = self._get_breakout_factor(current_price, h1_1h, l1_1h, h1_15m, l1_15m, net_position)
            #     breakout_reward = self.breakout_coeff * breakout_factor

            #     reward += trend_reward + breakout_reward

            # 3. 平仓奖励
            if self.env.last_close_position is not None:
                close_factor = self._get_close_factor(
                    current_price, h1_1h, l1_1h, h1_15m, l1_15m, atr_threshold, self.env.last_close_position, self.env.action
                )
                reward += self.close_coeff * close_factor

            self.last_action = self.env.action
            return float(reward)

        except Exception as e:
            print(f"Error in TrendReward: {e}")
            return 0.0

    def _get_trend_alignment(self, h1, l1, h2, l2, net_position):
        if h1 > h2 and l1 > l2:  # 上涨
            return Decimal('1.0') if net_position > 0 else (Decimal('0') if net_position < 0 else Decimal('0.0'))
        elif h1 < h2 and l1 < l2:  # 下跌
            return Decimal('1.0') if net_position < 0 else (Decimal('0') if net_position > 0 else Decimal('0.0'))
        return Decimal('0.0')  # 振荡或空仓

    def _get_open_trend_factor(self, h1, l1, h2, l2, action):
        is_long = 'LONG' in action.name
        if h1 > h2 and l1 > l2:  # 1h 上涨
            return Decimal('1.0') if is_long else Decimal('0.0')
        elif h1 < h2 and l1 < l2:  # 1h 下跌
            return Decimal('1.0') if not is_long else Decimal('0.0')
        return Decimal('0.5')  # 1h 振荡

    def _get_entry_proximity_factor(self, price, h1_5m, l1_5m, h1_15m, l1_15m, h1_1h, l1_1h, atr_threshold, action):
        """检查是否在 2 个 20 周期 ATR 内接近前低/前高"""
        is_long = 'LONG' in action.name
        levels = [(h1_5m, l1_5m), (h1_15m, l1_15m), (h1_1h, l1_1h)]
        for h, l in levels:
            if is_long and l > 0 and (l - atr_threshold) <= price <= (l + atr_threshold):
                return Decimal('1.0')
            elif not is_long and h > 0 and (h - atr_threshold) <= price <= (h + atr_threshold):
                return Decimal('1.0')
        return Decimal('0.0')

    def _get_breakout_factor(self, price, h1_1h, l1_1h, h1_15m, l1_15m, net_position):
        if net_position > 0:
            if price > h1_1h or price > h1_15m:
                return Decimal('1.0')
            elif price < l1_1h:
                return Decimal('-0.5')
        elif net_position < 0:
            if (price < l1_1h or price < l1_15m):
                return Decimal('1.0')
            elif price > h1_1h:
                return Decimal('-0.5')
        return Decimal('0.0')

    def _get_close_factor(self, price, h1_1h, l1_1h, h1_15m, l1_15m, atr_threshold, last_close, action):
        """计算平仓因子，基于 2 个 20 周期 ATR"""
        is_long = 'LONG' in action.name
        pnl = last_close['pnl']
        if pnl >= Decimal(0):
            # 止盈
            for h in [h1_1h, h1_15m]:
                if is_long and h > 0 and (h - atr_threshold) <= price <= (h + atr_threshold):
                    return Decimal('1.0')
            for l in [l1_1h, l1_15m]:
                if not is_long and l > 0 and (l - atr_threshold) <= price <= (l + atr_threshold):
                    return Decimal('1.0')
        else:
            # 止损
            if is_long and (price < l1_1h or price < l1_15m):
                return Decimal('1.5')
            elif not is_long and (price > h1_1h or price > h1_15m):
                return Decimal('1.5')
        return Decimal('0.0')

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

    def __call__(self, obs=None):
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

    def __call__(self, obs=None):
        reward = 0.0
        metrics = self.env.metrics.get_metrics()
        daily_lost_pct = decimal_to_float(metrics['current_day_lost_pct'] / Decimal('100.0'))
        drawdown_pct = decimal_to_float(metrics['current_drawdown_pct'] / Decimal('100.0')) 
        if daily_lost_pct > self.env.config.risk.daily_lost_ratio:
            reward = self.limit
        if drawdown_pct > self.env.config.risk.max_drawdown_ratio:
            reward = self.limit
        return float(reward)
    


class NoviceModeReward:
    def __init__(self, env):
        self.env = env
        self.lower_limit = -0.5
        self.upper_limit = 0.5
        self.step_count = 0
        self.rewards = [
            NoviceModeActionReward(env),
            NoviceModeProfitLossReward(env),
            NoviceModeTradeCompletionReward(env),
            NoviceModePositionHoldingReward(env),
            NoviceModeInactionPenalty(env),
        ]
        self.episode_trades = 0

    def __call__(self, obs=None):
        self.step_count += 1
        reward = sum([fn(obs) for fn in self.rewards])
        if 'OPEN' in self.env.action.name:
            self.episode_trades += 1
        if self.env.terminated:
            if self.episode_trades > 0:
                reward += 0.2
            self.episode_trades = 0
        self.env.last_close_position = None
        return max(self.lower_limit, min(self.upper_limit, reward))

class NoviceModeActionReward:
    def __init__(self, env):
        self.env = env
        self.was_empty = True

    def __call__(self, obs=None):
        if 'OPEN' not in self.env.action.name:
            return 0.0
        long_pos = self.env.position_manager.total_long_position()
        short_pos = self.env.position_manager.total_short_position()
        if long_pos > 0 or short_pos > 0:
            if self.was_empty:
                self.was_empty = False
                return 0.2
            return 0.1
        return 0.0

class NoviceModeProfitLossReward:
    def __init__(self, env, min_reward=-0.5, max_reward=0.75):
        self.env = env
        self.min_reward = min_reward
        self.max_reward = max_reward

    def __call__(self, obs=None):
        pt = float(self._get_unrealized_pnl(obs))
        if pt > 0:
            return min(pt * 0.2, 0.2)
        else:
            return max(pt * 0.05, -0.05)


    def _get_unrealized_pnl(self, obs):
        reward = Decimal('0')
        # 持仓收益奖励
        def pos_reward(pos, long: bool):
            if pos is None:
                return Decimal('0')

            pnl = calc_unrealized_pnl(self.env.current_price, pos, self.env.config.trading.lot_size, long)

            sign = copysign(1, float(pnl))
            reward = sign * (abs(float(pnl) / float(pos.initial_margin)))  # (abs(pnl / margin))
            
            return float_to_decimal(reward)

        for pos in self.env.position_manager.long_positions:
            reward += pos_reward(pos, True)
        for pos in self.env.position_manager.short_positions:
            reward += pos_reward(pos, False)
        return reward

class NoviceModeTradeCompletionReward:
    def __init__(self, env):
        self.env = env
        self.position_steps = 0

    def __call__(self, obs=None):
        long_pos = self.env.position_manager.total_long_position()
        short_pos = self.env.position_manager.total_short_position()
        if long_pos > 0 or short_pos > 0:
            self.position_steps += 1
        if self.env.last_close_position is not None:
            if self.position_steps >= 5:
                self.position_steps = 0
                return 0.1
            self.position_steps = 0
        return 0.0

class NoviceModePositionHoldingReward:
    def __init__(self, env):
        self.env = env
        self.episode_hold_reward = 0.0

    def __call__(self, obs=None):
        long_pos = self.env.position_manager.total_long_position()
        short_pos = self.env.position_manager.total_short_position()
        if long_pos > 0 or short_pos > 0:
            reward = 0.01
            self.episode_hold_reward += reward
            if self.episode_hold_reward > 0.2:
                reward = 0.0
                self.episode_hold_reward = 0.2
            return reward
        return 0.0

class NoviceModeInactionPenalty:
    def __init__(self, env):
        self.env = env
        self.empty_steps = 0
        self.episode_penalty = 0.0

    def __call__(self, obs=None):
        long_pos = self.env.position_manager.total_long_position()
        short_pos = self.env.position_manager.total_short_position()
        if long_pos == 0 and short_pos == 0:
            self.empty_steps += 1
            if self.empty_steps > 200:
                penalty = -0.01
                self.episode_penalty += -penalty
                if self.episode_penalty > 0.2:
                    penalty = 0.0
                    self.episode_penalty = 0.2
                return penalty
        else:
            self.empty_steps = 0
        return 0.0
            


reward_classes = {
    'current_balance_reward_function': CurrentBalanceReward,
    'total_pnl_reward_function': TotalPnlReward,
    'fast_car_racing_likely_reward_function': NoviceModeReward,
}
