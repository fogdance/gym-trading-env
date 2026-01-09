# src/gym_trading_env/rewards/reward_functions.py

from decimal import Decimal
from gym_trading_env.utils.decimal_util import decimal_to_float, float_to_decimal
from math import log1p, copysign
from collections import deque
from gym_trading_env.envs.action import ForexCode
import numpy as np
from gym_trading_env.utils.decimal_util import D, D0, D1, D100, quantize_money

from gym_trading_env.utils.trade_util import calc_unrealized_pnl

from math import tanh
from gym_trading_env.utils.decimal_util import D, D0, decimal_to_float
from decimal import Decimal


class EquityDeltaReward:
    """
    复式记账版：
    equity = user_cash + user_margin + unrealized_pnl(投影)
    reward:
      - mode="delta": equity(t) - equity(t-1)
      - mode="pct":   (equity(t)-equity(t-1)) / max(|equity(t-1)|, eps)
    """
    def __init__(self, env, precision=6, mode="pct", include_margin=True, include_unrealized=True, eps=Decimal("1e-8")):
        self.env = env
        self.precision = precision
        self.mode = mode
        self.include_margin = include_margin
        self.include_unrealized = include_unrealized
        self.eps = eps
        self.previous_equity = None

    def _equity(self) -> Decimal:
        b = self.env.ledger.balances()  # Dict[str, Decimal]

        cash = b.get("user_cash", Decimal("0"))
        margin = b.get("user_margin", Decimal("0")) if self.include_margin else Decimal("0")
        unreal = getattr(self.env.user_accounts, "unrealized_pnl", Decimal("0")) if self.include_unrealized else Decimal("0")

        return cash + margin + unreal

    def reset(self):
        self.previous_equity = self._equity()

    def __call__(self, obs=None):
        equity = self._equity()

        if self.previous_equity is None:
            self.previous_equity = equity
            return 0.0

        delta = equity - self.previous_equity

        if self.mode == "pct":
            denom = max(abs(self.previous_equity), self.eps)
            reward = delta / denom
        else:
            reward = delta

        self.previous_equity = equity
        return float(decimal_to_float(reward, precision=self.precision))


class CurrentBalanceReward:
    """复式记账版：返回当前现金(user_cash)"""
    def __init__(self, env, precision=2):
        self.env = env
        self.precision = precision

    def __call__(self, obs=None):
        cash = self.env.ledger.balances().get("user_cash", Decimal("0"))
        return float(decimal_to_float(cash, precision=self.precision))


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
            CloseReward(env, **self.config['close']),
            EventReward(env, repeated=self.config['event']['repeated'], max_profit_reward=self.config['event']['max_profit_reward'], max_loss_penalty=self.config['event']['max_loss_penalty']),
            TerminationReward(env, self.config['termination']['limit']),
        ]

    def __call__(self, obs=None):
        reward = sum([fn(obs) for fn in self.rewards])
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
            
class FuturesIntradayReward:
    """
    期货日内奖励（推荐 baseline）：
    - 主项：Δequity（含未实现）/ R_cash 归一化
    - 惩罚：手续费、回撤增量、临近EOD持仓、闭市乱操作、止损触发
    - NEW：无效操作小惩罚（如 flat 还 close / 有仓还 open / near_eod 禁开 等）
    期货日内奖励（baseline）+ 解决乱点：
    ✅ 方案B（主）：只罚“重复无效操作”（invalid streak 递增）
    ✅ 方案A（兜底）：每次无效操作一个极小时间/操作成本（按 episode 步数归一化）
    """
    def __init__(
        self,
        env,
        w_fee=0.35,
        w_dd=0.30,
        w_eod=0.25,
        eod_grace_minutes=10,
        w_close=0.10,
        w_stoploss=0.20,
        w_market_closed=0.05,
        clip=1.0,
        eps=Decimal("1e-6"),
    ):
        self.env = env
        self.w_fee = float(w_fee)
        self.w_dd = float(w_dd)
        self.w_eod = float(w_eod)
        self.eod_grace = int(eod_grace_minutes)
        self.w_close = float(w_close)
        self.w_stoploss = float(w_stoploss)
        self.w_market_closed = float(w_market_closed)
        self.clip = float(clip)
        self.eps = eps

        # 方案B：重复无效动作的基础惩罚强度
        self.w_invalid_action = float(getattr(self.env.config.trading, "invalid_action_punish", 0.02))

        # 方案A：每次无效动作的“时间/操作成本”，按 episode 步数均摊（不改经济账）
        invalid_time_cost_total = float(getattr(self.env.config.trading, "invalid_time_cost_total", 0.2))
        self.invalid_time_cost_total = max(0.0, invalid_time_cost_total)

        # streak 上限
        self.invalid_streak_cap = int(getattr(self.env.config.trading, "invalid_streak_cap", 10))
        if self.invalid_streak_cap <= 0:
            self.invalid_streak_cap = 10

        # 内部状态
        self.prev_equity = None
        self.prev_dd_cash = None
        self.prev_stoploss_fired = 0
        self.invalid_streak = 0  # 连续无效次数

        # 每步 time cost：初始化时先算一次；如果拿不到 episode 步数，就用 DAY_LEN/345 兜底
        self._invalid_time_cost_per_step = self._compute_invalid_time_cost_per_step()

    def reset(self):
        self.prev_equity = None
        self.prev_dd_cash = None
        self.prev_stoploss_fired = 0
        self.invalid_streak = 0
        self._invalid_time_cost_per_step = self._compute_invalid_time_cost_per_step()

    def _compute_invalid_time_cost_per_step(self) -> float:
        """
        把 invalid_time_cost_total 均摊到 episode 步数上，避免“不同数据要调参”。
        优先用 training.max_episode_steps / episode_length，其次 DAY_LEN。
        """
        steps = None
        cfg = getattr(self.env, "config", None)
        tr = getattr(cfg, "training", None) if cfg is not None else None

        if tr is not None:
            ms = int(getattr(tr, "max_episode_steps", 0) or 0)
            if ms > 0:
                steps = ms
            else:
                ep_len = getattr(tr, "episode_length", None)
                if ep_len is not None:
                    try:
                        steps = int(ep_len)
                    except Exception:
                        steps = None

        if steps is None:
            steps = int(getattr(self.env, "DAY_LEN", 345) or 345)
        steps = max(1, steps)

        return float(self.invalid_time_cost_total) / float(steps)

    def _equity(self) -> Decimal:
        # 含未实现：用 user_accounts.equity()（你现在就是这样算风控/metrics 的）
        return self.env.user_accounts.equity()

    def _R_cash(self) -> Decimal:
        rc = getattr(self.env, "_R_cash_last", None)
        if rc is None:
            # fallback：极端情况下给个不为0的尺度
            return Decimal("1")
        if isinstance(rc, Decimal):
            return max(rc, self.eps)
        # rc 可能是 float/np scalar
        return max(Decimal(str(rc)), self.eps)

    def _action_code(self) -> int | None:
        ar = getattr(self.env, "action_result", None)
        if ar is None:
            return None
        try:
            return int(getattr(ar, "value", ar))
        except Exception:
            return None

    def _invalid_weight(self, code: int) -> float:
        """
        可选：不同错误给不同权重（不改经济账，只是“控制成本” shaping）。
        这里默认：保证金不足稍轻（0.5），其他 1.0。
        """
        if code == int(ForexCode.ERROR_NO_ENOUGH_MONEY.value):
            return 0.5
        return 1.0

    def _is_invalid_action(self) -> bool:
        code = self._action_code()
        if code is None:
            return False

        if code == int(ForexCode.SUCCESS.value):
            return False

        # market closed 单独罚（r_mc），避免 double-penalty，也不计入 streak
        if code == int(ForexCode.ERROR_MARKET_CLOSED.value):
            return False

        # ✅ 其余全部 ERROR_* 都算 invalid
        return True

    def __call__(self, obs=None):
        eq = self._equity()

        # drawdown cash：用 env.max_equity（你在 agent features 里维护了 max_equity）
        peak = getattr(self.env, "max_equity", eq)
        dd_cash = peak - eq
        if dd_cash < D0:
            dd_cash = D0

        if self.prev_equity is None:
            self.prev_equity = eq
            self.prev_dd_cash = dd_cash
            self.prev_stoploss_fired = int(getattr(self.env, "stop_loss_fired", 0))
            self.invalid_streak = 0
            self._reward_debug = {
                "pnl": 0.0, "fee": 0.0, "dd": 0.0, "eod": 0.0,
                "close": 0.0, "sl": 0.0, "mkt_closed": 0.0,
                "invalid_time": 0.0,
                "invalid_streak": 0.0,
                "invalid_total": 0.0,
                "total": 0.0
            }
            self.env._reward_debug = self._reward_debug
            return 0.0

        scale = self._R_cash()

        # --- 主项：Δequity / R_cash ---
        dE = eq - self.prev_equity
        r_pnl = tanh(float(decimal_to_float(dE / scale)))

        # --- fee 惩罚（fee_step >= 0）---
        fee = getattr(self.env, "fee_step", D0)
        r_fee = -self.w_fee * tanh(float(decimal_to_float(fee / scale))) if fee > D0 else 0.0

        # --- 回撤“变差增量”惩罚（只罚 dd 上升的那部分）---
        prev_dd = self.prev_dd_cash if self.prev_dd_cash is not None else dd_cash
        dd_inc = dd_cash - prev_dd
        if dd_inc < D0:
            dd_inc = D0
        r_dd = -self.w_dd * tanh(float(decimal_to_float(dd_inc / scale))) if dd_inc > D0 else 0.0

        # --- EOD 引导：临近收盘仍持仓就罚（线性加大）---
        # 用 env 缓存 minutes_to_eod（推荐），没有就现算
        m2eod = getattr(self.env, "_minutes_to_eod_last", None)
        if m2eod is None:
            eod_idx = int(getattr(self.env, "_eod_idx", getattr(self.env, "end_idx", self.env.current_step)))
            m2eod = max(0, eod_idx - int(self.env.current_step))
        m2eod = int(m2eod)

        in_market = False
        try:
            in_market = (self.env.user_accounts.long_position > D0) or (self.env.user_accounts.short_position > D0)
        except Exception:
            in_market = False

        r_eod = 0.0
        if self.eod_grace > 0 and in_market and m2eod < self.eod_grace:
            frac = float(self.eod_grace - m2eod) / float(self.eod_grace)
            r_eod = -self.w_eod * frac

        # --- 平仓事件小奖励（避免太大）---
        r_close = 0.0
        lcp = getattr(self.env, "last_close_position", None)
        if lcp is not None and "pnl" in lcp:
            pnl = lcp["pnl"]
            try:
                pnl = pnl if isinstance(pnl, Decimal) else Decimal(str(pnl))
                r_close = self.w_close * tanh(float(decimal_to_float(pnl / scale)))
            except Exception:
                r_close = 0.0
            # 防止重复给奖
            self.env.last_close_position = None

        # --- 止损触发惩罚（按“触发次数”计）---
        sl_now = int(getattr(self.env, "stop_loss_fired", 0))
        dsl = max(0, sl_now - int(self.prev_stoploss_fired))
        r_sl = -self.w_stoploss * float(dsl) if dsl > 0 else 0.0

        # --- 闭市乱操作惩罚 ---
        r_mc = 0.0
        if getattr(self.env, "action_result", None) == ForexCode.ERROR_MARKET_CLOSED:
            r_mc = -self.w_market_closed

        # =========================================================
        # ✅ 方案A + 方案B：无效动作惩罚（不改手续费/经济账）
        #   A：每次无效动作极小 time cost（兜底）
        #   B：只罚“重复无效”（streak>1 才开始加重）
        # =========================================================
        r_invalid_time = 0.0
        r_invalid_streak = 0.0

        code = self._action_code()
        invalid = self._is_invalid_action()

        if invalid:
            self.invalid_streak += 1
            s = min(self.invalid_streak, self.invalid_streak_cap)

            # 方案A：每次无效都扣一点点（很轻）
            r_invalid_time = -float(self._invalid_time_cost_per_step)

            # 方案B：只罚重复（第1次不罚/极轻），第2次开始线性加重
            # 你要更狠也可以改成 (s-1)^2
            w = self._invalid_weight(int(code) if code is not None else 1)
            r_invalid_streak = -self.w_invalid_action * float(max(0, s - 1)) * float(w)
        else:
            self.invalid_streak = 0

        r_invalid = r_invalid_time + r_invalid_streak

        total = r_pnl + r_fee + r_dd + r_eod + r_close + r_sl + r_mc + r_invalid
        if total > self.clip:
            total = self.clip
        elif total < -self.clip:
            total = -self.clip

        self.env._reward_debug = {
            "pnl": float(r_pnl),
            "fee": float(r_fee),
            "dd": float(r_dd),
            "eod": float(r_eod),
            "close": float(r_close),
            "sl": float(r_sl),
            "mkt_closed": float(r_mc),
            "invalid_time": float(r_invalid_time),
            "invalid_streak": float(r_invalid_streak),
            "invalid_total": float(r_invalid),
            "invalid_streak_len": int(self.invalid_streak),
            "total": float(total),
        }

        # 更新 prev
        self.prev_equity = eq
        self.prev_dd_cash = dd_cash
        self.prev_stoploss_fired = sl_now

        return float(total)

reward_classes = {
    'current_balance_reward_function': CurrentBalanceReward,
    'total_pnl_reward_function': EquityDeltaReward,
    'fast_car_racing_likely_reward_function': NoviceModeReward,
    'futures_intraday_reward_function': FuturesIntradayReward,
}
