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
    - 主项：Δequity（含未实现打折）/ R_cash 归一化
    - 惩罚：回撤增量、临近EOD持仓、闭市乱操作、止损触发、无效操作
    - 注意：fee 已体现在 ledger 的 user_cash 变化里，因此不再显式 r_fee，避免 double-penalty
    """

    def __init__(
        self,
        env,
        w_fee=0.35,
        w_dd=0.0,
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

        # 未实现收益折扣系数（0~1），默认 0.5
        self.alpha_unrealized = float(getattr(self.env.config.trading, "alpha_unrealized", 0.5))
        self.alpha_unrealized = min(max(self.alpha_unrealized, 0.0), 1.0)

        # 方案B：重复无效动作的基础惩罚强度
        self.w_invalid_action = float(getattr(self.env.config.trading, "invalid_action_punish", 0.02))

        # 方案A：每次无效动作的“时间/操作成本”，按 episode 步数均摊
        invalid_time_cost_total = float(getattr(self.env.config.trading, "invalid_time_cost_total", 0.2))
        self.invalid_time_cost_total = max(0.0, invalid_time_cost_total)

        # streak 上限
        self.invalid_streak_cap = int(getattr(self.env.config.trading, "invalid_streak_cap", 10))
        if self.invalid_streak_cap <= 0:
            self.invalid_streak_cap = 10

        # 内部状态
        self.prev_eq = None              # 上一步 shaped equity
        self.peak_eq = None              # shaped equity peak
        self.prev_dd_cash = None
        self.prev_stoploss_fired = 0
        self.invalid_streak = 0

        self._invalid_time_cost_per_step = self._compute_invalid_time_cost_per_step()

        # --- ATR-based close shaping ---
        self.atr_takeprofit_ratio = float(getattr(self.env.config.trading, "atr_takeprofit_ratio", 0.70))
        self.w_atr_close = float(getattr(self.env.config.trading, "w_atr_close", 0.10))
        if self.atr_takeprofit_ratio < 0:
            self.atr_takeprofit_ratio = 0.0

    def reset(self):
        self.prev_eq = None
        self.peak_eq = None
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

    def _R_cash(self) -> Decimal:
        rc = getattr(self.env, "_R_cash_last", None)
        if rc is None:
            return Decimal("1")
        if isinstance(rc, Decimal):
            return max(rc, self.eps)
        return max(Decimal(str(rc)), self.eps)

    def _action_code(self) -> int | None:
        ar = getattr(self.env, "action_result", None)
        if ar is None:
            return None
        try:
            return int(getattr(ar, "value", ar))
        except Exception:
            return None

    def _tanh_scaled(self, x: Decimal, scale: Decimal) -> float:
        try:
            z = x / max(scale, self.eps)
            v = decimal_to_float(z)
        except Exception:
            v = 1e6 if x > 0 else -1e6

        if v > 20:
            v = 20.0
        elif v < -20:
            v = -20.0
        return tanh(float(v))

    # ========= 核心：shaped equity =========
    def _shaped_equity(self) -> Decimal:
        """
        shaped_equity = user_cash + user_margin + alpha * unrealized_pnl

        关键修复：
        - 与 Ledger 口径一致：fee 在开/平仓时直接扣到 user_cash
        - 原先用 realized_pnl（统计投影，不含 fee）会造成 reward 漏 fee + 再显式罚 fee 的 double-penalty
        """
        b = self.env.ledger.balances()
        cash = b.get("user_cash", D0)
        margin = b.get("user_margin", D0)

        upnl = getattr(self.env.user_accounts, "unrealized_pnl", D0)
        if not isinstance(upnl, Decimal):
            upnl = Decimal(str(upnl))

        a = Decimal(str(self.alpha_unrealized))
        return cash + margin + (upnl * a)

    def _invalid_weight(self, code: int) -> float:
        if code == int(ForexCode.ERROR_NO_ENOUGH_MONEY.value):
            return 0.5
        return 1.0

    def _is_invalid_action(self) -> bool:
        return bool(getattr(self.env, "_last_action_rejected", False))

    def __call__(self, obs=None):
        eq = self._shaped_equity()

        if self.prev_eq is None:
            self.prev_eq = eq
            self.peak_eq = eq
            self.prev_dd_cash = D0
            self.prev_stoploss_fired = int(getattr(self.env, "stop_loss_fired", 0))
            self.invalid_streak = 0

            self._reward_debug = {
                "pnl": 0.0, "fee": 0.0, "dd": 0.0, "eod": 0.0,
                "close": 0.0, "sl": 0.0, "mkt_closed": 0.0,
                "invalid_time": 0.0, "invalid_streak": 0.0, "invalid_total": 0.0,
                "total": 0.0,
                "alpha_unrealized": float(self.alpha_unrealized),
                "r_atr_close": 0.0,
            }
            self.env._reward_debug = self._reward_debug
            return 0.0

        scale = self._R_cash()

        # --- 主项：Δ shaped_equity / R_cash ---
        dE = eq - self.prev_eq
        r_pnl = self._tanh_scaled(dE, scale)

        # --- fee：已在 dE 中体现（cash 变化），不再显式 r_fee（避免 double-penalty） ---
        fee = getattr(self.env, "fee_step", D0)
        if not isinstance(fee, Decimal):
            fee = Decimal(str(fee))
        r_fee = 0.0

        # --- dd：基于 shaped equity 的 peak，罚 dd 上升增量（扣除当步 fee 影响，避免再罚一次） ---
        if self.peak_eq is None:
            self.peak_eq = eq
        if eq > self.peak_eq:
            self.peak_eq = eq

        dd_cash = self.peak_eq - eq
        if dd_cash < D0:
            dd_cash = D0

        prev_dd = self.prev_dd_cash if self.prev_dd_cash is not None else dd_cash
        dd_inc = dd_cash - prev_dd
        if dd_inc < D0:
            dd_inc = D0

        dd_inc_eff = dd_inc
        if fee > D0:
            dd_inc_eff = dd_inc - fee
            if dd_inc_eff < D0:
                dd_inc_eff = D0

        r_dd = -self.w_dd * self._tanh_scaled(dd_inc_eff, scale) if dd_inc_eff > D0 else 0.0

        # --- EOD：临近收盘仍持仓就罚 ---
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

        # --- 平仓事件小奖励 ---
        r_close = 0.0
        r_atr_close = 0.0

        lcp = getattr(self.env, "last_close_position", None)
        if lcp is not None and "pnl" in lcp:
            pnl = lcp["pnl"]
            try:
                pnl = pnl if isinstance(pnl, Decimal) else Decimal(str(pnl))
                r_close = self.w_close * self._tanh_scaled(pnl, scale)
            except Exception:
                r_close = 0.0

            # --- NEW: 0.7 * ATR take-profit shaping (no lookahead) ---
            try:
                store = getattr(self.env, "store", None)
                if store is None:
                    bar_source = getattr(self.env, "bar_source", None)
                    store = getattr(bar_source, "store", None) if bar_source is not None else None
                day_i = int(getattr(self.env, "_day_i", getattr(self.env, "day_i", -1)))
                atr_arr = getattr(store, "daily_atr_price", None) if store is not None else None

                if atr_arr is not None and 0 <= day_i < len(atr_arr):
                    atr_price = float(atr_arr[day_i])
                    if atr_price > 0:
                        lot = float(getattr(self.env.config.trading, "lot_size", 1.0))
                        atr_cash = Decimal(str(atr_price * lot))
                        thr = atr_cash * Decimal(str(self.atr_takeprofit_ratio))
                        if pnl >= thr:
                            r_atr_close = self.w_atr_close * self._tanh_scaled(pnl, scale)
            except Exception:
                r_atr_close = 0.0

            self.env.last_close_position = None

        # --- 止损触发惩罚 ---
        sl_now = int(getattr(self.env, "stop_loss_fired", 0))
        dsl = max(0, sl_now - int(self.prev_stoploss_fired))
        r_sl = -self.w_stoploss * float(dsl) if dsl > 0 else 0.0

        # --- 闭市乱操作惩罚 ---
        code = self._action_code()
        r_mc = 0.0
        if code == int(ForexCode.ERROR_MARKET_CLOSED.value):
            r_mc = -self.w_market_closed

        # --- 无效动作惩罚：time cost（A） + streak（B） ---
        r_invalid_time = 0.0
        r_invalid_streak = 0.0

        invalid = self._is_invalid_action()
        if invalid:
            self.invalid_streak += 1
            s = min(self.invalid_streak, self.invalid_streak_cap)

            r_invalid_time = -float(self._invalid_time_cost_per_step)

            mode = getattr(self.env.config.trading, "invalid_streak_mode", "linear")
            k = max(0, s - 1)
            mult = k if mode == "linear" else (k * k)

            w = self._invalid_weight(int(code) if code is not None else 1)
            r_invalid_streak = -self.w_invalid_action * float(mult) * float(w)
        else:
            self.invalid_streak = 0

        r_invalid = r_invalid_time + r_invalid_streak

        raw_total = r_pnl + r_fee + r_dd + r_eod + r_close + r_atr_close + r_sl + r_mc + r_invalid
        total = float(np.tanh(raw_total))   # 或 total = self.clip * np.tanh(raw_total / self.clip)


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
            "alpha_unrealized": float(self.alpha_unrealized),
            "r_atr_close": float(r_atr_close),
            "raw_total": float(raw_total),
            "total": float(total),
        }

        self.prev_eq = eq
        self.prev_dd_cash = dd_cash
        self.prev_stoploss_fired = sl_now

        return float(total)


reward_classes = {
    'current_balance_reward_function': CurrentBalanceReward,
    'total_pnl_reward_function': EquityDeltaReward,
    'fast_car_racing_likely_reward_function': NoviceModeReward,
    'futures_intraday_reward_function': FuturesIntradayReward,
}
