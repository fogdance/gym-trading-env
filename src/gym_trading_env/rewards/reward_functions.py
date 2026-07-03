# src/gym_trading_env/rewards/reward_functions.py

from decimal import Decimal
from gym_trading_env.utils.decimal_util import decimal_to_float, float_to_decimal
from math import log1p, copysign
from collections import deque
from gym_trading_env.envs.action import ForexCode
import numpy as np
from gym_trading_env.utils.decimal_util import D, D0, D1, D100, quantize_money

from gym_trading_env.utils.trade_util import calc_unrealized_pnl
from gym_trading_env.rewards.reward_audit import RewardAuditMixin

from math import tanh
from gym_trading_env.utils.decimal_util import D, D0, decimal_to_float
from decimal import Decimal


class EquityDeltaReward(RewardAuditMixin):
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
        self._set_reward_debug(self.default_reward_debug())

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
            debug = self.default_reward_debug()
            debug.update({
                "mtm_equity": decimal_to_float(equity),
                "prev_mtm_equity": decimal_to_float(equity),
                "scale_cash": decimal_to_float(max(abs(equity), self.eps)),
            })
            self._set_reward_debug(debug)
            return 0.0

        delta = equity - self.previous_equity

        if self.mode == "pct":
            denom = max(abs(self.previous_equity), self.eps)
            reward = delta / denom
        else:
            reward = delta

        reward_f = float(decimal_to_float(reward, precision=self.precision))
        debug = self.default_reward_debug()
        debug.update({
            "pnl": reward_f,
            "mtm_equity": decimal_to_float(equity),
            "prev_mtm_equity": decimal_to_float(self.previous_equity),
            "delta_equity": decimal_to_float(delta),
            "scale_cash": decimal_to_float(max(abs(self.previous_equity), self.eps)),
            "raw_total": reward_f,
            "total": reward_f,
        })
        self._set_reward_debug(debug)
        self.previous_equity = equity
        return reward_f


class CurrentBalanceReward(RewardAuditMixin):
    """复式记账版：返回当前现金(user_cash)"""
    def __init__(self, env, precision=2):
        self.env = env
        self.precision = precision
        self._set_reward_debug(self.default_reward_debug())

    def __call__(self, obs=None):
        cash = self.env.ledger.balances().get("user_cash", Decimal("0"))
        reward = float(decimal_to_float(cash, precision=self.precision))
        debug = self.default_reward_debug()
        debug.update({
            "pnl": reward,
            "mtm_equity": reward,
            "prev_mtm_equity": reward,
            "raw_total": reward,
            "total": reward,
        })
        self._set_reward_debug(debug)
        return reward


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

class FastCarRacingReward(RewardAuditMixin):
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
        self._set_reward_debug(self.default_reward_debug())

    def __call__(self, obs=None):
        reward = sum([fn(obs) for fn in self.rewards])
        self.env.last_close_position = None
        reward = max(self.lower_limit, min(self.upper_limit, reward))
        debug = self.default_reward_debug()
        debug.update({"raw_total": reward, "total": reward})
        self._set_reward_debug(debug)
        return reward



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
    


class NoviceModeReward(RewardAuditMixin):
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
        self._set_reward_debug(self.default_reward_debug())

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
        reward = max(self.lower_limit, min(self.upper_limit, reward))
        debug = self.default_reward_debug()
        debug.update({"raw_total": reward, "total": reward})
        self._set_reward_debug(debug)
        return reward

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
            
class FuturesIntradayReward(RewardAuditMixin):
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
        debug = self.default_reward_debug()
        debug["alpha_unrealized"] = float(self.alpha_unrealized)
        self._set_reward_debug(debug)

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
                "invalid_streak_len": 0,
                "invalid_action_debug": 0.0,
                "total": 0.0,
                "alpha_unrealized": float(self.alpha_unrealized),
                "r_atr_close": 0.0,
                "mtm_equity": decimal_to_float(eq),
                "prev_mtm_equity": decimal_to_float(eq),
                "delta_equity": 0.0,
                "scale_cash": decimal_to_float(self._R_cash()),
                "fee_cash_debug": 0.0,
                "raw_total": 0.0,
            }
            self._set_reward_debug(self._reward_debug)
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
            "invalid_action_debug": float(1.0 if invalid else 0.0),
            "alpha_unrealized": float(self.alpha_unrealized),
            "r_atr_close": float(r_atr_close),
            "mtm_equity": float(decimal_to_float(eq)),
            "prev_mtm_equity": float(decimal_to_float(self.prev_eq)),
            "delta_equity": float(decimal_to_float(dE)),
            "scale_cash": float(decimal_to_float(scale)),
            "fee_cash_debug": float(decimal_to_float(fee)),
            "raw_total": float(raw_total),
            "total": float(total),
        }
        self._set_reward_debug(self.env._reward_debug)

        self.prev_eq = eq
        self.prev_dd_cash = dd_cash
        self.prev_stoploss_fired = sl_now

        return float(total)


PROFIT_LOCK_DEBUG_KEYS = [
    "profit_lock_enabled",
    "profit_lock_active",
    "profit_lock_trigger_R",
    "profit_lock_mfe_cash",
    "profit_lock_mfe_R",
    "profit_lock_mae_cash",
    "profit_lock_unrealized_cash",
    "profit_lock_giveback_cash",
    "profit_lock_allowed_giveback_cash",
    "profit_lock_excess_giveback_cash",
    "profit_lock_excess_giveback_inc_cash",
    "profit_lock_capture_ratio",
    "profit_lock_lost_mfe_fired",
    "profit_lock_bars",
    "profit_lock_scale_cash",
    "profit_lock_scale_source_code",
    "profit_lock_atr_price",
    "profit_lock_atr_ratio",
    "profit_lock_tick_cash",
    "r_profit_giveback",
    "r_lost_mfe",
    "r_lock_close",
    "r_profit_lock_total",
    "r_old_raw_total",
    "r_raw_total_with_profit_lock",
]


class FuturesIntradayRewardProfitLock(FuturesIntradayReward):
    """
    Old intraday reward plus a small profit-lock overlay.

    The base FuturesIntradayReward path remains responsible for entry/close/EOD
    shaping. This class snapshots close events before the base reward consumes
    them, then applies a bounded overlay only after a trade has reached a
    configurable MFE threshold.
    """

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        trading = getattr(getattr(self.env, "config", None), "trading", None)
        self.reward_profit_lock_enabled = self._cfg_bool(
            getattr(trading, "reward_profit_lock_enabled", True)
        )
        self.reward_profit_lock_trigger_R = max(
            0.0,
            float(getattr(trading, "reward_profit_lock_trigger_R", 1.0)),
        )
        self.reward_profit_lock_scale_mode = str(
            getattr(trading, "reward_profit_lock_scale_mode", "daily_atr_ratio")
        )
        self.reward_profit_lock_atr_ratio = max(
            0.0,
            float(getattr(trading, "reward_profit_lock_atr_ratio", 0.55)),
        )
        self.reward_profit_lock_allowed_giveback_ratio = max(
            0.0,
            float(getattr(trading, "reward_profit_lock_allowed_giveback_ratio", 0.50)),
        )
        self.reward_profit_lock_min_capture_ratio = max(
            0.0,
            float(getattr(trading, "reward_profit_lock_min_capture_ratio", 0.50)),
        )
        self.reward_profit_lock_w_giveback = max(
            0.0,
            float(getattr(trading, "reward_profit_lock_w_giveback", 0.05)),
        )
        self.reward_profit_lock_w_lost_mfe = max(
            0.0,
            float(getattr(trading, "reward_profit_lock_w_lost_mfe", 0.10)),
        )
        self.reward_profit_lock_w_lock_close = max(
            0.0,
            float(getattr(trading, "reward_profit_lock_w_lock_close", 0.05)),
        )
        self.reward_profit_lock_max_overlay_abs = max(
            0.0,
            float(getattr(trading, "reward_profit_lock_max_overlay_abs", 0.20)),
        )
        self.reward_profit_lock_incremental_giveback = self._cfg_bool(
            getattr(trading, "reward_profit_lock_incremental_giveback", True)
        )
        self.reward_profit_lock_lost_mfe_one_shot = self._cfg_bool(
            getattr(trading, "reward_profit_lock_lost_mfe_one_shot", True)
        )

        self._reset_profit_lock_tracking()
        self._set_reward_debug(
            self._with_profit_lock_debug(
                self.reward_audit_debug(),
                self._profit_lock_zero_debug(),
            )
        )

    @staticmethod
    def _cfg_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off"}
        return bool(value)

    def reset(self):
        super().reset()
        self._reset_profit_lock_tracking()
        debug = self.default_reward_debug()
        debug["alpha_unrealized"] = float(self.alpha_unrealized)
        self._set_reward_debug(
            self._with_profit_lock_debug(debug, self._profit_lock_zero_debug())
        )

    def on_episode_start(self, context):
        self.reset()

    def _reset_profit_lock_tracking(self):
        self.trade_active = False
        self.trade_side = 0
        self.trade_mfe_cash = D0
        self.trade_mae_cash = D0
        self.trade_prev_excess_giveback_cash = D0
        self.trade_profit_lock_active = False
        self.trade_lost_mfe_fired = False
        self.trade_bars = 0

    def _profit_lock_zero_debug(self) -> dict[str, float]:
        debug = {key: 0.0 for key in PROFIT_LOCK_DEBUG_KEYS}
        if getattr(self, "reward_profit_lock_enabled", False):
            debug["profit_lock_enabled"] = 1.0
            debug["profit_lock_trigger_R"] = float(self.reward_profit_lock_trigger_R)
        return debug

    def _with_profit_lock_debug(
        self,
        base_debug,
        profit_lock_debug: dict[str, float],
    ) -> dict[str, float]:
        debug = dict(base_debug)
        debug.update(self._profit_lock_zero_debug())
        debug.update(profit_lock_debug)
        return debug

    def _to_decimal(self, value) -> Decimal:
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except Exception:
            return D0

    def _lot_cash_multiplier(self) -> Decimal:
        trading = getattr(getattr(self.env, "config", None), "trading", None)
        lot_size = self._to_decimal(getattr(trading, "lot_size", D1))
        trade_lot = self._to_decimal(getattr(trading, "trade_lot", D1))
        return max(lot_size, D0) * max(trade_lot, D0)

    def _profit_lock_tick_cash(self) -> Decimal:
        trading = getattr(getattr(self.env, "config", None), "trading", None)
        tick = self._to_decimal(getattr(trading, "tick_size", None))
        if tick <= D0:
            tick = self._to_decimal(getattr(trading, "spread", D0))
        if tick <= D0:
            return D0
        return tick * self._lot_cash_multiplier()

    def _daily_atr_price(self) -> Decimal:
        store = getattr(self.env, "store", None)
        if store is None:
            bar_source = getattr(self.env, "bar_source", None)
            store = getattr(bar_source, "store", None) if bar_source is not None else None
        day_i = int(getattr(self.env, "_day_i", getattr(self.env, "day_i", -1)))
        atr_arr = getattr(store, "daily_atr_price", None) if store is not None else None
        if atr_arr is None or day_i < 0:
            return D0
        try:
            if day_i >= len(atr_arr):
                return D0
            atr_price = self._to_decimal(atr_arr[day_i])
        except Exception:
            return D0
        return max(atr_price, D0)

    def _stop_loss_scale_cash(self) -> Decimal:
        trading = getattr(getattr(self.env, "config", None), "trading", None)
        mode = str(getattr(trading, "stop_loss_mode", "pct"))
        stop_value = self._to_decimal(getattr(trading, "stop_loss_value", D0))
        if stop_value <= D0:
            return D0
        if mode == "pct":
            ref_price = self._to_decimal(
                getattr(self.env, "current_price", getattr(self.env, "_last_valid_price", D0))
            )
            stop_distance = ref_price * stop_value
        else:
            stop_distance = stop_value
        if stop_distance <= D0:
            return D0
        return stop_distance * self._lot_cash_multiplier()

    def _profit_lock_scale(self) -> tuple[Decimal, dict[str, Decimal]]:
        atr_price = self._daily_atr_price()
        atr_ratio = Decimal(str(self.reward_profit_lock_atr_ratio))
        lot_cash = self._lot_cash_multiplier()
        tick_cash = self._profit_lock_tick_cash()

        scale = D0
        source_code = Decimal("0")
        mode = self.reward_profit_lock_scale_mode.strip().lower()
        if mode in {"daily_atr_ratio", "atr", "atr_ratio"}:
            scale = atr_price * lot_cash * atr_ratio
            if scale > self.eps:
                source_code = Decimal("1")

        if scale <= self.eps:
            stop_scale = self._stop_loss_scale_cash()
            if stop_scale > self.eps:
                scale = stop_scale
                source_code = Decimal("2")

        if scale <= self.eps:
            scale = max(self._R_cash(), self.eps)
            source_code = Decimal("3")

        return scale, {
            "source_code": source_code,
            "atr_price": atr_price,
            "atr_ratio": atr_ratio,
            "tick_cash": tick_cash,
        }

    def _position_side(self) -> int:
        accounts = getattr(self.env, "user_accounts", None)
        long_pos = self._to_decimal(getattr(accounts, "long_position", D0))
        short_pos = self._to_decimal(getattr(accounts, "short_position", D0))
        if long_pos > D0:
            return 1
        if short_pos > D0:
            return -1
        return 0

    def _unrealized_cash(self) -> Decimal:
        accounts = getattr(self.env, "user_accounts", None)
        return self._to_decimal(getattr(accounts, "unrealized_pnl", D0))

    def _snapshot_close_event(self):
        lcp = getattr(self.env, "last_close_position", None)
        if isinstance(lcp, dict):
            return dict(lcp)
        return None

    def _start_profit_lock_trade(self, side: int):
        self.trade_active = True
        self.trade_side = int(side)
        self.trade_mfe_cash = D0
        self.trade_mae_cash = D0
        self.trade_prev_excess_giveback_cash = D0
        self.trade_profit_lock_active = False
        self.trade_lost_mfe_fired = False
        self.trade_bars = 0

    def _update_trade_extremes(self, unrealized_cash: Decimal):
        if unrealized_cash > self.trade_mfe_cash:
            self.trade_mfe_cash = unrealized_cash
        if unrealized_cash < self.trade_mae_cash:
            self.trade_mae_cash = unrealized_cash
        self.trade_bars += 1

    def _mfe_active(self, mfe_cash: Decimal, scale: Decimal) -> bool:
        trigger_cash = Decimal(str(self.reward_profit_lock_trigger_R)) * max(scale, self.eps)
        return bool(mfe_cash >= trigger_cash)

    def _profit_lock_state_debug(
        self,
        *,
        scale: Decimal,
        unrealized_cash: Decimal,
        giveback_cash: Decimal = D0,
        allowed_giveback_cash: Decimal = D0,
        excess_giveback_cash: Decimal = D0,
        excess_giveback_inc_cash: Decimal = D0,
        capture_ratio: float = 0.0,
        r_profit_giveback: float = 0.0,
        r_lost_mfe: float = 0.0,
        r_lock_close: float = 0.0,
        r_profit_lock_total: float = 0.0,
        r_old_raw_total: float = 0.0,
        r_raw_total_with_profit_lock: float = 0.0,
        scale_meta: dict[str, Decimal] | None = None,
        active_override: bool | None = None,
        mfe_cash_override: Decimal | None = None,
        mae_cash_override: Decimal | None = None,
        bars_override: int | None = None,
        lost_mfe_fired_override: bool | None = None,
    ) -> dict[str, float]:
        mfe_cash = self.trade_mfe_cash if mfe_cash_override is None else mfe_cash_override
        mae_cash = self.trade_mae_cash if mae_cash_override is None else mae_cash_override
        bars = self.trade_bars if bars_override is None else bars_override
        active = self.trade_profit_lock_active if active_override is None else active_override
        lost_fired = (
            self.trade_lost_mfe_fired
            if lost_mfe_fired_override is None
            else lost_mfe_fired_override
        )
        mfe_R = Decimal("0")
        try:
            mfe_R = mfe_cash / max(scale, self.eps)
        except Exception:
            mfe_R = Decimal("0")
        scale_meta = scale_meta or {}
        source_code = scale_meta.get("source_code", D0)
        atr_price = scale_meta.get("atr_price", D0)
        atr_ratio = scale_meta.get(
            "atr_ratio",
            Decimal(str(self.reward_profit_lock_atr_ratio)),
        )
        tick_cash = scale_meta.get("tick_cash", D0)

        return {
            "profit_lock_enabled": 1.0 if self.reward_profit_lock_enabled else 0.0,
            "profit_lock_active": 1.0 if active else 0.0,
            "profit_lock_trigger_R": float(self.reward_profit_lock_trigger_R),
            "profit_lock_mfe_cash": float(decimal_to_float(mfe_cash)),
            "profit_lock_mfe_R": float(mfe_R),
            "profit_lock_mae_cash": float(decimal_to_float(mae_cash)),
            "profit_lock_unrealized_cash": float(decimal_to_float(unrealized_cash)),
            "profit_lock_giveback_cash": float(decimal_to_float(giveback_cash)),
            "profit_lock_allowed_giveback_cash": float(decimal_to_float(allowed_giveback_cash)),
            "profit_lock_excess_giveback_cash": float(decimal_to_float(excess_giveback_cash)),
            "profit_lock_excess_giveback_inc_cash": float(decimal_to_float(excess_giveback_inc_cash)),
            "profit_lock_capture_ratio": float(capture_ratio),
            "profit_lock_lost_mfe_fired": 1.0 if lost_fired else 0.0,
            "profit_lock_bars": float(int(bars)),
            "profit_lock_scale_cash": float(decimal_to_float(scale)),
            "profit_lock_scale_source_code": float(decimal_to_float(source_code)),
            "profit_lock_atr_price": float(decimal_to_float(atr_price)),
            "profit_lock_atr_ratio": float(atr_ratio),
            "profit_lock_tick_cash": float(decimal_to_float(tick_cash)),
            "r_profit_giveback": float(r_profit_giveback),
            "r_lost_mfe": float(r_lost_mfe),
            "r_lock_close": float(r_lock_close),
            "r_profit_lock_total": float(r_profit_lock_total),
            "r_old_raw_total": float(r_old_raw_total),
            "r_raw_total_with_profit_lock": float(r_raw_total_with_profit_lock),
        }

    def _compute_profit_lock_overlay(self, close_event) -> tuple[float, dict[str, float]]:
        scale, scale_meta = self._profit_lock_scale()
        current_side = self._position_side()
        current_unrealized = self._unrealized_cash()

        closed_mfe_cash = self.trade_mfe_cash
        closed_mae_cash = self.trade_mae_cash
        closed_bars = self.trade_bars
        closed_lost_mfe_fired = self.trade_lost_mfe_fired
        close_lost_mfe_fired = closed_lost_mfe_fired
        close_active = self._mfe_active(closed_mfe_cash, scale) if self.trade_active else False

        r_profit_giveback = 0.0
        r_lost_mfe = 0.0
        r_lock_close = 0.0
        capture_ratio = 0.0
        if close_event is not None and self.trade_active:
            realized_pnl = self._to_decimal(close_event.get("pnl", D0))
            denom = max(closed_mfe_cash, self.eps)
            capture_ratio = float(realized_pnl / denom) if denom > D0 else 0.0
            if (
                close_active
                and realized_pnl > D0
                and capture_ratio >= self.reward_profit_lock_min_capture_ratio
            ):
                r_lock_close = (
                    self.reward_profit_lock_w_lock_close
                    * self._tanh_scaled(realized_pnl, scale)
                )
            elif close_active and realized_pnl <= D0:
                if self.reward_profit_lock_lost_mfe_one_shot:
                    if not closed_lost_mfe_fired:
                        r_lost_mfe = -self.reward_profit_lock_w_lost_mfe
                        close_lost_mfe_fired = True
                else:
                    r_lost_mfe = (
                        -self.reward_profit_lock_w_lost_mfe
                        * self._tanh_scaled(abs(realized_pnl), scale)
                    )
                    close_lost_mfe_fired = True
            self._reset_profit_lock_tracking()

        if current_side == 0:
            if close_event is None and self.trade_active:
                self._reset_profit_lock_tracking()
        else:
            if (not self.trade_active) or self.trade_side != current_side:
                self._start_profit_lock_trade(current_side)
            self._update_trade_extremes(current_unrealized)
            self.trade_profit_lock_active = self._mfe_active(self.trade_mfe_cash, scale)

        giveback_cash = D0
        allowed_giveback_cash = D0
        excess_giveback_cash = D0
        excess_giveback_inc_cash = D0

        if self.trade_active and self.trade_profit_lock_active:
            current_positive_pnl = max(current_unrealized, D0)
            giveback_cash = max(self.trade_mfe_cash - current_positive_pnl, D0)
            allowed_giveback_cash = (
                Decimal(str(self.reward_profit_lock_allowed_giveback_ratio))
                * self.trade_mfe_cash
            )
            excess_giveback_cash = max(giveback_cash - allowed_giveback_cash, D0)
            if self.reward_profit_lock_incremental_giveback:
                excess_giveback_inc_cash = max(
                    excess_giveback_cash - self.trade_prev_excess_giveback_cash,
                    D0,
                )
            else:
                excess_giveback_inc_cash = excess_giveback_cash
            if excess_giveback_inc_cash > D0:
                r_profit_giveback = (
                    -self.reward_profit_lock_w_giveback
                    * self._tanh_scaled(excess_giveback_inc_cash, scale)
                )
            self.trade_prev_excess_giveback_cash = excess_giveback_cash

            if current_unrealized <= D0:
                if self.reward_profit_lock_lost_mfe_one_shot:
                    if not self.trade_lost_mfe_fired:
                        r_lost_mfe = -self.reward_profit_lock_w_lost_mfe
                        self.trade_lost_mfe_fired = True
                else:
                    r_lost_mfe = (
                        -self.reward_profit_lock_w_lost_mfe
                        * self._tanh_scaled(abs(current_unrealized), scale)
                    )

        overlay = r_profit_giveback + r_lost_mfe + r_lock_close
        overlay = float(np.clip(
            overlay,
            -self.reward_profit_lock_max_overlay_abs,
            self.reward_profit_lock_max_overlay_abs,
        ))

        if close_event is not None:
            debug = self._profit_lock_state_debug(
                scale=scale,
                unrealized_cash=current_unrealized,
                capture_ratio=capture_ratio,
                r_lock_close=r_lock_close,
                r_lost_mfe=r_lost_mfe,
                r_profit_lock_total=overlay,
                scale_meta=scale_meta,
                active_override=close_active,
                mfe_cash_override=closed_mfe_cash,
                mae_cash_override=closed_mae_cash,
                bars_override=closed_bars,
                lost_mfe_fired_override=close_lost_mfe_fired,
            )
        else:
            debug = self._profit_lock_state_debug(
                scale=scale,
                unrealized_cash=current_unrealized,
                giveback_cash=giveback_cash,
                allowed_giveback_cash=allowed_giveback_cash,
                excess_giveback_cash=excess_giveback_cash,
                excess_giveback_inc_cash=excess_giveback_inc_cash,
                r_profit_giveback=r_profit_giveback,
                r_lost_mfe=r_lost_mfe,
                r_profit_lock_total=overlay,
                scale_meta=scale_meta,
            )
        return overlay, debug

    def __call__(self, obs=None):
        close_event = self._snapshot_close_event()
        old_total = super().__call__(obs)
        old_debug = dict(self.reward_audit_debug())

        if not self.reward_profit_lock_enabled:
            self._set_reward_debug(
                self._with_profit_lock_debug(old_debug, self._profit_lock_zero_debug())
            )
            return float(old_total)

        overlay, profit_lock_debug = self._compute_profit_lock_overlay(close_event)

        old_raw_total = float(old_debug.get("raw_total", 0.0))
        raw_total = old_raw_total + float(overlay)
        total = float(np.tanh(raw_total))

        profit_lock_debug["r_profit_lock_total"] = float(overlay)
        profit_lock_debug["r_old_raw_total"] = float(old_raw_total)
        profit_lock_debug["r_raw_total_with_profit_lock"] = float(raw_total)

        debug = dict(old_debug)
        debug["raw_total"] = float(raw_total)
        debug["total"] = float(total)
        self._set_reward_debug(self._with_profit_lock_debug(debug, profit_lock_debug))
        return total


class FuturesIntradayMTMCleanReward(RewardAuditMixin):
    """
    v1-clean: Pure mark-to-market equity delta reward.

    Objective:
      reward is driven only by the change in real MTM equity.

    MTM equity:
      equity = user_cash + user_margin + unrealized_pnl

    Reward:
      raw_reward = (equity_t - equity_{t-1}) / R_cash
      reward = clip * tanh(raw_reward / clip)

    Intentional removals compared with FuturesIntradayReward:
      - no close-position bonus
      - no ATR take-profit shaping
      - no explicit drawdown penalty
      - no explicit EOD penalty
      - no explicit stop-loss penalty
      - no explicit invalid-action penalty
      - no explicit market-closed penalty
      - no explicit fee penalty

    Fees are already reflected by ledger cash changes.
    Spread/slippage is reflected through execution price and unrealized/realized PnL.

    Audit contract:
      - all REWARD_DEBUG_KEYS are present and numeric;
      - debug["total"] equals returned reward;
      - disabled reward components are present but exactly zero.
    """

    def __init__(
        self,
        env,
        clip=1.0,
        eps=Decimal("1e-6"),
        precision=8,
        fallback_scale_cash=None,
        fail_if_missing_scale=False,
    ):
        self.env = env
        self.clip = float(clip)
        self.eps = eps
        self.precision = int(precision)
        self.fail_if_missing_scale = bool(fail_if_missing_scale)

        if self.clip <= 0:
            self.clip = 1.0

        if fallback_scale_cash is None:
            fallback_scale_cash = getattr(
                getattr(self.env.config, "trading", object()),
                "reward_scale_cash_fallback",
                "1.0",
            )
        self.fallback_scale_cash = self._to_decimal(fallback_scale_cash, Decimal("1.0"))

        self.prev_equity = None
        self.prev_timestamp = None
        self.invalid_streak = 0

        # v1-clean hard contract: unrealized PnL is fully included.
        self.alpha_unrealized = 1.0

        # Initialize a valid all-numeric audit snapshot immediately.
        debug = self.default_reward_debug()
        debug.update({
            "alpha_unrealized": 1.0,
            "scale_cash": float(decimal_to_float(self._scale_cash(), precision=6)),
        })
        self._set_reward_debug(debug)

    def reward_audit_disabled_components(self) -> tuple[str, ...]:
        """
        These components are intentionally disabled in v1-clean.
        They must remain exactly zero, or validate_reward_audit() should fail.
        """
        return (
            "fee",
            "dd",
            "eod",
            "close",
            "sl",
            "mkt_closed",
            "invalid_time",
            "invalid_streak",
            "invalid_total",
            "r_atr_close",
        )

    def reset(self):
        self.prev_equity = None
        self.prev_timestamp = None
        self.invalid_streak = 0

        debug = self.default_reward_debug()
        debug.update({
            "alpha_unrealized": 1.0,
            "scale_cash": float(decimal_to_float(self._scale_cash(), precision=6)),
        })
        self._set_reward_debug(debug)

    def _to_decimal(self, value, default=D0) -> Decimal:
        if value is None:
            return default
        if isinstance(value, Decimal):
            return value
        try:
            if isinstance(value, float) and not np.isfinite(value):
                return default
        except Exception:
            pass
        try:
            return Decimal(str(value))
        except Exception:
            return default

    def _ledger_balance(self, key: str) -> Decimal:
        try:
            balances = self.env.ledger.balances()
            return self._to_decimal(balances.get(key, D0), D0)
        except Exception:
            return D0

    def _unrealized_pnl(self) -> Decimal:
        upnl = getattr(self.env.user_accounts, "unrealized_pnl", D0)
        return self._to_decimal(upnl, D0)

    def _mtm_equity(self) -> Decimal:
        """
        Full mark-to-market equity.

        cash + margin avoids creating a fake reward drop when cash is moved
        into margin on position open.
        """
        cash = self._ledger_balance("user_cash")
        margin = self._ledger_balance("user_margin")
        unrealized = self._unrealized_pnl()
        return cash + margin + unrealized

    def _scale_cash(self) -> Decimal:
        """
        Prefer env._R_cash_last because FuturesIntradayReward already uses it
        as the per-trade risk cash scale.
        """
        rc = getattr(self.env, "_R_cash_last", None)
        rc = self._to_decimal(rc, None)

        if rc is not None and rc > self.eps:
            return rc

        cfg_scale = getattr(
            getattr(self.env.config, "trading", object()),
            "reward_scale_cash",
            None,
        )
        cfg_scale = self._to_decimal(cfg_scale, None)
        if cfg_scale is not None and cfg_scale > self.eps:
            return cfg_scale

        if self.fallback_scale_cash is not None and self.fallback_scale_cash > self.eps:
            return self.fallback_scale_cash

        if self.fail_if_missing_scale:
            raise RuntimeError(
                "FuturesIntradayMTMCleanReward cannot resolve reward scale. "
                "Expected env._R_cash_last or config.trading.reward_scale_cash."
            )

        return Decimal("1.0")

    def _safe_ratio_float(self, x: Decimal, scale: Decimal) -> float:
        try:
            z = x / max(scale, self.eps)
            v = decimal_to_float(z, precision=self.precision)
        except Exception:
            v = 1e6 if x > D0 else -1e6

        if v > 20.0:
            return 20.0
        if v < -20.0:
            return -20.0
        return float(v)

    def _squash(self, raw: float) -> float:
        """
        Single bounded transform. No double tanh.
        """
        z = raw / self.clip
        if z > 20.0:
            z = 20.0
        elif z < -20.0:
            z = -20.0
        return float(self.clip * np.tanh(z))

    def _fee_cash_debug(self) -> Decimal:
        fee = getattr(self.env, "fee_step", D0)
        return self._to_decimal(fee, D0)

    def __call__(self, obs=None):
        equity = self._mtm_equity()
        scale = self._scale_cash()

        if self.prev_equity is None:
            self.prev_equity = equity

            debug = self.default_reward_debug()
            debug.update({
                "pnl": 0.0,
                "fee": 0.0,
                "dd": 0.0,
                "eod": 0.0,
                "close": 0.0,
                "sl": 0.0,
                "mkt_closed": 0.0,
                "invalid_time": 0.0,
                "invalid_streak": 0.0,
                "invalid_total": 0.0,
                "invalid_streak_len": 0.0,
                "invalid_action_debug": 0.0,
                "alpha_unrealized": 1.0,
                "r_atr_close": 0.0,
                "mtm_equity": float(decimal_to_float(equity, precision=6)),
                "prev_mtm_equity": float(decimal_to_float(equity, precision=6)),
                "delta_equity": 0.0,
                "scale_cash": float(decimal_to_float(scale, precision=6)),
                "fee_cash_debug": 0.0,
                "raw_total": 0.0,
                "total": 0.0,
            })
            self._set_reward_debug(debug)
            return 0.0

        delta_equity = equity - self.prev_equity

        raw = self._safe_ratio_float(delta_equity, scale)
        total = self._squash(raw)

        invalid = bool(getattr(self.env, "_last_action_rejected", False))
        if invalid:
            self.invalid_streak += 1
        else:
            self.invalid_streak = 0

        fee = self._fee_cash_debug()

        debug = self.default_reward_debug()
        debug.update({
            # Main component. This is the only non-zero reward component.
            # For audit, pnl equals returned reward, not pre-squash raw.
            "pnl": float(total),

            # Intentionally disabled reward components.
            "fee": 0.0,
            "dd": 0.0,
            "eod": 0.0,
            "close": 0.0,
            "sl": 0.0,
            "mkt_closed": 0.0,
            "invalid_time": 0.0,
            "invalid_streak": 0.0,
            "invalid_total": 0.0,
            "r_atr_close": 0.0,

            # Debug-only numeric fields.
            "invalid_streak_len": float(int(self.invalid_streak)),
            "invalid_action_debug": float(1.0 if invalid else 0.0),
            "alpha_unrealized": 1.0,
            "mtm_equity": float(decimal_to_float(equity, precision=6)),
            "prev_mtm_equity": float(decimal_to_float(self.prev_equity, precision=6)),
            "delta_equity": float(decimal_to_float(delta_equity, precision=6)),
            "scale_cash": float(decimal_to_float(scale, precision=6)),
            "fee_cash_debug": float(decimal_to_float(fee, precision=6)),
            "raw_total": float(raw),
            "total": float(total),
        })

        self._set_reward_debug(debug)

        self.prev_equity = equity

        return float(total)

class FuturesIntradayMTMRiskReward(RewardAuditMixin):
    """
    v2-risk: Pure MTM equity delta reward + auditable risk penalties.

    Objective:
      reward is primarily driven by real MTM equity delta, with small
      explicit penalties for drawdown expansion and holding losing positions.

    MTM equity:
      equity = user_cash + user_margin + unrealized_pnl

    Main reward:
      r_pnl = (equity_t - equity_{t-1}) / R_cash

    Risk penalties:
      r_dd:
        account-level drawdown increment penalty.

      r_adverse:
        position-level adverse unrealized PnL increment penalty.

      r_loss_time:
        small per-step penalty while an open position is underwater.

    Intentional removals compared with FuturesIntradayReward:
      - no close-position bonus
      - no ATR take-profit shaping
      - no EOD shaping
      - no stop-loss shaping
      - no invalid-action shaping
      - no market-closed shaping
      - no explicit fee penalty

    Fees are already reflected by ledger cash changes.
    Spread/slippage is reflected through execution price and unrealized/realized PnL.

    Audit contract:
      - all REWARD_DEBUG_KEYS are present and numeric;
      - debug["total"] equals returned reward;
      - disabled reward components are present but exactly zero;
      - risk penalty is reported through debug["dd"] and extra numeric debug keys.
    """

    def __init__(
        self,
        env,
        clip=1.0,
        eps=Decimal("1e-6"),
        precision=8,
        fallback_scale_cash=None,
        fail_if_missing_scale=False,
        w_dd=None,
        w_adverse=None,
        w_loss_time=None,
    ):
        self.env = env
        self.clip = float(clip)
        self.eps = eps
        self.precision = int(precision)
        self.fail_if_missing_scale = bool(fail_if_missing_scale)

        if self.clip <= 0:
            self.clip = 1.0

        trading_cfg = getattr(self.env.config, "trading", object())

        if fallback_scale_cash is None:
            fallback_scale_cash = getattr(
                trading_cfg,
                "reward_scale_cash_fallback",
                "1.0",
            )
        self.fallback_scale_cash = self._to_decimal(fallback_scale_cash, Decimal("1.0"))

        # Keep defaults deliberately small. This reward should remain MTM-first.
        if w_dd is None:
            w_dd = getattr(trading_cfg, "reward_v2_w_dd", 0.05)
        if w_adverse is None:
            w_adverse = getattr(trading_cfg, "reward_v2_w_adverse", 0.05)
        if w_loss_time is None:
            w_loss_time = getattr(trading_cfg, "reward_v2_w_loss_time", 0.001)

        self.w_dd = max(0.0, float(w_dd))
        self.w_adverse = max(0.0, float(w_adverse))
        self.w_loss_time = max(0.0, float(w_loss_time))

        self.prev_equity = None
        self.peak_equity = None
        self.prev_dd_cash = D0
        self.prev_adverse_cash = D0
        self.loss_steps = 0
        self.invalid_streak = 0

        # v2-risk hard contract: unrealized PnL is fully included.
        self.alpha_unrealized = 1.0

        self._set_reward_debug(self._initial_debug())

    def reward_audit_disabled_components(self) -> tuple[str, ...]:
        """
        These components are intentionally disabled in v2-risk.
        The dd key is NOT disabled because it carries the risk penalty.
        """
        return (
            "fee",
            "eod",
            "close",
            "sl",
            "mkt_closed",
            "invalid_time",
            "invalid_streak",
            "invalid_total",
            "r_atr_close",
        )

    def reset(self):
        self.prev_equity = None
        self.peak_equity = None
        self.prev_dd_cash = D0
        self.prev_adverse_cash = D0
        self.loss_steps = 0
        self.invalid_streak = 0

        self._set_reward_debug(self._initial_debug())

    def _initial_debug(self) -> dict[str, float]:
        debug = self.default_reward_debug()
        debug.update({
            "alpha_unrealized": 1.0,
            "scale_cash": float(decimal_to_float(self._scale_cash(), precision=6)),
            "risk_dd": 0.0,
            "risk_adverse": 0.0,
            "risk_loss_time": 0.0,
            "drawdown_cash": 0.0,
            "drawdown_inc_cash": 0.0,
            "adverse_cash": 0.0,
            "adverse_inc_cash": 0.0,
            "loss_steps": 0.0,
            "w_dd": float(self.w_dd),
            "w_adverse": float(self.w_adverse),
            "w_loss_time": float(self.w_loss_time),
        })
        return debug

    def _to_decimal(self, value, default=D0) -> Decimal:
        if value is None:
            return default
        if isinstance(value, Decimal):
            return value
        try:
            if isinstance(value, float) and not np.isfinite(value):
                return default
        except Exception:
            pass
        try:
            return Decimal(str(value))
        except Exception:
            return default

    def _ledger_balance(self, key: str) -> Decimal:
        try:
            balances = self.env.ledger.balances()
            return self._to_decimal(balances.get(key, D0), D0)
        except Exception:
            return D0

    def _unrealized_pnl(self) -> Decimal:
        upnl = getattr(self.env.user_accounts, "unrealized_pnl", D0)
        return self._to_decimal(upnl, D0)

    def _mtm_equity(self) -> Decimal:
        """
        Full mark-to-market equity.

        cash + margin avoids a fake reward drop when cash is moved into margin.
        """
        cash = self._ledger_balance("user_cash")
        margin = self._ledger_balance("user_margin")
        unrealized = self._unrealized_pnl()
        return cash + margin + unrealized

    def _scale_cash(self) -> Decimal:
        rc = getattr(self.env, "_R_cash_last", None)
        rc = self._to_decimal(rc, None)

        if rc is not None and rc > self.eps:
            return rc

        cfg_scale = getattr(
            getattr(self.env.config, "trading", object()),
            "reward_scale_cash",
            None,
        )
        cfg_scale = self._to_decimal(cfg_scale, None)
        if cfg_scale is not None and cfg_scale > self.eps:
            return cfg_scale

        if self.fallback_scale_cash is not None and self.fallback_scale_cash > self.eps:
            return self.fallback_scale_cash

        if self.fail_if_missing_scale:
            raise RuntimeError(
                "FuturesIntradayMTMRiskReward cannot resolve reward scale. "
                "Expected env._R_cash_last or config.trading.reward_scale_cash."
            )

        return Decimal("1.0")

    def _safe_ratio_float(self, x: Decimal, scale: Decimal) -> float:
        try:
            z = x / max(scale, self.eps)
            v = decimal_to_float(z, precision=self.precision)
        except Exception:
            v = 1e6 if x > D0 else -1e6

        if v > 20.0:
            return 20.0
        if v < -20.0:
            return -20.0
        return float(v)

    def _tanh_scaled(self, x: Decimal, scale: Decimal) -> float:
        raw = self._safe_ratio_float(x, scale)
        return float(np.tanh(raw))

    def _squash(self, raw_total: float) -> float:
        z = raw_total / self.clip
        if z > 20.0:
            z = 20.0
        elif z < -20.0:
            z = -20.0
        return float(self.clip * np.tanh(z))

    def _fee_cash_debug(self) -> Decimal:
        fee = getattr(self.env, "fee_step", D0)
        return self._to_decimal(fee, D0)

    def _in_market(self) -> bool:
        try:
            return (
                self.env.user_accounts.long_position > D0
                or self.env.user_accounts.short_position > D0
            )
        except Exception:
            return False

    def __call__(self, obs=None):
        equity = self._mtm_equity()
        scale = self._scale_cash()
        unrealized = self._unrealized_pnl()
        in_market = self._in_market()

        if self.prev_equity is None:
            self.prev_equity = equity
            self.peak_equity = equity
            self.prev_dd_cash = D0
            self.prev_adverse_cash = max(D0, -unrealized)
            self.loss_steps = 0
            self.invalid_streak = 0

            debug = self.default_reward_debug()
            debug.update({
                "pnl": 0.0,
                "fee": 0.0,
                "dd": 0.0,
                "eod": 0.0,
                "close": 0.0,
                "sl": 0.0,
                "mkt_closed": 0.0,
                "invalid_time": 0.0,
                "invalid_streak": 0.0,
                "invalid_total": 0.0,
                "invalid_streak_len": 0.0,
                "invalid_action_debug": 0.0,
                "alpha_unrealized": 1.0,
                "r_atr_close": 0.0,
                "mtm_equity": float(decimal_to_float(equity, precision=6)),
                "prev_mtm_equity": float(decimal_to_float(equity, precision=6)),
                "delta_equity": 0.0,
                "scale_cash": float(decimal_to_float(scale, precision=6)),
                "fee_cash_debug": 0.0,
                "raw_total": 0.0,
                "total": 0.0,

                # Extra numeric audit fields.
                "risk_dd": 0.0,
                "risk_adverse": 0.0,
                "risk_loss_time": 0.0,
                "drawdown_cash": 0.0,
                "drawdown_inc_cash": 0.0,
                "adverse_cash": float(decimal_to_float(self.prev_adverse_cash, precision=6)),
                "adverse_inc_cash": 0.0,
                "loss_steps": 0.0,
                "w_dd": float(self.w_dd),
                "w_adverse": float(self.w_adverse),
                "w_loss_time": float(self.w_loss_time),
            })
            self._set_reward_debug(debug)
            return 0.0

        # Main MTM delta component.
        delta_equity = equity - self.prev_equity
        r_pnl = self._safe_ratio_float(delta_equity, scale)

        # Account-level drawdown increment.
        if self.peak_equity is None:
            self.peak_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

        dd_cash = self.peak_equity - equity
        if dd_cash < D0:
            dd_cash = D0

        dd_inc_cash = dd_cash - self.prev_dd_cash
        if dd_inc_cash < D0:
            dd_inc_cash = D0

        # Avoid double-counting fee as drawdown risk.
        fee = self._fee_cash_debug()
        dd_inc_eff = dd_inc_cash
        if fee > D0:
            dd_inc_eff = dd_inc_eff - fee
            if dd_inc_eff < D0:
                dd_inc_eff = D0

        r_dd = 0.0
        if dd_inc_eff > D0 and self.w_dd > 0:
            r_dd = -self.w_dd * self._tanh_scaled(dd_inc_eff, scale)

        # Position-level adverse excursion increment.
        adverse_cash = max(D0, -unrealized) if in_market else D0
        adverse_inc_cash = adverse_cash - self.prev_adverse_cash
        if adverse_inc_cash < D0:
            adverse_inc_cash = D0

        r_adverse = 0.0
        if adverse_inc_cash > D0 and self.w_adverse > 0:
            r_adverse = -self.w_adverse * self._tanh_scaled(adverse_inc_cash, scale)

        # Time penalty while holding a losing position.
        r_loss_time = 0.0
        if in_market and unrealized < D0:
            self.loss_steps += 1
            if self.w_loss_time > 0:
                r_loss_time = -self.w_loss_time * self._tanh_scaled(-unrealized, scale)
        else:
            self.loss_steps = 0

        # v2-risk maps all explicit risk penalties to the standard dd audit key.
        r_risk_total = float(r_dd + r_adverse + r_loss_time)

        invalid = bool(getattr(self.env, "_last_action_rejected", False))
        if invalid:
            self.invalid_streak += 1
        else:
            self.invalid_streak = 0

        raw_total = float(r_pnl + r_risk_total)
        total = self._squash(raw_total)

        debug = self.default_reward_debug()
        debug.update({
            "pnl": float(r_pnl),
            "fee": 0.0,
            "dd": float(r_risk_total),
            "eod": 0.0,
            "close": 0.0,
            "sl": 0.0,
            "mkt_closed": 0.0,
            "invalid_time": 0.0,
            "invalid_streak": 0.0,
            "invalid_total": 0.0,
            "invalid_streak_len": float(int(self.invalid_streak)),
            "invalid_action_debug": float(1.0 if invalid else 0.0),
            "alpha_unrealized": 1.0,
            "r_atr_close": 0.0,
            "mtm_equity": float(decimal_to_float(equity, precision=6)),
            "prev_mtm_equity": float(decimal_to_float(self.prev_equity, precision=6)),
            "delta_equity": float(decimal_to_float(delta_equity, precision=6)),
            "scale_cash": float(decimal_to_float(scale, precision=6)),
            "fee_cash_debug": float(decimal_to_float(fee, precision=6)),
            "raw_total": float(raw_total),
            "total": float(total),

            # Extra numeric audit fields.
            "risk_dd": float(r_dd),
            "risk_adverse": float(r_adverse),
            "risk_loss_time": float(r_loss_time),
            "drawdown_cash": float(decimal_to_float(dd_cash, precision=6)),
            "drawdown_inc_cash": float(decimal_to_float(dd_inc_cash, precision=6)),
            "adverse_cash": float(decimal_to_float(adverse_cash, precision=6)),
            "adverse_inc_cash": float(decimal_to_float(adverse_inc_cash, precision=6)),
            "loss_steps": float(int(self.loss_steps)),
            "w_dd": float(self.w_dd),
            "w_adverse": float(self.w_adverse),
            "w_loss_time": float(self.w_loss_time),
        })

        self._set_reward_debug(debug)

        self.prev_equity = equity
        self.prev_dd_cash = dd_cash
        self.prev_adverse_cash = adverse_cash

        return float(total)


class FuturesIntradayPnLAlignedReward(RewardAuditMixin):
    """
    Profit-aligned reward.

    With a fixed positive episode scale and only rounding drift:

        sum(reward_t)
        ~= (final_mtm_equity - initial_mtm_equity) / scale

    If the episode starts and ends flat:

        sum(reward_t) ~= actual_net_pnl / scale

    No tanh, clipping, close bonus, ATR bonus, or risk shaping.
    """

    def __init__(
        self,
        env,
        *,
        scale_cash=None,
        eps=Decimal("1e-6"),
        precision=10,
    ):
        self.env = env
        self.eps = eps
        self.precision = int(precision)

        configured_scale = scale_cash
        if configured_scale is None:
            configured_scale = getattr(
                getattr(env.config, "trading", object()),
                "reward_scale_cash",
                None,
            )

        if configured_scale is None:
            raise ValueError(
                "FuturesIntradayPnLAlignedReward requires a fixed "
                "config.trading.reward_scale_cash."
            )

        self.fixed_scale_cash = self._to_decimal(configured_scale)
        if self.fixed_scale_cash <= self.eps:
            raise ValueError(
                f"reward_scale_cash must be positive, got {self.fixed_scale_cash}"
            )

        self.previous_equity = None
        self.initial_equity = None
        self.cumulative_delta_equity = D0

        self._set_reward_debug(self._initial_debug())

    def on_episode_start(self, context) -> None:
        equity = self._to_decimal(context.get("mtm_equity"), None)
        if equity is None:
            equity = self._mtm_equity()
        self._prime(equity)

    def reward_audit_disabled_components(self) -> tuple[str, ...]:
        return (
            "fee",
            "dd",
            "eod",
            "close",
            "sl",
            "mkt_closed",
            "invalid_time",
            "invalid_streak",
            "invalid_total",
            "r_atr_close",
        )

    def _to_decimal(self, value, default=D0) -> Decimal:
        if value is None:
            return default
        if isinstance(value, Decimal):
            return value
        try:
            converted = Decimal(str(value))
        except Exception:
            return default
        return converted if converted.is_finite() else default

    def _ledger_balance(self, key: str) -> Decimal:
        balances = self.env.ledger.balances()
        return self._to_decimal(balances.get(key, D0))

    def _mtm_equity(self) -> Decimal:
        cash = self._ledger_balance("user_cash")
        margin = self._ledger_balance("user_margin")
        unrealized = self._to_decimal(
            getattr(self.env.user_accounts, "unrealized_pnl", D0)
        )
        return cash + margin + unrealized

    def _initial_debug(self) -> dict[str, float]:
        debug = self.default_reward_debug()
        debug.update({
            "alpha_unrealized": 1.0,
            "scale_cash": float(self.fixed_scale_cash),
        })
        return debug

    def _prime(self, equity: Decimal) -> None:
        self.previous_equity = equity
        self.initial_equity = equity
        self.cumulative_delta_equity = D0

        debug = self._initial_debug()
        debug.update({
            "mtm_equity": float(equity),
            "prev_mtm_equity": float(equity),
        })
        self._set_reward_debug(debug)

    def reset(self) -> None:
        """
        Backward-compatible manual reset. Env lifecycle calls
        on_episode_start(context) instead.
        """
        self._prime(self._mtm_equity())

    def __call__(self, obs=None) -> float:
        if self.previous_equity is None:
            raise RuntimeError(
                "PnL-aligned reward was not primed. Call "
                "reward.on_episode_start(context) after account reset and "
                "before the first env action."
            )

        equity = self._mtm_equity()
        delta_equity = equity - self.previous_equity
        self.cumulative_delta_equity += delta_equity

        reward_decimal = delta_equity / self.fixed_scale_cash
        reward = float(
            decimal_to_float(reward_decimal, precision=self.precision)
        )

        if not np.isfinite(reward):
            raise RuntimeError(
                f"Non-finite PnL-aligned reward: {reward!r}"
            )

        fee = self._to_decimal(getattr(self.env, "fee_step", D0))
        invalid = bool(getattr(self.env, "_last_action_rejected", False))

        debug = self.default_reward_debug()
        debug.update({
            # Only reward-producing component.
            "pnl": reward,
            "raw_total": reward,
            "total": reward,

            # Full MTM accounting.
            "alpha_unrealized": 1.0,
            "mtm_equity": float(equity),
            "prev_mtm_equity": float(self.previous_equity),
            "delta_equity": float(delta_equity),
            "scale_cash": float(self.fixed_scale_cash),

            # Diagnostic only; fee is already included in equity.
            "fee": 0.0,
            "fee_cash_debug": float(fee),
            "invalid_action_debug": float(invalid),

            # Explicitly disabled shaping.
            "dd": 0.0,
            "eod": 0.0,
            "close": 0.0,
            "sl": 0.0,
            "mkt_closed": 0.0,
            "invalid_time": 0.0,
            "invalid_streak": 0.0,
            "invalid_total": 0.0,
            "invalid_streak_len": 0.0,
            "r_atr_close": 0.0,
        })
        self._set_reward_debug(debug)

        self.previous_equity = equity
        return reward


reward_classes = {
    "futures_intraday_pnl_aligned_reward_function": FuturesIntradayPnLAlignedReward,
    'current_balance_reward_function': CurrentBalanceReward,
    'total_pnl_reward_function': EquityDeltaReward,
    'fast_car_racing_likely_reward_function': NoviceModeReward,
    'futures_intraday_reward_function': FuturesIntradayReward,
    'futures_intraday_reward_profit_lock_function': FuturesIntradayRewardProfitLock,
    'futures_intraday_mtm_clean_reward_function': FuturesIntradayMTMCleanReward,
    'futures_intraday_mtm_risk_reward_function': FuturesIntradayMTMRiskReward,
}
