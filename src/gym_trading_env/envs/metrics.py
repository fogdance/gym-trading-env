# src/gym_trading_env/envs/metrics.py

from decimal import Decimal
import math
import numpy as np
from gym_trading_env.utils.decimal_util import D, D0, D1, D100


_SECONDS_IN_YEAR = 365.0 * 24.0 * 3600.0


class Metrics:
    """
    Per-episode metrics (because env.reset() recreates Metrics in your env).
    Main goals:
      1) Evaluate training quality: return/drawdown/sharpe/sortino/calmar
      2) Diagnose trading behavior: exposure, action ratios, success ratios
      3) Efficient: incremental trade stats (no O(T^2) scans each step)
    """

    def __init__(self, user_accounts, trade_record_manager, risk_free_rate=Decimal("0.012")):
        self.user_accounts = user_accounts
        self.trade_record_manager = trade_record_manager
        self.risk_free_rate = risk_free_rate  # annual risk-free (e.g. 1.2%)

        self._reset_state()

    def _reset_state(self):
        # --- time / equity curve state ---
        self._start_ts = None
        self._last_ts = None
        self._equity0 = self.user_accounts.initial_balance
        self._last_equity = None  # Decimal
        self._time_years = 0.0

        # Running stats for (excess) step returns via Welford
        self._ret_n = 0
        self._ret_mean = 0.0
        self._ret_M2 = 0.0
        self._downside_n = 0
        self._downside_ss = 0.0  # sum of squares of negative excess returns

        # --- daily loss ---
        self.current_day_key = None
        self.previous_day_equity = self.user_accounts.initial_balance

        # --- drawdown ---
        self.peak_equity = self.user_accounts.initial_balance
        self._peak_at_max_dd = self.peak_equity  # IMPORTANT: peak when max_dd happened

        # --- incremental trade stats ---
        self._th_i = 0  # processed trade_history index
        self._sum_win = 0.0
        self._sum_loss = 0.0  # negative
        self._win_n = 0
        self._loss_n = 0
        self._sum_pnl = 0.0
        self._fee_total = 0.0
        self._open_n = 0
        self._close_n = 0

        # --- metrics dict (mixed Decimal + float) ---
        self.metrics = {
            # equity curve core
            "equity": self.user_accounts.initial_balance,  # Decimal
            "return_pct": None,  # float
            "cagr": None,  # float
            "sharpe_ratio": None,  # float (annualized)
            "sortino_ratio": None,  # float (annualized)
            "calmar_ratio": None,  # float

            # daily loss
            "current_day_lost": D0,
            "current_day_lost_pct": D0,

            # drawdown
            "current_drawdown": D0,
            "current_drawdown_pct": D0,
            "max_drawdown": D0,
            "max_drawdown_pct": D0,

            # trade stats (close trades)
            "total_trades": 0,
            "winning_trades": 0,
            "win_rate": None,
            "profit_factor": None,
            "avg_win": None,
            "avg_loss": None,
            "avg_pnl": None,
            "expectancy": None,  # avg pnl per trade (cash)
            "fee_total": D0,

            # extremes
            "max_profit": D0,
            "max_loss": D0,  # negative

            # behavior metrics filled in get_metrics()
            "action_hold_ratio": None,
            "action_long_open_ratio": None,
            "action_long_close_ratio": None,
            "action_short_open_ratio": None,
            "action_short_close_ratio": None,
            "action_long_open_success_ratio": None,
            "action_long_close_success_ratio": None,
            "action_short_open_success_ratio": None,
            "action_short_close_success_ratio": None,
            "exposure_ratio": None,
            "trades_opened": 0,
            "trades_closed": 0,
            # behavior / cost derived
            "opens_per_1000_steps": None,
            "fee_drag_ratio": None,

            # invalid action (per-step + cumulative)
            "invalid_action": 0,          # 0/1 for last step
            "invalid_action_total": 0,    # cumulative count
            "invalid_action_ratio": None, # optional
        }

        # behavior counters
        self._counters = {
            "steps_total": 0,
            "hold_steps": 0,
            "long_open_steps": 0,
            "long_close_steps": 0,
            "short_open_steps": 0,
            "short_close_steps": 0,
            "long_open_steps_success": 0,
            "long_close_steps_success": 0,
            "short_open_steps_success": 0,
            "short_close_steps_success": 0,
            "empty_steps": 0,
            "in_market_steps": 0,
            "invalid_steps": 0,
        }

    # -------- public API --------

    def update(self, current_timestamp, day_id=None):
        """
        Update all metrics using current equity + timestamp.
        day_id: optional; if provided, use it as the 'day boundary' key instead of timestamp.date()
               (recommended for futures night session).
        """
        current_equity = self.user_accounts.equity()
        self.metrics["equity"] = current_equity

        # --- init timeline ---
        if self._start_ts is None:
            self._start_ts = current_timestamp
            self._last_ts = current_timestamp
            self._last_equity = current_equity
            # initialize day key
            self.current_day_key = str(day_id) if day_id is not None else (
                current_timestamp.date() if hasattr(current_timestamp, "date") else None
            )
            return

        # --- compute dt in years ---
        dt_sec = 0.0
        try:
            dt = current_timestamp - self._last_ts
            dt_sec = float(dt.total_seconds())
        except Exception:
            dt_sec = 0.0
        if dt_sec < 0:
            dt_sec = 0.0
        dt_years = dt_sec / _SECONDS_IN_YEAR
        self._time_years += dt_years

        # --- equity step return (simple) ---
        if self._last_equity is not None and self._last_equity != D0:
            r = (current_equity / self._last_equity) - D1  # Decimal
            r_f = float(r)
        else:
            r_f = 0.0

        # --- risk-free for this dt (simple linear approx) ---
        rf_f = float(self.risk_free_rate) * dt_years
        excess = r_f - rf_f

        # Welford update for excess returns
        self._ret_n += 1
        n = float(self._ret_n)
        delta = excess - self._ret_mean
        self._ret_mean += delta / n
        self._ret_M2 += delta * (excess - self._ret_mean)

        if excess < 0:
            self._downside_n += 1
            self._downside_ss += excess * excess

        # update last
        self._last_ts = current_timestamp
        self._last_equity = current_equity

        # --- return_pct / cagr (equity-based) ---
        eq0 = self._equity0
        if eq0 != D0:
            self.metrics["return_pct"] = float((current_equity / eq0) - D1)
        else:
            self.metrics["return_pct"] = None


        years = self._time_years
        if years and years > 0 and eq0 != D0 and current_equity > D0:
            ratio = float(current_equity / eq0)

            # intraday: 时间太短 CAGR 没意义，避免爆炸
            MIN_YEARS_FOR_CAGR = 1.0 / 365.0  # 至少 1 天
            if ratio > 0 and years >= MIN_YEARS_FOR_CAGR:
                # 用 log/exp 形式更稳定：cagr = exp(log(ratio)/years) - 1
                g = math.log(ratio) / years  # annualized log growth
                # float64 exp 溢出阈值约 709
                if -700.0 < g < 700.0:
                    self.metrics["cagr"] = math.expm1(g)  # exp(g)-1, 数值稳定
                else:
                    self.metrics["cagr"] = None
            else:
                self.metrics["cagr"] = None
        else:
            self.metrics["cagr"] = None


        # --- daily loss: use day_id if given (recommended) ---
        day_key = str(day_id) if day_id is not None else (
            current_timestamp.date() if hasattr(current_timestamp, "date") else None
        )
        if self.current_day_key is None or (day_key is not None and day_key != self.current_day_key):
            # day boundary crossed
            self.previous_day_equity = current_equity
            self.metrics["current_day_lost"] = D0
            self.metrics["current_day_lost_pct"] = D0
            self.current_day_key = day_key

        day_loss = self.previous_day_equity - current_equity
        self.metrics["current_day_lost"] = max(day_loss, D0)
        if self.previous_day_equity != D0:
            self.metrics["current_day_lost_pct"] = (self.metrics["current_day_lost"] / self.previous_day_equity) * Decimal("100.0")
        else:
            self.metrics["current_day_lost_pct"] = D0

        # --- drawdown ---
        self.peak_equity = max(self.peak_equity, current_equity)
        cur_dd = self.peak_equity - current_equity
        self.metrics["current_drawdown"] = max(cur_dd, D0)
        if self.peak_equity != D0:
            self.metrics["current_drawdown_pct"] = (self.metrics["current_drawdown"] / self.peak_equity) * Decimal("100.0")
        else:
            self.metrics["current_drawdown_pct"] = D0

        # max drawdown (store peak at time of max_dd to avoid dilution)
        if self.metrics["current_drawdown"] > self.metrics["max_drawdown"]:
            self.metrics["max_drawdown"] = self.metrics["current_drawdown"]
            self._peak_at_max_dd = self.peak_equity

        if self._peak_at_max_dd != D0:
            self.metrics["max_drawdown_pct"] = (self.metrics["max_drawdown"] / self._peak_at_max_dd) * Decimal("100.0")
        else:
            self.metrics["max_drawdown_pct"] = D0

        # --- incremental trade stats ---
        self._consume_new_trades()

        # --- sharpe / sortino (annualized using realized timeline) ---
        self._compute_ratios()

        # --- calmar ---
        self._compute_calmar()

    def get_metrics(self):
        # behavior ratios
        beh = self._behavior_metrics()
        self.metrics.update(beh)
        return self.metrics

    # -------- behavior metrics --------

    def on_step(self, action, action_success: bool, in_market: bool, invalid_action: bool = False):
        c = self._counters
        c["steps_total"] += 1
        if in_market:
            c["in_market_steps"] += 1

        try:
            aname = action.name
        except Exception:
            aname = str(action)

        if "HOLD" in aname:
            c["hold_steps"] += 1
        elif "LONG_OPEN" in aname:
            c["long_open_steps"] += 1
            if action_success:
                c["long_open_steps_success"] += 1
        elif "LONG_CLOSE" in aname:
            c["long_close_steps"] += 1
            if action_success:
                c["long_close_steps_success"] += 1
        elif "SHORT_OPEN" in aname:
            c["short_open_steps"] += 1
            if action_success:
                c["short_open_steps_success"] += 1
        elif "SHORT_CLOSE" in aname:
            c["short_close_steps"] += 1
            if action_success:
                c["short_close_steps_success"] += 1
        elif "EMPTY" in aname:
            c["empty_steps"] += 1
        elif "FLIP_LONG_TO_SHORT" in aname:
            c["long_close_steps"] += 1
            c["short_open_steps"] += 1
            if action_success:
                c["long_close_steps_success"] += 1
                c["short_open_steps_success"] += 1
        elif "FLIP_SHORT_TO_LONG" in aname:
            c["short_close_steps"] += 1
            c["long_open_steps"] += 1
            if action_success:
                c["short_close_steps_success"] += 1
                c["long_open_steps_success"] += 1

        if invalid_action:
            c["invalid_steps"] += 1

        # publish last-step + cumulative
        self.metrics["invalid_action"] = 1 if invalid_action else 0
        self.metrics["invalid_action_total"] = int(c["invalid_steps"])

    def _behavior_metrics(self):
        c = self._counters
        if c["steps_total"] <= 0:
            return {k: None for k in [
                "action_hold_ratio", "action_long_open_ratio", "action_long_close_ratio",
                "action_short_open_ratio", "action_short_close_ratio",
                "action_long_open_success_ratio", "action_long_close_success_ratio",
                "action_short_open_success_ratio", "action_short_close_success_ratio",
                "exposure_ratio",
                "opens_per_1000_steps",
            ]} | {"trades_opened": 0, "trades_closed": 0}

        steps = float(c["steps_total"])
        out = {
            "action_hold_ratio": float(c["hold_steps"]) / steps,
            "action_long_open_ratio": float(c["long_open_steps"]) / steps,
            "action_long_close_ratio": float(c["long_close_steps"]) / steps,
            "action_short_open_ratio": float(c["short_open_steps"]) / steps,
            "action_short_close_ratio": float(c["short_close_steps"]) / steps,

            "action_long_open_success_ratio": (float(c["long_open_steps_success"]) / float(c["long_open_steps"])) if c["long_open_steps"] else 0.0,
            "action_long_close_success_ratio": (float(c["long_close_steps_success"]) / float(c["long_close_steps"])) if c["long_close_steps"] else 0.0,
            "action_short_open_success_ratio": (float(c["short_open_steps_success"]) / float(c["short_open_steps"])) if c["short_open_steps"] else 0.0,
            "action_short_close_success_ratio": (float(c["short_close_steps_success"]) / float(c["short_close_steps"])) if c["short_close_steps"] else 0.0,

            "exposure_ratio": float(c["in_market_steps"]) / steps,
            "invalid_action_ratio": float(c.get("invalid_steps", 0)) / steps,
        }

        # NEW: robust counts from incremental stats (no scan)
        out["trades_opened"] = int(self.metrics.get("trades_opened", 0))
        out["trades_closed"] = int(self.metrics.get("trades_closed", 0))

        # NEW: opens per 1000 steps (over this episode so far)
        out["opens_per_1000_steps"] = (float(out["trades_opened"]) / steps) * 1000.0

        return out


    # -------- trade stats (incremental) --------

    def _consume_new_trades(self):
        th = self.trade_record_manager.trade_history
        if self._th_i >= len(th):
            return

        new = th[self._th_i:]
        self._th_i = len(th)

        for t in new:
            op = getattr(t, "operation_type", "") or ""

            # NEW: incremental open/close counts
            if "OPEN" in op:
                self._open_n += 1
            if "CLOSE" in op:
                self._close_n += 1

            # fee (accumulate regardless of open/close)
            fee = getattr(t, "fee", None)
            if fee is not None:
                try:
                    self._fee_total += float(fee)
                except Exception:
                    pass

            # pnl stats only for CLOSE trades
            pnl = getattr(t, "pnl", None)
            if pnl is None:
                continue
            if "CLOSE" not in op:
                continue

            try:
                pnl_f = float(pnl)
            except Exception:
                continue

            self._sum_pnl += pnl_f
            if pnl_f >= 0:
                self._sum_win += pnl_f
                self._win_n += 1
            else:
                self._sum_loss += pnl_f  # negative
                self._loss_n += 1

            # max profit/loss (keep Decimal for consistency)
            try:
                self.metrics["max_profit"] = max(self.metrics["max_profit"], pnl)
                self.metrics["max_loss"] = min(self.metrics["max_loss"], pnl)
            except Exception:
                pass

        # publish opened/closed counts (NEW)
        self.metrics["trades_opened"] = int(self._open_n)
        self.metrics["trades_closed"] = int(self._close_n)

        # existing close-trade stats
        total = self._win_n + self._loss_n
        self.metrics["total_trades"] = int(total)
        self.metrics["winning_trades"] = int(self._win_n)
        self.metrics["fee_total"] = D(str(self._fee_total))

        # NEW: fee drag ratio = fee_total / gross_pnl_abs
        # gross_pnl_abs = sum_win + abs(sum_loss)
        gross_pnl_abs = self._sum_win + abs(self._sum_loss)
        if gross_pnl_abs > 0:
            self.metrics["fee_drag_ratio"] = float(self._fee_total) / float(gross_pnl_abs)
        else:
            self.metrics["fee_drag_ratio"] = None

        if total > 0:
            win_rate = float(self._win_n) / float(total)
            self.metrics["win_rate"] = win_rate

            avg_pnl = self._sum_pnl / float(total)
            self.metrics["avg_pnl"] = avg_pnl
            self.metrics["expectancy"] = avg_pnl

            self.metrics["avg_win"] = (self._sum_win / float(self._win_n)) if self._win_n else 0.0
            self.metrics["avg_loss"] = (abs(self._sum_loss) / float(self._loss_n)) if self._loss_n else 0.0

            if self._sum_loss < 0:
                self.metrics["profit_factor"] = (self._sum_win / abs(self._sum_loss)) if abs(self._sum_loss) > 0 else None
            else:
                self.metrics["profit_factor"] = None
        else:
            self.metrics["win_rate"] = None
            self.metrics["avg_pnl"] = None
            self.metrics["expectancy"] = None
            self.metrics["avg_win"] = None
            self.metrics["avg_loss"] = None
            self.metrics["profit_factor"] = None


    # -------- equity-curve ratios --------

    def _compute_ratios(self):
        # need at least 2 returns for std
        if self._ret_n < 2:
            self.metrics["sharpe_ratio"] = None
            self.metrics["sortino_ratio"] = None
            return

        var = self._ret_M2 / float(self._ret_n - 1)
        if var <= 0:
            self.metrics["sharpe_ratio"] = None
            self.metrics["sortino_ratio"] = None
            return

        std = math.sqrt(var)

        # annualization: periods_per_year based on actual elapsed time
        if self._time_years and self._time_years > 0:
            periods_per_year = float(self._ret_n) / float(self._time_years)
        else:
            periods_per_year = 0.0

        if periods_per_year <= 0:
            self.metrics["sharpe_ratio"] = None
            self.metrics["sortino_ratio"] = None
            return

        ann = math.sqrt(periods_per_year)
        self.metrics["sharpe_ratio"] = (self._ret_mean / std) * ann

        # sortino: downside deviation
        if self._downside_n > 0:
            dd = math.sqrt(self._downside_ss / float(self._downside_n))
            self.metrics["sortino_ratio"] = (self._ret_mean / dd) * ann if dd > 0 else None
        else:
            self.metrics["sortino_ratio"] = None

    def _compute_calmar(self):
        cagr = self.metrics.get("cagr", None)
        if cagr is None:
            self.metrics["calmar_ratio"] = None
            return

        max_dd_pct = self.metrics.get("max_drawdown_pct", D0)  # Decimal percent
        try:
            max_dd_frac = float(max_dd_pct) / 100.0
        except Exception:
            max_dd_frac = 0.0

        if max_dd_frac > 0:
            self.metrics["calmar_ratio"] = float(cagr) / max_dd_frac
        else:
            self.metrics["calmar_ratio"] = None
