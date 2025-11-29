# src/gym_trading_env/envs/metrics.py

from decimal import Decimal
import numpy as np
from gym_trading_env.utils.decimal_util import D, D0, D1, D100, quantize_money

class Metrics:
    def __init__(self, user_accounts, trade_record_manager, risk_free_rate=Decimal('0.012')):
        self.user_accounts = user_accounts
        self.trade_record_manager = trade_record_manager
        self.risk_free_rate = risk_free_rate  # 默认年化无风险利率 1.2%
        self.current_date = None
        self.previous_day_equity = user_accounts.initial_balance
        self.peak_equity = user_accounts.initial_balance
        self.metrics = {
            'current_day_lost': D0,
            'current_day_lost_pct': D0,
            'current_drawdown': D0,
            'current_drawdown_pct': D0,
            'max_drawdown': D0,
            'max_drawdown_pct': D0,
            'total_trades': 0,
            'winning_trades': 0,
            'max_profit': D0,
            'max_loss': D0,
            'calmar_ratio': None,
            'sharpe_ratio': None,
            'win_rate': None,
        }

    def update(self, current_timestamp):
        """更新所有指标"""
        current_equity = self.user_accounts.equity()
        
        # 单日亏损
        current_day = current_timestamp.date() if hasattr(current_timestamp, 'date') else None
        if self.current_date is None or (current_day and current_day != self.current_date):
            self.previous_day_equity = current_equity if self.current_date is not None else self.previous_day_equity
            self.metrics['current_day_lost'] = D0
            self.metrics['current_day_lost_pct'] = D0
            self.current_date = current_day
        day_loss = self.previous_day_equity - current_equity
        self.metrics['current_day_lost'] = max(day_loss, D0)
        if self.previous_day_equity != D0:
            self.metrics['current_day_lost_pct'] = (self.metrics['current_day_lost'] / self.previous_day_equity) * Decimal('100.0')

        # 账户回撤
        self.peak_equity = max(self.peak_equity, current_equity)
        current_drawdown = self.peak_equity - current_equity
        self.metrics['current_drawdown'] = max(current_drawdown, D0)
        if self.peak_equity != D0:
            self.metrics['current_drawdown_pct'] = (self.metrics['current_drawdown'] / self.peak_equity) * Decimal('100.0')
        
        # 最大回撤
        self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], self.metrics['current_drawdown'])
        if self.peak_equity != D0:
            self.metrics['max_drawdown_pct'] = (self.metrics['max_drawdown'] / self.peak_equity) * Decimal('100.0')

        # 交易统计
        closed_trades = [t for t in self.trade_record_manager.trade_history if t.pnl is not None and "CLOSE" in t.operation_type]
        self.metrics['total_trades'] = len(closed_trades)
        self.metrics['winning_trades'] = sum(1 for t in closed_trades if t.pnl >= D0)
        for trade in closed_trades:
            self.metrics['max_profit'] = max(self.metrics['max_profit'], trade.pnl)
            self.metrics['max_loss'] = min(self.metrics['max_loss'], trade.pnl)

        # 胜率
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = self.metrics['winning_trades'] / self.metrics['total_trades']
        else:
            self.metrics['win_rate'] = None

        # 夏普比率（考虑无风险利率）
        if len(closed_trades) >= 2:
            returns = [float(t.pnl / self.user_accounts.initial_balance) for t in closed_trades]
            avg_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)  # 样本标准差
            if std_return > 0:
                self.metrics['sharpe_ratio'] = (avg_return - float(self.risk_free_rate) / 365) / std_return  # 日化无风险利率
            else:
                self.metrics['sharpe_ratio'] = None
        else:
            self.metrics['sharpe_ratio'] = None

        # 卡马比率
        if (closed_trades and self.metrics['max_drawdown'] > D0 and 
            len(self.trade_record_manager.trade_history) > 1):
            total_pnl = sum(t.pnl for t in closed_trades)
            time_span_days = (self.trade_record_manager.trade_history[-1].timestamp - 
                            self.trade_record_manager.trade_history[0].timestamp).days
            if time_span_days > 0:
                annualized_return = (Decimal(total_pnl) / self.user_accounts.initial_balance) / Decimal((time_span_days / 365.0))
                self.metrics['calmar_ratio'] = float(annualized_return / self.metrics['max_drawdown'])
            else:
                self.metrics['calmar_ratio'] = None
        else:
            self.metrics['calmar_ratio'] = None

    def get_metrics(self):
        # Merge behavior metrics
        beh = self._behavior_metrics()
        self.metrics.update({
            k: (D(v) if isinstance(v, float) else v)
            for k, v in beh.items() if v is not None
        })
        # Keep None values as-is (e.g., before any steps).
        for k, v in beh.items():
            if v is None:
                self.metrics[k] = None
        return self.metrics

    
    def on_step(self, action, action_success: bool, in_market: bool):
        """Lightweight per-step counters for behavior analysis."""
        # Lazily create once.
        if not hasattr(self, "_counters"):
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
            }
        c = self._counters
        c["steps_total"] += 1
        if in_market:
            c["in_market_steps"] += 1

        # Normalize action class without importing here.
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

    def _behavior_metrics(self):
        """Compute derived ratios safely from counters."""
        out = {}
        c = getattr(self, "_counters", None)
        if not c or c["steps_total"] == 0:
            return {
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
            }
        steps = float(c["steps_total"])
        out["action_hold_ratio"]  = float(c["hold_steps"])  / steps
        out["action_long_open_ratio"]  = float(c["long_open_steps"])  / steps
        out["action_long_close_ratio"] = float(c["long_close_steps"]) / steps
        out["action_short_open_ratio"]  = float(c["short_open_steps"])  / steps
        out["action_short_close_ratio"] = float(c["short_close_steps"]) / steps

        out["action_long_open_success_ratio"]  = float(c["long_open_steps_success"])  / float(c["long_open_steps"]) if float(c["long_open_steps"]) != 0 else 0
        out["action_long_close_success_ratio"] = float(c["long_close_steps_success"]) / float(c["long_close_steps"]) if float(c["long_close_steps"]) != 0 else 0
        out["action_short_open_success_ratio"]  = float(c["short_open_steps_success"])  / float(c["short_open_steps"])  if c["short_open_steps"]  else 0.0
        out["action_short_close_success_ratio"] = float(c["short_close_steps_success"]) / float(c["short_close_steps"]) if c["short_close_steps"] else 0.0

        out["exposure_ratio"]     = float(c["in_market_steps"]) / steps

        # Count open/close events from trade history (robust to multiple slots).
        th = self.trade_record_manager.trade_history
        out["trades_opened"] = sum(1 for t in th if "OPEN"  in t.operation_type)
        out["trades_closed"] = sum(1 for t in th if "CLOSE" in t.operation_type)
        return out
    