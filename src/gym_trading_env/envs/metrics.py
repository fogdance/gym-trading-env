# src/gym_trading_env/envs/metrics.py

from decimal import Decimal
import numpy as np

class Metrics:
    def __init__(self, user_accounts, trade_record_manager):
        self.user_accounts = user_accounts
        self.trade_record_manager = trade_record_manager
        self.current_date = None
        self.previous_day_equity = user_accounts.initial_balance
        self.peak_equity = user_accounts.initial_balance
        self.metrics = {
            'current_day_lost': Decimal('0.0'),
            'current_day_lost_pct': Decimal('0.0'),
            'current_drawdown': Decimal('0.0'),
            'current_drawdown_pct': Decimal('0.0'),
            'max_drawdown': Decimal('0.0'),
            'max_drawdown_pct': Decimal('0.0'),
            'total_trades': 0,
            'winning_trades': 0,
            'max_profit': Decimal('0.0'),
            'max_loss': Decimal('0.0')
        }

    def update(self, current_timestamp):
        """更新所有指标"""
        current_equity = self.user_accounts.equity()
        
        # 单日亏损
        current_day = current_timestamp.date() if hasattr(current_timestamp, 'date') else None
        if self.current_date is None or (current_day and current_day != self.current_date):
            self.previous_day_equity = current_equity if self.current_date is not None else self.previous_day_equity
            self.metrics['current_day_lost'] = Decimal('0.0')
            self.metrics['current_day_lost_pct'] = Decimal('0.0')
            self.current_date = current_day
        day_loss = self.previous_day_equity - current_equity
        self.metrics['current_day_lost'] = max(day_loss, Decimal('0.0'))
        if self.previous_day_equity != Decimal('0.0'):
            self.metrics['current_day_lost_pct'] = (self.metrics['current_day_lost'] / self.previous_day_equity) * Decimal('100.0')

        # 账户回撤
        self.peak_equity = max(self.peak_equity, current_equity)
        current_drawdown = self.peak_equity - current_equity
        self.metrics['current_drawdown'] = max(current_drawdown, Decimal('0.0'))
        if self.peak_equity != Decimal('0.0'):
            self.metrics['current_drawdown_pct'] = (self.metrics['current_drawdown'] / self.peak_equity) * Decimal('100.0')
        
        # 最大回撤
        self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], self.metrics['current_drawdown'])
        if self.peak_equity != Decimal('0.0'):
            self.metrics['max_drawdown_pct'] = (self.metrics['max_drawdown'] / self.peak_equity) * Decimal('100.0')

        # 交易统计
        self.metrics['total_trades'] = sum(1 for t in self.trade_record_manager.trade_history if t.pnl is not None)
        self.metrics['winning_trades'] = sum(1 for t in self.trade_record_manager.trade_history 
                                            if t.pnl is not None and t.pnl >= Decimal('0.0'))
        for trade in self.trade_record_manager.trade_history:
            if trade.pnl is not None:
                self.metrics['max_profit'] = max(self.metrics['max_profit'], trade.pnl)
                self.metrics['max_loss'] = min(self.metrics['max_loss'], trade.pnl)

    def get_metrics(self):
        """返回当前指标"""
        return self.metrics