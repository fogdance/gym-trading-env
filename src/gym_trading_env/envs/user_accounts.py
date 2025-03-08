# src/gym_trading_env/envs/user_accounts.py

from decimal import Decimal
from gym_trading_env.envs.account import Account


class UserAccounts:
    def __init__(self, initial_balance, position_manager):
        self.balance = Account(initial_balance)
        self.margin = Account(Decimal('0.0'))
        self.unrealized_pnl = Decimal('0.0')
        self.realized_pnl = Decimal('0.0')
        self.position_manager = position_manager
        
        # 用于计算单日亏损和回撤的属性
        self.initial_balance = initial_balance              # 初始余额
        self.previous_day_equity = initial_balance          # 前一天权益
        self.peak_equity = initial_balance                  # 历史最高权益
        self.max_drawdown = Decimal('0.0')                 # 最大回撤（绝对值）
        self.current_day_lost = Decimal('0.0')             # 当前单日亏损（绝对值）
        self.current_day_lost_pct = Decimal('0.0')         # 当前单日亏损（百分比）
        self.current_drawdown = Decimal('0.0')             # 当前回撤（绝对值）
        self.current_drawdown_pct = Decimal('0.0')         # 当前回撤（百分比）
        self.max_drawdown_pct = Decimal('0.0')             # 最大回撤（百分比）
        self.current_date = None                           # 当前日期

    @property
    def long_position(self):
        return self.position_manager.total_long_position() if self.position_manager else Decimal('0.0')
    
    @property
    def short_position(self):
        return self.position_manager.total_short_position() if self.position_manager else Decimal('0.0')
    
    def update_unrealized_pnl(self, pnl_change):
        assert isinstance(pnl_change, Decimal), "pnl_change must be a Decimal."
        self.unrealized_pnl += pnl_change
    
    def realize_pnl(self, pnl):
        self.realized_pnl += pnl
        self.balance.deposit(pnl)
    
    def allocate_margin(self, amount):
        if amount > (self.balance.get_balance() + self.realized_pnl):
            raise ValueError("Insufficient funds to allocate margin.")        
        self.balance.withdraw(amount)
        self.margin.deposit(amount)
    
    def release_margin(self, amount):
        if amount > self.margin.get_balance():
            raise ValueError("Cannot release more margin than allocated.")
        self.margin.withdraw(amount)
        self.balance.deposit(amount)

    def equity(self):
        """计算当前权益"""
        return self.balance.get_balance() + self.realized_pnl + self.unrealized_pnl

    def update_metrics(self, current_timestamp):
        """更新单日亏损和账户回撤，包括绝对值和百分比"""
        current_equity = self.equity()

        # 处理日期
        current_day = current_timestamp.date() if hasattr(current_timestamp, 'date') else None

        # 单日亏损：仅在新的一天开始时重置，并在每一步更新
        if self.current_date is None or (current_day and current_day != self.current_date):
            # 新的一天，记录前一天的权益并重置亏损
            self.previous_day_equity = current_equity if self.current_date is not None else self.previous_day_equity
            self.current_day_lost = Decimal('0.0')
            self.current_day_lost_pct = Decimal('0.0')
            self.current_date = current_day
        # 计算当前单日亏损（绝对值和百分比）
        day_loss = self.previous_day_equity - current_equity
        self.current_day_lost = max(day_loss, Decimal('0.0'))  # 只记录亏损（正值）
        if self.previous_day_equity != Decimal('0.0'):
            self.current_day_lost_pct = (self.current_day_lost / self.previous_day_equity) * Decimal('100.0')
        else:
            self.current_day_lost_pct = Decimal('0.0')

        # 账户回撤
        self.peak_equity = max(self.peak_equity, current_equity)  # 更新峰值
        current_drawdown = self.peak_equity - current_equity
        self.current_drawdown = max(current_drawdown, Decimal('0.0'))  # 当前回撤（正值）
        if self.peak_equity != Decimal('0.0'):
            self.current_drawdown_pct = (self.current_drawdown / self.peak_equity) * Decimal('100.0')
        else:
            self.current_drawdown_pct = Decimal('0.0')
        
        # 更新最大回撤（绝对值和百分比）
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        if self.peak_equity != Decimal('0.0'):
            self.max_drawdown_pct = (self.max_drawdown / self.peak_equity) * Decimal('100.0')
        else:
            self.max_drawdown_pct = Decimal('0.0')