# src/gym_trading_env/envs/user_accounts.py
from decimal import Decimal
from gym_trading_env.envs.account import Account

class UserAccounts:
    def __init__(self, initial_balance, position_manager):
        self.cash_balance = Account(initial_balance)
        # used_margin 现在是真正“锁定保证金资产账户”
        self.used_margin = Account(Decimal('0.0'))

        self.unrealized_pnl = Decimal('0.0')  # 档1：不入账（不走 Ledger），仅风控投影
        self.realized_pnl = Decimal('0.0')    # 档1：可先保留为统计字段（不当作 Ledger 账户）
        self.position_manager = position_manager
        self.initial_balance = initial_balance

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
        # 注意：现金入账由 Ledger 负责；这里仅做统计累计
        assert isinstance(pnl, Decimal), "pnl must be a Decimal."
        self.realized_pnl += pnl
    
    def allocate_margin(self, amount: Decimal):
        # 真实划转：cash -> used_margin
        assert isinstance(amount, Decimal), "amount must be a Decimal."
        self.cash_balance.withdraw(amount)
        self.used_margin.deposit(amount)
    
    def release_margin(self, amount: Decimal):
        # 真实划转：used_margin -> cash
        assert isinstance(amount, Decimal), "amount must be a Decimal."
        self.used_margin.withdraw(amount)
        self.cash_balance.deposit(amount)

    def equity(self):
        # Equity = cash(自由现金) + used_margin(锁定保证金) + UPnL(投影)
        return self.cash_balance.get_balance() + self.used_margin.get_balance() + self.unrealized_pnl
