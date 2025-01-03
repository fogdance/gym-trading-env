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
    
    @property
    def long_position(self):
        if self.position_manager:
            return self.position_manager.total_long_position()
    
    @property
    def short_position(self):
        if self.position_manager:
            return self.position_manager.total_short_position()
    
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
