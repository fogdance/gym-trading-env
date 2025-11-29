# src/gym_trading_env/envs/account.py
from decimal import Decimal

class Account:
    def __init__(self, initial_balance=Decimal('0.0')):
        self.balance = Decimal(initial_balance)
    
    def deposit(self, amount):
        assert isinstance(amount, Decimal), "Deposit amount must be a Decimal."
        self.balance += Decimal(amount)
    
    def withdraw(self, amount):
        assert isinstance(amount, Decimal), "Withdraw amount must be a Decimal."
        amount = Decimal(amount)
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount
    
    def get_balance(self):
        return self.balance

    # NEW: 给 Ledger snapshot/restore 用
    def set_balance(self, new_balance: Decimal):
        assert isinstance(new_balance, Decimal), "new_balance must be a Decimal."
        self.balance = new_balance
