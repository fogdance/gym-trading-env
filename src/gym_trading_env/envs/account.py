# src/gym_trading_env/envs/account.py

from decimal import Decimal

class Account:
    def __init__(self, initial_balance=Decimal('0.0')):
        self.balance = Decimal(initial_balance)
    
    def deposit(self, amount):
        self.balance += Decimal(amount)
    
    def withdraw(self, amount):
        amount = Decimal(amount)
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount
    
    def get_balance(self):
        return self.balance