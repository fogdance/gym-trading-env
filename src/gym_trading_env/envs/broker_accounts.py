# src/gym_trading_env/envs/broker_accounts.py

from decimal import Decimal

from gym_trading_env.envs.account import Account

class BrokerAccounts:
    def __init__(self):
        self.fees = Account(Decimal('0.0'))
        self.balance = Account(Decimal('0.0'))  # Optional
    
    def collect_fee(self, amount):
        self.fees.deposit(amount)