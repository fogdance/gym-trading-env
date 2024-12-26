# src/gym_trading_env/envs/broker_accounts.py

from decimal import Decimal
from gym_trading_env.envs.account import Account

class BrokerAccounts:
    def __init__(self):
        self.balance = Account(Decimal('0.0'))  # Broker's balance account for P&L hedging
        self.fees = Account(Decimal('0.0'))    # Broker's fees account for storing fees and spread profits

    def collect_fee(self, amount: Decimal):
        """
        Collects trading fees and stores them in the fees account.

        Args:
            amount (Decimal): The fee amount to be collected.
        """
        self.fees.deposit(amount)

    def adjust_balance(self, amount: Decimal):
        """
        Adjusts the broker's balance account for P&L hedging.

        Args:
            amount (Decimal): The amount to adjust the balance by.
        """
        self.balance.deposit(amount)

    def get_total_balance(self) -> Decimal:
        """
        Retrieves the total balance of the broker, including both balance and fees accounts.

        Returns:
            Decimal: The total balance.
        """
        return self.balance.get_balance() + self.fees.get_balance()
