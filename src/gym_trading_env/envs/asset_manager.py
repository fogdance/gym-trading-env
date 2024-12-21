# src/gym_trading_env/utils/asset_manager.py

from decimal import Decimal

class AssetManager:
    def __init__(self, initial_balance: Decimal, leverage: Decimal, lot_size: Decimal, trading_fees: Decimal):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.lot_size = lot_size
        self.trading_fees = trading_fees
        self.used_margin = Decimal('0.0')

    def update_balance(self, amount: Decimal):
        self.balance += amount

    def calculate_used_margin(self, position_size: Decimal, entry_price: Decimal):
        margin = (position_size * self.lot_size * entry_price) / self.leverage
        self.used_margin += margin

    def release_margin(self, position_size: Decimal, entry_price: Decimal):
        margin = (position_size * self.lot_size * entry_price) / self.leverage
        self.used_margin -= margin

    def get_free_margin(self, equity: Decimal) -> Decimal:
        return equity - self.used_margin

    def apply_fee(self, amount: Decimal) -> Decimal:
        return amount * self.trading_fees
