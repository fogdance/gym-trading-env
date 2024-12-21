# src/gym_trading_env/envs/position.py

from decimal import Decimal

class Position:
    def __init__(self, size: Decimal, entry_price: Decimal):
        self.size = size  # position size
        self.entry_price = entry_price

    def __repr__(self):
        return f"Position(size={self.size}, entry_price={self.entry_price})"
