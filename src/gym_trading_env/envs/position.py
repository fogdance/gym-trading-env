# src/gym_trading_env/envs/position.py

from decimal import Decimal
from typing import Optional
from gym_trading_env.utils.decimal_util import D, D0, D1, D100, quantize_money

class Position:
    def __init__(self, size: Decimal, entry_price: Decimal, initial_margin: Decimal, open_step: int = 0,  stop_loss_price: Optional[Decimal] = None):
        """
        Initializes a Position instance.

        Args:
            size (Decimal): The size of the position.
            entry_price (Decimal): The price at which the position was opened.
            initial_margin (Decimal): The margin allocated for this position.
        """
        if not isinstance(size, Decimal):
            size = D(size)
        if not isinstance(entry_price, Decimal):
            entry_price = D(entry_price)
        if not isinstance(initial_margin, Decimal):
            initial_margin = D(initial_margin)

        self.size = size
        self.entry_price = entry_price
        self.initial_margin = initial_margin
        self.open_step = open_step
        self.stop_loss_price = stop_loss_price

    def __repr__(self):
        """
        Returns a string representation of the Position instance.

        Returns:
            str: String representation.
        """
        return (f"Position(size={self.size}, entry_price={self.entry_price}, "
                f"initial_margin={self.initial_margin}, open_step={self.open_step}, stop_loss_price={self.stop_loss_price})")
