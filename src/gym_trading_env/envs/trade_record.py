# src/gym_trading_env/envs/trade_record.py

from decimal import Decimal
from datetime import datetime

class TradeRecord:
    def __init__(self, timestamp, operation_type: str, position_size: Decimal, price: Decimal, required_margin: Decimal,
                 fee: Decimal, balance: Decimal, leverage: Decimal, free_margin: Decimal, pnl: Decimal = Decimal('0.0'),
                 closed_size: Decimal = Decimal('0.0'), released_margin: Decimal = Decimal('0.0')):
        """
        Initialize a trade record

        Args:
            operation_type (str): The type of operation, 'Long/Open', 'Short/Open', 'Long/Close', 'Short/Close'
            position_size (Decimal): The size of the position
            price (Decimal): The trade price (opening or closing price)
            required_margin (Decimal): The required margin
            fee (Decimal): The trading fee
            balance (Decimal): The current balance
            leverage (Decimal): The current leverage
            free_margin (Decimal): The available margin
            pnl (Decimal, optional): The profit or loss, defaults to 0.0
            closed_size (Decimal, optional): The position size when closed, defaults to 0.0
            released_margin (Decimal, optional): The released margin when the position is closed, defaults to 0.0
        """
        self.timestamp = timestamp
        self.operation_type = operation_type  # Operation type: Long/Open, Short/Open, Long/Close, Short/Close
        self.position_size = position_size  # Position size
        self.price = price  # Trade price
        self.required_margin = required_margin  # Required margin
        self.fee = fee  # Trading fee
        self.balance = balance  # Current balance
        self.leverage = leverage  # Current leverage
        self.free_margin = free_margin  # Available margin
        self.pnl = pnl  # Profit or loss (only recorded when the position is closed)
        self.closed_size = closed_size  # Position size when closed
        self.released_margin = released_margin  # Released margin (when closing the position)

    def to_dict(self):
        """Convert the trade record to a dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "operation_type": self.operation_type,
            "position_size": str(self.position_size),
            "price": str(self.price),
            "required_margin": str(self.required_margin),
            "fee": str(self.fee),
            "balance": str(self.balance),
            "leverage": str(self.leverage),
            "free_margin": str(self.free_margin),
            "pnl": str(self.pnl),
            "closed_size": str(self.closed_size),
            "released_margin": str(self.released_margin)
        }

    def __repr__(self):
        return f"TradeRecord({self.operation_type}, {self.position_size}, {self.price}, {self.required_margin}, {self.fee})"
