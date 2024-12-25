# src/gym_trading_env/envs/position_manager.py

from collections import deque
from decimal import Decimal
from typing import Tuple
from gym_trading_env.envs.position import Position

class PositionManager:
    def __init__(self):
        """
        Initializes the PositionManager with separate deques for long and short positions.
        """
        self.long_positions = deque()
        self.short_positions = deque()
        self.realized_pnl = Decimal('0.0')
    
    def add_long_position(self, position: Position):
        """
        Adds a new long position.

        Args:
            position (Position): The long position to add.
        """
        self.long_positions.append(position)
    
    def add_short_position(self, position: Position):
        """
        Adds a new short position.

        Args:
            position (Position): The short position to add.
        """
        self.short_positions.append(position)
    
    def close_long_position(self, closing_price: Decimal, lot_size: Decimal) -> Tuple[Decimal, Decimal, Decimal]:
        """
        Closes the earliest long position, calculating realized P&L and releasing margin.

        Args:
            closing_price (Decimal): The price at which the position is closed.
            lot_size (Decimal): The standard lot size.

        Returns:
            Tuple[Decimal, Decimal, Decimal]: Realized P&L, released margin, and closed size.

        Raises:
            ValueError: If there are no long positions to close.
        """
        if not self.long_positions:
            raise ValueError("No long positions to close.")
        
        pos = self.long_positions.popleft()
        
        # Calculate P&L: (Closing Price - Entry Price) * Size * Lot Size
        pnl = (closing_price - pos.entry_price) * pos.size * lot_size
        self.realized_pnl += pnl
        
        # Release initial margin
        released_margin = pos.initial_margin
        
        # Closed size
        closed_size = pos.size
        
        return pnl, released_margin, closed_size
    
    def close_short_position(self, closing_price: Decimal, lot_size: Decimal) -> Tuple[Decimal, Decimal, Decimal]:
        """
        Closes the earliest short position, calculating realized P&L and releasing margin.

        Args:
            closing_price (Decimal): The price at which the position is closed.
            lot_size (Decimal): The standard lot size.

        Returns:
            Tuple[Decimal, Decimal, Decimal]: Realized P&L, released margin, and closed size.

        Raises:
            ValueError: If there are no short positions to close.
        """
        if not self.short_positions:
            raise ValueError("No short positions to close.")
        
        pos = self.short_positions.popleft()
        
        # Calculate P&L: (Entry Price - Closing Price) * Size * Lot Size
        pnl = (pos.entry_price - closing_price) * pos.size * lot_size
        self.realized_pnl += pnl
        
        # Release initial margin
        released_margin = pos.initial_margin
        
        # Closed size
        closed_size = pos.size
        
        return pnl, released_margin, closed_size
    
    def total_long_position(self) -> Decimal:
        """
        Calculates the total size of all long positions.

        Returns:
            Decimal: Total long position size.
        """
        return sum(pos.size for pos in self.long_positions)
    
    def total_short_position(self) -> Decimal:
        """
        Calculates the total size of all short positions.

        Returns:
            Decimal: Total short position size.
        """
        return sum(pos.size for pos in self.short_positions)
