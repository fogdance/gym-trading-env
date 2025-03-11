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
        self.closed_trade_profits = []  # 每笔交易的实际盈亏 (正负都存)

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
        
        self.closed_trade_profits.append(pnl)

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

        self.closed_trade_profits.append(pnl)

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
        return sum((pos.size for pos in self.long_positions), Decimal('0'))
    
    def total_short_position(self) -> Decimal:
        """
        Calculates the total size of all short positions.

        Returns:
            Decimal: Total short position size.
        """
        return sum((pos.size for pos in self.short_positions), Decimal('0'))
    
    def no_position(self):
        return (self.total_long_position() + self.total_short_position()) == Decimal('0.0')

    def close_all_position(self, closing_price: Decimal, lot_size: Decimal) -> Tuple[Decimal, Decimal, Decimal]:
        """
        Closes all long and short positions by repeatedly calling close_long_position / close_short_position
        until their respective deques are empty.

        Args:
            closing_price (Decimal): The price used to close every position (applied to both long and short).
            lot_size (Decimal): The standard lot size for each position.

        Returns:
            Tuple[Decimal, Decimal, Decimal]:
                (total_pnl, total_released_margin, total_closed_size)
                - total_pnl: Sum of realized PnL for all closed positions (both long and short)
                - total_released_margin: Sum of all margins released
                - total_closed_size: Combined lot size closed (long + short)
        """
        total_pnl = Decimal('0.0')
        total_released_margin = Decimal('0.0')
        total_closed_size = Decimal('0.0')

        # Close all long positions by repeatedly calling close_long_position
        while self.long_positions:
            pnl, released_margin, closed_size = self.close_long_position(closing_price, lot_size)
            total_pnl += pnl
            total_released_margin += released_margin
            total_closed_size += closed_size

        # Close all short positions by repeatedly calling close_short_position
        while self.short_positions:
            pnl, released_margin, closed_size = self.close_short_position(closing_price, lot_size)
            total_pnl += pnl
            total_released_margin += released_margin
            total_closed_size += closed_size

        return (total_pnl, total_released_margin, total_closed_size)
    
    def calc_profit_factor(self) -> Decimal:
        """
        根据给定的一组交易盈亏值(正负)计算盈亏比 (Profit Factor).
        若无亏损或无交易则需特殊处理.
        """
        if len(self.closed_trade_profits) < 1:
            return None
        
        sum_win = Decimal('0.0')
        sum_loss = Decimal('0.0')
        for p in self.closed_trade_profits:
            if p > 0:
                sum_win += p
            else:
                sum_loss += abs(p)
        
        if sum_loss == Decimal('0.0'):
            return Decimal('3.0')
        
        return sum_win / sum_loss
