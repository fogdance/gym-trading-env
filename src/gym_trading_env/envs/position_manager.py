# src/gym_trading_env/envs/position_manager.py

from collections import deque
from decimal import Decimal
from gym_trading_env.envs.position import Position

class PositionManager:
    def __init__(self):
        self.long_positions = deque()   # long position queue
        self.short_positions = deque()  # short position queue

    def add_long_position(self, position: Position):
        self.long_positions.append(position)

    def add_short_position(self, position: Position):
        self.short_positions.append(position)

    def close_long_position(self, size: Decimal):
        """
        Close the long position according to the FIFO principle,
        """
        closed_positions = []
        total_pnl = Decimal('0.0')
        remaining_size = size

        while self.long_positions and remaining_size > Decimal('0.0'):
            position = self.long_positions.popleft()
            if position.size <= remaining_size:
                closed_size = position.size
                remaining_size -= closed_size
            else:
                closed_size = remaining_size
                position.size -= closed_size
                self.long_positions.appendleft(position)
                remaining_size = Decimal('0.0')
            closed_positions.append(Position(closed_size, position.entry_price))
        
        return closed_positions, total_pnl

    def close_short_position(self, size: Decimal):
        """
        Close the short position according to the FIFO principle,
        """
        closed_positions = []
        total_pnl = Decimal('0.0')
        remaining_size = size

        while self.short_positions and remaining_size > Decimal('0.0'):
            position = self.short_positions.popleft()
            if position.size <= remaining_size:
                closed_size = position.size
                remaining_size -= closed_size
            else:
                closed_size = remaining_size
                position.size -= closed_size
                self.short_positions.appendleft(position)
                remaining_size = Decimal('0.0')
            closed_positions.append(Position(closed_size, position.entry_price))
        
        return closed_positions, total_pnl
