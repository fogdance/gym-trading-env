# src/gym_trading_env/envs/position_manager.py

from collections import deque
from decimal import Decimal
from gym_trading_env.envs.position import Position

class PositionManager:
    def __init__(self):
        self.long_positions = []
        self.short_positions = []
        self.realized_pnl = Decimal('0.0')
    
    def add_long_position(self, position: Position):
        self.long_positions.append(position)
    
    def add_short_position(self, position: Position):
        self.short_positions.append(position)
    
    def close_long_position(self, size: Decimal, closing_price: Decimal):
        size = Decimal(size)
        if size > self.total_long_position():
            raise ValueError("Attempting to close more than existing long positions")
        
        pnl = Decimal('0.0')
        while size > Decimal('0.0') and self.long_positions:
            pos = self.long_positions[0]
            if pos.size <= size:
                pnl += (closing_price - pos.entry_price) * pos.size
                size -= pos.size
                self.long_positions.pop(0)
            else:
                pnl += (closing_price - pos.entry_price) * size
                pos.size -= size
                size = Decimal('0.0')
        
        self.realized_pnl += pnl
        return pnl
    
    def close_short_position(self, size: Decimal, closing_price: Decimal):
        size = Decimal(size)
        if size > self.total_short_position():
            raise ValueError("Attempting to close more than existing short positions")
        
        pnl = Decimal('0.0')
        while size > Decimal('0.0') and self.short_positions:
            pos = self.short_positions[0]
            if pos.size <= size:
                pnl += (pos.entry_price - closing_price) * pos.size
                size -= pos.size
                self.short_positions.pop(0)
            else:
                pnl += (pos.entry_price - closing_price) * size
                pos.size -= size
                size = Decimal('0.0')
        
        self.realized_pnl += pnl
        return pnl
    
    def total_long_position(self):
        return sum(pos.size for pos in self.long_positions)
    
    def total_short_position(self):
        return sum(pos.size for pos in self.short_positions)

