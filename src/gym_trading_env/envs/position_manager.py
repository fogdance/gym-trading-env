# src/gym_trading_env/envs/position_manager.py
from decimal import Decimal
from typing import Tuple, Optional, List
from gym_trading_env.envs.position import Position

class PositionManager:
    def __init__(self, logger = None, long_slots: int = 2, short_slots: int = 2):
        """
        Initializes the PositionManager with configurable number of long and short slots.

        Args:
            long_slots (int): Number of long position slots (default: 2)
            short_slots (int): Number of short position slots (default: 2)
        """
        self.long_positions: List[Optional[Position]] = [None] * long_slots  # e.g., [None, None]
        self.short_positions: List[Optional[Position]] = [None] * short_slots  # e.g., [None, None]
        self.realized_pnl = Decimal('0.0')
        self.closed_trade_profits = []
        self.logger = logger

    def add_long_position(self, position: Position, slot: Optional[int] = None):
        """
        Adds a new long position. If slot is None, uses the first available slot (FIFO-like).
        If slot is specified, uses the exact slot.

        Args:
            position (Position): The long position to add
            slot (Optional[int]): Specific slot index (e.g., 0 for 'long1'), or None for auto-allocation

        Raises:
            ValueError: If slot is invalid or occupied, or no slots available
        """
        if slot is not None:
            if not (0 <= slot < len(self.long_positions)):
                raise ValueError(f"Invalid long slot: {slot}. Must be between 0 and {len(self.long_positions)-1}")
            if self.long_positions[slot] is not None:
                raise ValueError(f"Long slot {slot} is already occupied")
            self.long_positions[slot] = position
        else:
            # Find first empty slot
            for i in range(len(self.long_positions)):
                if self.long_positions[i] is None:
                    self.long_positions[i] = position
                    return
            raise ValueError("No available long slots")

    def add_short_position(self, position: Position, slot: Optional[int] = None):
        """
        Adds a new short position. If slot is None, uses the first available slot (FIFO-like).
        If slot is specified, uses the exact slot.

        Args:
            position (Position): The short position to add
            slot (Optional[int]): Specific slot index (e.g., 0 for 'short3'), or None for auto-allocation

        Raises:
            ValueError: If slot is invalid or occupied, or no slots available
        """
        if slot is not None:
            if not (0 <= slot < len(self.short_positions)):
                raise ValueError(f"Invalid short slot: {slot}. Must be between 0 and {len(self.short_positions)-1}")
            if self.short_positions[slot] is not None:
                raise ValueError(f"Short slot {slot} is already occupied")
            self.short_positions[slot] = position
        else:
            # Find first empty slot
            for i in range(len(self.short_positions)):
                if self.short_positions[i] is None:
                    self.short_positions[i] = position
                    return
            raise ValueError("No available short slots")

    def close_long_position(self, closing_price: Decimal, lot_size: Decimal, slot: Optional[int] = None) -> Tuple[Decimal, Decimal, Decimal]:
        """
        Closes a long position. If slot is None, closes the earliest position (FIFO).
        If slot is specified, closes the position at that slot.

        Args:
            closing_price (Decimal): The price at which the position is closed
            lot_size (Decimal): The standard lot size
            slot (Optional[int]): Specific slot index, or None for earliest position

        Returns:
            Tuple[Decimal, Decimal, Decimal]: Realized P&L, released margin, closed size

        Raises:
            ValueError: If no positions to close or specified slot is empty/invalid
        """
        if slot is not None:
            if not (0 <= slot < len(self.long_positions)):
                raise ValueError(f"Invalid long slot: {slot}. Must be between 0 and {len(self.long_positions)-1}")
            if self.long_positions[slot] is None:
                raise ValueError(f"No long position in slot {slot} to close")
            pos = self.long_positions[slot]
            self.long_positions[slot] = None
        else:
            # Find earliest non-empty slot
            for i in range(len(self.long_positions)):
                if self.long_positions[i] is not None:
                    pos = self.long_positions[i]
                    self.long_positions[i] = None
                    break
            else:
                raise ValueError("No long positions to close")

        # Calculate P&L: (Closing Price - Entry Price) * Size * Lot Size
        pnl = (closing_price - pos.entry_price) * pos.size * lot_size
        self.realized_pnl += pnl
        self.closed_trade_profits.append(pnl)
        released_margin = pos.initial_margin
        closed_size = pos.size
        return pnl, released_margin, closed_size

    def close_short_position(self, closing_price: Decimal, lot_size: Decimal, slot: Optional[int] = None) -> Tuple[Decimal, Decimal, Decimal]:
        """
        Closes a short position. If slot is None, closes the earliest position (FIFO).
        If slot is specified, closes the position at that slot.

        Args:
            closing_price (Decimal): The price at which the position is closed
            lot_size (Decimal): The standard lot size
            slot (Optional[int]): Specific slot index, or None for earliest position

        Returns:
            Tuple[Decimal, Decimal, Decimal]: Realized P&L, released margin, closed size

        Raises:
            ValueError: If no positions to close or specified slot is empty/invalid
        """
        if slot is not None:
            if not (0 <= slot < len(self.short_positions)):
                raise ValueError(f"Invalid short slot: {slot}. Must be between 0 and {len(self.short_positions)-1}")
            if self.short_positions[slot] is None:
                raise ValueError(f"No short position in slot {slot} to close")
            pos = self.short_positions[slot]
            self.short_positions[slot] = None
        else:
            # Find earliest non-empty slot
            for i in range(len(self.short_positions)):
                if self.short_positions[i] is not None:
                    pos = self.short_positions[i]
                    self.short_positions[i] = None
                    break
            else:
                raise ValueError("No short positions to close")

        # Calculate P&L: (Entry Price - Closing Price) * Size * Lot Size
        pnl = (pos.entry_price - closing_price) * pos.size * lot_size
        self.realized_pnl += pnl
        self.closed_trade_profits.append(pnl)
        released_margin = pos.initial_margin
        closed_size = pos.size
        return pnl, released_margin, closed_size

    def total_long_position(self) -> Decimal:
        """
        Calculates the total size of all long positions.

        Returns:
            Decimal: Total long position size
        """
        return sum((pos.size for pos in self.long_positions if pos is not None), Decimal('0'))

    def total_short_position(self) -> Decimal:
        """
        Calculates the total size of all short positions.

        Returns:
            Decimal: Total short position size
        """
        return sum((pos.size for pos in self.short_positions if pos is not None), Decimal('0'))

    def no_position(self) -> bool:
        """
        Checks if there are no open positions.

        Returns:
            bool: True if all slots are empty
        """
        return all(pos is None for pos in self.long_positions) and all(pos is None for pos in self.short_positions)

    def close_all_position(self, closing_price: Decimal, lot_size: Decimal) -> Tuple[Decimal, Decimal, Decimal]:
        """
        Closes all positions in all slots.

        Args:
            closing_price (Decimal): The price used to close every position
            lot_size (Decimal): The standard lot size

        Returns:
            Tuple[Decimal, Decimal, Decimal]: Total P&L, total released margin, total closed size
        """
        total_pnl = Decimal('0.0')
        total_released_margin = Decimal('0.0')
        total_closed_size = Decimal('0.0')

        # Close all long positions
        for i in range(len(self.long_positions)):
            if self.long_positions[i] is not None:
                pnl, released_margin, closed_size = self.close_long_position(closing_price, lot_size, slot=i)
                total_pnl += pnl
                total_released_margin += released_margin
                total_closed_size += closed_size

        # Close all short positions
        for i in range(len(self.short_positions)):
            if self.short_positions[i] is not None:
                pnl, released_margin, closed_size = self.close_short_position(closing_price, lot_size, slot=i)
                total_pnl += pnl
                total_released_margin += released_margin
                total_closed_size += closed_size

        return total_pnl, total_released_margin, total_closed_size

    def calc_profit_factor(self) -> Optional[Decimal]:
        """
        Calculates the profit factor based on closed trade profits.

        Returns:
            Decimal or None: Profit factor (sum of wins / sum of losses), or None if no trades
        """
        if not self.closed_trade_profits:
            return None

        sum_win = Decimal('0.0')
        sum_loss = Decimal('0.0')
        for p in self.closed_trade_profits:
            if p > 0:
                sum_win += p
            else:
                sum_loss += abs(p)

        if sum_loss == Decimal('0.0'):
            return Decimal('3.0')  # 默认值，避免除零

        return sum_win / sum_loss

    def get_position(self, slot: int, is_long: bool) -> Optional[Position]:
        """
        Gets the position in the specified slot.

        Args:
            slot (int): The slot index
            is_long (bool): True for long positions, False for short positions

        Returns:
            Position or None: The position object or None if slot is empty

        Raises:
            ValueError: If slot is invalid
        """
        positions = self.long_positions if is_long else self.short_positions
        if not (0 <= slot < len(positions)):
            raise ValueError(f"Invalid slot: {slot}. Must be between 0 and {len(positions)-1}")
        return positions[slot]