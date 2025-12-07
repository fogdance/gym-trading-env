# src/gym_trading_env/envs/position.py

from __future__ import annotations

from decimal import Decimal
from typing import Optional, Dict, Any

from gym_trading_env.utils.decimal_util import D


class Position:
    """
    A single position slot.

    IMPORTANT:
    - entry_price: the actual executed entry price (ask for long, bid for short)
    - stop_loss_price / take_profit_price: target EXECUTION prices for closing
        * long close executes at bid
        * short close executes at ask
      We store them in the same "close-exec-price" convention, so RR is consistent.
    """

    __slots__ = (
        "size",
        "entry_price",
        "initial_margin",
        "open_step",
        "stop_loss_price",
        "take_profit_price",
    )

    def __init__(
        self,
        size: Decimal,
        entry_price: Decimal,
        initial_margin: Decimal,
        open_step: int = 0,
        stop_loss_price: Optional[Decimal] = None,
        take_profit_price: Optional[Decimal] = None,
    ):
        if not isinstance(size, Decimal):
            size = D(size)
        if not isinstance(entry_price, Decimal):
            entry_price = D(entry_price)
        if not isinstance(initial_margin, Decimal):
            initial_margin = D(initial_margin)

        if size <= D(0):
            raise ValueError(f"Position.size must be > 0, got {size}")
        if entry_price <= D(0):
            raise ValueError(f"Position.entry_price must be > 0, got {entry_price}")
        if initial_margin < D(0):
            raise ValueError(f"Position.initial_margin must be >= 0, got {initial_margin}")

        if stop_loss_price is not None and not isinstance(stop_loss_price, Decimal):
            stop_loss_price = D(stop_loss_price)
        if take_profit_price is not None and not isinstance(take_profit_price, Decimal):
            take_profit_price = D(take_profit_price)

        self.size = size
        self.entry_price = entry_price
        self.initial_margin = initial_margin
        self.open_step = int(open_step)

        self.stop_loss_price = stop_loss_price
        self.take_profit_price = take_profit_price

    def to_dict(self) -> Dict[str, Any]:
        return {
            "size": str(self.size),
            "entry_price": str(self.entry_price),
            "initial_margin": str(self.initial_margin),
            "open_step": int(self.open_step),
            "stop_loss_price": None if self.stop_loss_price is None else str(self.stop_loss_price),
            "take_profit_price": None if self.take_profit_price is None else str(self.take_profit_price),
        }

    def __repr__(self) -> str:
        return (
            "Position("
            f"size={self.size}, "
            f"entry_price={self.entry_price}, "
            f"initial_margin={self.initial_margin}, "
            f"open_step={self.open_step}, "
            f"stop_loss_price={self.stop_loss_price}, "
            f"take_profit_price={self.take_profit_price}"
            ")"
        )
