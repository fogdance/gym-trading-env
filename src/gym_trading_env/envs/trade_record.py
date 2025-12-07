# src/gym_trading_env/envs/trade_record.py

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional


class TradeRecord:
    """
    Immutable-ish trade log record.

    meta: optional dict for extra structured info, e.g.
      {
        "side": "long"|"short",
        "slot": 0,
        "reason": "MANUAL"|"STOP_LOSS"|"TAKE_PROFIT"|"EOD",
        "sl": "...",
        "tp": "..."
      }
    """

    __slots__ = (
        "timestamp",
        "operation_type",
        "position_size",
        "open_price",
        "close_price",
        "required_margin",
        "fee",
        "balance",
        "leverage",
        "free_margin",
        "pnl",
        "closed_size",
        "released_margin",
        "meta",
    )

    def __init__(
        self,
        timestamp,
        operation_type: str,
        position_size: Decimal,
        open_price: Decimal,
        close_price: Decimal,
        required_margin: Decimal,
        fee: Decimal,
        balance: Decimal,
        leverage: Decimal,
        free_margin: Decimal,
        pnl: Decimal = Decimal("0.0"),
        closed_size: Decimal = Decimal("0.0"),
        released_margin: Decimal = Decimal("0.0"),
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.timestamp = timestamp
        self.operation_type = operation_type

        self.position_size = position_size
        self.open_price = open_price
        self.close_price = close_price

        self.required_margin = required_margin
        self.fee = fee
        self.balance = balance
        self.leverage = leverage
        self.free_margin = free_margin

        self.pnl = pnl
        self.closed_size = closed_size
        self.released_margin = released_margin

        self.meta = meta or {}

    def to_dict(self) -> Dict[str, Any]:
        # robust timestamp serialization (pd.Timestamp / datetime / str)
        ts = self.timestamp
        if hasattr(ts, "isoformat"):
            ts_s = ts.isoformat()
        else:
            ts_s = str(ts)

        return {
            "timestamp": ts_s,
            "operation_type": self.operation_type,
            "position_size": str(self.position_size),
            "open_price": str(self.open_price),
            "close_price": str(self.close_price),
            "required_margin": str(self.required_margin),
            "fee": str(self.fee),
            "balance": str(self.balance),
            "leverage": str(self.leverage),
            "free_margin": str(self.free_margin),
            "pnl": str(self.pnl),
            "closed_size": str(self.closed_size),
            "released_margin": str(self.released_margin),
            "meta": self.meta,
        }

    def __repr__(self) -> str:
        return (
            "TradeRecord("
            f"{self.operation_type}, size={self.position_size}, "
            f"open={self.open_price}, close={self.close_price}, "
            f"fee={self.fee}, pnl={self.pnl}, meta={self.meta})"
        )
