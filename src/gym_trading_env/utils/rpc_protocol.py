# src/gym_trading_env/utils/rpc_protocol.py

# coding: utf-8
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import json
import time


@dataclass
class TradeSignal:
    # 去重：通常用 f"{symbol}|{eob}"
    signal_id: str

    symbol: str                 # e.g. "DCE.jm2605"
    eob: str                    # "YYYY-mm-dd HH:MM:SS"

    # 你的 env.action_space 是 Discrete(len(valid_actions))，因此 action_index 是 0..4
    action_index: int           # 0..4
    action_name: str            # "HOLD" / "LONG_OPEN0" / ...

    volume: int = 1
    price: float = 0.0

    # 可选：携带账户/风控快照，方便执行端落库/审计
    meta: Optional[Dict[str, Any]] = None

    ts_client: float = 0.0

    def to_json(self) -> str:
        if self.meta is None:
            self.meta = {}
        if not self.ts_client:
            self.ts_client = time.time()
        d = {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "eob": self.eob,
            "action_index": int(self.action_index),
            "action_name": str(self.action_name),
            "volume": int(self.volume),
            "price": float(self.price),
            "meta": self.meta,
            "ts_client": float(self.ts_client),
        }
        return json.dumps(d, ensure_ascii=False)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TradeSignal":
        return TradeSignal(
            signal_id=str(d.get("signal_id", "")),
            symbol=str(d.get("symbol", "")),
            eob=str(d.get("eob", "")),
            action_index=int(d.get("action_index", 0)),
            action_name=str(d.get("action_name", "")),
            volume=int(d.get("volume", 1)),
            price=float(d.get("price", 0.0) or 0.0),
            meta=d.get("meta", None),
            ts_client=float(d.get("ts_client", 0.0) or 0.0),
        )
