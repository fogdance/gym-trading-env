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

    # RPC signal uses execution Action.value, not the env target-position action index.
    # The env action_space contract is target index 0..2: SHORT / FLAT / LONG.
    action_index: int           # execution Action.value
    action_name: str            # "HOLD" / "LONG_OPEN0" / ...

    volume: int = 1
    price: float = 0.0

    # ForexCode.value from env. Only SUCCESS(0) should be live-executable.
    action_result: int = 0

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
            "action_result": int(self.action_result),
            "meta": self.meta,
            "ts_client": float(self.ts_client),
        }
        return json.dumps(d, ensure_ascii=False)

    def is_success(self) -> bool:
        return int(self.action_result) == 0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TradeSignal":
        meta = d.get("meta", None)
        action_result = d.get("action_result", None)
        if action_result is None and isinstance(meta, dict):
            action_result = meta.get("result", meta.get("action_result", 0))
        if action_result is None:
            action_result = 0
        return TradeSignal(
            signal_id=str(d.get("signal_id", "")),
            symbol=str(d.get("symbol", "")),
            eob=str(d.get("eob", "")),
            action_index=int(d.get("action_index", 0)),
            action_name=str(d.get("exec_action", d.get("action_name", ""))),
            volume=int(d.get("volume", 1)),
            price=float(d.get("price", 0.0) or 0.0),
            action_result=int(action_result),
            meta=meta,
            ts_client=float(d.get("ts_client", 0.0) or 0.0),
        )
