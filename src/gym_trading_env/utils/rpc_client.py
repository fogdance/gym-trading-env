# src/gym_trading_env/utils/rpc_client.py

# coding: utf-8
from __future__ import annotations

import urllib.request
import urllib.error

from gym_trading_env.utils.rpc_protocol import TradeSignal


import json
import logging
from typing import Any


class LanOrderClient:
    def __init__(
        self,
        endpoint: str,
        token: str = "",
        timeout_sec: float = 1.0,
        logger: logging.Logger | None = None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.timeout_sec = float(timeout_sec)
        # 设置 logger，默认为模块级 logger
        self.logger = logger or logging.getLogger(__name__)

    def send(self, signal: Any) -> bool:  # 用 Any 或你的 TradeSignal 类型
        url = f"{self.endpoint}/signal"
        
        # 确保有 to_json 方法，或者直接用 json.dumps
        try:
            if hasattr(signal, "to_json"):
                data = signal.to_json().encode("utf-8")
            else:
                data = json.dumps(signal.__dict__).encode("utf-8")
        except Exception as e:
            self.logger.error(f"[RPC] serialize signal failed: {e}")
            return False

        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json; charset=utf-8")
        if self.token:
            req.add_header("X-Auth-Token", self.token)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                return resp.status == 200
        except urllib.error.HTTPError as e:
            # HTTP 错误（如 4xx, 5xx），可以读取响应
            self.logger.error(
                f"[RPC] send failed HTTP {e.code} signal_id={getattr(signal, 'signal_id', 'unknown')} "
                f"action={getattr(signal, 'action_name', 'unknown')} eob={getattr(signal, 'eob', 'unknown')}"
            )
            return False
        except urllib.error.URLError as e:
            # 通常是网络超时、连接拒绝等
            reason = str(getattr(e, 'reason', e))
            self.logger.error(
                f"[RPC] send failed URLError: {reason} "
                f"signal_id={getattr(signal, 'signal_id', 'unknown')} "
                f"action={getattr(signal, 'action_name', 'unknown')} eob={getattr(signal, 'eob', 'unknown')}"
            )
            return False
        except Exception as e:
            # 其他意外错误
            self.logger.error(f"[RPC] send unexpected error: {type(e).__name__}: {e}")
            return False