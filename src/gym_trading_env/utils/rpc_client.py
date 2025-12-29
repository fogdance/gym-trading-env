# src/gym_trading_env/utils/rpc_client.py

# coding: utf-8
from __future__ import annotations

import urllib.request
import urllib.error

from gym_trading_env.utils.rpc_protocol import TradeSignal
from typing import Dict, Any


import json
import logging


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

    def send(self, signal: Any) -> Dict[str, Any]:
        """
        发送信号并返回服务端的完整响应解析结果
        返回示例:
            {"success": True, "msg": "enqueued", "raw": {...}}
            {"success": False, "msg": "duplicate", "raw": {...}}
            {"success": False, "msg": "network_error", "error": "..."}
        """
        url = f"{self.endpoint}/signal"

        try:
            if hasattr(signal, "to_json"):
                data = signal.to_json().encode("utf-8")
            else:
                data = json.dumps(signal.__dict__, ensure_ascii=False).encode("utf-8")
        except Exception as e:
            self.logger.error(f"[RPC] serialize signal failed: {e}")
            return {"success": False, "msg": "serialize_failed", "error": str(e)}

        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json; charset=utf-8")
        if self.token:
            req.add_header("X-Auth-Token", self.token)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = resp.read().decode("utf-8")
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    payload = {"raw_body": body}

                # 推荐：以 payload["ok"] 为准，而不是只看 status
                success = payload.get("ok", False)
                msg = payload.get("msg", "no_msg")

                if success:
                    self.logger.info(
                        f"[RPC] sent OK signal_id={getattr(signal, 'signal_id', 'unknown')} "
                        f"action={getattr(signal, 'action_name', 'unknown')} eob={getattr(signal, 'eob', 'unknown')} "
                        f"msg={msg}"
                    )
                else:
                    self.logger.warning(
                        f"[RPC] sent but rejected signal_id={getattr(signal, 'signal_id', 'unknown')} "
                        f"action={getattr(signal, 'action_name', 'unknown')} eob={getattr(signal, 'eob', 'unknown')} "
                        f"status={resp.status} msg={msg}"
                    )

                return {"success": success, "msg": msg, "status": resp.status, "raw": payload}

        except urllib.error.HTTPError as e:
            # 读取错误响应体（如果有）
            try:
                err_body = e.read().decode("utf-8")
                try:
                    err_payload = json.loads(err_body)
                except:
                    err_payload = {"raw_body": err_body}
            except:
                err_payload = {}

            self.logger.error(
                f"[RPC] send failed HTTP {e.code} signal_id={getattr(signal, 'signal_id', 'unknown')} "
                f"action={getattr(signal, 'action_name', 'unknown')} eob={getattr(signal, 'eob', 'unknown')} "
                f"response={err_payload}"
            )
            return {"success": False, "msg": "http_error", "status": e.code, "raw": err_payload}

        except urllib.error.URLError as e:
            reason = str(getattr(e, 'reason', e))
            self.logger.error(
                f"[RPC] send failed URLError: {reason} "
                f"signal_id={getattr(signal, 'signal_id', 'unknown')} "
                f"action={getattr(signal, 'action_name', 'unknown')} eob={getattr(signal, 'eob', 'unknown')}"
            )
            return {"success": False, "msg": "url_error", "error": reason}

        except Exception as e:
            self.logger.error(f"[RPC] send unexpected error: {type(e).__name__}: {e}")
            return {"success": False, "msg": "unexpected", "error": str(e)}