# src/gym_trading_env/config/settings.py
from dataclasses import dataclass
import os


@dataclass
class DBConfig:
    host: str
    port: int
    db: str
    user: str
    password: str


# 从环境变量加载，方便在不同环境部署
DB_CONFIG = DBConfig(
    host=os.getenv("MD_MYSQL_HOST", "10.0.0.9"),
    port=int(os.getenv("MD_MYSQL_PORT", "3306")),
    db=os.getenv("MD_MYSQL_DB", "market_data"),
    user=os.getenv("MD_MYSQL_USER", "root"),
    password=os.getenv("MD_MYSQL_PASSWORD", "root"),
)


@dataclass
class RpcOrder:
    order_enabled: bool
    order_endpoint: str
    order_token: str
    order_timeout_sec: float
    order_volume: int

# 从环境变量读取配置（不存在则使用默认值）
RPC_ORDER_CONFIG = RpcOrder(
    order_enabled=os.getenv("RPC_ORDER_ENABLED", "true").lower() == "true",
    order_endpoint=os.getenv("RPC_ORDER_ENDPOINT", "http://10.0.0.33:9001"),
    order_token=os.getenv("RPC_ORDER_TOKEN", "devtoken"),
    order_timeout_sec=float(os.getenv("RPC_ORDER_TIMEOUT_SEC", "10")),
    order_volume=int(os.getenv("RPC_ORDER_VOLUME", "1")),
)