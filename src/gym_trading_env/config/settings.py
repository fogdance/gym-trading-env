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
