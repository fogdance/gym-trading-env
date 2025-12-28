# gym_trading_env/utils/juejin_db.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

import pandas as pd
import pymysql

from config.settings import DB_CONFIG  # 你刚贴的 DBConfig/DB_CONFIG


@dataclass
class JuejinDBClient:
    host: str
    port: int
    db: str
    user: str
    password: str

    @classmethod
    def from_settings(cls) -> "JuejinDBClient":
        return cls(
            host=DB_CONFIG.host,
            port=DB_CONFIG.port,
            db=DB_CONFIG.db,
            user=DB_CONFIG.user,
            password=DB_CONFIG.password,
        )

    def _connect(self):
        # 每次短连接，简单粗暴，方便部署
        return pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.db,
            cursorclass=pymysql.cursors.DictCursor,
            charset="utf8mb4",
        )

    def fetch_bars(
        self,
        *,
        underlying: Optional[str],
        symbol: Optional[str],
        start_eob: datetime,
        end_eob: datetime,
        sources: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        读取 [start_eob, end_eob) 的 1m bar，按 eob 升序。
        优先用 symbol 过滤，没传 symbol 就用 underlying。
        """
        if symbol is None and underlying is None:
            raise ValueError("fetch_bars: symbol 和 underlying 至少要给一个")

        conditions = ["eob >= %s", "eob < %s"]
        params: list = [start_eob, end_eob]

        if symbol is not None:
            conditions.append("symbol = %s")
            params.append(symbol)
        if underlying is not None:
            conditions.append("underlying = %s")
            params.append(underlying)
        if sources:
            # source in (...)
            placeholders = ", ".join(["%s"] * len(sources))
            conditions.append(f"source IN ({placeholders})")
            params.extend(list(sources))

        where_sql = " AND ".join(conditions)
        sql = f"""
            SELECT
                trading_date,
                symbol,
                underlying,
                bob,
                eob,
                `open`,
                `high`,
                `low`,
                `close`,
                `volume`,
                `position`,
                `source`,
                `provider`
            FROM fut_bar_1m_v2
            WHERE {where_sql}
            ORDER BY eob ASC
        """

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # 转成 pandas 时间戳，下一步在 BarSource 里再本地化时区
        df["bob"] = pd.to_datetime(df["bob"])
        df["eob"] = pd.to_datetime(df["eob"])
        df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
        return df

    def fetch_latest_eob(
        self,
        *,
        symbol: Optional[str],
        underlying: Optional[str],
    ) -> Optional[datetime]:
        """
        查询当前窗口内最新一根 bar 的 eob，方便轮询 live。
        """
        if symbol is None and underlying is None:
            raise ValueError("fetch_latest_eob: symbol 和 underlying 至少要给一个")

        conditions = []
        params: list = []

        if symbol is not None:
            conditions.append("symbol = %s")
            params.append(symbol)
        if underlying is not None:
            conditions.append("underlying = %s")
            params.append(underlying)

        where_sql = " AND ".join(conditions)
        sql = f"""
            SELECT eob
            FROM fut_bar_1m_v2
            WHERE {where_sql}
            ORDER BY eob DESC
            LIMIT 1
        """

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()

        if not row:
            return None
        return row["eob"]
