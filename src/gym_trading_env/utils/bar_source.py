# src/gym_trading_env/utils/bar_source.py
from __future__ import annotations

from typing import Optional, Literal

from datetime import datetime, time as dt_time, timedelta

import pandas as pd

from gym_trading_env.envs.config import TradingConfig  # 保持你原 import 口径
from gym_trading_env.utils.data_processing import load_data
from gym_trading_env.utils.market_features import build_market_features
from gym_trading_env.utils.market_store import MarketStore
from gym_trading_env.utils.session_futures_strict import DEFAULT_TZ

from gym_trading_env.config.settings import DB_CONFIG

try:
    import pymysql
except ImportError as e:  # pragma: no cover
    # 不在 import 时就 crash，只有真正用 JuejinBarSource 时才会报错
    pymysql = None


BarSourceKind = Literal["csv", "juejin"]

# 掘金期货 1m bar 表名
JUEJIN_FUT_BAR_TABLE = "fut_bar_1m_v2"


def _normalize_ohlcvi_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw OHLCVI dataframe:
    - Ensure DatetimeIndex
    - Sort by time
    - Drop duplicated timestamps (keep last)
    """
    if df is None:
        raise ValueError("df is None")

    df = df.copy()

    # Accept either 'Date' column or DatetimeIndex
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a 'Date' column or a DatetimeIndex.")

    df = df[~df.index.isna()]
    df = df.sort_index()

    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    return df


class BaseBarSource:
    """
    v2: env reads ONLY from self.store (numpy stable arrays).
    df_raw/df_market remain for debug/visualization.

    Must provide:
      - df_raw
      - df_market
      - store : MarketStore
    """

    kind: BarSourceKind = "csv"

    def __init__(self, config: TradingConfig, df: Optional[pd.DataFrame] = None):
        self.config = config
        self.df_raw: pd.DataFrame = pd.DataFrame()
        self.df_market: pd.DataFrame = pd.DataFrame()
        self.store: Optional[MarketStore] = None

        self._build(df=df)
        self._validate_basic()

    def _build(self, df: Optional[pd.DataFrame]):
        raise NotImplementedError

    def _validate_basic(self):
        if self.store is None:
            raise RuntimeError("BarSource.store is None (build failed)")

        ws = int(getattr(self.config.training, "window_size", 1))
        if ws <= 0:
            raise ValueError(f"window_size must be > 0, got {ws}")

        # futures: obs is day-based & right-padded to window_size
        if bool(getattr(self.config.trading, "is_future", False)):
            if ws > int(self.store.day_len):
                raise ValueError(f"window_size={ws} > day_len={self.store.day_len} (futures). Clamp in env or config.")

        # Basic availability: must have at least 1 day
        if len(self.store.days) < 1:
            raise ValueError("store has no days")

        # Required columns sanity (fail fast)
        required = {"day_id", "minute_index_t", "mask_t", "C_t"}
        if bool(getattr(self.config.trading, "stop_loss_enabled", False)) or bool(
            getattr(self.config.trading, "take_profit_enabled", False)
        ):
            required |= {"H_t", "L_t"}

        missing = required - set(self.df_market.columns)
        if missing:
            raise ValueError(f"df_market missing required columns: {missing}")


class CsvBarSource(BaseBarSource):
    kind: BarSourceKind = "csv"

    def _build(self, df: Optional[pd.DataFrame]):
        # 1) Load raw OHLCVI
        if df is None:
            sym = self.config.trading.data_path
            interval = self.config.trading.data_interval
            df = load_data(sym, interval)

        self.df_raw = _normalize_ohlcvi_df(df)

        # 2) Feature engineering
        self.df_market = build_market_features(
            self.df_raw,
            rollover_hour_local=5,
            is_future=bool(getattr(self.config.trading, "is_future", False)),
        )

        # 3) Build stable store (numpy arrays)
        self.store = MarketStore.from_frames(
            df_raw=self.df_raw,
            df_market=self.df_market,
            is_future=bool(getattr(self.config.trading, "is_future", False)),
            tz=DEFAULT_TZ,
            build_daily=bool(
                getattr(self.config.trading, "use_daily_context", False)
                or getattr(self.config.trading, "use_daily_seq_7", False)
            ),
        )


class JuejinBarSource(BaseBarSource):
    """
    期货实盘/回测专用 BarSource，从 MySQL 表 market_data.fut_bar_1m_v2 读取数据。

    约定：
      - 只支持期货（config.trading.is_future == True）
      - symbol 默认从 config.trading.symbol 或 config.trading.currency_pair 读取
      - 时间窗口：
          t1 = trading_date 15:01（当地时间，无时区）
          t0 = t1 - 14 天
        其中 trading_date：
          - 优先使用 config.trading.trading_date（可为 str/date/datetime）
          - 否则用当前本地日期（服务器时间）
      - 实盘时你可以在开盘前构造一次 env / BarSource，这个窗口固定，
        数据全部从 DB 读（HISTORY/EOD/LIVE 都写在一张表里），
        后续如需增量刷新，可调用 refresh_from_db()（下面也给出实现）
    """

    kind: BarSourceKind = "juejin"

    def _build(self, df: Optional[pd.DataFrame]):
        # 只支持 futures
        is_future = bool(getattr(self.config.trading, "is_future", False))
        if not is_future:
            raise ValueError("JuejinBarSource currently only supports futures (config.trading.is_future must be True)")

        # 测试/调试: 若外部直接给 df，则走跟 CsvBarSource 类似的流程
        if df is not None:
            self.df_raw = _normalize_ohlcvi_df(df)
        else:
            self.df_raw = self._load_from_db_initial_window()

        # 2) 特征工程 -> df_market
        #   使用 futures 严格 345 对齐逻辑（_build_market_future）
        limit_up_pct = getattr(self.config.trading, "limit_up_pct", None)
        limit_down_pct = getattr(self.config.trading, "limit_down_pct", None)

        self.df_market = build_market_features(
            self.df_raw,
            tz=DEFAULT_TZ,
            is_future=is_future,
            limit_up_pct=limit_up_pct,
            limit_down_pct=limit_down_pct,
        )

        # 3) 构建稳定 store
        self.store = MarketStore.from_frames(
            df_raw=self.df_raw,
            df_market=self.df_market,
            is_future=is_future,
            tz=DEFAULT_TZ,
            build_daily=bool(
                getattr(self.config.trading, "use_daily_context", False)
                or getattr(self.config.trading, "use_daily_seq_7", False)
            ),
        )

    # ------------------------------------------------------------------ #
    #   对外刷新入口（实盘可选用）
    # ------------------------------------------------------------------ #
    def refresh_from_db(self) -> None:
        """
        从 DB 重新拉取 [t0, t1] 窗口内所有 bar，并重建 df_raw/df_market/store。

        说明：
          - 窗口定义与 _load_from_db_initial_window 相同（同一个 trading_date、同一 t0/t1）
          - 用 MarketStore.rebuild 保持 tz/day_key_fn 等配置一致
          - 这是一个同步刷新，调用者必须自己控制调用频率（例如收盘后 / 每隔 N 分钟）
        """
        if self.store is None:
            # 第一次构建时 _build 已经处理，不在这里兜底
            raise RuntimeError("refresh_from_db called before initial build")

        is_future = bool(getattr(self.config.trading, "is_future", False))
        if not is_future:
            raise ValueError("JuejinBarSource.refresh_from_db only supports futures")

        new_df_raw = self._load_from_db_initial_window()

        limit_up_pct = getattr(self.config.trading, "limit_up_pct", None)
        limit_down_pct = getattr(self.config.trading, "limit_down_pct", None)

        new_df_market = build_market_features(
            new_df_raw,
            tz=DEFAULT_TZ,
            is_future=is_future,
            limit_up_pct=limit_up_pct,
            limit_down_pct=limit_down_pct,
        )

        # 基于旧 store 的 tz / day_key_fn / daily 配置重建
        self.df_raw = new_df_raw
        self.df_market = new_df_market
        self.store = MarketStore.rebuild(
            prev_store=self.store,
            df_raw=self.df_raw,
            df_market=self.df_market,
            is_future=is_future,
        )

    # ------------------------------------------------------------------ #
    #   内部：从 fut_bar_1m_v2 读取 [t0, t1] 窗口的原始 OHLCVI
    # ------------------------------------------------------------------ #
    def _load_from_db_initial_window(self) -> pd.DataFrame:
        """
        从 MySQL 表 market_data.fut_bar_1m_v2 读取原始 1m bar：
          - 窗口: [t0, t1]
            * t1 = trading_date 15:01
            * t0 = t1 - 14 天
          - 仅按 future_symbol 过滤（一个 env = 一个合约）
          - 返回格式：DatetimeIndex = eob，列至少包含:
                Open, High, Low, Close, Volume, OpenInterest
            其余字段（trading_date/future_symbol/...）保留作为附加列以便 debug
        """
        if pymysql is None:  # pragma: no cover
            raise ImportError(
                "pymysql is required for JuejinBarSource but is not installed. "
                "Please `pip install pymysql`."
            )

        symbol = getattr(self.config.trading, "future_symbol", None) or getattr(
            self.config.trading, "currency_pair", None
        )
        if not symbol:
            raise ValueError(
                "JuejinBarSource: config.trading must define 'future_symbol' or 'currency_pair' "
                "to be used as fut_bar_1m_v2.symbol"
            )

        # ---- 决定 trading_date ----
        trading_date_attr = getattr(self.config.trading, "trading_date", None)
        if trading_date_attr is not None:
            # 接受 str / date / datetime，统一转为 date
            td = pd.to_datetime(trading_date_attr).date()
        else:
            # 没指定则用当前本地日期（DEFAULT_TZ 下）
            now_local = pd.Timestamp.now(tz=DEFAULT_TZ)
            td = now_local.date()

        # t1 = 当天 15:01（无时区，按照本地时间直接存入 DATETIME）
        t1 = datetime.combine(td, dt_time(hour=15, minute=1, second=0))
        t0 = t1 - timedelta(days=14)

        # ---- 连接 MySQL ----
        conn = pymysql.connect(
            host=DB_CONFIG.host,
            port=DB_CONFIG.port,
            user=DB_CONFIG.user,
            password=DB_CONFIG.password,
            database=DB_CONFIG.db,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )

        try:
            with conn.cursor() as cursor:
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
                        volume,
                        `position`,
                        `source`,
                        provider,
                        created_at,
                        updated_at
                    FROM {JUEJIN_FUT_BAR_TABLE}
                    WHERE symbol = %s
                      AND eob >= %s
                      AND eob <= %s
                    ORDER BY eob ASC
                """
                cursor.execute(sql, (symbol, t0, t1))
                rows = cursor.fetchall()
        finally:
            conn.close()

        if not rows:
            raise ValueError(
                f"No rows found in {JUEJIN_FUT_BAR_TABLE} for symbol={symbol} "
                f"between eob>={t0!s} and eob<={t1!s}"
            )

        df = pd.DataFrame(rows)

        # eob 作为时间索引（你 futures 特征全部基于 eob 做 1m 档）
        df["eob"] = pd.to_datetime(df["eob"], errors="coerce")
        df = df[~df["eob"].isna()].copy()
        df = df.sort_values("eob")
        df = df.set_index("eob")

        # 构造 OHLCVI 视图，列名对齐 build_market_features 要求
        df_ohlc = pd.DataFrame(index=df.index)

        # 保留原始信息便于 debug/复盘
        for col in [
            "trading_date",
            "symbol",
            "underlying",
            "bob",
            "source",
            "provider",
            "created_at",
            "updated_at",
        ]:
            if col in df.columns:
                df_ohlc[col] = df[col]

        # 标准 OHLCVI
        # 注意 decimal(16,4) -> float；volume/position -> int -> float
        df_ohlc["Open"] = pd.to_numeric(df["open"], errors="coerce").astype(float)
        df_ohlc["High"] = pd.to_numeric(df["high"], errors="coerce").astype(float)
        df_ohlc["Low"] = pd.to_numeric(df["low"], errors="coerce").astype(float)
        df_ohlc["Close"] = pd.to_numeric(df["close"], errors="coerce").astype(float)
        df_ohlc["Volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(float)
        # position = OpenInterest
        df_ohlc["OpenInterest"] = pd.to_numeric(df["position"], errors="coerce").fillna(0).astype(float)

        # 基本清洗：去除完全 NaN 的行
        need_cols = ["Open", "High", "Low", "Close"]
        mask_valid_ohlc = df_ohlc[need_cols].notna().all(axis=1)
        if not mask_valid_ohlc.any():
            raise ValueError(
                f"All OHLC rows from {JUEJIN_FUT_BAR_TABLE} are NaN for symbol={symbol} between {t0} and {t1}"
            )
        df_ohlc = df_ohlc[mask_valid_ohlc].copy()

        # 最终再跑一遍通用规范化（保证时间索引/去重逻辑一致）
        df_ohlc = _normalize_ohlcvi_df(df_ohlc)

        return df_ohlc
