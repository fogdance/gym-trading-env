# src/gym_trading_env/utils/bar_source.py
from __future__ import annotations

from typing import Optional, Literal

import pandas as pd

from gym_trading_env.envs.config import TradingConfig  # 保持你原 import 口径
from gym_trading_env.utils.data_processing import load_data
from gym_trading_env.utils.market_features import build_market_features
from gym_trading_env.utils.market_store import MarketStore


BarSourceKind = Literal["csv", "juejin"]


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
            build_daily=bool(getattr(self.config.trading, "use_daily_context", False) or getattr(self.config.trading, "use_daily_seq_7", False)),
        )


class JuejinBarSource(BaseBarSource):
    kind: BarSourceKind = "juejin"

    def _build(self, df: Optional[pd.DataFrame]):
        raise NotImplementedError(
            "JuejinBarSource v1 is not implemented yet. Use CsvBarSource for now."
        )
