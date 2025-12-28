# src/gym_trading_env/utils/bar_source.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import pandas as pd

from gym_trading_env.envs.config import TradingConfig
from gym_trading_env.utils.data_processing import load_data
from gym_trading_env.utils.market_features import build_market_features


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

    # Basic cleanup
    df = df[~df.index.isna()]
    df = df.sort_index()

    # If duplicates exist, keep last (useful for live feeds overwriting the last bar)
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    return df


class BaseBarSource:
    """
    v1: does NOT hide df/df_market; env can still access them directly.

    Must provide:
      - df_raw: normalized OHLCVI dataframe
      - df_market: feature-engineered dataframe (build_market_features output)
    """

    kind: BarSourceKind = "csv"

    def __init__(self, config: TradingConfig, df: Optional[pd.DataFrame] = None):
        self.config = config
        self.df_raw: pd.DataFrame = pd.DataFrame()
        self.df_market: pd.DataFrame = pd.DataFrame()
        self._build(df=df)
        self._validate_basic()

    def _build(self, df: Optional[pd.DataFrame]):
        raise NotImplementedError

    def _validate_basic(self):
        # --- Basic length feasibility for Dreamer-style windowing ---
        df_len = len(self.df_market)
        ws = int(getattr(self.config.training, "window_size", 1))
        if df_len < ws:
            raise ValueError(
                f"df_market has only {df_len} rows, smaller than window_size={ws}. Not feasible."
            )

        ep_len = getattr(self.config.training, "episode_length", None)
        if ep_len is not None:
            ep_len = int(ep_len)
            if ep_len <= 0:
                raise ValueError(f"episode_length must be > 0, got {ep_len}")
            # 至少能跑 1 step（env.reset 的 start_row 逻辑还有更细的候选行过滤，这里只做硬下限）
            if df_len - ep_len < 1:
                raise ValueError(
                    f"df_market length={df_len} is insufficient for episode_length={ep_len}."
                )

        # --- Required columns sanity (fail fast) ---
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

        # 2) Feature engineering ONCE
        self.df_market = build_market_features(
            self.df_raw,
            rollover_hour_local=5,
            is_future=bool(getattr(self.config.trading, "is_future", False)),
        )


class JuejinBarSource(BaseBarSource):
    kind: BarSourceKind = "juejin"

    def _build(self, df: Optional[pd.DataFrame]):
        # v1 先占位：后续实现 ingest/append-only 逻辑
        raise NotImplementedError(
            "JuejinBarSource v1 is not implemented yet. Use CsvBarSource for now."
        )
