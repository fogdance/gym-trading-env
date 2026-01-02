# tests/integration/market_data_factory.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd


FEATURE_TZ = "Asia/Shanghai"

# 固定 futures strict_345 四段时钟（按分钟收盘 eob）
SEGMENTS: List[Tuple[str, str]] = [
    ("21:00", "23:00"),
    ("09:00", "10:15"),
    ("10:30", "11:30"),
    ("13:30", "15:00"),
]


def _build_clock_for_trading_day(trading_day: pd.Timestamp, tz: str) -> pd.DatetimeIndex:
    """与 strict_reindex_futures_345 同口径：每段分钟 +1min，得到 eob。"""
    td = pd.Timestamp(trading_day)
    if td.tz is None:
        td = td.tz_localize(tz)
    else:
        td = td.tz_convert(tz)
    trading_day = td.normalize()
    night_date = (trading_day - pd.Timedelta(days=1)).normalize()

    parts = []
    for i, (s_str, e_str) in enumerate(SEGMENTS):
        base = night_date if i == 0 else trading_day
        s = pd.Timestamp(f"{base.date()} {s_str}", tz=tz)
        e = pd.Timestamp(f"{base.date()} {e_str}", tz=tz)
        rng = pd.date_range(start=s, end=e, freq="1min", inclusive="left", tz=tz) + pd.Timedelta(minutes=1)
        parts.append(rng)

    clock = parts[0]
    for p in parts[1:]:
        clock = clock.append(p)

    assert len(clock) == 345
    return clock


@dataclass(frozen=True)
class MarketDataBundle:
    df_1m: pd.DataFrame
    df_prev_session: Optional[pd.DataFrame]
    symbol: str
    interval: str
    tz: str = FEATURE_TZ
    is_future: bool = True

    def write_csv_for_env(self, root: Path) -> Path:
        """
        写到 env 的默认加载路径：data/{symbol}_{interval}.csv
        注意：CsvBarSource 会用 normalize_ohlcvi(date_col="Date")，所以必须有 Date 列或 DatetimeIndex。
        这里写 Date 列（字符串），并保留 trading_day 以稳定 futures session_id。
        """
        data_dir = root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        out = data_dir / f"{self.symbol}_{self.interval}.csv"

        df = self.df_1m.copy()
        df = df.reset_index().rename(columns={"index": "Date"})
        df["Date"] = df["Date"].dt.tz_convert(self.tz).dt.strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(out, index=False)
        return out


class MarketDataFactory:
    """
    deterministic futures 1m “正常数据”：
    - 覆盖多交易日（>=2）保证 ref_close / dI_from_yclose / strict_345 都能对齐
    - Volume/OI 非 0
    - 平滑趋势 + 小波动
    """

    @staticmethod
    def make_futures_bundle(
        *,
        start_trading_day: str = "2025-01-06",  # 周一
        num_days: int = 3,
        seed: int = 7,
        symbol: str = "TEST",
        interval: str = "1m",
        tz: str = FEATURE_TZ,
    ) -> MarketDataBundle:
        rng = np.random.default_rng(seed)

        start = pd.Timestamp(start_trading_day, tz=tz).normalize()
        days = [start + pd.Timedelta(days=i) for i in range(num_days)]

        frames = []
        for di, day in enumerate(days):
            clock = _build_clock_for_trading_day(day, tz)

            n = len(clock)
            t = np.arange(n, dtype=float)

            # 平滑趋势 + 周期波动 + 小噪声（deterministic）
            base = 1000.0 + 5.0 * di
            drift = 0.02 * t
            wave = 2.5 * np.sin(2 * np.pi * t / 144.0) + 1.2 * np.sin(2 * np.pi * t / 37.0)
            noise = rng.normal(0.0, 0.05, size=n)

            close = base + drift + wave + noise
            open_ = close + rng.normal(0.0, 0.03, size=n)
            high = np.maximum(open_, close) + 0.12
            low = np.minimum(open_, close) - 0.12

            volume = 200.0 + 20.0 * np.sin(2 * np.pi * t / 50.0) + 5.0 * rng.normal(0.0, 1.0, size=n)
            volume = np.clip(volume, 1.0, None)

            oi = 12000.0 + 50.0 * di + 30.0 * np.cos(2 * np.pi * t / 80.0)
            oi = np.clip(oi, 10.0, None)

            df_day = pd.DataFrame(
                {
                    "Open": open_.astype(float),
                    "High": high.astype(float),
                    "Low": low.astype(float),
                    "Close": close.astype(float),
                    "Volume": volume.astype(float),
                    "OpenInterest": oi.astype(float),
                    # 强制 session_id 稳定：strict_reindex_futures_345 会优先用 trading_day
                    "trading_day": int(day.strftime("%Y%m%d")),
                },
                index=clock,
            )
            frames.append(df_day)

        df_all = pd.concat(frames).sort_index()
        df_all.index = df_all.index.tz_convert(tz)

        # prev_session：取第一天的完整 345，作为 df_prev_session（用于 unit 覆盖“第一天用 prev_session 兜底”）
        first_day = days[0]
        prev_day = first_day - pd.Timedelta(days=1)
        prev_clock = _build_clock_for_trading_day(prev_day, tz)
        # 造一份“前一交易日完整数据”，价格基准略低，保证 ref_close/dI 不为 0
        n = len(prev_clock)
        t = np.arange(n, dtype=float)
        base = 995.0
        close = base + 0.02 * t + 2.0 * np.sin(2 * np.pi * t / 144.0)
        open_ = close
        high = close + 0.12
        low = close - 0.12
        volume = np.full(n, 180.0, dtype=float)
        oi = np.full(n, 11800.0, dtype=float)

        df_prev = pd.DataFrame(
            {
                "Open": open_.astype(float),
                "High": high.astype(float),
                "Low": low.astype(float),
                "Close": close.astype(float),
                "Volume": volume.astype(float),
                "OpenInterest": oi.astype(float),
                "trading_day": int(prev_day.strftime("%Y%m%d")),
            },
            index=prev_clock,
        )

        return MarketDataBundle(
            df_1m=df_all,
            df_prev_session=df_prev,
            symbol=symbol,
            interval=interval,
            tz=tz,
            is_future=True,
        )
