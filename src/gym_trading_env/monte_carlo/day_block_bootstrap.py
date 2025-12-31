# src/gym_trading_env/monte_carlo/day_block_bootstrap.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from gym_trading_env.utils.timebase import FEATURE_TZ, yyyymmdd_int
from gym_trading_env.utils.session_futures_strict import build_clock_for_trading_day
from gym_trading_env.monte_carlo.session_library import SessionLibrary


@dataclass(frozen=True)
class DayBlockBootstrapConfig:
    warmup_days: int = 7
    eval_days: int = 20
    block_size_days: int = 5
    seed: int = 123

    # price continuity across day stitching
    relink_prices: bool = True
    relink_anchor: str = "first_open"  # 'first_open' or 'first_close'

    # synthetic calendar for session_id/weekday features
    start_date: str = "2015-01-05"     # a Monday; business-day stepping
    use_business_days: bool = True     # Mon-Fri

    tz: str = FEATURE_TZ


@dataclass(frozen=True)
class SynthPath:
    df_raw: pd.DataFrame
    meta: Dict[str, Any]


def _sample_day_indices(
    n_total: int,
    n_hist: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_hist <= 0:
        raise ValueError("n_hist must be > 0")
    if n_total <= 0:
        raise ValueError("n_total must be > 0")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    out = []
    while len(out) < n_total:
        # sample a start so that we can take a consecutive block within history
        if n_hist >= block_size:
            s = int(rng.integers(0, n_hist - block_size + 1))
            blk = list(range(s, s + block_size))
        else:
            # degenerate fallback: history shorter than block_size, just sample with replacement
            blk = [int(rng.integers(0, n_hist)) for _ in range(block_size)]
        out.extend(blk)
    return np.asarray(out[:n_total], dtype=np.int32)


def _make_synth_trading_days(cfg: DayBlockBootstrapConfig, n_days: int) -> pd.DatetimeIndex:
    tz = cfg.tz
    start = pd.Timestamp(cfg.start_date, tz=tz).normalize()

    if cfg.use_business_days:
        # Mon-Fri only (minimal viable)
        days = pd.bdate_range(start=start, periods=n_days, tz=tz)
    else:
        days = pd.date_range(start=start, periods=n_days, freq="D", tz=tz)

    return pd.DatetimeIndex(days)


def generate_synth_path_day_block(
    lib: SessionLibrary,
    cfg: DayBlockBootstrapConfig,
) -> SynthPath:
    """
    Generate a synthetic (warmup+eval) path by day-block bootstrap on complete sessions.

    Output df_raw:
      - index: tz-aware eob timestamps on canonical 345 clock for each synthetic trading_day
      - cols: Open/High/Low/Close/Volume/OpenInterest + trading_day(int)
    """
    if lib is None or lib.n_sessions <= 0:
        raise ValueError("lib is empty")
    if lib.day_len != 345 or lib.n_features != 6:
        raise ValueError(f"unexpected library shape: {lib.sessions.shape} (expect [N,345,6])")

    n_total_days = int(cfg.warmup_days + cfg.eval_days)
    rng = np.random.default_rng(int(cfg.seed))
    pick = _sample_day_indices(n_total_days, lib.n_sessions, int(cfg.block_size_days), rng)

    synth_days = _make_synth_trading_days(cfg, n_total_days)

    frames = []
    chosen_src_sid = []

    prev_last_close: Optional[float] = None
    eps = 1e-12

    for di in range(n_total_days):
        src_i = int(pick[di])
        day_arr = lib.sessions[src_i].copy()  # [345,6]
        chosen_src_sid.append(str(lib.session_id[src_i]))

        O = day_arr[:, 0]
        H = day_arr[:, 1]
        L = day_arr[:, 2]
        C = day_arr[:, 3]

        if cfg.relink_prices and prev_last_close is not None:
            if cfg.relink_anchor == "first_close":
                anchor = float(C[0])
            else:
                anchor = float(O[0])

            anchor = max(anchor, eps)
            scale = float(prev_last_close) / anchor

            # scale OHLC only; keep V/OI unchanged
            day_arr[:, 0:4] = (day_arr[:, 0:4] * scale).astype(np.float32)

        prev_last_close = float(day_arr[-1, 3])

        trading_day = synth_days[di]
        clock = build_clock_for_trading_day(trading_day, tz=cfg.tz, night_date=(trading_day - pd.Timedelta(days=1)))

        df_day = pd.DataFrame(
            day_arr,
            index=clock,
            columns=["Open", "High", "Low", "Close", "Volume", "OpenInterest"],
        )
        df_day["trading_day"] = yyyymmdd_int(trading_day)

        frames.append(df_day)

    df = pd.concat(frames, axis=0)
    # tz safety
    if df.index.tz is None:
        df.index = df.index.tz_localize(cfg.tz)

    meta = {
        "warmup_days": cfg.warmup_days,
        "eval_days": cfg.eval_days,
        "block_size_days": cfg.block_size_days,
        "seed": cfg.seed,
        "relink_prices": cfg.relink_prices,
        "relink_anchor": cfg.relink_anchor,
        "start_date": cfg.start_date,
        "picked_hist_indices": pick,
        "picked_source_session_id": np.array(chosen_src_sid, dtype=object),
        "synthetic_session_id": np.array([str(yyyymmdd_int(d)) for d in synth_days], dtype=object),
    }
    return SynthPath(df_raw=df, meta=meta)
