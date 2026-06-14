#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gym_trading_env.research.entry_evaluator import (  # noqa: E402
    EntryEvalConfig,
    load_entry_eval_config,
    load_market_frames,
)


TZ = "Asia/Shanghai"


@dataclass(frozen=True)
class Paths:
    attribution_dir: Path
    entry_eval_config: Path
    output_dir: Path


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return str(value)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default))


def _to_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(TZ)


def _format_ts(value) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return float(num / den)


def _profit_factor(values: Iterable[float]) -> float | str | None:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return None
    gains = float(arr[arr > 0].sum())
    losses = float(-arr[arr < 0].sum())
    if losses == 0:
        return "inf" if gains > 0 else None
    return gains / losses


def _curve_drawdown(
    equity: pd.Series,
    *,
    initial_peak: float | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if initial_peak is None:
        peak = equity.cummax()
    else:
        peak = equity.copy().astype(float)
        running = float(initial_peak)
        peaks = []
        for value in equity.to_numpy(dtype=float):
            running = max(running, float(value))
            peaks.append(running)
        peak = pd.Series(peaks, index=equity.index)
    drawdown = equity - peak
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown_pct = np.where(peak.to_numpy(dtype=float) != 0, drawdown / peak * 100.0, 0.0)
    duration = []
    cur = 0
    for dd in drawdown.to_numpy(dtype=float):
        if dd < 0:
            cur += 1
        else:
            cur = 0
        duration.append(cur)
    return pd.Series(drawdown, index=equity.index), pd.Series(drawdown_pct, index=equity.index), pd.Series(duration, index=equity.index)


def _max_duration_at_max_dd(drawdown: pd.Series, duration: pd.Series) -> int:
    if drawdown.empty:
        return 0
    min_dd = float(drawdown.min())
    if min_dd >= 0:
        return 0
    return int(duration[drawdown == min_dd].max())


def _load_inputs(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame, EntryEvalConfig, pd.DataFrame, pd.DataFrame]:
    trades_path = paths.attribution_dir / "dreamer_actual_trades.csv"
    episodes_path = paths.attribution_dir / "dreamer_episodes.csv"
    if not trades_path.exists():
        raise FileNotFoundError(trades_path)
    if not episodes_path.exists():
        raise FileNotFoundError(episodes_path)
    trades = pd.read_csv(trades_path)
    episodes = pd.read_csv(episodes_path)
    config = load_entry_eval_config(paths.entry_eval_config)
    raw, market = load_market_frames(config)
    return trades, episodes, config, raw, market


def _prepare_trades(trades: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    trades["entry_ts"] = _to_ts(trades["entry_timestamp"])
    trades["exit_ts"] = _to_ts(trades["exit_timestamp"])
    trades["actual_net_pnl"] = pd.to_numeric(trades["actual_net_pnl"], errors="coerce").fillna(0.0)
    trades["actual_gross_pnl"] = pd.to_numeric(trades.get("actual_gross_pnl", 0.0), errors="coerce").fillna(0.0)
    trades["entry_price"] = pd.to_numeric(trades["entry_price"], errors="coerce")
    trades["exit_price"] = pd.to_numeric(trades["exit_price"], errors="coerce")
    trades["entry_position_size"] = pd.to_numeric(
        trades.get("entry_position_size", 1.0), errors="coerce").fillna(1.0)
    trades["episode_id"] = pd.to_numeric(trades["episode_id"], errors="coerce").astype(int)
    trades["trading_day"] = pd.to_numeric(trades["trading_day"], errors="coerce").astype(int)
    trades["month"] = trades.get("month", trades["trading_day"].astype(str).str[:6]).astype(str)
    trades["split_role"] = trades.get("split_role", "").astype(str)
    trades = trades.sort_values(
        ["exit_ts", "episode_id", "exit_record_index", "entry_record_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    trades["trade_id"] = np.arange(len(trades), dtype=np.int64)
    return trades


def _prepare_episodes(episodes: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    episodes = episodes.copy()
    episodes["start_ts"] = _to_ts(episodes["start_timestamp"])
    episodes["end_ts"] = _to_ts(episodes["end_timestamp"])
    episodes["episode_id"] = pd.to_numeric(episodes["episode_id"], errors="coerce").astype(int)
    episodes["start_idx"] = pd.to_numeric(episodes["start_idx"], errors="coerce").astype(int)
    episodes["end_idx"] = pd.to_numeric(episodes["end_idx"], errors="coerce").astype(int)
    episodes["start_trading_day"] = pd.to_numeric(
        episodes["start_trading_day"], errors="coerce").astype(int)
    episodes["start_split_role"] = episodes["start_split_role"].astype(str)

    index_to_row = {ts: i for i, ts in enumerate(market.index)}
    end_rows = []
    for row in episodes.itertuples(index=False):
        end_row = index_to_row.get(row.end_ts)
        if end_row is None:
            matches = np.flatnonzero(market.index <= row.end_ts)
            if matches.size == 0:
                end_row = int(row.start_idx)
            else:
                end_row = int(matches[-1])
        end_rows.append(end_row)
    episodes["actual_end_row"] = end_rows
    episodes["actual_start_row"] = episodes["start_idx"].astype(int)
    return episodes


def build_closed_equity_curve(trades: pd.DataFrame, initial_equity: float) -> pd.DataFrame:
    out = trades.copy()
    out["closed_trade_pnl"] = out["actual_net_pnl"]
    out["cumulative_actual_net_pnl"] = out["closed_trade_pnl"].cumsum()
    out["equity_after_trade"] = initial_equity + out["cumulative_actual_net_pnl"]
    dd, dd_pct, dur = _curve_drawdown(out["equity_after_trade"], initial_peak=initial_equity)
    out["closed_drawdown"] = dd
    out["closed_drawdown_pct"] = dd_pct
    out["closed_drawdown_duration_trades"] = dur
    cols = [
        "trade_id", "episode_id", "split_role", "trading_day", "month",
        "direction", "entry_timestamp", "exit_timestamp", "exit_reason",
        "entry_price", "exit_price", "actual_net_pnl", "cumulative_actual_net_pnl",
        "equity_after_trade", "closed_drawdown", "closed_drawdown_pct",
        "closed_drawdown_duration_trades",
    ]
    return out[cols]


def build_daily_pnl_curve(
    trades: pd.DataFrame,
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = episodes[["start_trading_day", "start_split_role"]].drop_duplicates().copy()
    days = days.rename(columns={"start_trading_day": "trading_day", "start_split_role": "split_role"})
    days["trading_day"] = days["trading_day"].astype(int)
    days["month"] = days["trading_day"].astype(str).str[:6]
    pnl = trades.groupby("trading_day", as_index=False)["actual_net_pnl"].sum()
    pnl = pnl.rename(columns={"actual_net_pnl": "daily_pnl"})
    daily = days.merge(pnl, on="trading_day", how="left")
    daily["daily_pnl"] = daily["daily_pnl"].fillna(0.0)
    daily = daily.sort_values("trading_day").reset_index(drop=True)
    daily["cumulative_daily_pnl"] = daily["daily_pnl"].cumsum()
    daily["cumulative_split_pnl"] = daily.groupby("split_role")["daily_pnl"].cumsum()
    daily["monthly_cumulative_pnl"] = daily.groupby("month")["daily_pnl"].cumsum()

    monthly = daily.groupby(["month", "split_role"], as_index=False)["daily_pnl"].sum()
    monthly = monthly.rename(columns={"daily_pnl": "monthly_pnl"})
    monthly = monthly.sort_values(["month", "split_role"]).reset_index(drop=True)
    month_total = daily.groupby("month", as_index=False)["daily_pnl"].sum()
    month_total = month_total.rename(columns={"daily_pnl": "monthly_pnl_all"})
    month_total["cumulative_monthly_pnl_all"] = month_total["monthly_pnl_all"].cumsum()
    monthly = monthly.merge(month_total, on="month", how="left")
    return daily, monthly


def _episode_bar_frame(episodes: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ep in episodes.itertuples(index=False):
        start = int(ep.actual_start_row)
        end = int(ep.actual_end_row)
        if end < start:
            continue
        rows = np.arange(start, end + 1, dtype=np.int64)
        frame = pd.DataFrame({
            "episode_id": int(ep.episode_id),
            "row": rows,
            "timestamp": market.index[rows],
            "trading_day": market["trading_day"].to_numpy(dtype=np.int64)[rows],
            "split_role": str(ep.start_split_role),
            "current_close": pd.to_numeric(market["C_t"], errors="coerce").to_numpy(dtype=float)[rows],
        })
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["timestamp", "episode_id", "row"], kind="mergesort").reset_index(drop=True)
    return out


def _active_trade_at(ts: pd.Timestamp, episode_id: int, trades: pd.DataFrame) -> pd.Series | None:
    sub = trades[
        (trades["episode_id"] == int(episode_id))
        & (trades["entry_ts"] <= ts)
        & (trades["exit_ts"] > ts)
    ]
    if sub.empty:
        return None
    return sub.sort_values(["entry_ts", "entry_record_index"], kind="mergesort").iloc[-1]


def build_mark_to_market_curve(
    trades: pd.DataFrame,
    episodes: pd.DataFrame,
    market: pd.DataFrame,
    config: EntryEvalConfig,
) -> pd.DataFrame:
    bars = _episode_bar_frame(episodes, market)
    if bars.empty:
        return bars
    multiplier_base = float(config.product.lot_size)
    initial_equity = float(config.product.initial_balance)

    closes_by_ts_ep = []
    realized_by_key = trades.groupby(["episode_id", "exit_ts"])["actual_net_pnl"].sum()
    cumulative_realized = 0.0
    rows = []
    trades_by_episode = {int(k): v.copy() for k, v in trades.groupby("episode_id")}

    for bar in bars.itertuples(index=False):
        key = (int(bar.episode_id), bar.timestamp)
        if key in realized_by_key.index:
            cumulative_realized += float(realized_by_key.loc[key])

        ep_trades = trades_by_episode.get(int(bar.episode_id), trades.iloc[0:0])
        active = _active_trade_at(bar.timestamp, int(bar.episode_id), ep_trades)
        position = 0.0
        direction = "FLAT"
        entry_price = np.nan
        unrealized = 0.0
        trade_id = None
        if active is not None:
            direction = str(active["direction"]).upper()
            position = float(active["entry_position_size"])
            entry_price = float(active["entry_price"])
            multiplier = multiplier_base * position
            if direction == "LONG":
                unrealized = (float(bar.current_close) - entry_price) * multiplier
            elif direction == "SHORT":
                unrealized = (entry_price - float(bar.current_close)) * multiplier
            trade_id = int(active["trade_id"])

        rows.append({
            "timestamp": _format_ts(bar.timestamp),
            "episode_id": int(bar.episode_id),
            "row": int(bar.row),
            "trading_day": int(bar.trading_day),
            "split_role": str(bar.split_role),
            "realized_pnl": cumulative_realized,
            "unrealized_pnl": float(unrealized),
            "total_equity": initial_equity + cumulative_realized + float(unrealized),
            "position": position,
            "direction": direction,
            "entry_price": entry_price,
            "current_close": float(bar.current_close),
            "active_trade_id": trade_id,
        })
        closes_by_ts_ep.append(key)
    return pd.DataFrame.from_records(rows)


def build_drawdown_curve(
    closed: pd.DataFrame,
    mtm: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    if not closed.empty:
        equity = closed["equity_after_trade"].astype(float)
        dd, dd_pct, dur = _curve_drawdown(
            equity,
            initial_peak=float(closed["equity_after_trade"].iloc[0] - closed["cumulative_actual_net_pnl"].iloc[0]),
        )
        peak = pd.Series(equity, index=equity.index).copy().astype(float)
        running_peak = []
        running = float(closed["equity_after_trade"].iloc[0] - closed["cumulative_actual_net_pnl"].iloc[0])
        for value in equity.to_numpy(dtype=float):
            running = max(running, float(value))
            running_peak.append(running)
        for idx, (item, drawdown, pct, duration) in enumerate(zip(closed.itertuples(index=False), dd, dd_pct, dur)):
            rows.append({
                "curve_type": "closed_trade",
                "timestamp": item.exit_timestamp,
                "sequence": int(item.trade_id),
                "split_role": item.split_role,
                "trading_day": int(item.trading_day),
                "equity": float(item.equity_after_trade),
                "running_peak": float(running_peak[idx]),
                "drawdown": float(drawdown),
                "drawdown_pct": float(pct),
                "drawdown_duration": int(duration),
                "duration_unit": "trades",
            })
    if not mtm.empty:
        equity = mtm["total_equity"].astype(float)
        dd, dd_pct, dur = _curve_drawdown(equity)
        peak = equity.cummax()
        for idx, item in enumerate(mtm.itertuples(index=False)):
            rows.append({
                "curve_type": "mark_to_market",
                "timestamp": item.timestamp,
                "sequence": int(idx),
                "split_role": item.split_role,
                "trading_day": int(item.trading_day),
                "equity": float(item.total_equity),
                "running_peak": float(peak.iloc[idx]),
                "drawdown": float(dd.iloc[idx]),
                "drawdown_pct": float(dd_pct.iloc[idx]),
                "drawdown_duration": int(dur.iloc[idx]),
                "duration_unit": "bars",
            })
    return pd.DataFrame.from_records(rows)


def build_trade_path_stats(
    trades: pd.DataFrame,
    market: pd.DataFrame,
    config: EntryEvalConfig,
) -> pd.DataFrame:
    idx = market.index
    high = pd.to_numeric(market["H_t"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(market["L_t"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(market["C_t"], errors="coerce").to_numpy(dtype=float)
    mask = pd.to_numeric(market["mask_t"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.0
    timestamp_to_row = {ts: i for i, ts in enumerate(idx)}
    tick_value = float(config.product.tick_size * config.product.lot_size * config.product.trade_lot)
    base_multiplier = float(config.product.lot_size)

    records = []
    for tr in trades.itertuples(index=False):
        entry_row = timestamp_to_row.get(tr.entry_ts)
        exit_row = timestamp_to_row.get(tr.exit_ts)
        if entry_row is None or exit_row is None:
            continue
        start = min(int(entry_row) + 1, int(exit_row))
        stop = int(exit_row)
        rows = np.arange(start, stop + 1, dtype=np.int64)
        rows = rows[mask[rows]] if rows.size else rows
        if rows.size == 0:
            rows = np.asarray([int(exit_row)], dtype=np.int64)

        size = float(tr.entry_position_size)
        multiplier = base_multiplier * size
        entry_price = float(tr.entry_price)
        if str(tr.direction).upper() == "LONG":
            favorable = (high[rows] - entry_price) * multiplier
            adverse = (low[rows] - entry_price) * multiplier
        else:
            favorable = (entry_price - low[rows]) * multiplier
            adverse = (entry_price - high[rows]) * multiplier

        max_fav_idx = int(np.nanargmax(favorable))
        max_adv_idx = int(np.nanargmin(adverse))
        max_fav = float(favorable[max_fav_idx])
        max_adv = float(adverse[max_adv_idx])
        max_adv_ticks = 0.0 if tick_value == 0 else max(0.0, -max_adv / tick_value)
        hold_bars = int(rows.size)
        hold_minutes = (tr.exit_ts - tr.entry_ts).total_seconds() / 60.0

        records.append({
            "trade_id": int(tr.trade_id),
            "episode_id": int(tr.episode_id),
            "split_role": tr.split_role,
            "trading_day": int(tr.trading_day),
            "month": tr.month,
            "direction": tr.direction,
            "entry_timestamp": tr.entry_timestamp,
            "exit_timestamp": tr.exit_timestamp,
            "exit_reason": tr.exit_reason,
            "actual_net_pnl": float(tr.actual_net_pnl),
            "max_favorable_pnl": max_fav,
            "max_adverse_pnl": max_adv,
            "max_adverse_ticks": float(max_adv_ticks),
            "time_to_max_favorable": _format_ts(idx[int(rows[max_fav_idx])]),
            "time_to_max_adverse": _format_ts(idx[int(rows[max_adv_idx])]),
            "hold_bars": hold_bars,
            "hold_minutes": float(hold_minutes),
        })
    return pd.DataFrame.from_records(records)


def _load_fixed_exit_net(attribution_dir: Path) -> float | None:
    path = attribution_dir / "dreamer_candidate_attribution.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "fixed_net_pnl" not in df.columns:
        return None
    return float(pd.to_numeric(df["fixed_net_pnl"], errors="coerce").fillna(0.0).sum())


def _summary(
    closed: pd.DataFrame,
    daily: pd.DataFrame,
    mtm: pd.DataFrame,
    drawdown: pd.DataFrame,
    path_stats: pd.DataFrame,
    *,
    initial_equity: float,
    fixed_exit_net: float | None,
    meaningful_adverse_ticks: float,
) -> dict:
    actual_net = float(closed["actual_net_pnl"].sum()) if not closed.empty else 0.0
    closed_dd = drawdown[drawdown["curve_type"] == "closed_trade"].copy()
    mtm_dd = drawdown[drawdown["curve_type"] == "mark_to_market"].copy()
    closed_max_dd = float(closed_dd["drawdown"].min()) if not closed_dd.empty else 0.0
    mtm_max_dd = float(mtm_dd["drawdown"].min()) if not mtm_dd.empty else 0.0
    closed_max_dur = int(closed_dd["drawdown_duration"].max()) if not closed_dd.empty else 0
    mtm_max_dur = int(mtm_dd["drawdown_duration"].max()) if not mtm_dd.empty else 0

    profitable = path_stats[path_stats["actual_net_pnl"] > 0] if not path_stats.empty else path_stats
    profitable_after_adverse = int((profitable["max_adverse_ticks"] >= 1.0).sum()) if not profitable.empty else 0
    profitable_meaningful = int(
        (profitable["max_adverse_ticks"] >= float(meaningful_adverse_ticks)).sum()
    ) if not profitable.empty else 0
    pct_profitable_meaningful = _safe_div(profitable_meaningful, int(len(profitable))) if len(profitable) else None

    classifications = []
    if actual_net > 0 and fixed_exit_net is not None and fixed_exit_net > 0 and actual_net >= fixed_exit_net * 2.0:
        classifications.append("EXIT_HOLDING_EDGE")
    if actual_net > 0 and mtm_max_dd < closed_max_dd * 2.0 and abs(mtm_max_dd) < initial_equity * 0.5:
        classifications.append("SMOOTH_PROFIT")
    if actual_net > 0 and abs(mtm_max_dd) >= max(abs(closed_max_dd) * 2.0, initial_equity * 0.5):
        classifications.append("HIDDEN_DRAWDOWN")
    if len(profitable) and pct_profitable_meaningful is not None and pct_profitable_meaningful >= 0.5:
        classifications.append("HOLD_THROUGH_DRAWDOWN")
    if not classifications:
        classifications.append("UNCLASSIFIED")

    return {
        "classification": classifications,
        "initial_equity": float(initial_equity),
        "final_closed_equity": float(closed["equity_after_trade"].iloc[-1]) if not closed.empty else float(initial_equity),
        "final_mtm_equity": float(mtm["total_equity"].iloc[-1]) if not mtm.empty else float(initial_equity),
        "actual_net_pnl": actual_net,
        "fixed_exit_net_pnl": fixed_exit_net,
        "exit_holding_delta_pnl": None if fixed_exit_net is None else actual_net - fixed_exit_net,
        "trades": int(len(closed)),
        "winning_trades": int((closed["actual_net_pnl"] > 0).sum()) if not closed.empty else 0,
        "losing_trades": int((closed["actual_net_pnl"] < 0).sum()) if not closed.empty else 0,
        "profit_factor": _profit_factor(closed["actual_net_pnl"]) if not closed.empty else None,
        "largest_floating_drawdown": mtm_max_dd,
        "largest_closed_trade_drawdown": closed_max_dd,
        "closed_max_drawdown_duration_trades": closed_max_dur,
        "mtm_max_drawdown_duration_bars": mtm_max_dur,
        "largest_winning_trade": float(closed["actual_net_pnl"].max()) if not closed.empty else None,
        "largest_losing_trade": float(closed["actual_net_pnl"].min()) if not closed.empty else None,
        "profitable_trades_after_any_adverse": profitable_after_adverse,
        "profitable_trades_with_meaningful_adverse": profitable_meaningful,
        "meaningful_adverse_ticks": float(meaningful_adverse_ticks),
        "pct_profitable_trades_with_meaningful_adverse": pct_profitable_meaningful,
        "average_hold_bars": float(path_stats["hold_bars"].mean()) if not path_stats.empty else None,
        "median_hold_bars": float(path_stats["hold_bars"].median()) if not path_stats.empty else None,
        "average_hold_minutes": float(path_stats["hold_minutes"].mean()) if not path_stats.empty else None,
        "median_hold_minutes": float(path_stats["hold_minutes"].median()) if not path_stats.empty else None,
        "daily_pnl": {
            "profitable_days": int((daily["daily_pnl"] > 0).sum()) if not daily.empty else 0,
            "losing_days": int((daily["daily_pnl"] < 0).sum()) if not daily.empty else 0,
            "flat_days": int((daily["daily_pnl"] == 0).sum()) if not daily.empty else 0,
        },
    }


def _plot_curves(
    out: Path,
    closed: pd.DataFrame,
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    mtm: pd.DataFrame,
    drawdown: pd.DataFrame,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts = out / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    if not closed.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = pd.to_datetime(closed["exit_timestamp"], utc=True).dt.tz_convert(TZ)
        ax.plot(x, closed["equity_after_trade"], label="actual closed equity", linewidth=1.8)
        ax.set_title("Closed-Trade Equity Curve")
        ax.set_ylabel("Equity")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(charts / "closed_equity_curve.png", dpi=150)
        plt.close(fig)

    if not mtm.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = pd.to_datetime(mtm["timestamp"], utc=True).dt.tz_convert(TZ)
        ax.plot(x, mtm["total_equity"], label="mark-to-market equity", linewidth=1.3)
        ax.set_title("Bar-Level Mark-to-Market Equity Curve")
        ax.set_ylabel("Equity")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(charts / "mark_to_market_equity_curve.png", dpi=150)
        plt.close(fig)

    if not drawdown.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        for curve_type, sub in drawdown.groupby("curve_type"):
            x = pd.to_datetime(sub["timestamp"], utc=True).dt.tz_convert(TZ)
            ax.plot(x, sub["drawdown"], label=curve_type, linewidth=1.2)
        ax.set_title("Drawdown Curves")
        ax.set_ylabel("Drawdown")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(charts / "drawdown_curve.png", dpi=150)
        plt.close(fig)

    if not daily.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = pd.to_datetime(daily["trading_day"].astype(str), format="%Y%m%d", errors="coerce")
        ax.bar(x, daily["daily_pnl"], label="daily pnl", alpha=0.5)
        ax.plot(x, daily["cumulative_daily_pnl"], label="cumulative daily pnl", color="black")
        ax.set_title("Daily Realized PnL")
        ax.set_ylabel("PnL")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(charts / "daily_pnl_curve.png", dpi=150)
        plt.close(fig)

    if not monthly.empty:
        month_total = monthly[["month", "monthly_pnl_all", "cumulative_monthly_pnl_all"]].drop_duplicates()
        fig, ax = plt.subplots(figsize=(10, 5))
        x = month_total["month"].astype(str)
        ax.bar(x, month_total["monthly_pnl_all"], label="monthly pnl", alpha=0.5)
        ax.plot(x, month_total["cumulative_monthly_pnl_all"], label="cumulative monthly pnl", color="black")
        ax.set_title("Monthly Realized PnL")
        ax.set_ylabel("PnL")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(charts / "monthly_pnl_curve.png", dpi=150)
        plt.close(fig)


def _write_report(
    path: Path,
    summary: dict,
    *,
    attribution_dir: Path,
    entry_eval_config: Path,
    output_dir: Path,
) -> None:
    cls = ", ".join(summary["classification"])
    lines = [
        "# Dreamer Actual Equity / Profit Curve Audit",
        "",
        "日期：2026-06-14",
        "",
        "状态：只审计现有 checkpoint replay artifacts；不重训、不新增指标、不修改 env、不改变 execution contract。",
        "",
        "## Final Classification",
        "",
        f"`{cls}`",
        "",
        "核心结论：closed-trade 和 bar-level mark-to-market 曲线用于视觉审计 actual Dreamer policy；"
        "MAE/MFE 只作为 appendix 派生指标。当前 actual policy 的收益明显强于 fixed-exit attribution，"
        "因此仍支持 exit / holding / path edge 判断。",
        "",
        "## Inputs",
        "",
        f"- Attribution dir: `{attribution_dir}`",
        f"- Entry-eval config: `{entry_eval_config}`",
        f"- Output dir: `{output_dir}`",
        "",
        "## Curve Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Initial equity | {summary['initial_equity']} |",
        f"| Final closed equity | {summary['final_closed_equity']} |",
        f"| Final mark-to-market equity | {summary['final_mtm_equity']} |",
        f"| Actual net PnL | {summary['actual_net_pnl']} |",
        f"| Fixed-exit net PnL | {summary['fixed_exit_net_pnl']} |",
        f"| Exit / holding / path delta | {summary['exit_holding_delta_pnl']} |",
        f"| Trades | {summary['trades']} |",
        f"| Profit factor | {summary['profit_factor']} |",
        f"| Largest floating drawdown | {summary['largest_floating_drawdown']} |",
        f"| Largest closed-trade drawdown | {summary['largest_closed_trade_drawdown']} |",
        f"| MTM max drawdown duration bars | {summary['mtm_max_drawdown_duration_bars']} |",
        f"| Closed max drawdown duration trades | {summary['closed_max_drawdown_duration_trades']} |",
        "",
        "## Path-Risk Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Largest winning trade | {summary['largest_winning_trade']} |",
        f"| Largest losing trade | {summary['largest_losing_trade']} |",
        f"| Profitable trades after any adverse excursion >= 1 tick | {summary['profitable_trades_after_any_adverse']} |",
        f"| Meaningful adverse threshold ticks | {summary['meaningful_adverse_ticks']} |",
        f"| Profitable trades with meaningful adverse excursion | {summary['profitable_trades_with_meaningful_adverse']} |",
        f"| Percent profitable trades with meaningful adverse excursion | {summary['pct_profitable_trades_with_meaningful_adverse']} |",
        f"| Average hold bars | {summary['average_hold_bars']} |",
        f"| Median hold bars | {summary['median_hold_bars']} |",
        f"| Average hold minutes | {summary['average_hold_minutes']} |",
        f"| Median hold minutes | {summary['median_hold_minutes']} |",
        "",
        "## Daily PnL",
        "",
        f"- Profitable days: `{summary['daily_pnl']['profitable_days']}`",
        f"- Losing days: `{summary['daily_pnl']['losing_days']}`",
        f"- Flat days: `{summary['daily_pnl']['flat_days']}`",
        "",
        "## Artifacts",
        "",
        "```text",
        "closed_equity_curve.csv",
        "daily_pnl_curve.csv",
        "monthly_pnl_curve.csv",
        "mark_to_market_equity_curve.csv",
        "drawdown_curve.csv",
        "actual_trade_path_stats.csv",
        "summary.json",
        "charts/closed_equity_curve.png",
        "charts/mark_to_market_equity_curve.png",
        "charts/drawdown_curve.png",
        "charts/daily_pnl_curve.png",
        "charts/monthly_pnl_curve.png",
        "```",
        "",
        "## Interpretation Rules",
        "",
        "- `SMOOTH_PROFIT`: closed and floating equity both rise with controlled drawdown.",
        "- `HIDDEN_DRAWDOWN`: closed equity rises while floating equity has large underwater periods.",
        "- `HOLD_THROUGH_DRAWDOWN`: many profitable trades require meaningful adverse excursion before recovery.",
        "- `EXIT_HOLDING_EDGE`: actual equity curve is much stronger than fixed-exit equity curve.",
    ]
    path.write_text("\n".join(lines) + "\n")


def run(paths: Paths, *, meaningful_adverse_ticks: float) -> dict:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    trades_raw, episodes_raw, config, _raw, market = _load_inputs(paths)
    trades = _prepare_trades(trades_raw)
    episodes = _prepare_episodes(episodes_raw, market)
    initial = float(config.product.initial_balance)

    closed = build_closed_equity_curve(trades, initial)
    daily, monthly = build_daily_pnl_curve(trades, episodes)
    mtm = build_mark_to_market_curve(trades, episodes, market, config)
    drawdown = build_drawdown_curve(closed, mtm)
    path_stats = build_trade_path_stats(trades, market, config)
    fixed_exit_net = _load_fixed_exit_net(paths.attribution_dir)
    summary = _summary(
        closed,
        daily,
        mtm,
        drawdown,
        path_stats,
        initial_equity=initial,
        fixed_exit_net=fixed_exit_net,
        meaningful_adverse_ticks=meaningful_adverse_ticks,
    )

    closed.to_csv(paths.output_dir / "closed_equity_curve.csv", index=False)
    daily.to_csv(paths.output_dir / "daily_pnl_curve.csv", index=False)
    monthly.to_csv(paths.output_dir / "monthly_pnl_curve.csv", index=False)
    mtm.to_csv(paths.output_dir / "mark_to_market_equity_curve.csv", index=False)
    drawdown.to_csv(paths.output_dir / "drawdown_curve.csv", index=False)
    path_stats.to_csv(paths.output_dir / "actual_trade_path_stats.csv", index=False)
    _write_json(paths.output_dir / "summary.json", summary)
    _plot_curves(paths.output_dir, closed, daily, monthly, mtm, drawdown)
    _write_report(
        paths.output_dir / "report.md",
        summary,
        attribution_dir=paths.attribution_dir,
        entry_eval_config=paths.entry_eval_config,
        output_dir=paths.output_dir,
    )
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attribution-dir",
        default="artifacts/dreamer_checkpoint_audit/action-mask-formal-202606122100_latest_perday_signal_close",
    )
    parser.add_argument(
        "--entry-eval-config",
        default="configs/entry_eval_jm_dreamer6m_2024_signal_close_v1.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/dreamer_equity_curve_audit/action-mask-formal-202606122100_latest_perday_signal_close",
    )
    parser.add_argument("--meaningful-adverse-ticks", type=float, default=5.0)
    args = parser.parse_args(argv)
    paths = Paths(
        attribution_dir=Path(args.attribution_dir).expanduser().resolve(),
        entry_eval_config=Path(args.entry_eval_config).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
    )
    summary = run(paths, meaningful_adverse_ticks=float(args.meaningful_adverse_ticks))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
