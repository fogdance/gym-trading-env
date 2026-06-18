#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gym_trading_env.research.entry_evaluator import (  # noqa: E402
    load_entry_eval_config,
    load_market_frames,
)


TZ = "Asia/Shanghai"


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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default))


def _to_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(TZ)


def _profit_factor(values: pd.Series) -> float | str | None:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return None
    gains = float(arr[arr > 0].sum())
    losses = float(-arr[arr < 0].sum())
    if losses == 0:
        return "inf" if gains > 0 else None
    return float(gains / losses)


def _metric(df: pd.DataFrame, pnl_col: str) -> dict:
    if df.empty:
        return {
            "trades": 0,
            "net_pnl": 0.0,
            "expectancy": None,
            "win_rate": None,
            "profit_factor": None,
        }
    pnl = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)
    trades = int(len(df))
    return {
        "trades": trades,
        "net_pnl": float(pnl.sum()),
        "expectancy": float(pnl.mean()) if trades else None,
        "win_rate": float((pnl > 0).mean()) if trades else None,
        "profit_factor": _profit_factor(pnl),
    }


def _group_metrics(df: pd.DataFrame, group_cols: list[str], pnl_cols: list[str]) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for key, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key)}
        for pnl_col in pnl_cols:
            m = _metric(sub, pnl_col)
            prefix = pnl_col.replace("_net_pnl", "").replace("_pnl", "")
            row.update({
                f"{prefix}_trades": m["trades"],
                f"{prefix}_net_pnl": m["net_pnl"],
                f"{prefix}_expectancy": m["expectancy"],
                f"{prefix}_win_rate": m["win_rate"],
                f"{prefix}_profit_factor": m["profit_factor"],
            })
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _session_label(ts: pd.Timestamp) -> str:
    t = ts.tz_convert(TZ) if ts.tzinfo is not None else ts.tz_localize(TZ)
    minutes = t.hour * 60 + t.minute
    if minutes >= 21 * 60 or minutes < 3 * 60:
        return "night"
    return "day"


def _bucket_flags(ts: pd.Timestamp) -> dict[str, bool]:
    t = ts.tz_convert(TZ) if ts.tzinfo is not None else ts.tz_localize(TZ)
    minutes = t.hour * 60 + t.minute
    session_open = (
        21 * 60 + 1 <= minutes <= 21 * 60 + 5
        or 9 * 60 + 1 <= minutes <= 9 * 60 + 5
        or 10 * 60 + 31 <= minutes <= 10 * 60 + 35
        or 13 * 60 + 31 <= minutes <= 13 * 60 + 35
    )
    near_break = (
        10 * 60 + 10 <= minutes <= 10 * 60 + 15
        or 10 * 60 + 31 <= minutes <= 10 * 60 + 36
        or 11 * 60 + 25 <= minutes <= 11 * 60 + 30
        or 13 * 60 + 31 <= minutes <= 13 * 60 + 36
    )
    near_eod = 14 * 60 + 55 <= minutes <= 15 * 60
    return {
        "night_session": _session_label(t) == "night",
        "day_session": _session_label(t) == "day",
        "session_open": session_open,
        "near_break": near_break,
        "near_eod": near_eod,
    }


def _prepare_attribution(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path / "dreamer_candidate_attribution.csv")
    df = df.copy()
    df["period"] = label
    df["decision_ts"] = _to_ts(df["decision_timestamp"])
    df["entry_ts"] = _to_ts(df["entry_timestamp"])
    df["exit_ts"] = _to_ts(df["exit_timestamp"])
    df["direction"] = df["direction"].astype(str).str.upper()
    df["trading_day"] = pd.to_numeric(df["trading_day"], errors="coerce").astype(int)
    df["month"] = df["month"].astype(str)
    for col in ("actual_net_pnl", "fixed_net_pnl", "exit_holding_delta_pnl"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df = df.sort_values(["trading_day", "entry_ts", "episode_id", "candidate_id"], kind="mergesort")
    df["trade_ordinal"] = df.groupby("trading_day").cumcount() + 1
    df["session_phase"] = df["decision_ts"].map(_session_label)
    df["decision_hhmm"] = df["decision_ts"].dt.strftime("%H:%M")
    for name in ("night_session", "day_session", "session_open", "near_break", "near_eod"):
        df[name] = df["decision_ts"].map(lambda ts, n=name: _bucket_flags(ts)[n])
    return df


def _prepare_oracle(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path / "oracle_fixed_exit_trades.csv")
    df = df.copy()
    df["period"] = label
    df["decision_ts"] = _to_ts(df["decision_timestamp"])
    df["direction"] = df["direction"].astype(str).str.upper()
    df["trading_day"] = pd.to_numeric(df["trading_day"], errors="coerce").astype(int)
    df["month"] = df["month"].astype(str)
    df["net_pnl"] = pd.to_numeric(df["net_pnl"], errors="coerce").fillna(0.0)
    if "session_phase" not in df.columns:
        df["session_phase"] = df["decision_ts"].map(_session_label)
    return df


def _sequence_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day, sub in df.sort_values(["trading_day", "trade_ordinal"]).groupby("trading_day"):
        dirs = sub["direction"].tolist()
        seq = "->".join(dirs)
        if seq not in ("LONG->SHORT->LONG", "LONG->LONG->SHORT", "SHORT->LONG->SHORT"):
            seq_group = "other"
        else:
            seq_group = seq
        rows.append({
            "period": sub["period"].iloc[0],
            "trading_day": int(day),
            "sequence": seq,
            "sequence_group": seq_group,
            "trade_count": int(len(sub)),
            "actual_net_pnl": float(sub["actual_net_pnl"].sum()),
            "fixed_net_pnl": float(sub["fixed_net_pnl"].sum()),
            "exit_holding_delta_pnl": float(sub["exit_holding_delta_pnl"].sum()),
        })
    seq_days = pd.DataFrame.from_records(rows)
    return _group_metrics(
        seq_days,
        ["period", "sequence_group"],
        ["actual_net_pnl", "fixed_net_pnl", "exit_holding_delta_pnl"],
    )


def _ordinal_table(df: pd.DataFrame) -> pd.DataFrame:
    base = _group_metrics(
        df,
        ["period", "trade_ordinal", "direction"],
        ["actual_net_pnl", "fixed_net_pnl", "exit_holding_delta_pnl"],
    )
    hold = df.assign(
        hold_minutes=(df["exit_ts"] - df["entry_ts"]).dt.total_seconds() / 60.0
    ).groupby(["period", "trade_ordinal", "direction"], dropna=False).agg(
        hold_minutes_mean=("hold_minutes", "mean"),
        hold_minutes_median=("hold_minutes", "median"),
        exit_reason_top=("exit_reason", lambda x: x.value_counts().index[0] if len(x) else None),
    ).reset_index()
    return base.merge(hold, on=["period", "trade_ordinal", "direction"], how="left")


def _session_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, sub_period in df.groupby("period"):
        for bucket in ("night_session", "day_session", "session_open", "near_break", "near_eod"):
            sub = sub_period[sub_period[bucket]]
            row = {"period": period, "bucket": bucket}
            for pnl_col in ("actual_net_pnl", "fixed_net_pnl", "exit_holding_delta_pnl"):
                metric = _metric(sub, pnl_col)
                prefix = pnl_col.replace("_net_pnl", "").replace("_pnl", "")
                row.update({
                    f"{prefix}_trades": metric["trades"],
                    f"{prefix}_net_pnl": metric["net_pnl"],
                    f"{prefix}_expectancy": metric["expectancy"],
                    f"{prefix}_win_rate": metric["win_rate"],
                    f"{prefix}_profit_factor": metric["profit_factor"],
                })
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def _oracle_hit_table(dreamer: pd.DataFrame, oracle: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    dist_rows = []
    for period in sorted(dreamer["period"].unique()):
        d = dreamer[dreamer["period"] == period].copy()
        o = oracle[oracle["period"] == period].copy()
        oracle_pairs = set(zip(o["candidate_id"].astype(int), o["direction"]))
        oracle_rows = set(o["decision_row"].astype(int))
        oracle_by_day_dir = {
            (int(day), str(direction)): sub["decision_row"].astype(int).to_numpy()
            for (day, direction), sub in o.groupby(["trading_day", "direction"])
        }
        exact = []
        wrong_side = []
        nearest = []
        near5 = []
        for tr in d.itertuples(index=False):
            pair = (int(tr.candidate_id), str(tr.direction))
            exact_hit = pair in oracle_pairs
            exact.append(exact_hit)
            wrong_side.append((int(tr.decision_row) in oracle_rows) and not exact_hit)
            rows_for_key = oracle_by_day_dir.get((int(tr.trading_day), str(tr.direction)))
            if rows_for_key is None or rows_for_key.size == 0:
                nearest.append(np.nan)
                near5.append(False)
            else:
                distance = int(np.min(np.abs(rows_for_key - int(tr.decision_row))))
                nearest.append(distance)
                near5.append(distance <= 5)
        rows.append({
            "period": period,
            "dreamer_trades": int(len(d)),
            "oracle_trades": int(len(o)),
            "exact_hits": int(np.sum(exact)),
            "dreamer_exact_hit_rate": float(np.mean(exact)) if len(exact) else None,
            "oracle_coverage_rate": float(np.sum(exact) / len(o)) if len(o) else None,
            "same_row_wrong_side": int(np.sum(wrong_side)),
            "near_oracle_same_direction_within_5_rows": int(np.sum(near5)),
            "median_abs_row_distance_same_direction": (
                float(np.nanmedian(nearest)) if np.isfinite(nearest).any() else None
            ),
        })
        for group_cols in (["direction"], ["session_phase"], ["month"], ["direction", "session_phase"]):
            for key, sub in o.groupby(group_cols, dropna=False):
                if not isinstance(key, tuple):
                    key = (key,)
                item = {"period": period, "distribution": "+".join(group_cols)}
                item.update({col: val for col, val in zip(group_cols, key)})
                item.update(_metric(sub, "net_pnl"))
                dist_rows.append(item)
    return pd.DataFrame.from_records(rows), pd.DataFrame.from_records(dist_rows)


def _market_regime(config_path: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_entry_eval_config(config_path)
    _raw, market = load_market_frames(config)
    df = market.copy()
    df = df[pd.to_numeric(df["mask_t"], errors="coerce").fillna(0.0) > 0].copy()
    price_cols = {
        "open": "exec_open",
        "high": "H_t",
        "low": "L_t",
        "close": "C_t",
    }
    for col in price_cols.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    volume_col = "V_t" if "V_t" in df.columns else None
    oi_col = "I_t" if "I_t" in df.columns else None
    if volume_col:
        df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0.0)
    if oi_col:
        df[oi_col] = pd.to_numeric(df[oi_col], errors="coerce")
    rows = []
    prev_close = None
    for day, sub in df.groupby("trading_day"):
        sub = sub.sort_index()
        first_open = float(sub[price_cols["open"]].iloc[0])
        first_close = float(sub[price_cols["close"]].iloc[0])
        last_close = float(sub[price_cols["close"]].iloc[-1])
        high = float(sub[price_cols["high"]].max())
        low = float(sub[price_cols["low"]].min())
        close_diff = sub[price_cols["close"]].diff().dropna()
        row = {
            "period": label,
            "trading_day": int(day),
            "month": str(int(day) // 100),
            "first_open": first_open,
            "first_close": first_close,
            "last_close": last_close,
            "close_to_close": None if prev_close is None else last_close - prev_close,
            "intraday_drift": last_close - first_open,
            "realized_vol_close_diff": float(close_diff.std(ddof=0)) if len(close_diff) else 0.0,
            "range": high - low,
            "range_pct": (high - low) / first_open if first_open else None,
            "valid_bars": int(len(sub)),
        }
        if volume_col:
            row["volume_sum"] = float(sub[volume_col].sum())
        if oi_col:
            row["open_interest_change"] = float(sub[oi_col].iloc[-1] - sub[oi_col].iloc[0])
        rows.append(row)
        prev_close = last_close
    daily = pd.DataFrame.from_records(rows)
    summary_rows = []
    numeric_cols = [
        "close_to_close", "intraday_drift", "realized_vol_close_diff",
        "range", "range_pct", "volume_sum", "open_interest_change",
    ]
    for group_cols in (["period"], ["period", "month"]):
        for key, sub in daily.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            row = {col: val for col, val in zip(group_cols, key)}
            row["days"] = int(len(sub))
            for col in numeric_cols:
                if col not in sub.columns:
                    continue
                values = pd.to_numeric(sub[col], errors="coerce").dropna()
                row[f"{col}_mean"] = float(values.mean()) if len(values) else None
                row[f"{col}_median"] = float(values.median()) if len(values) else None
            row["positive_close_to_close_ratio"] = (
                float((pd.to_numeric(sub["close_to_close"], errors="coerce").dropna() > 0).mean())
                if "close_to_close" in sub else None
            )
            row["positive_intraday_drift_ratio"] = float((sub["intraday_drift"] > 0).mean())
            summary_rows.append(row)
    return daily, pd.DataFrame.from_records(summary_rows)


def _load_equity_summary(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text())


def _load_checkpoint_summary(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text())


def _reference_table(checkpoint: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for period, data in checkpoint.items():
        executable = data.get("dreamer_entry_fixed_exit_executable_nonoverlap", {}).get("metrics", {})
        matched = data.get("matched_random_fixed_exit_executable_nonoverlap", {})
        oracle = data.get("constrained_oracle_fixed_exit", {})
        rows.append({
            "period": period,
            "dreamer_executable_fixed_trades": executable.get("trades"),
            "dreamer_executable_fixed_net_pnl": executable.get("net_pnl"),
            "dreamer_executable_fixed_expectancy": executable.get("expectancy"),
            "matched_random_runs": matched.get("runs"),
            "matched_random_trade_count_mean": matched.get("trade_count_mean"),
            "matched_random_net_pnl_mean": matched.get("net_pnl_mean"),
            "matched_random_net_pnl_p05": matched.get("net_pnl_p05"),
            "matched_random_net_pnl_p50": matched.get("net_pnl_p50"),
            "matched_random_net_pnl_p95": matched.get("net_pnl_p95"),
            "oracle_trades": oracle.get("trades"),
            "oracle_net_pnl": oracle.get("net_pnl"),
            "oracle_expectancy": oracle.get("expectancy"),
            "oracle_long_trades": oracle.get("long_trades"),
            "oracle_short_trades": oracle.get("short_trades"),
        })
    return pd.DataFrame.from_records(rows)


def _classification(summary: dict) -> dict:
    dir_2025 = summary["direction"]["2025"]
    long_loss = abs(min(0.0, dir_2025.get("LONG", {}).get("actual_net_pnl", 0.0)))
    short_loss = abs(min(0.0, dir_2025.get("SHORT", {}).get("actual_net_pnl", 0.0)))
    total_loss = long_loss + short_loss
    long_loss_share = long_loss / total_loss if total_loss else 0.0
    delta_2024 = float(summary["equity"]["2024"]["exit_holding_delta_pnl"])
    delta_2025 = float(summary["equity"]["2025"]["exit_holding_delta_pnl"])
    fixed_2025 = float(summary["equity"]["2025"]["fixed_exit_net_pnl"])
    actual_2025 = float(summary["equity"]["2025"]["actual_net_pnl"])

    labels = []
    reasons = []
    if actual_2025 < 0 and delta_2024 > 0 and abs(delta_2025) <= max(1000.0, abs(delta_2024) * 0.05):
        labels.append("EXIT_EDGE_NOT_GENERALIZED")
        reasons.append(
            f"exit/holding delta collapsed from {delta_2024:.0f} in 2024 to {delta_2025:.0f} in 2025")
    if actual_2025 < 0 and long_loss_share >= 0.55:
        labels.append("LONG_BIAS_FAIL")
        reasons.append(f"LONG side explains {long_loss_share:.1%} of 2025 actual directional loss")
    if actual_2025 < 0 and fixed_2025 < 0 and total_loss and long_loss_share < 0.75:
        labels.append("BROAD_POLICY_NON_GENERALIZATION")
        reasons.append("2025 losses remain negative after fixed-exit attribution and affect both directions")
    if not labels:
        labels.append("INCONCLUSIVE")
        reasons.append("diagnostic thresholds did not isolate a dominant failure mode")
    return {
        "primary": labels[0],
        "secondary": labels[1:],
        "reasons": reasons,
    }


def _table_md(df: pd.DataFrame, max_rows: int = 20) -> list[str]:
    if df.empty:
        return ["(empty)"]
    show = df.head(max_rows).copy()
    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value)

    headers = [str(col) for col in show.columns]
    rows = [[fmt(value) for value in row] for row in show.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(headers))
    ]

    def render(values: list[str]) -> str:
        cells = [value.ljust(widths[idx]) for idx, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    lines = [
        render(headers),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(render(row) for row in rows)
    if len(df) > len(show):
        lines.append(f"(showing first {len(show)} of {len(df)} rows)")
    return lines


def _write_report(path: Path, summary: dict, tables: dict[str, pd.DataFrame]) -> None:
    classification = summary["classification"]
    lines = [
        "# Dreamer 2024 vs 2025 Failure Attribution",
        "",
        "日期：2026-06-14",
        "",
        "状态：只读取既有 2024/2025 signal-close artifacts 和 raw bars；不重训、不新增指标、不修改 env/reward/contract/max_entries。",
        "",
        "## Final Classification",
        "",
        f"Primary: `{classification['primary']}`",
        "",
        f"Secondary: `{classification['secondary']}`",
        "",
        "Reasons:",
    ]
    lines.extend([f"- {x}" for x in classification["reasons"]])
    lines.extend([
        "",
        "## Topline",
        "",
        "| Metric | 2024 | 2025 |",
        "|---|---:|---:|",
        f"| Actual net PnL | {summary['equity']['2024']['actual_net_pnl']} | {summary['equity']['2025']['actual_net_pnl']} |",
        f"| Fixed-exit net PnL | {summary['equity']['2024']['fixed_exit_net_pnl']} | {summary['equity']['2025']['fixed_exit_net_pnl']} |",
        f"| Exit / holding / path delta | {summary['equity']['2024']['exit_holding_delta_pnl']} | {summary['equity']['2025']['exit_holding_delta_pnl']} |",
        f"| Final closed equity | {summary['equity']['2024']['final_closed_equity']} | {summary['equity']['2025']['final_closed_equity']} |",
        f"| Largest floating drawdown | {summary['equity']['2024']['largest_floating_drawdown']} | {summary['equity']['2025']['largest_floating_drawdown']} |",
        f"| Trades | {summary['equity']['2024']['trades']} | {summary['equity']['2025']['trades']} |",
        "",
        "## Direction",
        "",
    ])
    lines.extend(_table_md(tables["direction"]))
    lines.extend(["", "## Trade Ordinal", ""])
    lines.extend(_table_md(tables["ordinal"], max_rows=30))
    lines.extend(["", "## Sequence Patterns", ""])
    lines.extend(_table_md(tables["sequence"]))
    lines.extend(["", "## Session / Timing Buckets", ""])
    lines.extend(_table_md(tables["session"]))
    lines.extend(["", "## Oracle Hit Rate", ""])
    lines.extend(_table_md(tables["oracle_hits"]))
    lines.extend(["", "## Fixed-Exit References", ""])
    lines.extend(_table_md(tables["fixed_exit_references"]))
    lines.extend(["", "## Oracle Opportunity Distribution", ""])
    lines.extend(_table_md(tables["oracle_distribution"], max_rows=40))
    lines.extend(["", "## Market Regime Summary", ""])
    lines.extend(_table_md(tables["market_regime"]))
    lines.extend([
        "",
        "## Interpretation",
        "",
        "2025 failure is not a data/config blocker: checkpoint replay completed with the same signal-on-close contract.",
        "",
        "The largest mechanical change is that 2024 actual profit depended heavily on exit/holding/path delta, while 2025 actual and fixed-exit PnL are almost identical and both negative. Directionally, 2025 LONG entries are strongly negative under fixed exit, while actual SHORT also becomes negative after exit/holding behavior. This points to a collapsed full-policy edge plus a LONG-side entry bias, not a simple isolated timing bug.",
    ])
    path.write_text("\n".join(lines) + "\n")


def run(args) -> dict:
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    docs_path = Path(args.docs_report).expanduser().resolve()
    docs_path.parent.mkdir(parents=True, exist_ok=True)

    dirs = {
        "2024": Path(args.dir_2024).expanduser().resolve(),
        "2025": Path(args.dir_2025).expanduser().resolve(),
    }
    eq_dirs = {
        "2024": Path(args.equity_2024).expanduser().resolve(),
        "2025": Path(args.equity_2025).expanduser().resolve(),
    }
    configs = {
        "2024": Path(args.config_2024).expanduser().resolve(),
        "2025": Path(args.config_2025).expanduser().resolve(),
    }

    attribution = pd.concat([
        _prepare_attribution(dirs["2024"], "2024"),
        _prepare_attribution(dirs["2025"], "2025"),
    ], ignore_index=True)
    oracle = pd.concat([
        _prepare_oracle(dirs["2024"], "2024"),
        _prepare_oracle(dirs["2025"], "2025"),
    ], ignore_index=True)

    direction = _group_metrics(
        attribution,
        ["period", "direction"],
        ["actual_net_pnl", "fixed_net_pnl", "exit_holding_delta_pnl"],
    )
    ordinal = _ordinal_table(attribution)
    sequence = _sequence_table(attribution)
    session = _session_table(attribution)
    oracle_hits, oracle_distribution = _oracle_hit_table(attribution, oracle)

    market_daily = []
    market_summary = []
    for period, config in configs.items():
        daily, summary = _market_regime(config, period)
        market_daily.append(daily)
        market_summary.append(summary)
    market_daily_df = pd.concat(market_daily, ignore_index=True)
    market_summary_df = pd.concat(market_summary, ignore_index=True)

    equity = {period: _load_equity_summary(path) for period, path in eq_dirs.items()}
    checkpoint = {period: _load_checkpoint_summary(path) for period, path in dirs.items()}
    fixed_exit_references = _reference_table(checkpoint)
    direction_nested = {}
    for period, sub in direction.groupby("period"):
        direction_nested[period] = {
            str(row.direction): {
                "actual_net_pnl": float(row.actual_net_pnl),
                "actual_expectancy": row.actual_expectancy,
                "actual_win_rate": row.actual_win_rate,
                "actual_profit_factor": row.actual_profit_factor,
                "fixed_net_pnl": float(row.fixed_net_pnl),
                "fixed_expectancy": row.fixed_expectancy,
                "fixed_win_rate": row.fixed_win_rate,
                "fixed_profit_factor": row.fixed_profit_factor,
                "exit_holding_delta_pnl": float(row.exit_holding_delta_net_pnl),
            }
            for row in sub.itertuples(index=False)
        }

    summary = {
        "inputs": {
            "dir_2024": str(dirs["2024"]),
            "dir_2025": str(dirs["2025"]),
            "config_2024": str(configs["2024"]),
            "config_2025": str(configs["2025"]),
        },
        "equity": equity,
        "fixed_exit_references": fixed_exit_references.to_dict(orient="records"),
        "direction": direction_nested,
        "oracle_hits": oracle_hits.to_dict(orient="records"),
        "classification": {},
    }
    summary["classification"] = _classification(summary)

    tables = {
        "direction": direction,
        "ordinal": ordinal,
        "sequence": sequence,
        "session": session,
        "oracle_hits": oracle_hits,
        "oracle_distribution": oracle_distribution,
        "fixed_exit_references": fixed_exit_references,
        "market_daily": market_daily_df,
        "market_regime": market_summary_df,
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    _write_json(out / "summary.json", summary)
    _write_report(out / "report.md", summary, tables)
    _write_report(docs_path, summary, tables)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-2024", required=True)
    parser.add_argument("--dir-2025", required=True)
    parser.add_argument("--equity-2024", required=True)
    parser.add_argument("--equity-2025", required=True)
    parser.add_argument("--config-2024", required=True)
    parser.add_argument("--config-2025", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--docs-report", default="docs/dreamer_2024_2025_failure_attribution_20260614.md")
    args = parser.parse_args(argv)
    summary = run(args)
    print(json.dumps(summary["classification"], ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
