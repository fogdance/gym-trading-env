#!/usr/bin/env python3
"""Compute the frozen JM8Y regime-aware validation checkpoint score.

This script scores one validation-only Dreamer checkpoint audit artifact. It does
not evaluate checkpoints and does not inspect test data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WEIGHTS = {
    "actual_regime_balance": 0.50,
    "fixed_entry_regime_balance": 0.25,
    "matched_random_margin": 0.15,
    "drawdown_penalty": -0.05,
    "regime_instability_penalty": -0.03,
    "extreme_no_trade_penalty": -0.02,
}

MIN_REGIME_DAYS = 5
NO_TRADE_PENALTY_START = 0.80
NO_TRADE_PENALTY_FULL = 1.00
SCORE_SCALE = 1000.0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _filter_split(df: pd.DataFrame, split_role: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "split_role" in out.columns:
        out = out[out["split_role"].astype(str) == split_role].copy()
    out["trading_day"] = pd.to_numeric(out["trading_day"], errors="coerce").astype("Int64")
    return out


def _daily_pnl(df: pd.DataFrame, pnl_col: str, name: str) -> pd.DataFrame:
    if df.empty or pnl_col not in df.columns:
        return pd.DataFrame(columns=["trading_day", name])
    out = df.copy()
    out[pnl_col] = pd.to_numeric(out[pnl_col], errors="coerce").fillna(0.0)
    return out.groupby("trading_day", as_index=False)[pnl_col].sum().rename(columns={pnl_col: name})


def _clip_ratio(num: float, denom: float, *, floor: float = 1.0) -> float:
    scale = max(abs(float(denom)), floor)
    return float(np.clip(float(num) / scale, -1.0, 1.0))


def _max_drawdown(cumulative: pd.Series) -> float:
    if cumulative.empty:
        return 0.0
    running_peak = cumulative.cummax()
    drawdown = cumulative - running_peak
    return float(drawdown.min())


def _summary_metrics(df: pd.DataFrame, days: pd.DataFrame) -> dict[str, Any]:
    total_days = int(len(days))
    trade_days = int(df["trading_day"].nunique()) if not df.empty else 0
    trades = int(len(df))
    actual_net = float(pd.to_numeric(df.get("actual_net_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    no_trade_ratio = 1.0 if total_days == 0 else float((total_days - trade_days) / total_days)
    trades_by_day = df.groupby("trading_day").size() if not df.empty else pd.Series(dtype=int)
    three_trade_day_ratio = 0.0 if total_days == 0 else float((trades_by_day >= 3).sum() / total_days)
    return {
        "validation_days": total_days,
        "trade_days": trade_days,
        "trades": trades,
        "actual_net_pnl": actual_net,
        "no_trade_day_ratio": no_trade_ratio,
        "three_or_more_trade_day_ratio": three_trade_day_ratio,
    }


def _matched_random_mean(summary: dict[str, Any], split_role: str) -> float | None:
    by_split = summary.get("matched_random_fixed_exit_by_split_role")
    if isinstance(by_split, dict):
        item = by_split.get(split_role) or by_split.get(str(split_role))
        if isinstance(item, dict) and item.get("net_pnl_mean") is not None:
            return float(item["net_pnl_mean"])
    item = summary.get("matched_random_fixed_exit_executable_nonoverlap")
    if isinstance(item, dict) and item.get("net_pnl_mean") is not None:
        return float(item["net_pnl_mean"])
    item = summary.get("matched_random_fixed_exit")
    if isinstance(item, dict) and item.get("net_pnl_mean") is not None:
        return float(item["net_pnl_mean"])
    return None


def _regime_scores(days: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for regime, part in days.groupby(group_col, dropna=False):
        day_count = int(len(part))
        actual = float(part["actual_pnl"].sum())
        fixed = float(part["fixed_executable_pnl"].sum())
        oracle = float(part["oracle_pnl"].sum())
        include = day_count >= MIN_REGIME_DAYS and str(regime) != "WARMUP"
        rows.append(
            {
                "group_type": group_col,
                "regime": regime,
                "days": day_count,
                "actual_net_pnl": actual,
                "fixed_executable_net_pnl": fixed,
                "oracle_net_pnl": oracle,
                "actual_vs_oracle_ratio": _clip_ratio(actual, oracle),
                "fixed_vs_oracle_ratio": _clip_ratio(fixed, oracle),
                "included_in_score": bool(include),
            }
        )
    return pd.DataFrame(rows)


def _build_validation_days(
    regime_day_table: Path,
    split_role: str,
    actual: pd.DataFrame,
    fixed: pd.DataFrame,
    oracle: pd.DataFrame,
) -> pd.DataFrame:
    days = pd.read_csv(regime_day_table)
    days["trading_day"] = pd.to_numeric(days["trading_day"], errors="coerce").astype("Int64")
    if "split_role" not in days.columns:
        raise ValueError(f"regime_day_table missing split_role: {regime_day_table}")
    days = days[days["split_role"].astype(str) == split_role].copy()
    for src, pnl_col, name in [
        (actual, "actual_net_pnl", "actual_pnl"),
        (fixed, "net_pnl", "fixed_executable_pnl"),
        (oracle, "net_pnl", "oracle_pnl"),
    ]:
        daily = _daily_pnl(src, pnl_col, name)
        days = days.merge(daily, on="trading_day", how="left")
        days[name] = pd.to_numeric(days[name], errors="coerce").fillna(0.0)
    return days


def score_validation(
    *,
    validation_audit_dir: Path,
    regime_day_table: Path,
    split_role: str,
    checkpoint_name: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = _read_json(validation_audit_dir / "summary.json")
    actual = _filter_split(_read_csv(validation_audit_dir / "dreamer_actual_trades.csv"), split_role)
    fixed = _filter_split(_read_csv(validation_audit_dir / "dreamer_fixed_exit_strategy.csv"), split_role)
    oracle = _filter_split(_read_csv(validation_audit_dir / "oracle_fixed_exit_trades.csv"), split_role)
    days = _build_validation_days(regime_day_table, split_role, actual, fixed, oracle)

    major = _regime_scores(days, "major_regime")
    structure = _regime_scores(days, "structure_regime")
    by_regime = pd.concat([major, structure], ignore_index=True)
    included = by_regime[by_regime["included_in_score"]].copy()
    if included.empty:
        raise ValueError("No regime groups eligible for scoring; check regime_day_table and split_role.")

    actual_regime_balance = float(included["actual_vs_oracle_ratio"].mean())
    fixed_regime_balance = float(included["fixed_vs_oracle_ratio"].mean())
    regime_instability = float(included["actual_vs_oracle_ratio"].std(ddof=0))

    totals = _summary_metrics(actual, days)
    actual_net = float(days["actual_pnl"].sum())
    oracle_net = float(days["oracle_pnl"].sum())
    fixed_net = float(days["fixed_executable_pnl"].sum())
    random_mean = _matched_random_mean(summary, split_role)
    if random_mean is None:
        random_mean = 0.0
        random_source = "missing_default_zero"
    else:
        random_source = "summary"
    random_margin = _clip_ratio(actual_net - random_mean, max(abs(random_mean), abs(oracle_net) * 0.10, 1.0))

    cumulative = days.sort_values("trading_day")["actual_pnl"].cumsum()
    max_dd = _max_drawdown(cumulative)
    drawdown_penalty = float(np.clip(abs(max_dd) / max(abs(oracle_net) * 0.10, 10000.0), 0.0, 1.0))
    no_trade_penalty = float(
        np.clip(
            (totals["no_trade_day_ratio"] - NO_TRADE_PENALTY_START)
            / max(NO_TRADE_PENALTY_FULL - NO_TRADE_PENALTY_START, 1e-9),
            0.0,
            1.0,
        )
    )

    components = {
        "actual_regime_balance": actual_regime_balance,
        "fixed_entry_regime_balance": fixed_regime_balance,
        "matched_random_margin": random_margin,
        "drawdown_penalty": drawdown_penalty,
        "regime_instability_penalty": regime_instability,
        "extreme_no_trade_penalty": no_trade_penalty,
    }
    weighted = {name: value * WEIGHTS[name] for name, value in components.items()}
    score = float(SCORE_SCALE * sum(weighted.values()))
    component_rows = [
        {
            "component": name,
            "value": value,
            "weight": WEIGHTS[name],
            "weighted_value": weighted[name],
        }
        for name, value in components.items()
    ]
    component_df = pd.DataFrame(component_rows)

    result = {
        "checkpoint_name": checkpoint_name,
        "classification": "REGIME_AWARE_VALIDATION_SCORE_COMPUTED",
        "score": score,
        "score_scale": SCORE_SCALE,
        "split_role": split_role,
        "validation_audit_dir": str(validation_audit_dir),
        "regime_day_table": str(regime_day_table),
        "weights": WEIGHTS,
        "actual_net_pnl": actual_net,
        "fixed_executable_net_pnl": fixed_net,
        "oracle_net_pnl": oracle_net,
        "matched_random_net_pnl_mean": random_mean,
        "matched_random_source": random_source,
        "max_drawdown": max_dd,
        **totals,
    }
    return result, component_df, by_regime, days


def _write_report(
    path: Path,
    result: dict[str, Any],
    components: pd.DataFrame,
    by_regime: pd.DataFrame,
) -> None:
    lines = [
        "# Regime-Aware Validation Score Report",
        "",
        "This is a validation-only checkpoint selection score. It must not be run on test artifacts for checkpoint selection.",
        "",
        "## Result",
        "",
        f"- checkpoint: `{result['checkpoint_name']}`",
        f"- score: `{result['score']:.6f}`",
        f"- classification: `{result['classification']}`",
        f"- actual net PnL: `{result['actual_net_pnl']:.2f}`",
        f"- fixed executable net PnL: `{result['fixed_executable_net_pnl']:.2f}`",
        f"- oracle net PnL: `{result['oracle_net_pnl']:.2f}`",
        f"- matched-random mean: `{result['matched_random_net_pnl_mean']:.2f}`",
        f"- max drawdown: `{result['max_drawdown']:.2f}`",
        "",
        "## Components",
        "",
        "| component | value | weight | weighted |",
        "|---|---:|---:|---:|",
    ]
    for _, row in components.iterrows():
        lines.append(
            f"| {row['component']} | {row['value']:.6f} | {row['weight']:.4f} | {row['weighted_value']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Regime Ratios",
            "",
            "| group | regime | days | actual/oracle | fixed/oracle | included |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for _, row in by_regime.iterrows():
        lines.append(
            f"| {row['group_type']} | {row['regime']} | {int(row['days'])} | "
            f"{row['actual_vs_oracle_ratio']:.6f} | {row['fixed_vs_oracle_ratio']:.6f} | "
            f"{bool(row['included_in_score'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-audit-dir", required=True, type=Path)
    parser.add_argument("--regime-day-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--doc-path", required=True, type=Path)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--split-role", default="validation")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result, components, by_regime, days = score_validation(
        validation_audit_dir=args.validation_audit_dir,
        regime_day_table=args.regime_day_table,
        split_role=args.split_role,
        checkpoint_name=args.checkpoint_name,
    )
    components.to_csv(args.output_dir / "validation_score_components.csv", index=False)
    by_regime.to_csv(args.output_dir / "validation_score_by_regime.csv", index=False)
    days.to_csv(args.output_dir / "validation_score_daily_pnl.csv", index=False)
    result["outputs"] = {
        "components": str(args.output_dir / "validation_score_components.csv"),
        "by_regime": str(args.output_dir / "validation_score_by_regime.csv"),
        "daily_pnl": str(args.output_dir / "validation_score_daily_pnl.csv"),
        "summary": str(args.output_dir / "summary.json"),
        "report": str(args.doc_path),
    }
    _write_json(args.output_dir / "summary.json", result)
    _write_report(args.doc_path, result, components, by_regime)


if __name__ == "__main__":
    main()
