#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gym_trading_env.research.entry_analysis import (
    constrained_oracle,
    fixed_direction_strategy,
    matched_random_strategies,
    strategy_metrics,
    summarize_random_runs,
)
from gym_trading_env.research.entry_evaluator import load_entry_eval_config
from gym_trading_env.research.walk_forward_split_builder import load_required_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit entry_eval dataset opportunity/oracle distribution by split/regime."
    )
    parser.add_argument("--entry-eval-dir", required=True, type=Path)
    parser.add_argument("--entry-eval-config", required=True, type=Path)
    parser.add_argument("--regime-day-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--doc-path", required=True, type=Path)
    parser.add_argument("--random-runs", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=20260620)
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")


def _read_table(path: Path, stem: str) -> pd.DataFrame:
    parquet = path / f"{stem}.parquet"
    csv = path / f"{stem}.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"missing {stem}.parquet/csv in {path}")


def _role_map(folds: list[dict]) -> dict[int, str]:
    fold = folds[-1] if isinstance(folds, list) else folds
    out: dict[int, str] = {}
    for role in ("train", "validation", "test"):
        for day in fold.get(f"{role}_days", []):
            out[int(day)] = role
    return out


def _metric_row(label: str, scope: str, trades: pd.DataFrame, all_days: list[int], initial_balance: float) -> dict[str, Any]:
    m = strategy_metrics(trades, all_days=all_days, initial_balance=initial_balance)
    return {"scope": scope, "strategy": label, **m}


def _candidate_distribution(candidates: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, sub in candidates.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        rec = {col: val for col, val in zip(group_cols, key)}
        best = sub[["long_net_pnl", "short_net_pnl"]].max(axis=1)
        rec.update({
            "candidates": int(len(sub)),
            "long_positive_rate": float((sub["long_net_pnl"] > 0).mean()),
            "short_positive_rate": float((sub["short_net_pnl"] > 0).mean()),
            "best_positive_rate": float((best > 0).mean()),
            "long_mean": float(sub["long_net_pnl"].mean()),
            "short_mean": float(sub["short_net_pnl"].mean()),
            "best_mean": float(best.mean()),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def _oracle_by_scope(
    candidates: pd.DataFrame,
    *,
    group_cols: list[str],
    max_entries_per_day: int,
    initial_balance: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, sub in candidates.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        scope = "|".join(f"{col}={val}" for col, val in zip(group_cols, key))
        all_days = sorted(int(x) for x in sub["trading_day"].unique())
        oracle = constrained_oracle(sub, max_entries_per_day=max_entries_per_day)
        long_only = fixed_direction_strategy(sub, "LONG", max_entries_per_day=max_entries_per_day)
        short_only = fixed_direction_strategy(sub, "SHORT", max_entries_per_day=max_entries_per_day)
        for name, trades in (("constrained_oracle", oracle), ("fixed_long", long_only), ("fixed_short", short_only)):
            rec = {col: val for col, val in zip(group_cols, key)}
            rec.update(_metric_row(name, scope, trades, all_days, initial_balance))
            rows.append(rec)
    return pd.DataFrame(rows)


def _matched_random_by_split(
    candidates: pd.DataFrame,
    *,
    max_entries_per_day: int,
    initial_balance: float,
    runs: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for role, sub in candidates.groupby("split_role", dropna=False, sort=True):
        all_days = sorted(int(x) for x in sub["trading_day"].unique())
        oracle = constrained_oracle(sub, max_entries_per_day=max_entries_per_day)
        random_runs = matched_random_strategies(
            sub,
            oracle,
            max_entries_per_day=max_entries_per_day,
            runs=runs,
            seed=seed,
        )
        row = summarize_random_runs(random_runs, all_days=all_days, initial_balance=initial_balance)
        row["split_role"] = role
        row["target_trades"] = int(len(oracle))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_report(
    path: Path,
    summary: dict[str, Any],
    split_metrics: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    random_summary: pd.DataFrame,
) -> None:
    oracle_split = split_metrics[split_metrics["strategy"] == "constrained_oracle"]
    lines = [
        "# JM8Y WF V1 Entry Eval Dataset Audit",
        "",
        "## Final Answer",
        "",
        f"Classification: `{summary['classification']}`.",
        "",
        "This report audits fixed-exit entry opportunities on the generated entry_eval dataset. "
        "It does not train Dreamer and does not train supervised models.",
        "",
        "## Dataset",
        "",
        f"- Candidates: {summary['candidates']}",
        f"- Outcomes: {summary['outcomes']}",
        f"- Split manifest hash: `{summary['split_manifest_hash']}`",
        f"- Execution timing: `{summary['execution_timing']}`",
        "",
        "## Constrained Oracle By Split",
        "",
        "| split | trades | net_pnl | expectancy | PF | long | short |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for _, row in oracle_split.iterrows():
        lines.append(
            f"| {row['split_role']} | {int(row['trades'])} | {float(row['net_pnl']):.2f} | "
            f"{float(row['expectancy']) if pd.notna(row['expectancy']) else 'NA'} | "
            f"{row['profit_factor']} | {int(row['long_trades'])} | {int(row['short_trades'])} |"
        )
    lines += [
        "",
        "## Matched Random By Split",
        "",
        "| split | runs | target_trades | pnl_mean | pnl_p05 | pnl_p50 | pnl_p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in random_summary.iterrows():
        lines.append(
            f"| {row['split_role']} | {int(row['runs'])} | {int(row['target_trades'])} | "
            f"{float(row['net_pnl_mean']):.2f} | {float(row['net_pnl_p05']):.2f} | "
            f"{float(row['net_pnl_p50']):.2f} | {float(row['net_pnl_p95']):.2f} |"
        )
    lines += [
        "",
        "## Regime Metrics",
        "",
        "See `oracle_by_regime.csv` for split/major/structure regime detail.",
        "",
        "## Outputs",
        "",
    ]
    for name, value in summary["outputs"].items():
        lines.append(f"- `{name}`: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_entry_eval_config(args.entry_eval_config)
    folds = load_required_split_manifest(args.entry_eval_config)
    split_hash_path = Path(args.entry_eval_config).resolve()
    role_by_day = _role_map(folds)
    candidates = _read_table(args.entry_eval_dir, "candidates").copy()
    outcomes = _read_table(args.entry_eval_dir, "outcomes")
    candidates["trading_day"] = candidates["trading_day"].astype(int)
    candidates["split_role"] = candidates["trading_day"].map(role_by_day).fillna("unknown")

    regime = pd.read_csv(args.regime_day_table)
    regime["trading_day"] = regime["trading_day"].astype(int)
    candidates = candidates.merge(
        regime[["trading_day", "major_regime", "structure_regime"]],
        on="trading_day",
        how="left",
    )
    candidates["major_regime"] = candidates["major_regime"].fillna("unknown")
    candidates["structure_regime"] = candidates["structure_regime"].fillna("unknown")

    candidate_by_split = _candidate_distribution(candidates, ["split_role"])
    candidate_by_regime = _candidate_distribution(candidates, ["split_role", "major_regime", "structure_regime"])
    split_metrics = _oracle_by_scope(
        candidates,
        group_cols=["split_role"],
        max_entries_per_day=config.product.max_entries_per_day,
        initial_balance=config.product.initial_balance,
    )
    regime_metrics = _oracle_by_scope(
        candidates,
        group_cols=["split_role", "major_regime", "structure_regime"],
        max_entries_per_day=config.product.max_entries_per_day,
        initial_balance=config.product.initial_balance,
    )
    random_summary = _matched_random_by_split(
        candidates[candidates["split_role"].isin(["train", "validation", "test"])],
        max_entries_per_day=config.product.max_entries_per_day,
        initial_balance=config.product.initial_balance,
        runs=args.random_runs,
        seed=args.random_seed,
    )

    candidate_by_split.to_csv(output_dir / "candidate_distribution_by_split.csv", index=False)
    candidate_by_regime.to_csv(output_dir / "candidate_distribution_by_regime.csv", index=False)
    split_metrics.to_csv(output_dir / "oracle_by_split.csv", index=False)
    regime_metrics.to_csv(output_dir / "oracle_by_regime.csv", index=False)
    random_summary.to_csv(output_dir / "matched_random_by_split.csv", index=False)

    manifest = _read_json(args.entry_eval_dir / "manifest.json")
    split_manifest_path = Path(args.entry_eval_dir / "split_manifest.json")
    split_hash = ""
    if split_manifest_path.exists():
        import hashlib
        split_hash = hashlib.sha256(split_manifest_path.read_bytes()).hexdigest()

    unknown_days = int((candidates["split_role"] == "unknown").sum())
    classification = "ENTRY_EVAL_DATASET_READY"
    if unknown_days:
        classification = "ENTRY_EVAL_DATASET_UNKNOWN_SPLIT_DAYS"
    summary = {
        "classification": classification,
        "entry_eval_dir": str(args.entry_eval_dir),
        "entry_eval_config": str(args.entry_eval_config),
        "candidates": int(len(candidates)),
        "outcomes": int(len(outcomes)),
        "unknown_split_candidates": unknown_days,
        "execution_timing": config.entry_evaluator.execution_timing,
        "split_manifest_hash": split_hash,
        "config_sha256": manifest.get("config_sha256"),
        "data_sha256": manifest.get("data_sha256"),
        "random_runs": int(args.random_runs),
        "random_seed": int(args.random_seed),
        "outputs": {
            "candidate_distribution_by_split": str(output_dir / "candidate_distribution_by_split.csv"),
            "candidate_distribution_by_regime": str(output_dir / "candidate_distribution_by_regime.csv"),
            "oracle_by_split": str(output_dir / "oracle_by_split.csv"),
            "oracle_by_regime": str(output_dir / "oracle_by_regime.csv"),
            "matched_random_by_split": str(output_dir / "matched_random_by_split.csv"),
            "summary": str(output_dir / "summary.json"),
            "report": str(args.doc_path),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    _write_report(args.doc_path, summary, split_metrics, regime_metrics, random_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
