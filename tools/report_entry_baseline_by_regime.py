#!/usr/bin/env python3
"""Summarize entry-eval supervised baselines by walk-forward regime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _profit_factor(values: pd.Series) -> tuple[float | None, str]:
    wins = float(values[values > 0].sum())
    losses = float(values[values < 0].sum())
    if len(values) == 0:
        return None, "not_computed"
    if losses == 0.0:
        if wins > 0.0:
            return float("inf"), "no_losing_trades"
        return None, "not_computed"
    if wins == 0.0:
        return 0.0, "no_winning_trades"
    return wins / abs(losses), "computed"


def _metrics(df: pd.DataFrame, pnl_col: str = "net_pnl") -> dict[str, Any]:
    if df.empty:
        return {
            "trades": 0,
            "net_pnl": 0.0,
            "expectancy": None,
            "win_rate": None,
            "profit_factor": None,
            "profit_factor_status": "not_computed",
            "long_trades": 0,
            "short_trades": 0,
        }
    pnl = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)
    pf, pf_status = _profit_factor(pnl)
    direction = df.get("direction", pd.Series([], dtype=object)).astype(str)
    return {
        "trades": int(len(df)),
        "net_pnl": float(pnl.sum()),
        "expectancy": float(pnl.mean()),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": pf,
        "profit_factor_status": pf_status,
        "long_trades": int((direction == "LONG").sum()) if len(direction) else 0,
        "short_trades": int((direction == "SHORT").sum()) if len(direction) else 0,
    }


def _group_metrics(
    df: pd.DataFrame,
    group_cols: list[str],
    pnl_col: str = "net_pnl",
    prefix_cols: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    for keys, part in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        if prefix_cols:
            row.update(prefix_cols)
        row.update(_metrics(part, pnl_col=pnl_col))
        rows.append(row)
    return pd.DataFrame(rows)


def _merge_regime(df: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["trading_day"] = pd.to_numeric(out["trading_day"], errors="coerce").astype("Int64")
    regime_small = regime[
        [
            "trading_day",
            "split_role",
            "major_regime",
            "structure_regime",
            "valid_ratio",
            "ret_20d",
            "ret_60d",
        ]
    ].copy()
    regime_small["trading_day"] = pd.to_numeric(regime_small["trading_day"], errors="coerce").astype("Int64")
    return out.merge(regime_small, on="trading_day", how="left", validate="many_to_one")


def _trade_file_specs(entry_eval_dir: Path) -> list[tuple[str, str, Path, Path]]:
    return [
        ("ridge", "ridge_window_summary", entry_eval_dir / "ridge_strategy_trades.csv", entry_eval_dir / "ridge_model_report.json"),
        (
            "xgboost_window",
            "window_summary_latest_mean_std_delta",
            entry_eval_dir / "xgboost_window_summary_trades.csv",
            entry_eval_dir / "xgboost_window_summary_report.json",
        ),
        (
            "xgboost_flattened",
            "flattened_60x18_market_seq",
            entry_eval_dir / "xgboost_flattened_trades.csv",
            entry_eval_dir / "xgboost_flattened_report.json",
        ),
        ("primary", "primary_selected", entry_eval_dir / "strategy_trades.csv", entry_eval_dir / "model_report.json"),
    ]


def _model_row(model_name: str, encoding: str, report: dict[str, Any]) -> dict[str, Any]:
    agg = report.get("aggregate_full") or {}
    if model_name == "primary":
        gate2 = report.get("gate2_status")
        gate3 = report.get("gate3_status") or "BLOCKED_BY_GATE2"
        encoding = report.get("primary_feature_encoding") or encoding
    else:
        gate2 = report.get("gate2_status")
        gate3 = "BLOCKED_BY_GATE2" if gate2 == "FAIL" else None
    validation = None
    folds = report.get("folds") or []
    if folds:
        variants = (folds[0].get("variants") or {})
        full = variants.get("full") or {}
        chosen = (full.get("validation") or {}).get("chosen") or {}
        validation = chosen.get("metrics")
    decile = agg.get("score_decile") or agg.get("score_decile_executable") or {}
    return {
        "model_name": model_name,
        "feature_encoding": encoding,
        "gate2_status": gate2,
        "gate3_status": gate3,
        "validation_chosen_trades": None if validation is None else validation.get("trades"),
        "validation_chosen_net_pnl": None if validation is None else validation.get("net_pnl"),
        "validation_chosen_expectancy": None if validation is None else validation.get("expectancy"),
        "test_trades": agg.get("trades"),
        "test_net_pnl": agg.get("net_pnl"),
        "test_expectancy": agg.get("expectancy"),
        "test_profit_factor": agg.get("profit_factor"),
        "test_profit_factor_status": agg.get("profit_factor_status"),
        "score_decile_spearman": decile.get("spearman"),
        "score_decile_top_above_bottom": decile.get("top_above_bottom"),
    }


def _resolve_split_manifest_hash(entry_eval_dir: Path, manifest: dict[str, Any]) -> str | None:
    value = manifest.get("split_manifest_hash")
    if value:
        return str(value)
    candidates = [
        entry_eval_dir / "split_manifest.sha256",
        entry_eval_dir.parent.parent / "walk_forward_splits" / entry_eval_dir.name / "split_manifest.sha256",
        entry_eval_dir.parent.parent.parent / "walk_forward_splits" / entry_eval_dir.name / "split_manifest.sha256",
    ]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    split_manifest = entry_eval_dir / "split_manifest.json"
    if split_manifest.exists():
        return hashlib.sha256(split_manifest.read_bytes()).hexdigest()
    return None


def _prediction_deciles(preds: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()
    merged = _merge_regime(preds, regime)
    merged = merged[pd.notna(merged["pred_score"])].copy()
    if merged.empty:
        return pd.DataFrame()
    try:
        merged["score_decile"] = pd.qcut(merged["pred_score"], 10, labels=False, duplicates="drop")
    except ValueError:
        merged["score_decile"] = 0
    rows = []
    for cols in [
        ["split_role", "score_decile"],
        ["split_role", "major_regime", "score_decile"],
        ["split_role", "structure_regime", "score_decile"],
    ]:
        for keys, part in merged.groupby(cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {col: key for col, key in zip(cols, keys)}
            row["scope"] = "|".join(f"{col}={row[col]}" for col in cols)
            row["candidate_count"] = int(len(part))
            row["mean_pred_score"] = float(part["pred_score"].mean())
            row["mean_realized_pred_direction_net_pnl"] = float(part["realized_pred_direction_net_pnl"].mean())
            row["net_realized_pred_direction_pnl"] = float(part["realized_pred_direction_net_pnl"].sum())
            row["pred_long_count"] = int((part["pred_best_direction"] == "LONG").sum())
            row["pred_short_count"] = int((part["pred_best_direction"] == "SHORT").sum())
            row["action_long_count"] = int((part["pred_action"] == "LONG").sum())
            row["action_short_count"] = int((part["pred_action"] == "SHORT").sum())
            row["action_flat_count"] = int((part["pred_action"] == "FLAT").sum())
            rows.append(row)
    return pd.DataFrame(rows)


def _format_metric(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if np.isnan(value):
            return "NA"
        if np.isinf(value):
            return "inf"
        return f"{value:.2f}"
    return str(value)


def _write_report(
    path: Path,
    summary: dict[str, Any],
    model_summary: pd.DataFrame,
    by_split: pd.DataFrame,
    by_regime: pd.DataFrame,
    oracle_by_split: pd.DataFrame,
    matched_random_by_split: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# JM8Y WF V1 Supervised Fixed-Exit Baseline By Regime")
    lines.append("")
    lines.append("Final answer: the JM8Y dataset and entry-eval dataset are usable under the user-confirmed main-continuous contract assumption, but the supervised fixed-exit baseline does not pass Gate 2. It remains a dataset/baseline diagnostic result, not Dreamer training admission.")
    lines.append("")
    lines.append("## Classification")
    lines.append("")
    lines.append(f"- `{summary['classification']}`")
    lines.append(f"- split manifest hash: `{summary['split_manifest_hash']}`")
    lines.append(f"- entry_eval: `{summary['entry_eval_dir']}`")
    lines.append("")
    lines.append("## Model Summary")
    lines.append("")
    lines.append("| model | encoding | gate2 | val trades | val net | test trades | test net | test exp | decile spearman |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in model_summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_name"]),
                    str(row["feature_encoding"]),
                    str(row["gate2_status"]),
                    _format_metric(row.get("validation_chosen_trades")),
                    _format_metric(row.get("validation_chosen_net_pnl")),
                    _format_metric(row.get("test_trades")),
                    _format_metric(row.get("test_net_pnl")),
                    _format_metric(row.get("test_expectancy")),
                    _format_metric(row.get("score_decile_spearman")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Important: primary flattened XGBoost selected a validation no-trade threshold, then produced only 9 test trades. Top 5% / 10% / 20% selective-entry diagnostics are all negative, so the small positive 9-trade executable result is not sufficient for Gate 2.")
    lines.append("")
    lines.append("## Baseline References")
    lines.append("")
    oracle = oracle_by_split[oracle_by_split["strategy"] == "constrained_oracle"].copy()
    lines.append("| split | oracle trades | oracle net | oracle exp | matched-random mean | matched-random p50 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for split in ["train", "validation", "test"]:
        o = oracle[oracle["split_role"] == split]
        r = matched_random_by_split[matched_random_by_split["split_role"] == split]
        lines.append(
            f"| {split} | "
            f"{_format_metric(o['trades'].iloc[0] if not o.empty else None)} | "
            f"{_format_metric(o['net_pnl'].iloc[0] if not o.empty else None)} | "
            f"{_format_metric(o['expectancy'].iloc[0] if not o.empty else None)} | "
            f"{_format_metric(r['net_pnl_mean'].iloc[0] if not r.empty else None)} | "
            f"{_format_metric(r['net_pnl_p50'].iloc[0] if not r.empty else None)} |"
        )
    lines.append("")
    lines.append("## Primary Test Trades By Regime")
    lines.append("")
    primary_regime = by_regime[by_regime["model_name"] == "primary"].copy()
    if primary_regime.empty:
        lines.append("No primary test trades were available for regime grouping.")
    else:
        show = primary_regime.sort_values(["split_role", "major_regime", "structure_regime"]).head(20)
        lines.append("| split | major | structure | trades | net | exp | long | short |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for _, row in show.iterrows():
            lines.append(
                f"| {row['split_role']} | {row['major_regime']} | {row['structure_regime']} | "
                f"{int(row['trades'])} | {_format_metric(row['net_pnl'])} | {_format_metric(row['expectancy'])} | "
                f"{int(row['long_trades'])} | {int(row['short_trades'])} |"
            )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for name, value in summary["outputs"].items():
        lines.append(f"- `{name}`: `{value}`")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("- Do not start Dreamer training from this result alone.")
    lines.append("- The JM8Y split remains useful as a broader dataset, but current fixed-exit supervised entry selection is still weak.")
    lines.append("- Next dataset-side step, if needed, is freezing a regime-aware validation score before any Dreamer pilot.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry-eval-dir", required=True, type=Path)
    parser.add_argument("--entry-eval-audit-dir", required=True, type=Path)
    parser.add_argument("--regime-day-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--doc-path", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    regime = pd.read_csv(args.regime_day_table)
    model_report = _read_json(args.entry_eval_dir / "model_report.json")
    manifest = _read_json(args.entry_eval_dir / "manifest.json")

    model_rows = []
    split_frames = []
    regime_frames = []
    direction_frames = []
    session_frames = []
    month_frames = []
    for model_name, encoding, trade_path, report_path in _trade_file_specs(args.entry_eval_dir):
        if not trade_path.exists() or not report_path.exists():
            continue
        report = _read_json(report_path)
        model_rows.append(_model_row(model_name, encoding, report))
        trades = pd.read_csv(trade_path)
        trades = _merge_regime(trades, regime)
        trades["model_name"] = model_name
        trades["feature_encoding"] = encoding
        split_frames.append(
            _group_metrics(trades, ["model_name", "feature_encoding", "split_role"])
        )
        regime_frames.append(
            _group_metrics(trades, ["model_name", "feature_encoding", "split_role", "major_regime", "structure_regime"])
        )
        direction_frames.append(
            _group_metrics(trades, ["model_name", "feature_encoding", "split_role", "direction"])
        )
        session_frames.append(
            _group_metrics(trades, ["model_name", "feature_encoding", "split_role", "session_phase"])
        )
        month_frames.append(
            _group_metrics(trades, ["model_name", "feature_encoding", "split_role", "month"])
        )

    model_summary = pd.DataFrame(model_rows)
    by_split = pd.concat(split_frames, ignore_index=True) if split_frames else pd.DataFrame()
    by_regime = pd.concat(regime_frames, ignore_index=True) if regime_frames else pd.DataFrame()
    by_direction = pd.concat(direction_frames, ignore_index=True) if direction_frames else pd.DataFrame()
    by_session = pd.concat(session_frames, ignore_index=True) if session_frames else pd.DataFrame()
    by_month = pd.concat(month_frames, ignore_index=True) if month_frames else pd.DataFrame()

    preds = pd.read_csv(args.entry_eval_dir / "baseline_predictions.csv")
    prediction_deciles = _prediction_deciles(preds, regime)

    oracle_by_split = pd.read_csv(args.entry_eval_audit_dir / "oracle_by_split.csv")
    oracle_by_regime = pd.read_csv(args.entry_eval_audit_dir / "oracle_by_regime.csv")
    matched_random_by_split = pd.read_csv(args.entry_eval_audit_dir / "matched_random_by_split.csv")

    outputs = {
        "model_summary": str(args.output_dir / "model_summary.csv"),
        "strategy_by_split": str(args.output_dir / "strategy_by_split.csv"),
        "strategy_by_regime": str(args.output_dir / "strategy_by_regime.csv"),
        "strategy_by_direction": str(args.output_dir / "strategy_by_direction.csv"),
        "strategy_by_session": str(args.output_dir / "strategy_by_session.csv"),
        "strategy_by_month": str(args.output_dir / "strategy_by_month.csv"),
        "prediction_deciles_by_regime": str(args.output_dir / "prediction_deciles_by_regime.csv"),
        "oracle_by_split": str(args.output_dir / "oracle_by_split.csv"),
        "oracle_by_regime": str(args.output_dir / "oracle_by_regime.csv"),
        "matched_random_by_split": str(args.output_dir / "matched_random_by_split.csv"),
        "summary": str(args.output_dir / "summary.json"),
        "report": str(args.doc_path),
    }

    model_summary.to_csv(outputs["model_summary"], index=False)
    by_split.to_csv(outputs["strategy_by_split"], index=False)
    by_regime.to_csv(outputs["strategy_by_regime"], index=False)
    by_direction.to_csv(outputs["strategy_by_direction"], index=False)
    by_session.to_csv(outputs["strategy_by_session"], index=False)
    by_month.to_csv(outputs["strategy_by_month"], index=False)
    prediction_deciles.to_csv(outputs["prediction_deciles_by_regime"], index=False)
    oracle_by_split.to_csv(outputs["oracle_by_split"], index=False)
    oracle_by_regime.to_csv(outputs["oracle_by_regime"], index=False)
    matched_random_by_split.to_csv(outputs["matched_random_by_split"], index=False)

    primary = model_summary[model_summary["model_name"] == "primary"]
    primary_gate2 = None if primary.empty else primary["gate2_status"].iloc[0]
    classification = "SUPERVISED_BASELINE_GATE2_FAIL" if primary_gate2 == "FAIL" else "SUPERVISED_BASELINE_REVIEW_REQUIRED"
    summary = {
        "classification": classification,
        "entry_eval_dir": str(args.entry_eval_dir),
        "entry_eval_audit_dir": str(args.entry_eval_audit_dir),
        "split_manifest_hash": _resolve_split_manifest_hash(args.entry_eval_dir, manifest),
        "primary_gate2_status": primary_gate2,
        "primary_feature_encoding": None if primary.empty else primary["feature_encoding"].iloc[0],
        "primary_test_net_pnl": None if primary.empty else float(primary["test_net_pnl"].iloc[0]),
        "primary_test_trades": None if primary.empty else int(primary["test_trades"].iloc[0]),
        "outputs": outputs,
    }
    _write_json(Path(outputs["summary"]), summary)
    _write_report(args.doc_path, summary, model_summary, by_split, by_regime, oracle_by_split, matched_random_by_split)


if __name__ == "__main__":
    main()
