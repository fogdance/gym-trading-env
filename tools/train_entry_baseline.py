from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gym_trading_env.research.entry_analysis import (
    add_diagnostic_context,
    run_ridge_walk_forward,
    run_xgboost_walk_forward,
)
from gym_trading_env.research.entry_dataset import (
    load_flattened_dataset_artifacts,
    load_dataset_artifacts,
    write_json,
    write_parquet,
    json_default,
)
from gym_trading_env.research.entry_evaluator import load_entry_eval_config
from gym_trading_env.research.walk_forward_split_builder import load_required_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    out = Path(args.output or args.input)
    out.mkdir(parents=True, exist_ok=True)
    config = load_entry_eval_config(args.config)
    candidates, X, names = load_dataset_artifacts(args.input)
    candidates = add_diagnostic_context(candidates, X, names)
    X_flat, flat_names = load_flattened_dataset_artifacts(args.input)
    folds = load_required_split_manifest(args.config)
    ridge_report, ridge_trades, ridge_predictions = run_ridge_walk_forward(
        candidates,
        X,
        names,
        folds,
        alpha=config.experiment.ridge_alpha,
        threshold_quantiles=config.experiment.threshold_quantiles,
        max_entries_per_day=config.product.max_entries_per_day,
        initial_balance=config.product.initial_balance,
        embargo_bars=max(config.data.window_size, config.entry_evaluator.max_hold_bars),
        bootstrap_runs=config.experiment.bootstrap_runs,
        random_seed=config.experiment.random_seed,
        cost_multipliers=config.sensitivity.cost_multipliers,
    )
    write_json(out / "ridge_model_report.json", ridge_report)
    write_parquet(ridge_predictions, out / "ridge_baseline_predictions.parquet")
    write_parquet(ridge_trades, out / "ridge_strategy_trades.parquet")

    summary_report, summary_trades, summary_predictions = run_xgboost_walk_forward(
        candidates,
        X,
        names,
        folds,
        threshold_quantiles=config.experiment.threshold_quantiles,
        max_entries_per_day=config.product.max_entries_per_day,
        initial_balance=config.product.initial_balance,
        embargo_bars=max(config.data.window_size, config.entry_evaluator.max_hold_bars),
        bootstrap_runs=config.experiment.bootstrap_runs,
        random_seed=config.experiment.random_seed,
        cost_multipliers=config.sensitivity.cost_multipliers,
        random_baseline_runs=config.experiment.random_baseline_runs,
        feature_encoding="window_summary_latest_mean_std_delta",
        variants=("full", "no_time", "time_only"),
    )
    flat_report, flat_trades, flat_predictions = run_xgboost_walk_forward(
        candidates,
        X_flat,
        flat_names,
        folds,
        threshold_quantiles=config.experiment.threshold_quantiles,
        max_entries_per_day=config.product.max_entries_per_day,
        initial_balance=config.product.initial_balance,
        embargo_bars=max(config.data.window_size, config.entry_evaluator.max_hold_bars),
        bootstrap_runs=config.experiment.bootstrap_runs,
        random_seed=config.experiment.random_seed,
        cost_multipliers=config.sensitivity.cost_multipliers,
        random_baseline_runs=config.experiment.random_baseline_runs,
        feature_encoding="flattened_60x18_market_seq",
        variants=("full",),
    )
    write_json(out / "xgboost_window_summary_report.json", summary_report)
    write_json(out / "xgboost_flattened_report.json", flat_report)
    write_parquet(summary_predictions, out / "xgboost_window_summary_predictions.parquet")
    write_parquet(summary_trades, out / "xgboost_window_summary_trades.parquet")
    write_parquet(flat_predictions, out / "xgboost_flattened_predictions.parquet")
    write_parquet(flat_trades, out / "xgboost_flattened_trades.parquet")

    def gate_rank(report: dict) -> tuple[int, float]:
        rank = {"FAIL": 0, "PROMISING": 1, "PASS": 2}.get(report["gate2_status"], -1)
        return rank, float(report["aggregate_full"]["net_pnl"])

    if gate_rank(flat_report) > gate_rank(summary_report):
        model_report, model_trades, model_predictions = flat_report, flat_trades, flat_predictions
        primary_encoding = "flattened_60x18_market_seq"
    else:
        model_report, model_trades, model_predictions = summary_report, summary_trades, summary_predictions
        primary_encoding = "window_summary_latest_mean_std_delta"
    model_report["primary_feature_encoding"] = primary_encoding
    model_report["feature_encoding_comparison"] = {
        "window_summary_latest_mean_std_delta": {
            "gate2_status": summary_report["gate2_status"],
            "trades": summary_report["aggregate_full"]["trades"],
            "net_pnl": summary_report["aggregate_full"]["net_pnl"],
            "expectancy": summary_report["aggregate_full"]["expectancy"],
            "score_decile_spearman": summary_report["aggregate_variants"]["full"]["score_decile_executable"]["spearman"],
        },
        "flattened_60x18_market_seq": {
            "gate2_status": flat_report["gate2_status"],
            "trades": flat_report["aggregate_full"]["trades"],
            "net_pnl": flat_report["aggregate_full"]["net_pnl"],
            "expectancy": flat_report["aggregate_full"]["expectancy"],
            "score_decile_spearman": flat_report["aggregate_variants"]["full"]["score_decile_executable"]["spearman"],
        },
        "small_temporal_model": {
            "status": "not_implemented",
            "reason": "Not added in this audit pass; scope limited to XGBoost feature encodings.",
        },
    }
    write_json(out / "model_report.json", model_report)
    write_parquet(model_predictions, out / "baseline_predictions.parquet")
    write_parquet(model_trades, out / "strategy_trades.parquet")
    print(json.dumps({
        "output": str(out),
        "model": model_report["model"],
        "gate2_status": model_report["gate2_status"],
        "trades": int(model_report["aggregate_full"]["trades"]),
        "net_pnl": float(model_report["aggregate_full"]["net_pnl"]),
        "ridge_gate2_status": ridge_report["gate2_status"],
        "primary_feature_encoding": primary_encoding,
    }, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
