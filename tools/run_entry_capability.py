from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from itertools import product as cartesian_product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gym_trading_env.research.entry_analysis import (
    add_diagnostic_context,
    constrained_oracle,
    fixed_direction_strategy,
    make_walk_forward_folds,
    matched_random_strategies,
    random_action_strategies,
    run_ridge_walk_forward,
    run_xgboost_walk_forward,
    simple_rule_actions,
    split_oracle_report,
    simulate_strategy,
    strategy_metrics,
    summarize_random_runs,
)
from gym_trading_env.research.entry_dataset import outcomes_from_candidates, write_parquet
from gym_trading_env.research.entry_evaluator import (
    build_flattened_entry_features,
    build_entry_dataset,
    load_entry_eval_config,
    load_market_frames,
)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default))


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _gate1_status(oracle_metrics: dict) -> str:
    if oracle_metrics["net_pnl"] <= 0:
        return "FAIL"
    if oracle_metrics["trades"] < 30:
        return "INSUFFICIENT"
    if (
        oracle_metrics["max_trade_profit_share"] is not None
        and oracle_metrics["max_trade_profit_share"] > 0.5
    ):
        return "FAIL"
    return "PASS"


def _parameter_sensitivity(market, config) -> list[dict]:
    rows = []
    for ticks, rr, hold in cartesian_product(
        config.sensitivity.stop_distance_ticks,
        config.sensitivity.take_profit_rr,
        config.sensitivity.max_hold_bars,
    ):
        rules = replace(
            config.entry_evaluator,
            stop_distance_ticks=int(ticks),
            stop_distance_price=float(ticks) * config.product.tick_size,
            take_profit_rr=float(rr),
            max_hold_bars=int(hold),
        )
        variant = replace(config, entry_evaluator=rules)
        candidates, _, _ = build_entry_dataset(market, variant)
        oracle = constrained_oracle(
            candidates, max_entries_per_day=config.product.max_entries_per_day)
        metrics = strategy_metrics(
            oracle,
            all_days=candidates["trading_day"].unique(),
            initial_balance=config.product.initial_balance,
        )
        rows.append({
            "scope": "constrained_oracle",
            "stop_distance_ticks": int(ticks),
            "stop_distance_price": float(rules.stop_distance_price),
            "take_profit_rr": float(rr),
            "max_hold_bars": int(hold),
            "is_canonical": bool(rules == config.entry_evaluator),
            **metrics,
        })
    return rows


def _write_markdown_report(path: Path, summary: dict) -> None:
    gate0 = summary["gate0"]
    oracle = summary["gate1"]["oracle"]
    model = summary["gate2"]["aggregate_full"]
    cost = summary["gate3"]["cost_stress"]
    ridge = summary["gate2"].get("ridge_baseline", {})
    comparison = summary["gate2"].get("feature_encoding_comparison", {})
    decile = model.get("score_decile_executable", {})
    alignment = model.get("alignment_audit", {})
    lines = [
        "# Entry Capability Report",
        "",
        f"- Dataset: `{summary['version']}`",
        f"- Gate 0 (frozen contract): **{gate0['status']}**",
        f"- Gate 1: **{summary['gate1']['status']}**",
        f"- Gate 2 (XGBoost Q baseline): **{summary['gate2']['status']}**",
        f"- Gate 3: **{summary['gate3']['status']}**",
        f"- Primary feature encoding: `{summary['gate2'].get('primary_feature_encoding')}`",
        "",
        "## Gate 0 Contract",
        "",
        f"- Entry delay: {gate0['contract']['entry_delay_bars']} valid 1m bar",
        f"- Stop distance: {gate0['contract']['stop_distance_price']} price "
        f"points / {gate0['contract']['stop_distance_ticks']} ticks",
        f"- Take profit RR: {gate0['contract']['take_profit_rr']}",
        f"- Max hold: {gate0['contract']['max_hold_bars']} "
        f"{gate0['contract']['max_hold_unit']}",
        f"- Across-break entries: {gate0['contract']['allow_entry_across_break']}",
        f"- Force flatten EOD: {gate0['contract']['force_flatten_eod']}",
        f"- Intrabar collision: {gate0['contract']['intrabar_collision']}",
        "",
        "## Constrained Oracle",
        "",
        f"- Trades: {oracle['trades']}",
        f"- Net PnL: {oracle['net_pnl']:.2f}",
        f"- Expectancy: {oracle['expectancy']:.2f}",
        f"- Profit factor: {oracle['profit_factor']}",
        "",
        "## XGBoost Q Walk-forward",
        "",
        f"- Trades: {model['trades']}",
        f"- Net PnL: {model['net_pnl']:.2f}",
        f"- Expectancy: {model['expectancy']}",
        f"- No-trade day ratio: {model['no_trade_day_ratio']}",
        f"- Daily-bootstrap expectancy 95% CI: "
        f"[{model['daily_bootstrap_expectancy']['lower_95']}, "
        f"{model['daily_bootstrap_expectancy']['upper_95']}]",
        "",
        "## Ridge Diagnostic",
        "",
        f"- Status: {ridge.get('gate2_status')}",
        f"- Net PnL: {ridge.get('net_pnl')}",
        f"- Expectancy: {ridge.get('expectancy')}",
        "",
        "## XGBoost Diagnostics",
        "",
        f"- Matched-random mean net PnL: "
        f"{model.get('matched_random', {}).get('net_pnl_mean')}",
        f"- Score decile top>bottom: "
        f"{decile.get('top_above_bottom')}",
        f"- Score decile spearman: "
        f"{decile.get('spearman')}",
        f"- Alignment audit: {alignment.get('status')}",
        f"- Action mapping audit: {alignment.get('action_mapping', {}).get('status')}",
        "",
        "### Feature Encoding Comparison",
        "",
    ]
    for name, item in comparison.items():
        if "gate2_status" in item:
            lines.append(
                f"- {name}: {item['gate2_status']}, trades {item['trades']}, "
                f"net PnL {item['net_pnl']:.2f}, expectancy {item['expectancy']}, "
                f"decile spearman {item.get('score_decile_spearman')}")
        else:
            lines.append(f"- {name}: {item.get('status')} ({item.get('reason')})")
    lines.extend([
        "",
        "### Top Selective Entry",
        "",
    ])
    for fraction, item in model.get("top_selective_entry", {}).items():
        metrics = item["metrics"]
        lines.append(
            f"- Top {float(fraction) * 100:.0f}%: trades {metrics['trades']}, "
            f"net PnL {metrics['net_pnl']:.2f}, expectancy {metrics['expectancy']}")
    lines.extend([
        "",
        "### Bottom Selective Entry (Debug Only)",
        "",
    ])
    for fraction, item in model.get("bottom_selective_entry", {}).items():
        metrics = item["metrics"]
        lines.append(
            f"- Bottom {float(fraction) * 100:.0f}%: trades {metrics['trades']}, "
            f"net PnL {metrics['net_pnl']:.2f}, expectancy {metrics['expectancy']}")
    inverted = model.get("inverted_score_sanity", {}).get("metrics", {})
    lines.extend([
        "",
        "### Inverted Score Sanity (Debug Only)",
        "",
        f"- Trades: {inverted.get('trades')}",
        f"- Net PnL: {inverted.get('net_pnl')}",
        f"- Expectancy: {inverted.get('expectancy')}",
        "",
        "### Full Score Decile Table",
        "",
        "| Decile | Candidates | Exec Trades | Mean Pred Q | Realized PnL | Realized Exp | LONG | SHORT | Exec PnL | Exec Exp |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in decile.get("bins", []):
        metrics = row["executable_metrics"]
        lines.append(
            f"| {row['decile']} | {row['candidate_count']} | "
            f"{row['executable_strategy_trades']} | {row['mean_pred_q']:.4f} | "
            f"{row['realized_net_pnl']:.2f} | {row['realized_expectancy']:.4f} | "
            f"{row['long_count']} | {row['short_count']} | "
            f"{metrics['net_pnl']:.2f} | {metrics['expectancy']} |")
    lines.extend([
        "",
        "### Alignment Audit",
        "",
        f"- Best-direction mismatches: {alignment.get('best_direction_mismatch_count')}",
        f"- Score mismatches: {alignment.get('score_mismatch_count')}",
        f"- Realized label mismatches: {alignment.get('realized_direction_label_mismatch_count')}",
        f"- Threshold action mismatches: {alignment.get('threshold_action_mismatch_count')}",
    ])
    for direction, item in alignment.get("selected_trade_alignment", {}).items():
        lines.append(
            f"- Selected {direction}: count {item['count']}, "
            f"mean pred Q {item['mean_pred_q']}, mean realized {item['mean_realized_net_pnl']}, "
            f"corr {item['pred_realized_corr']}, label mismatches "
            f"{item['strategy_net_matches_label_mismatches']}")
    lines.extend([
        "",
        "### Threshold Curve",
        "",
    ])
    for item in model.get("threshold_curve", []):
        metrics = item["metrics"]
        lines.append(
            f"- threshold {item['threshold']:.6g}, coverage "
            f"{item['candidate_coverage']:.3f}: trades {metrics['trades']}, "
            f"net PnL {metrics['net_pnl']:.2f}, expectancy {metrics['expectancy']}")
    lines.extend([
        "",
        "### Grouped Net PnL",
        "",
    ])
    grouped = model.get("grouped", {})
    for group_name in ("by_fold", "by_month", "by_contract", "by_direction", "by_session_phase"):
        values = grouped.get(group_name, {})
        if not values:
            continue
        compact = ", ".join(
            f"{key}: {metrics['net_pnl']:.2f}/{metrics['trades']}"
            for key, metrics in values.items()
        )
        lines.append(f"- {group_name}: {compact}")
    lines.extend([
        "",
        "### Split Oracle",
        "",
    ])
    for fold_name, fold_report in summary["gate2"].get("split_oracle", {}).items():
        split_parts = []
        for split, split_report in fold_report.get("splits", {}).items():
            metrics = split_report["oracle"]
            split_parts.append(
                f"{split}: {metrics['net_pnl']:.2f}/{metrics['trades']}")
        lines.append(f"- {fold_name}: " + ", ".join(split_parts))
    lines.extend([
        "",
        "## Cost Stress",
        "",
    ])
    for multiplier, metrics in cost.items():
        lines.append(
            f"- {multiplier}x cost: net PnL {metrics['net_pnl']:.2f}, "
            f"expectancy {metrics['expectancy']}")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This report keeps the same dataset, outcomes, split, cost model, and "
        "fixed exit rules. The final decision baseline is XGBoost Q_long/Q_short; "
        "classifier outputs are diagnostic only.",
        "",
        "If XGBoost Gate 2 is FAIL, the current 18-dimensional observation has "
        "not proved out-of-sample entry selection ability; the next step is "
        "single-feature ablation, not RL.",
        "",
    ])
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/entry_eval_jm_v1.yaml")
    parser.add_argument("--output", default="artifacts/entry_eval/entry_eval_jm_v1")
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = load_entry_eval_config(config_path)
    _, market = load_market_frames(config)
    candidates, X, names = build_entry_dataset(market, config)
    candidates = add_diagnostic_context(candidates, X, names)
    X_flat, flat_names = build_flattened_entry_features(market, candidates, config)
    outcomes = outcomes_from_candidates(candidates)

    shutil.copyfile(config_path, out / "config.yaml")
    write_parquet(candidates, out / "candidates.parquet")
    write_parquet(outcomes, out / "outcomes.parquet")
    candidates.to_csv(out / "candidates.csv", index=False)
    np.savez_compressed(out / "features.npz", X=X, feature_names=np.asarray(names, dtype=object))
    np.savez_compressed(
        out / "flat_features.npz",
        X=X_flat,
        feature_names=np.asarray(flat_names, dtype=object),
    )

    days = candidates["trading_day"].unique()
    product = config.product
    max_entries = product.max_entries_per_day
    oracle = constrained_oracle(candidates, max_entries_per_day=max_entries)
    long_only = fixed_direction_strategy(candidates, "LONG", max_entries_per_day=max_entries)
    short_only = fixed_direction_strategy(candidates, "SHORT", max_entries_per_day=max_entries)

    baselines = {
        "oracle": strategy_metrics(oracle, all_days=days, initial_balance=product.initial_balance),
        "long_only": strategy_metrics(long_only, all_days=days, initial_balance=product.initial_balance),
        "short_only": strategy_metrics(short_only, all_days=days, initial_balance=product.initial_balance),
    }
    for name, feature in {
        "vwap_direction": "latest:obs_cmp_C_vs_cumVWAP_t",
        "open_drift_direction": "latest:obs_open_drift_t",
        "recent_ref_return": "delta:obs_pct_chg_from_ref_t",
    }.items():
        actions, scores = simple_rule_actions(X, names, feature)
        trades = simulate_strategy(
            candidates, actions, scores, max_entries_per_day=max_entries)
        baselines[name] = strategy_metrics(
            trades, all_days=days, initial_balance=product.initial_balance)

    random_runs = matched_random_strategies(
        candidates,
        oracle,
        max_entries_per_day=max_entries,
        runs=config.experiment.random_baseline_runs,
        seed=config.experiment.random_seed,
    )
    random_metrics = [
        strategy_metrics(x, all_days=days, initial_balance=product.initial_balance)
        for x in random_runs
    ]
    random_pnl = np.asarray([x["net_pnl"] for x in random_metrics], dtype=float)
    baselines["oracle_trade_count_matched_random"] = {
        "runs": len(random_metrics),
        "target_trades": int(len(oracle)),
        "trade_count_min": int(min(x["trades"] for x in random_metrics)),
        "trade_count_max": int(max(x["trades"] for x in random_metrics)),
        "net_pnl_mean": float(random_pnl.mean()),
        "net_pnl_p05": float(np.quantile(random_pnl, 0.05)),
        "net_pnl_p50": float(np.quantile(random_pnl, 0.50)),
        "net_pnl_p95": float(np.quantile(random_pnl, 0.95)),
    }
    baselines["random_long_short_flat"] = summarize_random_runs(
        random_action_strategies(
            candidates,
            max_entries_per_day=max_entries,
            runs=config.experiment.random_baseline_runs,
            seed=config.experiment.random_seed,
        ),
        all_days=days,
        initial_balance=product.initial_balance,
    )

    opportunity = {
        "candidates": int(len(candidates)),
        "trading_days": int(len(days)),
        "long_positive_rate": float((candidates["long_net_pnl"] > 0).mean()),
        "short_positive_rate": float((candidates["short_net_pnl"] > 0).mean()),
        "best_direction_positive_rate": float(
            (candidates[["long_net_pnl", "short_net_pnl"]].max(axis=1) > 0).mean()),
        "collision_rate": float(candidates["stop_target_collision"].mean()),
        "long_exit_reasons": candidates["long_exit_reason"].value_counts().to_dict(),
        "short_exit_reasons": candidates["short_exit_reason"].value_counts().to_dict(),
        "baselines": baselines,
    }
    _write_json(out / "opportunity_report.json", opportunity)
    oracle.to_csv(out / "oracle_trades.csv", index=False)

    folds = make_walk_forward_folds(days)
    _write_json(out / "split_manifest.json", folds)
    split_oracle = split_oracle_report(
        candidates,
        folds,
        max_entries_per_day=max_entries,
        initial_balance=product.initial_balance,
        embargo_bars=max(config.data.window_size, config.entry_evaluator.max_hold_bars),
    )
    _write_json(out / "split_oracle_report.json", split_oracle)
    ridge_report, ridge_trades, ridge_predictions = run_ridge_walk_forward(
        candidates,
        X,
        names,
        folds,
        alpha=config.experiment.ridge_alpha,
        threshold_quantiles=config.experiment.threshold_quantiles,
        max_entries_per_day=max_entries,
        initial_balance=product.initial_balance,
        embargo_bars=max(config.data.window_size, config.entry_evaluator.max_hold_bars),
        bootstrap_runs=config.experiment.bootstrap_runs,
        random_seed=config.experiment.random_seed,
        cost_multipliers=config.sensitivity.cost_multipliers,
    )
    _write_json(out / "ridge_model_report.json", ridge_report)
    write_parquet(ridge_trades, out / "ridge_strategy_trades.parquet")
    write_parquet(ridge_predictions, out / "ridge_baseline_predictions.parquet")

    summary_report, summary_trades, summary_predictions = run_xgboost_walk_forward(
        candidates,
        X,
        names,
        folds,
        threshold_quantiles=config.experiment.threshold_quantiles,
        max_entries_per_day=max_entries,
        initial_balance=product.initial_balance,
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
        max_entries_per_day=max_entries,
        initial_balance=product.initial_balance,
        embargo_bars=max(config.data.window_size, config.entry_evaluator.max_hold_bars),
        bootstrap_runs=config.experiment.bootstrap_runs,
        random_seed=config.experiment.random_seed,
        cost_multipliers=config.sensitivity.cost_multipliers,
        random_baseline_runs=config.experiment.random_baseline_runs,
        feature_encoding="flattened_60x18_market_seq",
        variants=("full",),
    )
    _write_json(out / "xgboost_window_summary_report.json", summary_report)
    _write_json(out / "xgboost_flattened_report.json", flat_report)
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
    _write_json(out / "model_report.json", model_report)
    write_parquet(model_trades, out / "strategy_trades.parquet")
    write_parquet(model_predictions, out / "baseline_predictions.parquet")
    model_trades.to_csv(out / "strategy_trades.csv", index=False)
    model_predictions.to_csv(out / "baseline_predictions.csv", index=False)

    if args.skip_sensitivity:
        sensitivity = []
    else:
        sensitivity = _parameter_sensitivity(market, config)
    pd.DataFrame.from_records(sensitivity).to_csv(
        out / "parameter_sensitivity.csv", index=False)

    cost_stress = model_report["aggregate_full"]["cost_stress"]
    cost_positive = [
        metrics["net_pnl"] > 0
        for multiplier, metrics in cost_stress.items()
        if float(multiplier) > 1.0
    ]
    if model_report["gate2_status"] == "FAIL":
        gate3_status = "BLOCKED_BY_GATE2"
    elif not sensitivity:
        gate3_status = "INCOMPLETE"
    elif not all(cost_positive):
        gate3_status = "FAIL"
    elif all(row["net_pnl"] > 0 for row in sensitivity):
        gate3_status = "PASS" if model_report["gate2_status"] == "PASS" else "PROMISING"
    else:
        gate3_status = "INCOMPLETE_MODEL_PARAMETER_SENSITIVITY"

    strategy_report = {
        "version": config.version,
        "gate0": {
            "status": "PASS",
            "contract": {
                "entry_delay_bars": config.entry_evaluator.entry_delay_bars,
                "stop_distance_price": config.entry_evaluator.stop_distance_price,
                "stop_distance_ticks": config.entry_evaluator.stop_distance_ticks,
                "take_profit_rr": config.entry_evaluator.take_profit_rr,
                "max_hold_bars": config.entry_evaluator.max_hold_bars,
                "max_hold_unit": config.entry_evaluator.max_hold_unit,
                "allow_entry_across_break": config.entry_evaluator.allow_entry_across_break,
                "force_flatten_eod": config.entry_evaluator.force_flatten_eod,
                "intrabar_collision": config.entry_evaluator.intrabar_collision,
            },
        },
        "gate1": {
            "status": _gate1_status(baselines["oracle"]),
            "oracle": baselines["oracle"],
        },
        "gate2": {
            "model": model_report["model"],
            "feature_transform": model_report["feature_transform"],
            "primary_feature_encoding": model_report["primary_feature_encoding"],
            "status": model_report["gate2_status"],
            "aggregate_full": model_report["aggregate_full"],
            "aggregate_variants": model_report["aggregate_variants"],
            "feature_encoding_comparison": model_report["feature_encoding_comparison"],
            "split_oracle": split_oracle,
            "ridge_baseline": {
                "gate2_status": ridge_report["gate2_status"],
                "net_pnl": ridge_report["aggregate_full"]["net_pnl"],
                "expectancy": ridge_report["aggregate_full"]["expectancy"],
                "trades": ridge_report["aggregate_full"]["trades"],
            },
        },
        "gate3": {
            "status": gate3_status,
            "cost_stress": cost_stress,
            "oracle_sensitivity_combinations": len(sensitivity),
            "positive_oracle_sensitivity_combinations": sum(
                row["net_pnl"] > 0 for row in sensitivity),
        },
    }
    _write_json(out / "strategy_report.json", strategy_report)
    _write_markdown_report(out / "report.md", strategy_report)

    manifest = {
        "version": config.version,
        "git_commit": _git_commit(),
        "config": config.to_dict(),
        "config_sha256": _file_hash(config_path),
        "data_sha256": _file_hash(Path(config.data.path)),
        "rows": int(len(market)),
        "candidates": int(len(candidates)),
        "outcomes": int(len(outcomes)),
        "observation_shape": [config.data.window_size, len(names) // 4],
        "feature_transform": model_report["feature_transform"],
        "feature_count": int(X.shape[1]),
        "flat_feature_transform": "flattened_60x18_market_seq",
        "flat_feature_count": int(X_flat.shape[1]),
        "primary_model": model_report["model"],
        "primary_feature_encoding": primary_encoding,
        "auxiliary_model": ridge_report["model"],
        "sensitivity_skipped": bool(args.skip_sensitivity),
    }
    _write_json(out / "manifest.json", manifest)

    print(json.dumps({
        "output": str(out),
        "opportunity": opportunity,
        "model": {
            "aggregate_full": model_report["aggregate_full"],
            "gate2_status": model_report["gate2_status"],
            "gate3_status": gate3_status,
            "ridge_gate2_status": ridge_report["gate2_status"],
            "primary_feature_encoding": primary_encoding,
            "feature_encoding_comparison": model_report["feature_encoding_comparison"],
        },
    }, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
