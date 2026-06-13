from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import replace
from itertools import product as cartesian_product
from pathlib import Path

import numpy as np
import pandas as pd

from gym_trading_env.research.entry_analysis import (
    constrained_oracle,
    fixed_direction_strategy,
    make_walk_forward_folds,
    matched_random_strategies,
    run_ridge_walk_forward,
    simple_rule_actions,
    simulate_strategy,
    strategy_metrics,
)
from gym_trading_env.research.entry_evaluator import (
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
    oracle = summary["gate1"]["oracle"]
    model = summary["gate2"]["aggregate_full"]
    cost = summary["gate3"]["cost_stress"]
    lines = [
        "# Entry Capability Report",
        "",
        f"- Dataset: `{summary['version']}`",
        f"- Gate 1: **{summary['gate1']['status']}**",
        f"- Gate 2 (Ridge window-summary baseline): **{summary['gate2']['status']}**",
        f"- Gate 3: **{summary['gate3']['status']}**",
        "",
        "## Constrained Oracle",
        "",
        f"- Trades: {oracle['trades']}",
        f"- Net PnL: {oracle['net_pnl']:.2f}",
        f"- Expectancy: {oracle['expectancy']:.2f}",
        f"- Profit factor: {oracle['profit_factor']}",
        "",
        "## Ridge Walk-forward",
        "",
        f"- Trades: {model['trades']}",
        f"- Net PnL: {model['net_pnl']:.2f}",
        f"- Expectancy: {model['expectancy']}",
        f"- No-trade day ratio: {model['no_trade_day_ratio']}",
        f"- Daily-bootstrap expectancy 95% CI: "
        f"[{model['daily_bootstrap_expectancy']['lower_95']}, "
        f"{model['daily_bootstrap_expectancy']['upper_95']}]",
        "",
        "## Cost Stress",
        "",
    ]
    for multiplier, metrics in cost.items():
        lines.append(
            f"- {multiplier}x cost: net PnL {metrics['net_pnl']:.2f}, "
            f"expectancy {metrics['expectancy']}")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This report evaluates the frozen Ridge window-summary baseline only. "
        "A Ridge FAIL means Gate 2 is not passed; it does not prove that every "
        "possible supervised model must fail.",
        "",
        "Gate 2 PASS is required before any later-stage agent experiment.",
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

    candidates.to_csv(out / "candidates.csv", index=False)
    np.savez_compressed(out / "features.npz", X=X, feature_names=np.asarray(names, dtype=object))

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
    model_report, model_trades, model_predictions = run_ridge_walk_forward(
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
    _write_json(out / "model_report.json", model_report)
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
    if model_report["gate2_status"] != "PASS":
        gate3_status = "BLOCKED_BY_GATE2"
    elif not sensitivity:
        gate3_status = "INCOMPLETE"
    elif not any(cost_positive):
        gate3_status = "FAIL"
    else:
        gate3_status = "INCOMPLETE_MODEL_PARAMETER_SENSITIVITY"

    strategy_report = {
        "version": config.version,
        "gate1": {
            "status": _gate1_status(baselines["oracle"]),
            "oracle": baselines["oracle"],
        },
        "gate2": {
            "model": model_report["model"],
            "feature_transform": model_report["feature_transform"],
            "status": model_report["gate2_status"],
            "aggregate_full": model_report["aggregate_full"],
            "aggregate_variants": model_report["aggregate_variants"],
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
        "observation_shape": [config.data.window_size, len(names) // 4],
        "feature_transform": model_report["feature_transform"],
        "feature_count": int(X.shape[1]),
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
        },
    }, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
