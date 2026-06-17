from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import product as cartesian_product
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

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
    simulate_strategy,
    split_oracle_report,
    strategy_metrics,
    summarize_random_runs,
)
from gym_trading_env.research.entry_dataset import outcomes_from_candidates, write_parquet
from gym_trading_env.research.entry_evaluator import (
    build_entry_dataset,
    build_flattened_entry_features,
    load_entry_eval_config,
    load_market_frames,
)


StageLogger = Callable[[str], None]


@dataclass
class DatasetStage:
    config_path: Path
    output: Path
    config: object
    market: pd.DataFrame
    candidates: pd.DataFrame
    features: np.ndarray
    feature_names: list[str]
    flat_features: np.ndarray
    flat_feature_names: list[str]
    outcomes: pd.DataFrame
    folds: list[dict]
    manifest: dict


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def stage_log(message: str) -> None:
    print(f"[run_entry_capability] {message}", file=sys.stderr, flush=True)


def write_text_atomic(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json_atomic(path: str | Path, value) -> None:
    write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default),
    )


def write_csv_atomic(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_npz_atomic(path: str | Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_pipeline_status(
    output: str | Path,
    *,
    stage: str,
    state: str,
    last_completed_stage: str | None = None,
    extra: dict | None = None,
) -> None:
    payload = {
        "stage": stage,
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "last_completed_stage": last_completed_stage,
    }
    if extra:
        payload.update(extra)
    write_json_atomic(Path(output) / "pipeline_status.json", payload)


def _build_dataset_manifest(
    config_path: Path,
    config,
    market: pd.DataFrame,
    candidates: pd.DataFrame,
    outcomes: pd.DataFrame,
    features: np.ndarray,
    feature_names: list[str],
    flat_features: np.ndarray,
    *,
    stage: str,
) -> dict:
    return {
        "version": config.version,
        "git_commit": git_commit(),
        "config": config.to_dict(),
        "config_sha256": file_hash(config_path),
        "data_sha256": file_hash(Path(config.data.path)),
        "rows": int(len(market)),
        "candidates": int(len(candidates)),
        "outcomes": int(len(outcomes)),
        "observation_shape": [config.data.window_size, len(feature_names) // 4],
        "feature_transform": "latest_mean_std_delta_of_market_seq",
        "feature_count": int(features.shape[1]),
        "flat_feature_transform": "flattened_60x18_market_seq",
        "flat_feature_count": int(flat_features.shape[1]),
        "manifest_stage": stage,
    }


def build_dataset_stage(
    config_path: str | Path,
    output: str | Path,
    *,
    emit: StageLogger = stage_log,
) -> DatasetStage:
    config_path = Path(config_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    write_pipeline_status(output, stage="load_config", state="running")
    emit(f"loading config {config_path}")
    config = load_entry_eval_config(config_path)
    _, market = load_market_frames(config)
    write_pipeline_status(output, stage="load_config", state="completed", last_completed_stage="load_config")

    write_pipeline_status(output, stage="build_dataset", state="running", last_completed_stage="load_config")
    emit("building candidates and features")
    candidates, features, feature_names = build_entry_dataset(market, config)
    candidates = add_diagnostic_context(candidates, features, feature_names)
    flat_features, flat_feature_names = build_flattened_entry_features(market, candidates, config)
    outcomes = outcomes_from_candidates(candidates)
    folds = make_walk_forward_folds(candidates["trading_day"].unique())

    manifest = _build_dataset_manifest(
        config_path,
        config,
        market,
        candidates,
        outcomes,
        features,
        feature_names,
        flat_features,
        stage="dataset_ready",
    )
    stage = DatasetStage(
        config_path=config_path,
        output=output,
        config=config,
        market=market,
        candidates=candidates,
        features=features,
        feature_names=feature_names,
        flat_features=flat_features,
        flat_feature_names=flat_feature_names,
        outcomes=outcomes,
        folds=folds,
        manifest=manifest,
    )
    write_dataset_artifacts(stage, emit=emit)
    write_pipeline_status(
        output,
        stage="build_dataset",
        state="completed",
        last_completed_stage="dataset_ready",
        extra={
            "manifest_stage": "dataset_ready",
            "candidates": int(len(candidates)),
            "outcomes": int(len(outcomes)),
        },
    )
    return stage


def write_dataset_artifacts(stage: DatasetStage, *, emit: StageLogger = stage_log) -> None:
    out = stage.output
    emit(f"writing dataset artifacts to {out}")
    shutil.copyfile(stage.config_path, out / "config.yaml")
    write_parquet(stage.candidates, out / "candidates.parquet")
    write_parquet(stage.outcomes, out / "outcomes.parquet")
    write_csv_atomic(stage.candidates, out / "candidates.csv")
    write_npz_atomic(
        out / "features.npz",
        X=stage.features,
        feature_names=np.asarray(stage.feature_names, dtype=object),
    )
    write_npz_atomic(
        out / "flat_features.npz",
        X=stage.flat_features,
        feature_names=np.asarray(stage.flat_feature_names, dtype=object),
    )
    write_json_atomic(out / "split_manifest.json", stage.folds)
    write_json_atomic(out / "manifest.json", stage.manifest)
    emit(
        f"dataset_ready candidates={len(stage.candidates)} outcomes={len(stage.outcomes)} "
        f"feature_count={stage.features.shape[1]}")


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


def _parameter_sensitivity(stage: DatasetStage) -> list[dict]:
    config = stage.config
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
        candidates, _, _ = build_entry_dataset(stage.market, variant)
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


def run_opportunity_stage(stage: DatasetStage, *, emit: StageLogger = stage_log) -> tuple[dict, pd.DataFrame, dict]:
    write_pipeline_status(stage.output, stage="opportunity", state="running", last_completed_stage="dataset_ready")
    emit("running opportunity baselines")
    candidates = stage.candidates
    config = stage.config
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
        actions, scores = simple_rule_actions(stage.features, stage.feature_names, feature)
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
    write_json_atomic(stage.output / "opportunity_report.json", opportunity)
    write_csv_atomic(oracle, stage.output / "oracle_trades.csv")
    write_pipeline_status(stage.output, stage="opportunity", state="completed", last_completed_stage="opportunity")
    return opportunity, oracle, baselines


def run_model_stages(
    stage: DatasetStage,
    *,
    skip_sensitivity: bool,
    baselines: dict,
    emit: StageLogger = stage_log,
) -> dict:
    candidates = stage.candidates
    config = stage.config
    out = stage.output
    max_entries = config.product.max_entries_per_day

    write_pipeline_status(out, stage="split_oracle", state="running", last_completed_stage="opportunity")
    split_oracle = split_oracle_report(
        candidates,
        stage.folds,
        max_entries_per_day=max_entries,
        initial_balance=config.product.initial_balance,
        embargo_bars=max(config.data.window_size, config.entry_evaluator.max_hold_bars),
    )
    write_json_atomic(out / "split_oracle_report.json", split_oracle)
    write_pipeline_status(out, stage="split_oracle", state="completed", last_completed_stage="split_oracle")

    write_pipeline_status(out, stage="ridge", state="running", last_completed_stage="split_oracle")
    emit("running Ridge walk-forward baseline")
    ridge_report, ridge_trades, ridge_predictions = run_ridge_walk_forward(
        candidates,
        stage.features,
        stage.feature_names,
        stage.folds,
        alpha=config.experiment.ridge_alpha,
        threshold_quantiles=config.experiment.threshold_quantiles,
        max_entries_per_day=max_entries,
        initial_balance=config.product.initial_balance,
        embargo_bars=max(config.data.window_size, config.entry_evaluator.max_hold_bars),
        bootstrap_runs=config.experiment.bootstrap_runs,
        random_seed=config.experiment.random_seed,
        cost_multipliers=config.sensitivity.cost_multipliers,
    )
    write_json_atomic(out / "ridge_model_report.json", ridge_report)
    write_parquet(ridge_trades, out / "ridge_strategy_trades.parquet")
    write_parquet(ridge_predictions, out / "ridge_baseline_predictions.parquet")
    write_pipeline_status(out, stage="ridge", state="completed", last_completed_stage="ridge")

    write_pipeline_status(out, stage="xgboost_window_summary", state="running", last_completed_stage="ridge")
    emit("running XGBoost window-summary walk-forward baseline")
    summary_report, summary_trades, summary_predictions = run_xgboost_walk_forward(
        candidates,
        stage.features,
        stage.feature_names,
        stage.folds,
        threshold_quantiles=config.experiment.threshold_quantiles,
        max_entries_per_day=max_entries,
        initial_balance=config.product.initial_balance,
        embargo_bars=max(config.data.window_size, config.entry_evaluator.max_hold_bars),
        bootstrap_runs=config.experiment.bootstrap_runs,
        random_seed=config.experiment.random_seed,
        cost_multipliers=config.sensitivity.cost_multipliers,
        random_baseline_runs=config.experiment.random_baseline_runs,
        feature_encoding="window_summary_latest_mean_std_delta",
        variants=("full", "no_time", "time_only"),
    )
    write_pipeline_status(
        out,
        stage="xgboost_window_summary",
        state="completed",
        last_completed_stage="xgboost_window_summary",
    )

    write_pipeline_status(out, stage="xgboost_flattened", state="running", last_completed_stage="xgboost_window_summary")
    emit("running XGBoost flattened walk-forward baseline")
    flat_report, flat_trades, flat_predictions = run_xgboost_walk_forward(
        candidates,
        stage.flat_features,
        stage.flat_feature_names,
        stage.folds,
        threshold_quantiles=config.experiment.threshold_quantiles,
        max_entries_per_day=max_entries,
        initial_balance=config.product.initial_balance,
        embargo_bars=max(config.data.window_size, config.entry_evaluator.max_hold_bars),
        bootstrap_runs=config.experiment.bootstrap_runs,
        random_seed=config.experiment.random_seed,
        cost_multipliers=config.sensitivity.cost_multipliers,
        random_baseline_runs=config.experiment.random_baseline_runs,
        feature_encoding="flattened_60x18_market_seq",
        variants=("full",),
    )
    write_json_atomic(out / "xgboost_window_summary_report.json", summary_report)
    write_json_atomic(out / "xgboost_flattened_report.json", flat_report)
    write_parquet(summary_predictions, out / "xgboost_window_summary_predictions.parquet")
    write_parquet(summary_trades, out / "xgboost_window_summary_trades.parquet")
    write_parquet(flat_predictions, out / "xgboost_flattened_predictions.parquet")
    write_parquet(flat_trades, out / "xgboost_flattened_trades.parquet")
    write_pipeline_status(out, stage="xgboost_flattened", state="completed", last_completed_stage="xgboost_flattened")

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
    write_json_atomic(out / "model_report.json", model_report)
    write_parquet(model_trades, out / "strategy_trades.parquet")
    write_parquet(model_predictions, out / "baseline_predictions.parquet")
    write_csv_atomic(model_trades, out / "strategy_trades.csv")
    write_csv_atomic(model_predictions, out / "baseline_predictions.csv")

    if skip_sensitivity:
        sensitivity = []
    else:
        write_pipeline_status(out, stage="sensitivity", state="running", last_completed_stage="xgboost_flattened")
        emit("running parameter sensitivity")
        sensitivity = _parameter_sensitivity(stage)
    write_csv_atomic(pd.DataFrame.from_records(sensitivity), out / "parameter_sensitivity.csv")
    if not skip_sensitivity:
        write_pipeline_status(out, stage="sensitivity", state="completed", last_completed_stage="sensitivity")

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
    write_json_atomic(out / "strategy_report.json", strategy_report)
    write_markdown_report(out / "report.md", strategy_report)

    manifest = {
        **stage.manifest,
        "manifest_stage": "model_complete",
        "feature_transform": model_report["feature_transform"],
        "primary_model": model_report["model"],
        "primary_feature_encoding": primary_encoding,
        "auxiliary_model": ridge_report["model"],
        "sensitivity_skipped": bool(skip_sensitivity),
    }
    write_json_atomic(out / "manifest.json", manifest)
    write_pipeline_status(out, stage="model_complete", state="completed", last_completed_stage="model_complete")
    emit("model_complete")

    return {
        "opportunity": json.loads((out / "opportunity_report.json").read_text()),
        "model_report": model_report,
        "ridge_report": ridge_report,
        "gate3_status": gate3_status,
        "primary_encoding": primary_encoding,
    }


def write_markdown_report(path: str | Path, summary: dict) -> None:
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
    lines.extend(["", "### Top Selective Entry", ""])
    for fraction, item in model.get("top_selective_entry", {}).items():
        metrics = item["metrics"]
        lines.append(
            f"- Top {float(fraction) * 100:.0f}%: trades {metrics['trades']}, "
            f"net PnL {metrics['net_pnl']:.2f}, expectancy {metrics['expectancy']}")
    lines.extend(["", "### Bottom Selective Entry (Debug Only)", ""])
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
    lines.extend(["", "### Threshold Curve", ""])
    for item in model.get("threshold_curve", []):
        metrics = item["metrics"]
        lines.append(
            f"- threshold {item['threshold']:.6g}, coverage "
            f"{item['candidate_coverage']:.3f}: trades {metrics['trades']}, "
            f"net PnL {metrics['net_pnl']:.2f}, expectancy {metrics['expectancy']}")
    lines.extend(["", "### Grouped Net PnL", ""])
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
    lines.extend(["", "### Split Oracle", ""])
    for fold_name, fold_report in summary["gate2"].get("split_oracle", {}).items():
        split_parts = []
        for split, split_report in fold_report.get("splits", {}).items():
            metrics = split_report["oracle"]
            split_parts.append(
                f"{split}: {metrics['net_pnl']:.2f}/{metrics['trades']}")
        lines.append(f"- {fold_name}: " + ", ".join(split_parts))
    lines.extend(["", "## Cost Stress", ""])
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
    write_text_atomic(path, "\n".join(lines))


def run_entry_capability_pipeline(
    *,
    config_path: str | Path,
    output: str | Path,
    dataset_only: bool = False,
    skip_sensitivity: bool = False,
    emit: StageLogger = stage_log,
) -> dict:
    stage = build_dataset_stage(config_path, output, emit=emit)
    if dataset_only:
        result = {
            "output": str(stage.output),
            "manifest_stage": "dataset_ready",
            "candidates": int(len(stage.candidates)),
            "outcomes": int(len(stage.outcomes)),
            "feature_count": int(stage.features.shape[1]),
            "flat_feature_count": int(stage.flat_features.shape[1]),
        }
        emit("dataset_only complete")
        return result

    opportunity, _, baselines = run_opportunity_stage(stage, emit=emit)
    model_result = run_model_stages(
        stage,
        skip_sensitivity=skip_sensitivity,
        baselines=baselines,
        emit=emit,
    )
    return {
        "output": str(stage.output),
        "opportunity": opportunity,
        "model": {
            "aggregate_full": model_result["model_report"]["aggregate_full"],
            "gate2_status": model_result["model_report"]["gate2_status"],
            "gate3_status": model_result["gate3_status"],
            "ridge_gate2_status": model_result["ridge_report"]["gate2_status"],
            "primary_feature_encoding": model_result["primary_encoding"],
            "feature_encoding_comparison": model_result["model_report"]["feature_encoding_comparison"],
        },
    }
