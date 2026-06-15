#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def _metric(summary: dict[str, Any], section: str, key: str, default=None):
    return summary.get(section, {}).get("metrics", {}).get(key, default)


def _random_mean(summary: dict[str, Any]):
    random = summary.get("matched_random_fixed_exit_executable_nonoverlap", {})
    return random.get("net_pnl_mean")


def _seed_row(name: str, path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    actual_pnl = _metric(summary, "actual_dreamer", "net_pnl", 0.0)
    actual_pf = _metric(summary, "actual_dreamer", "profit_factor", None)
    fixed_candidate = _metric(
        summary, "dreamer_entry_fixed_exit_candidate_join", "net_pnl", 0.0)
    fixed_exec = _metric(
        summary, "dreamer_entry_fixed_exit_executable_nonoverlap", "net_pnl", 0.0)
    delta = _metric(
        summary, "dreamer_entry_fixed_exit_candidate_join",
        "exit_holding_delta_pnl", None)
    if delta is None:
        actual_matched = summary.get(
            "dreamer_entry_fixed_exit_candidate_join", {}).get(
                "actual_matched", {}).get("net_pnl")
        if actual_matched is not None:
            delta = float(actual_matched) - float(fixed_candidate)
    random_mean = _random_mean(summary)
    return {
        "seed_name": name,
        "summary_path": str(path),
        "checkpoint": summary.get("checkpoint"),
        "checkpoint_step": summary.get("limitations", {}).get("checkpoint_step"),
        "seed_protocol": summary.get("seed_protocol", {}),
        "entry_eval": {
            "version": summary.get("entry_eval", {}).get("version"),
            "execution_timing": summary.get("entry_eval", {}).get("execution_timing"),
            "split_manifest_hash": summary.get("entry_eval", {}).get("split_manifest_hash"),
        },
        "actual_net_pnl": actual_pnl,
        "actual_profit_factor": actual_pf,
        "fixed_exit_candidate_net_pnl": fixed_candidate,
        "fixed_exit_executable_net_pnl": fixed_exec,
        "exit_holding_path_delta": delta,
        "matched_random_fixed_exit_mean": random_mean,
        "beats_matched_random_mean": (
            bool(float(actual_pnl) > float(random_mean))
            if random_mean is not None else None),
    }


def aggregate(seed_reports: dict[str, Path], experiment_name: str) -> dict[str, Any]:
    rows = [
        _seed_row(name, path, _read_json(path))
        for name, path in sorted(seed_reports.items())
    ]
    actual = [float(x["actual_net_pnl"]) for x in rows]
    pfs = [
        float(x["actual_profit_factor"]) for x in rows
        if x["actual_profit_factor"] not in (None, "not_computed")
    ]
    fixed_exec = [float(x["fixed_exit_executable_net_pnl"]) for x in rows]
    deltas = [
        float(x["exit_holding_path_delta"]) for x in rows
        if x["exit_holding_path_delta"] is not None
    ]
    beat_random = [
        bool(x["beats_matched_random_mean"]) for x in rows
        if x["beats_matched_random_mean"] is not None
    ]
    summary = {
        "experiment_name": experiment_name,
        "seed_count": len(rows),
        "seeds": rows,
        "aggregate": {
            "mean_test_actual_pnl": mean(actual) if actual else None,
            "median_test_actual_pnl": median(actual) if actual else None,
            "positive_test_seeds": sum(x > 0 for x in actual),
            "seeds_beating_matched_random_mean": sum(beat_random),
            "mean_profit_factor": mean(pfs) if pfs else None,
            "fixed_exit_entry_negative_all_seeds": (
                all(x < 0 for x in fixed_exec) if fixed_exec else None),
            "exit_holding_path_delta_positive_all_seeds": (
                all(x > 0 for x in deltas) if deltas else None),
        },
        "manifest_fields_recorded": [
            "experiment_seed",
            "dreamer_seed",
            "train_env_seed",
            "replay_seed",
            "eval_env_seed",
            "matched_random_seed",
            "split_manifest_hash",
            "entry_eval_version",
            "execution_timing",
        ],
    }
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    lines = [
        "# Dreamer Walk-Forward Multi-Seed Repeatability",
        "",
        f"- Experiment: `{summary['experiment_name']}`",
        f"- Seed count: `{summary['seed_count']}`",
        "",
        "## Aggregate",
        "",
        f"- Mean test actual PnL: `{agg['mean_test_actual_pnl']}`",
        f"- Median test actual PnL: `{agg['median_test_actual_pnl']}`",
        f"- Positive test seeds: `{agg['positive_test_seeds']}`",
        f"- Seeds beating matched-random mean: `{agg['seeds_beating_matched_random_mean']}`",
        f"- Mean PF: `{agg['mean_profit_factor']}`",
        f"- Fixed-exit entry negative all seeds: `{agg['fixed_exit_entry_negative_all_seeds']}`",
        f"- Exit/holding/path delta positive all seeds: `{agg['exit_holding_path_delta_positive_all_seeds']}`",
        "",
        "## Seeds",
        "",
        "| Seed | Dreamer | Train Env | Replay | Eval Env | Matched Random | Actual PnL | Fixed Exec PnL | Delta | Split Hash |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["seeds"]:
        sp = row.get("seed_protocol", {})
        ee = row.get("entry_eval", {})
        lines.append(
            f"| {row['seed_name']} "
            f"| {sp.get('dreamer_seed')} "
            f"| {sp.get('train_env_seed')} "
            f"| {sp.get('replay_seed')} "
            f"| {sp.get('eval_env_seed')} "
            f"| {sp.get('matched_random_seed')} "
            f"| {row.get('actual_net_pnl')} "
            f"| {row.get('fixed_exit_executable_net_pnl')} "
            f"| {row.get('exit_holding_path_delta')} "
            f"| `{ee.get('split_manifest_hash')}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_seed_reports(values: list[str]) -> dict[str, Path]:
    reports = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--seed-report must be NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        reports[name] = Path(path).expanduser().resolve()
    return reports


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--seed-report",
        action="append",
        default=[],
        help="NAME=/path/to/checkpoint_audit/summary.json. Repeat for each seed.",
    )
    args = parser.parse_args(argv)
    if not args.seed_report:
        raise SystemExit("At least one --seed-report is required")
    output = Path(args.output_dir).expanduser().resolve()
    summary = aggregate(_parse_seed_reports(args.seed_report), args.experiment_name)
    _write_json(output / "summary.json", summary)
    write_report(output / "report.md", summary)
    print(json.dumps({
        "output_dir": str(output),
        "seed_count": summary["seed_count"],
        "aggregate": summary["aggregate"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
