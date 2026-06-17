from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gym_trading_env.envs.trading_env import CustomTradingEnv
from gym_trading_env.rewards.reward_audit import REWARD_DEBUG_KEYS


OUT = ROOT / "artifacts" / "reward_mtm_clean_risk_tests"
DOC = ROOT / "docs" / "reward_mtm_clean_risk_test_report.md"
DREAMER_ROOT = Path("/home/v/Documents/work/dreamerv3")
DREAMER_PYTHON = Path("/home/v/miniconda3/envs/dreamerv3/bin/python")
BASE_CONFIG = ROOT / "configs" / "env_trading_stage1_jm_walk_forward_train_20240603_20250731.yaml"

V1_REWARD = "futures_intraday_mtm_clean_reward_function"
V2_REWARD = "futures_intraday_mtm_risk_reward_function"

MANDATORY_METRIC_KEYS = [
    "total",
    "pnl",
    "raw_total",
    "mtm_equity",
    "delta_equity",
    "scale_cash",
    "close",
    "r_atr_close",
    "dd",
    "invalid_total",
]

V1_ZERO_KEYS = ["close", "r_atr_close", "dd", "eod", "invalid_total"]
V2_ZERO_KEYS = ["close", "r_atr_close", "eod", "invalid_total"]
V2_EXTRA_KEYS = ["risk_dd", "risk_adverse", "risk_loss_time"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_config(reward_name: str) -> Path:
    data = yaml.safe_load(BASE_CONFIG.read_text())
    data["trading"]["initial_balance"] = 100000.0
    data["training"]["reward_function"] = reward_name
    data["training"]["randomize_start"] = False
    data["debug"] = data.get("debug", {}) or {}
    data["debug"]["log_level"] = "ERROR"
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"env_{reward_name}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def choose_scripted_action(obs: dict[str, Any], env: CustomTradingEnv, closed_trades: int, hold_bars: int) -> int:
    mask = obs["action_mask"]
    have_long = env.user_accounts.long_position > 0
    have_short = env.user_accounts.short_position > 0
    if have_long:
        return 1 if hold_bars >= 1 and mask[1] > 0.5 else 2
    if have_short:
        return 1 if hold_bars >= 1 and mask[1] > 0.5 else 0
    if closed_trades >= 1:
        return 1 if mask[1] > 0.5 else int(max(range(len(mask)), key=lambda i: mask[i]))
    if mask[2] > 0.5:
        return 2
    if mask[0] > 0.5:
        return 0
    return 1 if mask[1] > 0.5 else int(max(range(len(mask)), key=lambda i: mask[i]))


def trade_direction(record: dict[str, Any]) -> str:
    meta = record.get("meta") or {}
    side = meta.get("side")
    if side:
        return str(side).lower()
    op = str(record.get("operation_type", "")).lower()
    if "long" in op:
        return "long"
    if "short" in op:
        return "short"
    return ""


def is_open_record(record: dict[str, Any]) -> bool:
    op = str(record.get("operation_type", "")).upper()
    return "OPEN" in op and "CLOSE" not in op


def is_close_record(record: dict[str, Any]) -> bool:
    op = str(record.get("operation_type", "")).upper()
    return "CLOSE" in op


def add_debug_sums(target: dict[str, Any], debug: dict[str, Any]) -> None:
    target["reward_sum_during_trade"] += float(debug.get("total", 0.0) or 0.0)
    target["pnl_reward_sum"] += float(debug.get("pnl", 0.0) or 0.0)
    target["close_reward_sum"] += float(debug.get("close", 0.0) or 0.0)
    target["atr_close_reward_sum"] += float(debug.get("r_atr_close", 0.0) or 0.0)
    target["dd_reward_sum"] += float(debug.get("dd", 0.0) or 0.0)


def run_scripted_rollout(reward_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config = make_config(reward_name)
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    previous_cwd = Path.cwd()
    env = None
    try:
        os.chdir(DREAMER_ROOT)
        env = CustomTradingEnv(config_path=str(config))
        obs, info = env.reset(seed=20260617)

        seen_records = 0
        active: dict[str, Any] | None = None

        hold_bars = 0
        for step in range(1, 121):
            action = choose_scripted_action(obs, env, len(trades), hold_bars)
            obs, reward, terminated, truncated, info = env.step(action)
            debug = dict(getattr(env, "_reward_debug", {}) or {})

            row = {
                "step": step,
                "action": action,
                "reward": float(reward),
                "equity": float(env._calculate_equity()),
                "position_long": float(env.user_accounts.long_position),
                "position_short": float(env.user_accounts.short_position),
            }
            for key, value in debug.items():
                row[key] = float(value)
            rows.append(row)

            new_records = [
                rec.to_dict()
                for rec in env.trade_record_manager.trade_history[seen_records:]
            ]
            for rec in new_records:
                if is_open_record(rec) and active is None:
                    hold_bars = 0
                    active = {
                        "trade_id": len(trades),
                        "entry_timestamp": rec.get("timestamp", ""),
                        "exit_timestamp": "",
                        "direction": trade_direction(rec),
                        "actual_net_pnl": 0.0,
                        "reward_sum_during_trade": 0.0,
                        "pnl_reward_sum": 0.0,
                        "close_reward_sum": 0.0,
                        "atr_close_reward_sum": 0.0,
                        "dd_reward_sum": 0.0,
                        "hold_bars": 0,
                        "exit_reason": "",
                    }

            if active is not None:
                add_debug_sums(active, debug)
                active["hold_bars"] += 1
                hold_bars += 1

            for rec in new_records:
                if is_close_record(rec) and active is not None:
                    active["exit_timestamp"] = rec.get("timestamp", "")
                    active["actual_net_pnl"] = float(rec.get("pnl", "0") or 0)
                    active["exit_reason"] = (rec.get("meta") or {}).get("reason", rec.get("operation_type", ""))
                    trades.append(active)
                    active = None
                    hold_bars = 0

            seen_records = len(env.trade_record_manager.trade_history)
            if len(trades) >= 1 and not (
                env.user_accounts.long_position > 0 or env.user_accounts.short_position > 0
            ):
                break
            if terminated or truncated:
                break
    finally:
        if env is not None:
            env.close()
        os.chdir(previous_cwd)

    required = set(REWARD_DEBUG_KEYS)
    if reward_name == V2_REWARD:
        required |= set(V2_EXTRA_KEYS)
    missing_step_keys = sorted(required - set(rows[-1].keys())) if rows else sorted(required)
    summary = {
        "reward_name": reward_name,
        "steps": len(rows),
        "trades": len(trades),
        "missing_last_step_keys": missing_step_keys,
        "has_trade_attribution": bool(trades),
    }
    return rows, trades, summary


def metric_key(name: str) -> str:
    return f"epstats/log/env/reward/{name}/sum"


def run_dreamer_metrics_smoke(reward_name: str) -> dict[str, Any]:
    config = make_config(reward_name)
    logdir = Path("/tmp/reward-mtm-smoke-" + reward_name.replace("_", "-") + "-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC}:{env.get('PYTHONPATH', '')}"
    env["MPLCONFIGDIR"] = "/tmp/matplotlib-reward-mtm-smoke"
    cmd = [
        str(DREAMER_PYTHON),
        "dreamerv3/main.py",
        "--configs", "action_mask_formal", "debug",
        "--logdir", str(logdir),
        "--run.steps", "760",
        "--run.envs", "1",
        "--run.eval_envs", "0",
        "--run.actor_threads", "1",
        "--run.save_every", "10000",
        "--run.report_every", "0",
        "--run.log_every", "-1",
        "--env.gymnasium.config_path", str(config),
        "--experiment_seed", "923",
        "--dreamer.seed", "923",
        "--env.train_seed", "923101",
        "--env.eval_seed", "0",
        "--replay.seed", "923202",
        "--audit.matched_random_seed", "20260615",
        "--audit.entry_eval_version", "reward_mtm_smoke",
        "--audit.split_manifest_hash", "smoke",
        "--audit.execution_timing", "signal_on_close_plus_spread",
    ]
    proc = subprocess.run(
        cmd,
        cwd=DREAMER_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    metrics = read_jsonl(logdir / "metrics.jsonl")
    keys = sorted({key for row in metrics for key in row})
    required_metric_keys = [metric_key(key) for key in MANDATORY_METRIC_KEYS]
    if reward_name == V2_REWARD:
        required_metric_keys.extend(metric_key(key) for key in V2_EXTRA_KEYS)
    missing = [key for key in required_metric_keys if key not in keys]

    def values_for(key: str) -> list[float]:
        out = []
        for row in metrics:
            if key in row:
                out.append(float(row[key]))
        return out

    zero_keys = V1_ZERO_KEYS if reward_name == V1_REWARD else V2_ZERO_KEYS
    zero_sums = {}
    for key in zero_keys:
        vals = values_for(metric_key(key))
        zero_sums[key] = all(abs(v) <= 1e-9 for v in vals) if vals else False

    return {
        "reward_name": reward_name,
        "logdir": str(logdir),
        "returncode": proc.returncode,
        "metrics_rows": len(metrics),
        "missing_metric_keys": missing,
        "mandatory_metric_keys_present": not missing,
        "zero_disabled_metric_sums": zero_sums,
        "stdout_tail": proc.stdout[-3000:],
    }


def write_report(summary: dict[str, Any]) -> None:
    status = summary["classification"]
    lines = [
        "# MTM Clean/Risk Reward Test Report",
        "",
        "Date: 2026-06-17",
        "",
        f"Final classification: `{status}`",
        "",
        "## Scope",
        "",
        "This report covers `futures_intraday_mtm_clean_reward_function` (v1-clean) and `futures_intraday_mtm_risk_reward_function` (v2-risk). No formal Dreamer training was run; only unit tests, scripted rollouts, and 760-step metrics smoke runs were used.",
        "",
        "## Results",
        "",
    ]
    for name in [V1_REWARD, V2_REWARD]:
        scripted = summary["scripted"][name]
        metrics = summary["metrics"][name]
        lines += [
            f"### {name}",
            "",
            f"- Scripted steps: `{scripted['steps']}`",
            f"- Attributed trades: `{scripted['trades']}`",
            f"- Missing scripted step keys: `{scripted['missing_last_step_keys']}`",
            f"- Dreamer smoke metrics rows: `{metrics['metrics_rows']}`",
            f"- Missing metrics keys: `{metrics['missing_metric_keys']}`",
            f"- Disabled component sums zero: `{metrics['zero_disabled_metric_sums']}`",
            f"- Dreamer smoke logdir: `{metrics['logdir']}`",
            "",
        ]
    lines += [
        "## Notes",
        "",
        "- v1-clean disables fee, dd, EOD, close, stop-loss, market-closed, invalid-action, and ATR-close shaping. Fees still affect reward through ledger cash/equity deltas.",
        "- v2-risk disables close/ATR/EOD/invalid/market-closed/stop-loss shaping but keeps `dd` active for explicit risk penalties.",
        "- v2-risk exposes extra numeric audit fields: `risk_dd`, `risk_adverse`, `risk_loss_time`, `drawdown_cash`, `drawdown_inc_cash`, `adverse_cash`, `adverse_inc_cash`, and `loss_steps`.",
        "- Trade-level attribution is a smoke attribution from env trade records and reward debug sums; it is sufficient for visibility, not a full accounting report.",
        "",
        "## Outputs",
        "",
        "- `artifacts/reward_mtm_clean_risk_tests/summary.json`",
        "- `artifacts/reward_mtm_clean_risk_tests/v1_smoke_steps.csv`",
        "- `artifacts/reward_mtm_clean_risk_tests/v2_smoke_steps.csv`",
        "- `artifacts/reward_mtm_clean_risk_tests/v1_trade_attribution.csv`",
        "- `artifacts/reward_mtm_clean_risk_tests/v2_trade_attribution.csv`",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines))


def classify(summary: dict[str, Any]) -> str:
    scripted_ok = all(
        not item["missing_last_step_keys"]
        for item in summary["scripted"].values()
    )
    metrics_ok = all(
        item["returncode"] == 0 and item["mandatory_metric_keys_present"]
        for item in summary["metrics"].values()
    )
    zero_ok = all(
        all(item["zero_disabled_metric_sums"].values())
        for item in summary["metrics"].values()
    )
    trade_ok = all(
        item["has_trade_attribution"]
        for item in summary["scripted"].values()
    )
    if not metrics_ok:
        return "REWARD_LOGGING_FAIL"
    if not scripted_ok or not zero_ok:
        return "REWARD_TEST_FAIL"
    if not trade_ok:
        return "REWARD_TEST_PARTIAL"
    return "REWARD_TEST_PASS"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scripted: dict[str, dict[str, Any]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for reward_name, step_file, trade_file in [
        (V1_REWARD, "v1_smoke_steps.csv", "v1_trade_attribution.csv"),
        (V2_REWARD, "v2_smoke_steps.csv", "v2_trade_attribution.csv"),
    ]:
        rows, trades, step_summary = run_scripted_rollout(reward_name)
        write_csv(OUT / step_file, rows)
        write_csv(OUT / trade_file, trades)
        scripted[reward_name] = step_summary
        metrics[reward_name] = run_dreamer_metrics_smoke(reward_name)

    summary = {
        "scripted": scripted,
        "metrics": metrics,
    }
    summary["classification"] = classify(summary)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    write_report(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["classification"] not in {"REWARD_TEST_PASS", "REWARD_TEST_PARTIAL"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
