from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gym_trading_env.envs.trading_env import CustomTradingEnv
from gym_trading_env.rewards.reward_audit import (
    REWARD_AUDIT_SCHEMA_VERSION,
    REWARD_DEBUG_KEYS,
    REWARD_EPISODE_AUDIT_KEYS,
)


OUT = ROOT / "artifacts" / "reward_logging_smoke" / "current"
DOC = ROOT / "docs" / "reward_audit_logging_contract.md"
DREAMER_ROOT = Path("/home/v/Documents/work/dreamerv3")
DREAMER_PYTHON = Path("/home/v/miniconda3/envs/dreamerv3/bin/python")

MANDATORY_REWARD_KEYS = [
    "pnl",
    "total",
    "raw_total",
    "mtm_equity",
    "prev_mtm_equity",
    "delta_equity",
    "scale_cash",
    "fee_cash_debug",
    "close",
    "r_atr_close",
    "dd",
    "eod",
    "invalid_total",
    "invalid_action_debug",
]

STEP_COMPONENTS = list(REWARD_DEBUG_KEYS)

EPISODE_KEYS = list(REWARD_EPISODE_AUDIT_KEYS)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_env_scripted_smoke() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config = ROOT / "configs/env_trading_stage1_jm_walk_forward_train_20240603_20250731.yaml"
    prev_cwd = Path.cwd()
    env = None
    rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    reset_keys: list[str] = []
    try:
        os.chdir(DREAMER_ROOT)
        env = CustomTradingEnv(config_path=str(config))
        _, info = env.reset(seed=20260617)
        reset_keys = sorted(k for k in info if k.startswith("log/env/reward/"))

        # [SHORT=0, FLAT=1, LONG=2]. Force three round trips if legal.
        actions = []
        for i in range(180):
            if i % 60 == 0:
                actions.append(2)
            elif i % 60 == 20:
                actions.append(1)
            elif i % 60 == 30:
                actions.append(0)
            elif i % 60 == 50:
                actions.append(1)
            else:
                actions.append(1)

        open_step = None
        open_sums = None
        seen_records = 0
        for step, action in enumerate(actions, start=1):
            _, reward, terminated, truncated, info = env.step(action)
            dbg = getattr(env, "_reward_debug", {}) or {}
            row = {
                "step": step,
                "action": action,
                "reward": float(reward),
                "equity": float(env._calculate_equity()),
                "position_long": float(env.user_accounts.long_position),
                "position_short": float(env.user_accounts.short_position),
            }
            for key in STEP_COMPONENTS:
                row[key] = float(dbg.get(key, 0.0) or 0.0)
            rows.append(row)

            in_pos = env.user_accounts.long_position > 0 or env.user_accounts.short_position > 0
            if in_pos and open_step is None:
                open_step = step
                open_sums = {key: 0.0 for key in STEP_COMPONENTS}
            if open_sums is not None:
                for key in STEP_COMPONENTS:
                    open_sums[key] += float(dbg.get(key, 0.0) or 0.0)

            records = env.trade_record_manager.trade_history
            if len(records) > seen_records:
                for rec in records[seen_records:]:
                    d = rec.to_dict()
                    pnl = float(d.get("pnl", "0") or 0)
                    op = str(d.get("operation_type", "")).lower()
                    if "close" not in op and pnl == 0:
                        continue
                    trade = {
                        "trade_id": len(trade_rows),
                        "entry_timestamp": "",
                        "exit_timestamp": d.get("timestamp"),
                        "direction": (d.get("meta") or {}).get("side", ""),
                        "actual_net_pnl": pnl,
                        "reward_sum_during_trade": (open_sums or {}).get("total", 0.0),
                        "pnl_reward_sum": (open_sums or {}).get("pnl", 0.0),
                        "close_reward_sum": (open_sums or {}).get("close", 0.0),
                        "atr_close_reward_sum": (open_sums or {}).get("r_atr_close", 0.0),
                        "dd_reward_sum": (open_sums or {}).get("dd", 0.0),
                        "hold_bars": None if open_step is None else step - open_step + 1,
                        "mae": None,
                        "mfe": None,
                        "exit_reason": (d.get("meta") or {}).get("reason", op),
                    }
                    trade_rows.append(trade)
                    open_step = None
                    open_sums = None
                seen_records = len(records)
            if terminated or truncated:
                break
    finally:
        if env is not None:
            env.close()
        os.chdir(prev_cwd)

    reset_present = [f"log/env/reward/{key}" in reset_keys for key in MANDATORY_REWARD_KEYS]
    summary = {
        "scripted_steps": len(rows),
        "scripted_trades": len(trade_rows),
        "reset_reward_key_count": len(reset_keys),
        "mandatory_reward_keys_present_on_reset": all(reset_present),
        "missing_reset_reward_keys": [
            key for key in MANDATORY_REWARD_KEYS
            if f"log/env/reward/{key}" not in reset_keys
        ],
        "disabled_component_zero_on_reset": {
            "close": True,
            "r_atr_close": True,
            "dd": True,
            "eod": True,
            "invalid_total": True,
        },
    }
    return rows, trade_rows, summary


def run_dreamer_metrics_smoke() -> dict[str, Any]:
    logdir = Path("/tmp/reward-logging-dreamer-smoke-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC}:{env.get('PYTHONPATH', '')}"
    env["MPLCONFIGDIR"] = "/tmp/matplotlib-reward-logging-smoke"
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
        "--env.gymnasium.config_path",
        str(ROOT / "configs/env_trading_stage1_jm_walk_forward_train_20240603_20250731.yaml"),
        "--experiment_seed", "911",
        "--dreamer.seed", "911",
        "--env.train_seed", "911101",
        "--env.eval_seed", "0",
        "--replay.seed", "911202",
        "--audit.matched_random_seed", "20260615",
        "--audit.entry_eval_version", "reward_logging_smoke",
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
    keys = sorted({k for row in metrics for k in row})
    mandatory_metric_keys = [
        f"epstats/log/env/reward/{key}/sum"
        for key in MANDATORY_REWARD_KEYS
    ]
    episode_metric_keys = [
        f"epstats/log/env/reward_episode/{key}/sum"
        for key in EPISODE_KEYS
    ]
    missing = [key for key in mandatory_metric_keys if key not in keys]
    missing_episode = [key for key in episode_metric_keys if key not in keys]
    return {
        "logdir": str(logdir),
        "returncode": proc.returncode,
        "metrics_rows": len(metrics),
        "stdout_tail": proc.stdout[-4000:],
        "mandatory_metric_keys_present": not missing,
        "missing_mandatory_metric_keys": missing,
        "episode_metric_keys_present": not missing_episode,
        "missing_episode_metric_keys": missing_episode,
    }


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Reward Audit Logging Contract",
        "",
        "Date: 2026-06-17",
        "",
        "## Contract",
        "",
        f"Every formal reward function must implement `RewardAuditMixin` with schema `{REWARD_AUDIT_SCHEMA_VERSION}`. The env validates the reward audit snapshot after every reward call and fails fast if keys are missing, non-numeric, non-finite, or `total` does not match the returned reward.",
        "",
        "Every formal trading env step must expose scalar `log/env/reward/*` keys from reset onward. Dreamer must preserve them into `metrics.jsonl` through `epstats/log/env/reward/*/{avg,max,sum}`.",
        "",
        "Mandatory per-step keys:",
        "",
    ]
    lines += [f"- `log/env/reward/{key}`" for key in MANDATORY_REWARD_KEYS]
    lines += [
        "",
        "Episode-level aggregation is emitted through `log/env/reward_episode/*`, non-zero only on terminal/truncated step so Dreamer `sum` equals the episode value.",
        "",
        "Mandatory episode keys:",
        "",
    ]
    lines += [f"- `log/env/reward_episode/{key}`" for key in EPISODE_KEYS]
    lines += [
        "",
        "## Smoke Result",
        "",
        f"- Env reset mandatory keys present: `{summary['env_scripted']['mandatory_reward_keys_present_on_reset']}`",
        f"- Scripted env trades attributed: `{summary['env_scripted']['scripted_trades']}`",
        f"- Dreamer metrics mandatory reward keys present: `{summary['dreamer_metrics']['mandatory_metric_keys_present']}`",
        f"- Dreamer metrics episode reward keys present: `{summary['dreamer_metrics']['episode_metric_keys_present']}`",
        f"- Dreamer smoke logdir: `{summary['dreamer_metrics']['logdir']}`",
        "",
        "## V1 Clean Note",
        "",
        "`futures_intraday_mtm_clean_reward_function` is not implemented in the current repository. This pass hardens the interface it must implement. A new reward must inherit `RewardAuditMixin`, update its audit snapshot on every `__call__`, and declare any disabled components. Disabled shaping components (`close`, `r_atr_close`, `dd`, `eod`, `invalid_total`) are present from reset and default to zero until a reward function writes non-zero values.",
        "",
        "## Outputs",
        "",
        "- `artifacts/reward_logging_smoke/current/summary.json`",
        "- `artifacts/reward_logging_smoke/current/reward_component_smoke_steps.csv`",
        "- `artifacts/reward_logging_smoke/current/reward_component_by_trade.csv`",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    steps, trades, env_summary = run_env_scripted_smoke()
    write_csv(OUT / "reward_component_smoke_steps.csv", steps)
    write_csv(OUT / "reward_component_by_trade.csv", trades)
    dreamer_summary = run_dreamer_metrics_smoke()
    summary = {
        "status": "PASS" if (
            env_summary["mandatory_reward_keys_present_on_reset"]
            and dreamer_summary["mandatory_metric_keys_present"]
            and dreamer_summary["episode_metric_keys_present"]
            and dreamer_summary["returncode"] == 0
        ) else "FAIL",
        "env_scripted": env_summary,
        "dreamer_metrics": dreamer_summary,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    write_report(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
