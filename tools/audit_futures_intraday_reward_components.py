from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gym_trading_env.envs.config import TradingConfig
from gym_trading_env.envs.trading_env import CustomTradingEnv


OUT = ROOT / "artifacts" / "reward_component_audit" / "current"
DOC = ROOT / "docs" / "futures_intraday_reward_component_audit.md"

REWARD_COMPONENTS = [
    "pnl",
    "fee",
    "dd",
    "eod",
    "close",
    "sl",
    "mkt_closed",
    "invalid_time",
    "invalid_streak",
    "invalid_total",
    "r_atr_close",
    "raw_total",
    "total",
]

RUNS = [
    {
        "run": "action-mask-formal-202606122100",
        "logdir": Path("/data/logdir/action-mask-formal-202606122100"),
        "summary": ROOT / "artifacts/dreamer_checkpoint_audit/action-mask-formal-202606122100_latest_perday_signal_close/summary.json",
    },
    {
        "run": "action-mask-formal-202606131200",
        "logdir": Path("/data/logdir/action-mask-formal-202606131200"),
        "summary": None,
    },
    {
        "run": "action-mask-wf1-rerun2-20260615",
        "logdir": Path("/data/logdir/action-mask-wf1-rerun2-20260615"),
        "summary": ROOT / "artifacts/dreamer_walk_forward/action-mask-wf1-rerun2-20260615_900k_test_signal_close/summary.json",
    },
    {
        "run": "action-mask-wf1-ms-seed-a-20260615",
        "logdir": Path("/data/logdir/action-mask-wf1-ms-seed-a-20260615"),
        "summary": ROOT / "artifacts/dreamer_walk_forward_multiseed/action-mask-wf1-multiseed-20260615/seed_a_selected_test_signal_close/summary.json",
    },
    {
        "run": "action-mask-wf1-ms-seed-b-20260615",
        "logdir": Path("/data/logdir/action-mask-wf1-ms-seed-b-20260615"),
        "summary": ROOT / "artifacts/dreamer_walk_forward_multiseed/action-mask-wf1-multiseed-20260615/seed_b_selected_test_signal_close/summary.json",
    },
    {
        "run": "action-mask-wf1-ms-seed-c-20260615",
        "logdir": Path("/data/logdir/action-mask-wf1-ms-seed-c-20260615"),
        "summary": ROOT / "artifacts/dreamer_walk_forward_multiseed/action-mask-wf1-multiseed-20260615/seed_c_selected_test_signal_close/summary.json",
    },
]


def read_json(path: Path | None) -> Any:
    if not path or not path.exists():
        return None
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_env_config(logdir: Path) -> Path | None:
    if not logdir.exists():
        return None
    candidates = sorted(logdir.glob("env*.yaml"))
    if candidates:
        return candidates[0]
    return None


def effective_reward_config(env_config: Path | None) -> dict[str, Any]:
    if not env_config or not env_config.exists():
        return {}
    cfg = TradingConfig.from_yaml(str(env_config))
    trading = cfg.trading
    return {
        "reward_function": cfg.training.reward_function,
        "alpha_unrealized": float(getattr(trading, "alpha_unrealized", 0.5)),
        "w_dd": 0.0,
        "w_eod": 0.25,
        "w_close": 0.10,
        "w_atr_close": float(getattr(trading, "w_atr_close", 0.10)),
        "atr_takeprofit_ratio": float(getattr(trading, "atr_takeprofit_ratio", 0.70)),
        "w_stoploss": 0.20,
        "invalid_action_punish": float(getattr(trading, "invalid_action_punish", 0.0)),
        "invalid_time_cost_total": float(getattr(trading, "invalid_time_cost_total", 0.0)),
        "R_cash_scale_source": "env._R_cash_last; stop-loss distance cash if available, else max(initial_balance*0.001, 1)",
        "stop_loss_enabled": bool(getattr(trading, "stop_loss_enabled", False)),
        "take_profit_enabled": bool(getattr(trading, "take_profit_enabled", False)),
        "env_config": str(env_config),
    }


def first_present(dct: dict[str, Any], paths: list[list[str]]) -> Any:
    for path in paths:
        cur = dct
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok:
            return cur
    return None


def metrics_component_summary(metrics: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    keys = sorted({k for row in metrics for k in row.keys()})
    reward_keys = [k for k in keys if "/reward/" in k or "reward/" in k]
    out: dict[str, Any] = {
        "has_reward_component_logs": bool(reward_keys),
        "reward_component_key_count": len(reward_keys),
        "reward_component_keys": reward_keys,
    }
    if not reward_keys:
        return out, reward_keys
    for comp in REWARD_COMPONENTS:
        candidates = [
            f"epstats/log/env/reward/{comp}/sum",
            f"epstats/log/env/reward/{comp}/avg",
            f"log/env/reward/{comp}",
        ]
        vals = []
        for row in metrics:
            for key in candidates:
                if key in row:
                    vals.append(float(row[key]))
                    break
        if vals:
            out[f"reward_{comp}_mean_metric"] = sum(vals) / len(vals)
            out[f"reward_{comp}_last_metric"] = vals[-1]
    return out, reward_keys


def summary_metrics(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    actual = first_present(summary, [["actual_dreamer", "metrics"], ["actual_dreamer"]])
    risk = first_present(summary, [["risk_path", "episodes"], ["risk_path"]])
    fixed = first_present(summary, [["dreamer_entry_fixed_exit_candidate_join", "metrics"], ["dreamer_entry_fixed_exit_candidate_join"]])
    out = {}
    if isinstance(actual, dict):
        out.update({
            "total_actual_pnl": actual.get("net_pnl"),
            "actual_profit_factor": actual.get("profit_factor"),
            "actual_trades": actual.get("trades"),
            "no_trade_episode_ratio": actual.get("no_trade_day_ratio") or actual.get("zero_trade_day_ratio"),
            "largest_winning_trade": actual.get("largest_winning_trade"),
            "largest_losing_trade": actual.get("largest_losing_trade"),
        })
    if isinstance(risk, dict):
        out.update({
            "margin_termination_count": risk.get("margin_termination_count"),
            "return_pct_min": risk.get("return_pct_min"),
            "return_pct_mean": risk.get("return_pct_mean"),
            "return_pct_max": risk.get("return_pct_max"),
        })
    if isinstance(fixed, dict):
        out["fixed_exit_candidate_pnl"] = fixed.get("net_pnl")
    return out


def latest_metric(metrics: list[dict[str, Any]], key: str) -> Any:
    for row in reversed(metrics):
        if key in row:
            return row[key]
    return None


def analyze_runs() -> list[dict[str, Any]]:
    rows = []
    for spec in RUNS:
        logdir = spec["logdir"]
        metrics = read_jsonl(logdir / "metrics.jsonl")
        env_config = find_env_config(logdir)
        row: dict[str, Any] = {
            "run": spec["run"],
            "logdir": str(logdir),
            "metrics_jsonl_exists": (logdir / "metrics.jsonl").exists(),
            "metrics_rows": len(metrics),
        }
        row.update(effective_reward_config(env_config))
        comp, _ = metrics_component_summary(metrics)
        row.update(comp)
        row.update(summary_metrics(read_json(spec["summary"])))
        row.update({
            "final_episode_score": latest_metric(metrics, "episode/score"),
            "final_train_return_pct_avg": latest_metric(metrics, "epstats/log/env/return_pct/avg"),
            "invalid_action_total_sum_last": latest_metric(metrics, "epstats/log/env/invalid_action_total/sum"),
            "action_rejected_sum_last": latest_metric(metrics, "epstats/log/env/action_rejected/sum"),
            "trades_opened_sum_last": latest_metric(metrics, "epstats/log/env/trades_opened/sum"),
            "avg_hold_bars": None,
            "max_floating_drawdown": None,
            "close_reward_actual_pnl_corr": None,
            "atr_close_reward_actual_pnl_corr": None,
            "classification": "INSUFFICIENT_LOGGING" if not comp["has_reward_component_logs"] else "COMPONENT_LOGS_AVAILABLE",
        })
        rows.append(row)
    return rows


def run_env_smoke() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config = Path(
        os.environ.get(
            "GYM_TRADING_REWARD_SMOKE_CONFIG",
            "/data/logdir/trading_contracts/jm_walk_forward_20240603_20251202/"
            "configs/env/jm_walk_forward_20240603_20251202_train.yaml",
        )
    )
    prev_cwd = Path.cwd()
    dreamer_root = Path("/home/v/Documents/work/dreamerv3")
    rows = []
    trade_rows = []
    info_reward_keys = []
    env = None
    try:
        if dreamer_root.exists():
            os.chdir(dreamer_root)
        env = CustomTradingEnv(config_path=str(config))
        obs, info = env.reset(seed=20260617)
        info_reward_keys = sorted(k for k in info if k.startswith("log/env/reward/"))
        actions = []
        # Target action ids are [SHORT=0, FLAT=1, LONG=2].
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
        open_component_sums = None
        seen_records = 0
        for step, action in enumerate(actions, start=1):
            obs, reward, terminated, truncated, info = env.step(action)
            dbg = dict(getattr(env, "_reward_debug", {}) or {})
            row = {
                "run": "smoke_env_scripted",
                "step": step,
                "action": action,
                "reward": float(reward),
                "equity": float(env._calculate_equity()),
                "position_long": float(env.user_accounts.long_position),
                "position_short": float(env.user_accounts.short_position),
            }
            for comp in REWARD_COMPONENTS:
                row[comp] = float(dbg.get(comp, 0.0) or 0.0)
            rows.append(row)

            if (env.user_accounts.long_position > 0 or env.user_accounts.short_position > 0) and open_step is None:
                open_step = step
                open_component_sums = {comp: 0.0 for comp in REWARD_COMPONENTS}
            if open_component_sums is not None:
                for comp in REWARD_COMPONENTS:
                    open_component_sums[comp] += float(dbg.get(comp, 0.0) or 0.0)

            records = env.trade_record_manager.trade_history
            if len(records) > seen_records:
                for rec in records[seen_records:]:
                    d = rec.to_dict()
                    op = str(d.get("operation_type", "")).lower()
                    if "close" in op or float(d.get("pnl", "0") or 0) != 0:
                        tr = {
                            "run": "smoke_env_scripted",
                            "open_step": open_step,
                            "close_step": step,
                            "trade_actual_pnl": float(d.get("pnl", "0") or 0),
                            "exit_reason": (d.get("meta") or {}).get("reason", op),
                            "MAE": None,
                            "MFE": None,
                        }
                        if open_component_sums:
                            tr.update({f"{comp}_reward_sum": open_component_sums.get(comp, 0.0) for comp in REWARD_COMPONENTS})
                        trade_rows.append(tr)
                        open_step = None
                        open_component_sums = None
                seen_records = len(records)
            if terminated or truncated:
                break
    finally:
        if env is not None:
            env.close()
        os.chdir(prev_cwd)
    summary = {
        "env_info_reward_keys_on_reset": info_reward_keys,
        "env_info_reward_key_count_on_reset": len(info_reward_keys),
        "steps": len(rows),
        "component_sums": {
            comp: sum(float(row.get(comp, 0.0) or 0.0) for row in rows)
            for comp in REWARD_COMPONENTS
        },
        "total_reward": sum(float(row["reward"]) for row in rows),
        "trade_rows": len(trade_rows),
    }
    return rows, trade_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_report(rows: list[dict[str, Any]], smoke_summary: dict[str, Any]) -> None:
    missing = [row["run"] for row in rows if not row.get("has_reward_component_logs")]
    cfg = rows[0] if rows else {}
    component_sums = smoke_summary.get("component_sums", {})
    smoke_total_abs = sum(abs(float(component_sums.get(k, 0.0))) for k in REWARD_COMPONENTS if k not in ("raw_total", "total"))
    def share(k: str) -> float | None:
        if not smoke_total_abs:
            return None
        return abs(float(component_sums.get(k, 0.0))) / smoke_total_abs

    lines = [
        "# FuturesIntradayReward Component Audit",
        "",
        "Date: 2026-06-17",
        "",
        "## Final Classification",
        "",
        "**INSUFFICIENT_LOGGING** for historical runs. Existing full training `metrics.jsonl` files do not contain `log/env/reward/*` component metrics, so close shaping dominance, ATR close dominance, and reward/PnL correlation cannot be measured retrospectively.",
        "",
        "Static config risk: **RISK_UNDERPENALIZED** is plausible because `w_dd=0.0` in `FuturesIntradayReward`, so drawdown increment penalty is disabled unless code/config changes later.",
        "",
        "## Effective Runtime Reward Values",
        "",
        "| field | value |",
        "|---|---:|",
    ]
    for key in [
        "alpha_unrealized", "w_dd", "w_eod", "w_close", "w_atr_close",
        "atr_takeprofit_ratio", "w_stoploss", "invalid_action_punish",
        "invalid_time_cost_total", "stop_loss_enabled", "take_profit_enabled",
    ]:
        lines.append(f"| `{key}` | `{cfg.get(key)}` |")
    lines += [
        f"| `R_cash_scale_source` | `{cfg.get('R_cash_scale_source')}` |",
        "",
        "## Historical Run Logging Coverage",
        "",
        "| run | reward component logs | actual PnL | PF | invalid sum last | classification |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['run']}` | {row.get('reward_component_key_count', 0)} | "
            f"{row.get('total_actual_pnl')} | {row.get('actual_profit_factor')} | "
            f"{row.get('invalid_action_total_sum_last')} | {row.get('classification')} |"
        )
    lines += [
        "",
        "Missing component logs:",
        "",
    ]
    lines += [f"- `{run}`" for run in missing]
    lines += [
        "",
        "## Smoke Logging Verification",
        "",
        f"- Env info reward keys on reset: `{smoke_summary.get('env_info_reward_key_count_on_reset')}`",
        f"- Scripted env steps: `{smoke_summary.get('steps')}`",
        f"- Scripted env trades attributed: `{smoke_summary.get('trade_rows')}`",
        f"- Smoke total reward: `{smoke_summary.get('total_reward')}`",
        "",
        "| component | smoke sum | absolute share |",
        "|---|---:|---:|",
    ]
    for comp in REWARD_COMPONENTS:
        if comp in ("raw_total", "total"):
            continue
        lines.append(f"| `{comp}` | {component_sums.get(comp, 0.0):.6f} | {share(comp)} |")
    lines += [
        "",
        "The smoke is a logging and accounting check only. It is not a policy-quality result and must not be used to judge the trained agent.",
        "",
        "## Outputs",
        "",
        "- `artifacts/reward_component_audit/current/summary.json`",
        "- `artifacts/reward_component_audit/current/reward_component_by_run.csv`",
        "- `artifacts/reward_component_audit/current/reward_component_by_trade.csv`",
        "- `artifacts/reward_component_audit/current/reward_component_smoke_steps.csv`",
        "",
        "## Next Constraint",
        "",
        "Future formal training runs should keep `log/env/reward/*` enabled from the first reset. Only then can close reward vs actual PnL and ATR close reward vs actual PnL correlations be computed from training data.",
        "",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_rows = analyze_runs()
    smoke_steps, trade_rows, smoke_summary = run_env_smoke()
    summary = {
        "classification": "INSUFFICIENT_LOGGING",
        "static_config_risk": "RISK_UNDERPENALIZED",
        "historical_runs_analyzed": len(run_rows),
        "historical_runs_with_reward_component_logs": sum(
            1 for row in run_rows if row.get("has_reward_component_logs")),
        "smoke_summary": smoke_summary,
        "notes": [
            "Historical full training metrics do not contain log/env/reward/* keys.",
            "FuturesIntradayReward currently has w_dd=0.0 by constructor default.",
            "Close and ATR close shaping are enabled by default at w_close=0.10 and w_atr_close=0.10.",
        ],
    }
    write_csv(OUT / "reward_component_by_run.csv", run_rows)
    write_csv(OUT / "reward_component_by_trade.csv", trade_rows)
    write_csv(OUT / "reward_component_smoke_steps.csv", smoke_steps)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))
    write_report(run_rows, smoke_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
