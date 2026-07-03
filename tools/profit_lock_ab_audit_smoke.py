#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.dreamer_checkpoint_audit import (  # noqa: E402
    PROFIT_LOCK_DEBUG_FIELDS,
    STEP_REWARD_COMPONENT_COLUMNS,
    _ensure_step_reward_schema,
    build_profit_lock_trade_rollup,
    write_profit_lock_ab_audit_outputs,
)


DATE = "20260703"
IN_STEPS = ROOT / "artifacts" / "reward_profit_lock_smoke" / "step_reward_components.csv"
OUT = ROOT / "artifacts" / "profit_lock_ab_audit_script_fixes"
DOC = ROOT / "docs" / f"profit_lock_ab_audit_script_fixes_{DATE}.md"


def _standard_step_rows(smoke: pd.DataFrame) -> pd.DataFrame:
    base_ts = pd.Timestamp("2026-06-22 09:01:00", tz="Asia/Shanghai")
    rows = []
    for idx, row in smoke.reset_index(drop=True).iterrows():
        step = int(row.get("step", idx + 1))
        episode_id = 0 if step <= 6 else 1
        event = str(row.get("event", ""))
        close_reason = "MANUAL" if event.startswith("close") else ""
        std = {
            "checkpoint_label": "sample_profit_lock",
            "episode_id": episode_id,
            "trading_day": 20260622,
            "timestamp": (base_ts + pd.Timedelta(minutes=step - 1)).isoformat(),
            "step_index": step if episode_id == 0 else step - 6,
            "requested_action": "LONG" if float(row.get("long_position", 0.0)) > 0 else "FLAT",
            "selected_action": 2 if float(row.get("long_position", 0.0)) > 0 else 1,
            "executed_action": 2 if float(row.get("long_position", 0.0)) > 0 else 1,
            "action_mask_short": 1.0,
            "action_mask_flat": 1.0,
            "action_mask_long": 1.0,
            "action_rejected": 0.0,
            "reject_code": 0.0,
            "execution_failed": 0.0,
            "position_side": "LONG" if float(row.get("long_position", 0.0)) > 0 else "FLAT",
            "user_cash": 10000.0,
            "user_margin": 0.0,
            "unrealized_pnl": float(row.get("unrealized_pnl", 0.0)),
            "mtm_equity": float(row.get("mtm_equity", 0.0)),
            "prev_mtm_equity": float(row.get("prev_mtm_equity", 0.0)),
            "delta_equity": float(row.get("delta_equity", 0.0)),
            "reward_total": float(row.get("reward", row.get("total", 0.0))),
            "raw_total": float(row.get("raw_total", 0.0)),
            "pnl": float(row.get("pnl", 0.0)),
            "close": float(row.get("close", 0.0)),
            "r_atr_close": float(row.get("r_atr_close", 0.0)),
            "eod": float(row.get("eod", 0.0)),
            "sl": float(row.get("sl", 0.0)),
            "invalid_total": float(row.get("invalid_total", 0.0)),
            "fee_cash_debug": float(row.get("fee_cash_debug", 0.0)),
            "close_reason": close_reason,
            "forced_close_reason": "",
            "is_last": step in {6, 8},
            "is_terminal": step in {6, 8},
            "source_reward_has_profit_lock": True,
        }
        for key in PROFIT_LOCK_DEBUG_FIELDS:
            std[key] = float(row.get(key, 0.0))
        rows.append(std)
    return _ensure_step_reward_schema(pd.DataFrame.from_records(rows))


def _old_reward_step_rows(profit_lock_steps: pd.DataFrame) -> pd.DataFrame:
    old_cols = [
        col for col in STEP_REWARD_COMPONENT_COLUMNS
        if col not in set(PROFIT_LOCK_DEBUG_FIELDS) | {"source_reward_has_profit_lock"}
    ]
    old = profit_lock_steps[old_cols].copy()
    old["checkpoint_label"] = "sample_old_reward"
    old["source_reward_has_profit_lock"] = False
    return _ensure_step_reward_schema(old)


def _sample_trades(label: str) -> pd.DataFrame:
    base_ts = pd.Timestamp("2026-06-22 09:01:00", tz="Asia/Shanghai")
    return pd.DataFrame.from_records([
        {
            "checkpoint_label": label,
            "episode_id": 0,
            "collect_mode": "sample",
            "start_trading_day": 20260622,
            "trading_day": 20260622,
            "direction": "LONG",
            "entry_timestamp": (base_ts + pd.Timedelta(minutes=0)).isoformat(),
            "exit_timestamp": (base_ts + pd.Timedelta(minutes=5)).isoformat(),
            "entry_price": 100.0,
            "exit_price": 90.0,
            "entry_fee": 3.0,
            "exit_fee": 3.0,
            "actual_gross_pnl": -594.0,
            "actual_cost": 6.0,
            "actual_net_pnl": -600.0,
            "exit_reason": "MANUAL",
        },
        {
            "checkpoint_label": label,
            "episode_id": 1,
            "collect_mode": "sample",
            "start_trading_day": 20260622,
            "trading_day": 20260622,
            "direction": "LONG",
            "entry_timestamp": (base_ts + pd.Timedelta(minutes=6)).isoformat(),
            "exit_timestamp": (base_ts + pd.Timedelta(minutes=7)).isoformat(),
            "entry_price": 100.0,
            "exit_price": 140.0,
            "entry_fee": 3.0,
            "exit_fee": 3.0,
            "actual_gross_pnl": 2406.0,
            "actual_cost": 6.0,
            "actual_net_pnl": 2400.0,
            "exit_reason": "MANUAL",
        },
    ])


def _sample_fixed_exit(trades: pd.DataFrame) -> pd.DataFrame:
    fixed = trades[["checkpoint_label", "episode_id", "entry_timestamp", "direction"]].copy()
    fixed["fixed_net_pnl"] = [300.0, 2100.0][:len(fixed)]
    fixed["exit_holding_delta_pnl"] = trades["actual_net_pnl"].to_numpy(dtype=float) - fixed["fixed_net_pnl"].to_numpy(dtype=float)
    return fixed


def _finite_numeric(df: pd.DataFrame) -> bool:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return True
    return bool(np.isfinite(numeric.to_numpy(dtype=float)).all())


def _write_ab_report_schema(path: Path) -> None:
    schema = {
        "schema_version": "jm3_profit_lock_ab_report_v1",
        "required_questions": [
            "Did profit-lock version still trade?",
            "Did it preserve entry quality?",
            "Did profit-lock actually activate?",
            "Did capture ratio for MFE>=1R improve?",
            "Did big-MFE-to-loss decrease?",
            "Did giveback after MFE>=1R decrease?",
            "Did exit/holding delta improve?",
            "Did it avoid over-early exits?",
            "Did overlay avoid dominating old reward?",
        ],
        "required_artifacts": [
            "checkpoint_summary.csv",
            "profit_lock_trade_rollup.csv",
            "profit_lock_component_summary.csv",
            "trade_mfe_capture_summary.csv",
            "mfe_bucket_breakdown.csv",
            "close_reason_breakdown.csv",
            "overlay_dominance_summary.csv",
        ],
        "minimum_columns": {
            "profit_lock_trade_rollup.csv": [
                "checkpoint_label",
                "trade_id",
                "direction",
                "actual_pnl",
                "mfe_R",
                "capture_ratio",
                "big_mfe_to_loss",
                "sum_r_profit_lock_total",
                "overlay_abs_share",
            ]
        },
    }
    path.write_text(json.dumps(schema, ensure_ascii=False, indent=2))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not IN_STEPS.exists():
        raise FileNotFoundError(IN_STEPS)

    smoke = pd.read_csv(IN_STEPS)
    profit_steps = _standard_step_rows(smoke)
    profit_trades = _sample_trades("sample_profit_lock")
    fixed = _sample_fixed_exit(profit_trades)
    paths = write_profit_lock_ab_audit_outputs(OUT, profit_steps, profit_trades, fixed)
    rollup = pd.read_csv(paths["profit_lock_trade_rollup"])

    old_steps = _old_reward_step_rows(profit_steps)
    old_trades = _sample_trades("sample_old_reward")
    old_rollup = build_profit_lock_trade_rollup(
        old_steps,
        old_trades,
        _sample_fixed_exit(old_trades),
    )
    old_rollup.to_csv(OUT / "old_baseline_metric_sample.csv", index=False)

    # User-facing sample names for Step 0.
    profit_steps.head(8).to_csv(OUT / "step_reward_components_sample.csv", index=False)
    rollup.to_csv(OUT / "profit_lock_trade_rollup_sample.csv", index=False)
    pd.read_csv(paths["profit_lock_component_summary"], keep_default_na=False).to_csv(
        OUT / "profit_lock_component_summary_sample.csv", index=False)
    _write_ab_report_schema(OUT / "ab_report_schema.json")

    required = {
        "step_reward_components": OUT / "step_reward_components.csv",
        "profit_lock_trade_rollup": OUT / "profit_lock_trade_rollup.csv",
        "profit_lock_component_summary": OUT / "profit_lock_component_summary.csv",
        "trade_mfe_capture_summary": OUT / "trade_mfe_capture_summary.csv",
        "mfe_bucket_breakdown": OUT / "mfe_bucket_breakdown.csv",
        "close_reason_breakdown": OUT / "close_reason_breakdown.csv",
        "overlay_dominance_summary": OUT / "overlay_dominance_summary.csv",
        "old_baseline_metric_sample": OUT / "old_baseline_metric_sample.csv",
        "ab_report_schema": OUT / "ab_report_schema.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    close_reason_ok = not pd.read_csv(required["close_reason_breakdown"]).empty
    overlay_ok = not pd.read_csv(required["overlay_dominance_summary"]).empty
    old_absent_ok = (
        "source_reward_has_profit_lock" in old_steps.columns
        and not bool(old_steps["source_reward_has_profit_lock"].any())
        and _finite_numeric(old_rollup)
    )
    profit_fields_ok = all(key in profit_steps.columns for key in PROFIT_LOCK_DEBUG_FIELDS)
    finite_ok = all(
        _finite_numeric(pd.read_csv(path, keep_default_na=False))
        for name, path in required.items()
        if name != "ab_report_schema"
    )
    smoke_pass = not missing and close_reason_ok and overlay_ok and old_absent_ok and profit_fields_ok and finite_ok
    classification = (
        "PROFIT_LOCK_AB_AUDIT_SCRIPTS_READY"
        if smoke_pass
        else "PROFIT_LOCK_AB_AUDIT_SCRIPTS_FAIL"
    )
    summary = {
        "classification": classification,
        "smoke_pass": smoke_pass,
        "trained": False,
        "official_validation_run": False,
        "official_test_run": False,
        "inputs": {
            "reward_profit_lock_smoke_steps": str(IN_STEPS),
        },
        "checks": {
            "missing_outputs": missing,
            "profit_lock_fields_present": profit_fields_ok,
            "old_reward_absent_profit_lock_fields_ok": old_absent_ok,
            "numeric_fields_finite": finite_ok,
            "close_reason_breakdown_produced": close_reason_ok,
            "overlay_dominance_summary_produced": overlay_ok,
            "rollup_rows": int(len(rollup)),
            "old_rollup_rows": int(len(old_rollup)),
        },
        "outputs": {name: str(path) for name, path in required.items()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    DOC.write_text(
        "\n".join([
            f"# Profit-Lock A/B Audit Script Fixes - {DATE}",
            "",
            "## Classification",
            "",
            f"`{classification}`",
            "",
            "## Evidence",
            "",
            f"- Input smoke step components: `{IN_STEPS}`",
            "- Step reward components schema was normalized.",
            "- Profit-lock trade rollup and summary CSVs were produced.",
            "- Old reward sample with absent profit-lock fields was normalized with zero profit-lock fields.",
            f"- Numeric finite check: `{finite_ok}`.",
            f"- Close reason breakdown produced: `{close_reason_ok}`.",
            f"- Overlay dominance summary produced: `{overlay_ok}`.",
            "",
            "## Outputs",
            "",
            *[f"- `{path}`" for path in required.values()],
            "- `artifacts/profit_lock_ab_audit_script_fixes/summary.json`",
            "",
            "## Scope",
            "",
            "No training, official validation, official test, reward logic change, env behavior change, obs/action/action_mask change, or execution change was performed.",
            "",
        ])
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if smoke_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
