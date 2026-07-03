#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/jm3_old_reward_baseline_reaudit"
CKPT_DIRS = [
    BASE / "checkpoints/step_000000500000",
    BASE / "checkpoints/step_000000700000",
    BASE / "checkpoints/step_000000900000",
    BASE / "checkpoints/latest_000000982672",
]
INITIAL_EQUITY = 10000.0
FALLBACK_R_CASH = 300.0


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def nested(data: dict[str, Any], *keys: str, default=None):
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def safe_float(value, default=np.nan) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def max_drawdown_from_daily(trades: pd.DataFrame) -> tuple[float, float]:
    if trades.empty or "trading_day" not in trades or "actual_net_pnl" not in trades:
        return 0.0, 0.0
    daily = trades.groupby("trading_day", dropna=False)["actual_net_pnl"].sum().sort_index()
    equity = INITIAL_EQUITY + daily.cumsum()
    peak = equity.cummax()
    drawdown = peak - equity
    if not len(drawdown):
        return 0.0, 0.0
    max_dd = float(drawdown.max())
    peak_at_max = float(peak.iloc[drawdown.to_numpy().argmax()])
    max_dd_pct = max_dd / peak_at_max * 100.0 if peak_at_max else 0.0
    return max_dd, max_dd_pct


def mfe_bucket(mfe_r) -> str:
    value = safe_float(mfe_r, 0.0)
    if value < 1.0:
        return "<1R"
    if value < 2.0:
        return "1R-2R"
    if value < 3.0:
        return "2R-3R"
    return ">=3R"


def enrich_would_have(label: str, trades: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    out = trades.copy()
    if "checkpoint_label" not in out:
        out.insert(0, "checkpoint_label", label)
    for col in ("entry_timestamp", "exit_timestamp"):
        if col in out:
            out[f"{col}_dt"] = pd.to_datetime(out[col], utc=True, errors="coerce")
    if not steps.empty and "timestamp" in steps:
        steps = steps.copy()
        steps["timestamp_dt"] = pd.to_datetime(steps["timestamp"], utc=True, errors="coerce")

    rows: list[dict[str, Any]] = []
    for _, trade in out.iterrows():
        step_part = steps
        episode_id = trade.get("episode_id")
        entry_dt = trade.get("entry_timestamp_dt", pd.NaT)
        exit_dt = trade.get("exit_timestamp_dt", pd.NaT)
        if not step_part.empty and "episode_id" in step_part and pd.notna(episode_id):
            step_part = step_part[step_part["episode_id"] == episode_id]
        if not step_part.empty and pd.notna(entry_dt) and pd.notna(exit_dt):
            step_part = step_part[
                (step_part["timestamp_dt"] >= entry_dt) & (step_part["timestamp_dt"] <= exit_dt)
            ]

        unrealized = (
            pd.to_numeric(step_part.get("unrealized_pnl", pd.Series(dtype=float)), errors="coerce")
            if not step_part.empty
            else pd.Series(dtype=float)
        )
        scale_cash = FALLBACK_R_CASH
        for col in ("profit_lock_scale_cash", "scale_cash"):
            if not step_part.empty and col in step_part:
                scale_values = pd.to_numeric(step_part[col], errors="coerce")
                positive = scale_values[scale_values > 0]
                if len(positive):
                    scale_cash = float(positive.median())
                    break

        if len(unrealized.dropna()):
            mfe_cash = max(float(unrealized.max()), 0.0)
            mae_cash = min(float(unrealized.min()), 0.0)
        else:
            gross = safe_float(trade.get("actual_gross_pnl"), 0.0)
            mfe_cash = max(gross, 0.0)
            mae_cash = min(gross, 0.0)

        actual_pnl = safe_float(trade.get("actual_net_pnl"), 0.0)
        mfe_r = mfe_cash / scale_cash if scale_cash > 0 else 0.0
        capture = actual_pnl / mfe_cash if mfe_cash > 1e-12 else 0.0
        giveback = max(mfe_cash - max(actual_pnl, 0.0), 0.0) if mfe_r >= 1.0 else 0.0
        rows.append(
            {
                "would_have_scale_cash": scale_cash,
                "would_have_mfe_cash": mfe_cash,
                "would_have_mfe_R": mfe_r,
                "would_have_mae_cash": mae_cash,
                "would_have_capture_ratio": capture,
                "would_have_giveback_after_1R": giveback,
                "would_have_profit_lock_active": bool(mfe_r >= 1.0),
                "would_have_mfe_ge_1R": bool(mfe_r >= 1.0),
                "would_have_mfe_ge_2R": bool(mfe_r >= 2.0),
                "would_have_mfe_ge_3R": bool(mfe_r >= 3.0),
                "would_have_big_mfe_to_loss": bool(mfe_r >= 1.0 and actual_pnl <= 0.0),
                "would_have_low_capture": bool(mfe_r >= 1.0 and capture < 0.5),
                "would_have_profit_lock_trade_population": bool(mfe_r >= 1.0),
            }
        )
    result = pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return result.drop(
        columns=[c for c in ("entry_timestamp_dt", "exit_timestamp_dt") if c in result],
        errors="ignore",
    )


def summarize_group(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out_cols = cols + [
        "trades",
        "actual_net_pnl",
        "mean_actual_pnl",
        "mfe_ge_1R",
        "mfe_ge_2R",
        "mfe_ge_3R",
        "big_mfe_to_loss",
        "big_mfe_low_capture",
        "mean_mfe_R",
        "median_mfe_R",
        "mean_capture_ratio_mfe_ge_1R",
        "mean_giveback_after_1R",
        "mean_hold_bars",
    ]
    if df.empty:
        return pd.DataFrame(columns=out_cols)

    rows: list[dict[str, Any]] = []
    grouped = df.groupby(cols, dropna=False) if cols else [((), df)]
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        actual = pd.to_numeric(
            group.get("actual_net_pnl", group.get("actual_pnl", pd.Series(dtype=float))),
            errors="coerce",
        )
        active = group[group["would_have_mfe_ge_1R"].astype(bool)]
        row = dict(zip(cols, key))
        row.update(
            {
                "trades": int(len(group)),
                "actual_net_pnl": float(actual.sum()) if len(actual) else 0.0,
                "mean_actual_pnl": float(actual.mean()) if len(actual) else np.nan,
                "mfe_ge_1R": int(group["would_have_mfe_ge_1R"].astype(bool).sum()),
                "mfe_ge_2R": int(group["would_have_mfe_ge_2R"].astype(bool).sum()),
                "mfe_ge_3R": int(group["would_have_mfe_ge_3R"].astype(bool).sum()),
                "big_mfe_to_loss": int(group["would_have_big_mfe_to_loss"].astype(bool).sum()),
                "big_mfe_low_capture": int(group["would_have_low_capture"].astype(bool).sum()),
                "mean_mfe_R": float(pd.to_numeric(group["would_have_mfe_R"], errors="coerce").mean()),
                "median_mfe_R": float(pd.to_numeric(group["would_have_mfe_R"], errors="coerce").median()),
                "mean_capture_ratio_mfe_ge_1R": float(
                    pd.to_numeric(active["would_have_capture_ratio"], errors="coerce").mean()
                )
                if len(active)
                else np.nan,
                "mean_giveback_after_1R": float(
                    pd.to_numeric(active["would_have_giveback_after_1R"], errors="coerce").mean()
                )
                if len(active)
                else np.nan,
                "mean_hold_bars": float(
                    pd.to_numeric(
                        group.get("hold_bars", group.get("fixed_holding_bars", pd.Series(dtype=float))),
                        errors="coerce",
                    ).mean()
                )
                if len(group)
                else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=out_cols)


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    def fmt(value) -> str:
        if isinstance(value, float):
            return "" if np.isnan(value) else f"{value:.6g}"
        return str(value)

    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def main() -> int:
    checkpoint_rows: list[dict[str, Any]] = []
    all_trades: list[pd.DataFrame] = []
    all_rollups: list[pd.DataFrame] = []

    for ckpt_dir in CKPT_DIRS:
        label = ckpt_dir.name
        summary_path = ckpt_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text())
        trades = read_csv(ckpt_dir / "dreamer_actual_trades.csv")
        steps = read_csv(ckpt_dir / "step_reward_components.csv")
        rollup = read_csv(ckpt_dir / "profit_lock_trade_rollup.csv")
        enriched = enrich_would_have(label, trades, steps)
        if not enriched.empty:
            all_trades.append(enriched)
        if not rollup.empty and not enriched.empty:
            join_cols = [
                col
                for col in ("episode_id", "entry_timestamp", "exit_timestamp")
                if col in rollup and col in enriched
            ]
            extra_cols = [col for col in enriched if col.startswith("would_have_")]
            merged = rollup.merge(enriched[join_cols + extra_cols], on=join_cols, how="left")
            all_rollups.append(merged)
        elif not enriched.empty:
            all_rollups.append(
                enriched.rename(columns={"actual_net_pnl": "actual_pnl", "exit_reason": "close_reason"})
            )
        else:
            all_rollups.append(pd.DataFrame({"checkpoint_label": pd.Series(dtype=str)}))

        actual_val = nested(summary, "actual_dreamer", "by_split_role", "validation", default={}) or {}
        fixed_val = (
            nested(summary, "dreamer_entry_fixed_exit_candidate_join", "by_split_role", "validation", default={})
            or {}
        )
        actual_matched = (
            nested(
                summary,
                "dreamer_entry_fixed_exit_candidate_join",
                "actual_matched_by_split_role",
                "validation",
                default={},
            )
            or {}
        )
        delta_val = (
            nested(summary, "dreamer_entry_fixed_exit_candidate_join", "delta_by_split_role", "validation", default={})
            or {}
        )
        exec_val = (
            nested(summary, "dreamer_entry_fixed_exit_executable_nonoverlap", "by_split_role", "validation", default={})
            or nested(summary, "dreamer_entry_fixed_exit_executable_nonoverlap", "metrics", default={})
            or {}
        )
        matched_random = nested(summary, "matched_random_fixed_exit", default={}) or {}
        oracle = nested(summary, "constrained_oracle_fixed_exit", default={}) or {}

        validation_days = int(nested(summary, "episodes", "by_start_role", "validation", default=0) or 0)
        if validation_days <= 0 and not trades.empty and "trading_day" in trades:
            validation_days = int(trades["trading_day"].nunique())
        daily_counts = (
            trades.groupby("trading_day").size()
            if not trades.empty and "trading_day" in trades
            else pd.Series(dtype=int)
        )
        daily_pnl = (
            trades.groupby("trading_day")["actual_net_pnl"].sum()
            if not trades.empty and "trading_day" in trades and "actual_net_pnl" in trades
            else pd.Series(dtype=float)
        )
        max_dd_cash, max_dd_pct = max_drawdown_from_daily(trades)
        checkpoint_rows.append(
            {
                "checkpoint_label": label,
                "checkpoint_path": str(ckpt_dir),
                "episodes_collected": int(nested(summary, "episodes", "collected", default=0) or 0),
                "validation_days": validation_days,
                "trade_count": int(actual_val.get("trades", 0) or 0),
                "actual_net_pnl": safe_float(actual_val.get("net_pnl"), 0.0),
                "final_equity": INITIAL_EQUITY + safe_float(actual_val.get("net_pnl"), 0.0),
                "negative_days": int((daily_pnl < 0).sum()) if len(daily_pnl) else 0,
                "negative_day_ratio": float((daily_pnl < 0).sum() / validation_days)
                if validation_days
                else np.nan,
                "profit_factor": actual_val.get("profit_factor"),
                "max_drawdown_cash": max_dd_cash,
                "max_drawdown_pct": max_dd_pct,
                "daily_quota_ratio": float((daily_counts >= 3).sum() / validation_days)
                if validation_days
                else np.nan,
                "no_trade_day_ratio": safe_float(actual_val.get("no_trade_day_ratio"), np.nan),
                "long_trades": int(actual_val.get("long_trades", 0) or 0),
                "short_trades": int(actual_val.get("short_trades", 0) or 0),
                "fixed_candidate_pnl": safe_float(fixed_val.get("net_pnl"), 0.0),
                "actual_matched_pnl": safe_float(actual_matched.get("net_pnl"), 0.0),
                "exit_holding_delta": safe_float(delta_val.get("net_pnl"), 0.0),
                "executable_fixed_pnl": safe_float(exec_val.get("net_pnl"), 0.0),
                "executable_fixed_trades": int(exec_val.get("trades", 0) or 0),
                "matched_random_mean": safe_float(matched_random.get("net_pnl_mean"), np.nan),
                "matched_random_p05": safe_float(matched_random.get("net_pnl_p05"), np.nan),
                "matched_random_p50": safe_float(matched_random.get("net_pnl_p50"), np.nan),
                "matched_random_p95": safe_float(matched_random.get("net_pnl_p95"), np.nan),
                "oracle_fixed_pnl": safe_float(oracle.get("net_pnl"), np.nan),
                "fixed_join_matched_trades": int(
                    nested(summary, "dreamer_entry_fixed_exit_candidate_join", "matched_trades", default=0)
                    or 0
                ),
                "fixed_join_unmatched_trades": int(
                    nested(summary, "dreamer_entry_fixed_exit_candidate_join", "unmatched_trades", default=0)
                    or 0
                ),
                "audit_protocol_caveat": "fallback_full_validation_old_checkpoint_same_protocol",
            }
        )

    checkpoint_summary = pd.DataFrame(checkpoint_rows)
    combined_trades = pd.concat(all_trades, ignore_index=True, sort=False) if all_trades else pd.DataFrame()
    combined_rollup = pd.concat(all_rollups, ignore_index=True, sort=False) if all_rollups else pd.DataFrame()
    for df in (combined_trades, combined_rollup):
        if df.empty:
            continue
        if "month" not in df and "trading_day" in df:
            df["month"] = df["trading_day"].astype(str).str.slice(0, 6)
        if "would_have_mfe_R" in df:
            df["would_have_mfe_bucket"] = df["would_have_mfe_R"].map(mfe_bucket)
        reason = df.get("close_reason", df.get("exit_reason", pd.Series("", index=df.index))).fillna("").astype(str)
        df["eod_flag"] = np.where(
            reason.str.contains("EOD|SESSION_END|TRUNCATE_SESSION_END", case=False, regex=True),
            "EOD",
            "NON_EOD",
        )

    if not combined_trades.empty:
        combined_trades["close_reason_for_summary"] = combined_trades.get(
            "exit_reason", combined_trades.get("close_reason", pd.Series("", index=combined_trades.index))
        ).fillna("")

    mfe_summary = pd.concat(
        [
            summarize_group(combined_trades, ["checkpoint_label"]),
            summarize_group(combined_trades, ["checkpoint_label", "month"])
            if not combined_trades.empty and "month" in combined_trades
            else pd.DataFrame(),
            summarize_group(combined_trades, ["checkpoint_label", "direction"])
            if not combined_trades.empty and "direction" in combined_trades
            else pd.DataFrame(),
        ],
        ignore_index=True,
        sort=False,
    )
    monthly = (
        summarize_group(combined_trades, ["checkpoint_label", "month"])
        if not combined_trades.empty and "month" in combined_trades
        else pd.DataFrame()
    )
    mfe_bucket_summary = (
        summarize_group(combined_trades, ["checkpoint_label", "would_have_mfe_bucket"])
        if not combined_trades.empty and "would_have_mfe_bucket" in combined_trades
        else pd.DataFrame()
    )
    close_reason = (
        summarize_group(combined_trades, ["checkpoint_label", "close_reason_for_summary", "eod_flag"])
        if not combined_trades.empty
        else pd.DataFrame()
    )
    direction = (
        summarize_group(combined_trades, ["checkpoint_label", "direction"])
        if not combined_trades.empty and "direction" in combined_trades
        else pd.DataFrame()
    )
    would_have = summarize_group(combined_trades, ["checkpoint_label"])
    if not would_have.empty:
        extras = []
        for label, group in combined_trades.groupby("checkpoint_label", dropna=False):
            active = group[group["would_have_profit_lock_active"].astype(bool)]
            extras.append(
                {
                    "checkpoint_label": label,
                    "would_have_profit_lock_trade_population": int(len(active)),
                    "would_have_profit_lock_active_ratio": float(len(active) / len(group))
                    if len(group)
                    else np.nan,
                    "would_have_total_giveback_after_1R": float(active["would_have_giveback_after_1R"].sum())
                    if len(active)
                    else 0.0,
                    "would_have_mean_capture_ratio": float(active["would_have_capture_ratio"].mean())
                    if len(active)
                    else np.nan,
                }
            )
        would_have = would_have.merge(pd.DataFrame(extras), on="checkpoint_label", how="left")

    BASE.mkdir(parents=True, exist_ok=True)
    checkpoint_summary.to_csv(BASE / "checkpoint_summary.csv", index=False)
    combined_rollup.to_csv(BASE / "old_baseline_trade_rollup.csv", index=False)
    mfe_summary.to_csv(BASE / "old_baseline_mfe_capture_summary.csv", index=False)
    close_reason.to_csv(BASE / "old_baseline_close_reason_breakdown.csv", index=False)
    monthly.to_csv(BASE / "old_baseline_monthly_breakdown.csv", index=False)
    mfe_bucket_summary.to_csv(BASE / "old_baseline_mfe_bucket_breakdown.csv", index=False)
    would_have.to_csv(BASE / "old_baseline_would_have_profit_lock_summary.csv", index=False)
    direction.to_csv(BASE / "old_baseline_direction_breakdown.csv", index=False)

    best = (
        checkpoint_summary.sort_values("actual_net_pnl", ascending=False).iloc[0].to_dict()
        if len(checkpoint_summary)
        else {}
    )
    summary = {
        "classification": "JM3_OLD_BASELINE_REAUDIT_READY_WITH_CAVEATS",
        "generated_at": "2026-07-03",
        "strict_scope": {
            "trained": False,
            "official_test": False,
            "policy_modified": False,
            "reward_modified_for_old_baseline": False,
        },
        "old_training_root": "/data/logdir/action-mask-wf1-ms-seed-c-20260615",
        "old_live_root": "/data/logdir/future_jm3_live_seed_c_700k_20260618",
        "env_config_path": "artifacts/jm3_pnl_aligned_no_trade_root_cause/old_baseline_full_eval_old_reward_env.yaml",
        "entry_eval_dir": "artifacts/entry_eval/entry_eval_jm_walk_forward_20240603_20251202_signal_close_v1",
        "checkpoints": checkpoint_summary.to_dict(orient="records"),
        "best_by_actual_net_pnl": best,
        "caveats": [
            "Old checkpoint was re-audited with the same fallback validation protocol used in prior no-trade audit because the original old training env data does not align with the current validation role map.",
            "Would-have profit-lock metrics are diagnostic-only and were reconstructed from old reward step unrealized_pnl; the old policy was not trained with profit-lock.",
            "Would-have 1R uses exported profit_lock_scale_cash when present, then scale_cash, then fallback reward_scale_cash=300.0.",
        ],
    }
    (BASE / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    cols = [
        "checkpoint_label",
        "episodes_collected",
        "trade_count",
        "actual_net_pnl",
        "fixed_candidate_pnl",
        "executable_fixed_pnl",
        "matched_random_mean",
        "oracle_fixed_pnl",
        "exit_holding_delta",
        "no_trade_day_ratio",
        "daily_quota_ratio",
        "negative_days",
        "profit_factor",
    ]
    doc = ROOT / "docs/jm3_old_reward_baseline_reaudit_20260703.md"
    lines = [
        "# JM3 Old Reward Baseline Re-Audit - 2026-07-03",
        "",
        "Classification: `JM3_OLD_BASELINE_REAUDIT_READY_WITH_CAVEATS`",
        "",
        "## Scope",
        "",
        "- No Dreamer training was run.",
        "- No official test replay was run.",
        "- Old checkpoint, reward, observation, action, action mask, and execution behavior were not modified.",
        "- Audit was offline and used CUDA for policy replay.",
        "",
        "## Evidence",
        "",
        "- Old training root: `/data/logdir/action-mask-wf1-ms-seed-c-20260615`.",
        f"- Old retained checkpoints audited: `{', '.join(path.name for path in CKPT_DIRS)}`.",
        "- Eval config: `artifacts/jm3_pnl_aligned_no_trade_root_cause/old_baseline_full_eval_old_reward_env.yaml`.",
        "- Reward function: `futures_intraday_reward_function`.",
        "- Entry eval protocol: `artifacts/entry_eval/entry_eval_jm_walk_forward_20240603_20251202_signal_close_v1`.",
        "",
        "## Checkpoint Summary",
        "",
        markdown_table(checkpoint_summary, cols),
        "",
        "## Caveats",
        "",
        "- This is a fallback same-protocol validation re-audit, not strict proof on the original old training split layout. Prior audit evidence showed the old training env data did not align with the current validation role map.",
        "- Would-have profit-lock metrics are diagnostic-only. They were reconstructed from old reward replay step `unrealized_pnl` and trade intervals; old policy training did not see profit-lock reward.",
        "- Would-have 1R uses exported `profit_lock_scale_cash` when present, then `scale_cash`, then fallback `trading.reward_scale_cash=300.0`.",
        "- Margin requirement termination logs appeared for 900k/latest during replay; those are old-policy/env behavior observations, not audit-script crashes.",
        "",
        "## Outputs",
        "",
        "- `checkpoint_summary`: `artifacts/jm3_old_reward_baseline_reaudit/checkpoint_summary.csv`",
        "- `old_baseline_trade_rollup`: `artifacts/jm3_old_reward_baseline_reaudit/old_baseline_trade_rollup.csv`",
        "- `old_baseline_mfe_capture_summary`: `artifacts/jm3_old_reward_baseline_reaudit/old_baseline_mfe_capture_summary.csv`",
        "- `old_baseline_close_reason_breakdown`: `artifacts/jm3_old_reward_baseline_reaudit/old_baseline_close_reason_breakdown.csv`",
        "- `old_baseline_monthly_breakdown`: `artifacts/jm3_old_reward_baseline_reaudit/old_baseline_monthly_breakdown.csv`",
        "- `old_baseline_mfe_bucket_breakdown`: `artifacts/jm3_old_reward_baseline_reaudit/old_baseline_mfe_bucket_breakdown.csv`",
        "- `old_baseline_would_have_profit_lock_summary`: `artifacts/jm3_old_reward_baseline_reaudit/old_baseline_would_have_profit_lock_summary.csv`",
        "",
        "## Result",
        "",
        "The old baseline A-group metrics are available for final profit-lock A/B comparison, with the fallback and would-have caveats above.",
    ]
    doc.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "classification": summary["classification"],
                "checkpoint_rows": len(checkpoint_summary),
                "trade_rows": len(combined_trades),
                "best_checkpoint": best.get("checkpoint_label"),
                "best_actual_net_pnl": best.get("actual_net_pnl"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
