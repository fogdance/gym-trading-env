#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dreamer_checkpoint_audit import _ensure_step_reward_schema


EPS = 1e-12

COMPONENT_COLUMNS = {
    "reward_total_sum": "reward_total",
    "reward_pnl_sum": "pnl",
    "reward_close_sum": "close",
    "reward_atr_close_sum": "r_atr_close",
    "reward_eod_sum": "eod",
    "reward_invalid_sum": "invalid_total",
    "reward_sl_sum": "sl",
    "reward_mkt_closed_sum": "mkt_closed",
    "raw_total_sum": "raw_total",
}

TRADE_REWARD_COMPONENT_COLUMNS = [
    "checkpoint_label",
    "trade_id",
    "episode_id",
    "trading_day",
    "direction",
    "entry_time",
    "exit_time",
    "actual_pnl",
    "fixed_exit_pnl",
    "exit_holding_delta",
    "mfe_cash",
    "mfe_R",
    "mae_cash",
    "realized_pnl",
    "capture_ratio",
    "giveback_cash",
    "hold_bars",
    *COMPONENT_COLUMNS.keys(),
    "total_score",
    "trade_reward_positive_actual_pnl_negative",
    "trade_reward_negative_actual_pnl_positive",
]

EPISODE_SCORE_PNL_COLUMNS = [
    "checkpoint_label",
    "episode_id",
    "trading_day",
    "score",
    "actual_net_pnl",
    "final_equity",
    "trade_count",
    *COMPONENT_COLUMNS.keys(),
    "score_positive_pnl_negative",
    "score_negative_pnl_positive",
]

CHECKPOINT_COMPONENT_COLUMNS = [
    "checkpoint_label",
    "episodes",
    "trades",
    *COMPONENT_COLUMNS.keys(),
    "total_score",
    "actual_net_pnl",
    "score_positive_pnl_negative_count",
    "score_negative_pnl_positive_count",
    "trade_reward_positive_actual_pnl_negative_count",
    "trade_reward_negative_actual_pnl_positive_count",
    "score_pnl_corr",
]

ACTION_COUNT_COLUMNS = [
    "checkpoint_label",
    "action_field",
    "action_value",
    "action_name",
    "count",
    "ratio",
]

SELECTED_ACTION_NAMES = {
    0: "SHORT",
    1: "FLAT",
    2: "LONG",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _to_timestamp(value) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _safe_int(value, default: int = -1) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _first_present(row: pd.Series, names: Iterable[str], default=""):
    for name in names:
        if name in row.index and not pd.isna(row[name]):
            return row[name]
    return default


def _checkpoint_from_frames(*frames: pd.DataFrame) -> str:
    for frame in frames:
        if "checkpoint_label" in frame.columns and not frame.empty:
            values = frame["checkpoint_label"].dropna().astype(str)
            if len(values):
                return str(values.iloc[0])
    return ""


def _normalize_steps(step_rows: pd.DataFrame, checkpoint_label: str = "") -> pd.DataFrame:
    steps = _ensure_step_reward_schema(step_rows)
    if "checkpoint_label" in steps.columns and checkpoint_label:
        steps.loc[steps["checkpoint_label"].astype(str).eq(""), "checkpoint_label"] = checkpoint_label
    steps["_timestamp_ts"] = pd.to_datetime(steps["timestamp"], utc=True, errors="coerce")
    return steps


def _component_sums(rows: pd.DataFrame) -> dict[str, float]:
    out = {}
    for output_col, source_col in COMPONENT_COLUMNS.items():
        if source_col in rows.columns:
            out[output_col] = float(pd.to_numeric(rows[source_col], errors="coerce").fillna(0.0).sum())
        else:
            out[output_col] = 0.0
    return out


def _scale_cash_from_steps(rows: pd.DataFrame, fallback: float = 300.0) -> float:
    if rows.empty:
        return fallback
    for col in ("profit_lock_scale_cash", "scale_cash"):
        if col in rows.columns:
            values = pd.to_numeric(rows[col], errors="coerce")
            positive = values[values > 0.0]
            if len(positive):
                return float(positive.median())
    return fallback


def _mfe_mae_from_steps(rows: pd.DataFrame) -> tuple[float, float]:
    if rows.empty or "unrealized_pnl" not in rows.columns:
        return 0.0, 0.0
    unrealized = pd.to_numeric(rows["unrealized_pnl"], errors="coerce").dropna()
    if not len(unrealized):
        return 0.0, 0.0
    return max(float(unrealized.max()), 0.0), min(float(unrealized.min()), 0.0)


def build_trade_reward_component_rollup(
    step_rows: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    checkpoint_label: str = "",
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=TRADE_REWARD_COMPONENT_COLUMNS)
    checkpoint_label = checkpoint_label or _checkpoint_from_frames(step_rows, trades)
    steps = _normalize_steps(step_rows, checkpoint_label=checkpoint_label)
    rows: list[dict] = []
    for idx, trade in trades.reset_index(drop=True).iterrows():
        episode_id = _safe_int(_first_present(trade, ["episode_id"], -1))
        entry_time = _first_present(trade, ["entry_time", "entry_timestamp"], "")
        exit_time = _first_present(trade, ["exit_time", "exit_timestamp"], "")
        entry_ts = _to_timestamp(entry_time)
        exit_ts = _to_timestamp(exit_time)
        in_trade = steps[steps["episode_id"].astype(int) == episode_id].copy()
        if pd.notna(entry_ts):
            in_trade = in_trade[in_trade["_timestamp_ts"] >= entry_ts]
        if pd.notna(exit_ts):
            in_trade = in_trade[in_trade["_timestamp_ts"] <= exit_ts]
        components = _component_sums(in_trade)
        total_score = components["reward_total_sum"]
        actual_pnl = _safe_float(_first_present(trade, ["actual_pnl", "actual_net_pnl"], 0.0))
        realized_pnl = _safe_float(_first_present(trade, ["realized_pnl", "actual_gross_pnl", "actual_pnl", "actual_net_pnl"], actual_pnl))
        step_mfe, step_mae = _mfe_mae_from_steps(in_trade)
        scale_cash = _scale_cash_from_steps(in_trade)
        mfe_cash = _safe_float(_first_present(trade, ["mfe_cash", "would_have_mfe_cash"], np.nan), np.nan)
        if not np.isfinite(mfe_cash) or (mfe_cash <= 0.0 and step_mfe > 0.0):
            mfe_cash = step_mfe
        mae_cash = _safe_float(_first_present(trade, ["mae_cash", "would_have_mae_cash"], np.nan), np.nan)
        if not np.isfinite(mae_cash) or (mae_cash == 0.0 and step_mae < 0.0):
            mae_cash = step_mae
        mfe_r = _safe_float(_first_present(trade, ["mfe_R", "would_have_mfe_R"], np.nan), np.nan)
        if not np.isfinite(mfe_r) or (mfe_r <= 0.0 and mfe_cash > 0.0):
            mfe_r = mfe_cash / scale_cash if scale_cash > 0.0 else 0.0
        capture_ratio = _safe_float(_first_present(trade, ["capture_ratio", "would_have_capture_ratio"], np.nan), np.nan)
        if not np.isfinite(capture_ratio) or (capture_ratio == 0.0 and mfe_cash > EPS):
            capture_ratio = realized_pnl / max(mfe_cash, EPS) if mfe_cash > EPS else 0.0
        giveback_cash = _safe_float(_first_present(trade, ["giveback_cash", "would_have_giveback_after_1R"], np.nan), np.nan)
        if not np.isfinite(giveback_cash) or (giveback_cash == 0.0 and mfe_r >= 1.0):
            giveback_cash = max(mfe_cash - max(realized_pnl, 0.0), 0.0) if mfe_r >= 1.0 else 0.0
        row = {
            "checkpoint_label": str(checkpoint_label or _first_present(trade, ["checkpoint_label"], "")),
            "trade_id": _safe_int(_first_present(trade, ["trade_id"], idx), idx),
            "episode_id": episode_id,
            "trading_day": _safe_int(_first_present(trade, ["trading_day", "start_trading_day"], -1)),
            "direction": str(_first_present(trade, ["direction"], "")),
            "entry_time": entry_time,
            "exit_time": exit_time,
            "actual_pnl": actual_pnl,
            "fixed_exit_pnl": _safe_float(_first_present(trade, ["fixed_exit_pnl", "fixed_net_pnl"], np.nan), np.nan),
            "exit_holding_delta": _safe_float(_first_present(trade, ["exit_holding_delta", "exit_holding_delta_pnl"], np.nan), np.nan),
            "mfe_cash": mfe_cash,
            "mfe_R": mfe_r,
            "mae_cash": mae_cash,
            "realized_pnl": realized_pnl,
            "capture_ratio": capture_ratio,
            "giveback_cash": giveback_cash,
            "hold_bars": int(len(in_trade)),
            **components,
            "total_score": total_score,
            "trade_reward_positive_actual_pnl_negative": bool(total_score > 0.0 and actual_pnl < 0.0),
            "trade_reward_negative_actual_pnl_positive": bool(total_score < 0.0 and actual_pnl > 0.0),
        }
        rows.append(row)
    return pd.DataFrame.from_records(rows, columns=TRADE_REWARD_COMPONENT_COLUMNS)


def build_episode_score_pnl_mismatch(
    step_rows: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    checkpoint_label: str = "",
) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame(columns=EPISODE_SCORE_PNL_COLUMNS)
    checkpoint_label = checkpoint_label or _checkpoint_from_frames(step_rows, episodes)
    steps = _normalize_steps(step_rows, checkpoint_label=checkpoint_label)
    grouped_steps = {
        int(ep): part
        for ep, part in steps.groupby("episode_id", dropna=False)
    }
    rows: list[dict] = []
    for _, episode in episodes.reset_index(drop=True).iterrows():
        episode_id = _safe_int(_first_present(episode, ["episode_id"], -1))
        part = grouped_steps.get(episode_id, pd.DataFrame())
        components = _component_sums(part)
        score = _safe_float(_first_present(episode, ["score", "reward_episode_total"], components["reward_total_sum"]))
        actual_pnl = _safe_float(
            _first_present(episode, ["reward_episode_actual_net_pnl", "actual_net_pnl"], np.nan),
            np.nan,
        )
        if not np.isfinite(actual_pnl):
            actual_pnl = _safe_float(components.get("actual_net_pnl", 0.0), 0.0)
        row = {
            "checkpoint_label": str(checkpoint_label or _first_present(episode, ["checkpoint_label"], "")),
            "episode_id": episode_id,
            "trading_day": _safe_int(_first_present(episode, ["start_trading_day", "trading_day"], -1)),
            "score": score,
            "actual_net_pnl": actual_pnl,
            "final_equity": _safe_float(_first_present(episode, ["reward_episode_final_equity", "final_equity"], np.nan), np.nan),
            "trade_count": _safe_float(_first_present(episode, ["reward_episode_trade_count", "trade_count", "trades_closed"], np.nan), np.nan),
            **components,
            "score_positive_pnl_negative": bool(score > 0.0 and actual_pnl < 0.0),
            "score_negative_pnl_positive": bool(score < 0.0 and actual_pnl > 0.0),
        }
        rows.append(row)
    return pd.DataFrame.from_records(rows, columns=EPISODE_SCORE_PNL_COLUMNS)


def build_checkpoint_reward_component_summary(
    episode_mismatch: pd.DataFrame,
    trade_rollup: pd.DataFrame,
) -> pd.DataFrame:
    checkpoints = sorted(
        set(episode_mismatch.get("checkpoint_label", pd.Series(dtype=str)).astype(str))
        | set(trade_rollup.get("checkpoint_label", pd.Series(dtype=str)).astype(str))
    )
    rows = []
    for checkpoint in checkpoints:
        ep = episode_mismatch[episode_mismatch["checkpoint_label"].astype(str) == str(checkpoint)]
        tr = trade_rollup[trade_rollup["checkpoint_label"].astype(str) == str(checkpoint)]
        row = {
            "checkpoint_label": checkpoint,
            "episodes": int(len(ep)),
            "trades": int(len(tr)),
        }
        for output_col in COMPONENT_COLUMNS:
            row[output_col] = float(ep[output_col].sum()) if output_col in ep.columns else 0.0
        row["total_score"] = float(ep["score"].sum()) if "score" in ep.columns else row["reward_total_sum"]
        row["actual_net_pnl"] = float(ep["actual_net_pnl"].sum()) if "actual_net_pnl" in ep.columns else 0.0
        row["score_positive_pnl_negative_count"] = int(ep["score_positive_pnl_negative"].sum()) if "score_positive_pnl_negative" in ep.columns else 0
        row["score_negative_pnl_positive_count"] = int(ep["score_negative_pnl_positive"].sum()) if "score_negative_pnl_positive" in ep.columns else 0
        row["trade_reward_positive_actual_pnl_negative_count"] = int(
            tr["trade_reward_positive_actual_pnl_negative"].sum()
        ) if "trade_reward_positive_actual_pnl_negative" in tr.columns else 0
        row["trade_reward_negative_actual_pnl_positive_count"] = int(
            tr["trade_reward_negative_actual_pnl_positive"].sum()
        ) if "trade_reward_negative_actual_pnl_positive" in tr.columns else 0
        if len(ep) >= 2 and ep["score"].nunique(dropna=True) > 1 and ep["actual_net_pnl"].nunique(dropna=True) > 1:
            row["score_pnl_corr"] = float(ep["score"].corr(ep["actual_net_pnl"]))
        else:
            row["score_pnl_corr"] = np.nan
        rows.append(row)
    return pd.DataFrame.from_records(rows, columns=CHECKPOINT_COMPONENT_COLUMNS)


def build_action_count_summary(step_rows: pd.DataFrame, *, checkpoint_label: str = "") -> pd.DataFrame:
    if step_rows.empty:
        return pd.DataFrame(columns=ACTION_COUNT_COLUMNS)
    steps = _normalize_steps(step_rows, checkpoint_label=checkpoint_label)
    rows = []
    for checkpoint, part in steps.groupby("checkpoint_label", dropna=False):
        for field in ["selected_action", "executed_action"]:
            counts = pd.to_numeric(part[field], errors="coerce").fillna(-1).astype(int).value_counts().sort_index()
            denom = int(counts.sum())
            for value, count in counts.items():
                action_name = (
                    SELECTED_ACTION_NAMES.get(int(value), f"UNKNOWN_{int(value)}")
                    if field == "selected_action"
                    else f"EXEC_{int(value)}"
                )
                rows.append({
                    "checkpoint_label": str(checkpoint),
                    "action_field": field,
                    "action_value": int(value),
                    "action_name": action_name,
                    "count": int(count),
                    "ratio": float(count / denom) if denom else 0.0,
                })
    return pd.DataFrame.from_records(rows, columns=ACTION_COUNT_COLUMNS)


def _discover_checkpoint_dirs(input_dir: Path) -> list[Path]:
    if (input_dir / "step_reward_components.csv").exists() or (input_dir / "dreamer_episodes.csv").exists():
        return [input_dir]
    checkpoints = input_dir / "checkpoints"
    if checkpoints.exists():
        return sorted(p for p in checkpoints.iterdir() if p.is_dir())
    return []


def _load_checkpoint_inputs(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    steps = _read_csv(path / "step_reward_components.csv")
    episodes = _read_csv(path / "dreamer_episodes.csv")
    trades = _read_csv(path / "profit_lock_trade_rollup.csv")
    if trades.empty:
        trades = _read_csv(path / "dreamer_actual_trades.csv")
    checkpoint = path.name
    for frame in (steps, episodes, trades):
        if not frame.empty:
            frame["checkpoint_label"] = checkpoint
    return steps, episodes, trades, checkpoint


def build_reports(input_dir: Path) -> dict[str, pd.DataFrame]:
    trade_reports = []
    episode_reports = []
    action_reports = []
    processed = []
    for checkpoint_dir in _discover_checkpoint_dirs(input_dir):
        steps, episodes, trades, checkpoint = _load_checkpoint_inputs(checkpoint_dir)
        if steps.empty and episodes.empty and trades.empty:
            continue
        trade_reports.append(build_trade_reward_component_rollup(steps, trades, checkpoint_label=checkpoint))
        episode_reports.append(build_episode_score_pnl_mismatch(steps, episodes, checkpoint_label=checkpoint))
        action_reports.append(build_action_count_summary(steps, checkpoint_label=checkpoint))
        processed.append(str(checkpoint_dir))
    trade_reports = [frame for frame in trade_reports if not frame.empty]
    episode_reports = [frame for frame in episode_reports if not frame.empty]
    action_reports = [frame for frame in action_reports if not frame.empty]
    trade_rollup = (
        pd.concat(trade_reports, ignore_index=True)
        if trade_reports
        else pd.DataFrame(columns=TRADE_REWARD_COMPONENT_COLUMNS)
    )
    episode_mismatch = (
        pd.concat(episode_reports, ignore_index=True)
        if episode_reports
        else pd.DataFrame(columns=EPISODE_SCORE_PNL_COLUMNS)
    )
    action_counts = (
        pd.concat(action_reports, ignore_index=True)
        if action_reports
        else pd.DataFrame(columns=ACTION_COUNT_COLUMNS)
    )
    checkpoint_summary = build_checkpoint_reward_component_summary(episode_mismatch, trade_rollup)
    return {
        "trade_reward_component_rollup": trade_rollup,
        "episode_score_pnl_mismatch": episode_mismatch,
        "checkpoint_reward_component_summary": checkpoint_summary,
        "action_count_summary": action_counts,
        "_processed": pd.DataFrame({"checkpoint_dir": processed}),
    }


def write_reports(
    input_dir: Path,
    output_dir: Path,
    *,
    write_sample_aliases: bool = False,
    sample_rows: int = 20,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = build_reports(input_dir)
    paths: dict[str, str] = {}
    for name, frame in reports.items():
        if name.startswith("_"):
            continue
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = str(path)
    if write_sample_aliases:
        sample_trade = output_dir / "sample_trade_reward_component_rollup.csv"
        sample_mismatch = output_dir / "sample_score_pnl_mismatch.csv"
        reports["trade_reward_component_rollup"].head(sample_rows).to_csv(sample_trade, index=False)
        reports["episode_score_pnl_mismatch"].head(sample_rows).to_csv(sample_mismatch, index=False)
        paths["sample_trade_reward_component_rollup"] = str(sample_trade)
        paths["sample_score_pnl_mismatch"] = str(sample_mismatch)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "processed_checkpoint_dirs": reports["_processed"]["checkpoint_dir"].tolist(),
        "rows": {
            name: int(len(frame))
            for name, frame in reports.items()
            if not name.startswith("_")
        },
        "outputs": paths,
    }
    summary_path = output_dir / "old_reward_metric_reports_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    paths["old_reward_metric_reports_summary"] = str(summary_path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Build old-reward component and score/PnL audit reports.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--write-sample-aliases", action="store_true")
    parser.add_argument("--sample-rows", type=int, default=20)
    args = parser.parse_args()
    paths = write_reports(
        args.input_dir,
        args.output_dir,
        write_sample_aliases=bool(args.write_sample_aliases),
        sample_rows=int(args.sample_rows),
    )
    print(json.dumps(paths, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
