#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from decimal import Decimal
from functools import partial as bind
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@dataclass(frozen=True)
class AuditPaths:
    dreamer_root: Path
    run_logdir: Path
    checkpoint: Path
    entry_eval_dir: Path
    output_dir: Path
    env_config_path: Path | None = None


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Decimal):
        return float(value)
    if pd.isna(value):
        return None
    return str(value)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default))


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_run_config(run_logdir: Path) -> dict:
    path = run_logdir / "config.yaml"
    if not path.exists():
        return {}
    try:
        import ruamel.yaml as yaml
        data = yaml.YAML(typ="safe").load(path.read_text())
        return data or {}
    except Exception:
        return {}


def _read_retention_metadata(run_logdir: Path) -> dict:
    path = run_logdir / "ckpt_retained" / "retention_manifest.json"
    if not path.exists():
        return {}
    try:
        return dict((json.loads(path.read_text()) or {}).get("metadata") or {})
    except Exception:
        return {}


def _seed_metadata(
    paths: "AuditPaths",
    *,
    matched_random_seed: int,
) -> dict:
    run_config = _read_run_config(paths.run_logdir)
    config_protocol = dict(run_config.get("seed_protocol") or {})
    retention_protocol = _read_retention_metadata(paths.run_logdir)
    if config_protocol and retention_protocol:
        for key, value in retention_protocol.items():
            if key in config_protocol and config_protocol[key] != value:
                raise ValueError(
                    f"Seed protocol mismatch for {key}: "
                    f"config={config_protocol[key]!r} retention={value!r}")
    protocol = config_protocol or retention_protocol
    if not protocol:
        raise ValueError(
            "Dreamer run is missing seed_protocol in config.yaml and "
            f"ckpt_retained/retention_manifest.json: {paths.run_logdir}")
    protocol_random_seed = protocol.get("matched_random_seed")
    if (
        protocol_random_seed is not None
        and int(protocol_random_seed) != int(matched_random_seed)
    ):
        raise ValueError(
            "Seed protocol mismatch for matched_random_seed: "
            f"metadata={protocol_random_seed!r} cli={matched_random_seed!r}")
    return {
        "experiment_seed": protocol.get("experiment_seed", run_config.get("experiment_seed")),
        "dreamer_seed": protocol.get("dreamer_seed", run_config.get("seed")),
        "train_env_seed": protocol.get("train_env_seed"),
        "replay_seed": protocol.get("replay_seed"),
        "eval_env_seed": protocol.get("eval_env_seed"),
        "matched_random_seed": int(matched_random_seed),
        "seed_protocol": protocol,
        "source": {
            "run_config": str(paths.run_logdir / "config.yaml"),
            "retention_manifest": str(
                paths.run_logdir / "ckpt_retained" / "retention_manifest.json"),
            "matched_random_seed": "audit CLI --matched-random-seed",
        },
    }


def _read_latest_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        return path
    target = path.read_text().strip()
    resolved = path.parent / target
    if not resolved.is_dir():
        raise FileNotFoundError(f"Cannot resolve checkpoint pointer {path} -> {resolved}")
    return resolved


def _read_table(base: Path, stem: str) -> pd.DataFrame:
    parquet = base / f"{stem}.parquet"
    csv = base / f"{stem}.csv"
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except Exception:
            if csv.exists():
                return pd.read_csv(csv)
            from gym_trading_env.research.entry_dataset import read_parquet
            return read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing {stem}.parquet or {stem}.csv under {base}")


def _to_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert("Asia/Shanghai")


def _unwrap_base_env(env):
    cur = env
    seen = set()
    while hasattr(cur, "env") and id(cur) not in seen:
        seen.add(id(cur))
        nxt = cur.env
        if nxt is cur:
            break
        cur = nxt
    return cur


def _scalar(value, default=None):
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return arr.item()
        return arr.reshape(-1)[-1].item()
    except Exception:
        return default


def _name(value) -> str | None:
    if value is None:
        return None
    return getattr(value, "name", str(value))


def _timestamp_text(value) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _zero_action(act_space: dict) -> dict:
    acts = {k: np.zeros(v.shape, v.dtype) for k, v in act_space.items()}
    acts["reset"] = np.asarray(True)
    return acts


def _batch_obs(obs: dict, obs_space: dict) -> dict:
    return {
        k: np.expand_dims(np.asarray(obs[k]), 0)
        for k in obs_space
        if k in obs
    }


def _decision_meta_before_step(base_env, action: dict) -> dict:
    if bool(_scalar(action.get("reset"), False)):
        return {}
    action_index = int(_scalar(action.get("action"), -1))
    store = base_env.bar_source.store
    row = int(getattr(base_env, "current_step", -1))
    meta = {
        "decision_row": row,
        "decision_timestamp": _timestamp_text(store.index[row]) if row >= 0 else None,
        "decision_trading_day": int(store.row_trading_day[row]) if row >= 0 else None,
        "requested_action_index": action_index,
    }
    table = getattr(base_env, "_pending_transition_table", None)
    if table is None or action_index < 0 or action_index >= len(table.decisions):
        return meta
    decision = table.for_index(action_index)
    quote = decision.quote
    meta.update({
        "requested_target": _name(decision.requested_target),
        "planned_action": _name(decision.planned_action),
        "decision_allowed": bool(decision.allowed),
        "decision_reason": str(decision.reason),
        "decision_result_code": _name(decision.result_code),
    })
    if quote is not None:
        meta.update({
            "scheduled_entry_timestamp": _timestamp_text(quote.timestamp),
            "entry_execution_price": float(quote.exec_price),
            "decision_action_price": float(quote.action_price),
            "open_side": _name(quote.open_side),
            "close_side": _name(quote.close_side),
        })
    return meta


def _select_start_rows(
    base_env,
    *,
    role_by_day: dict[int, str],
    roles: set[str],
    start_clock: str,
    episode_cap: int,
) -> list[int]:
    rows = base_env.bar_source.store.candidate_start_rows_by_clock(start_clock)
    selected: list[int] = []
    for row in rows:
        row_i = int(row)
        day = int(base_env.bar_source.store.row_trading_day[row_i])
        role = role_by_day.get(day, "outside_split")
        if roles and role not in roles:
            continue
        selected.append(row_i)
    if episode_cap > 0:
        selected = selected[:episode_cap]
    return selected


def _role_from_split(days: Iterable[int], split_manifest: list[dict]) -> dict[int, str]:
    if not split_manifest:
        return {int(day): "unknown" for day in days}
    # Use the last walk-forward fold as the overall chronological role split.
    final = split_manifest[-1]
    roles = {}
    for role_key, role_name in (
        ("train_days", "train"),
        ("validation_days", "validation"),
        ("test_days", "test"),
    ):
        for day in final.get(role_key, []):
            roles[int(day)] = role_name
    return {int(day): roles.get(int(day), "outside_split") for day in days}


def _month_from_day(day: int) -> int:
    return int(int(day) // 100)


def _direction_from_operation(operation: str) -> str | None:
    op = str(operation).upper()
    if "LONG" in op:
        return "LONG"
    if "SHORT" in op:
        return "SHORT"
    return None


def _direction_from_meta(meta: dict[str, Any]) -> str | None:
    side = str(meta.get("open_side") or meta.get("requested_target") or "").upper()
    if "LONG" in side:
        return "LONG"
    if "SHORT" in side:
        return "SHORT"
    planned = str(meta.get("planned_action") or "").upper()
    if "LONG" in planned and ("OPEN" in planned or "FLIP" in planned):
        return "LONG"
    if "SHORT" in planned and ("OPEN" in planned or "FLIP" in planned):
        return "SHORT"
    return None


def _is_open(operation: str) -> bool:
    return "OPEN" in str(operation).upper()


def _is_close(operation: str) -> bool:
    return "CLOSE" in str(operation).upper()


def _as_float(value) -> float:
    try:
        return float(Decimal(str(value)))
    except Exception:
        return float("nan")


def _records_to_trades(
    records: list[dict],
    *,
    episode_id: int,
    episode_meta: dict,
    record_meta_by_index: dict[int, dict] | None = None,
) -> list[dict]:
    trades = []
    pending = None
    record_meta_by_index = record_meta_by_index or {}
    for idx, rec in enumerate(records):
        operation = str(rec.get("operation_type", ""))
        direction = _direction_from_operation(operation)
        if direction is None:
            continue
        if _is_open(operation):
            decision_meta = dict(record_meta_by_index.get(idx, {}))
            action_direction = _direction_from_meta(decision_meta)
            pending = {
                "episode_id": episode_id,
                "direction": direction,
                "decision_direction": action_direction,
                "decision_row": decision_meta.get("decision_row"),
                "decision_timestamp": decision_meta.get("decision_timestamp"),
                "decision_trading_day": decision_meta.get("decision_trading_day"),
                "requested_action_index": decision_meta.get("requested_action_index"),
                "requested_target": decision_meta.get("requested_target"),
                "planned_action": decision_meta.get("planned_action"),
                "decision_allowed": decision_meta.get("decision_allowed"),
                "decision_reason": decision_meta.get("decision_reason"),
                "scheduled_entry_timestamp": decision_meta.get("scheduled_entry_timestamp"),
                "entry_execution_price": decision_meta.get("entry_execution_price"),
                "decision_action_price": decision_meta.get("decision_action_price"),
                "entry_record_index": idx,
                "entry_timestamp": rec.get("timestamp"),
                "entry_operation_type": operation,
                "entry_price": _as_float(rec.get("open_price", 0)),
                "entry_fee": _as_float(rec.get("fee", 0)),
                "entry_position_size": _as_float(rec.get("position_size", 0)),
                **episode_meta,
            }
            continue
        if _is_close(operation):
            if pending is None:
                continue
            close_direction = direction
            if close_direction != pending["direction"]:
                # Single-position env should not hit this, but do not fabricate a pair.
                pending = None
                continue
            close_fee = _as_float(rec.get("fee", 0))
            gross_pnl = _as_float(rec.get("pnl", 0))
            net_pnl = gross_pnl - float(pending["entry_fee"]) - close_fee
            meta = rec.get("meta", {}) or {}
            trades.append({
                **pending,
                "exit_record_index": idx,
                "exit_timestamp": rec.get("timestamp"),
                "exit_operation_type": operation,
                "exit_reason": meta.get("reason", operation),
                "exit_price": _as_float(rec.get("close_price", 0)),
                "exit_fee": close_fee,
                "actual_gross_pnl": gross_pnl,
                "actual_cost": float(pending["entry_fee"]) + close_fee,
                "actual_net_pnl": net_pnl,
            })
            pending = None
    return trades


def _strategy_metrics(trades: pd.DataFrame, *, all_days: Iterable[int], pnl_col: str) -> dict:
    days = np.asarray(list(all_days), dtype=np.int64)
    if trades.empty:
        zero_ratio = 1.0 if len(days) else None
        return {
            "trades": 0,
            "net_pnl": 0.0,
            "expectancy": None,
            "win_rate": None,
            "profit_factor": None,
            "profit_factor_status": "not_computed",
            "no_trade_day_ratio": zero_ratio,
            "mean_trades_per_day": 0.0 if len(days) else None,
            "max_trades_per_day": 0 if len(days) else None,
            "zero_trade_day_ratio": zero_ratio,
            "one_trade_day_ratio": 0.0 if len(days) else None,
            "two_trade_day_ratio": 0.0 if len(days) else None,
            "three_trade_day_ratio": 0.0 if len(days) else None,
            "trades_per_day": {"0": int(len(days))} if len(days) else {},
            "trades_per_day_ratio": {"0": 1.0} if len(days) else {},
            "long_trades": 0,
            "short_trades": 0,
            "largest_winning_trade": None,
            "largest_losing_trade": None,
        }
    pnl = trades[pnl_col].to_numpy(dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    if len(losses) == 0 and len(wins) == 0:
        pf, pf_status = None, "not_computed"
    elif len(losses) == 0:
        pf, pf_status = "inf", "no_losing_trades"
    elif wins.sum() <= 0:
        pf, pf_status = 0.0, "no_winning_trades"
    else:
        pf, pf_status = float(wins.sum() / abs(losses.sum())), "computed"
    per_day = trades.groupby("trading_day").size().reindex(days, fill_value=0)
    dist = per_day.value_counts().sort_index()
    denom = float(len(per_day)) if len(per_day) else 0.0
    return {
        "trades": int(len(trades)),
        "net_pnl": float(pnl.sum()),
        "expectancy": float(pnl.mean()),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": pf,
        "profit_factor_status": pf_status,
        "no_trade_day_ratio": float((per_day == 0).mean()) if len(per_day) else None,
        "mean_trades_per_day": float(per_day.mean()) if len(per_day) else None,
        "max_trades_per_day": int(per_day.max()) if len(per_day) else None,
        "zero_trade_day_ratio": float((per_day == 0).mean()) if len(per_day) else None,
        "one_trade_day_ratio": float((per_day == 1).mean()) if len(per_day) else None,
        "two_trade_day_ratio": float((per_day == 2).mean()) if len(per_day) else None,
        "three_trade_day_ratio": float((per_day == 3).mean()) if len(per_day) else None,
        "trades_per_day": {str(int(k)): int(v) for k, v in dist.items()},
        "trades_per_day_ratio": {
            str(int(k)): float(v / denom) for k, v in dist.items()
        } if denom else {},
        "long_trades": int((trades["direction"] == "LONG").sum()),
        "short_trades": int((trades["direction"] == "SHORT").sum()),
        "largest_winning_trade": float(wins.max()) if len(wins) else None,
        "largest_losing_trade": float(losses.min()) if len(losses) else None,
    }


def _random_summary_as_metrics(summary: dict) -> dict:
    return {
        "trades": summary.get("trade_count_mean"),
        "net_pnl": summary.get("net_pnl_mean"),
        "expectancy": None,
        "win_rate": None,
        "profit_factor": None,
        "profit_factor_status": "random_distribution",
        "long_trades": None,
        "short_trades": None,
    }


def _group_metrics(trades: pd.DataFrame, *, all_days: Iterable[int], pnl_col: str, group_col: str) -> dict:
    if group_col not in trades.columns or trades.empty:
        return {}
    out = {}
    for key, part in trades.groupby(group_col, dropna=False):
        out[str(key)] = _strategy_metrics(part, all_days=all_days, pnl_col=pnl_col)
    return out


def _days_by_role(all_days: Iterable[int], role_by_day: dict[int, str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for day in all_days:
        role = role_by_day.get(int(day), "outside_split")
        out.setdefault(role, []).append(int(day))
    return out


def _group_metrics_by_split_role(
    trades: pd.DataFrame,
    *,
    days_by_role: dict[str, list[int]],
    pnl_col: str,
) -> dict:
    roles = set(days_by_role)
    if not trades.empty and "split_role" in trades.columns:
        roles |= {str(x) for x in trades["split_role"].dropna().unique()}
    out = {}
    for role in sorted(roles):
        if trades.empty or "split_role" not in trades.columns:
            part = pd.DataFrame()
        else:
            part = trades[trades["split_role"].astype(str) == str(role)].copy()
        out[str(role)] = _strategy_metrics(
            part,
            all_days=days_by_role.get(str(role), []),
            pnl_col=pnl_col,
        )
    return out


def _prepare_candidates(entry_eval_dir: Path) -> tuple[pd.DataFrame, dict[int, str], list[int]]:
    candidates = _read_table(entry_eval_dir, "candidates")
    candidates = candidates.copy()
    timestamp_cols = [
        "decision_timestamp",
        "long_entry_timestamp",
        "short_entry_timestamp",
        "long_exit_timestamp",
        "short_exit_timestamp",
    ]
    for col in timestamp_cols:
        candidates[col] = _to_timestamp(candidates[col])
    candidates["trading_day"] = candidates["trading_day"].astype(int)
    if "month" not in candidates.columns:
        candidates["month"] = candidates["trading_day"].map(_month_from_day)
    split_path = entry_eval_dir / "split_manifest.json"
    split_manifest = json.loads(split_path.read_text()) if split_path.exists() else []
    all_days = sorted(int(x) for x in candidates["trading_day"].unique())
    role_by_day = _role_from_split(all_days, split_manifest)
    candidates["split_role"] = candidates["trading_day"].map(role_by_day)
    return candidates, role_by_day, all_days


def _entry_eval_manifest(entry_eval_dir: Path) -> dict:
    manifest_path = entry_eval_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text())


def _entry_eval_execution_timing_text(manifest: dict) -> str:
    config = manifest.get("config", {})
    evaluator = config.get("entry_evaluator", {})
    timing = evaluator.get("execution_timing", "canonical_next_open")
    if timing == "signal_on_close_plus_spread":
        return (
            "Dreamer env trade records use signal_on_close_plus_spread fills: "
            "observation includes completed bar t, action is generated after close[t], "
            "LONG fills at close[t] + spread, SHORT fills at close[t] - spread. "
            "candidate exact join maps Dreamer decision_row + direction to the matching "
            "signal-on-close entry evaluator candidate_id and is the primary attribution key. "
            "entry_timestamp and decision_timestamp joins are timing diagnostics."
        )
    if timing == "canonical_next_open":
        return (
            "Dreamer env trade records use the env execution timestamp, while the "
            "fixed-exit evaluator uses decision timestamp t -> next valid 1m open. "
            "candidate exact join maps Dreamer decision_row + direction to the entry "
            "evaluator candidate_id and is the primary attribution key. entry_timestamp "
            "and decision_timestamp joins are timing diagnostics."
        )
    return (
        f"Entry evaluator execution_timing={timing!r}. candidate exact join maps "
        "Dreamer decision_row + direction to the entry evaluator candidate_id and is "
        "the primary attribution key."
    )


def _fixed_exit_join(
    actual_trades: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    mode: str,
) -> pd.DataFrame:
    if actual_trades.empty:
        return actual_trades.copy()
    actual = actual_trades.copy()
    actual["entry_timestamp_ts"] = _to_timestamp(actual["entry_timestamp"])
    if "decision_row" in actual.columns:
        actual["decision_row_join"] = pd.to_numeric(
            actual["decision_row"], errors="coerce").astype("Int64")
    else:
        actual["decision_row_join"] = pd.Series(pd.NA, index=actual.index, dtype="Int64")
    if "decision_timestamp" in actual.columns:
        actual["decision_timestamp_ts"] = _to_timestamp(actual["decision_timestamp"])
        actual["decision_timestamp_ts"] = actual["decision_timestamp_ts"].fillna(actual["entry_timestamp_ts"])
    else:
        actual["decision_timestamp_ts"] = actual["entry_timestamp_ts"]

    pieces = []
    for direction in ("LONG", "SHORT"):
        prefix = direction.lower()
        cand = candidates[candidates[f"action_mask_{prefix}"] > 0.5].copy()
        cand["_candidate_entry_timestamp"] = cand[f"{prefix}_entry_timestamp"]
        if mode == "decision_row":
            cand["decision_row_join"] = pd.to_numeric(
                cand["decision_row"], errors="coerce").astype("Int64")
            left_key = "decision_row_join"
            right_key = "decision_row_join"
            rename_key = False
        elif mode == "entry_timestamp":
            left_key = "entry_timestamp_ts"
            right_key = f"{prefix}_entry_timestamp"
            rename_key = True
        elif mode == "decision_timestamp":
            left_key = "decision_timestamp_ts"
            right_key = "decision_timestamp"
            rename_key = True
        else:
            raise ValueError(mode)
        if rename_key:
            cand = cand.rename(columns={right_key: "_join_key"})
        sub = actual[actual["direction"] == direction].merge(
            cand,
            left_on=left_key,
            right_on=("_join_key" if rename_key else right_key),
            how="left",
            suffixes=("", "_candidate"),
        )
        sub["fixed_join_mode"] = mode
        sub["fixed_net_pnl"] = sub[f"{prefix}_net_pnl"]
        sub["fixed_gross_pnl"] = sub[f"{prefix}_gross_pnl"]
        sub["fixed_spread_cost"] = sub[f"{prefix}_spread_cost"]
        sub["fixed_fee_cost"] = sub[f"{prefix}_fee_cost"]
        sub["fixed_exit_reason"] = sub[f"{prefix}_exit_reason"]
        sub["fixed_exit_timestamp"] = sub[f"{prefix}_exit_timestamp"]
        sub["fixed_entry_row"] = sub[f"{prefix}_entry_row"]
        sub["fixed_entry_timestamp"] = sub["_candidate_entry_timestamp"]
        sub["fixed_exit_row"] = sub[f"{prefix}_exit_row"]
        sub["fixed_holding_bars"] = sub[f"{prefix}_holding_bars"]
        sub["fixed_mfe_gross"] = sub[f"{prefix}_mfe_gross"]
        sub["fixed_mae_gross"] = sub[f"{prefix}_mae_gross"]
        sub["exit_holding_delta_pnl"] = sub["actual_net_pnl"] - sub["fixed_net_pnl"]
        pieces.append(sub)
    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    out["matched_candidate"] = out["candidate_id"].notna()
    return out


def _candidate_actions_from_trades(candidates: pd.DataFrame, trades: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    actions = np.full(len(candidates), "FLAT", dtype=object)
    scores = np.zeros(len(candidates), dtype=float)
    if trades.empty:
        return actions, scores
    id_to_pos = {int(cid): idx for idx, cid in enumerate(candidates["candidate_id"].astype(int))}
    for row in trades.itertuples(index=False):
        cid = getattr(row, "candidate_id", None)
        if pd.isna(cid):
            continue
        pos = id_to_pos.get(int(cid))
        if pos is None:
            continue
        actions[pos] = str(row.direction)
        scores[pos] = 1.0
    return actions, scores


def _attribution_table(primary: pd.DataFrame) -> pd.DataFrame:
    if primary.empty:
        return pd.DataFrame()
    cols = [
        "episode_id",
        "collect_mode",
        "split_role",
        "month",
        "trading_day",
        "direction",
        "candidate_id",
        "decision_row",
        "decision_timestamp",
        "decision_trading_day",
        "requested_action_index",
        "requested_target",
        "planned_action",
        "decision_allowed",
        "decision_reason",
        "entry_timestamp",
        "scheduled_entry_timestamp",
        "fixed_entry_timestamp",
        "entry_execution_price",
        "entry_price",
        "fixed_entry_row",
        "exit_timestamp",
        "fixed_exit_timestamp",
        "exit_reason",
        "fixed_exit_reason",
        "actual_net_pnl",
        "fixed_net_pnl",
        "exit_holding_delta_pnl",
        "fixed_mfe_gross",
        "fixed_mae_gross",
        "matched_candidate",
    ]
    out = primary.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[cols].copy()


def _matched_random(
    candidates: pd.DataFrame,
    target_trades: pd.DataFrame,
    *,
    runs: int,
    seed: int,
    max_entries_per_day: int,
) -> list[pd.DataFrame]:
    from gym_trading_env.research.entry_analysis import matched_random_strategies

    if target_trades.empty:
        return []
    cols = [
        "trading_day",
        "candidate_id",
        "direction",
        "decision_row",
        "exit_row",
        "net_pnl",
    ]
    target = target_trades.rename(columns={"fixed_net_pnl": "net_pnl"}).copy()
    for col in cols:
        if col not in target.columns:
            target[col] = np.nan
    return matched_random_strategies(
        candidates,
        target[cols],
        max_entries_per_day=max_entries_per_day,
        runs=runs,
        seed=seed,
    )


def _attach_roles_to_runs(
    runs: list[pd.DataFrame],
    role_by_day: dict[int, str],
) -> list[pd.DataFrame]:
    out = []
    for df in runs:
        if df.empty:
            out.append(df)
            continue
        item = df.copy()
        item["split_role"] = item["trading_day"].astype(int).map(role_by_day).fillna("outside_split")
        out.append(item)
    return out


def _random_summary_by_split(
    runs: list[pd.DataFrame],
    *,
    all_days: Iterable[int],
    role_by_day: dict[int, str],
    initial_balance: float,
) -> dict:
    from gym_trading_env.research.entry_analysis import summarize_random_runs

    days_by_role: dict[str, list[int]] = {}
    for day in all_days:
        role = role_by_day.get(int(day), "outside_split")
        days_by_role.setdefault(role, []).append(int(day))
    roles = sorted(days_by_role)
    reports = {}
    for role in roles:
        role_runs = []
        for df in runs:
            if df.empty or "split_role" not in df.columns:
                role_runs.append(pd.DataFrame())
            else:
                role_runs.append(df[df["split_role"] == role].copy())
        reports[role] = summarize_random_runs(
            role_runs,
            all_days=days_by_role[role],
            initial_balance=initial_balance,
        )
    return reports


def _episode_risk_report(episodes: pd.DataFrame, *, role_by_day: dict[int, str]) -> dict:
    if episodes.empty:
        return {"margin_termination_count": 0, "negative_return_episodes": 0}
    work = episodes.copy()
    if "start_split_role" in work.columns:
        work["split_role"] = work["start_split_role"].fillna("outside_split")
    else:
        work["split_role"] = work["start_trading_day"].map(role_by_day).fillna("outside_split")
    report = {
        "episodes": int(len(work)),
        "negative_return_episodes": int((pd.to_numeric(work.get("return_pct"), errors="coerce") < 0).sum()),
        "margin_termination_count": 0,
        "return_pct_min": float(pd.to_numeric(work.get("return_pct"), errors="coerce").min()),
        "return_pct_mean": float(pd.to_numeric(work.get("return_pct"), errors="coerce").mean()),
        "return_pct_max": float(pd.to_numeric(work.get("return_pct"), errors="coerce").max()),
        "by_split_role": {},
    }
    if "trades_opened" in work.columns:
        report["episodes_opening_daily_quota"] = int(
            (pd.to_numeric(work["trades_opened"], errors="coerce") >= 3).sum())
    for role, part in work.groupby("split_role", dropna=False):
        ret = pd.to_numeric(part.get("return_pct"), errors="coerce")
        opened = pd.to_numeric(part.get("trades_opened"), errors="coerce")
        report["by_split_role"][str(role)] = {
            "episodes": int(len(part)),
            "negative_return_episodes": int((ret < 0).sum()),
            "return_pct_min": float(ret.min()),
            "return_pct_mean": float(ret.mean()),
            "return_pct_max": float(ret.max()),
            "episodes_opening_daily_quota": int((opened >= 3).sum()) if len(opened) else 0,
            "daily_quota_episode_ratio": float((opened >= 3).mean()) if len(opened) else None,
        }
    return report


def _trade_risk_report(trades: pd.DataFrame, *, pnl_col: str, group_col: str = "split_role") -> dict:
    if trades.empty or pnl_col not in trades.columns:
        return {"trades": 0}
    pnl = pd.to_numeric(trades[pnl_col], errors="coerce")
    report = {
        "trades": int(len(trades)),
        "largest_winning_trade": float(pnl.max()),
        "largest_losing_trade": float(pnl.min()),
        "loss_trade_count": int((pnl < 0).sum()),
        "win_trade_count": int((pnl > 0).sum()),
        "by_split_role": {},
    }
    if "fixed_mae_gross" in trades.columns:
        mae = pd.to_numeric(trades["fixed_mae_gross"], errors="coerce")
        report["fixed_mae_min"] = float(mae.min())
        report["fixed_mae_mean"] = float(mae.mean())
    if "fixed_mfe_gross" in trades.columns:
        mfe = pd.to_numeric(trades["fixed_mfe_gross"], errors="coerce")
        report["fixed_mfe_max"] = float(mfe.max())
        report["fixed_mfe_mean"] = float(mfe.mean())
    if group_col in trades.columns:
        for role, part in trades.groupby(group_col, dropna=False):
            part_pnl = pd.to_numeric(part[pnl_col], errors="coerce")
            item = {
                "trades": int(len(part)),
                "largest_winning_trade": float(part_pnl.max()),
                "largest_losing_trade": float(part_pnl.min()),
                "loss_trade_count": int((part_pnl < 0).sum()),
                "win_trade_count": int((part_pnl > 0).sum()),
            }
            if "fixed_mae_gross" in part.columns:
                item["fixed_mae_min"] = float(pd.to_numeric(part["fixed_mae_gross"], errors="coerce").min())
                item["fixed_mae_mean"] = float(pd.to_numeric(part["fixed_mae_gross"], errors="coerce").mean())
            if "fixed_mfe_gross" in part.columns:
                item["fixed_mfe_max"] = float(pd.to_numeric(part["fixed_mfe_gross"], errors="coerce").max())
                item["fixed_mfe_mean"] = float(pd.to_numeric(part["fixed_mfe_gross"], errors="coerce").mean())
            report["by_split_role"][str(role)] = item
    return report


def collect_dreamer_episodes(
    paths: AuditPaths,
    *,
    episodes: int,
    max_steps: int,
    jax_platform: str | None,
    collect_mode: str,
    role_by_day: dict[int, str],
    roles: set[str],
    start_clock: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if collect_mode == "sampled" and int(episodes) <= 0:
        return pd.DataFrame(), pd.DataFrame()
    sys.path.insert(0, str(paths.dreamer_root))
    os.chdir(paths.dreamer_root)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import elements
    import ruamel.yaml as yaml
    from dreamerv3 import main as dreamer_main

    raw = yaml.YAML(typ="safe").load((paths.run_logdir / "config.yaml").read_text())
    config = elements.Config(raw)
    if paths.env_config_path is not None:
        config = config.update({
            "env.gymnasium.config_path": str(paths.env_config_path),
        })
    eval_logdir = paths.output_dir / "dreamer_eval_runtime"
    eval_logdir.mkdir(parents=True, exist_ok=True)
    config = config.update({
        "logdir": str(eval_logdir),
        "run.debug": True,
        "run.envs": 1,
    })
    if jax_platform:
        config = config.update({"jax.platform": str(jax_platform)})
    os.environ["DREAMER_RUN_DIR"] = str(eval_logdir)

    agent = dreamer_main.make_agent(config)
    import elements.checkpoint as checkpoint

    checkpoint.load(paths.checkpoint, {"agent": bind(agent.load, regex=None)})

    env = dreamer_main.make_env(config, 0)
    base_env = _unwrap_base_env(env)
    forced_start = {"row": None}
    original_candidate_start_rows = base_env._candidate_start_rows_by_clock

    def candidate_start_rows_override(clock):
        if forced_start["row"] is None:
            return original_candidate_start_rows(clock)
        return np.asarray([int(forced_start["row"])], dtype=np.int64)

    if collect_mode == "per_day":
        start_clock = start_clock or str(getattr(base_env.config.training, "start_clock", "future_night"))
        start_rows = _select_start_rows(
            base_env,
            role_by_day=role_by_day,
            roles=roles,
            start_clock=start_clock,
            episode_cap=int(episodes),
        )
        if not start_rows:
            raise RuntimeError(
                f"No per-day start rows for roles={sorted(roles)} start_clock={start_clock}")
        base_env._candidate_start_rows_by_clock = candidate_start_rows_override
        forced_start["row"] = int(start_rows[0])
        target_episodes = len(start_rows)
    else:
        start_rows = []
        target_episodes = int(episodes)

    carry = agent.init_policy(1)
    action = _zero_action(env.act_space)

    episode_rows = []
    trade_rows = []
    completed = 0
    steps = 0
    current_episode_id = -1
    current_meta = {}
    current_score = 0.0
    current_length = 0
    record_meta_by_index: dict[int, dict] = {}
    try:
        while completed < int(target_episodes) and steps < int(max_steps):
            action_meta = _decision_meta_before_step(base_env, action)
            before_records = len(getattr(base_env.trade_record_manager, "trade_history", []))
            obs = env.step(action)
            steps += 1
            after_records = len(getattr(base_env.trade_record_manager, "trade_history", []))
            if action_meta and after_records > before_records:
                for rec_idx in range(before_records, after_records):
                    record_meta_by_index[int(rec_idx)] = dict(action_meta)
            if bool(_scalar(obs.get("is_first"), False)):
                current_episode_id += 1
                current_score = 0.0
                current_length = 0
                record_meta_by_index = {}
                start_day = int(base_env.bar_source.store.row_trading_day[int(getattr(base_env, "start_idx", 0))])
                current_meta = {
                    "episode_id": current_episode_id,
                    "collect_mode": collect_mode,
                    "start_idx": int(getattr(base_env, "start_idx", -1)),
                    "end_idx": int(getattr(base_env, "end_idx", -1)),
                    "start_timestamp": str(base_env.bar_source.store.index[int(getattr(base_env, "start_idx", 0))]),
                    "start_trading_day": start_day,
                    "start_split_role": role_by_day.get(start_day, "outside_split"),
                }
            else:
                current_length += 1
                current_score += float(_scalar(obs.get("reward"), 0.0) or 0.0)

            policy_obs = _batch_obs(obs, agent.obs_space)
            carry, next_action, _ = agent.policy(carry, policy_obs, mode="eval")
            action = {k: np.asarray(v[0]) for k, v in next_action.items()}

            if bool(_scalar(obs.get("is_last"), False)):
                records = [
                    rec.to_dict()
                    for rec in getattr(base_env.trade_record_manager, "trade_history", [])
                ]
                episode_meta = {
                    **current_meta,
                    "episode_id": current_episode_id,
                    "length": int(current_length),
                    "score": float(current_score),
                    "end_timestamp": str(base_env.bar_source.store.index[int(getattr(base_env, "current_step", 0))]),
                    "end_trading_day": int(base_env.bar_source.store.row_trading_day[int(getattr(base_env, "current_step", 0))]),
                    "return_pct": float(_scalar(obs.get("log/env/return_pct"), np.nan)),
                    "trades_opened": float(_scalar(obs.get("log/env/trades_opened"), np.nan)),
                    "trades_closed": float(_scalar(obs.get("log/env/trades_closed"), np.nan)),
                    "profit_factor": _scalar(obs.get("log/env/profit_factor"), None),
                }
                episode_rows.append(episode_meta)
                trade_rows.extend(_records_to_trades(
                    records,
                    episode_id=current_episode_id,
                    episode_meta=episode_meta,
                    record_meta_by_index=record_meta_by_index,
                ))
                completed += 1
                if collect_mode == "per_day" and completed < target_episodes:
                    forced_start["row"] = int(start_rows[completed])
                action = _zero_action(env.act_space)
            else:
                action["reset"] = np.asarray(False)
    finally:
        env.close()

    return pd.DataFrame.from_records(episode_rows), pd.DataFrame.from_records(trade_rows)


def summarize(
    paths: AuditPaths,
    episodes: pd.DataFrame,
    actual_trades: pd.DataFrame,
    *,
    random_runs: int,
    matched_random_seed: int,
) -> dict:
    candidates, role_by_day, all_days = _prepare_candidates(paths.entry_eval_dir)
    days_by_role = _days_by_role(all_days, role_by_day)
    if not actual_trades.empty:
        actual_trades = actual_trades.copy()
        actual_trades["trading_day"] = actual_trades["start_trading_day"].astype(int)
        actual_trades["month"] = actual_trades["trading_day"].map(_month_from_day)
        actual_trades["split_role"] = actual_trades["trading_day"].map(role_by_day).fillna("outside_split")

    fixed_candidate = _fixed_exit_join(actual_trades, candidates, mode="decision_row")
    fixed_entry = _fixed_exit_join(actual_trades, candidates, mode="entry_timestamp")
    fixed_decision = _fixed_exit_join(actual_trades, candidates, mode="decision_timestamp")
    fixed_candidate_matched = (
        fixed_candidate[fixed_candidate.get("matched_candidate", False) == True].copy()
        if not fixed_candidate.empty
        else fixed_candidate
    )
    fixed_matched = fixed_entry[fixed_entry.get("matched_candidate", False) == True].copy() if not fixed_entry.empty else fixed_entry
    fixed_decision_matched = (
        fixed_decision[fixed_decision.get("matched_candidate", False) == True].copy()
        if not fixed_decision.empty
        else fixed_decision
    )

    from gym_trading_env.research.entry_analysis import (
        constrained_oracle,
        simulate_strategy,
        strategy_metrics,
        summarize_random_runs,
    )

    product = json.loads((paths.entry_eval_dir / "manifest.json").read_text())["config"]["product"]
    entry_eval_manifest = _entry_eval_manifest(paths.entry_eval_dir)
    product = entry_eval_manifest.get("config", {}).get("product", product)
    entry_evaluator_config = entry_eval_manifest.get("config", {}).get("entry_evaluator", {})
    max_entries = int(product["max_entries_per_day"])
    initial_balance = float(product["initial_balance"])

    if not fixed_candidate_matched.empty:
        actions, scores = _candidate_actions_from_trades(candidates, fixed_candidate_matched)
        dreamer_fixed_strategy = simulate_strategy(
            candidates, actions, scores, max_entries_per_day=max_entries)
    else:
        dreamer_fixed_strategy = pd.DataFrame()
    if not dreamer_fixed_strategy.empty:
        dreamer_fixed_strategy = dreamer_fixed_strategy.copy()
        dreamer_fixed_strategy["trading_day"] = dreamer_fixed_strategy["trading_day"].astype(int)
        dreamer_fixed_strategy["month"] = dreamer_fixed_strategy["trading_day"].map(_month_from_day)
        dreamer_fixed_strategy["split_role"] = (
            dreamer_fixed_strategy["trading_day"].map(role_by_day).fillna("outside_split")
        )

    oracle = constrained_oracle(candidates, max_entries_per_day=max_entries)
    random_strategies = _matched_random(
        candidates,
        dreamer_fixed_strategy,
        runs=random_runs,
        seed=matched_random_seed,
        max_entries_per_day=max_entries,
    )
    random_strategies = _attach_roles_to_runs(random_strategies, role_by_day)
    random_summary = summarize_random_runs(
        random_strategies,
        all_days=all_days,
        initial_balance=initial_balance,
    )
    random_summary["seed"] = int(matched_random_seed)
    random_by_split = _random_summary_by_split(
        random_strategies,
        all_days=all_days,
        role_by_day=role_by_day,
        initial_balance=initial_balance,
    )
    for item in random_by_split.values():
        item["seed"] = int(matched_random_seed)
    collect_modes = (
        sorted(str(x) for x in episodes.get("collect_mode", pd.Series(dtype=object)).dropna().unique())
        if not episodes.empty
        else []
    )
    split_eval = (
        "Per-day deterministic replay: one forced reset per selected trading day, "
        "grouped by entry-eval final-fold roles."
        if collect_modes == ["per_day"]
        else (
            "Episodes are sampled from the env start distribution and grouped by "
            "entry-eval final-fold roles; this is not exhaustive per-day replay."
        )
    )

    split_manifest_path = paths.entry_eval_dir / "split_manifest.json"
    reports = {
        "checkpoint": str(paths.checkpoint),
        "run_logdir": str(paths.run_logdir),
        "env_config_path": str(paths.env_config_path) if paths.env_config_path is not None else None,
        "seed_protocol": _seed_metadata(
            paths, matched_random_seed=int(matched_random_seed)),
        "entry_eval": {
            "dir": str(paths.entry_eval_dir),
            "version": entry_eval_manifest.get("version"),
            "execution_timing": entry_evaluator_config.get("execution_timing", "canonical_next_open"),
            "entry_evaluator": entry_evaluator_config,
            "candidate_filter_audit": entry_eval_manifest.get("candidate_filter_audit", {}),
            "split_manifest_hash": _file_sha256(split_manifest_path),
        },
        "limitations": {
            "available_checkpoints": "latest_only",
            "checkpoint_step": _checkpoint_step(paths.checkpoint),
            "collect_modes": collect_modes,
            "split_eval": split_eval,
            "execution_timing": _entry_eval_execution_timing_text(entry_eval_manifest),
            "random_comparison": (
                "Matched random is compared against the executable non-overlapping "
                "fixed-exit strategy, not against independent per-trade counterfactuals."),
        },
        "episodes": {
            "collected": int(len(episodes)),
            "by_start_role": episodes.assign(
                split_role=episodes["start_trading_day"].map(role_by_day).fillna("outside_split")
            ).groupby("split_role").size().to_dict() if not episodes.empty else {},
        },
        "actual_dreamer": {
            "metrics": _strategy_metrics(actual_trades, all_days=all_days, pnl_col="actual_net_pnl"),
            "by_split_role": _group_metrics_by_split_role(
                actual_trades,
                days_by_role=days_by_role,
                pnl_col="actual_net_pnl",
            ),
            "by_month": _group_metrics(actual_trades, all_days=all_days, pnl_col="actual_net_pnl", group_col="month"),
            "by_direction": _group_metrics(actual_trades, all_days=all_days, pnl_col="actual_net_pnl", group_col="direction"),
        },
        "dreamer_entry_fixed_exit_candidate_join": {
            "matched_trades": int(fixed_candidate["matched_candidate"].sum()) if not fixed_candidate.empty else 0,
            "unmatched_trades": int((~fixed_candidate["matched_candidate"]).sum()) if not fixed_candidate.empty else 0,
            "metrics": _strategy_metrics(fixed_candidate_matched, all_days=all_days, pnl_col="fixed_net_pnl"),
            "actual_matched": _strategy_metrics(fixed_candidate_matched, all_days=all_days, pnl_col="actual_net_pnl"),
            "exit_holding_delta": _strategy_metrics(
                fixed_candidate_matched,
                all_days=all_days,
                pnl_col="exit_holding_delta_pnl",
            ),
            "by_split_role": _group_metrics_by_split_role(
                fixed_candidate_matched,
                days_by_role=days_by_role,
                pnl_col="fixed_net_pnl",
            ),
            "actual_matched_by_split_role": _group_metrics_by_split_role(
                fixed_candidate_matched,
                days_by_role=days_by_role,
                pnl_col="actual_net_pnl",
            ),
            "delta_by_split_role": _group_metrics_by_split_role(
                fixed_candidate_matched,
                days_by_role=days_by_role,
                pnl_col="exit_holding_delta_pnl",
            ),
            "by_month": _group_metrics(fixed_candidate_matched, all_days=all_days, pnl_col="fixed_net_pnl", group_col="month"),
            "by_direction": _group_metrics(fixed_candidate_matched, all_days=all_days, pnl_col="fixed_net_pnl", group_col="direction"),
        },
        "dreamer_entry_fixed_exit_entry_timestamp_join": {
            "matched_trades": int(fixed_entry["matched_candidate"].sum()) if not fixed_entry.empty else 0,
            "unmatched_trades": int((~fixed_entry["matched_candidate"]).sum()) if not fixed_entry.empty else 0,
            "metrics": _strategy_metrics(fixed_matched, all_days=all_days, pnl_col="fixed_net_pnl"),
            "actual_matched": _strategy_metrics(fixed_matched, all_days=all_days, pnl_col="actual_net_pnl"),
            "exit_holding_delta": _strategy_metrics(
                fixed_matched,
                all_days=all_days,
                pnl_col="exit_holding_delta_pnl",
            ),
            "by_split_role": _group_metrics_by_split_role(
                fixed_matched,
                days_by_role=days_by_role,
                pnl_col="fixed_net_pnl",
            ),
            "actual_matched_by_split_role": _group_metrics_by_split_role(
                fixed_matched,
                days_by_role=days_by_role,
                pnl_col="actual_net_pnl",
            ),
            "delta_by_split_role": _group_metrics_by_split_role(
                fixed_matched,
                days_by_role=days_by_role,
                pnl_col="exit_holding_delta_pnl",
            ),
            "by_month": _group_metrics(fixed_matched, all_days=all_days, pnl_col="fixed_net_pnl", group_col="month"),
            "by_direction": _group_metrics(fixed_matched, all_days=all_days, pnl_col="fixed_net_pnl", group_col="direction"),
        },
        "dreamer_entry_fixed_exit_decision_timestamp_join": {
            "matched_trades": int(fixed_decision["matched_candidate"].sum()) if not fixed_decision.empty else 0,
            "unmatched_trades": int((~fixed_decision["matched_candidate"]).sum()) if not fixed_decision.empty else 0,
            "metrics": _strategy_metrics(
                fixed_decision_matched,
                all_days=all_days,
                pnl_col="fixed_net_pnl",
            ),
            "actual_matched": _strategy_metrics(
                fixed_decision_matched,
                all_days=all_days,
                pnl_col="actual_net_pnl",
            ),
            "exit_holding_delta": _strategy_metrics(
                fixed_decision_matched,
                all_days=all_days,
                pnl_col="exit_holding_delta_pnl",
            ),
            "by_split_role": _group_metrics_by_split_role(
                fixed_decision_matched,
                days_by_role=days_by_role,
                pnl_col="fixed_net_pnl",
            ),
            "actual_matched_by_split_role": _group_metrics_by_split_role(
                fixed_decision_matched,
                days_by_role=days_by_role,
                pnl_col="actual_net_pnl",
            ),
            "delta_by_split_role": _group_metrics_by_split_role(
                fixed_decision_matched,
                days_by_role=days_by_role,
                pnl_col="exit_holding_delta_pnl",
            ),
        },
        "dreamer_entry_fixed_exit_executable_nonoverlap": {
            "metrics": strategy_metrics(
                dreamer_fixed_strategy,
                all_days=all_days,
                initial_balance=initial_balance,
            ),
            "by_split_role": _group_metrics_by_split_role(
                dreamer_fixed_strategy,
                days_by_role=days_by_role,
                pnl_col="net_pnl",
            ),
            "by_month": _group_metrics(
                dreamer_fixed_strategy,
                all_days=all_days,
                pnl_col="net_pnl",
                group_col="month",
            ),
            "by_direction": _group_metrics(
                dreamer_fixed_strategy,
                all_days=all_days,
                pnl_col="net_pnl",
                group_col="direction",
            ),
            "source": "candidate_join_matched_trades",
        },
        "matched_random_fixed_exit_executable_nonoverlap": random_summary,
        "matched_random_fixed_exit_by_split_role": random_by_split,
        "matched_random_fixed_exit": random_summary,
        "risk_path": {
            "episodes": _episode_risk_report(episodes, role_by_day=role_by_day),
            "actual_trades": _trade_risk_report(actual_trades, pnl_col="actual_net_pnl"),
            "candidate_attribution": _trade_risk_report(
                fixed_candidate_matched,
                pnl_col="fixed_net_pnl",
            ),
            "candidate_delta": _trade_risk_report(
                fixed_candidate_matched,
                pnl_col="exit_holding_delta_pnl",
            ),
        },
        "constrained_oracle_fixed_exit": strategy_metrics(
            oracle,
            all_days=all_days,
            initial_balance=initial_balance,
        ),
    }

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    attribution = _attribution_table(fixed_candidate)
    episodes.to_csv(paths.output_dir / "dreamer_episodes.csv", index=False)
    actual_trades.to_csv(paths.output_dir / "dreamer_actual_trades.csv", index=False)
    attribution.to_csv(paths.output_dir / "dreamer_candidate_attribution.csv", index=False)
    fixed_candidate.to_csv(paths.output_dir / "dreamer_fixed_exit_candidate_join.csv", index=False)
    fixed_entry.to_csv(paths.output_dir / "dreamer_fixed_exit_entry_join.csv", index=False)
    fixed_decision.to_csv(paths.output_dir / "dreamer_fixed_exit_decision_join.csv", index=False)
    dreamer_fixed_strategy.to_csv(paths.output_dir / "dreamer_fixed_exit_strategy.csv", index=False)
    oracle.to_csv(paths.output_dir / "oracle_fixed_exit_trades.csv", index=False)
    _write_json(paths.output_dir / "summary.json", reports)
    _write_markdown(paths.output_dir / "report.md", reports)
    return reports


def _checkpoint_step(checkpoint: Path) -> int | None:
    step_path = checkpoint / "step.pkl"
    if not step_path.exists():
        return None
    import pickle
    try:
        return int(pickle.loads(step_path.read_bytes()))
    except Exception:
        return None


def _write_markdown(path: Path, summary: dict) -> None:
    actual = summary["actual_dreamer"]["metrics"]
    fixed_candidate = summary["dreamer_entry_fixed_exit_candidate_join"]["metrics"]
    fixed_entry = summary["dreamer_entry_fixed_exit_entry_timestamp_join"]["metrics"]
    fixed_decision = summary["dreamer_entry_fixed_exit_decision_timestamp_join"]["metrics"]
    fixed_exec = summary["dreamer_entry_fixed_exit_executable_nonoverlap"]["metrics"]
    random = summary["matched_random_fixed_exit_executable_nonoverlap"]
    oracle = summary["constrained_oracle_fixed_exit"]
    candidate_join = summary["dreamer_entry_fixed_exit_candidate_join"]
    entry_join = summary["dreamer_entry_fixed_exit_entry_timestamp_join"]
    decision_join = summary["dreamer_entry_fixed_exit_decision_timestamp_join"]
    split_rows = _split_rows(summary)
    delta_rows = _delta_rows(summary)
    trade_day_rows = _trade_day_rows(summary)
    risk_rows = _risk_rows(summary)
    random_split_rows = _random_split_rows(summary)
    lines = [
        "# Dreamer Checkpoint Generalization Audit",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Checkpoint step: `{summary['limitations']['checkpoint_step']}`",
        f"- Source run: `{summary['run_logdir']}`",
        f"- Entry-eval dir: `{summary.get('entry_eval', {}).get('dir')}`",
        f"- Entry-eval version: `{summary.get('entry_eval', {}).get('version')}`",
        f"- Entry-eval execution timing: `{summary.get('entry_eval', {}).get('execution_timing')}`",
        f"- Split manifest hash: `{summary.get('entry_eval', {}).get('split_manifest_hash')}`",
        "",
        "## Seed Protocol",
        "",
        f"- Experiment seed: `{summary.get('seed_protocol', {}).get('experiment_seed')}`",
        f"- Dreamer/JAX seed: `{summary.get('seed_protocol', {}).get('dreamer_seed')}`",
        f"- Train env seed: `{summary.get('seed_protocol', {}).get('train_env_seed')}`",
        f"- Replay seed: `{summary.get('seed_protocol', {}).get('replay_seed')}`",
        f"- Eval env seed: `{summary.get('seed_protocol', {}).get('eval_env_seed')}`",
        f"- Matched-random seed: `{summary.get('seed_protocol', {}).get('matched_random_seed')}`",
        "",
        "## Limitations",
        "",
        f"- Available checkpoints: {summary['limitations']['available_checkpoints']}",
        f"- Split eval: {summary['limitations']['split_eval']}",
        f"- Execution timing: {summary['limitations']['execution_timing']}",
        f"- Random comparison: {summary['limitations']['random_comparison']}",
        f"- Candidate filter audit: `{summary.get('entry_eval', {}).get('candidate_filter_audit', {})}`",
        "",
        "## Episode Coverage",
        "",
        f"- Episodes collected: {summary['episodes']['collected']}",
        f"- By final-fold role: `{summary['episodes']['by_start_role']}`",
        "",
        "## Comparison",
        "",
        "| Strategy | Trades | Net PnL | Expectancy | Profit Factor | Long | Short |",
        "|---|---:|---:|---:|---|---:|---:|",
        _metric_row("Dreamer actual entry + Dreamer actual exit", actual),
        _metric_row("Dreamer entries + fixed exit, candidate-id exact join", fixed_candidate),
        _metric_row("Dreamer entries + fixed exit, individual entry-time join", fixed_entry),
        _metric_row("Dreamer entries + fixed exit, individual decision-time join", fixed_decision),
        _metric_row("Dreamer entries + fixed exit, executable non-overlap from candidate join", fixed_exec),
        (
            f"| Matched random + fixed exit, executable non-overlap | "
            f"{random.get('trade_count_mean', 'NA')} | "
            f"{random.get('net_pnl_mean', 'NA')} | NA | NA | NA | NA |"
        ),
        _metric_row("Constrained oracle + fixed evaluator exit", oracle),
        "",
        "## Join Diagnostics",
        "",
        (
            f"- Candidate exact join matched {candidate_join['matched_trades']} trades and "
            f"missed {candidate_join['unmatched_trades']} trades."
        ),
        (
            f"- Entry-time join matched {entry_join['matched_trades']} trades and "
            f"missed {entry_join['unmatched_trades']} trades."
        ),
        (
            f"- Decision-time join matched {decision_join['matched_trades']} trades and "
            f"missed {decision_join['unmatched_trades']} trades."
        ),
        "- Candidate exact join is the primary attribution key and executable comparison source for this report.",
        "- Entry-time join is a strict same-execution-time timing diagnostic.",
        "- Entry-time and decision-time joins are timing diagnostics.",
        "",
        "## Split Diagnostics",
        "",
        "| Split | Actual Trades | Actual Net | Candidate Fixed Trades | Candidate Fixed Net | Fixed Executable Trades | Fixed Executable Net |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *split_rows,
        "",
        "## Attribution Diagnostics",
        "",
        "| Split | Actual All Net | Candidate Actual Matched Net | Candidate Fixed Net | Candidate Unmatched Actual Net | Delta On Candidate Matched | Entry-Time Fixed Net | Decision-Time Fixed Net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *delta_rows,
        "",
        "## Trades Per Day",
        "",
        "| Split | Mean | Max | 0/day | 1/day | 2/day | 3/day | Distribution |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        *trade_day_rows,
        "",
        "## Risk Path",
        "",
        "| Scope | Split | Trades/Episodes | Largest Win | Largest Loss | Fixed MAE Min | Fixed MAE Mean | Negative Episodes | Daily Quota Ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *risk_rows,
        "",
        "## Matched Random By Split",
        "",
        "| Split | Runs | Trade Count Mean | Net PnL Mean | Net PnL P05 | Net PnL P50 | Net PnL P95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *random_split_rows,
        "",
        "## Preliminary Conclusion",
        "",
        _preliminary_conclusion(summary),
        "",
        "## Attribution Rule",
        "",
        "- If Dreamer actual is positive but fixed-exit is negative, improvement is exit/holding/path driven.",
        "- If both actual and fixed-exit are positive and beat matched random across validation/test roles, entry selection is plausible.",
        "- If gains concentrate in train role only, treat the 70w+ lift as in-sample until a heldout run proves otherwise.",
    ]
    path.write_text("\n".join(lines) + "\n")


def _split_rows(summary: dict) -> list[str]:
    actual = summary["actual_dreamer"].get("by_split_role", {})
    fixed = summary["dreamer_entry_fixed_exit_candidate_join"].get("by_split_role", {})
    executable = summary["dreamer_entry_fixed_exit_executable_nonoverlap"].get("by_split_role", {})
    roles = sorted(set(actual) | set(fixed) | set(executable))
    rows = []
    for role in roles:
        a = actual.get(role, {})
        f = fixed.get(role, {})
        e = executable.get(role, {})
        rows.append(
            f"| {role} | {a.get('trades', 0)} | {a.get('net_pnl', 0)} | "
            f"{f.get('trades', 0)} | {f.get('net_pnl', 0)} | "
            f"{e.get('trades', 0)} | {e.get('net_pnl', 0)} |"
        )
    return rows


def _delta_rows(summary: dict) -> list[str]:
    actual = summary["actual_dreamer"].get("by_split_role", {})
    candidate_actual = summary["dreamer_entry_fixed_exit_candidate_join"].get("actual_matched_by_split_role", {})
    candidate_fixed = summary["dreamer_entry_fixed_exit_candidate_join"].get("by_split_role", {})
    candidate_delta = summary["dreamer_entry_fixed_exit_candidate_join"].get("delta_by_split_role", {})
    fixed_entry = summary["dreamer_entry_fixed_exit_entry_timestamp_join"].get("by_split_role", {})
    fixed_decision = summary["dreamer_entry_fixed_exit_decision_timestamp_join"].get("by_split_role", {})
    roles = sorted(set(actual) | set(candidate_actual) | set(candidate_fixed) | set(candidate_delta) | set(fixed_entry) | set(fixed_decision))
    rows = []
    for role in roles:
        a = actual.get(role, {})
        ca = candidate_actual.get(role, {})
        cf = candidate_fixed.get(role, {})
        cd = candidate_delta.get(role, {})
        f = fixed_entry.get(role, {})
        fd = fixed_decision.get(role, {})
        actual_all = float(a.get("net_pnl", 0) or 0)
        actual_matched = float(ca.get("net_pnl", 0) or 0)
        rows.append(
            f"| {role} | {a.get('net_pnl', 0)} | {ca.get('net_pnl', 0)} | "
            f"{cf.get('net_pnl', 0)} | {actual_all - actual_matched} | "
            f"{cd.get('net_pnl', 0)} | {f.get('net_pnl', 0)} | {fd.get('net_pnl', 0)} |"
        )
    return rows


def _trade_day_rows(summary: dict) -> list[str]:
    actual = summary["actual_dreamer"].get("by_split_role", {})
    rows = []
    for role, metrics in sorted(actual.items()):
        rows.append(
            f"| {role} | {metrics.get('mean_trades_per_day')} | {metrics.get('max_trades_per_day')} | "
            f"{metrics.get('zero_trade_day_ratio')} | {metrics.get('one_trade_day_ratio')} | "
            f"{metrics.get('two_trade_day_ratio')} | {metrics.get('three_trade_day_ratio')} | "
            f"`{metrics.get('trades_per_day', {})}` |"
        )
    return rows


def _risk_rows(summary: dict) -> list[str]:
    risk = summary.get("risk_path", {})
    rows = []
    episode_roles = risk.get("episodes", {}).get("by_split_role", {})
    for role, item in sorted(episode_roles.items()):
        rows.append(
            f"| episodes | {role} | {item.get('episodes')} | NA | NA | NA | NA | "
            f"{item.get('negative_return_episodes')} | {item.get('daily_quota_episode_ratio')} |"
        )
    for scope in ("actual_trades", "candidate_attribution", "candidate_delta"):
        by_role = risk.get(scope, {}).get("by_split_role", {})
        for role, item in sorted(by_role.items()):
            rows.append(
                f"| {scope} | {role} | {item.get('trades')} | "
                f"{item.get('largest_winning_trade')} | {item.get('largest_losing_trade')} | "
                f"{item.get('fixed_mae_min', 'NA')} | {item.get('fixed_mae_mean', 'NA')} | NA | NA |"
            )
    return rows


def _random_split_rows(summary: dict) -> list[str]:
    random_by_split = summary.get("matched_random_fixed_exit_by_split_role", {})
    rows = []
    for role, item in sorted(random_by_split.items()):
        rows.append(
            f"| {role} | {item.get('runs', 0)} | {item.get('trade_count_mean', 'NA')} | "
            f"{item.get('net_pnl_mean', 'NA')} | {item.get('net_pnl_p05', 'NA')} | "
            f"{item.get('net_pnl_p50', 'NA')} | {item.get('net_pnl_p95', 'NA')} |"
        )
    return rows


def _preliminary_conclusion(summary: dict) -> str:
    actual = summary["actual_dreamer"]["metrics"]
    fixed_exec = summary["dreamer_entry_fixed_exit_executable_nonoverlap"]["metrics"]
    fixed_exec_test = (
        summary["dreamer_entry_fixed_exit_executable_nonoverlap"]
        .get("by_split_role", {})
        .get("test", {})
    )
    random = summary["matched_random_fixed_exit_executable_nonoverlap"]
    actual_net = float(actual.get("net_pnl") or 0.0)
    fixed_net = float(fixed_exec.get("net_pnl") or 0.0)
    if not fixed_exec_test:
        return (
            "Status: INCONCLUSIVE_NO_TEST. This report has no executable fixed-exit "
            "test-split trades, so it can validate mechanics but cannot classify "
            "Dreamer entry attribution."
        )
    test_fixed_net = float(fixed_exec_test.get("net_pnl") or 0.0)
    random_mean = float(random.get("net_pnl_mean") or 0.0)
    if actual_net > 0 and fixed_net > random_mean and test_fixed_net <= 0:
        return (
            "Status: PROMISING_BUT_NOT_PASS. Actual Dreamer exits are strongly positive "
            "and executable fixed-exit entries beat matched-random overall, but the "
            "test split fixed-exit executable PnL is not positive. This supports further "
            "checkpoint attribution and env timing audit, not Entry-only Dreamer admission."
        )
    if actual_net > 0 and fixed_net > random_mean and test_fixed_net > 0:
        return (
            "Status: PROMISING. Actual Dreamer exits are positive and executable fixed-exit "
            "entries beat matched-random overall and on the test split. This still needs "
            "larger deterministic per-day coverage before a formal PASS."
        )
    if actual_net > 0 and fixed_net <= 0:
        return (
            "Status: EXIT_OR_PATH_DRIVEN. Actual Dreamer exits are positive but fixed-exit "
            "entry attribution is not, so the observed lift is more likely from exit, "
            "holding, or path mechanics than fixed-exit entry selection."
        )
    return (
        "Status: INCONCLUSIVE_OR_FAIL. The sampled checkpoint does not provide enough "
        "positive attribution evidence under this audit."
    )


def _metric_row(name: str, metrics: dict) -> str:
    return (
        f"| {name} | {metrics.get('trades')} | {metrics.get('net_pnl')} | "
        f"{metrics.get('expectancy')} | {metrics.get('profit_factor')} | "
        f"{metrics.get('long_trades')} | {metrics.get('short_trades')} |"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dreamer-root", default="/home/v/Documents/work/dreamerv3")
    parser.add_argument("--run-logdir", default="/data/logdir/action-mask-formal-202606131200")
    parser.add_argument("--checkpoint", default="/data/logdir/action-mask-formal-202606131200/ckpt/latest")
    parser.add_argument("--entry-eval-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--env-config-path",
        default=None,
        help="Optional CustomTradingEnv YAML for evaluation only; checkpoint weights and env code are unchanged.",
    )
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--random-runs", type=int, default=100)
    parser.add_argument("--matched-random-seed", type=int, default=20260615)
    parser.add_argument("--jax-platform", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--collect", action="store_true", help="Run Dreamer eval to collect fresh episodes.")
    parser.add_argument(
        "--collect-mode",
        choices=["sampled", "per_day"],
        default="sampled",
        help="sampled uses the env start distribution; per_day forces one deterministic episode per selected trading day.",
    )
    parser.add_argument(
        "--roles",
        default="train,validation,test",
        help="Comma-separated split roles for --collect-mode per_day. Use empty string for all roles.",
    )
    parser.add_argument(
        "--start-clock",
        default=None,
        help="Start clock for --collect-mode per_day; defaults to the checkpoint env config start_clock.",
    )
    args = parser.parse_args(argv)

    checkpoint = _read_latest_checkpoint(Path(args.checkpoint))
    paths = AuditPaths(
        dreamer_root=Path(args.dreamer_root).expanduser().resolve(),
        run_logdir=Path(args.run_logdir).expanduser().resolve(),
        checkpoint=checkpoint,
        entry_eval_dir=Path(args.entry_eval_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        env_config_path=(
            Path(args.env_config_path).expanduser().resolve()
            if args.env_config_path else None
        ),
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    _, role_by_day, _ = _prepare_candidates(paths.entry_eval_dir)
    roles = {x.strip() for x in str(args.roles).split(",") if x.strip()}

    episode_path = paths.output_dir / "dreamer_episodes.csv"
    trade_path = paths.output_dir / "dreamer_actual_trades.csv"
    if args.collect or not (episode_path.exists() and trade_path.exists()):
        episodes, trades = collect_dreamer_episodes(
            paths,
            episodes=args.episodes,
            max_steps=args.max_steps,
            jax_platform=args.jax_platform,
            collect_mode=args.collect_mode,
            role_by_day=role_by_day,
            roles=roles,
            start_clock=args.start_clock,
        )
        episodes.to_csv(episode_path, index=False)
        trades.to_csv(trade_path, index=False)
    else:
        episodes = pd.read_csv(episode_path)
        trades = pd.read_csv(trade_path)
    summary = summarize(
        paths,
        episodes,
        trades,
        random_runs=args.random_runs,
        matched_random_seed=args.matched_random_seed,
    )
    print(json.dumps({
        "output_dir": str(paths.output_dir),
        "episodes": summary["episodes"],
        "actual": summary["actual_dreamer"]["metrics"],
        "fixed_exit": summary["dreamer_entry_fixed_exit_candidate_join"]["metrics"],
        "fixed_exit_join": summary["dreamer_entry_fixed_exit_candidate_join"],
        "matched_random": summary["matched_random_fixed_exit"],
        "fixed_exit_executable": summary["dreamer_entry_fixed_exit_executable_nonoverlap"]["metrics"],
        "oracle": summary["constrained_oracle_fixed_exit"],
    }, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
