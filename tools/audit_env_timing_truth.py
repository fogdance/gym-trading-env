#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for item in (ROOT, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))


def _unwrap(env):
    cur = env
    seen = set()
    while hasattr(cur, "env") and id(cur) not in seen:
        seen.add(id(cur))
        nxt = cur.env
        if nxt is cur:
            break
        cur = nxt
    return cur


def _timestamp(value) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _num(value) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _find_row_by_timestamp(store, timestamp) -> int | None:
    if timestamp is None:
        return None
    target = pd.Timestamp(timestamp)
    if target.tzinfo is None:
        target = target.tz_localize("Asia/Shanghai")
    else:
        target = target.tz_convert("Asia/Shanghai")
    index = pd.DatetimeIndex(store.index)
    if index.tz is None:
        index = index.tz_localize("Asia/Shanghai")
    else:
        index = index.tz_convert("Asia/Shanghai")
    matches = np.flatnonzero(index == target)
    return int(matches[0]) if len(matches) else None


def _position_state(base_env) -> str:
    ua = base_env.user_accounts
    try:
        if ua.long_position > 0:
            return "LONG"
        if ua.short_position > 0:
            return "SHORT"
    except Exception:
        pass
    return "FLAT"


def _next_valid_row(store, row: int) -> int | None:
    mask = np.asarray(store.row_mask)
    for idx in range(int(row) + 1, len(mask)):
        if float(mask[idx]) >= 0.5:
            return int(idx)
    return None


def _prev_valid_row(store, row: int) -> int | None:
    mask = np.asarray(store.row_mask)
    for idx in range(int(row) - 1, -1, -1):
        if float(mask[idx]) >= 0.5:
            return int(idx)
    return None


def _row_ohlcv(store, row: int | None) -> dict[str, Any]:
    if row is None:
        return {
            "row": None,
            "timestamp": None,
            "open": None,
            "high": None,
            "low": None,
            "close": None,
            "volume": None,
            "mask": None,
        }
    ts = store.index[int(row)]
    raw = _raw_bar_by_timestamp(store, ts)
    out = {
        "row": int(row),
        "timestamp": _timestamp(ts),
        "open": _num(raw.get("Open")) if raw is not None else None,
        "high": _num(store.row_H[int(row)]),
        "low": _num(store.row_L[int(row)]),
        "close": _num(store.row_C[int(row)]),
        "volume": _num(raw.get("Volume")) if raw is not None else None,
        "mask": _num(store.row_mask[int(row)]),
    }
    return out


def _raw_bar_by_timestamp(store, timestamp) -> dict[str, Any] | None:
    try:
        raw = store.df_raw
        idx = pd.DatetimeIndex(raw.index)
        target = pd.Timestamp(timestamp)
        if target.tzinfo is None:
            target = target.tz_localize("Asia/Shanghai")
        else:
            target = target.tz_convert("Asia/Shanghai")
        if idx.tz is None:
            idx = idx.tz_localize("Asia/Shanghai")
        else:
            idx = idx.tz_convert("Asia/Shanghai")
        matches = np.flatnonzero(idx == target)
        if len(matches) == 0:
            return None
        row = raw.iloc[int(matches[0])]
        return row.to_dict()
    except Exception:
        return None


def _nearest_price_match(base_env, direction: str | None, execution_row: int | None, execution_price: float | None) -> str:
    if execution_row is None or execution_price is None:
        return "none"
    store = base_env.bar_source.store
    spread = float(base_env.config.trading.spread)
    current_close = float(store.row_C[execution_row])
    current_raw = _raw_bar_by_timestamp(store, store.index[execution_row])
    current_open = _num(current_raw.get("Open")) if current_raw is not None else None
    next_row = _next_valid_row(store, execution_row)
    next_open = None
    if next_row is not None:
        next_raw = _raw_bar_by_timestamp(store, store.index[next_row])
        next_open = _num(next_raw.get("Open")) if next_raw is not None else None
    candidates = []
    if current_open is not None:
        candidates.append(("open", current_open))
    candidates.append(("current_close", current_close))
    if next_open is not None:
        candidates.append(("next_open", next_open))
    if direction == "LONG":
        if current_open is not None:
            candidates.append(("open_plus_spread", current_open + spread))
        candidates.append(("current_close_plus_spread", current_close + spread))
        if next_open is not None:
            candidates.append(("next_open_plus_spread", next_open + spread))
    elif direction == "SHORT":
        if current_open is not None:
            candidates.append(("open_minus_spread", current_open - spread))
        candidates.append(("current_close_minus_spread", current_close - spread))
        if next_open is not None:
            candidates.append(("next_open_minus_spread", next_open - spread))
    best = min(candidates, key=lambda item: abs(float(execution_price) - item[1]))
    if abs(float(execution_price) - best[1]) <= 1e-9:
        return best[0]
    return "other"


def _quote_details_before_step(base_env, action_index: int) -> dict[str, Any]:
    table = getattr(base_env, "_pending_transition_table", None)
    if table is None:
        return {}
    try:
        decision = table.for_index(int(action_index))
    except Exception:
        return {}
    quote = decision.quote
    out = {
        "quote_action_price": _num(getattr(table, "action_price", None)),
        "quote_market_open": bool(getattr(table, "market_open", False)),
        "decision_allowed": bool(getattr(decision, "allowed", False)),
        "decision_reason": str(getattr(decision, "reason", "")),
        "planned_action_name": getattr(decision.planned_action, "name", str(decision.planned_action)),
    }
    if quote is not None:
        out.update({
            "quote_timestamp": _timestamp(getattr(quote, "timestamp", None)),
            "quote_exec_price": _num(getattr(quote, "exec_price", None)),
            "quote_rpc_price": _num(getattr(quote, "rpc_price", None)),
        })
    return out


def _code_price_source(base_env, direction: str | None, execution_row: int | None, quote: dict[str, Any]) -> str:
    if execution_row is None:
        return "none"
    action_price = quote.get("quote_action_price")
    exec_price = quote.get("quote_exec_price")
    if action_price is None or exec_price is None:
        return "none"
    store = base_env.bar_source.store
    row_c = float(store.row_C[int(execution_row)])
    spread = float(base_env.config.trading.spread)
    if abs(float(action_price) - row_c) > 1e-9:
        return "other"
    if direction == "LONG" and abs(float(exec_price) - (row_c + spread)) <= 1e-9:
        return "current_close_plus_spread"
    if direction == "SHORT" and abs(float(exec_price) - (row_c - spread)) <= 1e-9:
        return "current_close_minus_spread"
    if abs(float(exec_price) - row_c) <= 1e-9:
        return "current_close"
    return "current_close_derived_other"


def _classify(
    *,
    obs_end: int,
    obs_start: int,
    execution_row: int | None,
    decision_row: int,
    next_valid: int | None,
    execution_price_source: str,
) -> tuple[str, str]:
    if execution_row is None:
        return "UNKNOWN", "No execution record was produced for this action."
    includes_execution = obs_start <= execution_row <= obs_end
    if (
        obs_end < execution_row
        and execution_price_source.startswith("open")
        and execution_row == obs_end + 1
    ):
        return "Mode_A", "Observation ends before execution row and fill is the first unseen bar open."
    if (
        obs_end == decision_row
        and next_valid is not None
        and execution_row == next_valid
        and execution_price_source.startswith("next_open")
    ):
        return "Mode_B", "Observation includes decision bar and fill is next valid bar open."
    if includes_execution and execution_row == decision_row:
        return (
            "Mode_C",
            "market_seq includes the execution row and the action is filled on that same row.",
        )
    if obs_end >= execution_row:
        return "Mode_C", "Observation window reaches or passes the execution row."
    return "UNKNOWN", "Observed rows and execution row do not match Mode A/B/C exactly."


def _candidate_id(entry_eval_dir: Path | None, decision_row: int, direction: str | None) -> int | None:
    if entry_eval_dir is None or direction not in ("LONG", "SHORT"):
        return None
    path = entry_eval_dir / "candidates.parquet"
    csv = entry_eval_dir / "candidates.csv"
    try:
        if path.exists():
            candidates = pd.read_parquet(path)
        elif csv.exists():
            candidates = pd.read_csv(csv)
        else:
            return None
    except Exception:
        return None
    mask_col = f"action_mask_{direction.lower()}"
    if mask_col not in candidates.columns:
        return None
    rows = candidates[
        (pd.to_numeric(candidates["decision_row"], errors="coerce") == int(decision_row))
        & (pd.to_numeric(candidates[mask_col], errors="coerce") > 0.5)
    ]
    if rows.empty:
        return None
    return int(rows.iloc[0]["candidate_id"])


def _reset_to_row(env, base_env, row: int):
    original = base_env._candidate_start_rows_by_clock

    def forced(_clock):
        return np.asarray([int(row)], dtype=np.int64)

    old_randomize = bool(getattr(base_env.config.training, "randomize_start", True))
    base_env._candidate_start_rows_by_clock = forced
    base_env.config.training.randomize_start = False
    try:
        obs, info = env.reset()
    finally:
        base_env._candidate_start_rows_by_clock = original
        base_env.config.training.randomize_start = old_randomize
    return obs, info


def _first_new_execution(records_before: int, base_env) -> dict | None:
    records = base_env.trade_record_manager.trade_history
    if len(records) <= records_before:
        return None
    for rec in records[records_before:]:
        data = rec.to_dict()
        op = str(data.get("operation_type", ""))
        if "OPEN" in op or "CLOSE" in op:
            return data
    return records[records_before].to_dict()


def _planned_order_type(base_env, action_index: int) -> str:
    table = getattr(base_env, "_pending_transition_table", None)
    if table is None:
        return "no_pending_table"
    try:
        decision = table.for_index(int(action_index))
    except Exception:
        return "invalid_action_index"
    return getattr(decision.planned_action, "name", str(decision.planned_action))


def audit_current_state(
    env,
    base_env,
    *,
    episode_id: int,
    label: str,
    action_index: int,
    entry_eval_dir: Path | None,
) -> dict[str, Any]:
    obs = base_env._get_obs()
    store = base_env.bar_source.store
    row_before = int(base_env.current_step)
    obs_end = row_before
    obs_start = row_before - int(base_env.window_size) + 1
    market_seq_last_row = obs_end
    market_seq_last = np.asarray(obs["market_seq"][-1], dtype=np.float32)
    mode = getattr(base_env.config.trading, "obs_feature_mode", "raw")
    use_obs = mode == "obs"
    X_all = store.X_market_obs if use_obs else store.X_market_raw
    market_seq_matches_raw_last = bool(np.allclose(
        market_seq_last,
        np.asarray(X_all[market_seq_last_row], dtype=np.float32),
        atol=1e-6,
        rtol=0,
    ))
    quote = _quote_details_before_step(base_env, action_index)
    planned_action = quote.get("planned_action_name") or _planned_order_type(base_env, action_index)
    position_before = _position_state(base_env)
    records_before = len(base_env.trade_record_manager.trade_history)
    next_valid = _next_valid_row(store, row_before)
    prev_valid = _prev_valid_row(store, row_before)

    obs_after, reward, terminated, truncated, info = env.step(int(action_index))
    position_after = _position_state(base_env)
    rec = _first_new_execution(records_before, base_env)
    direction = None
    execution_timestamp = None
    execution_price = None
    order_type = planned_action
    if rec:
        op = str(rec.get("operation_type", ""))
        order_type = op
        if "LONG" in op:
            direction = "LONG"
        elif "SHORT" in op:
            direction = "SHORT"
        execution_timestamp = rec.get("timestamp")
        execution_price = rec.get("open_price") if "OPEN" in op else rec.get("close_price")
        execution_price = _num(execution_price)
    execution_row = _find_row_by_timestamp(store, execution_timestamp)
    source = _code_price_source(base_env, direction, execution_row, quote)
    nearest_match = _nearest_price_match(base_env, direction, execution_row, execution_price)
    classification, reason = _classify(
        obs_end=obs_end,
        obs_start=obs_start,
        execution_row=execution_row,
        decision_row=row_before,
        next_valid=next_valid,
        execution_price_source=source,
    )
    includes_exec = (
        execution_row is not None
        and int(obs_start) <= int(execution_row) <= int(obs_end)
    )
    cur = _row_ohlcv(store, row_before)
    prev_row = _row_ohlcv(store, prev_valid)
    next_row = _row_ohlcv(store, next_valid)
    exec_row_data = _row_ohlcv(store, execution_row)
    return {
        "case": label,
        "episode_id": int(episode_id),
        "env_step": int(getattr(base_env, "episode_step_count", 0)),
        "env_current_row_before_step": row_before,
        "env_current_timestamp_before_step": _timestamp(store.index[row_before]),
        "observation_window_start_row": int(obs_start),
        "observation_window_start_timestamp": _timestamp(store.index[max(0, obs_start)]),
        "observation_window_end_row": int(obs_end),
        "observation_window_end_timestamp": _timestamp(store.index[obs_end]),
        "market_seq_last_row": int(market_seq_last_row),
        "market_seq_last_timestamp": _timestamp(store.index[market_seq_last_row]),
        "market_seq_matches_raw_last_row": market_seq_matches_raw_last,
        "does_market_seq_include_execution_bar": bool(includes_exec),
        "action_submitted": int(action_index),
        "position_before": position_before,
        "position_after": position_after,
        "order_type": order_type,
        "execution_row": execution_row,
        "execution_timestamp": _timestamp(execution_timestamp),
        "execution_price": execution_price,
        "execution_price_source": source,
        "execution_price_nearest_numeric_match": nearest_match,
        "quote_action_price": quote.get("quote_action_price"),
        "quote_exec_price": quote.get("quote_exec_price"),
        "quote_timestamp": quote.get("quote_timestamp"),
        "quote_market_open": quote.get("quote_market_open"),
        "decision_allowed": quote.get("decision_allowed"),
        "decision_reason": quote.get("decision_reason"),
        "next_valid_bar_row": next_valid,
        "next_valid_bar_timestamp": _timestamp(store.index[next_valid]) if next_valid is not None else None,
        "decision_row_if_available": row_before,
        "candidate_id_if_available": _candidate_id(entry_eval_dir, row_before, direction),
        "classification": classification,
        "reason": reason,
        "reward": _num(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "prev_row": json.dumps(prev_row, ensure_ascii=False),
        "current_row": json.dumps(cur, ensure_ascii=False),
        "next_valid_row": json.dumps(next_row, ensure_ascii=False),
        "execution_row_data": json.dumps(exec_row_data, ensure_ascii=False),
    }


def audit_one(
    env,
    base_env,
    *,
    episode_id: int,
    label: str,
    row: int,
    action_index: int,
    entry_eval_dir: Path | None,
) -> dict[str, Any]:
    _reset_to_row(env, base_env, row)
    return audit_current_state(
        env,
        base_env,
        episode_id=episode_id,
        label=label,
        action_index=action_index,
        entry_eval_dir=entry_eval_dir,
    )


def _select_rows(base_env) -> list[tuple[str, int]]:
    store = base_env.bar_source.store
    mask = np.asarray(store.row_mask)
    days = sorted(set(int(x) for x in store.row_trading_day[mask >= 0.5]))
    if not days:
        raise RuntimeError("No valid trading days in store")
    day = days[min(3, len(days) - 1)]
    day_rows = np.flatnonzero((store.row_trading_day == day) & (mask >= 0.5))
    if len(day_rows) < 20:
        raise RuntimeError(f"Too few valid rows for day {day}")
    session_open = int(day_rows[0])
    session_after_open = int(day_rows[min(5, len(day_rows) - 1)])
    near_eod = int(day_rows[-3]) if len(day_rows) >= 3 else int(day_rows[-1])

    valid = np.flatnonzero(mask >= 0.5)
    def can_afford_long(row: int) -> bool:
        try:
            price = float(store.row_C[int(row)]) + float(base_env.config.trading.spread)
            size = float(base_env.config.trading.trade_lot)
            lot_size = float(base_env.config.trading.lot_size)
            leverage = float(base_env.config.trading.leverage)
            fee = float(base_env.config.trading.trading_fee_per_lot) * size
            required = size * lot_size * price / leverage
            return required + fee <= float(base_env.config.trading.initial_balance)
        except Exception:
            return True

    gaps = []
    for prev, cur in zip(valid[:-1], valid[1:]):
        same_day = int(store.row_trading_day[int(prev)]) == int(store.row_trading_day[int(cur)])
        if not same_day:
            continue
        prev_ts = pd.Timestamp(store.index[int(prev)])
        cur_ts = pd.Timestamp(store.index[int(cur)])
        if cur_ts - prev_ts > pd.Timedelta(minutes=1) and can_afford_long(prev) and can_afford_long(cur):
            gaps.append((int(prev), int(cur)))
    before_break, after_break = gaps[0] if gaps else (int(day_rows[-2]), int(day_rows[-1]))
    return [
        ("normal_continuous", session_after_open),
        ("session_open", session_open),
        ("before_break", before_break),
        ("after_break", after_break),
        ("near_eod", near_eod),
    ]


def run_audit(args) -> pd.DataFrame:
    if args.runtime_cwd:
        os.chdir(args.runtime_cwd)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import gymnasium as gym
    import gym_trading_env  # noqa: F401  register env

    env = gym.make("CustomTradingEnv-v0", config_path=str(args.config_path))
    base_env = _unwrap(env)
    rows = _select_rows(base_env)
    entry_eval_dir = Path(args.entry_eval_dir).resolve() if args.entry_eval_dir else None
    records = []
    for idx, (label, row) in enumerate(rows):
        records.append(audit_one(
            env,
            base_env,
            episode_id=idx,
            label=label,
            row=row,
            action_index=2,
            entry_eval_dir=entry_eval_dir,
        ))
    if not args.skip_flip:
        flip_row = rows[0][1]
        _reset_to_row(env, base_env, flip_row)
        env.step(2)
        records.append(audit_current_state(
            env,
            base_env,
            episode_id=len(records),
            label="flip_long_to_short_continuation",
            action_index=0,
            entry_eval_dir=entry_eval_dir,
        ))
    env.close()
    return pd.DataFrame.from_records(records)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        default="/home/v/Documents/work/dreamerv3/data/trading_stage1.yaml",
    )
    parser.add_argument(
        "--runtime-cwd",
        default=None,
        help=(
            "Working directory used while loading env data. Defaults to the "
            "parent of the config directory, matching the Dreamer run layout."
        ),
    )
    parser.add_argument(
        "--entry-eval-dir",
        default="artifacts/entry_eval/entry_eval_jm_dreamer6m_2024_v1",
    )
    parser.add_argument(
        "--output",
        default="artifacts/env_timing_truth/env_timing_truth.csv",
    )
    parser.add_argument("--skip-flip", action="store_true")
    args = parser.parse_args(argv)
    args.config_path = Path(args.config_path).expanduser().resolve()
    if args.runtime_cwd is None:
        args.runtime_cwd = args.config_path.parent.parent
    else:
        args.runtime_cwd = Path(args.runtime_cwd).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df = run_audit(args)
    df.to_csv(out, index=False)
    print(json.dumps({
        "output": str(out),
        "rows": int(len(df)),
        "classification_counts": df["classification"].value_counts().to_dict(),
        "cases": df[[
            "case",
            "env_current_row_before_step",
            "market_seq_last_row",
            "execution_row",
            "execution_price_source",
            "classification",
        ]].to_dict(orient="records"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
