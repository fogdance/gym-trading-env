from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from gym_trading_env.research.entry_analysis import matched_random_strategies, simulate_strategy


def _reference_simulate_strategy(
    candidates: pd.DataFrame,
    actions,
    scores,
    *,
    max_entries_per_day: int,
) -> pd.DataFrame:
    work = candidates.copy()
    work["action"] = np.asarray(list(actions), dtype=object)
    work["score"] = np.asarray(list(scores), dtype=float)
    work = work.sort_values(["decision_row", "candidate_id"]).reset_index(drop=True)

    trades = []
    active_exit_row = -1
    day_entries: dict[int, int] = {}
    for row in work.itertuples(index=False):
        action = str(row.action)
        if action == "FLAT":
            continue
        if int(row.decision_row) <= active_exit_row:
            continue
        day = int(row.trading_day)
        if day_entries.get(day, 0) >= int(max_entries_per_day):
            continue

        prefix = action.lower()
        exit_row = int(getattr(row, f"{prefix}_exit_row"))
        trades.append({
            "candidate_id": int(row.candidate_id),
            "contract": getattr(row, "contract", None),
            "month": getattr(row, "month", None),
            "session_phase": getattr(row, "session_phase", None),
            "trading_day": day,
            "decision_row": int(row.decision_row),
            "decision_timestamp": row.decision_timestamp,
            "direction": action,
            "score": float(row.score),
            "entry_row": int(getattr(row, f"{prefix}_entry_row")),
            "entry_timestamp": getattr(row, f"{prefix}_entry_timestamp"),
            "exit_row": exit_row,
            "exit_timestamp": getattr(row, f"{prefix}_exit_timestamp"),
            "exit_reason": getattr(row, f"{prefix}_exit_reason"),
            "holding_bars": int(getattr(row, f"{prefix}_holding_bars")),
            "gross_pnl": float(getattr(row, f"{prefix}_gross_pnl")),
            "spread_cost": float(getattr(row, f"{prefix}_spread_cost")),
            "fee_cost": float(getattr(row, f"{prefix}_fee_cost")),
            "net_pnl": float(getattr(row, f"{prefix}_net_pnl")),
            "mfe_gross": float(getattr(row, f"{prefix}_mfe_gross")),
            "mae_gross": float(getattr(row, f"{prefix}_mae_gross")),
        })
        active_exit_row = exit_row
        day_entries[day] = day_entries.get(day, 0) + 1
    return pd.DataFrame.from_records(trades)


def _candidates() -> pd.DataFrame:
    rows = []
    for idx, row in enumerate([12, 10, 14, 30, 31, 36]):
        day = 20250101 if row < 20 else 20250102
        rows.append({
            "candidate_id": idx,
            "contract": "RB",
            "month": day // 100,
            "session_phase": "night" if idx % 2 == 0 else "day",
            "trading_day": day,
            "decision_row": row,
            "decision_timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=row),
            "long_entry_row": row,
            "long_entry_timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=row),
            "long_exit_row": row + 3,
            "long_exit_timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=row + 3),
            "long_exit_reason": "take_profit",
            "long_holding_bars": 3,
            "long_gross_pnl": float(idx + 1),
            "long_spread_cost": 0.5,
            "long_fee_cost": 1.0,
            "long_net_pnl": float(idx),
            "long_mfe_gross": float(idx + 2),
            "long_mae_gross": float(-idx),
            "short_entry_row": row,
            "short_entry_timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=row),
            "short_exit_row": row + 2,
            "short_exit_timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=row + 2),
            "short_exit_reason": "stop_loss",
            "short_holding_bars": 2,
            "short_gross_pnl": float(-idx - 1),
            "short_spread_cost": 0.5,
            "short_fee_cost": 1.0,
            "short_net_pnl": float(-idx - 2),
            "short_mfe_gross": float(idx),
            "short_mae_gross": float(-idx - 3),
        })
    return pd.DataFrame.from_records(rows)


def test_simulate_strategy_matches_reference_implementation():
    candidates = _candidates()
    actions = np.asarray(["LONG", "SHORT", "FLAT", "LONG", "SHORT", "LONG"], dtype=object)
    scores = np.asarray([0.5, 0.7, 0.0, 1.1, 0.3, 1.8], dtype=float)

    expected = _reference_simulate_strategy(
        candidates, actions, scores, max_entries_per_day=2)
    actual = simulate_strategy(candidates, actions, scores, max_entries_per_day=2)

    assert_frame_equal(actual, expected, check_dtype=False)


def test_matched_random_strategies_remains_deterministic():
    candidates = _candidates()
    target = simulate_strategy(
        candidates,
        ["LONG", "FLAT", "FLAT", "LONG", "FLAT", "FLAT"],
        [1, 0, 0, 1, 0, 0],
        max_entries_per_day=1,
    )

    first = matched_random_strategies(
        candidates, target, max_entries_per_day=1, runs=3, seed=42)
    second = matched_random_strategies(
        candidates, target, max_entries_per_day=1, runs=3, seed=42)

    assert len(first) == len(second) == 3
    for lhs, rhs in zip(first, second):
        assert_frame_equal(lhs, rhs, check_dtype=False)
