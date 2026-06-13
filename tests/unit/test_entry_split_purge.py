import numpy as np
import pandas as pd
import pytest

from gym_trading_env.research.entry_analysis import (
    constrained_oracle,
    daily_bootstrap_expectancy,
    matched_random_strategies,
    purged_fold_masks,
)


pytestmark = pytest.mark.unit


def test_purge_and_embargo_remove_boundary_overlap():
    candidates = pd.DataFrame({
        "decision_row": [0, 10, 20, 30, 40, 50],
        "long_exit_row": [5, 25, 25, 45, 45, 55],
        "short_exit_row": [5, 25, 25, 45, 45, 55],
        "trading_day": [1, 1, 2, 2, 3, 3],
    })
    fold = {
        "name": "fold_1",
        "train_days": [1],
        "validation_days": [2],
        "test_days": [3],
    }
    masks, audit = purged_fold_masks(candidates, fold, embargo_bars=3)

    np.testing.assert_array_equal(np.flatnonzero(masks["train"]), [0])
    np.testing.assert_array_equal(np.flatnonzero(masks["validation"]), [2])
    np.testing.assert_array_equal(np.flatnonzero(masks["test"]), [4, 5])
    assert audit["boundaries"][0]["crossing_labels_removed"] == 1
    assert audit["boundaries"][1]["crossing_labels_removed"] == 1


def test_daily_bootstrap_is_deterministic_and_samples_days():
    trades = pd.DataFrame({
        "trading_day": [1, 1, 2, 3],
        "net_pnl": [10.0, -2.0, 4.0, -1.0],
    })
    first = daily_bootstrap_expectancy(trades, all_days=[1, 2, 3, 4], runs=200, seed=7)
    second = daily_bootstrap_expectancy(trades, all_days=[1, 2, 3, 4], runs=200, seed=7)
    assert first == second
    assert first["lower_95"] <= first["mean"] <= first["upper_95"]


def test_random_baseline_matches_target_trade_count():
    rows = []
    for day in (1, 2):
        for decision in range((day - 1) * 100, (day - 1) * 100 + 20):
            rows.append({
                "candidate_id": len(rows),
                "trading_day": day,
                "decision_row": decision,
                "decision_timestamp": str(decision),
                "long_entry_row": decision + 1,
                "long_entry_timestamp": str(decision + 1),
                "long_exit_row": decision + 2,
                "long_exit_timestamp": str(decision + 2),
                "long_exit_reason": "MAX_HOLD",
                "long_holding_bars": 2,
                "long_gross_pnl": 1.0,
                "long_spread_cost": 0.1,
                "long_fee_cost": 0.1,
                "long_net_pnl": 0.8,
                "long_mfe_gross": 1.0,
                "long_mae_gross": -1.0,
                "short_entry_row": decision + 1,
                "short_entry_timestamp": str(decision + 1),
                "short_exit_row": decision + 2,
                "short_exit_timestamp": str(decision + 2),
                "short_exit_reason": "MAX_HOLD",
                "short_holding_bars": 2,
                "short_gross_pnl": 1.0,
                "short_spread_cost": 0.1,
                "short_fee_cost": 0.1,
                "short_net_pnl": 0.8,
                "short_mfe_gross": 1.0,
                "short_mae_gross": -1.0,
            })
    candidates = pd.DataFrame.from_records(rows)
    target = pd.DataFrame({"trading_day": [1, 1, 1, 2, 2, 2]})
    random = matched_random_strategies(
        candidates, target, max_entries_per_day=3, runs=5, seed=11)
    assert [len(x) for x in random] == [6] * 5


def test_constrained_oracle_optimizes_non_overlapping_trade_combination():
    rows = []
    for candidate_id, decision, exit_row, pnl in (
        (0, 0, 10, 5.0),
        (1, 1, 2, 8.0),
        (2, 3, 4, 8.0),
    ):
        row = {
            "candidate_id": candidate_id,
            "trading_day": 1,
            "decision_row": decision,
            "decision_timestamp": str(decision),
        }
        for direction in ("long", "short"):
            value = pnl if direction == "long" else -1.0
            row.update({
                f"{direction}_entry_row": decision + 1,
                f"{direction}_entry_timestamp": str(decision + 1),
                f"{direction}_exit_row": exit_row,
                f"{direction}_exit_timestamp": str(exit_row),
                f"{direction}_exit_reason": "MAX_HOLD",
                f"{direction}_holding_bars": exit_row - decision,
                f"{direction}_gross_pnl": value,
                f"{direction}_spread_cost": 0.0,
                f"{direction}_fee_cost": 0.0,
                f"{direction}_net_pnl": value,
                f"{direction}_mfe_gross": max(value, 0.0),
                f"{direction}_mae_gross": min(value, 0.0),
            })
        rows.append(row)

    trades = constrained_oracle(pd.DataFrame.from_records(rows), max_entries_per_day=3)
    assert trades["candidate_id"].tolist() == [1, 2]
    assert trades["net_pnl"].sum() == 16.0
