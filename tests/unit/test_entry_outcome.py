import pandas as pd

from gym_trading_env.research.entry_dataset import outcomes_from_candidates
from gym_trading_env.research.entry_outcome import EntryOutcome


def test_entry_outcome_schema_exposes_required_fields():
    names = set(EntryOutcome.__dataclass_fields__)
    assert {
        "decision_timestamp",
        "entry_timestamp",
        "direction",
        "gross_pnl",
        "spread_cost",
        "fee_cost",
        "net_pnl",
        "stop_target_collision",
    }.issubset(names)


def test_outcomes_from_candidates_creates_long_and_short_rows():
    candidates = pd.DataFrame([{
        "dataset_version": "v1",
        "exit_policy_version": "fixed",
        "product": "DCE.JM",
        "contract": "JMTEST",
        "candidate_id": 1,
        "decision_row": 10,
        "decision_timestamp": "t10",
        "trading_day": 20250106,
        "long_entry_row": 11,
        "long_entry_timestamp": "t11",
        "long_exit_row": 12,
        "long_exit_timestamp": "t12",
        "long_exit_reason": "STOP",
        "long_holding_bars": 1,
        "long_gross_pnl": -240.0,
        "long_spread_cost": 60.0,
        "long_fee_cost": 6.0,
        "long_net_pnl": -306.0,
        "long_mfe_gross": 0.0,
        "long_mae_gross": -240.0,
        "short_entry_row": 11,
        "short_entry_timestamp": "t11",
        "short_exit_row": 12,
        "short_exit_timestamp": "t12",
        "short_exit_reason": "TARGET",
        "short_holding_bars": 1,
        "short_gross_pnl": 504.0,
        "short_spread_cost": 60.0,
        "short_fee_cost": 6.0,
        "short_net_pnl": 438.0,
        "short_mfe_gross": 504.0,
        "short_mae_gross": 0.0,
    }])

    outcomes = outcomes_from_candidates(candidates)

    assert outcomes["direction"].tolist() == ["LONG", "SHORT"]
    assert outcomes["net_pnl"].tolist() == [-306.0, 438.0]
