import pytest
import pandas as pd

from tools.jm3_old_baseline_reaudit_summary import enrich_would_have


def test_would_have_capture_uses_gross_but_loss_uses_net():
    trades = pd.DataFrame.from_records([{
        "episode_id": 0,
        "trading_day": 20250801,
        "direction": "LONG",
        "entry_timestamp": "2025-07-31T21:01:00+08:00",
        "exit_timestamp": "2025-07-31T21:03:00+08:00",
        "actual_gross_pnl": 180.0,
        "actual_net_pnl": -6.0,
    }])
    steps = pd.DataFrame.from_records([
        {
            "episode_id": 0,
            "timestamp": "2025-07-31T21:01:00+08:00",
            "unrealized_pnl": 0.0,
            "scale_cash": 300.0,
        },
        {
            "episode_id": 0,
            "timestamp": "2025-07-31T21:02:00+08:00",
            "unrealized_pnl": 600.0,
            "scale_cash": 300.0,
        },
        {
            "episode_id": 0,
            "timestamp": "2025-07-31T21:03:00+08:00",
            "unrealized_pnl": 120.0,
            "scale_cash": 300.0,
        },
    ])

    row = enrich_would_have("ckpt", trades, steps).iloc[0]

    assert row["would_have_mfe_cash"] == 600.0
    assert row["would_have_mfe_R"] == 2.0
    assert row["would_have_capture_ratio"] == pytest.approx(180.0 / 600.0)
    assert row["would_have_giveback_after_1R"] == pytest.approx(420.0)
    assert bool(row["would_have_big_mfe_to_loss"]) is True
    assert bool(row["would_have_low_capture"]) is True
