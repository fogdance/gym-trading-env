import pandas as pd

from tools.dreamer_checkpoint_audit import (
    PROFIT_LOCK_DEBUG_FIELDS,
    _ensure_step_reward_schema,
    build_profit_lock_trade_rollup,
)


def _rows(has_profit_lock=True):
    base = {
        "checkpoint_label": "ckpt",
        "episode_id": 0,
        "trading_day": 20260622,
        "timestamp": "2026-06-22T09:01:00+08:00",
        "step_index": 1,
        "position_side": "LONG",
        "reward_total": 0.1,
        "raw_total": 0.2,
        "r_old_raw_total": 0.1,
        "source_reward_has_profit_lock": has_profit_lock,
    }
    row1 = dict(base)
    row1.update({
        "profit_lock_active": 1.0,
        "profit_lock_mfe_cash": 3600.0,
        "profit_lock_mfe_R": 1.2,
        "profit_lock_giveback_cash": 1200.0,
        "profit_lock_excess_giveback_cash": 100.0,
        "r_profit_giveback": -0.01,
        "r_profit_lock_total": -0.01,
    })
    row2 = dict(base)
    row2.update({
        "timestamp": "2026-06-22T09:02:00+08:00",
        "step_index": 2,
        "position_side": "FLAT",
        "close_reason": "MANUAL",
        "profit_lock_active": 1.0,
        "profit_lock_mfe_cash": 3600.0,
        "profit_lock_mfe_R": 1.2,
        "profit_lock_capture_ratio": -0.1,
        "r_lost_mfe": -0.1,
        "r_profit_lock_total": -0.1,
    })
    rows = [row1, row2]
    if not has_profit_lock:
        for row in rows:
            for key in PROFIT_LOCK_DEBUG_FIELDS:
                row.pop(key, None)
            row["source_reward_has_profit_lock"] = False
    return _ensure_step_reward_schema(pd.DataFrame.from_records(rows))


def _trades():
    return pd.DataFrame.from_records([{
        "checkpoint_label": "ckpt",
        "episode_id": 0,
        "trading_day": 20260622,
        "direction": "LONG",
        "entry_timestamp": "2026-06-22T09:01:00+08:00",
        "exit_timestamp": "2026-06-22T09:02:00+08:00",
        "entry_price": 100.0,
        "exit_price": 90.0,
        "actual_gross_pnl": -594.0,
        "actual_net_pnl": -600.0,
        "exit_reason": "MANUAL",
    }])


def test_profit_lock_trade_rollup_flags_big_mfe_loss():
    rollup = build_profit_lock_trade_rollup(_rows(True), _trades())

    row = rollup.iloc[0]
    assert row["capture_ratio"] == -594.0 / 3600.0
    assert bool(row["big_mfe_ge_1R"]) is True
    assert bool(row["big_mfe_to_loss"]) is True
    assert bool(row["profit_lock_was_active"]) is True
    assert bool(row["giveback_penalty_fired"]) is True
    assert bool(row["lost_mfe_fired"]) is True
    assert row["sum_r_profit_lock_total"] < 0
    assert row["overlay_abs_share"] > 0


def test_old_reward_absent_profit_lock_fields_fill_zero():
    rollup = build_profit_lock_trade_rollup(_rows(False), _trades())

    row = rollup.iloc[0]
    assert row["mfe_R"] == 0.0
    assert bool(row["profit_lock_was_active"]) is False
    assert row["sum_r_profit_lock_total"] == 0.0
    assert row["sum_abs_old_raw_total"] > 0.0
    assert bool(row["reward_sign_changed_by_overlay"]) is False


def test_session_end_close_is_eod_and_forced_reason():
    trades = _trades().copy()
    trades.loc[0, "exit_reason"] = "TRUNCATE_SESSION_END"

    rollup = build_profit_lock_trade_rollup(_rows(False), trades)

    row = rollup.iloc[0]
    assert row["eod_flag"] == "EOD"
    assert row["forced_close_reason"] == "TRUNCATE_SESSION_END"
