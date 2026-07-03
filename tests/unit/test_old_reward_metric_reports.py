import pandas as pd

from tools.old_reward_metric_reports import (
    build_action_count_summary,
    build_checkpoint_reward_component_summary,
    build_reports,
    build_episode_score_pnl_mismatch,
    build_trade_reward_component_rollup,
)


def _steps():
    return pd.DataFrame.from_records([
        {
            "checkpoint_label": "ckpt",
            "episode_id": 0,
            "trading_day": 20260622,
            "timestamp": "2026-06-22T09:01:00+08:00",
            "step_index": 1,
            "selected_action": 2,
            "executed_action": 2,
            "position_side": "LONG",
            "reward_total": 0.30,
            "raw_total": 0.31,
            "pnl": -0.05,
            "close": 0.20,
            "r_atr_close": 0.10,
            "eod": 0.0,
            "sl": -0.01,
            "mkt_closed": -0.02,
            "invalid_total": 0.0,
        },
        {
            "checkpoint_label": "ckpt",
            "episode_id": 0,
            "trading_day": 20260622,
            "timestamp": "2026-06-22T09:02:00+08:00",
            "step_index": 2,
            "selected_action": 1,
            "executed_action": 1,
            "position_side": "FLAT",
            "reward_total": 0.40,
            "raw_total": 0.42,
            "pnl": -0.02,
            "close": 0.30,
            "r_atr_close": 0.15,
            "eod": 0.0,
            "sl": 0.0,
            "mkt_closed": 0.0,
            "invalid_total": 0.0,
        },
    ])


def _trades():
    return pd.DataFrame.from_records([{
        "checkpoint_label": "ckpt",
        "trade_id": 0,
        "episode_id": 0,
        "trading_day": 20260622,
        "direction": "LONG",
        "entry_time": "2026-06-22T09:01:00+08:00",
        "exit_time": "2026-06-22T09:02:00+08:00",
        "actual_pnl": -60.0,
        "fixed_exit_pnl": 120.0,
        "exit_holding_delta": -180.0,
        "mfe_cash": 180.0,
        "mfe_R": 1.2,
        "mae_cash": -90.0,
        "capture_ratio": -0.3,
        "giveback_cash": 240.0,
    }])


def _episodes():
    return pd.DataFrame.from_records([{
        "checkpoint_label": "ckpt",
        "episode_id": 0,
        "start_trading_day": 20260622,
        "score": 0.70,
        "reward_episode_actual_net_pnl": -60.0,
        "reward_episode_final_equity": 9940.0,
        "reward_episode_trade_count": 1,
    }])


def test_trade_reward_component_rollup_sums_old_reward_fields():
    rollup = build_trade_reward_component_rollup(_steps(), _trades(), checkpoint_label="ckpt")

    row = rollup.iloc[0]
    assert row["reward_total_sum"] == 0.70
    assert row["reward_close_sum"] == 0.50
    assert row["reward_atr_close_sum"] == 0.25
    assert row["reward_sl_sum"] == -0.01
    assert row["reward_mkt_closed_sum"] == -0.02
    assert bool(row["trade_reward_positive_actual_pnl_negative"]) is True


def test_trade_rollup_reconstructs_mfe_from_unrealized_when_old_reward_has_no_mfe_fields():
    steps = _steps()
    steps["unrealized_pnl"] = [120.0, -30.0]
    steps["scale_cash"] = 60.0
    trades = pd.DataFrame.from_records([{
        "checkpoint_label": "ckpt",
        "episode_id": 0,
        "trading_day": 20260622,
        "direction": "LONG",
        "entry_timestamp": "2026-06-22T09:01:00+08:00",
        "exit_timestamp": "2026-06-22T09:02:00+08:00",
        "actual_gross_pnl": 60.0,
        "actual_net_pnl": 54.0,
    }])

    rollup = build_trade_reward_component_rollup(steps, trades, checkpoint_label="ckpt")

    row = rollup.iloc[0]
    assert row["mfe_cash"] == 120.0
    assert row["mfe_R"] == 2.0
    assert row["mae_cash"] == -30.0
    assert row["realized_pnl"] == 60.0
    assert row["capture_ratio"] == 0.5
    assert row["giveback_cash"] == 60.0


def test_episode_score_pnl_mismatch_and_checkpoint_summary():
    mismatch = build_episode_score_pnl_mismatch(_steps(), _episodes(), checkpoint_label="ckpt")
    trade_rollup = build_trade_reward_component_rollup(_steps(), _trades(), checkpoint_label="ckpt")
    summary = build_checkpoint_reward_component_summary(mismatch, trade_rollup)

    row = mismatch.iloc[0]
    assert row["score"] == 0.70
    assert row["actual_net_pnl"] == -60.0
    assert row["reward_mkt_closed_sum"] == -0.02
    assert bool(row["score_positive_pnl_negative"]) is True

    summary_row = summary.iloc[0]
    assert summary_row["episodes"] == 1
    assert summary_row["trades"] == 1
    assert summary_row["score_positive_pnl_negative_count"] == 1
    assert summary_row["trade_reward_positive_actual_pnl_negative_count"] == 1


def test_action_count_summary_uses_selected_and_executed_actions():
    action_counts = build_action_count_summary(_steps(), checkpoint_label="ckpt")

    selected = action_counts[action_counts["action_field"] == "selected_action"]
    assert int(selected["count"].sum()) == 2
    assert set(selected["action_value"]) == {1, 2}
    assert set(selected["action_name"]) == {"FLAT", "LONG"}


def test_build_reports_uses_checkpoint_directory_label(tmp_path):
    ckpt = tmp_path / "checkpoints" / "latest_000000123456"
    ckpt.mkdir(parents=True)
    steps = _steps()
    episodes = _episodes()
    trades = _trades()
    trades["checkpoint_label"] = "raw_checkpoint_file_name"
    steps.to_csv(ckpt / "step_reward_components.csv", index=False)
    episodes.to_csv(ckpt / "dreamer_episodes.csv", index=False)
    trades.to_csv(ckpt / "profit_lock_trade_rollup.csv", index=False)

    reports = build_reports(tmp_path)

    summary = reports["checkpoint_reward_component_summary"]
    assert list(summary["checkpoint_label"]) == ["latest_000000123456"]
    assert int(summary.iloc[0]["episodes"]) == 1
    assert int(summary.iloc[0]["trades"]) == 1
