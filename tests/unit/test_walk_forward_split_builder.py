from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from gym_trading_env.research.walk_forward_split_builder import (
    BuilderPaths,
    SplitDates,
    build_walk_forward_split,
    day_int,
    file_sha256,
    load_configured_split_manifest,
    load_required_split_manifest,
)
from gym_trading_env.utils.data_processing import load_data


pytestmark = pytest.mark.unit


def _write_market_csv(path: Path) -> None:
    rows = []
    for day in ("2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"):
        for minute in ("09:00:00", "09:01:00"):
            rows.append({
                "Date": f"{day} {minute}",
                "Open": 100.0 + len(rows),
                "High": 101.0 + len(rows),
                "Low": 99.0 + len(rows),
                "Close": 100.5 + len(rows),
                "Volume": 10 + len(rows),
                "OpenInterest": 1000 + len(rows),
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_templates(env_template: Path, entry_eval_template: Path) -> None:
    env_template.write_text(yaml.safe_dump({
        "trading": {
            "currency_pair": "TEST",
            "data_path": "OLD_SYMBOL",
            "data_interval": "1m",
        },
        "training": {
            "reward_function": "futures_intraday_mtm_risk_reward_function",
        },
    }, sort_keys=False))
    entry_eval_template.write_text(yaml.safe_dump({
        "version": "old_version",
        "data": {
            "path": "old.csv",
            "timezone": "Asia/Shanghai",
            "product": "TEST",
            "contract": "TEST_CONTRACT",
            "window_size": 60,
            "observation_mode": "obs",
        },
        "product": {
            "initial_balance": 10000.0,
            "tick_size": 1.0,
            "spread": 1.0,
            "trading_fee_per_lot": 3.0,
            "is_round_turn": True,
            "leverage": 10.0,
            "lot_size": 10.0,
            "trade_lot": 1.0,
            "max_entries_per_day": 3,
        },
        "entry_evaluator": {
            "execution_timing": "signal_on_close_plus_spread",
            "entry_delay_bars": 0,
            "signal_on_close_safety_filters": True,
            "near_eod_bars": 2,
            "stop_distance_price": 5.0,
            "stop_distance_ticks": 5,
            "take_profit_rr": 1.5,
            "max_hold_bars": 60,
            "max_hold_unit": "valid_1m_bars",
            "allow_entry_across_break": False,
            "force_flatten_eod": True,
            "intrabar_collision": "stop_first",
        },
        "experiment": {"random_seed": 1},
        "sensitivity": {},
    }, sort_keys=False))


def test_build_walk_forward_split_assets_are_chronological_and_train_only(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    env_template = tmp_path / "env.yaml"
    entry_eval_template = tmp_path / "entry_eval.yaml"
    _write_market_csv(raw_csv)
    _write_templates(env_template, entry_eval_template)

    summary = build_walk_forward_split(
        paths=BuilderPaths(
            raw_csv=raw_csv,
            env_template=env_template,
            entry_eval_template=entry_eval_template,
            output_root=tmp_path,
            split_name="test_split_v1",
            raw_output_name="test_raw.csv",
            train_output_name="test_train.csv",
        ),
        dates=SplitDates(
            train_start=day_int("2025-01-02"),
            train_end=day_int("2025-01-03"),
            validation_start=day_int("2025-01-06"),
            validation_end=day_int("2025-01-06"),
            test_start=day_int("2025-01-07"),
            test_end=day_int("2025-01-07"),
        ),
    )

    train_csv = Path(summary["train_csv"])
    raw_copy = Path(summary["raw_csv"])
    split_manifest = Path(summary["split_manifest"])
    assert train_csv.exists()
    assert raw_copy.exists()
    assert split_manifest.exists()
    assert summary["split_manifest_hash"] == file_sha256(split_manifest)

    train_days = pd.to_datetime(pd.read_csv(train_csv)["Date"]).dt.strftime("%Y%m%d").astype(int)
    assert sorted(train_days.unique().tolist()) == [20250102, 20250103]

    entry_eval = yaml.safe_load(Path(summary["entry_eval_config"]).read_text())
    assert entry_eval["version"] == "test_split_v1"
    assert entry_eval["data"]["path"] == str(raw_copy.resolve())
    assert entry_eval["data"]["contract"] == "test_split_v1"
    assert entry_eval["walk_forward"]["split_manifest_path"] == str(split_manifest.resolve())

    train_env = yaml.safe_load(Path(summary["train_env_config"]).read_text())
    full_env = yaml.safe_load(Path(summary["full_env_config"]).read_text())
    assert train_env["trading"]["data_path"] == str(train_csv.resolve())
    assert full_env["trading"]["data_path"] == str(raw_copy.resolve())

    folds = load_configured_split_manifest(summary["entry_eval_config"])
    assert folds == [{
        "name": "test_split_v1",
        "train_days": [20250102, 20250103],
        "validation_days": [20250106],
        "test_days": [20250107],
    }]
    assert load_required_split_manifest(summary["entry_eval_config"]) == folds


def test_load_required_split_manifest_rejects_missing_configured_manifest(tmp_path):
    config_path = tmp_path / "entry_eval.yaml"
    config_path.write_text("version: missing_split\n")

    with pytest.raises(ValueError, match="walk_forward.split_manifest_path"):
        load_required_split_manifest(config_path)


def test_split_builder_rejects_overlap(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    env_template = tmp_path / "env.yaml"
    entry_eval_template = tmp_path / "entry_eval.yaml"
    _write_market_csv(raw_csv)
    _write_templates(env_template, entry_eval_template)

    with pytest.raises(ValueError, match="train must end before validation"):
        build_walk_forward_split(
            paths=BuilderPaths(
                raw_csv=raw_csv,
                env_template=env_template,
                entry_eval_template=entry_eval_template,
                output_root=tmp_path,
                split_name="bad_split",
                raw_output_name="raw_out.csv",
                train_output_name="train_out.csv",
            ),
            dates=SplitDates(
                train_start=20250102,
                train_end=20250106,
                validation_start=20250103,
                validation_end=20250106,
                test_start=20250107,
                test_end=20250107,
            ),
        )


def test_load_data_accepts_explicit_csv_path(tmp_path):
    raw_csv = tmp_path / "explicit.csv"
    _write_market_csv(raw_csv)

    loaded = load_data(str(raw_csv), "1m")
    assert len(loaded) == 8
    assert loaded.index.name == "Date"
    assert "Close" in loaded.columns
