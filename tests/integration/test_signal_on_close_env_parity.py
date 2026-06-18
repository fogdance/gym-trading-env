from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("talib")
pytest.importorskip("zigzag")

from gym_trading_env.envs.trading_env import CustomTradingEnv
from gym_trading_env.research.entry_evaluator import (
    build_evaluation_context,
    evaluate_entry,
    load_entry_eval_config,
    load_market_frames,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = Path(os.environ.get(
    "GYM_TRADING_TEST_CONTRACT_ROOT",
    "/data/logdir/trading_contracts/jm_walk_forward_20240603_20251202",
))
CONFIG_PATH = CONTRACT_ROOT / "configs/entry_eval/jm_walk_forward_20240603_20251202_signal_close.yaml"
ENV_CONFIG = CONTRACT_ROOT / "configs/env/jm_walk_forward_20240603_20251202_full.yaml"
DREAMER_ROOT = Path("/home/v/Documents/work/dreamerv3")


def _require_data():
    if not CONFIG_PATH.exists() or not ENV_CONFIG.exists():
        pytest.skip(f"external JM signal-on-close contract is unavailable: {CONTRACT_ROOT}")


def _reset_to_row(env: CustomTradingEnv, row: int):
    original = env._candidate_start_rows_by_clock
    old_randomize = bool(getattr(env.config.training, "randomize_start", True))

    def forced(_clock):
        return np.asarray([int(row)], dtype=np.int64)

    env._candidate_start_rows_by_clock = forced
    env.config.training.randomize_start = False
    try:
        env.reset()
    finally:
        env._candidate_start_rows_by_clock = original
        env.config.training.randomize_start = old_randomize


def _first_open_record(env: CustomTradingEnv, start_len: int):
    for rec in env.trade_record_manager.trade_history[start_len:]:
        data = rec.to_dict()
        if "OPEN" in str(data.get("operation_type", "")):
            return data
    raise AssertionError("no open trade record was produced")


def _case_rows(context) -> list[tuple[str, int]]:
    mask = np.asarray(context.mask)
    valid = np.flatnonzero(mask)
    rows = []
    # Same rows as the audit script, chosen to be affordable and deterministic.
    rows.append(("normal_continuous", 1040))
    rows.append(("session_open", 1035))
    rows.append(("before_break", 944))
    rows.append(("after_break", 945))
    rows.append(("near_eod", 1377))
    for _name, row in rows:
        if row >= len(mask) or not mask[row]:
            pytest.skip("expected deterministic audit row is unavailable")
    return rows


@pytest.mark.parametrize("direction,action_index", [("LONG", 2), ("SHORT", 0)])
def test_signal_on_close_evaluator_matches_env_open_fills(monkeypatch, direction, action_index):
    _require_data()
    monkeypatch.chdir(DREAMER_ROOT)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    config = load_entry_eval_config(CONFIG_PATH)
    # Fill parity deliberately disables formal break/EOD filters so every audited
    # row can compare the raw env fill contract.
    config = replace(
        config,
        entry_evaluator=replace(
            config.entry_evaluator,
            signal_on_close_safety_filters=False,
        ),
    )
    _, market = load_market_frames(config)
    context = build_evaluation_context(market)
    env = CustomTradingEnv(config_path=str(ENV_CONFIG))
    try:
        for _case, row in _case_rows(context):
            _reset_to_row(env, row)
            before = len(env.trade_record_manager.trade_history)
            obs, reward, terminated, truncated, info = env.step(action_index)
            rec = _first_open_record(env, before)
            outcome = evaluate_entry(
                market,
                row,
                direction,
                config,
                context=context,
            )
            assert pd.Timestamp(rec["timestamp"]) == pd.Timestamp(outcome.entry_timestamp)
            assert float(rec["open_price"]) == pytest.approx(
                outcome.entry_execution_price)
            assert int(outcome.entry_row) == int(row)
            assert str(rec["operation_type"]).upper().startswith(direction)
    finally:
        env.close()


def test_signal_on_close_flip_open_matches_env_fill(monkeypatch):
    _require_data()
    monkeypatch.chdir(DREAMER_ROOT)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    config = load_entry_eval_config(CONFIG_PATH)
    _, market = load_market_frames(config)
    context = build_evaluation_context(market)
    env = CustomTradingEnv(config_path=str(ENV_CONFIG))
    try:
        row = 1040
        _reset_to_row(env, row)
        env.step(2)
        row = int(env.current_step)
        before = len(env.trade_record_manager.trade_history)
        env.step(0)
        open_rec = _first_open_record(env, before)
        outcome = evaluate_entry(
            market,
            row,
            "SHORT",
            config,
            context=context,
        )
        assert "FLIP_OPEN_SHORT" == str(open_rec["operation_type"])
        assert pd.Timestamp(open_rec["timestamp"]) == pd.Timestamp(outcome.entry_timestamp)
        assert float(open_rec["open_price"]) == pytest.approx(
            outcome.entry_execution_price)
    finally:
        env.close()
