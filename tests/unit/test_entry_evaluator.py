from dataclasses import replace
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from gym_trading_env.envs.position import Position
from gym_trading_env.envs.position_manager import PositionManager
from gym_trading_env.research.entry_evaluator import (
    DataConfig,
    EntryEvalConfig,
    EvaluatorConfig,
    ExperimentConfig,
    ProductConfig,
    SensitivityConfig,
    _market_window,
    build_entry_dataset,
    candidate_rows,
    evaluate_entry,
    summarize_market_window,
)
from gym_trading_env.utils.market_features import FEATURES_MARKET_OBS


pytestmark = pytest.mark.unit


def _config(**rule_overrides) -> EntryEvalConfig:
    rules = EvaluatorConfig(
        entry_delay_bars=1,
        stop_distance_price=5.0,
        stop_distance_ticks=10,
        take_profit_rr=1.5,
        max_hold_bars=60,
        max_hold_unit="valid_1m_bars",
        allow_entry_across_break=False,
        force_flatten_eod=True,
        intrabar_collision="stop_first",
    )
    rules = replace(rules, **rule_overrides)
    return EntryEvalConfig(
        version="test_entry_eval_v1",
        data=DataConfig(
            path="unused.csv",
            timezone="Asia/Shanghai",
            product="DCE.JM",
            contract="JMTEST",
            window_size=3,
            observation_mode="obs",
        ),
        product=ProductConfig(
            initial_balance=10000.0,
            tick_size=0.5,
            spread=0.5,
            trading_fee_per_lot=3.0,
            is_round_turn=True,
            leverage=10.0,
            lot_size=60.0,
            trade_lot=1.0,
            max_entries_per_day=3,
        ),
        entry_evaluator=rules,
        experiment=ExperimentConfig(bootstrap_runs=100),
        sensitivity=SensitivityConfig(),
    )


def _market(
    *,
    minutes=(120, 121, 122, 123, 124),
    days=None,
    mask=None,
    opens=None,
    highs=None,
    lows=None,
    closes=None,
) -> pd.DataFrame:
    count = len(minutes)
    days = list(days or [20250106] * count)
    mask = list(mask or [1.0] * count)
    opens = list(opens or [100.0] * count)
    highs = list(highs or [101.0] * count)
    lows = list(lows or [99.0] * count)
    closes = list(closes or [100.0] * count)
    index = pd.date_range("2025-01-06 09:01", periods=count, freq="1min", tz="Asia/Shanghai")
    frame = pd.DataFrame({
        "trading_day": days,
        "minute_index_t": minutes,
        "mask_t": mask,
        "exec_open": opens,
        "H_t": highs,
        "L_t": lows,
        "C_t": closes,
    }, index=index)
    for idx, name in enumerate(FEATURES_MARKET_OBS):
        frame[name] = np.arange(count, dtype=float) + idx / 100.0
    frame["obs_mask_t"] = np.asarray(mask, dtype=float)
    return frame


def test_stop_distance_tick_mismatch_fails_fast():
    config = _config(stop_distance_price=4.5)
    with pytest.raises(ValueError, match="stop_distance_price"):
        config.validate()


def test_next_open_stop_first_and_costs_match_environment_position_math():
    market = _market(
        opens=[99.0, 100.0, 100.0],
        highs=[100.0, 109.0, 100.0],
        lows=[98.0, 95.0, 99.0],
        closes=[99.0, 100.0, 100.0],
        minutes=(120, 121, 122),
    )
    outcome = evaluate_entry(market, 0, "LONG", _config())

    assert outcome.entry_row == 1
    assert outcome.entry_reference_price == 100.0
    assert outcome.entry_execution_price == 100.5
    assert outcome.stop_target_collision
    assert outcome.exit_reason == "STOP"
    assert outcome.exit_execution_price == 95.5
    assert outcome.gross_pnl == -240.0
    assert outcome.spread_cost == 60.0
    assert outcome.fee_cost == 6.0
    assert outcome.net_pnl == -306.0

    manager = PositionManager(long_slots=1, short_slots=1)
    manager.add_long_position(Position(
        size=Decimal("1"),
        entry_price=Decimal("100.5"),
        initial_margin=Decimal("0"),
    ))
    _, quote = manager.quote_close_long(
        closing_price=Decimal("95.5"), lot_size=Decimal("60"), slot=0)
    assert float(quote.pnl - Decimal("6")) == outcome.net_pnl


def test_signal_on_close_enters_on_decision_close_and_manages_from_next_bar():
    market = _market(
        opens=[99.0, 100.0, 100.0],
        highs=[100.0, 109.0, 100.0],
        lows=[98.0, 95.0, 99.0],
        closes=[100.0, 100.0, 100.0],
        minutes=(120, 121, 122),
    )
    outcome = evaluate_entry(
        market,
        0,
        "LONG",
        _config(entry_delay_bars=0, execution_timing="signal_on_close_plus_spread"),
    )

    assert outcome.entry_row == 0
    assert outcome.entry_reference_price == 100.0
    assert outcome.entry_execution_price == 100.5
    assert outcome.exit_row == 1
    assert outcome.stop_target_collision
    assert outcome.exit_reason == "STOP"


def test_signal_on_close_safety_filters_reject_break_boundary_but_diagnostic_allows_it():
    market = _market(
        minutes=(118, 119, 120, 121, 122),
        mask=(1, 1, 1, 0, 1),
    )
    formal = _config(
        entry_delay_bars=0,
        execution_timing="signal_on_close_plus_spread",
        signal_on_close_safety_filters=True,
    )
    diagnostic = _config(
        entry_delay_bars=0,
        execution_timing="signal_on_close_plus_spread",
        signal_on_close_safety_filters=False,
    )

    assert candidate_rows(market, formal).tolist() == [0]
    assert candidate_rows(market, diagnostic).tolist() == [0, 1, 2]


def test_max_hold_counts_only_valid_bars_and_eod_never_crosses_day():
    market = _market(
        minutes=(120, 121, 122, 123, 124, 125),
        days=(20250106, 20250106, 20250106, 20250106, 20250106, 20250107),
        mask=(1, 1, 0, 1, 1, 1),
        closes=(100, 100, 0, 102, 104, 200),
        highs=(101, 101, 0, 103, 105, 201),
        lows=(99, 99, 0, 101, 103, 199),
    )
    outcome = evaluate_entry(market, 0, "LONG", _config(max_hold_bars=2))
    assert outcome.entry_row == 1
    assert outcome.exit_row == 3
    assert outcome.holding_bars == 2
    assert outcome.exit_reason == "MAX_HOLD"
    assert outcome.trading_day == 20250106

    eod = evaluate_entry(market, 3, "LONG", _config(max_hold_bars=60))
    assert eod.exit_row == 4
    assert eod.exit_reason == "EOD"
    assert eod.trading_day == 20250106


def test_candidates_exclude_segment_break_missing_slot_and_insufficient_margin():
    market = _market(
        minutes=(118, 119, 120, 121, 122),
        mask=(1, 1, 1, 0, 1),
    )
    assert candidate_rows(market, _config()).tolist() == [0]

    poor = replace(_config(), product=replace(_config().product, initial_balance=100.0))
    assert candidate_rows(market, poor).size == 0


def test_future_outcome_changes_do_not_change_decision_input():
    market = _market()
    config = _config(max_hold_bars=3)
    X = market[FEATURES_MARKET_OBS].to_numpy(dtype=np.float32)
    before = summarize_market_window(_market_window(X, 0, config.data.window_size))
    first = evaluate_entry(market, 0, "LONG", config)

    changed = market.copy()
    changed.iloc[2:, changed.columns.get_loc("H_t")] = 120.0
    second = evaluate_entry(changed, 0, "LONG", config)
    after = summarize_market_window(_market_window(
        changed[FEATURES_MARKET_OBS].to_numpy(dtype=np.float32),
        0,
        config.data.window_size,
    ))

    np.testing.assert_array_equal(before, after)
    assert first.exit_reason != second.exit_reason


def test_dataset_build_is_deterministic():
    market = _market()
    first, X_first, names_first = build_entry_dataset(market, _config(max_hold_bars=2))
    second, X_second, names_second = build_entry_dataset(market, _config(max_hold_bars=2))
    pd.testing.assert_frame_equal(first, second)
    np.testing.assert_array_equal(X_first, X_second)
    assert names_first == names_second
