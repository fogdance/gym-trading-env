# tests/utils/test_agent_features_match_spec.py

from decimal import Decimal

import numpy as np

from tests.oracles.agent_spec_oracle import (
    AgentSnapshot,
    compute_agent_features_raw_spec,
    compute_agent_features_obs_spec,
    FEATURES_AGENT,
    FEATURES_AGENT_OBS,
    to_float32_vector,
)
from tests.utils.assertions import assert_array_allclose_with_diff

from gym_trading_env.utils.decimal_util import D, D0
from gym_trading_env.utils.agent_features import (
    AgentFeatureInput,
    compute_agent_features_raw,
    compute_agent_features_obs,
    agent_feature_vector,
)
from gym_trading_env.envs.action import ForexCode

from gym_trading_env.envs.position import Position

def _mk_pos(
    size: str,
    entry: str,
    open_step: int,
    *,
    initial_margin: str = "0",
    stop_loss_price: str | None = None,
    take_profit_price: str | None = None,
) -> Position:
    return Position(
        size=D(size),
        entry_price=D(entry),
        initial_margin=D(initial_margin),
        open_step=int(open_step),
        stop_loss_price=None if stop_loss_price is None else D(stop_loss_price),
        take_profit_price=None if take_profit_price is None else D(take_profit_price),
    )



def _max_code() -> int:
    return max(int(x.value) for x in ForexCode)


def _run_case(inp: AgentFeatureInput, *, title: str):
    got_raw = compute_agent_features_raw(inp)
    got_obs = compute_agent_features_obs(inp, got_raw)

    snap = AgentSnapshot(
        long_positions=inp.long_positions,
        short_positions=inp.short_positions,
        current_step=int(inp.current_step),
        current_price=Decimal(str(inp.current_price)),
        lot_size=Decimal(str(inp.lot_size)),
        realized_pnl_step=Decimal(str(inp.realized_pnl_step)),
        realized_pnl_cum=Decimal(str(inp.realized_pnl_cum)),
        fee_step=Decimal(str(inp.fee_step)),
        fee_cum=Decimal(str(inp.fee_cum)),
        cash_balance=Decimal(str(inp.cash_balance)),
        used_margin=Decimal(str(inp.used_margin)),
        prev_max_equity=Decimal(str(inp.prev_max_equity)),
        entries_used_today=int(inp.entries_used_today),
        max_entries_per_day=int(inp.max_entries_per_day),
        minutes_to_eod=int(inp.minutes_to_eod),
        day_len=int(inp.day_len),
        initial_balance=Decimal(str(inp.initial_balance)),
        realized_today_cash=Decimal(str(inp.realized_today_cash)),
        R_cash=Decimal(str(inp.R_cash)),
        market_open=int(inp.market_open),
        can_open=int(inp.can_open),
        can_close=int(inp.can_close),
        action_result_code=int(inp.action_result_code),
        action_result_max_code=_max_code(),
    )

    exp_raw = compute_agent_features_raw_spec(snap)
    exp_obs = compute_agent_features_obs_spec(snap, exp_raw)

    # dict -> float32 vectors（与 env 训练口径一致）
    got_raw_vec = agent_feature_vector(got_raw, FEATURES_AGENT)
    got_obs_vec = agent_feature_vector(got_obs, FEATURES_AGENT_OBS)

    exp_raw_vec = to_float32_vector(exp_raw, FEATURES_AGENT)
    exp_obs_vec = to_float32_vector(exp_obs, FEATURES_AGENT_OBS)

    assert_array_allclose_with_diff(
        got_raw_vec.reshape(1, -1),
        exp_raw_vec.reshape(1, -1),
        col_labels=FEATURES_AGENT,
        prefix=f"[{title} RAW]",
        rtol=1e-6,
        atol=1e-6,
    )
    assert_array_allclose_with_diff(
        got_obs_vec.reshape(1, -1),
        exp_obs_vec.reshape(1, -1),
        col_labels=FEATURES_AGENT_OBS,
        prefix=f"[{title} OBS]",
        rtol=1e-6,
        atol=1e-6,
    )


def test_agent_features_match_spec_cases():
    # Case1: flat
    inp = AgentFeatureInput(
        long_positions=[None],
        short_positions=[None],
        current_step=10,
        current_price=D("1000"),
        lot_size=D("10"),
        realized_pnl_step=D0,
        realized_pnl_cum=D0,
        fee_step=D0,
        fee_cum=D0,
        cash_balance=D("100000"),
        used_margin=D0,
        prev_max_equity=D("100000"),
        entries_used_today=0,
        max_entries_per_day=2,
        minutes_to_eod=200,
        day_len=345,
        initial_balance=D("100000"),
        realized_today_cash=D0,
        R_cash=D("100"),
        market_open=1,
        can_open=1,
        can_close=0,
        action_result_code=int(ForexCode.SUCCESS.value),
    )
    _run_case(inp, title="flat")

    # Case2: long single
    inp2 = AgentFeatureInput(
        long_positions=[_mk_pos("1", "1000", 5)],
        short_positions=[None],
        current_step=10,
        current_price=D("1010"),
        lot_size=D("10"),
        realized_pnl_step=D0,
        realized_pnl_cum=D0,
        fee_step=D("1.5"),
        fee_cum=D("1.5"),
        cash_balance=D("99998.5"),
        used_margin=D("5000"),
        prev_max_equity=D("105000"),
        entries_used_today=1,
        max_entries_per_day=2,
        minutes_to_eod=200,
        day_len=345,
        initial_balance=D("100000"),
        realized_today_cash=D0,
        R_cash=D("100"),
        market_open=1,
        can_open=0,
        can_close=1,
        action_result_code=int(ForexCode.SUCCESS.value),
    )
    _run_case(inp2, title="long1")

    # Case3: short multi
    inp3 = AgentFeatureInput(
        long_positions=[None],
        short_positions=[_mk_pos("1", "1005", 2), _mk_pos("2", "1002", 6)],
        current_step=10,
        current_price=D("995"),
        lot_size=D("10"),
        realized_pnl_step=D("0"),
        realized_pnl_cum=D("50"),
        fee_step=D("2"),
        fee_cum=D("10"),
        cash_balance=D("100000"),
        used_margin=D("8000"),
        prev_max_equity=D("110000"),
        entries_used_today=2,
        max_entries_per_day=3,
        minutes_to_eod=10,
        day_len=345,
        initial_balance=D("100000"),
        realized_today_cash=D("50"),
        R_cash=D("100"),
        market_open=1,
        can_open=0,
        can_close=1,
        action_result_code=int(ForexCode.SUCCESS.value),
    )
    _run_case(inp3, title="short_multi")
