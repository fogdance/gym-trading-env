import numpy as np
from decimal import Decimal
from gym_trading_env.envs.position import Position
from gym_trading_env.utils.decimal_util import D
from gym_trading_env.utils.agent_features import (
    AgentFeatureInput, compute_agent_features, agent_feature_vector
)

def test_agent_features_single_long():
    # long: 0.02 lots, entry 1.1000, opened at step=10
    p = Position(size=D("0.02"), entry_price=D("1.1000"), initial_margin=D("100"), open_step=10)

    inp = AgentFeatureInput(
        long_positions=[p],
        short_positions=[None],
        current_step=70,
        current_price=D("1.1010"),
        lot_size=D("100000"),

        realized_pnl_step=D("0"),
        realized_pnl_cum=D("5"),
        fee_step=D("0.5"),
        fee_cum=D("1.0"),

        cash_balance=D("10000"),
        used_margin=D("1000"),
        prev_max_equity=D("11050"),
    )

    feat = compute_agent_features(inp)

    # 手算 upnl = (1.1010-1.1000)*0.02*100000 = 2
    assert feat["pos_t"] == D("0.02")
    assert feat["have_long_t"] == 1
    assert feat["have_short_t"] == 0
    assert feat["entry_price_t"] == D("1.1000")
    assert feat["holding_minutes_t"] == D("60")
    assert feat["upnl_t"] == D("2")

    # equity = cash + margin + upnl = 10000 + 1000 + 2 = 11002
    assert feat["equity_t"] == D("11002")
    # prev_max=11050 -> dd=48
    assert feat["max_equity_t"] == D("11050")
    assert feat["drawdown_t"] == D("48")

    vec = agent_feature_vector(feat)
    assert vec.dtype == np.float32
    assert np.isclose(vec[0], 0.02, atol=1e-6)


def test_agent_features_hedge_not_flat():
    # long == short -> net=0，但不是 flat（have_long/have_short 都应该是 1）
    L = Position(size=D("0.02"), entry_price=D("1.1000"), initial_margin=D("100"), open_step=10)
    S = Position(size=D("0.02"), entry_price=D("1.0995"), initial_margin=D("100"), open_step=12)

    inp = AgentFeatureInput(
        long_positions=[L],
        short_positions=[S],
        current_step=20,
        current_price=D("1.1000"),
        lot_size=D("100000"),
        realized_pnl_step=D("0"),
        realized_pnl_cum=D("0"),
        fee_step=D("0"),
        fee_cum=D("0"),
        cash_balance=D("10000"),
        used_margin=D("0"),
        prev_max_equity=D("10000"),
    )
    feat = compute_agent_features(inp)
    assert feat["pos_t"] == D("0")          # 净仓为 0
    assert feat["have_long_t"] == 1
    assert feat["have_short_t"] == 1        # 关键：不是 flat
