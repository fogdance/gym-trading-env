# tests/unit/test_features_schema.py
import pytest
from gym_trading_env.utils.market_features import FEATURES_MARKET, build_market_features
from gym_trading_env.utils.agent_features import FEATURES_AGENT

pytestmark = pytest.mark.unit

def _assert_unique(seq, name):
    assert len(seq) == len(set(seq)), f"{name} contains duplicates: {[x for x in seq if seq.count(x) > 1]}"

def test_features_market_schema():
    assert isinstance(FEATURES_MARKET, list)
    assert all(isinstance(x, str) for x in FEATURES_MARKET)
    _assert_unique(FEATURES_MARKET, "FEATURES_MARKET")
    # 强约束：顺序即契约（改动需要显式更新测试）
    assert FEATURES_MARKET[0] == "C_t"
    assert FEATURES_MARKET[-1] == "weekday_cos_t"
    assert "mask_t" in FEATURES_MARKET

def test_features_agent_schema():
    assert isinstance(FEATURES_AGENT, list)
    assert all(isinstance(x, str) for x in FEATURES_AGENT)
    _assert_unique(FEATURES_AGENT, "FEATURES_AGENT")
    assert FEATURES_AGENT[0] == "pos_t"
    assert FEATURES_AGENT[-1] == "minutes_to_timeout_t"
    # 关键字段必须存在
    must = {"equity_t", "max_equity_t", "drawdown_t", "fee_cum_t", "realized_pnl_cum_t"}
    assert must.issubset(set(FEATURES_AGENT))
