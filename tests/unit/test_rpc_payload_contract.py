import json

import pytest

from gym_trading_env.envs.action import Action
from gym_trading_env.envs.action import ForexCode
from gym_trading_env.utils.rpc_protocol import TradeSignal


pytestmark = pytest.mark.unit


def test_trade_signal_uses_execution_action_and_top_level_result():
    sig = TradeSignal(
        signal_id="DCE.jm2605|2025-01-01 21:01:00|FLIP_LONG_TO_SHORT",
        symbol="DCE.jm2605",
        eob="2025-01-01 21:01:00",
        action_index=int(Action.FLIP_LONG_TO_SHORT.value),
        action_name=Action.FLIP_LONG_TO_SHORT.name,
        volume=1,
        price=1234.5,
        action_result=int(ForexCode.SUCCESS.value),
        meta={"result": int(ForexCode.SUCCESS.value), "equity": 100000.0},
    )

    payload = json.loads(sig.to_json())

    assert payload["action_index"] == int(Action.FLIP_LONG_TO_SHORT.value)
    assert payload["action_name"] == "FLIP_LONG_TO_SHORT"
    assert payload["action_result"] == int(ForexCode.SUCCESS.value)
    assert payload["meta"]["result"] == int(ForexCode.SUCCESS.value)
    assert TradeSignal.from_dict(payload).is_success()


def test_trade_signal_from_dict_accepts_live_logger_exec_action_and_meta_result():
    sig = TradeSignal.from_dict(
        {
            "signal_id": "s1",
            "symbol": "DCE.jm2605",
            "eob": "2025-01-01 21:01:00",
            "action_index": 16,
            "action_name": "legacy_name",
            "exec_action": "FLIP_LONG_TO_SHORT",
            "volume": 1,
            "price": 1234.5,
            "meta": {"result": int(ForexCode.ERROR_MARKET_CLOSED.value)},
        }
    )

    assert sig.action_name == "FLIP_LONG_TO_SHORT"
    assert sig.action_result == int(ForexCode.ERROR_MARKET_CLOSED.value)
    assert not sig.is_success()
