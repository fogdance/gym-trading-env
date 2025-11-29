# tests/unit/test_ledger_basic.py

import pandas as pd
from decimal import Decimal
from pathlib import Path
import tempfile
import yaml
import pytest
pytestmark = pytest.mark.unit

pytest.importorskip("talib")
pytest.importorskip("zigzag")

from gym_trading_env.envs.trading_env import CustomTradingEnv, Action
from gym_trading_env.utils.trade_util import step_wrapper

def test_open_close_updates_ledger_balances():
    # 构造最小 1min 数据（避免 hard gap）
    idx = pd.date_range("2023-01-02 09:00:00", periods=10, freq="1min")
    df = pd.DataFrame({
        "Open":  [100]*10,
        "High":  [101]*10,
        "Low":   [99]*10,
        "Close": [100,100,100,101,101,101,101,101,101,101],
        "Volume":[1]*10
    }, index=idx)

    cfg = {
        "trading": {
            "currency_pair": "TEST",
            "initial_balance": 10000.0,
            "trading_fee_per_lot": 5.0,
            "is_round_turn": False,
            "spread": 0.1,
            "leverage": 10,
            "lot_size": 1,
            "trade_lot": 1.0,
            "max_long_position": 10.0,
            "max_short_position": 10.0,
            "data_path": "TEST",
            "data_interval": "1m",
            "is_future": False
        },
        "risk": {"max_drawdown_ratio": 0.9, "daily_lost_ratio": 0.9, "risk_reward_ratio_enable": False},
        "training": {"window_size": 1, "max_episode_steps": 100, "randomize_start": False, "episode_length": 8, "start_clock": "any"},
        "visualization": {},
        "debug": {"log_level":"ERROR","debug_enabled": False},
    }

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        env = CustomTradingEnv(df=df, config_path=str(p))
        # action mapping: [HOLD, LONG_OPEN0, LONG_CLOSE0, SHORT_OPEN0, SHORT_CLOSE0]
        obs, reward, term, trunc, info = step_wrapper(env, Action.LONG_OPEN0)  # LONG_OPEN0
        # 开仓后：现金减少(保证金+手续费)，保证金账户增加，broker_fee_income 增加
        b1 = env.ledger.balances()
        assert b1["broker_fee_income"] > Decimal("0")
        assert b1["user_margin"] > Decimal("0")
        assert b1["user_cash"] < Decimal("10000")

        obs, reward, term, trunc, info = step_wrapper(env, Action.LONG_CLOSE0)  # LONG_CLOSE0
        b2 = env.ledger.balances()
        # 平仓后保证金释放回 0
        assert b2["user_margin"] == Decimal("0")
        # 至少产生两条分录：open / close
        assert len(env.ledger.entries) >= 2
