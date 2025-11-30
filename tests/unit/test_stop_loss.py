# tests/unit/test_stop_loss.py
import pandas as pd
import numpy as np
from decimal import Decimal
from pathlib import Path
import tempfile
import yaml
import pytest

pytestmark = pytest.mark.unit

from gym_trading_env.envs.trading_env import CustomTradingEnv, Action
from gym_trading_env.utils.trade_util import step_wrapper
from gym_trading_env.utils.decimal_util import D, D0


def _make_cfg(stop_loss_mode: str, stop_loss_value: float):
    return {
        "trading": {
            "currency_pair": "TEST",
            "initial_balance": 10000.0,
            "trading_fee_per_lot": 0.0,
            "is_round_turn": False,
            "spread": 0.0,
            "leverage": 10,
            "lot_size": 1,
            "trade_lot": 1.0,
            "max_long_position": 10.0,
            "max_short_position": 10.0,
            "data_path": "TEST",
            "data_interval": "1m",
            "is_future": True,
            # --- stop loss ---
            "stop_loss_enabled": True,
            "stop_loss_mode": stop_loss_mode,   # "pct" | "abs"
            "stop_loss_value": stop_loss_value,
        },
        "risk": {"max_drawdown_ratio": 0.9, "daily_lost_ratio": 0.9, "risk_reward_ratio_enable": False},
        "training": {"window_size": 1, "max_episode_steps": 100, "randomize_start": False, "episode_length": 2, "start_clock": "any"},
        "visualization": {},
        "debug": {"log_level": "INFO", "debug_enabled": True},
    }


def _run_env_once(df: pd.DataFrame, cfg: dict, act: Action):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        env = CustomTradingEnv(df=df, config_path=str(p))
        env.config.training.randomize_start = False
        env.reset()

        obs, r, term, trunc, info = step_wrapper(env, act)
        env.close()
        return obs, r, term, trunc, info, env  # env 已 close，但 info 足够


@pytest.mark.parametrize(
    "mode, sl_value",
    [
        ("pct", 0.005),  # 0.5%
        ("abs", 1.0),    # 1.0 price unit
    ],
)
def test_stop_loss_long_triggers(mode, sl_value):
    """
    t0 开多；t1 low 打穿止损 -> 在同一次 step(LONG_OPEN0) 末尾被自动平仓
    """
    entry = 100.0
    if mode == "pct":
        stop = entry * (1.0 - sl_value)  # 99.5
    else:
        stop = entry - sl_value          # 99.0

    # t1 low 越过止损（<= stop）
    low_t1 = stop - 0.1

    idx = pd.date_range("2023-01-02 21:01:00", periods=3, freq="1min")
    df = pd.DataFrame(
        {
            "Open":   [entry, entry, entry],
            "High":   [entry, entry, entry],
            "Low":    [entry, low_t1, entry],
            "Close":  [entry, entry - 0.2, entry],
            "Volume": [1, 1, 1],
        },
        index=idx,
    )

    cfg = _make_cfg(mode, sl_value)

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        env = CustomTradingEnv(df=df, config_path=str(p))
        env.config.training.randomize_start = False
        env.reset()

        # 开仓（但会在 t1 被 SL 自动平）
        _, _, _, _, info = step_wrapper(env, Action.LONG_OPEN0)

        assert D(info["long_position"]) == D0
        assert D(info["used_margin"]) == D0
        assert int(info.get("stop_loss_fired", 0)) >= 1
        assert len(env.ledger.entries) >= 2  # OPEN + SL CLOSE

        env.close()


@pytest.mark.parametrize(
    "mode, sl_value",
    [
        ("pct", 0.005),  # 0.5%
        ("abs", 1.0),    # 1.0 price unit
    ],
)
def test_stop_loss_short_triggers(mode, sl_value):
    """
    t0 开空；t1 high 打穿止损 -> 在同一次 step(SHORT_OPEN0) 末尾被自动平仓
    """
    entry = 100.0
    if mode == "pct":
        stop = entry * (1.0 + sl_value)  # 100.5
    else:
        stop = entry + sl_value          # 101.0

    # t1 high 越过止损（>= stop）
    high_t1 = stop + 0.1

    idx = pd.date_range("2023-01-02 21:01:00", periods=3, freq="1min")
    df = pd.DataFrame(
        {
            "Open":   [entry, entry, entry],
            "High":   [entry, high_t1, entry],
            "Low":    [entry, entry, entry],
            "Close":  [entry, entry + 0.2, entry],
            "Volume": [1, 1, 1],
        },
        index=idx,
    )

    cfg = _make_cfg(mode, sl_value)

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        env = CustomTradingEnv(df=df, config_path=str(p))
        env.config.training.randomize_start = False
        env.reset()

        # 开仓（但会在 t1 被 SL 自动平）
        _, _, _, _, info = step_wrapper(env, Action.SHORT_OPEN0)

        assert D(info["short_position"]) == D0
        assert D(info["used_margin"]) == D0
        assert int(info.get("stop_loss_fired", 0)) >= 1
        assert len(env.ledger.entries) >= 2  # OPEN + SL CLOSE

        env.close()
