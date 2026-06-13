from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytestmark = pytest.mark.integration

pytest.importorskip("talib")
pytest.importorskip("zigzag")

from gym_trading_env.envs.trading_env import CustomTradingEnv
from gym_trading_env.research.entry_evaluator import (
    DataConfig,
    EntryEvalConfig,
    EvaluatorConfig,
    ExperimentConfig,
    ProductConfig,
    SensitivityConfig,
    evaluate_entry,
)
from gym_trading_env.utils.decimal_util import D


def _evaluator_config() -> EntryEvalConfig:
    return EntryEvalConfig(
        version="ledger_parity_v1",
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
        entry_evaluator=EvaluatorConfig(
            entry_delay_bars=1,
            stop_distance_price=5.0,
            stop_distance_ticks=10,
            take_profit_rr=1.5,
            max_hold_bars=60,
            max_hold_unit="valid_1m_bars",
            allow_entry_across_break=False,
            force_flatten_eod=True,
            intrabar_collision="stop_first",
        ),
        experiment=ExperimentConfig(),
        sensitivity=SensitivityConfig(),
    )


def test_fixed_long_trade_matches_environment_ledger(tmp_path: Path):
    market = pd.DataFrame({
        "trading_day": [20250106, 20250106, 20250106],
        "minute_index_t": [120, 121, 122],
        "mask_t": [1.0, 1.0, 1.0],
        "exec_open": [99.0, 100.0, 100.0],
        "H_t": [100.0, 101.0, 100.0],
        "L_t": [98.0, 95.0, 99.0],
        "C_t": [99.0, 100.0, 100.0],
    }, index=pd.date_range(
        "2025-01-06 09:01", periods=3, freq="1min", tz="Asia/Shanghai"))
    outcome = evaluate_entry(market, 0, "LONG", _evaluator_config())
    assert outcome.exit_reason == "STOP"

    env_df = pd.DataFrame({
        "Open": [100.0] * 4,
        "High": [101.0] * 4,
        "Low": [99.0] * 4,
        "Close": [100.0] * 4,
        "Volume": [1.0] * 4,
    }, index=pd.date_range("2025-01-06 09:00", periods=4, freq="1min"))
    config = {
        "trading": {
            "currency_pair": "LEDGER_PARITY",
            "initial_balance": 10000.0,
            "trading_fee_per_lot": 3.0,
            "is_round_turn": True,
            "spread": 0.5,
            "leverage": 10,
            "lot_size": 60,
            "trade_lot": 1.0,
            "max_long_position": 1.0,
            "max_short_position": 1.0,
            "max_entries_per_day": 3,
            "data_path": "unused",
            "data_interval": "1m",
            "is_future": False,
        },
        "risk": {
            "max_drawdown_ratio": 0.99,
            "daily_lost_ratio": 0.99,
            "risk_reward_ratio_enable": False,
        },
        "training": {
            "window_size": 1,
            "max_episode_steps": 10,
            "randomize_start": False,
            "start_clock": "any",
        },
        "visualization": {},
        "debug": {"log_level": "ERROR", "debug_enabled": False},
    }
    config_path = tmp_path / "ledger_parity.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    env = CustomTradingEnv(df=env_df, config_path=str(config_path))
    assert env._long_open(price=D("100"), spread=D("0.5")).name == "SUCCESS"
    assert env._long_close(price=D("96"), spread=D("0.5")).name == "SUCCESS"

    balances = env.ledger.balances()
    assert balances["user_margin"] == Decimal("0")
    assert balances["broker_fee_income"] == Decimal(str(outcome.fee_cost))
    assert balances["user_cash"] == Decimal("10000") + Decimal(str(outcome.net_pnl))
