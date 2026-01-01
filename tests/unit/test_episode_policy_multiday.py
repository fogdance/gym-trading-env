# tests/unit/test_episode_policy_multiday.py

import logging
import numpy as np
from types import SimpleNamespace
from decimal import Decimal

from gym_trading_env.envs.trading_env import CustomTradingEnv


def _mk_env_stub(truncate_on_session_end: bool):
    """
    Create a minimal env instance WITHOUT calling __init__,
    only wiring fields that _should_terminated() needs.
    """
    env = CustomTradingEnv.__new__(CustomTradingEnv)
    env.logger = logging.getLogger("test_episode_policy")
    env.logger.setLevel(logging.CRITICAL)

    # 2 sessions (2 trading days), each 345 bars
    day_ranges = [(0, 345), (345, 690)]
    store = SimpleNamespace(
        n_rows=690,
        day_ranges=day_ranges,
    )
    env.bar_source = SimpleNamespace(store=store)

    env.config = SimpleNamespace(
        trading=SimpleNamespace(
            is_future=True,
            intraday_mode=True,   # keep session semantics ON
        ),
        training=SimpleNamespace(
            episode_length=None,
            max_episode_steps=10_000_000,
            episode_policy=SimpleNamespace(truncate_on_session_end=truncate_on_session_end),
        ),
        risk=SimpleNamespace(
            daily_lost_ratio=1.0,
            max_drawdown_ratio=1.0,
            risk_reward_ratio_enable=False,
            risk_reward_ratio=Decimal("0"),
        ),
        debug=SimpleNamespace(debug_enabled=False),
    )

    # required runtime fields
    env.start_idx = 0
    env.end_idx = 690
    env.episode_step_count = 0

    env.terminated = False
    env.truncated = False

    # stubs to avoid touching other subsystems
    env._check_margin = lambda: False
    env._force_flatten_if_any = lambda reason: None
    env.metrics = SimpleNamespace(get_metrics=lambda: {
        "current_day_lost_pct": Decimal("0"),
        "current_drawdown_pct": Decimal("0"),
    })
    env.position_manager = SimpleNamespace(calc_profit_factor=lambda: None)

    return env


def test_multiday_episode_not_truncated_at_session_end_when_policy_false():
    env = _mk_env_stub(truncate_on_session_end=False)

    # at end of day1 (index 344), should NOT truncate
    env._day_i = 0
    env.current_step = 344

    done = env._should_terminated()
    assert done is False
    assert env.truncated is False
    assert env.terminated is False

    # at end of whole episode (index 689), should truncate (end_idx boundary)
    env._day_i = 1
    env.current_step = 689

    done = env._should_terminated()
    assert done is True
    assert env.truncated is True
    assert env.terminated is False


def test_intraday_episode_truncated_at_session_end_when_policy_true():
    env = _mk_env_stub(truncate_on_session_end=True)

    # at end of day1, should truncate
    env._day_i = 0
    env.current_step = 344

    done = env._should_terminated()
    assert done is True
    assert env.truncated is True
    assert env.terminated is False
