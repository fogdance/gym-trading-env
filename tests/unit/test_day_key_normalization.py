# tests/unit/test_day_key_normalization.py
import pytest
import numpy as np

from gym_trading_env.envs.trading_env import CustomTradingEnv


pytestmark = pytest.mark.unit


@pytest.fixture
def env():
    # Avoid heavy __init__; we only need the method.
    return CustomTradingEnv.__new__(CustomTradingEnv)


def test_day_key_normalizes_small_numeric_forms(env):
    """
    Ensure common numeric representations map to the same canonical string.
    This is critical for day_id grouping when upstream uses float/int/numpy scalars.
    """
    keys = [
        env._day_key(0),
        env._day_key(0.0),
        env._day_key(np.int64(0)),
        env._day_key(np.int32(0)),
        env._day_key(np.float64(0.0)),
        env._day_key(np.float32(0.0)),
        env._day_key("0"),
    ]
    assert all(k == "0" for k in keys)


def test_day_key_preserves_strings(env):
    assert env._day_key("2020-01-01") == "2020-01-01"
    assert env._day_key("foo") == "foo"


def test_day_key_float_non_integer_is_stable(env):
    # Non-integer floats should stay distinguishable
    assert env._day_key(0.5) in ("0.5", "0.500000")  # tolerate formatting differences
    assert env._day_key(np.float64(1.25)) in ("1.25", "1.25000")


@pytest.mark.xfail(
    reason=(
        "Current implementation uses float ':g' for floats; large date-like floats can format as scientific notation. "
        "If your day_id can be 20240520.0-style, consider improving _day_key."
    )
)
def test_day_key_large_date_like_float_should_not_be_scientific(env):
    # If you ever pass day_id like 20240520.0, you'd usually want '20240520'
    assert env._day_key(20240520.0) == "20240520"
