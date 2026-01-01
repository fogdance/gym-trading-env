# tests/unit/test_day_key_normalization.py
import pytest
import numpy as np

from gym_trading_env.utils.market_store import _day_key_default


pytestmark = pytest.mark.unit



def test_day_key_normalizes_small_numeric_forms():
    """
    Ensure common numeric representations map to the same canonical string.
    This is critical for day_id grouping when upstream uses float/int/numpy scalars.
    """
    keys = [
        _day_key_default(0),
        _day_key_default(0.0),
        _day_key_default(np.int64(0)),
        _day_key_default(np.int32(0)),
        _day_key_default(np.float64(0.0)),
        _day_key_default(np.float32(0.0)),
        _day_key_default("0"),
    ]
    assert all(k == "0" for k in keys)


def test_day_key_preserves_strings():
    assert _day_key_default("2020-01-01") == "2020-01-01"
    assert _day_key_default("foo") == "foo"


def test_day_key_float_non_integer_is_stable():
    # Non-integer floats should stay distinguishable
    assert _day_key_default(0.5) in ("0.5", "0.500000")  # tolerate formatting differences
    assert _day_key_default(np.float64(1.25)) in ("1.25", "1.25000")


@pytest.mark.xfail(
    reason=(
        "Current implementation uses float ':g' for floats; large date-like floats can format as scientific notation. "
        "If your day_id can be 20240520.0-style, consider improving _day_key."
    )
)
def test_day_key_large_date_like_float_should_not_be_scientific():
    # If you ever pass day_id like 20240520.0, you'd usually want '20240520'
    assert _day_key_default(20240520.0) == "20240520"
