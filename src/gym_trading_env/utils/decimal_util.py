# src/gym_trading_env/utils/decimal_util.py

from decimal import Decimal, getcontext, ROUND_HALF_UP

getcontext().prec = 28
getcontext().rounding = ROUND_HALF_UP

D  = lambda x: x if isinstance(x, Decimal) else Decimal(str(x))

D0  = Decimal('0')
D1  = Decimal('1')
D100= Decimal('100')

def quantize_money(x: Decimal, places='0.01') -> Decimal:
    return x.quantize(Decimal(places))

def to_decimal_series(values) -> list[Decimal]:
    return [D(v) for v in values]

def to_float_array(values) -> "np.ndarray":
    import numpy as np
    return np.asarray(values, dtype='float64')

def safe_sum(decimals) -> Decimal:
    from decimal import Decimal
    return sum(decimals, Decimal('0'))

def decimal_to_float(value, precision=5):
    """
    Converts a Decimal to float with specified precision.
    
    Args:
        value (Decimal): The Decimal value to convert.
        precision (int): Number of decimal places.
    
    Returns:
        float: The converted float value.
    """
    quantize_str = '1.' + '0' * precision
    if isinstance(value, Decimal):
        return float(value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP))
    else:
        raise TypeError("Value must be a Decimal.")
    
def float_to_decimal(value):
    return D(value)

