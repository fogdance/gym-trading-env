# src/gym_trading_env/utils/conversion.py

from decimal import Decimal, ROUND_HALF_UP

def decimal_to_float(value, precision=2):
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
