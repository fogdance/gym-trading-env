"""Canonical reversible exact rational witnesses for new DATA publications."""
from fractions import Fraction
from functools import lru_cache


ENCODING = 'FG_S3_EXACT_RATIONAL_HEX_V1'


def encode(value):
    return f'h:{value.numerator:x}/{value.denominator:x}'


@lru_cache(maxsize=128)
def decode(text):
    if not isinstance(text,str) or not text.startswith('h:'):
        raise ValueError('Exact witness requires canonical hexadecimal rational encoding')
    try:
        numerator,denominator=text[2:].split('/')
        value=Fraction(int(numerator,16),int(denominator,16))
    except (ValueError,ZeroDivisionError) as error:
        raise ValueError('Malformed exact rational witness') from error
    if encode(value)!=text:raise ValueError('Noncanonical exact rational witness')
    return value
