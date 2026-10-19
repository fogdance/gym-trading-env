"""New exact evidence is reversible without decimal conversion or float ranking."""
from fractions import Fraction
import sys

import pytest

from gym_trading_env.research.fg_s3_exact_witness import encode,decode


def test_long_witness_exact_roundtrip_with_default_decimal_limit():
    value=Fraction(332,333)**32000
    before=sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(4300)
        for expected in (Fraction(0),Fraction(-7,3),value,2*value):
            assert decode(encode(expected))==expected
        assert decode(encode(value))<decode(encode(2*value))
        assert decode(encode(value))==decode(encode(value))
    finally:sys.set_int_max_str_digits(before)


@pytest.mark.parametrize('text',['1/2','h:1/0','h:01/2','h:2/4','h:1/-2','h:1/2/3'])
def test_malformed_or_old_witnesses_are_rejected(text):
    with pytest.raises(ValueError):decode(text)


def test_exact_suffix_cache_is_shared_readonly_and_detects_mutation(tmp_path):
    import pickle
    from gym_trading_env.research.fg_s3_suffix_store import ExactSuffixStore
    values={0:Fraction(332,333)**2000,7:Fraction(-7,3),8:Fraction(0)}
    store=ExactSuffixStore.write(tmp_path/'store',values)
    clone=pickle.loads(pickle.dumps(store))
    try:
        assert len(pickle.dumps(store))<512
        for cursor,value in values.items():assert store.get(cursor)==clone.get(cursor)==value
        with pytest.raises(TypeError):clone.mapping[0]=1
    finally:clone.close();store.close()
    body=tmp_path/'store/values.bin';data=bytearray(body.read_bytes());data[0]^=1;body.write_bytes(data)
    with pytest.raises(ValueError,match='values changed'):
        pickle.loads(pickle.dumps(store))
