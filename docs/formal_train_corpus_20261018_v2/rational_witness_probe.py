"""Read-only exact-arithmetic runtime check; no market, Env or optimizer."""
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys


def probe():
    # Covers the largest approved main-qualified source (31950 positions).
    value = Fraction(332, 333) ** 32000
    witness = str(value)
    restored = Fraction(json.loads(json.dumps(witness)))
    assert restored == value
    assert float(restored) == float(value)
    small = Fraction(332, 333) ** 3
    assert str(small) == '36594368/36926037'
    numerator, denominator = witness.split('/')
    return dict(python_int_max_str_digits=sys.get_int_max_str_digits(),
                numerator_digits=len(numerator), denominator_digits=len(denominator),
                witness_sha256=hashlib.sha256(witness.encode()).hexdigest(),
                exact_roundtrip=True, short_witness_unchanged=True)


def main():
    try:
        if '--child' in sys.argv:
            print(json.dumps(probe()))
            return 0
        if os.environ.get('PYTHONINTMAXSTRDIGITS') != '0' or sys.get_int_max_str_digits() != 0:
            raise ValueError('Exact rational witness runtime requires PYTHONINTMAXSTRDIGITS=0')
        parent = probe()
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
            spawned = pool.submit(probe).result()
        # Native G3 launches a new interpreter with an inherited environment.
        child = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--child'],
                               check=True, capture_output=True, text=True)
        verifier = json.loads(child.stdout)
        assert parent == spawned == verifier
        print(json.dumps(dict(status='PASS_EXACT_RATIONAL_RUNTIME', **parent,
                              spawned_worker_matches=True, verifier_process_matches=True,
                              exponent=32000, optimizer_updates=0, market_prices_opened=0)))
        return 0
    except (ValueError, AssertionError, subprocess.CalledProcessError) as exc:
        print(json.dumps(dict(status='NO_GO', error=str(exc), optimizer_updates=0)))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
