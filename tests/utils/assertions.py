# tests/utils/assertions.py

from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd


def assert_frame_allclose(
    got: pd.DataFrame,
    exp: pd.DataFrame,
    *,
    cols: Optional[List[str]] = None,
    rtol: float = 1e-6,
    atol: float = 1e-6,
):
    if cols is None:
        cols = list(exp.columns)

    missing = [c for c in cols if c not in got.columns]
    if missing:
        raise AssertionError(f"missing columns in got: {missing}")

    g = got[cols].to_numpy(dtype=float, copy=False)
    e = exp[cols].to_numpy(dtype=float, copy=False)

    if g.shape != e.shape:
        raise AssertionError(f"shape mismatch: got={g.shape}, exp={e.shape}")

    ok = np.isclose(g, e, rtol=rtol, atol=atol) | (np.isnan(g) & np.isnan(e))
    if ok.all():
        return

    bad = np.argwhere(~ok)
    # 打印前 30 条
    lines = []
    for k in range(min(len(bad), 30)):
        i, j = bad[k]
        col = cols[j]
        lines.append(f"row={i} col={col} got={g[i,j]!r} exp={e[i,j]!r} diff={g[i,j]-e[i,j]}")
    raise AssertionError("DataFrame allclose failed:\n" + "\n".join(lines))


def assert_array_allclose_with_diff(
    got: np.ndarray,
    exp: np.ndarray,
    *,
    row_labels: Optional[List[int]] = None,
    col_labels: Optional[List[str]] = None,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    max_print: int = 30,
    prefix: str = "",
):
    g = np.asarray(got, dtype=float)
    e = np.asarray(exp, dtype=float)

    if g.shape != e.shape:
        raise AssertionError(f"{prefix} shape mismatch: got={g.shape}, exp={e.shape}")

    ok = np.isclose(g, e, rtol=rtol, atol=atol) | (np.isnan(g) & np.isnan(e))
    if ok.all():
        return

    bad = np.argwhere(~ok)
    lines = []
    for k in range(min(len(bad), max_print)):
        i, j = bad[k]
        r = row_labels[i] if row_labels is not None else i
        c = col_labels[j] if col_labels is not None else j
        lines.append(f"{prefix} row={r} col={c} got={g[i,j]!r} exp={e[i,j]!r} diff={g[i,j]-e[i,j]}")
    raise AssertionError(f"{prefix} allclose failed:\n" + "\n".join(lines))
