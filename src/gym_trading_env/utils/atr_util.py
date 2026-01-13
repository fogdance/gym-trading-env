# src/gym_trading_env/utils/atr_util.py

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import talib
except Exception:  # pragma: no cover
    talib = None# src/gym_trading_env/utils/atr_util.py


def compute_daily_atr_from_summary_talib(
    daily_summary: pd.DataFrame,
    *,
    period: int = 14,
    shift: int = 1,
) -> pd.Series:
    """
    用 TA-Lib 计算日 ATR（单位=价格），并做 shift 防止偷看。

    - period 动态缩短：p = min(period, n_valid-1)；保证至少 1
      （TA-Lib 的 ATR 需要 prev_close，所以 n_valid<2 时直接返回 0）
    - shift=1 => 当天用到的是前一日及更早的统计
    """
    if daily_summary is None or len(daily_summary) == 0:
        return pd.Series(dtype=float)

    if talib is None:
        raise ImportError("talib is not available. Please install TA-Lib or switch to non-talib ATR.")

    # 取 float arrays
    high = pd.to_numeric(daily_summary["high"], errors="coerce").astype(float).to_numpy()
    low  = pd.to_numeric(daily_summary["low"], errors="coerce").astype(float).to_numpy()
    close = pd.to_numeric(daily_summary["close"], errors="coerce").astype(float).to_numpy()

    # valid mask：三者都得有
    m = np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
    if m.sum() < 2:
        # ATR 需要 prev_close，少于2天没意义
        out = np.zeros(len(daily_summary), dtype=float)
        return pd.Series(out, index=daily_summary.index)

    # 为了保持 index 对齐（不打乱 day 顺序），我们只在 valid 段上算，然后再塞回去
    # 这里假设 daily_summary 是按 days_order 对齐的（你的 build_daily_context_and_seq 已保证）
    idx = daily_summary.index

    # 连续 valid 的场景最常见；如果中间有缺失天，直接在 full arrays 上算会让 talib 传入 nan 产生 nan
    # 所以：把 valid 的点抽出来算，再回填
    hv = high[m]
    lv = low[m]
    cv = close[m]

    n = len(hv)
    # p 至少 1，且不能超过 n-1（因为需要 prev_close 形成 TR）
    p_cfg = int(period) if int(period) > 0 else 14
    p = max(1, min(p_cfg, n - 1))

    atr_v = talib.ATR(hv, lv, cv, timeperiod=p)  # length=n, 前面会有 nan
    if int(shift) > 0:
        atr_v = np.roll(atr_v, int(shift))
        atr_v[: int(shift)] = np.nan

    # 回填到 full length
    out = np.zeros(len(daily_summary), dtype=float)
    out[:] = np.nan
    out[m] = atr_v

    # 训练早期 nan 变 0（后续 reward 里再做兜底）
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    return pd.Series(out, index=idx)

