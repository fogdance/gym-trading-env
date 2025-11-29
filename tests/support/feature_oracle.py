# tests/support/feature_oracle.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from gym_trading_env.utils.market_features import FEATURES_MARKET, build_market_features
from gym_trading_env.utils.agent_features import FEATURES_AGENT


@dataclass
class FeatureOracle:
    """
    测试侧的“期望值计算器”：
    - market：以 env.df_market 作为 build_xt 的真值来源
    - agent：先从“无持仓契约”开始（后续再加入开/平仓的严格推导）
    """

    @staticmethod
    def frontier_from_obs_market(market_seq: np.ndarray) -> int:
        """
        右 padding 合约下：
        - frontier = 最后一个“可见(非 padding)”行的 index
        """
        # 优先用 mask_t（如果它在 FEATURES_MARKET 里）
        if "mask_t" in FEATURES_MARKET:
            mi = FEATURES_MARKET.index("mask_t")
            mask_col = market_seq[:, mi]
            nz = np.flatnonzero(mask_col > 0.0)
            return int(nz[-1]) if nz.size else 0

        # fallback：用 row 是否全 0 判定
        row_nz = np.flatnonzero(np.any(np.abs(market_seq) > 0.0, axis=1))
        return int(row_nz[-1]) if row_nz.size else 0

    @staticmethod
    def expected_market_seq(env, frontier: int) -> np.ndarray:
        """
        跟 env._get_obs() 同构：
        - 从 env._daily_X 取窗口
        - 右侧补 0
        """
        X_day = env._daily_X[env._day_i]  # (DAY_LEN, F)
        ws = int(env.window_size)

        end = int(min(frontier, env.DAY_LEN - 1))
        start = max(0, end - ws + 1)

        window = X_day[start:end + 1, :]
        L = window.shape[0]
        if L < ws:
            pad = np.zeros((ws - L, env._F_MARKET), dtype=np.float32)
            out = np.concatenate([window, pad], axis=0)
        else:
            out = window.astype(np.float32, copy=False)
        return out.astype(np.float32, copy=False)

    @staticmethod
    def assert_market_columnwise(actual: np.ndarray, expected: np.ndarray, frontier: int):
        assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} vs {expected.shape}"

        # 逐列对比（更容易定位哪一列错）
        for j in range(actual.shape[1]):
            a = actual[:, j]
            e = expected[:, j]
            if not np.allclose(a, e, atol=1e-6, rtol=0):
                # 找第一处 mismatch
                idx = int(np.argmax(np.abs(a - e) > 1e-6))
                raise AssertionError(
                    f"market column {j} mismatch at row={idx}, frontier={frontier}: "
                    f"actual={a[idx]} expected={e[idx]}"
                )

    @staticmethod
    def expected_agent_state_no_position(env) -> np.ndarray:
        """
        无持仓契约 version（先把 obs 返回错误/缺失的问题抓出来）：
        - pos/have_long/have_short/entry_price/holding_minutes/upnl/realized/fee 全部 0
        - equity = 当前现金（无持仓时）
        - max_equity = equity（至少在 reset~前几步 hold 时应成立）
        - drawdown = 0
        - 其它风险/止损止盈/超时字段默认 0
        """
        # 这里我们用 env._get_info() 作为“对外契约”的单一来源（推荐你让 _get_info 保持全量且稳定）
        info = env._get_info() if hasattr(env, "_get_info") else {}
        # 最低限度：得能拿到 equity 或 balance
        equity = None
        if "equity" in info:
            equity = float(info["equity"])
        elif "balance" in info:
            equity = float(info["balance"])
        else:
            # 如果连 equity/balance 都没有，那 agent_state 基本没法做严谨校验
            raise AssertionError("env._get_info() must provide at least 'equity' or 'balance' to validate agent_state")

        vec = {
            "pos_t": 0.0,
            "have_long_t": 0.0,
            "have_short_t": 0.0,
            "entry_price_t": 0.0,
            "holding_minutes_t": 0.0,
            "upnl_t": 0.0,
            "realized_pnl_step_t": 0.0,
            "realized_pnl_cum_t": 0.0,
            "fee_step_t": 0.0,
            "fee_cum_t": 0.0,
            "equity_t": equity,
            "max_equity_t": equity,
            "drawdown_t": 0.0,
            "sigma_entry_t": 0.0,
            "sl_ticks_t": 0.0,
            "tp_ticks_t": 0.0,
            "sl_price_t": 0.0,
            "tp_price_t": 0.0,
            "minutes_to_timeout_t": 0.0,
        }
        return np.asarray([vec[k] for k in FEATURES_AGENT], dtype=np.float32)

    @staticmethod
    def assert_agent_columnwise(actual: np.ndarray, expected: np.ndarray):
        assert actual.shape == expected.shape == (len(FEATURES_AGENT),)
        a = actual.astype(np.float64)
        e = expected.astype(np.float64)
        assert np.isfinite(a).all(), "obs agent_state must be finite"
        assert np.isfinite(e).all(), "expected agent_state must be finite"

        for j, name in enumerate(FEATURES_AGENT):
            if not np.isclose(a[j], e[j], rtol=1e-6, atol=1e-6):
                raise AssertionError(f"[AGENT] feature='{name}' mismatch: actual={a[j]} expected={e[j]}")
