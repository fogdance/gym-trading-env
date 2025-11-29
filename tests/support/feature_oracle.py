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
        nz = np.where(np.any(market_seq != 0.0, axis=1))[0]
        assert len(nz) >= 1, "market_seq should reveal at least 1 row (t=0)"
        return int(nz[-1])

    @staticmethod
    def expected_market_seq(env, frontier: int) -> np.ndarray:
        assert hasattr(env, "df_market"), "env.df_market must exist (build_xt output)"
        for c in FEATURES_MARKET:
            assert c in env.df_market.columns, f"env.df_market missing col: {c}"

        full = env.df_market[FEATURES_MARKET].to_numpy(dtype=np.float32, copy=False)
        out = np.zeros_like(full, dtype=np.float32)
        out[: frontier + 1, :] = full[: frontier + 1, :]
        return out

    @staticmethod
    def assert_market_columnwise(actual: np.ndarray, expected: np.ndarray, frontier: int):
        assert actual.shape == expected.shape
        # 只对可见区间逐列对齐（不可见区间 expected 已经是 0）
        vis = slice(0, frontier + 1)

        a = actual[vis, :].astype(np.float64)
        e = expected[vis, :].astype(np.float64)

        assert np.isfinite(a).all(), "obs market_seq visible part must be finite"
        assert np.isfinite(e).all(), "expected market_seq visible part must be finite"

        for j, name in enumerate(FEATURES_MARKET):
            if not np.allclose(a[:, j], e[:, j], rtol=1e-6, atol=1e-6):
                k = int(np.argmax(np.abs(a[:, j] - e[:, j])))
                raise AssertionError(
                    f"[MARKET] feature='{name}' mismatch at row={k}: actual={a[k, j]} expected={e[k, j]}"
                )

        # 不可见区间必须全 0
        if frontier + 1 < actual.shape[0]:
            assert np.all(actual[frontier + 1 :, :] == 0.0), "Future rows must be all-zero"

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
