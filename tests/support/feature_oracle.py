from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from gym_trading_env.utils.agent_features import (
    AgentFeatureInput,
    compute_agent_features_raw,
    compute_agent_features_obs,
    agent_feature_vector,
    FEATURES_AGENT,
    FEATURES_AGENT_OBS,
)
from gym_trading_env.utils.decimal_util import D, D0


@dataclass
class FeatureOracle:
    """
    测试侧 oracle（对齐现有 env._get_obs()）：
    - market_seq：全局滑窗回看 end_i（允许跨天），数据不足左补 0（不是右补）
    - agent_state：用同一套 agent_features 计算，保证 obs 输出与计算逻辑一致
    """

    @staticmethod
    def pad_len_from_obs_market(env, market_seq: np.ndarray) -> int:
        """
        左 padding 合约下：
        - pad_len = 左侧连续 padding 行数
        优先用 mask 列判断，否则 fallback 用“全 0 行”判断。
        """
        feats = list(getattr(env, "_OBS_FEATURES_MARKET", []))

        # 优先：mask_t / obs_mask_t（看你选择的 feature_mode）
        for mname in ("mask_t", "obs_mask_t"):
            if mname in feats:
                mi = feats.index(mname)
                mask_col = market_seq[:, mi]
                nz = np.flatnonzero(mask_col > 0.0)
                return int(nz[0]) if nz.size else market_seq.shape[0]

        # fallback：用 row 是否全 0
        row_nz = np.flatnonzero(np.any(np.abs(market_seq) > 0.0, axis=1))
        return int(row_nz[0]) if row_nz.size else market_seq.shape[0]

    @staticmethod
    def expected_market_seq(env, end_i: int | None = None) -> np.ndarray:
        """
        完全复刻 env._get_obs() 的 market_seq 逻辑（当前版本）：
        - 全局切片 [end_i-ws+1, end_i]（含）
        - 不足左补 0
        - nan_to_num
        """
        if end_i is None:
            end_i = int(env.current_step)

        ws = int(env.window_size)
        feats = list(env._OBS_FEATURES_MARKET)

        start_i = end_i - ws + 1
        if start_i >= 0:
            window = env.df_market.iloc[start_i:end_i + 1][feats].to_numpy(np.float32, copy=False)
            out = window
        else:
            pad_len = -start_i
            window = env.df_market.iloc[0:end_i + 1][feats].to_numpy(np.float32, copy=False)
            pad = np.zeros((pad_len, len(feats)), dtype=np.float32)
            out = np.concatenate([pad, window], axis=0)

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        assert out.shape == (ws, len(feats))
        return out

    @staticmethod
    def assert_market_columnwise(actual: np.ndarray, expected: np.ndarray, end_i: int):
        assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} vs {expected.shape}"

        for j in range(actual.shape[1]):
            a = actual[:, j]
            e = expected[:, j]
            if not np.allclose(a, e, atol=1e-6, rtol=0):
                idx = int(np.argmax(np.abs(a - e) > 1e-6))
                raise AssertionError(
                    f"market column {j} mismatch at row={idx}, end_i={end_i}: "
                    f"actual={a[idx]} expected={e[idx]}"
                )

    @staticmethod
    def expected_agent_state(env) -> np.ndarray:
        """
        对齐 env._refresh_agent_state 的计算路径（但不依赖 env 内部缓存向量）：
        - 直接用 AgentFeatureInput + compute_agent_features_raw/obs 计算
        - 根据 config.trading.obs_feature_mode 选择 raw/obs 版本
        """
        prev_max = getattr(env, "max_equity", D(getattr(env.config.trading, "initial_balance", 0)))
        eod_idx = int(getattr(env, "_eod_idx", env.end_idx))
        minutes_to_eod = max(0, eod_idx - int(env.current_step))

        day_start_realized = getattr(env, "_day_start_realized_cum", env.user_accounts.realized_pnl)
        realized_today_cash = env.user_accounts.realized_pnl - day_start_realized

        ref_price = getattr(env, "current_price", getattr(env, "_last_valid_price", D0))

        # 复刻 env._refresh_agent_state 的 R_cash
        R_cash = D0
        if bool(getattr(env.config.trading, "stop_loss_enabled", False)):
            mode = getattr(env.config.trading, "stop_loss_mode", "pct")
            slv = getattr(env.config.trading, "stop_loss_value", D0) or D0
            if mode == "pct":
                sl_dist = ref_price * slv
            else:
                sl_dist = slv
            R_cash = sl_dist * env.config.trading.lot_size * env.config.trading.trade_lot

        if R_cash <= D0:
            B0 = D(env.config.trading.initial_balance)
            R_cash = max(B0 * D("0.001"), D("1"))

        inp = AgentFeatureInput(
            long_positions=env.position_manager.long_positions,
            short_positions=env.position_manager.short_positions,
            current_step=int(env.current_step),
            current_price=ref_price,
            lot_size=env.config.trading.lot_size,

            realized_pnl_step=getattr(env, "realized_step", D0),
            realized_pnl_cum=env.user_accounts.realized_pnl,
            fee_step=getattr(env, "fee_step", D0),
            fee_cum=env.broker_accounts.fee_income.get_balance(),

            cash_balance=env.user_accounts.cash_balance.get_balance(),
            used_margin=env.user_accounts.used_margin.get_balance(),

            prev_max_equity=prev_max,

            entries_used_today=int(getattr(env, "_entries_used_today", 0)),
            max_entries_per_day=int(getattr(env.config.trading, "max_entries_per_day", 1)),
            minutes_to_eod=int(minutes_to_eod),
            day_len=int(env.DAY_LEN),

            initial_balance=D(env.config.trading.initial_balance),
            realized_today_cash=realized_today_cash,
            R_cash=R_cash,
        )

        raw = compute_agent_features_raw(inp)
        obs = compute_agent_features_obs(inp, raw)

        mode = getattr(env.config.trading, "obs_feature_mode", "raw")
        use_obs = (mode == "obs")

        if use_obs:
            vec = agent_feature_vector(obs, FEATURES_AGENT_OBS).astype(np.float32, copy=False)
            names = FEATURES_AGENT_OBS
        else:
            vec = agent_feature_vector(raw, FEATURES_AGENT).astype(np.float32, copy=False)
            names = FEATURES_AGENT

        # sanity
        assert vec.shape == (len(names),)
        assert np.isfinite(vec).all()
        return vec

    @staticmethod
    def assert_agent_columnwise(env, actual: np.ndarray, expected: np.ndarray):
        mode = getattr(env.config.trading, "obs_feature_mode", "raw")
        names = FEATURES_AGENT_OBS if mode == "obs" else FEATURES_AGENT

        assert actual.shape == expected.shape == (len(names),)
        a = actual.astype(np.float64)
        e = expected.astype(np.float64)

        for j, name in enumerate(names):
            if not np.isclose(a[j], e[j], rtol=1e-6, atol=1e-6):
                raise AssertionError(f"[AGENT] feature='{name}' mismatch: actual={a[j]} expected={e[j]}")
