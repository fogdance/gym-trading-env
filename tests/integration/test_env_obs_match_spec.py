# tests/integration/test_env_obs_match_spec.py

from decimal import Decimal
import numpy as np
import pytest
from gym_trading_env.envs.action import Action
from tests.factories.market_data_factory import MarketDataFactory
from tests.oracles.market_spec_oracle import (
    build_market_features_spec,
    FEATURES_MARKET,
    FEATURES_MARKET_OBS,
    window_slice_spec_by_cols,
)
from tests.oracles.agent_spec_oracle import (
    AgentSnapshot,
    compute_agent_features_raw_spec,
    compute_agent_features_obs_spec,
    FEATURES_AGENT,
    FEATURES_AGENT_OBS,
    to_float32_vector,
)
from tests.utils.assertions import assert_array_allclose_with_diff

from gym_trading_env.envs.trading_env import CustomTradingEnv
from gym_trading_env.envs.action import ForexCode
from gym_trading_env.envs.position import Position


def _max_code():
    return max(int(x.value) for x in ForexCode)


def _to_decimal(x) -> Decimal:
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


def _pos_from_any(p):
    """
    snapshot 里 position 可能是：
    - None
    - Position 实例
    - dict (Position.to_dict())
    这里统一还原为 Position/None，喂给 agent_spec_oracle。
    """
    if p is None:
        return None
    if isinstance(p, Position):
        return p
    if isinstance(p, dict):
        sl = p.get("stop_loss_price", None)
        tp = p.get("take_profit_price", None)
        # 兼容 "None" 字符串
        if sl in ("None", ""):
            sl = None
        if tp in ("None", ""):
            tp = None
        return Position(
            size=_to_decimal(p["size"]),
            entry_price=_to_decimal(p["entry_price"]),
            initial_margin=_to_decimal(p.get("initial_margin", "0")),
            open_step=int(p.get("open_step", 0)),
            stop_loss_price=None if sl is None else _to_decimal(sl),
            take_profit_price=None if tp is None else _to_decimal(tp),
        )
    raise TypeError(f"Unsupported position type in snapshot: {type(p)}")

def _allow_cross_day_episode_for_test(env):
    """
    integration 为了 skip day0 + flush window，需要 episode 能跨 session/day。
    只在测试里改 config，不侵入 env 实现。
    """
    cfg = getattr(env, "config", None)
    if cfg is None or not hasattr(cfg, "training"):
        return

    tr = cfg.training

    # 1) 不要在 session end 截断，否则永远到不了 day_i==1
    ep = getattr(tr, "episode_policy", None)
    if ep is not None and hasattr(ep, "truncate_on_session_end"):
        ep.truncate_on_session_end = False

    # 2) episode_length 放开，避免太短导致提前 truncated
    if hasattr(tr, "episode_length"):
        tr.episode_length = None

    # 3) max_episode_steps 放开（0 表示不限制：以你们 _should_terminated 的逻辑为准）
    if hasattr(tr, "max_episode_steps"):
        tr.max_episode_steps = 0


def _warmup_skip_first_day_and_flush_window(env, obs):
    """
    跳过 day_i=0（第一交易日），并确保 market_seq 的 window 不再包含 day0 的行。
    依赖 env.get_oracle_snapshot().meta.day_i + meta.step。
    """
    win = int(getattr(env, "window_size"))
    snap = env.get_oracle_snapshot()
    meta = snap.get("meta", {})
    if "day_i" not in meta or "step" not in meta:
        # 兜底：strict_345 futures，直接走 345 + win-1（侵入性最低）
        for _ in range(345 + win - 1):
            obs, *_ = env.step(0)
        return obs

    # 1) 找到 day_i==1 的起点 step（第二个交易日开始的那一根）
    day1_start = None
    guard = 0
    while True:
        snap = env.get_oracle_snapshot()
        meta = snap["meta"]
        if int(meta["day_i"]) >= 1:
            day1_start = int(meta["step"])
            break
        obs, *_ = env.step(0)
        guard += 1
        if guard > 10000:
            raise RuntimeError("warmup guard exceeded while seeking day_i>=1")

    # 2) 再推进到 window 完全落在 day1（避免 window 回看 day0）
    target = day1_start + (win - 1)
    while True:
        snap = env.get_oracle_snapshot()
        cur = int(snap["meta"]["step"])
        if cur >= target:
            break
        obs, *_ = env.step(0)

    return obs

def extract_agent_snapshot_from_env(env) -> AgentSnapshot:
    """
    ✅ Integration 只允许“取数”，不允许在这里算特征。
    统一通过 env.get_oracle_snapshot() 获取 stable schema。
    """
    if not hasattr(env, "get_oracle_snapshot"):
        raise RuntimeError("env.get_oracle_snapshot() missing; please implement it for integration tests")

    snap = env.get_oracle_snapshot()
    acc = snap["accounting"]
    ain = snap["agent_input"]

    pos_block = snap.get("positions", {})
    long_any = pos_block.get("long", pos_block.get("long_positions", []))
    short_any = pos_block.get("short", pos_block.get("short_positions", []))

    long_pos = [_pos_from_any(p) for p in long_any]
    short_pos = [_pos_from_any(p) for p in short_any]

    return AgentSnapshot(
        long_positions=long_pos,
        short_positions=short_pos,
        current_step=int(ain["current_step"]),
        current_price=_to_decimal(ain["current_price"]),
        lot_size=_to_decimal(ain["lot_size"]),
        realized_pnl_step=_to_decimal(acc["realized_pnl_step"]),
        realized_pnl_cum=_to_decimal(acc["realized_pnl_cum"]),
        fee_step=_to_decimal(acc["fee_step"]),
        fee_cum=_to_decimal(acc["fee_cum"]),
        cash_balance=_to_decimal(acc["cash_balance"]),
        used_margin=_to_decimal(acc["used_margin"]),
        prev_max_equity=_to_decimal(ain["prev_max_equity"]),
        entries_used_today=int(ain["entries_used_today"]),
        max_entries_per_day=int(ain["max_entries_per_day"]),
        minutes_to_eod=int(ain["minutes_to_eod"]),
        day_len=int(ain["day_len"]),
        initial_balance=_to_decimal(acc["initial_balance"]),
        realized_today_cash=_to_decimal(ain["realized_today_cash"]),
        R_cash=_to_decimal(ain["R_cash"]),
        market_open=int(ain["market_open"]),
        can_long_open=int(ain["can_long_open"]),
        can_short_open=int(ain["can_short_open"]),
        can_long_close=int(ain["can_long_close"]),
        can_short_close=int(ain["can_short_close"]),
        action_result_code=int(ain["action_result_code"]),
        action_result_max_code=_max_code(),
    )


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["raw", "obs"])
def test_env_obs_match_spec(mode):
    # 1) 固定数据
    bundle = MarketDataFactory.make_futures_bundle(num_days=9)
    df = bundle.df_1m

    # 2) Spec market full table（oracle）
    df_market_full = build_market_features_spec(df, tz=bundle.tz, df_prev_session=bundle.df_prev_session)

    # 3) 跑真实 env
    env = CustomTradingEnv(df=df, config_path="tests/test.yaml")

    # 强制切 mode（兜底）
    if hasattr(env, "config") and hasattr(env.config, "trading"):
        setattr(env.config.trading, "obs_feature_mode", mode)

    _allow_cross_day_episode_for_test(env)

    # 用 seed 固定起点，避免随机起点导致 flaky
    obs, _ = env.reset(seed=123)
    obs = _warmup_skip_first_day_and_flush_window(env, obs)

    market_cols = FEATURES_MARKET_OBS if mode == "obs" else FEATURES_MARKET
    agent_cols = FEATURES_AGENT_OBS if mode == "obs" else FEATURES_AGENT

    steps_to_check = 30

    for step_k in range(steps_to_check):
        snap0 = env.get_oracle_snapshot()
        cur_i = int(snap0["meta"]["step"]) if "meta" in snap0 else int(getattr(env, "current_step"))

        got_mkt = obs["market_seq"]
        got_agent = obs["agent_state"]

        exp_mkt = window_slice_spec_by_cols(
            df_market_full,
            row_i=cur_i,
            window_size=int(getattr(env, "window_size")),
            cols=market_cols,
        )

        # agent oracle：只取数 → spec 算特征
        snap = extract_agent_snapshot_from_env(env)
        raw = compute_agent_features_raw_spec(snap)
        obs_feat = compute_agent_features_obs_spec(snap, raw)

        exp_agent = (
            to_float32_vector(obs_feat, FEATURES_AGENT_OBS)
            if mode == "obs"
            else to_float32_vector(raw, FEATURES_AGENT)
        )

        assert_array_allclose_with_diff(
            got_mkt,
            exp_mkt,
            row_labels=list(range(got_mkt.shape[0])),
            col_labels=market_cols,
            prefix=f"[mode={mode} step={step_k} cur_i={cur_i} market_seq]",
            rtol=1e-6,
            atol=1e-6,
        )
        assert_array_allclose_with_diff(
            got_agent.reshape(1, -1),
            exp_agent.reshape(1, -1),
            col_labels=agent_cols,
            prefix=f"[mode={mode} step={step_k} cur_i={cur_i} agent_state]",
            rtol=1e-6,
            atol=1e-6,
        )

        # action：这里用 HOLD
        obs, *_ = env.step(0)


@pytest.mark.integration
def test_market_not_affected_by_action():
    bundle = MarketDataFactory.make_futures_bundle(num_days=9)
    df = bundle.df_1m

    env_hold = CustomTradingEnv(df=df, config_path="tests/test.yaml")
    env_act = CustomTradingEnv(df=df, config_path="tests/test.yaml")

    # 固定 seed，确保两边起点一致（否则 market_seq 不可比）
    obs_h, _ = env_hold.reset(seed=777)
    obs_a, _ = env_act.reset(seed=777)

    # 一个简单脚本：开仓->持有->平仓（需要你们 env.valid_actions 保持一致 index）
    script = [Action.LONG_OPEN0, Action.HOLD, Action.HOLD, Action.LONG_CLOSE0, Action.HOLD, Action.HOLD]  # LONG_OPEN0, HOLD..., LONG_CLOSE0...

    for k, a in enumerate(script):
        # 行情必须完全一致
        assert np.allclose(obs_h["market_seq"], obs_a["market_seq"], rtol=0, atol=0), f"market changed at step {k}"

        # agent_state 可以不同（且应当不同），但至少 shape 一致
        assert obs_h["agent_state"].shape == obs_a["agent_state"].shape

        obs_h, *_ = env_hold.step(Action.HOLD)  # HOLD
        obs_a, *_ = env_act.step(a)
