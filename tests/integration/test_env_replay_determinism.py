# tests/integration/test_env_replay_determinism.py
import numpy as np
from pathlib import Path

import pytest

from gym_trading_env.envs.trading_env import CustomTradingEnv
from gym_trading_env.envs.action import Action, JsonlActionLogger
from gym_trading_env.utils.trade_util import step_wrapper

pytestmark = pytest.mark.integration


def _is_db_unavailable(exc: BaseException) -> bool:
    text = str(exc)
    if "Can't connect to MySQL server" in text:
        return True
    original = getattr(exc, "original_exception", None)
    if isinstance(original, (OSError, PermissionError)):
        return True
    return False


def _make_env(config_path: Path) -> CustomTradingEnv:
    """
    构造 env，数据源走 juejin（由 config 决定），不注入 df。
    """
    try:
        env = CustomTradingEnv(df=None, config_path=str(config_path))
    except Exception as exc:
        if _is_db_unavailable(exc):
            pytest.skip(f"External MySQL data source is unavailable: {exc}")
        raise

    # 测试里不要随机起点，方便比较
    if hasattr(env.config.training, "randomize_start"):
        env.config.training.randomize_start = False

    return env


def _to_py_scalar(x):
    if isinstance(x, np.generic):
        return x.item()
    return x


def _normalize_info(info: dict) -> dict:
    """
    info 里有 np.scalar/0d ndarray，统一转成 python 标量便于比较。
    """
    out = {}
    for k, v in info.items():
        if isinstance(v, np.ndarray):
            if v.shape == ():
                out[k] = _to_py_scalar(v.reshape(()).item())
            else:
                out[k] = v.copy()
        else:
            out[k] = _to_py_scalar(v)
    return out


def _array_diff_debug(a: np.ndarray, b: np.ndarray, atol: float = 1e-6, max_show: int = 10) -> str:
    """
    返回一个可读的 diff 字符串，用于 assertion message。
    """
    if a.shape != b.shape:
        return f"shape mismatch: {a.shape} != {b.shape}"

    # allclose mask（支持 NaN==NaN）
    close = np.isclose(a, b, atol=atol, rtol=0.0, equal_nan=True)
    if bool(np.all(close)):
        return "arrays are allclose (unexpected debug call)"

    # 计算 diff（NaN 位置置 0，避免 nanmax）
    diff = np.abs(np.nan_to_num(a, nan=0.0) - np.nan_to_num(b, nan=0.0))
    max_abs = float(np.max(diff))

    # 找一个代表性的最大差异位置（diff argmax）
    flat_idx = int(np.argmax(diff))
    idx = np.unravel_index(flat_idx, diff.shape)
    va = a[idx]
    vb = b[idx]

    # mismatch 点列表
    mis = np.argwhere(~close)
    n_mis = int(mis.shape[0])
    show = mis[: min(max_show, n_mis)]

    lines = []
    lines.append(f"max_abs_diff={max_abs} at idx={tuple(int(x) for x in idx)}")
    lines.append(f"value_a={va} value_b={vb}")

    # 如果是 2D，额外打印该 row 的整行（特别适合 market_seq）
    if a.ndim == 2:
        r = int(idx[0])
        lines.append(f"row_a[{r}]= {a[r, :]}")
        lines.append(f"row_b[{r}]= {b[r, :]}")

    lines.append(f"mismatch_count={n_mis}, showing first {len(show)} mismatches:")
    for it in show:
        it = tuple(int(x) for x in it)
        lines.append(f"  idx={it} a={a[it]} b={b[it]} diff={diff[it]}")

    return "\n".join(lines)


def _assert_obs_equal(obs_a: dict, obs_b: dict, atol: float = 1e-6):
    assert isinstance(obs_a, dict) and isinstance(obs_b, dict)
    assert set(obs_a.keys()) == set(obs_b.keys()), f"obs keys mismatch: {set(obs_a.keys())} != {set(obs_b.keys())}"

    for k in obs_a.keys():
        va = obs_a[k]
        vb = obs_b[k]

        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            assert isinstance(va, np.ndarray) and isinstance(vb, np.ndarray), f"obs[{k}] type mismatch: {type(va)} != {type(vb)}"
            if not np.allclose(va, vb, atol=atol, rtol=0.0, equal_nan=True):
                dbg = _array_diff_debug(va, vb, atol=atol)
                raise AssertionError(f"obs[{k}] values mismatch\n{dbg}")
        else:
            if va != vb:
                raise AssertionError(f"obs[{k}] mismatch: {va} != {vb}")


def _assert_info_equal(info_a: dict, info_b: dict, atol: float = 1e-6):
    ia = _normalize_info(info_a)
    ib = _normalize_info(info_b)

    if set(ia.keys()) != set(ib.keys()):
        only_a = sorted(set(ia.keys()) - set(ib.keys()))
        only_b = sorted(set(ib.keys()) - set(ia.keys()))
        raise AssertionError(f"info keys mismatch. only_in_a={only_a} only_in_b={only_b}")

    for k in ia.keys():
        va = ia[k]
        vb = ib[k]

        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            assert isinstance(va, np.ndarray) and isinstance(vb, np.ndarray), f"info[{k}] type mismatch: {type(va)} != {type(vb)}"
            if not np.allclose(va, vb, atol=atol, rtol=0.0, equal_nan=True):
                dbg = _array_diff_debug(va, vb, atol=atol)
                raise AssertionError(f"info[{k}] values mismatch\n{dbg}")
        else:
            # float 类字段允许一点点误差（log/env/* 是 float32）
            if isinstance(va, float) or isinstance(vb, float):
                if abs(float(va) - float(vb)) > atol:
                    raise AssertionError(f"info[{k}] mismatch: {va} != {vb}")
            else:
                if va != vb:
                    raise AssertionError(f"info[{k}] mismatch: {va} != {vb}")


def test_live_replay_restores_oracle_snapshot(tmp_path: Path):
    """
    集成测试：验证 live 模式下的 action JSONL + reset replay 能把 env 状态恢复一致。

    流程：
      1) env_a：live 模式运行一段固定 Action 序列，依赖 env 内部写 JSONL。
      2) env_b：reset 时触发 _replay_from_action_log()，恢复到 env_a 最终状态。
      3) 比较：最终 obs/info/snapshot 必须一致。
         注意：env_b.reset() 返回的 obs/info 是 replay 完成后的最终态，
               应当与 env_a 跑完动作后的最终态一致。
    """

    cfg_path = Path("tests/test_replay.yaml")
    if not cfg_path.exists():
        pytest.skip(f"Missing integration config: {cfg_path}")

    # --------- env_a：真实跑一遍，写 action JSONL ---------
    env_a = _make_env(cfg_path)

    # 确保 live 模式打开（否则不会写 action log）
    env_a._live_mode = True

    # 用一个单独的目录做 action JSONL 存储，避免和真实路径串起来
    action_logger = JsonlActionLogger(base_dir=tmp_path, logger=env_a.logger)
    env_a.set_action_logger(action_logger)

    # reset（此时 JSONL 为空，不会 replay）
    env_a.reset(seed=123)

    # 用 Action 枚举（不直接写 int），通过 step_wrapper 调 env.step()
    base_actions = [
        Action.HOLD,
        Action.LONG_OPEN0,
        Action.HOLD,
        Action.HOLD,
        Action.LONG_CLOSE0,
        Action.HOLD,
        Action.SHORT_OPEN0,
        Action.HOLD,
        Action.SHORT_CLOSE0,
        Action.HOLD,
    ]
    max_steps = 50

    for k in range(max_steps):
        if env_a.terminated or env_a.truncated:
            break
        act = base_actions[k % len(base_actions)]
        _, _, term_a, trunc_a, _ = step_wrapper(env_a, act)
        if term_a or trunc_a:
            break

    # env_a 最终态（我们要恢复到这里）
    obs_a_final = env_a._get_obs()
    info_a_final = env_a._get_info()
    snap_a = env_a.get_oracle_snapshot()

    # --------- env_b：只 reset（内部 replay），不手动 step ---------
    env_b = _make_env(cfg_path)

    # 使用同一个 JsonlActionLogger（同一份 JSONL 文件）
    env_b.set_action_logger(action_logger)

    # 确保 reset 会走 replay 分支
    env_b._live_mode = True
    env_b._live_replay_on_reset = True

    # reset 返回时，env_b 已经 replay 完成 -> 应等于 env_a_final
    obs_b_reset, info_b_reset = env_b.reset(seed=123)

    # 确认 action log 里确实有记录
    td = int(env_a.bar_source.store.row_trading_day[int(env_a.current_step)])
    actions = list(action_logger.load_for_day(symbol=env_a.config.trading.future_symbol, trading_day=td))
    assert len(actions) > 0

    # 确认 env_b 起点就是第一条 action 的 step
    first_step = min(int(r["step"]) for r in actions)
    assert env_b.start_idx == first_step

    # 确认 env_b 最终 step == env_a 最终 step
    assert int(env_b.current_step) == int(env_a.current_step)

    # 额外再取一次最终态（理论上应与 reset 返回一致）
    obs_b_final = env_b._get_obs()
    info_b_final = env_b._get_info()
    snap_b = env_b.get_oracle_snapshot()

    # --------- 断言：obs/info 必须一致 ---------
    _assert_obs_equal(obs_a_final, obs_b_reset, atol=1e-6)
    _assert_info_equal(info_a_final, info_b_reset, atol=1e-6)

    _assert_obs_equal(obs_a_final, obs_b_final, atol=1e-6)
    _assert_info_equal(info_a_final, info_b_final, atol=1e-6)

    # --------- 断言：oracle snapshot 必须一致 ---------

    # 1) meta：步数 / 起止边界 / day_i 等必须一致
    assert snap_a["meta"]["step"] == snap_b["meta"]["step"]
    assert snap_a["meta"]["day_i"] == snap_b["meta"]["day_i"]
    assert snap_a["meta"]["start_idx"] == snap_b["meta"]["start_idx"]
    assert snap_a["meta"]["end_idx"] == snap_b["meta"]["end_idx"]
    assert snap_a["meta"]["day_len"] == snap_b["meta"]["day_len"]

    # 2) positions：long / short 列表的 dict 必须完全一样
    assert snap_a["positions"] == snap_b["positions"]

    # 3) accounting：现金、保证金、realized/unrealized pnl、fee 等必须一致
    for key in [
        "cash_balance",
        "used_margin",
        "realized_pnl_step",
        "realized_pnl_cum",
        "unrealized_pnl",
        "fee_step",
        "fee_cum",
        "initial_balance",
    ]:
        va = snap_a["accounting"][key]
        vb = snap_b["accounting"][key]
        assert va == vb, f"accounting[{key}] mismatch: {va} != {vb}"

    # 4) agent_input：核心 gate / entries / 时间 等必须一致
    ai_a = snap_a["agent_input"]
    ai_b = snap_b["agent_input"]

    scalar_keys = [
        "current_step",
        "entries_used_today",
        "max_entries_per_day",
        "minutes_to_eod",
        "day_len",
        "market_open",
        "can_long_open",
        "can_short_open",
        "can_long_close",
        "can_short_close",
        "action_result_code",
    ]
    for k in scalar_keys:
        assert ai_a[k] == ai_b[k], f"agent_input[{k}] mismatch: {ai_a[k]} != {ai_b[k]}"

    # current_price / R_cash / realized_today_cash 也必须一致
    for k in ["current_price", "R_cash", "realized_today_cash"]:
        assert ai_a[k] == ai_b[k], f"agent_input[{k}] mismatch: {ai_a[k]} != {ai_b[k]}"

    # 5) market_seq：shape 相同，数值 allclose
    seq_a = snap_a["market"]["market_seq"]
    seq_b = snap_b["market"]["market_seq"]

    assert isinstance(seq_a, np.ndarray)
    assert isinstance(seq_b, np.ndarray)
    assert seq_a.shape == seq_b.shape
    if not np.allclose(seq_a, seq_b, atol=1e-6, rtol=0.0, equal_nan=True):
        dbg = _array_diff_debug(seq_a, seq_b, atol=1e-6)
        raise AssertionError(f"snapshot market_seq mismatch\n{dbg}")
