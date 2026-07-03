#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for item in (ROOT, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from tools.dreamer_checkpoint_audit import (  # noqa: E402
    _batch_obs,
    _prepare_candidates,
    _scalar,
    _select_start_rows,
    _unwrap_base_env,
    _zero_action,
)


DATE = "20260703"
OUT = ROOT / "artifacts" / "jm3_pnl_aligned_no_trade_root_cause"
DOC = ROOT / "docs" / f"jm3_pnl_aligned_no_trade_root_cause_{DATE}.md"

DREAMER_ROOT = Path("/home/v/Documents/work/dreamerv3")
OLD_RUN = Path("/data/logdir/action-mask-wf1-ms-seed-c-20260615")
OLD_LIVE_RUN = Path("/data/logdir/future_jm3_live_seed_c_700k_20260618")
OLD_ENV = OLD_RUN / "env_env_trading_stage1_jm_walk_forward_train_20240603_20250731.yaml"
OLD_LIVE_ENV = OLD_LIVE_RUN / "env_jm_live_jm2609_v1_20260622_replay.yaml"
OLD_CHECKPOINT = OLD_RUN / "ckpt_retained" / "step_000000700000"
OLD_BASELINE_FULL_EVAL_ENV = OUT / "old_baseline_full_eval_old_reward_env.yaml"

NEW_RUN = Path("/data/logdir/future_jm3_pnl_aligned_diag_seed_c_20260702")
NEW_TRAIN_ENV = ROOT / "configs" / "jm3_pnl_aligned_diag_seed_a_20260702.yaml"
NEW_FULL_ENV = ROOT / "configs" / "jm3_pnl_aligned_diag_full_eval_20260702.yaml"
ENTRY_EVAL_DIR = Path(
    "/data/logdir/trading_contracts/jm_walk_forward_20240603_20251202"
    "/artifacts/entry_eval/jm_walk_forward_20240603_20251202"
)

CHECKPOINTS = [
    ("300k", NEW_RUN / "ckpt_retained" / "step_000000300000"),
    ("500k", NEW_RUN / "ckpt_retained" / "step_000000500000"),
    ("700k", NEW_RUN / "ckpt_retained" / "step_000000700000"),
]

ACTION_NAMES = {0: "SHORT", 1: "FLAT", 2: "LONG"}


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _safe_float(value, default=0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _safe_int(value, default=0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _load_yaml(path: Path) -> dict:
    import ruamel.yaml as yaml

    if not path.exists():
        return {}
    return yaml.YAML(typ="safe").load(path.read_text()) or {}


def _write_yaml(path: Path, data: dict) -> None:
    import ruamel.yaml as yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = yaml.YAML()
    writer.default_flow_style = False
    with path.open("w") as f:
        writer.dump(data, f)


def _nested(data: dict, path: str, default=None):
    cur = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _stringify(value) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def _feature_counts() -> dict:
    def const_list(path: Path, name: str) -> list[str]:
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == name and isinstance(node.value, ast.List):
                    return [x.value for x in node.value.elts if isinstance(x, ast.Constant)]
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == name and isinstance(node.value, ast.List):
                        return [x.value for x in node.value.elts if isinstance(x, ast.Constant)]
        raise KeyError(f"Cannot find {name} in {path}")

    market_path = SRC / "gym_trading_env" / "utils" / "market_features.py"
    agent_path = SRC / "gym_trading_env" / "utils" / "agent_features.py"
    FEATURES_MARKET_OBS_OUTPUT = const_list(market_path, "FEATURES_MARKET_OBS_OUTPUT")
    FEATURES_RISK_CONTEXT_OUTPUT = const_list(market_path, "FEATURES_RISK_CONTEXT_OUTPUT")
    FEATURES_HTF_CONTEXT_OUTPUT = const_list(market_path, "FEATURES_HTF_CONTEXT_OUTPUT")
    FEATURES_AGENT_OBS = const_list(agent_path, "FEATURES_AGENT_OBS")

    return {
        "market_seq_features_current_code": len(FEATURES_MARKET_OBS_OUTPUT),
        "risk_context_features_current_code": len(FEATURES_RISK_CONTEXT_OUTPUT),
        "htf_context_features_current_code": len(FEATURES_HTF_CONTEXT_OUTPUT),
        "agent_state_features_current_code": len(FEATURES_AGENT_OBS),
        "market_seq_feature_names_current_code": FEATURES_MARKET_OBS_OUTPUT,
        "risk_context_feature_names_current_code": FEATURES_RISK_CONTEXT_OUTPUT,
        "htf_context_feature_names_current_code": FEATURES_HTF_CONTEXT_OUTPUT,
        "agent_state_feature_names_current_code": FEATURES_AGENT_OBS,
    }


def _classify_config_diff(field: str, old_value, new_value) -> str:
    if old_value == new_value:
        return "SAME"
    if field == "training.reward_function":
        return "EXPECTED_REWARD_ONLY"
    if field == "trading.reward_scale_cash":
        return "EXPECTED_REWARD_ONLY"
    if field in {"trading.live_mode", "trading.live_replay_on_reset"}:
        return "EXPECTED_LIVE_SAFETY"
    if field in {"logdir", "run.checkpoint_retention.steps"}:
        return "EXPECTED_LOGDIR"
    if field in {"run.steps"}:
        return "EXPECTED_LOGDIR"
    if field in {
        "trading.data_path",
        "audit.split_manifest_hash",
        "audit.entry_eval_version",
        "audit.entry_eval_config",
        "env.gymnasium.config_path",
    }:
        return "POTENTIALLY_BEHAVIOR_CHANGING"
    return "UNKNOWN"


def build_config_diff_matrix() -> list[dict]:
    old_env = _load_yaml(OLD_ENV)
    old_live_env = _load_yaml(OLD_LIVE_ENV)
    old_run = _load_yaml(OLD_RUN / "config.yaml")
    new_env = _load_yaml(NEW_TRAIN_ENV)
    new_run = _load_yaml(NEW_RUN / "config.yaml")
    features = _feature_counts()

    rows: list[dict] = []
    env_fields = [
        "trading.data_path",
        "training.bar_source",
        "trading.future_symbol",
        "trading.obs_feature_mode",
        "trading.trading_fee_per_lot",
        "trading.spread",
        "trading.lot_size",
        "trading.trade_lot",
        "trading.max_long_position",
        "trading.max_short_position",
        "trading.stop_loss_enabled",
        "trading.take_profit_enabled",
        "trading.session_policy",
        "training.reward_function",
        "trading.reward_scale_cash",
        "trading.live_mode",
        "trading.live_replay_on_reset",
        "training.window_size",
        "training.max_episode_steps",
        "training.start_clock",
        "training.episode_policy",
    ]
    for field in env_fields:
        old_value = _nested(old_env, field)
        new_value = _nested(new_env, field)
        rows.append({
            "scope": "env_train_old_vs_new",
            "field": field,
            "old_value": _stringify(old_value),
            "new_value": _stringify(new_value),
            "classification": _classify_config_diff(field, old_value, new_value),
            "notes": "old training env backup vs new PnL training env",
        })

    live_fields = [
        "trading.data_path",
        "training.bar_source",
        "trading.future_symbol",
        "trading.live_mode",
        "trading.live_replay_on_reset",
        "training.reward_function",
    ]
    for field in live_fields:
        rows.append({
            "scope": "old_live_config_reference",
            "field": field,
            "old_value": _stringify(_nested(old_live_env, field)),
            "new_value": _stringify(_nested(new_env, field)),
            "classification": "UNKNOWN" if field not in {"trading.live_mode", "trading.live_replay_on_reset", "training.reward_function"} else _classify_config_diff(field, _nested(old_live_env, field), _nested(new_env, field)),
            "notes": "old live config is a 2026-06-22 live replay reference, not the training config",
        })

    run_fields = [
        "batch_size",
        "batch_length",
        "replay_context",
        "run.envs",
        "run.eval_envs",
        "run.train_ratio",
        "run.steps",
        "run.checkpoint_retention.steps",
        "seed",
        "dreamer.seed",
        "env.train_seed",
        "env.eval_seed",
        "replay.seed",
        "agent.policy.unimix",
        "agent.imag_loss.actent",
        "agent.avail_threshold",
        "agent.avail_actor_enabled",
        "env.gymnasium.config_path",
        "audit.entry_eval_version",
        "audit.entry_eval_config",
        "audit.split_manifest_hash",
    ]
    for field in run_fields:
        old_value = _nested(old_run, field)
        new_value = _nested(new_run, field)
        rows.append({
            "scope": "dreamer_run_old_vs_new",
            "field": field,
            "old_value": _stringify(old_value),
            "new_value": _stringify(new_value),
            "classification": _classify_config_diff(field, old_value, new_value),
            "notes": "old seed-c run config vs new PnL run config",
        })

    for key, value in features.items():
        rows.append({
            "scope": "current_code_obs_runtime",
            "field": key,
            "old_value": "not recorded in old config",
            "new_value": _stringify(value),
            "classification": "UNKNOWN",
            "notes": "runtime feature count is determined by current code, not YAML",
        })
    return rows


def _patch_policy_debug() -> None:
    sys.path.insert(0, str(DREAMER_ROOT))
    import dreamerv3.agent as agent_mod

    if getattr(agent_mod.Agent, "_codex_policy_debug_patch", False):
        return

    def policy_debug(self, carry, obs, mode="train"):
        enc_carry, dyn_carry, dec_carry, prevact = carry
        kw = dict(training=False, single=True)
        reset = obs["is_first"]
        enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
        dyn_carry, dyn_entry, feat = self.dyn.observe(
            dyn_carry, tokens, prevact, reset, **kw)
        dec_entry = {}
        if dec_carry:
            dec_carry, dec_entry, _ = self.dec(dec_carry, feat, reset, **kw)
        policy = self._masked_policy(
            self.pol(self.feat2tensor(feat), bdims=1),
            obs["action_mask"],
            allow_fallback=False,
        )
        dist = policy[self.action_key]
        logits = dist.logits
        probs = agent_mod.jax.nn.softmax(logits, -1)
        logprob = agent_mod.jax.nn.log_softmax(logits, -1)
        entropy = -(probs * logprob).sum(-1)
        act = (
            agent_mod.jax.tree.map(lambda x: x.pred(), policy)
            if mode == "eval" else agent_mod.sample(policy)
        )
        out = {
            "action_logits": logits,
            "action_probs": probs,
            "action_entropy": entropy,
        }
        out["finite"] = agent_mod.elements.tree.flatdict(agent_mod.jax.tree.map(
            lambda x: agent_mod.jnp.isfinite(x).all(range(1, x.ndim)),
            dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act,
                 action_logits=logits, action_probs=probs)))
        carry = (enc_carry, dyn_carry, dec_carry, act)
        if self.config.replay_context:
            out.update(agent_mod.elements.tree.flatdict(dict(
                enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
        return carry, act, out

    agent_mod.Agent.policy = policy_debug
    agent_mod.Agent._codex_policy_debug_patch = True


def _dreamer_jax_platform_value(jax_platform: str) -> str:
    if jax_platform == "auto":
        return ""
    return jax_platform


def _assert_gpu_backend(context: str) -> None:
    import jax

    backend = jax.default_backend()
    devices = jax.devices()
    gpu_devices = [dev for dev in devices if getattr(dev, "platform", "") == "gpu"]
    if backend != "gpu" or not gpu_devices:
        raise RuntimeError(
            f"{context}: expected JAX GPU backend, got backend={backend!r}, "
            f"devices={[str(dev) for dev in devices]}"
        )
    print(f"{context}: JAX backend={backend}, devices={[str(dev) for dev in gpu_devices]}")


def _load_config(run_logdir: Path, env_config: Path, runtime_dir: Path, *, jax_platform: str = "auto"):
    sys.path.insert(0, str(DREAMER_ROOT))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("RPC_ORDER_ENABLED", "false")
    import elements
    import ruamel.yaml as yaml

    raw = yaml.YAML(typ="safe").load((run_logdir / "config.yaml").read_text())
    raw.setdefault("jax", {})["platform"] = _dreamer_jax_platform_value(jax_platform)
    raw.setdefault("jax", {})["prealloc"] = False
    raw.setdefault("jax", {})["precompile"] = False
    config = elements.Config(raw).update({
        "env.gymnasium.config_path": str(env_config),
        "logdir": str(runtime_dir),
        "run.debug": True,
        "run.envs": 1,
        "jax.platform": _dreamer_jax_platform_value(jax_platform),
        "jax.prealloc": False,
    })
    os.environ["DREAMER_RUN_DIR"] = str(runtime_dir)
    return config


def _make_env(run_logdir: Path, env_config: Path, runtime_dir: Path, *, jax_platform: str):
    sys.path.insert(0, str(DREAMER_ROOT))
    from dreamerv3 import main as dreamer_main

    config = _load_config(run_logdir, env_config, runtime_dir, jax_platform=jax_platform)
    env = dreamer_main.make_env(config, 0)
    return config, env


def _load_agent(
    run_logdir: Path,
    env_config: Path,
    checkpoint: Path,
    runtime_dir: Path,
    *,
    jax_platform: str,
):
    sys.path.insert(0, str(DREAMER_ROOT))
    from functools import partial as bind

    import elements.checkpoint as checkpoint_mod
    from dreamerv3 import main as dreamer_main

    _patch_policy_debug()
    config = _load_config(run_logdir, env_config, runtime_dir, jax_platform=jax_platform)
    _assert_gpu_backend(f"policy_audit_preload:{checkpoint.name}")
    agent = dreamer_main.make_agent(config)
    _assert_gpu_backend(f"policy_audit:{checkpoint.name}")
    checkpoint_mod.load(checkpoint, {"agent": bind(agent.load, regex=None)})
    return config, agent


def _role_days(role_by_day: dict[int, str], role: str, limit: int | None = None) -> set[int]:
    days = [int(day) for day, r in sorted(role_by_day.items()) if r == role]
    if limit:
        days = days[:limit]
    return set(days)


def _forced_rows(base_env, role_by_day, roles: set[str], *, start_clock: str, limit: int | None):
    rows = _select_start_rows(
        base_env,
        role_by_day=role_by_day,
        roles=roles,
        start_clock=start_clock,
        episode_cap=int(limit or 0),
    )
    if limit:
        rows = rows[:limit]
    return [int(x) for x in rows]


def _position_name(base_env) -> str:
    try:
        return base_env._current_target().name
    except Exception:
        return "UNKNOWN"


def _timestamp(base_env) -> str:
    try:
        row = int(getattr(base_env, "current_step", 0))
        return str(base_env.bar_source.store.index[row])
    except Exception:
        return ""


def scan_action_mask_availability(role_by_day: dict[int, str], *, jax_platform: str) -> tuple[list[dict], list[dict]]:
    runtime_dir = OUT / "runtime_mask_scan"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    os.chdir(DREAMER_ROOT)
    try:
        _, env = _make_env(NEW_RUN, NEW_FULL_ENV, runtime_dir, jax_platform=jax_platform)
        base_env = _unwrap_base_env(env)
        original_candidate_start_rows = base_env._candidate_start_rows_by_clock
        forced_start = {"row": None}

        def override(clock):
            if forced_start["row"] is None:
                return original_candidate_start_rows(clock)
            return np.asarray([int(forced_start["row"])], dtype=np.int64)

        base_env._candidate_start_rows_by_clock = override
        start_clock = str(getattr(base_env.config.training, "start_clock", "future_night"))
        start_rows = _forced_rows(
            base_env, role_by_day, {"validation"}, start_clock=start_clock, limit=None)
        day_rows = []
        state_rows = []
        for ep_idx, start_row in enumerate(start_rows):
            forced_start["row"] = int(start_row)
            action = _zero_action(env.act_space)
            done = False
            day_counts = defaultdict(float)
            while not done:
                obs = env.step(action)
                mask = np.asarray(obs.get("action_mask"), dtype=float).reshape(-1)
                is_last = bool(_scalar(obs.get("is_last"), False))
                if not is_last:
                    pos = _position_name(base_env)
                    day = int(base_env.bar_source.store.row_trading_day[int(getattr(base_env, "current_step", 0))])
                    row = {
                        "episode_index": ep_idx,
                        "trading_day": day,
                        "timestamp": _timestamp(base_env),
                        "position_side": pos,
                        "short_available": float(mask[0]),
                        "flat_available": float(mask[1]),
                        "long_available": float(mask[2]),
                    }
                    state_rows.append(row)
                    prefix = f"pos_{pos.lower()}"
                    day_counts["steps"] += 1
                    day_counts["short_available_steps"] += float(mask[0] > 0.5)
                    day_counts["flat_available_steps"] += float(mask[1] > 0.5)
                    day_counts["long_available_steps"] += float(mask[2] > 0.5)
                    day_counts[f"{prefix}_steps"] += 1
                    day_counts[f"{prefix}_short_available_steps"] += float(mask[0] > 0.5)
                    day_counts[f"{prefix}_flat_available_steps"] += float(mask[1] > 0.5)
                    day_counts[f"{prefix}_long_available_steps"] += float(mask[2] > 0.5)
                if is_last:
                    done = True
                else:
                    action = {"action": np.asarray(1, dtype=np.int32), "reset": np.asarray(False)}
            trading_day = int(base_env.bar_source.store.row_trading_day[int(start_row)])
            steps = max(1.0, day_counts["steps"])
            flat_steps = max(1.0, day_counts.get("pos_flat_steps", 0.0))
            day_rows.append({
                "trading_day": trading_day,
                "steps": int(day_counts["steps"]),
                "short_available_steps": int(day_counts["short_available_steps"]),
                "flat_available_steps": int(day_counts["flat_available_steps"]),
                "long_available_steps": int(day_counts["long_available_steps"]),
                "short_available_ratio": day_counts["short_available_steps"] / steps,
                "long_available_ratio": day_counts["long_available_steps"] / steps,
                "flat_position_steps": int(day_counts.get("pos_flat_steps", 0)),
                "flat_position_short_available_steps": int(day_counts.get("pos_flat_short_available_steps", 0)),
                "flat_position_long_available_steps": int(day_counts.get("pos_flat_long_available_steps", 0)),
                "flat_position_short_available_ratio": day_counts.get("pos_flat_short_available_steps", 0.0) / flat_steps,
                "flat_position_long_available_ratio": day_counts.get("pos_flat_long_available_steps", 0.0) / flat_steps,
            })
        env.close()
        return day_rows, state_rows
    finally:
        os.chdir(cwd)


def _summarize_probs(rows: list[dict], prefix: str = "") -> dict:
    out = {}
    for action in ("short", "flat", "long"):
        vals = [float(r[f"prob_{action}"]) for r in rows if f"prob_{action}" in r]
        if vals:
            out[f"{prefix}prob_{action}_mean"] = float(np.mean(vals))
            out[f"{prefix}prob_{action}_median"] = float(np.median(vals))
            out[f"{prefix}prob_{action}_p90"] = float(np.percentile(vals, 90))
        else:
            out[f"{prefix}prob_{action}_mean"] = None
            out[f"{prefix}prob_{action}_median"] = None
            out[f"{prefix}prob_{action}_p90"] = None
    ent = [float(r["entropy"]) for r in rows if "entropy" in r]
    out[f"{prefix}entropy_mean"] = float(np.mean(ent)) if ent else None
    out[f"{prefix}entropy_median"] = float(np.median(ent)) if ent else None
    return out


def rollout_policy(
    label: str,
    run_logdir: Path,
    checkpoint: Path,
    env_config: Path,
    role_by_day: dict[int, str],
    *,
    split_role: str,
    mode: str,
    limit_days: int | None,
    jax_platform: str,
) -> tuple[list[dict], list[dict]]:
    runtime_dir = OUT / f"runtime_policy_{label}_{split_role}_{mode}"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    os.chdir(DREAMER_ROOT)
    try:
        config, agent = _load_agent(
            run_logdir, env_config, checkpoint, runtime_dir, jax_platform=jax_platform)
        return rollout_policy_loaded(
            label, config, agent, role_by_day,
            split_role=split_role, mode=mode, limit_days=limit_days)
    finally:
        os.chdir(cwd)


def rollout_policy_loaded(
    label: str,
    config,
    agent,
    role_by_day: dict[int, str],
    *,
    split_role: str,
    mode: str,
    limit_days: int | None,
) -> tuple[list[dict], list[dict]]:
    cwd = Path.cwd()
    os.chdir(DREAMER_ROOT)
    try:
        env = __import__("dreamerv3.main", fromlist=["make_env"]).make_env(config, 0)
        base_env = _unwrap_base_env(env)
        original_candidate_start_rows = base_env._candidate_start_rows_by_clock
        forced_start = {"row": None}

        def override(clock):
            if forced_start["row"] is None:
                return original_candidate_start_rows(clock)
            return np.asarray([int(forced_start["row"])], dtype=np.int64)

        base_env._candidate_start_rows_by_clock = override
        start_clock = str(getattr(base_env.config.training, "start_clock", "future_night"))
        start_rows = _forced_rows(
            base_env, role_by_day, {split_role}, start_clock=start_clock, limit=limit_days)
        carry = agent.init_policy(1)
        step_rows: list[dict] = []
        episode_rows: list[dict] = []
        for ep_idx, start_row in enumerate(start_rows):
            forced_start["row"] = int(start_row)
            action = _zero_action(env.act_space)
            score = 0.0
            length = 0
            done = False
            while not done:
                obs = env.step(action)
                is_first = bool(_scalar(obs.get("is_first"), False))
                is_last = bool(_scalar(obs.get("is_last"), False))
                if not is_first:
                    score += float(_scalar(obs.get("reward"), 0.0) or 0.0)
                    length += 1
                if not is_last:
                    policy_obs = _batch_obs(obs, agent.obs_space)
                    carry, next_action, outs = agent.policy(carry, policy_obs, mode=mode)
                    selected = int(_scalar(next_action.get("action"), -1))
                    probs = np.asarray(outs.get("action_probs"))[0].reshape(-1)
                    logits = np.asarray(outs.get("action_logits"))[0].reshape(-1)
                    entropy = float(np.asarray(outs.get("action_entropy"))[0].reshape(-1)[0])
                    mask = np.asarray(obs.get("action_mask"), dtype=float).reshape(-1)
                    pos = _position_name(base_env)
                    day = int(base_env.bar_source.store.row_trading_day[int(getattr(base_env, "current_step", 0))])
                    row = {
                        "checkpoint_label": label,
                        "split_role": split_role,
                        "mode": mode,
                        "episode_index": ep_idx,
                        "trading_day": day,
                        "timestamp": _timestamp(base_env),
                        "position_side": pos,
                        "selected_action": selected,
                        "selected_action_name": ACTION_NAMES.get(selected, str(selected)),
                        "mask_short": float(mask[0]),
                        "mask_flat": float(mask[1]),
                        "mask_long": float(mask[2]),
                        "prob_short": float(probs[0]),
                        "prob_flat": float(probs[1]),
                        "prob_long": float(probs[2]),
                        "logit_short": float(logits[0]),
                        "logit_flat": float(logits[1]),
                        "logit_long": float(logits[2]),
                        "entropy": entropy,
                        "flat_state_entry_available": bool(pos == "FLAT" and (mask[0] > 0.5 or mask[2] > 0.5)),
                    }
                    step_rows.append(row)
                    action = {k: np.asarray(v[0]) for k, v in next_action.items()}
                    action["reset"] = np.asarray(False)
                else:
                    records = getattr(base_env.trade_record_manager, "trade_history", [])
                    episode_rows.append({
                        "checkpoint_label": label,
                        "split_role": split_role,
                        "mode": mode,
                        "episode_index": ep_idx,
                        "start_row": int(start_row),
                        "start_trading_day": int(base_env.bar_source.store.row_trading_day[int(start_row)]),
                        "length": int(length),
                        "score": float(score),
                        "score_scaled_cash": float(score * 300.0),
                        "return_pct": _safe_float(_scalar(obs.get("log/env/return_pct"), np.nan), default=np.nan),
                        "trades_opened": _safe_float(_scalar(obs.get("log/env/trades_opened"), np.nan), default=0.0),
                        "trades_closed": _safe_float(_scalar(obs.get("log/env/trades_closed"), np.nan), default=0.0),
                        "actual_net_pnl": _safe_float(_scalar(obs.get("log/env/reward_episode/actual_net_pnl"), 0.0), default=0.0),
                        "final_equity": _safe_float(_scalar(obs.get("log/env/reward_episode/final_equity"), np.nan), default=np.nan),
                        "action_rejected": _safe_float(_scalar(obs.get("log/env/action_rejected"), 0.0), default=0.0),
                        "execution_failed": _safe_float(_scalar(obs.get("log/env/execution_failed"), 0.0), default=0.0),
                        "trade_records": len(records),
                    })
                    done = True
        env.close()
        return step_rows, episode_rows
    finally:
        os.chdir(cwd)


def _action_distribution_summary(step_rows: list[dict], episode_rows: list[dict]) -> dict:
    flat_available = [
        r for r in step_rows
        if r.get("position_side") == "FLAT" and (float(r.get("mask_short", 0)) > 0.5 or float(r.get("mask_long", 0)) > 0.5)
    ]
    selected_counts = defaultdict(int)
    for row in step_rows:
        selected_counts[row.get("selected_action_name", "UNKNOWN")] += 1
    opened = sum(_safe_float(row.get("trades_opened")) for row in episode_rows)
    net = sum(_safe_float(row.get("actual_net_pnl")) for row in episode_rows)
    out = {
        "steps": len(step_rows),
        "episodes": len(episode_rows),
        "selected_short_count": selected_counts.get("SHORT", 0),
        "selected_flat_count": selected_counts.get("FLAT", 0),
        "selected_long_count": selected_counts.get("LONG", 0),
        "flat_available_steps": len(flat_available),
        "days_with_nonzero_open_probability": len({
            r["trading_day"] for r in flat_available
            if float(r.get("prob_short", 0)) + float(r.get("prob_long", 0)) > 1e-6
        }),
        "trades_opened": opened,
        "actual_net_pnl": net,
        "no_trade_day_ratio": (
            sum(1 for row in episode_rows if _safe_float(row.get("trades_opened")) == 0)
            / max(1, len(episode_rows))
        ),
    }
    out.update(_summarize_probs(flat_available, prefix="flat_available_"))
    out.update(_summarize_probs(step_rows, prefix="all_steps_"))
    return out


def summarize_policy_outputs(
    det_steps: dict[str, list[dict]],
    det_eps: dict[str, list[dict]],
    stochastic_steps: dict[str, list[dict]],
    stochastic_eps: dict[str, list[dict]],
) -> tuple[list[dict], list[dict], list[dict]]:
    checkpoint_rows = []
    day_rows = []
    flat_prob_rows = []
    for label, rows in det_steps.items():
        summary = _action_distribution_summary(rows, det_eps[label])
        stochastic = _action_distribution_summary(
            stochastic_steps.get(label, []), stochastic_eps.get(label, []))
        checkpoint_rows.append({
            "checkpoint_label": label,
            "eval_steps": summary["steps"],
            "eval_episodes": summary["episodes"],
            "eval_selected_short_count": summary["selected_short_count"],
            "eval_selected_flat_count": summary["selected_flat_count"],
            "eval_selected_long_count": summary["selected_long_count"],
            "eval_trades_opened": summary["trades_opened"],
            "eval_actual_net_pnl": summary["actual_net_pnl"],
            "eval_no_trade_day_ratio": summary["no_trade_day_ratio"],
            "eval_flat_available_steps": summary["flat_available_steps"],
            "eval_flat_available_prob_short_mean": summary["flat_available_prob_short_mean"],
            "eval_flat_available_prob_flat_mean": summary["flat_available_prob_flat_mean"],
            "eval_flat_available_prob_long_mean": summary["flat_available_prob_long_mean"],
            "eval_flat_available_prob_flat_p90": summary["flat_available_prob_flat_p90"],
            "eval_flat_available_entropy_mean": summary["flat_available_entropy_mean"],
            "stochastic_selected_short_count": stochastic["selected_short_count"],
            "stochastic_selected_flat_count": stochastic["selected_flat_count"],
            "stochastic_selected_long_count": stochastic["selected_long_count"],
            "stochastic_trades_opened": stochastic["trades_opened"],
            "stochastic_actual_net_pnl": stochastic["actual_net_pnl"],
            "stochastic_no_trade_day_ratio": stochastic["no_trade_day_ratio"],
        })
        by_day = defaultdict(list)
        for row in rows:
            by_day[int(row["trading_day"])].append(row)
            if row.get("flat_state_entry_available"):
                flat_prob_rows.append({
                    "checkpoint_label": label,
                    "trading_day": row["trading_day"],
                    "timestamp": row["timestamp"],
                    "selected_action_name": row["selected_action_name"],
                    "prob_short": row["prob_short"],
                    "prob_flat": row["prob_flat"],
                    "prob_long": row["prob_long"],
                    "entropy": row["entropy"],
                    "mask_short": row["mask_short"],
                    "mask_flat": row["mask_flat"],
                    "mask_long": row["mask_long"],
                })
        eps_by_day = {int(e["start_trading_day"]): e for e in det_eps[label]}
        for day, part in sorted(by_day.items()):
            dist = _action_distribution_summary(part, [eps_by_day[day]] if day in eps_by_day else [])
            day_rows.append({
                "checkpoint_label": label,
                "trading_day": day,
                "steps": dist["steps"],
                "selected_short_count": dist["selected_short_count"],
                "selected_flat_count": dist["selected_flat_count"],
                "selected_long_count": dist["selected_long_count"],
                "flat_available_steps": dist["flat_available_steps"],
                "flat_available_prob_short_mean": dist["flat_available_prob_short_mean"],
                "flat_available_prob_flat_mean": dist["flat_available_prob_flat_mean"],
                "flat_available_prob_long_mean": dist["flat_available_prob_long_mean"],
                "entropy_mean": dist["all_steps_entropy_mean"],
                "trades_opened": dist["trades_opened"],
                "actual_net_pnl": dist["actual_net_pnl"],
            })
    return checkpoint_rows, day_rows, flat_prob_rows


def training_metrics_summary() -> tuple[list[dict], list[dict], dict]:
    path = NEW_RUN / "metrics.jsonl"
    rows = []
    dist_rows = []
    collapse_step = None
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            step = item.get("step")
            if "epstats/log/env/trades_opened/avg" not in item:
                continue
            opened_avg = _safe_float(item.get("epstats/log/env/trades_opened/avg"))
            opened_sum = _safe_float(item.get("epstats/log/env/trades_opened/sum"))
            trade_count_sum = _safe_float(item.get("epstats/log/env/reward_episode/trade_count/sum"))
            selected_flat_avg = _safe_float(item.get("epstats/log/env/chosen_action_index/avg"))
            entropy = _safe_float(item.get("train/ent/action"))
            actual_sum = _safe_float(item.get("epstats/log/env/reward_episode/actual_net_pnl/sum"))
            reward_sum = _safe_float(item.get("epstats/log/env/reward_episode/reward_total/sum"))
            invariant_error = reward_sum * 300.0 - actual_sum
            row = {
                "step": step,
                "trades_opened_avg": opened_avg,
                "trades_opened_sum": opened_sum,
                "reward_episode_trade_count_sum": trade_count_sum,
                "chosen_action_index_avg": selected_flat_avg,
                "actor_entropy": entropy,
                "actual_net_pnl_sum": actual_sum,
                "reward_total_sum": reward_sum,
                "reward_total_scaled_cash": reward_sum * 300.0,
                "reward_pnl_invariant_error_cash": invariant_error,
                "action_rejected_sum": _safe_float(item.get("epstats/log/env/action_rejected/sum")),
                "execution_failed_sum": _safe_float(item.get("epstats/log/env/execution_failed/sum")),
                "train_loss_rew": _safe_float(item.get("train/loss/rew")),
                "train_loss_policy": _safe_float(item.get("train/loss/policy")),
                "train_loss_value": _safe_float(item.get("train/loss/value")),
                "train_avail_ready": _safe_float(item.get("train/avail/ready")),
                "train_avail_img_action_invalid_rate": _safe_float(item.get("train/avail/img_action_invalid_rate")),
                "train_avail_img_fallback_rate": _safe_float(item.get("train/avail/img_fallback_rate")),
            }
            rows.append(row)
            dist_rows.append({
                "step": step,
                "actual_net_pnl_sum": actual_sum,
                "reward_total_sum": reward_sum,
                "reward_total_scaled_cash": reward_sum * 300.0,
                "reward_pnl_invariant_error_cash": invariant_error,
            })
            if collapse_step is None and opened_avg <= 0.01 and _safe_float(item.get("train/avail/ready")) >= 1.0:
                collapse_step = step
    compact = {
        "metrics_rows_with_epstats": len(rows),
        "first_step": rows[0]["step"] if rows else None,
        "last_step": rows[-1]["step"] if rows else None,
        "last_trades_opened_avg": rows[-1]["trades_opened_avg"] if rows else None,
        "last_reward_episode_trade_count_sum": rows[-1]["reward_episode_trade_count_sum"] if rows else None,
        "last_actor_entropy": rows[-1]["actor_entropy"] if rows else None,
        "collapse_step_by_trades_opened_avg_le_0p01_after_avail_ready": collapse_step,
        "max_abs_reward_pnl_invariant_error_cash": max((abs(r["reward_pnl_invariant_error_cash"]) for r in rows), default=None),
        "max_action_rejected_sum": max((abs(r["action_rejected_sum"]) for r in rows), default=None),
        "max_execution_failed_sum": max((abs(r["execution_failed_sum"]) for r in rows), default=None),
        "max_img_action_invalid_rate": max((abs(r["train_avail_img_action_invalid_rate"]) for r in rows), default=None),
        "max_img_fallback_rate": max((abs(r["train_avail_img_fallback_rate"]) for r in rows), default=None),
    }
    return rows, dist_rows, compact


def summarize_train_vs_validation(
    train_steps: dict[str, list[dict]],
    train_eps: dict[str, list[dict]],
    val_steps: dict[str, list[dict]],
    val_eps: dict[str, list[dict]],
) -> tuple[list[dict], list[dict]]:
    action_rows = []
    pnl_rows = []
    for label in sorted(val_steps):
        for split_name, steps, eps in [
            ("train_subset", train_steps.get(label, []), train_eps.get(label, [])),
            ("validation", val_steps.get(label, []), val_eps.get(label, [])),
        ]:
            dist = _action_distribution_summary(steps, eps)
            action_rows.append({
                "checkpoint_label": label,
                "split_eval": split_name,
                "episodes": dist["episodes"],
                "steps": dist["steps"],
                "selected_short_count": dist["selected_short_count"],
                "selected_flat_count": dist["selected_flat_count"],
                "selected_long_count": dist["selected_long_count"],
                "flat_available_steps": dist["flat_available_steps"],
                "flat_available_prob_short_mean": dist["flat_available_prob_short_mean"],
                "flat_available_prob_flat_mean": dist["flat_available_prob_flat_mean"],
                "flat_available_prob_long_mean": dist["flat_available_prob_long_mean"],
                "entropy_mean": dist["all_steps_entropy_mean"],
                "trades_opened": dist["trades_opened"],
            })
            pnl_rows.append({
                "checkpoint_label": label,
                "split_eval": split_name,
                "episodes": dist["episodes"],
                "actual_net_pnl": dist["actual_net_pnl"],
                "score_scaled_cash": sum(_safe_float(e.get("score_scaled_cash")) for e in eps),
                "no_trade_day_ratio": dist["no_trade_day_ratio"],
                "negative_days": sum(1 for e in eps if _safe_float(e.get("actual_net_pnl")) < 0),
            })
    return action_rows, pnl_rows


def old_baseline_same_protocol(role_by_day: dict[int, str], *, jax_platform: str) -> list[dict]:
    def row_from_dist(dist: dict, *, status: str, env_config: Path, notes: str, primary_episodes: int = 0) -> dict:
        return {
            "checkpoint_label": "old_baseline_700k",
            "status": status,
            "episodes": dist["episodes"],
            "steps": dist["steps"],
            "selected_short_count": dist["selected_short_count"],
            "selected_flat_count": dist["selected_flat_count"],
            "selected_long_count": dist["selected_long_count"],
            "trades_opened": dist["trades_opened"],
            "actual_net_pnl": dist["actual_net_pnl"],
            "no_trade_day_ratio": dist["no_trade_day_ratio"],
            "env_config": str(env_config),
            "primary_old_env_episodes": primary_episodes,
            "notes": notes,
            "error_type": "",
            "error_message": "",
        }

    try:
        steps, eps = rollout_policy(
            "old_baseline_700k",
            OLD_RUN,
            OLD_CHECKPOINT,
            OLD_ENV,
            role_by_day,
            split_role="validation",
            mode="eval",
            limit_days=5,
            jax_platform=jax_platform,
        )
        dist = _action_distribution_summary(steps, eps)
        if dist["episodes"] > 0:
            return [row_from_dist(
                dist,
                status="RUN_OK_OLD_ENV",
                env_config=OLD_ENV,
                notes="Old baseline training env had validation start rows for the requested role map.",
                primary_episodes=int(dist["episodes"]),
            )]

        full_eval = _load_yaml(NEW_FULL_ENV)
        old_env = _load_yaml(OLD_ENV)
        old_reward = _nested(old_env, "training.reward_function", "futures_intraday_reward_function")
        full_eval.setdefault("training", {})["reward_function"] = old_reward
        full_eval.setdefault("trading", {})["live_mode"] = False
        full_eval.setdefault("trading", {})["live_replay_on_reset"] = False
        _write_yaml(OLD_BASELINE_FULL_EVAL_ENV, full_eval)
        steps, eps = rollout_policy(
            "old_baseline_700k_full_validation_old_reward",
            OLD_RUN,
            OLD_CHECKPOINT,
            OLD_BASELINE_FULL_EVAL_ENV,
            role_by_day,
            split_role="validation",
            mode="eval",
            limit_days=5,
            jax_platform=jax_platform,
        )
        dist = _action_distribution_summary(steps, eps)
        status = (
            "RUN_OK_FULL_VALIDATION_FALLBACK"
            if dist["episodes"] > 0 else
            "NO_VALIDATION_EPISODES"
        )
        return [row_from_dist(
            dist,
            status=status,
            env_config=OLD_BASELINE_FULL_EVAL_ENV,
            notes=(
                "Old training env produced 0 current validation episodes because its data ends before "
                "the current validation role map. Fallback uses current full JM2601 validation data "
                "with the old reward function and unchanged old checkpoint to verify same-pipeline "
                "action behavior."
            ),
            primary_episodes=0,
        )]
    except Exception as exc:
        return [{
            "checkpoint_label": "old_baseline_700k",
            "status": "RUN_FAILED",
            "episodes": 0,
            "steps": 0,
            "selected_short_count": "",
            "selected_flat_count": "",
            "selected_long_count": "",
            "trades_opened": "",
            "actual_net_pnl": "",
            "no_trade_day_ratio": "",
            "env_config": str(OLD_ENV),
            "primary_old_env_episodes": "",
            "notes": "",
            "error_type": type(exc).__name__,
            "error_message": str(exc).splitlines()[0][:500],
            "traceback_tail": "\n".join(traceback.format_exc().splitlines()[-8:]),
        }]


def classify_root_cause(
    mask_summary: list[dict],
    policy_summary: list[dict],
    train_vs_val: list[dict],
    old_summary: list[dict],
) -> tuple[str, str, str]:
    flat_short = np.mean([
        _safe_float(r.get("flat_position_short_available_ratio"))
        for r in mask_summary
    ]) if mask_summary else 0.0
    flat_long = np.mean([
        _safe_float(r.get("flat_position_long_available_ratio"))
        for r in mask_summary
    ]) if mask_summary else 0.0
    if flat_short < 0.1 and flat_long < 0.1:
        return (
            "JM3_NO_TRADE_ROOT_CAUSE_MASK",
            "SHORT/LONG were rarely available while flat.",
            "fix/action-mask audit first, no training",
        )
    stochastic_trades = sum(_safe_float(r.get("stochastic_trades_opened")) for r in policy_summary)
    eval_trades = sum(_safe_float(r.get("eval_trades_opened")) for r in policy_summary)
    if eval_trades == 0 and stochastic_trades > 0:
        return (
            "JM3_NO_TRADE_ROOT_CAUSE_EVAL_ARGMAX",
            "Deterministic eval selected no entries, but stochastic rollout opened trades.",
            "run stochastic evaluation audit or adjust eval selection protocol, no reward change",
        )
    flat_probs = [
        _safe_float(r.get("eval_flat_available_prob_flat_mean"))
        for r in policy_summary
        if r.get("eval_flat_available_prob_flat_mean") not in ("", None)
    ]
    if flat_probs and min(flat_probs) > 0.95:
        return (
            "JM3_NO_TRADE_ROOT_CAUSE_ACTOR_FLAT_COLLAPSE",
            "SHORT/LONG were available but masked policy probability mass was concentrated on FLAT.",
            "demonstration / replay prefill from old policy or oracle candidates",
        )
    train_eval = [r for r in train_vs_val if r.get("split_eval") == "train_subset"]
    val_eval = [r for r in train_vs_val if r.get("split_eval") == "validation"]
    if train_eval and val_eval:
        train_trades = sum(_safe_float(r.get("trades_opened")) for r in train_eval)
        val_trades = sum(_safe_float(r.get("trades_opened")) for r in val_eval)
        if train_trades == 0 and val_trades == 0:
            return (
                "JM3_NO_TRADE_ROOT_CAUSE_TRAIN_GLOBAL_NO_TRADE",
                "Deterministic evaluation did not trade on either train subset or validation.",
                "demonstration / replay prefill from old policy or oracle candidates",
            )
        if train_trades > 0 and val_trades == 0:
            return (
                "JM3_NO_TRADE_ROOT_CAUSE_VALIDATION_GENERALIZATION",
                "Deterministic evaluation trades on train subset but not validation.",
                "run train/regime split comparison before new training",
            )
    if old_summary and old_summary[0].get("status") != "RUN_OK":
        return (
            "JM3_NO_TRADE_ROOT_CAUSE_OLD_BASELINE_PROTOCOL_MISMATCH",
            "Old baseline checkpoint could not be replayed under the same current-code protocol.",
            "fix parity first",
        )
    return (
        "JM3_NO_TRADE_ROOT_CAUSE_INCONCLUSIVE",
        "Available evidence did not isolate a single root cause.",
        "run stochastic evaluation audit or adjust eval selection protocol, no reward change",
    )


def write_report(
    root_cause: str,
    reason: str,
    next_experiment: str,
    config_rows: list[dict],
    mask_summary: list[dict],
    policy_summary: list[dict],
    train_vs_action: list[dict],
    old_summary: list[dict],
    training_compact: dict,
) -> None:
    behavior_changes = [r for r in config_rows if r["classification"] == "POTENTIALLY_BEHAVIOR_CHANGING"]
    mask_short = np.mean([_safe_float(r.get("flat_position_short_available_ratio")) for r in mask_summary]) if mask_summary else 0
    mask_long = np.mean([_safe_float(r.get("flat_position_long_available_ratio")) for r in mask_summary]) if mask_summary else 0
    lines = [
        "# JM3 PnL-Aligned No-Trade Root Cause Audit",
        "",
        "Date: 2026-07-03",
        "",
        "## Classification",
        "",
        f"`{root_cause}`",
        "",
        reason,
        "",
        "## Key Evidence",
        "",
        f"- Validation flat-state SHORT availability mean: `{mask_short}`.",
        f"- Validation flat-state LONG availability mean: `{mask_long}`.",
        f"- Training metrics rows with episode stats: `{training_compact.get('metrics_rows_with_epstats')}`.",
        f"- Last training trades_opened_avg: `{training_compact.get('last_trades_opened_avg')}`.",
        f"- Last training reward_episode_trade_count_sum: `{training_compact.get('last_reward_episode_trade_count_sum')}`.",
        f"- Training max reward/PnL invariant error: `{training_compact.get('max_abs_reward_pnl_invariant_error_cash')}` cash.",
        f"- Old baseline same-protocol status: `{old_summary[0].get('status') if old_summary else 'missing'}`.",
        "",
        "## PnL Checkpoint Policy Summary",
        "",
        "| checkpoint | eval trades | stochastic trades | eval FLAT prob mean on flat-entry-available states | eval selected FLAT | eval selected SHORT | eval selected LONG |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in policy_summary:
        lines.append(
            f"| {row.get('checkpoint_label')} | {row.get('eval_trades_opened')} | "
            f"{row.get('stochastic_trades_opened')} | {row.get('eval_flat_available_prob_flat_mean')} | "
            f"{row.get('eval_selected_flat_count')} | {row.get('eval_selected_short_count')} | "
            f"{row.get('eval_selected_long_count')} |"
        )
    lines.extend([
        "",
        "## Config Parity",
        "",
        f"Behavior-changing or unknown differences are recorded in `{OUT / 'config_diff_matrix.csv'}`.",
        f"Potentially behavior-changing diff count: `{len(behavior_changes)}`.",
        "",
        "## Train vs Validation",
        "",
        "| checkpoint | split | episodes | trades | selected FLAT | selected SHORT | selected LONG |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in train_vs_action:
        lines.append(
            f"| {row.get('checkpoint_label')} | {row.get('split_eval')} | {row.get('episodes')} | "
            f"{row.get('trades_opened')} | {row.get('selected_flat_count')} | "
            f"{row.get('selected_short_count')} | {row.get('selected_long_count')} |"
        )
    lines.extend([
        "",
        "## Next Experiment",
        "",
        f"Recommend exactly one next experiment: `{next_experiment}`.",
        "",
        "No training, official test replay, checkpoint promotion, reward change, obs change, action-mask change, or JM8Y run was performed in this audit.",
    ])
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-subset-days", type=int, default=5)
    parser.add_argument("--skip-policy", action="store_true")
    parser.add_argument("--jax-platform", default="auto", choices=["auto", "gpu", "cuda", "cpu"])
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    _, role_by_day, _ = _prepare_candidates(ENTRY_EVAL_DIR)

    config_rows = build_config_diff_matrix()
    _write_csv(OUT / "config_diff_matrix.csv", config_rows)

    mask_summary, mask_by_day = scan_action_mask_availability(
        role_by_day, jax_platform=args.jax_platform)
    _write_csv(OUT / "action_mask_availability_summary.csv", mask_summary)
    _write_csv(OUT / "action_mask_availability_by_day.csv", mask_by_day)

    if not args.skip_policy:
        _assert_gpu_backend("policy_audit_main_preload")

    det_steps: dict[str, list[dict]] = {}
    det_eps: dict[str, list[dict]] = {}
    stochastic_steps: dict[str, list[dict]] = {}
    stochastic_eps: dict[str, list[dict]] = {}
    train_steps: dict[str, list[dict]] = {}
    train_eps: dict[str, list[dict]] = {}

    if not args.skip_policy:
        for label, checkpoint in CHECKPOINTS:
            runtime_dir = OUT / f"runtime_policy_{label}"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            cwd = Path.cwd()
            os.chdir(DREAMER_ROOT)
            try:
                config, agent = _load_agent(
                    NEW_RUN, NEW_FULL_ENV, checkpoint, runtime_dir,
                    jax_platform=args.jax_platform)
            finally:
                os.chdir(cwd)
            steps, eps = rollout_policy_loaded(
                label, config, agent, role_by_day,
                split_role="validation", mode="eval", limit_days=None)
            det_steps[label] = steps
            det_eps[label] = eps
            steps_s, eps_s = rollout_policy_loaded(
                label, config, agent, role_by_day,
                split_role="validation", mode="train", limit_days=None)
            stochastic_steps[label] = steps_s
            stochastic_eps[label] = eps_s
            steps_t, eps_t = rollout_policy_loaded(
                label, config, agent, role_by_day,
                split_role="train", mode="eval", limit_days=args.train_subset_days)
            train_steps[label] = steps_t
            train_eps[label] = eps_t

    policy_summary, policy_by_day, flat_prob_rows = summarize_policy_outputs(
        det_steps, det_eps, stochastic_steps, stochastic_eps)
    _write_csv(OUT / "policy_action_distribution_by_checkpoint.csv", policy_summary)
    _write_csv(OUT / "policy_action_distribution_by_day.csv", policy_by_day)
    _write_csv(OUT / "policy_probs_flat_available_steps.csv", flat_prob_rows)

    train_vs_action, train_vs_pnl = summarize_train_vs_validation(
        train_steps, train_eps, det_steps, det_eps)
    _write_csv(OUT / "train_vs_validation_action_summary.csv", train_vs_action)
    _write_csv(OUT / "train_vs_validation_pnl_summary.csv", train_vs_pnl)

    old_summary = (
        old_baseline_same_protocol(role_by_day, jax_platform=args.jax_platform)
        if not args.skip_policy else []
    )
    _write_csv(OUT / "old_baseline_same_protocol_summary.csv", old_summary)

    metrics_rows, reward_dist_rows, training_compact = training_metrics_summary()
    _write_csv(OUT / "training_no_trade_metrics_summary.csv", metrics_rows)
    _write_csv(OUT / "training_reward_pnl_distribution.csv", reward_dist_rows)

    root_cause, reason, next_experiment = classify_root_cause(
        mask_summary, policy_summary, train_vs_action, old_summary)
    summary = {
        "root_cause_classification": root_cause,
        "reason": reason,
        "next_recommended_experiment": next_experiment,
        "scope": {
            "jax_platform": args.jax_platform,
            "trained": False,
            "official_test_run": False,
            "checkpoint_promoted": False,
            "reward_changed": False,
            "obs_changed": False,
            "action_mask_changed": False,
            "jm8y_used": False,
            "live_orders_enabled": False,
        },
        "paths": {
            "output_dir": str(OUT),
            "report": str(DOC),
            "new_run": str(NEW_RUN),
            "old_run": str(OLD_RUN),
        },
        "training_metrics": training_compact,
        "old_baseline_same_protocol": old_summary,
        "policy_summary": policy_summary,
        "mask_availability_mean": {
            "flat_position_short_available_ratio": float(np.mean([
                _safe_float(r.get("flat_position_short_available_ratio"))
                for r in mask_summary
            ])) if mask_summary else None,
            "flat_position_long_available_ratio": float(np.mean([
                _safe_float(r.get("flat_position_long_available_ratio"))
                for r in mask_summary
            ])) if mask_summary else None,
        },
        "required_outputs": [
            "config_diff_matrix.csv",
            "action_mask_availability_summary.csv",
            "policy_action_distribution_by_checkpoint.csv",
            "policy_probs_flat_available_steps.csv",
            "train_vs_validation_action_summary.csv",
            "old_baseline_same_protocol_summary.csv",
            "training_no_trade_metrics_summary.csv",
            "summary.json",
        ],
    }
    _write_json(OUT / "summary.json", summary)
    write_report(
        root_cause,
        reason,
        next_experiment,
        config_rows,
        mask_summary,
        policy_summary,
        train_vs_action,
        old_summary,
        training_compact,
    )
    print(json.dumps({
        "root_cause_classification": root_cause,
        "reason": reason,
        "next_recommended_experiment": next_experiment,
        "output_dir": str(OUT),
        "report": str(DOC),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
