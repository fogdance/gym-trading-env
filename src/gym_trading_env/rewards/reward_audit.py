from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from decimal import Decimal


REWARD_AUDIT_SCHEMA_VERSION = "reward_audit_v1"

REWARD_DEBUG_KEYS = [
    "pnl",
    "fee",
    "dd",
    "eod",
    "close",
    "sl",
    "mkt_closed",
    "invalid_time",
    "invalid_streak",
    "invalid_total",
    "invalid_streak_len",
    "invalid_action_debug",
    "alpha_unrealized",
    "r_atr_close",
    "mtm_equity",
    "prev_mtm_equity",
    "delta_equity",
    "scale_cash",
    "fee_cash_debug",
    "raw_total",
    "total",
]

REWARD_EPISODE_AUDIT_KEYS = [
    "reward_total",
    "reward_pnl_sum",
    "reward_close_sum",
    "reward_atr_close_sum",
    "reward_dd_sum",
    "reward_eod_sum",
    "reward_invalid_sum",
    "actual_net_pnl",
    "final_equity",
    "max_floating_drawdown",
    "trade_count",
]


class RewardAuditError(RuntimeError):
    pass


class RewardAuditMixin:
    """
    Explicit interface for reward component audit logging.

    New formal reward functions must inherit this mixin and update
    `_last_reward_debug` on every `__call__`. The env validates the snapshot
    after reward calculation and fails fast if the reward is not auditable.
    """

    audit_schema_version = REWARD_AUDIT_SCHEMA_VERSION
    audit_required_keys = tuple(REWARD_DEBUG_KEYS)

    def default_reward_debug(self) -> dict[str, float]:
        return {key: 0.0 for key in REWARD_DEBUG_KEYS}

    def reward_audit_components_supported(self) -> tuple[str, ...]:
        return tuple(REWARD_DEBUG_KEYS)

    def reward_audit_disabled_components(self) -> tuple[str, ...]:
        return ()

    def reward_audit_debug(self) -> Mapping[str, Any]:
        debug = getattr(self, "_last_reward_debug", None)
        if debug is None:
            debug = self.default_reward_debug()
            self._set_reward_debug(debug)
        return debug

    def _set_reward_debug(self, debug: Mapping[str, Any]) -> None:
        normalized = dict(debug)
        self._last_reward_debug = normalized
        env = getattr(self, "env", None)
        if env is not None:
            env._reward_debug = normalized


def _audit_float(value: Any) -> float:
    if value is None:
        raise TypeError("None is not a numeric reward audit value")
    if isinstance(value, str):
        raise TypeError("str is not a numeric reward audit value")
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def normalize_reward_debug(debug: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(debug, Mapping):
        raise RewardAuditError(
            f"reward_audit_debug() must return a mapping, got {type(debug).__name__}")

    missing = [key for key in REWARD_DEBUG_KEYS if key not in debug]
    if missing:
        raise RewardAuditError(f"reward audit debug missing keys: {missing}")

    normalized: dict[str, float] = {}
    for key in REWARD_DEBUG_KEYS:
        value = debug[key]
        try:
            converted = _audit_float(value)
        except Exception as exc:
            raise RewardAuditError(
                f"reward audit key {key!r} is not numeric: {value!r}") from exc
        if not math.isfinite(converted):
            raise RewardAuditError(
                f"reward audit key {key!r} must be finite, got {converted!r}")
        normalized[key] = converted

    # Keep streak length integral after normalization for human-facing logs.
    normalized["invalid_streak_len"] = float(int(normalized["invalid_streak_len"]))

    for key, value in debug.items():
        if key in normalized:
            continue
        if not isinstance(key, str):
            raise RewardAuditError(f"reward audit extra key must be str, got {key!r}")
        try:
            converted = _audit_float(value)
        except Exception as exc:
            raise RewardAuditError(
                f"reward audit extra key {key!r} is not numeric: {value!r}") from exc
        if not math.isfinite(converted):
            raise RewardAuditError(
                f"reward audit extra key {key!r} must be finite, got {converted!r}")
        normalized[key] = converted

    return normalized


def validate_reward_audit(
    reward_function: Any,
    *,
    reward_name: str,
    returned_reward: float | None = None,
) -> dict[str, float]:
    if not isinstance(reward_function, RewardAuditMixin):
        raise RewardAuditError(
            f"Reward function {reward_name!r} must implement RewardAuditMixin")

    schema = getattr(reward_function, "audit_schema_version", None)
    if schema != REWARD_AUDIT_SCHEMA_VERSION:
        raise RewardAuditError(
            f"Reward function {reward_name!r} has unsupported audit schema "
            f"{schema!r}; expected {REWARD_AUDIT_SCHEMA_VERSION!r}")

    supported = set(reward_function.reward_audit_components_supported())
    missing_supported = [key for key in REWARD_DEBUG_KEYS if key not in supported]
    if missing_supported:
        raise RewardAuditError(
            f"Reward function {reward_name!r} does not declare support for "
            f"audit keys: {missing_supported}")

    debug = normalize_reward_debug(reward_function.reward_audit_debug())

    if returned_reward is not None:
        expected = float(returned_reward)
        actual = debug["total"]
        tolerance = max(1e-6, 1e-5 * abs(expected))
        if abs(actual - expected) > tolerance:
            raise RewardAuditError(
                f"Reward function {reward_name!r} audit total {actual} does not "
                f"match returned reward {expected}")

    disabled = set(reward_function.reward_audit_disabled_components())
    unknown_disabled = [key for key in disabled if key not in debug]
    if unknown_disabled:
        raise RewardAuditError(
            f"Reward function {reward_name!r} declares unknown disabled audit "
            f"components: {unknown_disabled}")
    for key in disabled:
        if abs(debug[key]) > 1e-12:
            raise RewardAuditError(
                f"Reward function {reward_name!r} disabled audit component "
                f"{key!r} is non-zero: {debug[key]}")

    return debug
