from __future__ import annotations

import csv
import json
import math
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gym_trading_env.envs.action import ForexCode
from gym_trading_env.rewards.reward_audit import REWARD_DEBUG_KEYS, validate_reward_audit
from gym_trading_env.rewards.reward_functions import (
    FuturesIntradayRewardProfitLock,
    PROFIT_LOCK_DEBUG_KEYS,
)


OUT = ROOT / "artifacts" / "reward_profit_lock_smoke"
DOC = ROOT / "docs" / "reward_profit_lock_design_and_smoke_20260703.md"


class Ledger:
    def __init__(self):
        self.cash = Decimal("10000")
        self.margin = Decimal("0")

    def balances(self):
        return {
            "user_cash": self.cash,
            "user_margin": self.margin,
        }


class SmokeEnv:
    def __init__(self):
        self.config = SimpleNamespace(
            trading=SimpleNamespace(
                alpha_unrealized=0.5,
                invalid_action_punish=0.02,
                invalid_time_cost_total=0.2,
                invalid_streak_cap=10,
                invalid_streak_mode="linear",
                atr_takeprofit_ratio=0.70,
                w_atr_close=0.10,
                lot_size=Decimal("60"),
                trade_lot=Decimal("1"),
                tick_size=Decimal("0.5"),
                spread=Decimal("0.5"),
                stop_loss_mode="abs",
                stop_loss_value=Decimal("5"),
                reward_profit_lock_enabled=True,
                reward_profit_lock_trigger_R=1.0,
                reward_profit_lock_scale_mode="daily_atr_ratio",
                reward_profit_lock_atr_ratio=0.55,
                reward_profit_lock_allowed_giveback_ratio=0.50,
                reward_profit_lock_min_capture_ratio=0.50,
                reward_profit_lock_w_giveback=0.05,
                reward_profit_lock_w_lost_mfe=0.10,
                reward_profit_lock_w_lock_close=0.05,
                reward_profit_lock_max_overlay_abs=0.20,
                reward_profit_lock_incremental_giveback=True,
                reward_profit_lock_lost_mfe_one_shot=True,
            ),
            training=SimpleNamespace(max_episode_steps=100, episode_length=None),
        )
        self.ledger = Ledger()
        self.user_accounts = SimpleNamespace(
            unrealized_pnl=Decimal("0"),
            long_position=Decimal("0"),
            short_position=Decimal("0"),
        )
        self.bar_source = SimpleNamespace(
            store=SimpleNamespace(daily_atr_price=[100.0])
        )
        self.fee_step = Decimal("0")
        self.stop_loss_fired = 0
        self.last_close_position = None
        self.current_step = 0
        self.end_idx = 0
        self._eod_idx = 100
        self._day_i = 0
        self._minutes_to_eod_last = 100
        self._R_cash_last = Decimal("100")
        self.action_result = ForexCode.SUCCESS
        self._last_action_rejected = False
        self._reward_debug = {}


def apply_state(
    env: SmokeEnv,
    *,
    unrealized: str,
    long_pos: str = "0",
    short_pos: str = "0",
    close_pnl: str | None = None,
) -> None:
    env.user_accounts.unrealized_pnl = Decimal(str(unrealized))
    env.user_accounts.long_position = Decimal(str(long_pos))
    env.user_accounts.short_position = Decimal(str(short_pos))
    env.last_close_position = (
        {"pnl": Decimal(str(close_pnl))}
        if close_pnl is not None
        else None
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite_debug(debug: dict[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in debug.values())


def run_smoke() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = SmokeEnv()
    reward_fn = FuturesIntradayRewardProfitLock(env)
    init_reward = reward_fn()
    validate_reward_audit(
        reward_fn,
        reward_name="futures_intraday_reward_profit_lock_function",
        returned_reward=init_reward,
    )

    script = [
        {
            "scenario": "below_trigger",
            "event": "open_one_tick_mfe",
            "unrealized": "30",
            "long_pos": "1",
        },
        {
            "scenario": "below_trigger",
            "event": "close_below_trigger",
            "unrealized": "0",
            "close_pnl": "30",
        },
        {
            "scenario": "giveback",
            "event": "reach_mfe_1_09R",
            "unrealized": "3600",
            "long_pos": "1",
        },
        {
            "scenario": "giveback",
            "event": "first_excess_giveback",
            "unrealized": "1200",
            "long_pos": "1",
        },
        {
            "scenario": "giveback",
            "event": "same_excess_giveback",
            "unrealized": "1200",
            "long_pos": "1",
        },
        {
            "scenario": "giveback",
            "event": "close_after_giveback",
            "unrealized": "0",
            "close_pnl": "1200",
        },
        {
            "scenario": "lock_close",
            "event": "reach_mfe_1_09R_again",
            "unrealized": "3600",
            "long_pos": "1",
        },
        {
            "scenario": "lock_close",
            "event": "close_good_capture",
            "unrealized": "0",
            "close_pnl": "2400",
        },
    ]

    rows: list[dict[str, Any]] = []
    audit_ok = True
    for step, spec in enumerate(script, start=1):
        env.current_step = step
        apply_state(
            env,
            unrealized=spec["unrealized"],
            long_pos=spec.get("long_pos", "0"),
            short_pos=spec.get("short_pos", "0"),
            close_pnl=spec.get("close_pnl"),
        )
        reward = reward_fn()
        debug = validate_reward_audit(
            reward_fn,
            reward_name="futures_intraday_reward_profit_lock_function",
            returned_reward=reward,
        )
        audit_ok = audit_ok and finite_debug(debug)
        row = {
            "step": step,
            "scenario": spec["scenario"],
            "event": spec["event"],
            "reward": float(reward),
            "long_position": float(env.user_accounts.long_position),
            "short_position": float(env.user_accounts.short_position),
            "unrealized_pnl": float(env.user_accounts.unrealized_pnl),
        }
        for key in REWARD_DEBUG_KEYS + PROFIT_LOCK_DEBUG_KEYS:
            row[key] = float(debug.get(key, 0.0))
        rows.append(row)

    by_event = {row["event"]: row for row in rows}
    inactive_rows = [row for row in rows if row["scenario"] == "below_trigger"]
    inactive_overlay_zero = all(
        abs(row["r_profit_lock_total"]) <= 1e-12 for row in inactive_rows
    )
    one_tick_not_active = by_event["open_one_tick_mfe"]["profit_lock_active"] == 0.0
    scale_cash = by_event["open_one_tick_mfe"]["profit_lock_scale_cash"]
    scale_source_code = by_event["open_one_tick_mfe"]["profit_lock_scale_source_code"]
    giveback_negative = by_event["first_excess_giveback"]["r_profit_giveback"] < 0.0
    giveback_incremental = (
        by_event["same_excess_giveback"]["profit_lock_excess_giveback_inc_cash"] == 0.0
        and by_event["same_excess_giveback"]["r_profit_giveback"] == 0.0
    )
    lock_close_positive = by_event["close_good_capture"]["r_lock_close"] > 0.0

    summary = {
        "reward_name": "futures_intraday_reward_profit_lock_function",
        "steps": len(rows),
        "audit_ok": bool(audit_ok),
        "inactive_overlay_zero": bool(inactive_overlay_zero),
        "one_tick_not_active": bool(one_tick_not_active),
        "profit_lock_scale_cash": float(scale_cash),
        "profit_lock_scale_source_code": float(scale_source_code),
        "giveback_negative": bool(giveback_negative),
        "giveback_incremental_no_repeat": bool(giveback_incremental),
        "lock_close_positive": bool(lock_close_positive),
        "smoke_pass": bool(
            audit_ok
            and inactive_overlay_zero
            and one_tick_not_active
            and abs(float(scale_cash) - 3300.0) <= 1e-9
            and float(scale_source_code) == 1.0
            and giveback_negative
            and giveback_incremental
            and lock_close_positive
        ),
        "config": {
            "trigger_R": 1.0,
            "scale_mode": "daily_atr_ratio",
            "atr_ratio": 0.55,
            "daily_atr_price": 100.0,
            "lot_size": 60.0,
            "trade_lot": 1.0,
            "tick_size": 0.5,
            "allowed_giveback_ratio": 0.50,
            "min_capture_ratio": 0.50,
            "w_giveback": 0.05,
            "w_lost_mfe": 0.10,
            "w_lock_close": 0.05,
            "max_overlay_abs": 0.20,
            "incremental_giveback": True,
            "lost_mfe_one_shot": True,
        },
    }
    return rows, summary


def write_doc(summary: dict[str, Any]) -> None:
    DOC.parent.mkdir(parents=True, exist_ok=True)
    status = "PASS" if summary["smoke_pass"] else "FAIL"
    DOC.write_text(
        "\n".join(
            [
                "# Reward Profit Lock Design and Smoke - 20260703",
                "",
                "## Design",
                "",
                "`futures_intraday_reward_profit_lock_function` subclasses the old "
                "`FuturesIntradayReward` and adds a bounded profit-lock overlay.",
                "The old reward remains registered as `futures_intraday_reward_function`.",
                "",
                "Overlay activation requires current-trade MFE >= "
                "`reward_profit_lock_trigger_R * profit_lock_scale_cash`. The default "
                "profit-lock scale is `0.55 * previous daily ATR * lot_size * trade_lot`. "
                "Entry discovery before the trigger is unchanged except for numeric audit fields.",
                "",
                "## Smoke Result",
                "",
                f"Status: {status}",
                "",
                f"- steps: {summary['steps']}",
                f"- audit_ok: {summary['audit_ok']}",
                f"- inactive_overlay_zero: {summary['inactive_overlay_zero']}",
                f"- one_tick_not_active: {summary['one_tick_not_active']}",
                f"- profit_lock_scale_cash: {summary['profit_lock_scale_cash']}",
                f"- profit_lock_scale_source_code: {summary['profit_lock_scale_source_code']}",
                f"- giveback_negative: {summary['giveback_negative']}",
                f"- giveback_incremental_no_repeat: {summary['giveback_incremental_no_repeat']}",
                f"- lock_close_positive: {summary['lock_close_positive']}",
                "",
                "## Outputs",
                "",
                "- `artifacts/reward_profit_lock_smoke/summary.json`",
                "- `artifacts/reward_profit_lock_smoke/step_reward_components.csv`",
                "",
            ]
        )
        + "\n"
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, summary = run_smoke()
    write_csv(OUT / "step_reward_components.csv", rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_doc(summary)
    return 0 if summary["smoke_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
