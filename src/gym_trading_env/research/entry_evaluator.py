from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml

from gym_trading_env.utils.market_features import FEATURES_MARKET_OBS, build_market_features
from gym_trading_env.utils.ohlcvi_contract import normalize_ohlcvi
from gym_trading_env.utils.time_contract import ensure_feature_tz_index


Direction = Literal["LONG", "SHORT"]
SEGMENT_STARTS = np.asarray([0, 120, 195, 255], dtype=np.int32)
SEGMENT_ENDS = np.asarray([120, 195, 255, 345], dtype=np.int32)


@dataclass(frozen=True)
class ProductConfig:
    initial_balance: float
    tick_size: float
    spread: float
    trading_fee_per_lot: float
    is_round_turn: bool
    leverage: float
    lot_size: float
    trade_lot: float
    max_entries_per_day: int

    def validate(self) -> None:
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be > 0")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be > 0")
        if self.spread < 0:
            raise ValueError("spread must be >= 0")
        if self.trading_fee_per_lot < 0:
            raise ValueError("trading_fee_per_lot must be >= 0")
        if self.leverage <= 0:
            raise ValueError("leverage must be > 0")
        if self.lot_size <= 0 or self.trade_lot <= 0:
            raise ValueError("lot_size and trade_lot must be > 0")
        if self.max_entries_per_day <= 0:
            raise ValueError("max_entries_per_day must be > 0")


@dataclass(frozen=True)
class EvaluatorConfig:
    entry_delay_bars: int
    stop_distance_price: float
    stop_distance_ticks: int
    take_profit_rr: float
    max_hold_bars: int
    max_hold_unit: str
    allow_entry_across_break: bool
    force_flatten_eod: bool
    intrabar_collision: str

    def validate(self, product: ProductConfig) -> None:
        if self.entry_delay_bars != 1:
            raise ValueError("entry_delay_bars=1 is the only supported canonical mode")
        if self.stop_distance_price <= 0 or self.stop_distance_ticks <= 0:
            raise ValueError("stop distance must be > 0")
        expected = self.stop_distance_ticks * product.tick_size
        if not np.isclose(self.stop_distance_price, expected, rtol=0, atol=1e-12):
            raise ValueError(
                "stop_distance_price must equal stop_distance_ticks * tick_size: "
                f"{self.stop_distance_price} != {self.stop_distance_ticks} * {product.tick_size}")
        if self.take_profit_rr <= 0:
            raise ValueError("take_profit_rr must be > 0")
        if self.max_hold_bars <= 0:
            raise ValueError("max_hold_bars must be > 0")
        if self.max_hold_unit != "valid_1m_bars":
            raise ValueError("max_hold_unit must be valid_1m_bars")
        if self.intrabar_collision != "stop_first":
            raise ValueError("intrabar_collision must be stop_first")
        if not self.force_flatten_eod:
            raise ValueError("force_flatten_eod must be true for entry capability evaluation")


@dataclass(frozen=True)
class DataConfig:
    path: str
    timezone: str
    product: str
    contract: str
    window_size: int
    observation_mode: str

    def validate(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        if self.observation_mode != "obs":
            raise ValueError("entry evaluator currently requires observation_mode=obs")


@dataclass(frozen=True)
class ExperimentConfig:
    random_seed: int = 0
    ridge_alpha: float = 10.0
    threshold_quantiles: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 0.95)
    random_baseline_runs: int = 100
    bootstrap_runs: int = 2000


@dataclass(frozen=True)
class SensitivityConfig:
    stop_distance_ticks: tuple[int, ...] = (6, 10, 16)
    take_profit_rr: tuple[float, ...] = (1.0, 1.5, 2.0)
    max_hold_bars: tuple[int, ...] = (15, 30, 60)
    cost_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)

    def validate(self) -> None:
        if not self.stop_distance_ticks or any(x <= 0 for x in self.stop_distance_ticks):
            raise ValueError("sensitivity.stop_distance_ticks must contain positive values")
        if not self.take_profit_rr or any(x <= 0 for x in self.take_profit_rr):
            raise ValueError("sensitivity.take_profit_rr must contain positive values")
        if not self.max_hold_bars or any(x <= 0 for x in self.max_hold_bars):
            raise ValueError("sensitivity.max_hold_bars must contain positive values")
        if not self.cost_multipliers or any(x <= 0 for x in self.cost_multipliers):
            raise ValueError("sensitivity.cost_multipliers must contain positive values")


@dataclass(frozen=True)
class EntryEvalConfig:
    version: str
    data: DataConfig
    product: ProductConfig
    entry_evaluator: EvaluatorConfig
    experiment: ExperimentConfig
    sensitivity: SensitivityConfig

    def validate(self) -> None:
        self.data.validate()
        self.product.validate()
        self.entry_evaluator.validate(self.product)
        self.sensitivity.validate()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EntryOutcome:
    outcome_version: str
    dataset_version: str
    exit_policy_version: str
    product: str
    contract: str
    candidate_id: int
    decision_row: int
    entry_row: int
    exit_row: int
    decision_timestamp: str
    entry_timestamp: str
    exit_timestamp: str
    trading_day: int
    direction: Direction
    entry_reference_price: float
    entry_execution_price: float
    stop_price: float
    target_price: float
    exit_reference_price: float
    exit_execution_price: float
    exit_reason: str
    holding_bars: int
    entry_crosses_break: bool
    gross_pnl: float
    spread_cost: float
    fee_cost: float
    net_pnl: float
    return_on_initial_equity: float
    mfe_gross: float
    mae_gross: float
    mfe_net_after_entry_cost: float
    mae_net_after_entry_cost: float
    stop_target_collision: bool


@dataclass(frozen=True)
class EvaluationContext:
    mask: np.ndarray
    days: np.ndarray
    segments: np.ndarray
    next_valid: np.ndarray
    valid_rows_by_day: dict[int, np.ndarray]
    opens: np.ndarray
    closes: np.ndarray
    highs: np.ndarray
    lows: np.ndarray


def load_entry_eval_config(path: str | Path) -> EntryEvalConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    exp = dict(raw.get("experiment", {}) or {})
    exp["threshold_quantiles"] = tuple(float(x) for x in exp.get(
        "threshold_quantiles", (0.5, 0.7, 0.8, 0.9, 0.95)))
    sensitivity = dict(raw.get("sensitivity", {}) or {})
    sensitivity["stop_distance_ticks"] = tuple(int(x) for x in sensitivity.get(
        "stop_distance_ticks", (6, 10, 16)))
    sensitivity["take_profit_rr"] = tuple(float(x) for x in sensitivity.get(
        "take_profit_rr", (1.0, 1.5, 2.0)))
    sensitivity["max_hold_bars"] = tuple(int(x) for x in sensitivity.get(
        "max_hold_bars", (15, 30, 60)))
    sensitivity["cost_multipliers"] = tuple(float(x) for x in sensitivity.get(
        "cost_multipliers", (1.0, 1.5, 2.0)))
    config = EntryEvalConfig(
        version=str(raw["version"]),
        data=DataConfig(**raw["data"]),
        product=ProductConfig(**raw["product"]),
        entry_evaluator=EvaluatorConfig(**raw["entry_evaluator"]),
        experiment=ExperimentConfig(**exp),
        sensitivity=SensitivityConfig(**sensitivity),
    )
    config.validate()
    return config


def _segment_ids(minutes: np.ndarray) -> np.ndarray:
    minutes = np.asarray(minutes, dtype=np.int32)
    seg = np.searchsorted(SEGMENT_STARTS, minutes, side="right") - 1
    clipped = np.clip(seg, 0, len(SEGMENT_STARTS) - 1)
    valid = (seg >= 0) & (minutes < SEGMENT_ENDS[clipped])
    if not np.all(valid):
        raise ValueError(f"minute_index outside futures segments: {minutes[~valid][:5].tolist()}")
    return seg.astype(np.int8)


def load_market_frames(config: EntryEvalConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(config.data.path)
    raw = normalize_ohlcvi(raw, date_col="Date")
    raw.index = ensure_feature_tz_index(raw.index, assume_tz=config.data.timezone)
    market = build_market_features(raw, tz=config.data.timezone, is_future=True)

    aligned = raw.reindex(market.index)
    for col in ("Open", "High", "Low", "Close"):
        market[f"exec_{col.lower()}"] = pd.to_numeric(
            aligned[col], errors="coerce").to_numpy(dtype=float)
    return raw, market


def _next_valid_same_day(mask: np.ndarray, days: np.ndarray) -> np.ndarray:
    out = np.full(len(mask), -1, dtype=np.int64)
    next_row = -1
    next_day = None
    for row in range(len(mask) - 1, -1, -1):
        day = days[row]
        if next_day != day:
            next_row = -1
        out[row] = next_row
        if mask[row]:
            next_row = row
        next_day = day
    return out


def build_evaluation_context(market: pd.DataFrame) -> EvaluationContext:
    mask = market["mask_t"].to_numpy(dtype=float) > 0.0
    days = market["trading_day"].to_numpy(dtype=np.int64)
    minutes = market["minute_index_t"].to_numpy(dtype=np.int32)
    segments = _segment_ids(minutes)
    nxt = _next_valid_same_day(mask, days)
    valid_rows_by_day = {
        int(day): rows.astype(np.int64, copy=False)
        for day in np.unique(days)
        if (rows := np.flatnonzero(mask & (days == day))).size
    }
    return EvaluationContext(
        mask=mask,
        days=days,
        segments=segments,
        next_valid=nxt,
        valid_rows_by_day=valid_rows_by_day,
        opens=market["exec_open"].to_numpy(dtype=float),
        closes=market["C_t"].to_numpy(dtype=float),
        highs=market["H_t"].to_numpy(dtype=float),
        lows=market["L_t"].to_numpy(dtype=float),
    )


def _can_open_one_lot(entry_reference: float, config: EntryEvalConfig) -> bool:
    product = config.product
    fee = product.trading_fee_per_lot * product.trade_lot
    size_cash = product.trade_lot * product.lot_size
    long_required = size_cash * (entry_reference + product.spread) / product.leverage + fee
    short_required = size_cash * (entry_reference - product.spread) / product.leverage + fee
    return max(long_required, short_required) <= product.initial_balance


def candidate_rows(
    market: pd.DataFrame,
    config: EntryEvalConfig,
    *,
    context: EvaluationContext | None = None,
) -> np.ndarray:
    context = context or build_evaluation_context(market)

    rows = []
    for row in np.flatnonzero(context.mask):
        entry_row = int(context.next_valid[row])
        if entry_row < 0:
            continue
        if not config.entry_evaluator.allow_entry_across_break:
            if context.segments[row] != context.segments[entry_row]:
                continue
            # A missing/invalid slot inside a segment is also not a canonical next-open entry.
            if entry_row != row + 1:
                continue
        entry_open = float(context.opens[entry_row])
        if not np.isfinite(entry_open):
            continue
        if not _can_open_one_lot(entry_open, config):
            continue
        rows.append(int(row))
    return np.asarray(rows, dtype=np.int64)


def _exit_prices(
    direction: Direction,
    reference: float,
    spread: float,
) -> float:
    return reference - spread if direction == "LONG" else reference + spread


def evaluate_entry(
    market: pd.DataFrame,
    decision_row: int,
    direction: Direction,
    config: EntryEvalConfig,
    *,
    candidate_id: int = 0,
    context: EvaluationContext | None = None,
) -> EntryOutcome:
    config.validate()
    if direction not in ("LONG", "SHORT"):
        raise ValueError(f"unsupported direction: {direction}")
    context = context or build_evaluation_context(market)
    if not context.mask[decision_row]:
        raise ValueError("decision row must be a valid market bar")

    day = int(context.days[decision_row])
    valid_rows = context.valid_rows_by_day[day]
    entry_row = int(context.next_valid[decision_row])
    if entry_row < 0:
        raise ValueError("decision row has no next valid bar in the same trading day")
    crosses_break = bool(
        context.segments[decision_row] != context.segments[entry_row]
        or entry_row != decision_row + 1
    )
    if crosses_break and not config.entry_evaluator.allow_entry_across_break:
        raise ValueError("entry crosses a trading break")

    entry_ref = float(context.opens[entry_row])
    if not np.isfinite(entry_ref):
        raise ValueError("entry open is not finite")
    if not _can_open_one_lot(entry_ref, config):
        raise ValueError("flat canonical account cannot open one lot")

    product = config.product
    rules = config.entry_evaluator
    spread = product.spread
    distance = rules.stop_distance_price
    size_cash = product.trade_lot * product.lot_size

    if direction == "LONG":
        entry_exec = entry_ref + spread
        stop_exec = entry_exec - distance
        target_exec = entry_exec + distance * rules.take_profit_rr
    else:
        entry_exec = entry_ref - spread
        stop_exec = entry_exec + distance
        target_exec = entry_exec - distance * rules.take_profit_rr

    first = int(np.searchsorted(valid_rows, entry_row, side="left"))
    day_rows = valid_rows[first:]
    if day_rows.size == 0:
        raise ValueError("entry row is not valid")

    exit_row = int(day_rows[-1])
    exit_reason = "EOD"
    exit_exec = _exit_prices(direction, float(context.closes[exit_row]), spread)
    collision = False
    processed = 0
    max_favorable = 0.0
    max_adverse = 0.0

    for row in day_rows:
        row = int(row)
        processed += 1
        high = float(context.highs[row])
        low = float(context.lows[row])

        if direction == "LONG":
            max_favorable = max(max_favorable, (high - entry_ref) * size_cash)
            max_adverse = min(max_adverse, (low - entry_ref) * size_cash)
            hit_stop = low <= stop_exec
            hit_target = high >= target_exec
        else:
            max_favorable = max(max_favorable, (entry_ref - low) * size_cash)
            max_adverse = min(max_adverse, (entry_ref - high) * size_cash)
            hit_stop = high >= stop_exec
            hit_target = low <= target_exec

        collision = collision or bool(hit_stop and hit_target)
        if hit_stop:
            exit_row = row
            exit_reason = "STOP"
            exit_exec = stop_exec
            break
        if hit_target:
            exit_row = row
            exit_reason = "TARGET"
            exit_exec = target_exec
            break
        if processed >= rules.max_hold_bars:
            exit_row = row
            exit_reason = "MAX_HOLD"
            exit_exec = _exit_prices(direction, float(context.closes[row]), spread)
            break

    if direction == "LONG":
        exit_ref = exit_exec + spread
        execution_pnl = (exit_exec - entry_exec) * size_cash
        gross_pnl = (exit_ref - entry_ref) * size_cash
    else:
        exit_ref = exit_exec - spread
        execution_pnl = (entry_exec - exit_exec) * size_cash
        gross_pnl = (entry_ref - exit_ref) * size_cash

    spread_cost = gross_pnl - execution_pnl
    fee_sides = 2 if product.is_round_turn else 1
    fee_cost = product.trading_fee_per_lot * product.trade_lot * fee_sides
    net_pnl = execution_pnl - fee_cost
    entry_cost = spread * product.trade_lot * product.lot_size

    idx = market.index
    return EntryOutcome(
        outcome_version="entry_outcome_v1",
        dataset_version=config.version,
        exit_policy_version="fixed_barrier_v1",
        product=config.data.product,
        contract=config.data.contract,
        candidate_id=int(candidate_id),
        decision_row=int(decision_row),
        entry_row=entry_row,
        exit_row=int(exit_row),
        decision_timestamp=str(idx[decision_row]),
        entry_timestamp=str(idx[entry_row]),
        exit_timestamp=str(idx[exit_row]),
        trading_day=day,
        direction=direction,
        entry_reference_price=entry_ref,
        entry_execution_price=float(entry_exec),
        stop_price=float(stop_exec),
        target_price=float(target_exec),
        exit_reference_price=float(exit_ref),
        exit_execution_price=float(exit_exec),
        exit_reason=exit_reason,
        holding_bars=int(processed),
        entry_crosses_break=crosses_break,
        gross_pnl=float(gross_pnl),
        spread_cost=float(spread_cost),
        fee_cost=float(fee_cost),
        net_pnl=float(net_pnl),
        return_on_initial_equity=float(net_pnl / product.initial_balance),
        mfe_gross=float(max_favorable),
        mae_gross=float(max_adverse),
        mfe_net_after_entry_cost=float(max_favorable - entry_cost - fee_cost),
        mae_net_after_entry_cost=float(max_adverse - entry_cost - fee_cost),
        stop_target_collision=bool(collision),
    )


def _market_window(X: np.ndarray, row: int, window_size: int) -> np.ndarray:
    start = row - window_size + 1
    if start >= 0:
        return X[start:row + 1]
    pad = np.zeros((-start, X.shape[1]), dtype=np.float32)
    return np.concatenate([pad, X[:row + 1]], axis=0)


def summarize_market_window(window: np.ndarray) -> np.ndarray:
    window = np.asarray(window, dtype=np.float64)
    latest = window[-1]
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    delta = window[-1] - window[0]
    return np.concatenate([latest, mean, std, delta]).astype(np.float32)


def feature_names() -> list[str]:
    return [
        f"{agg}:{name}"
        for agg in ("latest", "mean", "std", "delta")
        for name in FEATURES_MARKET_OBS
    ]


def build_entry_dataset(
    market: pd.DataFrame,
    config: EntryEvalConfig,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    context = build_evaluation_context(market)
    rows = candidate_rows(market, config, context=context)
    X_all = market[FEATURES_MARKET_OBS].to_numpy(dtype=np.float32)
    records = []
    features = []

    for candidate_id, row in enumerate(rows):
        long_out = evaluate_entry(
            market, int(row), "LONG", config, candidate_id=candidate_id, context=context)
        short_out = evaluate_entry(
            market, int(row), "SHORT", config, candidate_id=candidate_id, context=context)
        records.append({
            "dataset_version": config.version,
            "exit_policy_version": long_out.exit_policy_version,
            "product": config.data.product,
            "contract": config.data.contract,
            "candidate_id": candidate_id,
            "decision_row": int(row),
            "decision_timestamp": long_out.decision_timestamp,
            "trading_day": long_out.trading_day,
            "action_mask_short": 1.0,
            "action_mask_flat": 1.0,
            "action_mask_long": 1.0,
            "flat_net_pnl": 0.0,
            "long_entry_row": long_out.entry_row,
            "long_entry_timestamp": long_out.entry_timestamp,
            "long_exit_row": long_out.exit_row,
            "long_exit_timestamp": long_out.exit_timestamp,
            "long_exit_reason": long_out.exit_reason,
            "long_holding_bars": long_out.holding_bars,
            "long_gross_pnl": long_out.gross_pnl,
            "long_spread_cost": long_out.spread_cost,
            "long_fee_cost": long_out.fee_cost,
            "long_net_pnl": long_out.net_pnl,
            "long_mfe_gross": long_out.mfe_gross,
            "long_mae_gross": long_out.mae_gross,
            "short_entry_row": short_out.entry_row,
            "short_entry_timestamp": short_out.entry_timestamp,
            "short_exit_row": short_out.exit_row,
            "short_exit_timestamp": short_out.exit_timestamp,
            "short_exit_reason": short_out.exit_reason,
            "short_holding_bars": short_out.holding_bars,
            "short_gross_pnl": short_out.gross_pnl,
            "short_spread_cost": short_out.spread_cost,
            "short_fee_cost": short_out.fee_cost,
            "short_net_pnl": short_out.net_pnl,
            "short_mfe_gross": short_out.mfe_gross,
            "short_mae_gross": short_out.mae_gross,
            "stop_target_collision": bool(
                long_out.stop_target_collision or short_out.stop_target_collision),
        })
        features.append(summarize_market_window(
            _market_window(X_all, int(row), config.data.window_size)))

    return pd.DataFrame.from_records(records), np.asarray(features, dtype=np.float32), feature_names()
