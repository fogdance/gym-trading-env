from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xn = (np.asarray(X, dtype=np.float64) - self.mean) / self.scale
        return Xn @ self.coef + self.intercept


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> RidgeModel:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-12] = 1.0
    Xn = (X - mean) / scale
    intercept = float(y.mean())
    yc = y - intercept
    gram = Xn.T @ Xn
    coef = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0], dtype=np.float64),
        Xn.T @ yc,
    )
    return RidgeModel(mean=mean, scale=scale, coef=coef, intercept=intercept)


def predict_actions(
    pred_long: np.ndarray,
    pred_short: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    pred_long = np.asarray(pred_long, dtype=float)
    pred_short = np.asarray(pred_short, dtype=float)
    best = np.maximum(pred_long, pred_short)
    direction = np.where(pred_long >= pred_short, "LONG", "SHORT").astype(object)
    direction[best <= float(threshold)] = "FLAT"
    return direction, best


def apply_direction_score_threshold(
    directions: Iterable[str],
    scores: Iterable[float],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(list(directions), dtype=object).copy()
    score_arr = np.asarray(list(scores), dtype=float)
    actions[score_arr <= float(threshold)] = "FLAT"
    return actions, score_arr


def simulate_strategy(
    candidates: pd.DataFrame,
    actions: Iterable[str],
    scores: Iterable[float],
    *,
    max_entries_per_day: int,
) -> pd.DataFrame:
    work = candidates.copy()
    work["action"] = np.asarray(list(actions), dtype=object)
    work["score"] = np.asarray(list(scores), dtype=float)
    work = work.sort_values(["decision_row", "candidate_id"]).reset_index(drop=True)

    trades = []
    active_exit_row = -1
    day_entries: dict[int, int] = {}
    for row in work.itertuples(index=False):
        action = str(row.action)
        if action == "FLAT":
            continue
        if int(row.decision_row) <= active_exit_row:
            continue
        day = int(row.trading_day)
        if day_entries.get(day, 0) >= int(max_entries_per_day):
            continue

        prefix = action.lower()
        exit_row = int(getattr(row, f"{prefix}_exit_row"))
        trade = {
            "candidate_id": int(row.candidate_id),
            "contract": getattr(row, "contract", None),
            "month": getattr(row, "month", None),
            "session_phase": getattr(row, "session_phase", None),
            "trading_day": day,
            "decision_row": int(row.decision_row),
            "decision_timestamp": row.decision_timestamp,
            "direction": action,
            "score": float(row.score),
            "entry_row": int(getattr(row, f"{prefix}_entry_row")),
            "entry_timestamp": getattr(row, f"{prefix}_entry_timestamp"),
            "exit_row": exit_row,
            "exit_timestamp": getattr(row, f"{prefix}_exit_timestamp"),
            "exit_reason": getattr(row, f"{prefix}_exit_reason"),
            "holding_bars": int(getattr(row, f"{prefix}_holding_bars")),
            "gross_pnl": float(getattr(row, f"{prefix}_gross_pnl")),
            "spread_cost": float(getattr(row, f"{prefix}_spread_cost")),
            "fee_cost": float(getattr(row, f"{prefix}_fee_cost")),
            "net_pnl": float(getattr(row, f"{prefix}_net_pnl")),
            "mfe_gross": float(getattr(row, f"{prefix}_mfe_gross")),
            "mae_gross": float(getattr(row, f"{prefix}_mae_gross")),
        }
        trades.append(trade)
        active_exit_row = exit_row
        day_entries[day] = day_entries.get(day, 0) + 1
    return pd.DataFrame.from_records(trades)


def _profit_factor(wins: np.ndarray, losses: np.ndarray) -> tuple[float | str | None, str]:
    if wins.size == 0 and losses.size == 0:
        return None, "not_computed"
    if losses.size == 0:
        return "inf", "no_losing_trades"
    if float(wins.sum()) <= 0.0:
        return 0.0, "no_winning_trades"
    return float(wins.sum() / abs(losses.sum())), "computed"


def strategy_metrics(
    trades: pd.DataFrame,
    *,
    all_days: Iterable[int],
    initial_balance: float,
) -> dict:
    days = np.asarray(list(all_days), dtype=np.int64)
    if trades.empty:
        return {
            "trades": 0,
            "net_pnl": 0.0,
            "expectancy": None,
            "win_rate": None,
            "profit_factor": None,
            "profit_factor_status": "not_computed",
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "no_trade_day_ratio": 1.0 if len(days) else None,
            "trades_per_day": {"0": int(len(days))},
            "max_trade_profit_share": None,
            "var_05": None,
            "cvar_05": None,
            "long_trades": 0,
            "short_trades": 0,
            "profitable_month_ratio": None,
        }

    pnl = trades["net_pnl"].to_numpy(dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor, profit_factor_status = _profit_factor(wins, losses)
    daily = trades.groupby("trading_day")["net_pnl"].sum().reindex(days, fill_value=0.0)
    equity = float(initial_balance) + daily.cumsum().to_numpy(dtype=float)
    peak = np.maximum.accumulate(np.concatenate([[float(initial_balance)], equity]))
    curve = np.concatenate([[float(initial_balance)], equity])
    drawdown = np.where(peak > 0, (peak - curve) / peak, 0.0)
    per_day = trades.groupby("trading_day").size().reindex(days, fill_value=0)
    dist = per_day.value_counts().sort_index()
    total_profit = float(wins.sum())
    var_05 = float(np.quantile(pnl, 0.05))
    cvar_tail = pnl[pnl <= var_05]
    month = (trades["trading_day"].astype(int) // 100).astype(int)
    monthly = trades.groupby(month)["net_pnl"].sum()

    return {
        "trades": int(len(trades)),
        "net_pnl": float(pnl.sum()),
        "expectancy": float(pnl.mean()),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": profit_factor,
        "profit_factor_status": profit_factor_status,
        "return_pct": float(pnl.sum() / float(initial_balance) * 100.0),
        "max_drawdown_pct": float(drawdown.max() * 100.0),
        "no_trade_day_ratio": float((per_day == 0).mean()) if len(per_day) else None,
        "trades_per_day": {str(int(k)): int(v) for k, v in dist.items()},
        "long_trades": int((trades["direction"] == "LONG").sum()),
        "short_trades": int((trades["direction"] == "SHORT").sum()),
        "max_trade_profit_share": (
            None if total_profit <= 0 else float(wins.max() / total_profit)),
        "var_05": var_05,
        "cvar_05": float(cvar_tail.mean()),
        "profitable_month_ratio": float((monthly > 0).mean()) if len(monthly) else None,
    }


def constrained_oracle(
    candidates: pd.DataFrame,
    *,
    max_entries_per_day: int,
) -> pd.DataFrame:
    actions = np.full(len(candidates), "FLAT", dtype=object)
    scores = np.zeros(len(candidates), dtype=float)
    day_values = candidates["trading_day"].to_numpy(dtype=np.int64)

    for day in np.unique(day_values):
        positions = np.flatnonzero(day_values == day)
        intervals = []
        for pos in positions:
            row = candidates.iloc[int(pos)]
            for direction in ("LONG", "SHORT"):
                prefix = direction.lower()
                pnl = float(row[f"{prefix}_net_pnl"])
                if pnl <= 0.0:
                    continue
                intervals.append({
                    "pos": int(pos),
                    "direction": direction,
                    "start": int(row["decision_row"]),
                    "end": int(row[f"{prefix}_exit_row"]),
                    "pnl": pnl,
                })
        intervals.sort(key=lambda x: (x["end"], x["start"], x["pos"], x["direction"]))
        if not intervals:
            continue

        ends = [x["end"] for x in intervals]
        predecessors = [
            bisect.bisect_left(ends, interval["start"]) - 1
            for interval in intervals
        ]
        count = len(intervals)
        limit = int(max_entries_per_day)
        dp = np.zeros((limit + 1, count + 1), dtype=float)
        take = np.zeros((limit + 1, count + 1), dtype=bool)
        for trades in range(1, limit + 1):
            for idx in range(1, count + 1):
                interval = intervals[idx - 1]
                take_value = interval["pnl"] + dp[trades - 1, predecessors[idx - 1] + 1]
                skip_value = dp[trades, idx - 1]
                if take_value > skip_value:
                    dp[trades, idx] = take_value
                    take[trades, idx] = True
                else:
                    dp[trades, idx] = skip_value

        trades = limit
        idx = count
        while trades > 0 and idx > 0:
            if not take[trades, idx]:
                idx -= 1
                continue
            interval = intervals[idx - 1]
            actions[interval["pos"]] = interval["direction"]
            scores[interval["pos"]] = interval["pnl"]
            idx = predecessors[idx - 1] + 1
            trades -= 1

    return simulate_strategy(
        candidates, actions, scores, max_entries_per_day=max_entries_per_day)


def fixed_direction_strategy(
    candidates: pd.DataFrame,
    direction: str,
    *,
    max_entries_per_day: int,
) -> pd.DataFrame:
    return simulate_strategy(
        candidates,
        np.full(len(candidates), direction, dtype=object),
        np.ones(len(candidates), dtype=float),
        max_entries_per_day=max_entries_per_day,
    )


def simple_rule_actions(
    X: np.ndarray,
    feature_names: list[str],
    feature: str,
) -> tuple[np.ndarray, np.ndarray]:
    idx = feature_names.index(feature)
    value = np.asarray(X[:, idx], dtype=float)
    actions = np.where(value > 0, "LONG", np.where(value < 0, "SHORT", "FLAT"))
    return actions.astype(object), np.abs(value)


def add_diagnostic_context(
    candidates: pd.DataFrame,
    X: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    out = candidates.copy()
    out["month"] = (out["trading_day"].astype(int) // 100).astype(int)
    if "latest:obs_session_phase_t" in feature_names:
        idx = feature_names.index("latest:obs_session_phase_t")
        phase = np.asarray(X[:, idx], dtype=float)
        out["session_phase"] = np.where(phase >= 0.5, "day", "night")
    else:
        out["session_phase"] = "unknown"
    return out


def random_action_strategies(
    candidates: pd.DataFrame,
    *,
    max_entries_per_day: int,
    runs: int,
    seed: int,
) -> list[pd.DataFrame]:
    out = []
    for run in range(int(runs)):
        rng = np.random.default_rng(int(seed) + 10000 + run)
        actions = rng.choice(
            np.asarray(["LONG", "SHORT", "FLAT"], dtype=object),
            size=len(candidates),
            p=np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        )
        scores = rng.random(len(candidates))
        out.append(simulate_strategy(
            candidates, actions, scores, max_entries_per_day=max_entries_per_day))
    return out


def summarize_random_runs(
    runs: list[pd.DataFrame],
    *,
    all_days: Iterable[int],
    initial_balance: float,
) -> dict:
    if not runs:
        return {"runs": 0}
    metrics = [
        strategy_metrics(run, all_days=all_days, initial_balance=initial_balance)
        for run in runs
    ]
    pnl = np.asarray([x["net_pnl"] for x in metrics], dtype=float)
    trades = np.asarray([x["trades"] for x in metrics], dtype=float)
    return {
        "runs": int(len(runs)),
        "trade_count_min": int(trades.min()),
        "trade_count_max": int(trades.max()),
        "trade_count_mean": float(trades.mean()),
        "net_pnl_mean": float(pnl.mean()),
        "net_pnl_p05": float(np.quantile(pnl, 0.05)),
        "net_pnl_p50": float(np.quantile(pnl, 0.50)),
        "net_pnl_p95": float(np.quantile(pnl, 0.95)),
    }


def matched_random_summary(
    candidates: pd.DataFrame,
    target_trades: pd.DataFrame,
    *,
    max_entries_per_day: int,
    runs: int,
    seed: int,
    all_days: Iterable[int],
    initial_balance: float,
) -> dict:
    random_runs = matched_random_strategies(
        candidates,
        target_trades,
        max_entries_per_day=max_entries_per_day,
        runs=runs,
        seed=seed,
    )
    summary = summarize_random_runs(
        random_runs, all_days=all_days, initial_balance=initial_balance)
    summary["target_trades"] = int(len(target_trades))
    return summary


def matched_random_strategies(
    candidates: pd.DataFrame,
    target_trades: pd.DataFrame,
    *,
    max_entries_per_day: int,
    runs: int,
    seed: int,
) -> list[pd.DataFrame]:
    target_by_day = target_trades.groupby("trading_day").size().to_dict()
    out = []
    for run in range(int(runs)):
        rng = np.random.default_rng(int(seed) + run)
        parts = []
        for day, target in target_by_day.items():
            day_rows = candidates[candidates["trading_day"] == day].copy().reset_index(drop=True)
            if day_rows.empty:
                continue
            target = min(int(target), int(max_entries_per_day))
            selected = pd.DataFrame()
            # Random sparse signals create random entry timing. The all-active fallback
            # guarantees the requested trade count whenever the constrained target did.
            for attempt in range(20):
                probability = float(rng.uniform(0.01, 0.25)) if attempt < 19 else 1.0
                active = rng.random(len(day_rows)) < probability
                actions = np.where(
                    active,
                    np.where(rng.random(len(day_rows)) >= 0.5, "LONG", "SHORT"),
                    "FLAT",
                )
                selected = simulate_strategy(
                    day_rows,
                    actions,
                    rng.random(len(day_rows)),
                    max_entries_per_day=target,
                )
                if len(selected) == target:
                    break
            if len(selected) != target:
                raise AssertionError(
                    f"random baseline could not match {target} trades on day {day}")
            parts.append(selected)
        out.append(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())
    return out


def _realized_predicted_direction_pnl(
    candidates: pd.DataFrame,
    pred_long: np.ndarray,
    pred_short: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_long = np.asarray(pred_long, dtype=float)
    pred_short = np.asarray(pred_short, dtype=float)
    is_long = pred_long >= pred_short
    score = np.maximum(pred_long, pred_short)
    direction = np.where(is_long, "LONG", "SHORT").astype(object)
    realized = np.where(
        is_long,
        candidates["long_net_pnl"].to_numpy(dtype=float),
        candidates["short_net_pnl"].to_numpy(dtype=float),
    )
    return direction, score, realized


def candidate_score_expectancy_report(
    candidates: pd.DataFrame,
    pred_long: np.ndarray,
    pred_short: np.ndarray,
    *,
    bins: int = 10,
) -> dict:
    if len(candidates) < 2:
        return {"bins": [], "spearman": None, "top_above_bottom": None}
    direction, score, realized = _realized_predicted_direction_pnl(
        candidates, pred_long, pred_short)
    work = pd.DataFrame({
        "score": score,
        "direction": direction,
        "realized_net_pnl": realized,
    })
    count = min(int(bins), int(work["score"].nunique()), len(work))
    if count < 2:
        return {"bins": [], "spearman": None, "top_above_bottom": None}
    work["score_decile"] = pd.qcut(
        work["score"].rank(method="first"), count, labels=False, duplicates="drop")
    grouped = work.groupby("score_decile", observed=True).agg(
        candidates=("realized_net_pnl", "size"),
        score_min=("score", "min"),
        score_max=("score", "max"),
        score_mean=("score", "mean"),
        expectancy=("realized_net_pnl", "mean"),
        net_pnl=("realized_net_pnl", "sum"),
        positive_rate=("realized_net_pnl", lambda x: float((x > 0).mean())),
        long_share=("direction", lambda x: float((x == "LONG").mean())),
    ).reset_index()
    expectancy = grouped["expectancy"].to_numpy(dtype=float)
    rank = np.arange(len(expectancy), dtype=float)
    spearman = (
        float(np.corrcoef(rank, expectancy)[0, 1])
        if len(expectancy) > 1 and float(expectancy.std()) > 0.0
        else None
    )
    return {
        "bins": grouped.to_dict(orient="records"),
        "spearman": spearman,
        "top_above_bottom": bool(expectancy[-1] > expectancy[0]),
        "nondecreasing_step_ratio": float((np.diff(expectancy) >= 0).mean()),
    }


def selective_entry_report(
    candidates: pd.DataFrame,
    pred_long: np.ndarray,
    pred_short: np.ndarray,
    *,
    max_entries_per_day: int,
    initial_balance: float,
    fractions: Iterable[float] = (0.05, 0.10, 0.20),
) -> dict:
    best = np.maximum(np.asarray(pred_long, dtype=float), np.asarray(pred_short, dtype=float))
    report = {}
    for fraction in fractions:
        if not len(best):
            threshold = 0.0
        else:
            threshold = float(np.quantile(best, max(0.0, 1.0 - float(fraction))))
        actions, scores = predict_actions(pred_long, pred_short, threshold)
        trades = simulate_strategy(
            candidates, actions, scores, max_entries_per_day=max_entries_per_day)
        report[str(float(fraction))] = {
            "threshold": threshold,
            "candidate_coverage": float((best > threshold).mean()) if len(best) else 0.0,
            "metrics": strategy_metrics(
                trades,
                all_days=candidates["trading_day"].unique(),
                initial_balance=initial_balance,
            ),
        }
    return report


def bottom_selective_entry_report(
    candidates: pd.DataFrame,
    pred_long: np.ndarray,
    pred_short: np.ndarray,
    *,
    max_entries_per_day: int,
    initial_balance: float,
    fractions: Iterable[float] = (0.05, 0.10, 0.20),
) -> dict:
    direction, best, _ = _realized_predicted_direction_pnl(candidates, pred_long, pred_short)
    inverted = -best
    report = {}
    for fraction in fractions:
        if not len(best):
            threshold = 0.0
        else:
            threshold = float(np.quantile(inverted, max(0.0, 1.0 - float(fraction))))
        actions, scores = apply_direction_score_threshold(direction, inverted, threshold)
        trades = simulate_strategy(
            candidates, actions, scores, max_entries_per_day=max_entries_per_day)
        report[str(float(fraction))] = {
            "inverted_score_threshold": threshold,
            "original_score_max_threshold": float(-threshold),
            "candidate_coverage": float((inverted > threshold).mean()) if len(inverted) else 0.0,
            "metrics": strategy_metrics(
                trades,
                all_days=candidates["trading_day"].unique(),
                initial_balance=initial_balance,
            ),
        }
    return report


def threshold_curve_report(
    candidates: pd.DataFrame,
    pred_long: np.ndarray,
    pred_short: np.ndarray,
    quantiles: Iterable[float],
    *,
    max_entries_per_day: int,
    initial_balance: float,
) -> list[dict]:
    best = np.maximum(np.asarray(pred_long, dtype=float), np.asarray(pred_short, dtype=float))
    if not len(best):
        return []
    thresholds = sorted(set(
        [0.0, float(np.nextafter(best.max(), np.inf))]
        + [float(np.quantile(best, q)) for q in quantiles]
    ))
    rows = []
    for threshold in thresholds:
        actions, scores = predict_actions(pred_long, pred_short, threshold)
        trades = simulate_strategy(
            candidates, actions, scores, max_entries_per_day=max_entries_per_day)
        rows.append({
            "threshold": threshold,
            "candidate_coverage": float((best > threshold).mean()),
            "metrics": strategy_metrics(
                trades,
                all_days=candidates["trading_day"].unique(),
                initial_balance=initial_balance,
            ),
        })
    return rows


def score_decile_executable_report(
    candidates: pd.DataFrame,
    pred_long: np.ndarray,
    pred_short: np.ndarray,
    *,
    max_entries_per_day: int,
    initial_balance: float,
    bins: int = 10,
) -> dict:
    if len(candidates) < 2:
        return {"bins": [], "spearman": None, "top_above_bottom": None}
    direction, score, realized = _realized_predicted_direction_pnl(
        candidates, pred_long, pred_short)
    work = candidates.copy()
    work["pred_best_direction"] = direction
    work["pred_score"] = score
    work["realized_pred_direction_net_pnl"] = realized
    count = min(int(bins), int(pd.Series(score).nunique()), len(work))
    if count < 2:
        return {"bins": [], "spearman": None, "top_above_bottom": None}
    work["score_decile"] = pd.qcut(
        pd.Series(score).rank(method="first"), count, labels=False, duplicates="drop")
    rows = []
    for decile, part in work.groupby("score_decile", observed=True):
        part = part.reset_index(drop=True)
        part_actions = part["pred_best_direction"].to_numpy(dtype=object)
        part_scores = part["pred_score"].to_numpy(dtype=float)
        trades = simulate_strategy(
            part, part_actions, part_scores, max_entries_per_day=max_entries_per_day)
        realized_part = part["realized_pred_direction_net_pnl"].to_numpy(dtype=float)
        rows.append({
            "decile": int(decile),
            "candidate_count": int(len(part)),
            "executable_strategy_trades": int(len(trades)),
            "mean_pred_q": float(part_scores.mean()),
            "realized_net_pnl": float(realized_part.sum()),
            "realized_expectancy": float(realized_part.mean()),
            "long_count": int((part["pred_best_direction"] == "LONG").sum()),
            "short_count": int((part["pred_best_direction"] == "SHORT").sum()),
            "executable_metrics": strategy_metrics(
                trades,
                all_days=part["trading_day"].unique(),
                initial_balance=initial_balance,
            ),
        })
    expectancy = np.asarray([row["realized_expectancy"] for row in rows], dtype=float)
    rank = np.arange(len(expectancy), dtype=float)
    spearman = (
        float(np.corrcoef(rank, expectancy)[0, 1])
        if len(expectancy) > 1 and float(expectancy.std()) > 0.0
        else None
    )
    return {
        "bins": rows,
        "spearman": spearman,
        "top_above_bottom": bool(expectancy[-1] > expectancy[0]),
        "nondecreasing_step_ratio": float((np.diff(expectancy) >= 0).mean()),
    }


def grouped_strategy_report(
    trades: pd.DataFrame,
    *,
    all_days: Iterable[int],
    initial_balance: float,
) -> dict:
    if trades.empty:
        return {"by_fold": {}, "by_month": {}, "by_contract": {}, "by_direction": {}, "by_session_phase": {}}

    def groupby_metrics(column: str) -> dict:
        if column not in trades.columns:
            return {}
        out = {}
        for key, part in trades.groupby(column, dropna=False):
            out[str(key)] = strategy_metrics(
                part,
                all_days=part["trading_day"].unique(),
                initial_balance=initial_balance,
            )
        return out

    return {
        "by_fold": groupby_metrics("fold"),
        "by_month": groupby_metrics("month"),
        "by_contract": groupby_metrics("contract"),
        "by_direction": groupby_metrics("direction"),
        "by_session_phase": groupby_metrics("session_phase"),
    }


def action_mapping_audit() -> dict:
    try:
        from gym_trading_env.envs.action import TargetPos
        mapping = {
            "SHORT": int(TargetPos.SHORT),
            "FLAT": int(TargetPos.FLAT),
            "LONG": int(TargetPos.LONG),
        }
        status = "PASS" if mapping == {"SHORT": 0, "FLAT": 1, "LONG": 2} else "FAIL"
    except Exception as exc:  # pragma: no cover - defensive only.
        mapping = {}
        status = f"ERROR: {type(exc).__name__}: {exc}"
    return {
        "status": status,
        "target_pos_mapping": mapping,
        "expected_order": ["SHORT", "FLAT", "LONG"],
    }


def prediction_alignment_audit(
    predictions: pd.DataFrame,
    selected_trades: pd.DataFrame,
) -> dict:
    if predictions.empty:
        return {"status": "EMPTY"}
    pred = predictions.copy()
    pred_long = pred["pred_long_net_pnl"].to_numpy(dtype=float)
    pred_short = pred["pred_short_net_pnl"].to_numpy(dtype=float)
    expected_direction = np.where(pred_long >= pred_short, "LONG", "SHORT")
    expected_score = np.maximum(pred_long, pred_short)
    expected_realized = np.where(
        expected_direction == "LONG",
        pred["long_net_pnl"].to_numpy(dtype=float),
        pred["short_net_pnl"].to_numpy(dtype=float),
    )
    expected_action = expected_direction.astype(object)
    expected_action[expected_score <= pred["threshold"].to_numpy(dtype=float)] = "FLAT"

    def corr(x: pd.Series, y: pd.Series) -> float | None:
        if len(x) < 2:
            return None
        xv = x.to_numpy(dtype=float)
        yv = y.to_numpy(dtype=float)
        if float(np.std(xv)) < 1e-12 or float(np.std(yv)) < 1e-12:
            return None
        return float(np.corrcoef(xv, yv)[0, 1])

    selected = selected_trades.merge(
        pred[[
            "candidate_id", "pred_long_net_pnl", "pred_short_net_pnl",
            "pred_score", "long_net_pnl", "short_net_pnl",
        ]],
        on="candidate_id",
        how="left",
        suffixes=("", "_pred"),
    ) if not selected_trades.empty else pd.DataFrame()

    selected_report = {}
    for direction in ("LONG", "SHORT"):
        part = selected[selected["direction"] == direction] if not selected.empty else pd.DataFrame()
        pred_col = "pred_long_net_pnl" if direction == "LONG" else "pred_short_net_pnl"
        label_col = "long_net_pnl" if direction == "LONG" else "short_net_pnl"
        selected_report[direction] = {
            "count": int(len(part)),
            "mean_pred_q": None if part.empty else float(part[pred_col].mean()),
            "mean_realized_net_pnl": None if part.empty else float(part[label_col].mean()),
            "pred_realized_corr": None if part.empty else corr(part[pred_col], part[label_col]),
            "strategy_net_matches_label_mismatches": (
                0 if part.empty else int(
                    (~np.isclose(
                        part["net_pnl"].to_numpy(dtype=float),
                        part[label_col].to_numpy(dtype=float),
                        rtol=0,
                        atol=1e-9,
                    )).sum()
                )
            ),
        }

    mismatches = {
        "best_direction_mismatch_count": int(
            (pred["pred_best_direction"].to_numpy(dtype=object) != expected_direction).sum()),
        "score_mismatch_count": int(
            (~np.isclose(pred["pred_score"].to_numpy(dtype=float), expected_score, rtol=0, atol=1e-9)).sum()),
        "realized_direction_label_mismatch_count": int(
            (~np.isclose(
                pred["realized_pred_direction_net_pnl"].to_numpy(dtype=float),
                expected_realized,
                rtol=0,
                atol=1e-9,
            )).sum()),
        "threshold_action_mismatch_count": int(
            (pred["pred_action"].to_numpy(dtype=object) != expected_action).sum()),
    }
    status = "PASS" if all(value == 0 for value in mismatches.values()) else "FAIL"
    return {
        "status": status,
        "action_mapping": action_mapping_audit(),
        **mismatches,
        "threshold_rule": "pred_action = FLAT when max(pred_q_long, pred_q_short) <= threshold",
        "selected_trade_alignment": selected_report,
    }


def classifier_diagnostics(
    candidates: pd.DataFrame,
    long_prob: np.ndarray,
    short_prob: np.ndarray,
) -> dict:
    eps = 1e-12
    report = {}
    for direction, prob in (("long", long_prob), ("short", short_prob)):
        y = (candidates[f"{direction}_net_pnl"].to_numpy(dtype=float) > 0.0).astype(float)
        p = np.clip(np.asarray(prob, dtype=float), eps, 1.0 - eps)
        pred = p >= 0.5
        report[direction] = {
            "positive_rate": float(y.mean()) if len(y) else None,
            "prob_mean": float(p.mean()) if len(p) else None,
            "accuracy_at_0_5": float((pred == y.astype(bool)).mean()) if len(y) else None,
            "brier": float(np.mean((p - y) ** 2)) if len(y) else None,
            "logloss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))) if len(y) else None,
        }
    return report


def split_oracle_report(
    candidates: pd.DataFrame,
    folds: list[dict],
    *,
    max_entries_per_day: int,
    initial_balance: float,
    embargo_bars: int,
) -> dict:
    report = {}
    for fold in folds:
        masks, purge_audit = purged_fold_masks(
            candidates, fold, embargo_bars=embargo_bars)
        fold_report = {"purge_audit": purge_audit, "splits": {}}
        for split in ("train", "validation", "test"):
            part = candidates.loc[masks[split]].reset_index(drop=True)
            oracle = constrained_oracle(part, max_entries_per_day=max_entries_per_day)
            split_days = part["trading_day"].unique()
            by_direction = {}
            if not oracle.empty:
                for direction, trades in oracle.groupby("direction"):
                    by_direction[str(direction)] = strategy_metrics(
                        trades,
                        all_days=trades["trading_day"].unique(),
                        initial_balance=initial_balance,
                    )
            fold_report["splits"][split] = {
                "candidate_count": int(len(part)),
                "oracle": strategy_metrics(
                    oracle,
                    all_days=split_days,
                    initial_balance=initial_balance,
                ),
                "by_direction": by_direction,
            }
        report[fold["name"]] = fold_report
    return report


def make_walk_forward_folds(days: Iterable[int]) -> list[dict]:
    days = np.asarray(sorted(set(int(x) for x in days)), dtype=np.int64)
    blocks = [x for x in np.array_split(days, 7) if len(x)]
    if len(blocks) < 6:
        raise ValueError("at least 6 non-empty day blocks are required for 3 folds")
    folds = []
    for i in range(3):
        folds.append({
            "name": f"fold_{i + 1}",
            "train_days": np.concatenate(blocks[:2 + i]).tolist(),
            "validation_days": blocks[2 + i].tolist(),
            "test_days": blocks[3 + i].tolist(),
        })
    return folds


def purged_fold_masks(
    candidates: pd.DataFrame,
    fold: dict,
    *,
    embargo_bars: int,
) -> tuple[dict[str, np.ndarray], dict]:
    decision = candidates["decision_row"].to_numpy(dtype=np.int64)
    max_exit = candidates[["long_exit_row", "short_exit_row"]].max(axis=1).to_numpy(dtype=np.int64)
    day = candidates["trading_day"]
    masks = {
        "train": day.isin(fold["train_days"]).to_numpy(),
        "validation": day.isin(fold["validation_days"]).to_numpy(),
        "test": day.isin(fold["test_days"]).to_numpy(),
    }
    audit = {"embargo_bars": int(embargo_bars), "boundaries": []}
    for earlier, later in (("train", "validation"), ("validation", "test")):
        later_rows = decision[masks[later]]
        if not later_rows.size:
            raise ValueError(f"{fold['name']} has no {later} candidates")
        boundary = int(later_rows.min())
        crossing = masks[earlier] & (max_exit >= boundary)
        embargoed = masks[earlier] & (decision >= boundary - int(embargo_bars))
        removed = crossing | embargoed
        masks[earlier] = masks[earlier] & ~removed
        audit["boundaries"].append({
            "earlier": earlier,
            "later": later,
            "boundary_row": boundary,
            "crossing_labels_removed": int(crossing.sum()),
            "embargo_rows_removed": int((embargoed & ~crossing).sum()),
        })
        if np.any(masks[earlier] & (max_exit >= boundary)):
            raise AssertionError("purge failed: earlier outcome crosses split boundary")
    audit["counts"] = {name: int(mask.sum()) for name, mask in masks.items()}
    if any(count == 0 for count in audit["counts"].values()):
        raise ValueError(f"purge/embargo emptied a split: {audit['counts']}")
    return masks, audit


def _feature_indices(names: list[str], variant: str) -> np.ndarray:
    time_tokens = (
        "obs_minute_index_t", "obs_session_phase_t",
        "obs_weekday_sin_t", "obs_weekday_cos_t",
    )
    is_time = np.asarray([any(token in name for token in time_tokens) for name in names])
    if variant == "full":
        return np.arange(len(names), dtype=np.int64)
    if variant == "no_time":
        return np.flatnonzero(~is_time)
    if variant == "time_only":
        return np.flatnonzero(is_time)
    raise ValueError(f"unknown feature variant: {variant}")


def _choose_threshold(
    candidates: pd.DataFrame,
    pred_long: np.ndarray,
    pred_short: np.ndarray,
    quantiles: Iterable[float],
    max_entries_per_day: int,
) -> tuple[float, dict]:
    best = np.maximum(pred_long, pred_short)
    all_flat = float(np.nextafter(best.max(), np.inf)) if len(best) else 0.0
    options = sorted(set(
        [0.0, all_flat] + [float(np.quantile(best, q)) for q in quantiles]))
    results = []
    for threshold in options:
        actions, scores = predict_actions(pred_long, pred_short, threshold)
        trades = simulate_strategy(
            candidates, actions, scores, max_entries_per_day=max_entries_per_day)
        metrics = strategy_metrics(
            trades,
            all_days=candidates["trading_day"].unique(),
            initial_balance=1.0,
        )
        results.append({"threshold": threshold, "metrics": metrics})
    chosen = max(results, key=lambda x: (
        x["metrics"]["net_pnl"],
        -x["metrics"]["trades"],
    ))
    return float(chosen["threshold"]), {"options": results, "chosen": chosen}


def _choose_direction_score_threshold(
    candidates: pd.DataFrame,
    directions: Iterable[str],
    scores: Iterable[float],
    quantiles: Iterable[float],
    max_entries_per_day: int,
) -> tuple[float, dict]:
    score_arr = np.asarray(list(scores), dtype=float)
    directions_arr = np.asarray(list(directions), dtype=object)
    all_flat = float(np.nextafter(score_arr.max(), np.inf)) if len(score_arr) else 0.0
    options = sorted(set(
        [0.0, all_flat] + [float(np.quantile(score_arr, q)) for q in quantiles]))
    results = []
    for threshold in options:
        actions, strategy_scores = apply_direction_score_threshold(
            directions_arr, score_arr, threshold)
        trades = simulate_strategy(
            candidates, actions, strategy_scores, max_entries_per_day=max_entries_per_day)
        metrics = strategy_metrics(
            trades,
            all_days=candidates["trading_day"].unique(),
            initial_balance=1.0,
        )
        results.append({"threshold": threshold, "metrics": metrics})
    chosen = max(results, key=lambda x: (
        x["metrics"]["net_pnl"],
        -x["metrics"]["trades"],
    ))
    return float(chosen["threshold"]), {"options": results, "chosen": chosen}


def score_expectancy_report(trades: pd.DataFrame, bins: int = 10) -> dict:
    if len(trades) < 2:
        return {"bins": [], "spearman": None, "top_above_bottom": None}
    work = trades[["score", "net_pnl"]].copy()
    count = min(int(bins), int(work["score"].nunique()), len(work))
    if count < 2:
        return {"bins": [], "spearman": None, "top_above_bottom": None}
    work["score_bin"] = pd.qcut(
        work["score"].rank(method="first"), count, labels=False, duplicates="drop")
    grouped = work.groupby("score_bin", observed=True).agg(
        trades=("net_pnl", "size"),
        score_mean=("score", "mean"),
        expectancy=("net_pnl", "mean"),
        net_pnl=("net_pnl", "sum"),
    ).reset_index()
    expectancy = grouped["expectancy"].to_numpy(dtype=float)
    rank = np.arange(len(expectancy), dtype=float)
    spearman = (
        float(np.corrcoef(rank, expectancy)[0, 1])
        if len(expectancy) > 1 and float(expectancy.std()) > 0.0
        else None
    )
    return {
        "bins": grouped.to_dict(orient="records"),
        "spearman": spearman,
        "top_above_bottom": bool(expectancy[-1] > expectancy[0]),
        "nondecreasing_step_ratio": float((np.diff(expectancy) >= 0).mean()),
    }


def daily_bootstrap_expectancy(
    trades: pd.DataFrame,
    *,
    all_days: Iterable[int],
    runs: int,
    seed: int,
) -> dict:
    days = np.asarray(list(all_days), dtype=np.int64)
    if not len(days) or trades.empty:
        return {"runs": int(runs), "mean": None, "lower_95": None, "upper_95": None}
    grouped = trades.groupby("trading_day")["net_pnl"].agg(["sum", "size"]).reindex(
        days, fill_value=0.0)
    pnl = grouped["sum"].to_numpy(dtype=float)
    count = grouped["size"].to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(runs), dtype=float)
    for run in range(int(runs)):
        sampled = rng.integers(0, len(days), size=len(days))
        sampled_count = float(count[sampled].sum())
        values[run] = (
            float(pnl[sampled].sum() / sampled_count)
            if sampled_count > 0 else 0.0
        )
    return {
        "runs": int(runs),
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def cost_stress_report(
    trades: pd.DataFrame,
    *,
    all_days: Iterable[int],
    initial_balance: float,
    multipliers: Iterable[float] = (1.0, 1.5, 2.0),
) -> dict:
    report = {}
    for multiplier in multipliers:
        stressed = trades.copy()
        if not stressed.empty:
            stressed["net_pnl"] = (
                stressed["gross_pnl"]
                - float(multiplier) * (stressed["spread_cost"] + stressed["fee_cost"])
            )
        report[str(float(multiplier))] = strategy_metrics(
            stressed, all_days=all_days, initial_balance=initial_balance)
    return report


@dataclass(frozen=True)
class ConstantModel:
    value: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), float(self.value), dtype=np.float64)


def _require_xgboost():
    try:
        import xgboost as xgb
    except Exception as exc:  # pragma: no cover - exercised when dependency is absent.
        raise RuntimeError(
            "XGBoost is required for the boosted Q baseline. Install xgboost "
            "in the active training environment."
        ) from exc
    return xgb


def fit_xgb_regressor(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    num_boost_round: int = 120,
):
    y = np.asarray(y, dtype=np.float64)
    if len(y) == 0:
        raise ValueError("cannot train XGBoost regressor with no samples")
    if float(np.nanstd(y)) < 1e-12:
        return ConstantModel(float(np.nanmean(y)))
    xgb = _require_xgboost()
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_depth": 3,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 20,
        "lambda": 5.0,
        "alpha": 0.0,
        "seed": int(seed),
        "nthread": 1,
        "verbosity": 0,
    }
    dtrain = xgb.DMatrix(np.asarray(X, dtype=np.float32), label=y)
    booster = xgb.train(params, dtrain, num_boost_round=int(num_boost_round))

    class BoosterModel:
        def predict(self, X_new: np.ndarray) -> np.ndarray:
            dtest = xgb.DMatrix(np.asarray(X_new, dtype=np.float32))
            return np.asarray(booster.predict(dtest), dtype=np.float64)

    return BoosterModel()


def fit_xgb_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    num_boost_round: int = 80,
):
    y = np.asarray(y, dtype=np.float64)
    if len(y) == 0:
        raise ValueError("cannot train XGBoost classifier with no samples")
    if np.unique(y).size < 2:
        return ConstantModel(float(y[0]))
    xgb = _require_xgboost()
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_depth": 3,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 20,
        "lambda": 5.0,
        "alpha": 0.0,
        "seed": int(seed),
        "nthread": 1,
        "verbosity": 0,
    }
    dtrain = xgb.DMatrix(np.asarray(X, dtype=np.float32), label=y)
    booster = xgb.train(params, dtrain, num_boost_round=int(num_boost_round))

    class BoosterModel:
        def predict(self, X_new: np.ndarray) -> np.ndarray:
            dtest = xgb.DMatrix(np.asarray(X_new, dtype=np.float32))
            return np.asarray(booster.predict(dtest), dtype=np.float64)

    return BoosterModel()


def run_ridge_walk_forward(
    candidates: pd.DataFrame,
    X: np.ndarray,
    feature_names: list[str],
    folds: list[dict],
    *,
    alpha: float,
    threshold_quantiles: Iterable[float],
    max_entries_per_day: int,
    initial_balance: float,
    embargo_bars: int,
    bootstrap_runs: int,
    random_seed: int,
    cost_multipliers: Iterable[float],
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    report = {
        "model": "ridge_window_summary_v1",
        "feature_transform": "latest_mean_std_delta_of_market_seq",
        "folds": [],
    }
    all_test_trades: dict[str, list[pd.DataFrame]] = {
        "full": [], "no_time": [], "time_only": []}
    all_predictions = []
    for fold in folds:
        masks, purge_audit = purged_fold_masks(
            candidates, fold, embargo_bars=embargo_bars)
        fold_report = {
            "name": fold["name"],
            "purge_audit": purge_audit,
            "variants": {},
        }
        for variant in ("full", "no_time", "time_only"):
            cols = _feature_indices(feature_names, variant)
            train = masks["train"]
            val = masks["validation"]
            test = masks["test"]

            long_model = fit_ridge(X[train][:, cols], candidates.loc[train, "long_net_pnl"], alpha)
            short_model = fit_ridge(X[train][:, cols], candidates.loc[train, "short_net_pnl"], alpha)
            val_long = long_model.predict(X[val][:, cols])
            val_short = short_model.predict(X[val][:, cols])
            threshold, threshold_report = _choose_threshold(
                candidates.loc[val].reset_index(drop=True),
                val_long,
                val_short,
                threshold_quantiles,
                max_entries_per_day,
            )

            test_long = long_model.predict(X[test][:, cols])
            test_short = short_model.predict(X[test][:, cols])
            actions, scores = predict_actions(test_long, test_short, threshold)
            test_candidates = candidates.loc[test].reset_index(drop=True)
            predictions = test_candidates[[
                "candidate_id", "decision_row", "decision_timestamp", "trading_day",
                "long_net_pnl", "short_net_pnl",
            ]].copy()
            predictions["fold"] = fold["name"]
            predictions["variant"] = variant
            predictions["pred_long_net_pnl"] = test_long
            predictions["pred_short_net_pnl"] = test_short
            predictions["pred_action"] = actions
            predictions["pred_score"] = scores
            predictions["threshold"] = threshold
            all_predictions.append(predictions)
            trades = simulate_strategy(
                test_candidates, actions, scores,
                max_entries_per_day=max_entries_per_day)
            metrics = strategy_metrics(
                trades,
                all_days=fold["test_days"],
                initial_balance=initial_balance,
            )
            fold_report["variants"][variant] = {
                "feature_count": int(len(cols)),
                "threshold": threshold,
                "validation": threshold_report,
                "test": metrics,
                "score_expectancy": score_expectancy_report(trades),
            }
            if not trades.empty:
                trades = trades.copy()
                trades["fold"] = fold["name"]
                trades["variant"] = variant
                all_test_trades[variant].append(trades)
        report["folds"].append(fold_report)

    test_days = np.concatenate([f["test_days"] for f in folds])
    merged_by_variant = {
        variant: (
            pd.concat(parts, ignore_index=True)
            if parts else pd.DataFrame()
        )
        for variant, parts in all_test_trades.items()
    }
    report["aggregate_variants"] = {}
    for variant, trades in merged_by_variant.items():
        report["aggregate_variants"][variant] = {
            "metrics": strategy_metrics(
                trades, all_days=test_days, initial_balance=initial_balance),
            "score_expectancy": score_expectancy_report(trades),
        }
    merged = merged_by_variant["full"]
    report["aggregate_full"] = report["aggregate_variants"]["full"]["metrics"]
    report["aggregate_full"]["daily_bootstrap_expectancy"] = daily_bootstrap_expectancy(
        merged, all_days=test_days, runs=bootstrap_runs, seed=random_seed)
    report["aggregate_full"]["cost_stress"] = cost_stress_report(
        merged,
        all_days=test_days,
        initial_balance=initial_balance,
        multipliers=cost_multipliers,
    )
    report["gate2_status"] = classify_gate2(report)
    predictions = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions else pd.DataFrame())
    return report, merged, predictions


def run_xgboost_walk_forward(
    candidates: pd.DataFrame,
    X: np.ndarray,
    feature_names: list[str],
    folds: list[dict],
    *,
    threshold_quantiles: Iterable[float],
    max_entries_per_day: int,
    initial_balance: float,
    embargo_bars: int,
    bootstrap_runs: int,
    random_seed: int,
    cost_multipliers: Iterable[float],
    random_baseline_runs: int,
    feature_encoding: str = "window_summary",
    variants: Iterable[str] = ("full", "no_time", "time_only"),
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    _require_xgboost()
    variants = tuple(variants)
    report = {
        "model": "xgboost_q_regression_v1",
        "feature_transform": feature_encoding,
        "decision_rule": "argmax(pred_q_long, pred_q_short) with validation-selected threshold",
        "classifier_role": "diagnostic_only",
        "alignment_audit_scope": "test predictions and selected executable trades",
        "folds": [],
    }
    all_test_trades: dict[str, list[pd.DataFrame]] = {variant: [] for variant in variants}
    all_test_candidates: dict[str, list[pd.DataFrame]] = {variant: [] for variant in variants}
    all_inverted_trades: dict[str, list[pd.DataFrame]] = {variant: [] for variant in variants}
    all_predictions = []

    for fold_idx, fold in enumerate(folds):
        masks, purge_audit = purged_fold_masks(
            candidates, fold, embargo_bars=embargo_bars)
        fold_report = {
            "name": fold["name"],
            "purge_audit": purge_audit,
            "variants": {},
        }
        for variant in variants:
            cols = _feature_indices(feature_names, variant)
            train = masks["train"]
            val = masks["validation"]
            test = masks["test"]
            seed = int(random_seed) + fold_idx * 100 + len(cols)

            X_train = X[train][:, cols]
            X_val = X[val][:, cols]
            X_test = X[test][:, cols]
            long_model = fit_xgb_regressor(
                X_train, candidates.loc[train, "long_net_pnl"], seed=seed)
            short_model = fit_xgb_regressor(
                X_train, candidates.loc[train, "short_net_pnl"], seed=seed + 1)
            long_cls = fit_xgb_classifier(
                X_train,
                (candidates.loc[train, "long_net_pnl"].to_numpy(dtype=float) > 0).astype(float),
                seed=seed + 2,
            )
            short_cls = fit_xgb_classifier(
                X_train,
                (candidates.loc[train, "short_net_pnl"].to_numpy(dtype=float) > 0).astype(float),
                seed=seed + 3,
            )

            val_long = long_model.predict(X_val)
            val_short = short_model.predict(X_val)
            val_candidates = candidates.loc[val].reset_index(drop=True)
            val_direction, val_score, _ = _realized_predicted_direction_pnl(
                val_candidates, val_long, val_short)
            threshold, threshold_report = _choose_threshold(
                val_candidates,
                val_long,
                val_short,
                threshold_quantiles,
                max_entries_per_day,
            )
            inverted_threshold, inverted_threshold_report = _choose_direction_score_threshold(
                val_candidates,
                val_direction,
                -val_score,
                threshold_quantiles,
                max_entries_per_day,
            )

            test_long = long_model.predict(X_test)
            test_short = short_model.predict(X_test)
            test_long_prob = long_cls.predict(X_test)
            test_short_prob = short_cls.predict(X_test)
            actions, scores = predict_actions(test_long, test_short, threshold)
            test_candidates = candidates.loc[test].reset_index(drop=True)
            direction, best_score, realized = _realized_predicted_direction_pnl(
                test_candidates, test_long, test_short)
            inverted_actions, inverted_scores = apply_direction_score_threshold(
                direction, -best_score, inverted_threshold)
            predictions = test_candidates[[
                "candidate_id", "contract", "month", "session_phase",
                "decision_row", "decision_timestamp", "trading_day",
                "long_net_pnl", "short_net_pnl",
            ]].copy()
            predictions["fold"] = fold["name"]
            predictions["variant"] = variant
            predictions["pred_long_net_pnl"] = test_long
            predictions["pred_short_net_pnl"] = test_short
            predictions["pred_long_positive_prob"] = test_long_prob
            predictions["pred_short_positive_prob"] = test_short_prob
            predictions["pred_best_direction"] = direction
            predictions["pred_action"] = actions
            predictions["pred_score"] = scores
            predictions["realized_pred_direction_net_pnl"] = realized
            predictions["threshold"] = threshold
            predictions["inverted_score"] = -best_score
            predictions["inverted_threshold"] = inverted_threshold
            all_predictions.append(predictions)

            diagnostic_candidates = test_candidates.copy()
            diagnostic_candidates["fold"] = fold["name"]
            diagnostic_candidates["variant"] = variant
            diagnostic_candidates["pred_long_net_pnl"] = test_long
            diagnostic_candidates["pred_short_net_pnl"] = test_short
            diagnostic_candidates["pred_long_positive_prob"] = test_long_prob
            diagnostic_candidates["pred_short_positive_prob"] = test_short_prob
            diagnostic_candidates["pred_best_direction"] = direction
            diagnostic_candidates["pred_action"] = actions
            diagnostic_candidates["pred_score"] = scores
            diagnostic_candidates["realized_pred_direction_net_pnl"] = realized
            diagnostic_candidates["threshold"] = threshold
            diagnostic_candidates["inverted_score"] = -best_score
            diagnostic_candidates["inverted_threshold"] = inverted_threshold

            trades = simulate_strategy(
                test_candidates, actions, scores,
                max_entries_per_day=max_entries_per_day)
            inverted_trades = simulate_strategy(
                test_candidates,
                inverted_actions,
                inverted_scores,
                max_entries_per_day=max_entries_per_day,
            )
            metrics = strategy_metrics(
                trades,
                all_days=fold["test_days"],
                initial_balance=initial_balance,
            )
            fold_report["variants"][variant] = {
                "feature_count": int(len(cols)),
                "threshold": threshold,
                "validation": threshold_report,
                "validation_threshold_curve": threshold_report["options"],
                "test": metrics,
                "inverted_score_sanity": {
                    "debug_only": True,
                    "threshold": inverted_threshold,
                    "validation": inverted_threshold_report,
                    "test": strategy_metrics(
                        inverted_trades,
                        all_days=fold["test_days"],
                        initial_balance=initial_balance,
                    ),
                },
                "classifier_diagnostics": classifier_diagnostics(
                    test_candidates, test_long_prob, test_short_prob),
                "score_decile_expectancy": candidate_score_expectancy_report(
                    test_candidates, test_long, test_short),
                "score_decile_executable": score_decile_executable_report(
                    test_candidates,
                    test_long,
                    test_short,
                    max_entries_per_day=max_entries_per_day,
                    initial_balance=initial_balance,
                ),
                "selected_trade_score_expectancy": score_expectancy_report(trades),
                "top_selective_entry": selective_entry_report(
                    test_candidates,
                    test_long,
                    test_short,
                    max_entries_per_day=max_entries_per_day,
                    initial_balance=initial_balance,
                ),
                "bottom_selective_entry": bottom_selective_entry_report(
                    test_candidates,
                    test_long,
                    test_short,
                    max_entries_per_day=max_entries_per_day,
                    initial_balance=initial_balance,
                ),
                "test_threshold_curve": threshold_curve_report(
                    test_candidates,
                    test_long,
                    test_short,
                    threshold_quantiles,
                    max_entries_per_day=max_entries_per_day,
                    initial_balance=initial_balance,
                ),
            }
            if not trades.empty:
                trades = trades.copy()
                trades["fold"] = fold["name"]
                trades["variant"] = variant
                all_test_trades[variant].append(trades)
            if not inverted_trades.empty:
                inverted_trades = inverted_trades.copy()
                inverted_trades["fold"] = fold["name"]
                inverted_trades["variant"] = variant
                all_inverted_trades[variant].append(inverted_trades)
            all_test_candidates[variant].append(diagnostic_candidates)
        report["folds"].append(fold_report)

    test_days = np.concatenate([f["test_days"] for f in folds])
    merged_by_variant = {
        variant: (
            pd.concat(parts, ignore_index=True)
            if parts else pd.DataFrame()
        )
        for variant, parts in all_test_trades.items()
    }
    pred_by_variant = {
        variant: (
            pd.concat(parts, ignore_index=True)
            if parts else pd.DataFrame()
        )
        for variant, parts in all_test_candidates.items()
    }
    inverted_by_variant = {
        variant: (
            pd.concat(parts, ignore_index=True)
            if parts else pd.DataFrame()
        )
        for variant, parts in all_inverted_trades.items()
    }

    report["aggregate_variants"] = {}
    for variant, trades in merged_by_variant.items():
        preds = pred_by_variant[variant]
        report["aggregate_variants"][variant] = {
            "metrics": strategy_metrics(
                trades, all_days=test_days, initial_balance=initial_balance),
            "selected_trade_score_expectancy": score_expectancy_report(trades),
            "score_decile_expectancy": (
                _prediction_decile_report(preds) if not preds.empty
                else {"bins": [], "spearman": None, "top_above_bottom": None}
            ),
            "score_decile_executable": (
                score_decile_executable_report(
                    preds,
                    preds["pred_long_net_pnl"].to_numpy(dtype=float),
                    preds["pred_short_net_pnl"].to_numpy(dtype=float),
                    max_entries_per_day=max_entries_per_day,
                    initial_balance=initial_balance,
                ) if not preds.empty
                else {"bins": [], "spearman": None, "top_above_bottom": None}
            ),
            "inverted_score_sanity": {
                "debug_only": True,
                "metrics": strategy_metrics(
                    inverted_by_variant[variant],
                    all_days=test_days,
                    initial_balance=initial_balance,
                ),
            },
        }

    merged = merged_by_variant["full"]
    full_predictions = pred_by_variant["full"]
    report["aggregate_full"] = report["aggregate_variants"]["full"]["metrics"]
    report["aggregate_full"]["daily_bootstrap_expectancy"] = daily_bootstrap_expectancy(
        merged, all_days=test_days, runs=bootstrap_runs, seed=random_seed)
    report["aggregate_full"]["cost_stress"] = cost_stress_report(
        merged,
        all_days=test_days,
        initial_balance=initial_balance,
        multipliers=cost_multipliers,
    )
    if not full_predictions.empty:
        report["aggregate_full"]["top_selective_entry"] = selective_entry_report(
            full_predictions,
            full_predictions["pred_long_net_pnl"].to_numpy(dtype=float),
            full_predictions["pred_short_net_pnl"].to_numpy(dtype=float),
            max_entries_per_day=max_entries_per_day,
            initial_balance=initial_balance,
        )
        report["aggregate_full"]["threshold_curve"] = threshold_curve_report(
            full_predictions,
            full_predictions["pred_long_net_pnl"].to_numpy(dtype=float),
            full_predictions["pred_short_net_pnl"].to_numpy(dtype=float),
            threshold_quantiles,
            max_entries_per_day=max_entries_per_day,
            initial_balance=initial_balance,
        )
        report["aggregate_full"]["bottom_selective_entry"] = bottom_selective_entry_report(
            full_predictions,
            full_predictions["pred_long_net_pnl"].to_numpy(dtype=float),
            full_predictions["pred_short_net_pnl"].to_numpy(dtype=float),
            max_entries_per_day=max_entries_per_day,
            initial_balance=initial_balance,
        )
        report["aggregate_full"]["score_decile_executable"] = score_decile_executable_report(
            full_predictions,
            full_predictions["pred_long_net_pnl"].to_numpy(dtype=float),
            full_predictions["pred_short_net_pnl"].to_numpy(dtype=float),
            max_entries_per_day=max_entries_per_day,
            initial_balance=initial_balance,
        )
    else:
        report["aggregate_full"]["top_selective_entry"] = {}
        report["aggregate_full"]["threshold_curve"] = []
        report["aggregate_full"]["bottom_selective_entry"] = {}
        report["aggregate_full"]["score_decile_executable"] = {
            "bins": [], "spearman": None, "top_above_bottom": None}
    report["aggregate_full"]["matched_random"] = matched_random_summary(
        candidates[candidates["trading_day"].isin(test_days)].reset_index(drop=True),
        merged,
        max_entries_per_day=max_entries_per_day,
        runs=random_baseline_runs,
        seed=random_seed + 20000,
        all_days=test_days,
        initial_balance=initial_balance,
    )
    report["aggregate_full"]["grouped"] = grouped_strategy_report(
        merged, all_days=test_days, initial_balance=initial_balance)
    report["aggregate_full"]["inverted_score_sanity"] = {
        "debug_only": True,
        "metrics": strategy_metrics(
            inverted_by_variant["full"],
            all_days=test_days,
            initial_balance=initial_balance,
        ),
    }
    report["aggregate_full"]["alignment_audit"] = prediction_alignment_audit(
        full_predictions, merged)
    report["gate2_status"] = classify_gate2(report)

    predictions = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions else pd.DataFrame())
    return report, merged, predictions


def _prediction_decile_report(predictions: pd.DataFrame, bins: int = 10) -> dict:
    if predictions.empty:
        return {"bins": [], "spearman": None, "top_above_bottom": None}
    work = predictions[[
        "pred_score", "pred_best_direction", "realized_pred_direction_net_pnl"
    ]].copy()
    count = min(int(bins), int(work["pred_score"].nunique()), len(work))
    if count < 2:
        return {"bins": [], "spearman": None, "top_above_bottom": None}
    work["score_decile"] = pd.qcut(
        work["pred_score"].rank(method="first"), count, labels=False, duplicates="drop")
    grouped = work.groupby("score_decile", observed=True).agg(
        candidates=("realized_pred_direction_net_pnl", "size"),
        score_min=("pred_score", "min"),
        score_max=("pred_score", "max"),
        score_mean=("pred_score", "mean"),
        expectancy=("realized_pred_direction_net_pnl", "mean"),
        net_pnl=("realized_pred_direction_net_pnl", "sum"),
        positive_rate=("realized_pred_direction_net_pnl", lambda x: float((x > 0).mean())),
        long_share=("pred_best_direction", lambda x: float((x == "LONG").mean())),
    ).reset_index()
    expectancy = grouped["expectancy"].to_numpy(dtype=float)
    rank = np.arange(len(expectancy), dtype=float)
    spearman = (
        float(np.corrcoef(rank, expectancy)[0, 1])
        if len(expectancy) > 1 and float(expectancy.std()) > 0.0
        else None
    )
    return {
        "bins": grouped.to_dict(orient="records"),
        "spearman": spearman,
        "top_above_bottom": bool(expectancy[-1] > expectancy[0]),
        "nondecreasing_step_ratio": float((np.diff(expectancy) >= 0).mean()),
    }


def classify_gate2(report: dict) -> str:
    full = [fold["variants"]["full"]["test"] for fold in report["folds"]]
    positive_folds = sum(float(x["net_pnl"]) > 0 for x in full)
    aggregate = report["aggregate_full"]
    if float(aggregate["net_pnl"]) <= 0 or positive_folds < 2:
        return "FAIL"
    variants = report["aggregate_variants"]
    no_time = variants.get("no_time", {}).get("metrics")
    time_only = variants.get("time_only", {}).get("metrics")
    score = (
        variants["full"].get("score_decile_expectancy")
        or variants["full"].get("score_expectancy")
        or variants["full"].get("selected_trade_score_expectancy")
    )
    if no_time is not None and float(no_time["net_pnl"]) <= 0:
        return "FAIL"
    if time_only is not None and float(time_only["net_pnl"]) >= float(aggregate["net_pnl"]):
        return "FAIL"
    if score and (
        score["top_above_bottom"] is False
        or (score["spearman"] is not None and float(score["spearman"]) <= 0)
    ):
        return "FAIL"
    concentration = aggregate["max_trade_profit_share"]
    if concentration is not None and float(concentration) > 0.5:
        return "FAIL"
    bootstrap = aggregate["daily_bootstrap_expectancy"]
    if int(aggregate["trades"]) < 30 or bootstrap["lower_95"] is None:
        return "PROMISING"
    if float(bootstrap["lower_95"]) <= 0:
        return "PROMISING"
    return "PASS"
