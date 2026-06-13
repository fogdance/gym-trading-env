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
        trades.append({
            "candidate_id": int(row.candidate_id),
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
        })
        active_exit_row = exit_row
        day_entries[day] = day_entries.get(day, 0) + 1
    return pd.DataFrame.from_records(trades)


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
        "profit_factor": (
            None if losses.size == 0 else float(wins.sum() / abs(losses.sum()))),
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


def classify_gate2(report: dict) -> str:
    full = [fold["variants"]["full"]["test"] for fold in report["folds"]]
    positive_folds = sum(float(x["net_pnl"]) > 0 for x in full)
    aggregate = report["aggregate_full"]
    if float(aggregate["net_pnl"]) <= 0 or positive_folds < 2:
        return "FAIL"
    variants = report["aggregate_variants"]
    no_time = variants["no_time"]["metrics"]
    time_only = variants["time_only"]["metrics"]
    score = variants["full"]["score_expectancy"]
    if float(no_time["net_pnl"]) <= 0:
        return "FAIL"
    if float(time_only["net_pnl"]) >= float(aggregate["net_pnl"]):
        return "FAIL"
    if score["top_above_bottom"] is False or (
        score["spearman"] is not None and float(score["spearman"]) <= 0
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
