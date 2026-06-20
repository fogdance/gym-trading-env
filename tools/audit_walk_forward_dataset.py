#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gym_trading_env.utils.session_futures_strict import strict_reindex_futures_345


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a walk-forward trading dataset contract root."
    )
    parser.add_argument("--contract-root", required=True, type=Path)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--roll-audit-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--doc-path", type=Path, required=True)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument(
        "--accept-main-continuous",
        action="store_true",
        help=(
            "Treat the raw CSV as user-confirmed main-continuous contract data. "
            "The report still records missing per-row symbol metadata and roll-risk windows."
        ),
    )
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")


def _load_csv(path: Path, tz: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"CSV must contain Date column: {path}")
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.sort_values("Date", kind="mergesort").drop_duplicates("Date", keep="last")
    df = df.set_index("Date")
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)
    return df


def _role_map(split_manifest: Path) -> tuple[dict[int, str], dict[str, list[int]]]:
    raw = _read_json(split_manifest)
    fold = raw[-1] if isinstance(raw, list) else raw
    roles: dict[int, str] = {}
    by_role: dict[str, list[int]] = {}
    for role in ("train", "validation", "test"):
        days = [int(x) for x in fold.get(f"{role}_days", [])]
        by_role[role] = days
        for day in days:
            roles[day] = role
    return roles, by_role


def _daily_from_csv(path: Path, tz: str) -> pd.DataFrame:
    raw = _load_csv(path, tz)
    aligned = strict_reindex_futures_345(raw, tz=tz)
    df = aligned["aligned"].copy()
    df["mask"] = aligned["mask"].to_numpy(dtype=int)
    df["trading_day"] = aligned["session_id"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for day, sub in df.groupby("trading_day", sort=True):
        valid = sub[sub["mask"] == 1]
        if valid.empty:
            continue
        first = valid.iloc[0]
        last = valid.iloc[-1]
        idx = valid.index
        rows.append({
            "trading_day": int(day),
            "first_timestamp": idx[0].isoformat(),
            "last_timestamp": idx[-1].isoformat(),
            "expected_rows": int(len(sub)),
            "valid_rows": int(len(valid)),
            "valid_ratio": float(len(valid) / max(len(sub), 1)),
            "open": float(first["Open"]),
            "high": float(valid["High"].max()),
            "low": float(valid["Low"].min()),
            "close": float(last["Close"]),
            "volume": float(valid["Volume"].sum()),
            "oi_first": float(first["OpenInterest"]),
            "oi_last": float(last["OpenInterest"]),
            "night_rows": int(((idx.hour >= 21) | (idx.hour < 3)).sum()),
            "day_rows": int(((idx.hour >= 9) & (idx.hour < 16)).sum()),
        })
    out = pd.DataFrame(rows).sort_values("trading_day", kind="mergesort").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"no valid trading days in {path}")
    return out


def _rolling_slope(values: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = float(np.sum(x * x)) or 1.0

    def calc(arr: np.ndarray) -> float:
        if len(arr) != window or np.any(~np.isfinite(arr)):
            return np.nan
        y = arr.astype(float)
        return float(np.sum((y - y.mean()) * x) / denom)

    return values.rolling(window, min_periods=window).apply(calc, raw=True)


def add_regimes(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    close = pd.Series(out["close"].to_numpy(dtype=float))
    high = pd.Series(out["high"].to_numpy(dtype=float))
    low = pd.Series(out["low"].to_numpy(dtype=float))
    log_close = np.log(close.clip(lower=1e-12))
    out["ret_20d"] = log_close.diff(20)
    out["ret_60d"] = log_close.diff(60)
    out["slope_20d"] = _rolling_slope(log_close, 20)
    out["slope_60d"] = _rolling_slope(log_close, 60)
    out["range_60d"] = (high.rolling(60, min_periods=20).max() - low.rolling(60, min_periods=20).min()) / close.clip(lower=1e-12)
    daily_ret = log_close.diff()
    out["vol_20d"] = daily_ret.rolling(20, min_periods=10).std()

    vol_q80 = float(out["vol_20d"].quantile(0.80))
    major = []
    for _, row in out.iterrows():
        ret60 = row["ret_60d"]
        vol20 = row["vol_20d"]
        if pd.isna(ret60):
            major.append("WARMUP")
        elif abs(float(ret60)) < 0.08 and pd.notna(vol20) and float(vol20) >= vol_q80:
            major.append("HIGH_VOL_RANGE")
        elif float(ret60) >= 0.08:
            major.append("UP")
        elif float(ret60) <= -0.08:
            major.append("DOWN")
        else:
            major.append("RANGE")
    out["major_regime"] = major

    high20 = high.rolling(20, min_periods=20).max()
    low20 = low.rolling(20, min_periods=20).min()
    prev_high20 = high20.shift(20)
    prev_low20 = low20.shift(20)
    structure = []
    for i in range(len(out)):
        if pd.isna(high20.iloc[i]) or pd.isna(prev_high20.iloc[i]):
            structure.append("WARMUP")
        elif high20.iloc[i] > prev_high20.iloc[i] and low20.iloc[i] > prev_low20.iloc[i]:
            structure.append("UP")
        elif high20.iloc[i] < prev_high20.iloc[i] and low20.iloc[i] < prev_low20.iloc[i]:
            structure.append("DOWN")
        else:
            structure.append("RANGE")
    out["structure_regime"] = structure
    return out


def _split_table(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for role, sub in daily.groupby("split_role", sort=False):
        rows.append({
            "split_role": role,
            "start_day": int(sub["trading_day"].min()),
            "end_day": int(sub["trading_day"].max()),
            "trading_days": int(len(sub)),
            "valid_rows": int(sub["valid_rows"].sum()),
            "expected_rows": int(sub["expected_rows"].sum()),
            "valid_ratio": float(sub["valid_rows"].sum() / max(sub["expected_rows"].sum(), 1)),
            "low_valid_ratio_days": int((sub["valid_ratio"] < 0.8).sum()),
            "total_return": float(np.log(float(sub["close"].iloc[-1]) / max(float(sub["open"].iloc[0]), 1e-12))),
            "night_rows": int(sub["night_rows"].sum()),
            "day_rows": int(sub["day_rows"].sum()),
        })
    order = {"train": 0, "validation": 1, "test": 2, "unknown": 9}
    return pd.DataFrame(rows).sort_values("split_role", key=lambda s: s.map(order).fillna(8)).reset_index(drop=True)


def _regime_distribution(daily: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for role, sub in daily.groupby("split_role", sort=False):
        for col in ("major_regime", "structure_regime"):
            counts = sub[col].value_counts(dropna=False)
            for regime, count in counts.items():
                records.append({
                    "split_role": role,
                    "regime_family": col,
                    "regime": str(regime),
                    "days": int(count),
                    "ratio": float(count / max(len(sub), 1)),
                })
    order = {"train": 0, "validation": 1, "test": 2, "unknown": 9}
    return pd.DataFrame(records).sort_values(
        ["split_role", "regime_family", "regime"],
        key=lambda s: s.map(order).fillna(s) if s.name == "split_role" else s,
    ).reset_index(drop=True)


def _roll_by_split(roll_dir: Path | None, roles: dict[int, str]) -> pd.DataFrame:
    if roll_dir is None:
        return pd.DataFrame()
    path = roll_dir / "roll_event_table.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or "next_trading_day" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["split_role"] = [roles.get(int(x), "unknown") for x in df["next_trading_day"]]
    keep = [
        "split_role",
        "prev_trading_day",
        "next_trading_day",
        "open_gap_pct",
        "oi_gap_pct",
        "volume_ratio",
        "roll_risk_score",
        "large_price_gap",
        "large_oi_gap",
        "large_volume_shift",
        "large_calendar_gap",
    ]
    return df[[c for c in keep if c in df.columns]]


def _write_report(path: Path, summary: dict[str, Any], split_table: pd.DataFrame, regime_dist: pd.DataFrame, roll_by_split: pd.DataFrame) -> None:
    if summary.get("main_continuous_confirmed"):
        data_scope = (
            "The raw CSV is user-confirmed as JM main-continuous contract data. The file itself has no "
            "contract symbol metadata, so conclusions must remain scoped to main-continuous JM and not to one "
            "verified single listed contract."
        )
        limitation = (
            "The source CSV has no symbol/contract column. The data owner has confirmed it is JM "
            "main-continuous contract data. Do not claim verified single-contract semantics. Future checkpoint "
            "reports must split PnL around roll-boundary candidate windows so we can see whether results depend "
            "on continuous-contract construction artifacts."
        )
    else:
        data_scope = (
            "It remains an anonymous-continuous JM experiment because the raw CSV has no contract symbol metadata."
        )
        limitation = (
            "The source CSV has no symbol/contract column. This audit accepts it as an anonymous continuous JM "
            "series only. Do not claim verified single-contract, verified dominant-contract, or verified roll-rule semantics."
        )
    lines = [
        "# JM8Y WF V1 Dataset Audit",
        "",
        "## Final Answer",
        "",
        f"Classification: `{summary['classification']}`.",
        "",
        "The `jm8y_wf_v1` dataset split is chronological and train-only data is separated from validation/test. "
        + data_scope,
        "",
        "## Split Table",
        "",
        "| role | start | end | days | valid_rows | valid_ratio | low_valid_days | total_log_return |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in split_table.iterrows():
        lines.append(
            f"| {row['split_role']} | {int(row['start_day'])} | {int(row['end_day'])} | "
            f"{int(row['trading_days'])} | {int(row['valid_rows'])} | {float(row['valid_ratio']):.4f} | "
            f"{int(row['low_valid_ratio_days'])} | {float(row['total_return']):.4f} |"
        )
    lines += [
        "",
        "## Regime Distribution",
        "",
        "| role | family | regime | days | ratio |",
        "|---|---|---|---:|---:|",
    ]
    for _, row in regime_dist.iterrows():
        lines.append(f"| {row['split_role']} | {row['regime_family']} | {row['regime']} | {int(row['days'])} | {float(row['ratio']):.4f} |")
    lines += [
        "",
        "## Roll-Risk Distribution",
        "",
    ]
    if roll_by_split.empty:
        lines.append("No roll-risk table was provided.")
    else:
        counts = roll_by_split["split_role"].value_counts().to_dict()
        for role in ("train", "validation", "test", "unknown"):
            if role in counts:
                lines.append(f"- {role}: {int(counts[role])} candidate roll-risk boundaries")
    lines += [
        "",
        "## Data Limitation",
        "",
        limitation,
        "",
        "## Outputs",
        "",
    ]
    for name, value in summary["outputs"].items():
        lines.append(f"- `{name}`: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = args.contract_root.resolve()
    split_dir = root / "artifacts" / "walk_forward_splits" / args.split_name
    summary_path = split_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    split_summary = _read_json(summary_path)
    split_manifest = Path(split_summary["split_manifest"])
    roles, by_role = _role_map(split_manifest)

    output_dir = args.output_dir or (root / "artifacts" / "dataset_audit" / args.split_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_daily = _daily_from_csv(Path(split_summary["raw_csv"]), args.timezone)
    full_daily["split_role"] = [roles.get(int(x), "unknown") for x in full_daily["trading_day"]]
    full_daily = add_regimes(full_daily)
    full_daily.to_csv(output_dir / "regime_day_table.csv", index=False)

    split_table = _split_table(full_daily[full_daily["split_role"].isin(["train", "validation", "test"])])
    split_table.to_csv(output_dir / "split_table.csv", index=False)
    regime_dist = _regime_distribution(full_daily[full_daily["split_role"].isin(["train", "validation", "test"])])
    regime_dist.to_csv(output_dir / "regime_distribution_by_split.csv", index=False)

    roll_by_split = _roll_by_split(args.roll_audit_dir, roles)
    if not roll_by_split.empty:
        roll_by_split.to_csv(output_dir / "roll_risk_by_split.csv", index=False)

    unknown_days = full_daily[full_daily["split_role"] == "unknown"]
    strict_train_days = set(full_daily.loc[full_daily["split_role"] == "train", "trading_day"].astype(int).tolist())
    manifest_train_days = set(by_role["train"])
    classification = "DATASET_SPLIT_READY_MAIN_CONTINUOUS" if args.accept_main_continuous else "DATASET_SPLIT_READY_ANONYMOUS_CONTINUOUS"
    if len(unknown_days):
        classification = "DATASET_SPLIT_UNKNOWN_DAYS"
    elif strict_train_days != manifest_train_days:
        classification = "DATASET_SPLIT_TRAIN_DAY_MISMATCH"

    summary = {
        "classification": classification,
        "contract_root": str(root),
        "split_name": args.split_name,
        "main_continuous_confirmed": bool(args.accept_main_continuous),
        "anonymous_continuous_assumption": bool(not args.accept_main_continuous),
        "split_manifest": str(split_manifest),
        "split_manifest_hash": split_summary["split_manifest_hash"],
        "raw_csv": split_summary["raw_csv"],
        "train_csv": split_summary["train_csv"],
        "roles": split_summary["roles"],
        "strict_checks": {
            "unknown_days": int(len(unknown_days)),
            "train_days_match_manifest": bool(strict_train_days == manifest_train_days),
            "train_intersects_validation": int(len(strict_train_days & set(by_role["validation"]))),
            "train_intersects_test": int(len(strict_train_days & set(by_role["test"]))),
        },
        "roll_risk": {
            "provided": bool(not roll_by_split.empty),
            "candidate_boundaries": int(len(roll_by_split)) if not roll_by_split.empty else 0,
            "by_split": {str(k): int(v) for k, v in roll_by_split["split_role"].value_counts().to_dict().items()} if not roll_by_split.empty else {},
        },
        "outputs": {
            "split_table": str(output_dir / "split_table.csv"),
            "regime_day_table": str(output_dir / "regime_day_table.csv"),
            "regime_distribution_by_split": str(output_dir / "regime_distribution_by_split.csv"),
            "roll_risk_by_split": str(output_dir / "roll_risk_by_split.csv") if not roll_by_split.empty else "",
            "summary": str(output_dir / "summary.json"),
            "report": str(args.doc_path),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    _write_report(args.doc_path, summary, split_table, regime_dist, roll_by_split)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
