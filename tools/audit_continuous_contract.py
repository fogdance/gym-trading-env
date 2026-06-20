#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from gym_trading_env.utils.session_futures_strict import strict_reindex_futures_345


CONTRACT_COLUMN_CANDIDATES = (
    "symbol",
    "contract",
    "instrument",
    "instrument_id",
    "vt_symbol",
    "code",
    "ticker",
)


@dataclass(frozen=True)
class AuditThresholds:
    price_gap_pct: float = 0.03
    oi_gap_pct: float = 0.20
    volume_ratio_hi: float = 4.0
    volume_ratio_lo: float = 0.25
    robust_z: float = 8.0
    low_valid_ratio: float = 0.80
    large_calendar_gap_days: int = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit continuous futures CSV contract/roll integrity."
    )
    parser.add_argument("--raw-csv", required=True, type=Path)
    parser.add_argument("--product", default="UNKNOWN")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--doc-path", required=True, type=Path)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument(
        "--accept-main-continuous",
        action="store_true",
        help=(
            "Treat a CSV without per-row symbol metadata as user-confirmed main-continuous "
            "contract data. Roll-risk candidates remain reportable risk points."
        ),
    )
    return parser.parse_args()


def _ensure_input(df: pd.DataFrame, *, tz: str) -> pd.DataFrame:
    required = {"Date", "Open", "High", "Low", "Close", "Volume", "OpenInterest"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"raw csv missing required columns: {missing}")
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="raise")
    out = out.sort_values("Date", kind="mergesort").drop_duplicates("Date", keep="last")
    out = out.set_index("Date")
    if out.index.tz is None:
        out.index = out.index.tz_localize(tz)
    else:
        out.index = out.index.tz_convert(tz)
    return out


def _robust_z(values: pd.Series) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = float(clean.median(skipna=True))
    mad = float((clean - med).abs().median(skipna=True))
    if not np.isfinite(mad) or mad <= 0:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return 0.6745 * (clean - med) / mad


def _safe_pct(num: pd.Series, den: pd.Series) -> pd.Series:
    den_abs = den.abs().replace(0.0, np.nan)
    return num / den_abs


def _session_phase_counts(index: pd.DatetimeIndex, valid: np.ndarray) -> dict[str, int]:
    idx = index[valid.astype(bool)]
    counts = {
        "night": int(((idx.hour >= 21) | (idx.hour < 3)).sum()),
        "day": int(((idx.hour >= 9) & (idx.hour < 16)).sum()),
        "other": 0,
    }
    counts["other"] = int(len(idx) - counts["night"] - counts["day"])
    return counts


def _find_contract_columns(columns: Iterable[str]) -> list[str]:
    found: list[str] = []
    lower_to_original = {c.lower(): c for c in columns}
    for name in CONTRACT_COLUMN_CANDIDATES:
        if name in lower_to_original:
            found.append(lower_to_original[name])
    return found


def build_daily_table(raw: pd.DataFrame, tz: str) -> tuple[pd.DataFrame, dict[str, int]]:
    aligned = strict_reindex_futures_345(raw, tz=tz)
    df = aligned["aligned"].copy()
    mask = aligned["mask"].astype(int)
    session_id = aligned["session_id"].astype(str)
    df["mask"] = mask.to_numpy(dtype=int)
    df["trading_day"] = session_id.to_numpy(dtype=str)

    rows: list[dict[str, object]] = []
    for sid, sub in df.groupby("trading_day", sort=True):
        valid_sub = sub[sub["mask"] == 1]
        if valid_sub.empty:
            continue
        first = valid_sub.iloc[0]
        last = valid_sub.iloc[-1]
        valid_bool = sub["mask"].to_numpy(dtype=bool)
        phase_counts = _session_phase_counts(sub.index, valid_bool)
        rows.append(
            {
                "trading_day": int(sid),
                "first_timestamp": valid_sub.index[0].isoformat(),
                "last_timestamp": valid_sub.index[-1].isoformat(),
                "expected_rows": int(len(sub)),
                "valid_rows": int(len(valid_sub)),
                "valid_ratio": float(len(valid_sub) / max(len(sub), 1)),
                "open_first": float(first["Open"]),
                "high": float(valid_sub["High"].max()),
                "low": float(valid_sub["Low"].min()),
                "close_last": float(last["Close"]),
                "volume_sum": float(valid_sub["Volume"].sum()),
                "oi_first": float(first["OpenInterest"]),
                "oi_last": float(last["OpenInterest"]),
                "oi_mean": float(valid_sub["OpenInterest"].mean()),
                "oi_min": float(valid_sub["OpenInterest"].min()),
                "oi_max": float(valid_sub["OpenInterest"].max()),
                "night_rows": phase_counts["night"],
                "day_rows": phase_counts["day"],
                "other_rows": phase_counts["other"],
            }
        )
    daily = pd.DataFrame(rows).sort_values("trading_day", kind="mergesort").reset_index(drop=True)
    phase_totals = {
        "night_rows": int(daily.get("night_rows", pd.Series(dtype=float)).sum()),
        "day_rows": int(daily.get("day_rows", pd.Series(dtype=float)).sum()),
        "other_rows": int(daily.get("other_rows", pd.Series(dtype=float)).sum()),
    }
    return daily, phase_totals


def build_pairwise_table(daily: pd.DataFrame, thresholds: AuditThresholds) -> pd.DataFrame:
    if len(daily) < 2:
        return pd.DataFrame()

    prev = daily.iloc[:-1].reset_index(drop=True).add_prefix("prev_")
    nxt = daily.iloc[1:].reset_index(drop=True).add_prefix("next_")
    pairs = pd.concat([prev, nxt], axis=1)
    prev_date = pd.to_datetime(pairs["prev_trading_day"].astype(str), format="%Y%m%d")
    next_date = pd.to_datetime(pairs["next_trading_day"].astype(str), format="%Y%m%d")
    pairs["calendar_gap_days"] = (next_date - prev_date).dt.days.astype(int)

    pairs["open_gap"] = pairs["next_open_first"] - pairs["prev_close_last"]
    pairs["open_gap_pct"] = _safe_pct(pairs["open_gap"], pairs["prev_close_last"])
    pairs["close_to_close_ret"] = _safe_pct(
        pairs["next_close_last"] - pairs["prev_close_last"],
        pairs["prev_close_last"],
    )
    pairs["oi_gap"] = pairs["next_oi_first"] - pairs["prev_oi_last"]
    pairs["oi_gap_pct"] = _safe_pct(pairs["oi_gap"], pairs["prev_oi_last"])
    pairs["volume_ratio"] = pairs["next_volume_sum"] / pairs["prev_volume_sum"].replace(0.0, np.nan)
    pairs["log_volume_ratio_abs"] = np.log(pairs["volume_ratio"].replace(0.0, np.nan)).abs()

    pairs["open_gap_robust_z"] = _robust_z(pairs["open_gap_pct"]).abs()
    pairs["oi_gap_robust_z"] = _robust_z(pairs["oi_gap_pct"]).abs()
    pairs["volume_ratio_robust_z"] = _robust_z(pairs["log_volume_ratio_abs"]).abs()

    pairs["large_price_gap"] = (
        pairs["open_gap_pct"].abs().ge(thresholds.price_gap_pct)
        | pairs["open_gap_robust_z"].ge(thresholds.robust_z)
    )
    pairs["large_oi_gap"] = (
        pairs["oi_gap_pct"].abs().ge(thresholds.oi_gap_pct)
        | pairs["oi_gap_robust_z"].ge(thresholds.robust_z)
    )
    pairs["large_volume_shift"] = (
        pairs["volume_ratio"].ge(thresholds.volume_ratio_hi)
        | pairs["volume_ratio"].le(thresholds.volume_ratio_lo)
        | pairs["volume_ratio_robust_z"].ge(thresholds.robust_z)
    )
    pairs["large_calendar_gap"] = pairs["calendar_gap_days"].gt(thresholds.large_calendar_gap_days)
    pairs["low_valid_ratio_next"] = pairs["next_valid_ratio"].lt(thresholds.low_valid_ratio)
    pairs["low_valid_ratio_prev"] = pairs["prev_valid_ratio"].lt(thresholds.low_valid_ratio)
    pairs["candidate_roll_risk"] = (
        pairs["large_oi_gap"]
        | (pairs["large_price_gap"] & pairs["large_volume_shift"])
        | (pairs["large_price_gap"] & pairs["large_calendar_gap"])
    )

    score = (
        pairs["open_gap_robust_z"].fillna(0.0)
        + pairs["oi_gap_robust_z"].fillna(0.0)
        + pairs["volume_ratio_robust_z"].fillna(0.0)
        + pairs["large_calendar_gap"].astype(float)
    )
    pairs["roll_risk_score"] = score
    return pairs


def build_report(
    *,
    raw_csv: Path,
    product: str,
    output_dir: Path,
    daily: pd.DataFrame,
    pairs: pd.DataFrame,
    contract_columns: list[str],
    thresholds: AuditThresholds,
    phase_totals: dict[str, int],
    accept_main_continuous: bool,
) -> tuple[dict[str, object], str]:
    flagged = pairs[pairs["candidate_roll_risk"]].copy() if not pairs.empty else pd.DataFrame()
    low_valid = daily[daily["valid_ratio"].lt(thresholds.low_valid_ratio)].copy()
    if contract_columns:
        classification = "ROLL_AUDIT_READY_FOR_REVIEW"
    elif accept_main_continuous:
        classification = "USER_CONFIRMED_MAIN_CONTINUOUS_CONTRACT"
    else:
        classification = "CONTRACT_METADATA_BLOCKED"

    summary = {
        "product": product,
        "raw_csv": str(raw_csv),
        "raw_contract_columns": contract_columns,
        "symbol_metadata_available": bool(contract_columns),
        "main_continuous_confirmed": bool(accept_main_continuous),
        "classification": classification,
        "trading_days": int(len(daily)),
        "date_start": int(daily["trading_day"].min()) if len(daily) else None,
        "date_end": int(daily["trading_day"].max()) if len(daily) else None,
        "candidate_roll_risk_days": int(len(flagged)),
        "low_valid_ratio_days": int(len(low_valid)),
        "max_abs_open_gap_pct": float(pairs["open_gap_pct"].abs().max()) if not pairs.empty else None,
        "max_abs_oi_gap_pct": float(pairs["oi_gap_pct"].abs().max()) if not pairs.empty else None,
        "max_volume_ratio": float(pairs["volume_ratio"].max()) if not pairs.empty else None,
        "min_volume_ratio": float(pairs["volume_ratio"].min()) if not pairs.empty else None,
        "thresholds": asdict(thresholds),
        "phase_totals": phase_totals,
        "outputs": {
            "daily_contract_audit": str(output_dir / "daily_contract_audit.csv"),
            "roll_gap_distribution": str(output_dir / "roll_gap_distribution.csv"),
            "roll_event_table": str(output_dir / "roll_event_table.csv"),
            "summary": str(output_dir / "summary.json"),
        },
    }

    top = flagged.sort_values("roll_risk_score", ascending=False).head(20)
    top_rows = []
    for _, row in top.iterrows():
        top_rows.append(
            "| {prev} -> {nxt} | {gap:.4f} | {oi:.4f} | {vol:.2f} | {score:.2f} | {flags} |".format(
                prev=int(row["prev_trading_day"]),
                nxt=int(row["next_trading_day"]),
                gap=float(row["open_gap_pct"]) if pd.notna(row["open_gap_pct"]) else float("nan"),
                oi=float(row["oi_gap_pct"]) if pd.notna(row["oi_gap_pct"]) else float("nan"),
                vol=float(row["volume_ratio"]) if pd.notna(row["volume_ratio"]) else float("nan"),
                score=float(row["roll_risk_score"]),
                flags=", ".join(
                    name
                    for name in (
                        "large_price_gap" if row["large_price_gap"] else "",
                        "large_oi_gap" if row["large_oi_gap"] else "",
                        "large_volume_shift" if row["large_volume_shift"] else "",
                        "large_calendar_gap" if row["large_calendar_gap"] else "",
                    )
                    if name
                ),
            )
        )

    symbol_line = (
        f"Contract metadata columns found: `{contract_columns}`."
        if contract_columns
        else "No symbol / contract metadata column is present in the raw CSV."
    )
    if contract_columns:
        final_answer = "ROLL_AUDIT_READY_FOR_REVIEW: symbol metadata exists; review flagged roll-risk rows before training."
        readiness = (
            "Review symbol/roll metadata and flagged roll-risk rows before using this dataset for formal training."
        )
        method_note = "Detected events are roll-risk candidates inferred from price / OI / volume behavior."
    elif accept_main_continuous:
        final_answer = (
            "USER_CONFIRMED_MAIN_CONTINUOUS_CONTRACT: the data is accepted as main-continuous "
            f"{product} contract data for this experiment."
        )
        readiness = (
            "Training is allowed as a main-continuous contract experiment. Do not describe results as one verified "
            "single listed contract, and report future PnL around roll-boundary candidate windows."
        )
        method_note = (
            "Detected events are continuous-contract roll-boundary risk or data-jump candidates inferred from "
            "price / OI / volume behavior. They are useful for reporting and attribution, not proof that the "
            "dataset is dirty or unusable."
        )
    else:
        final_answer = (
            "CONTRACT_METADATA_BLOCKED: the data can be profiled as an anonymous continuous series, "
            "but formal training is not ready until the continuous-contract/roll rule is documented or accepted."
        )
        readiness = (
            "Do not start formal training from this CSV until a symbol/roll metadata source is reviewed or the team "
            "explicitly accepts this file as main-continuous contract data."
        )
        method_note = "Detected events are roll-risk or data-jump candidates, not confirmed roll dates."
    report = f"""# {product} 8Y Continuous Contract / Roll Audit

## Final Answer

{final_answer}

{symbol_line}

This audit does not train Dreamer, does not change env behavior, and does not build a walk-forward split.

## Dataset

| item | value |
|---|---:|
| raw_csv | `{raw_csv}` |
| trading_days | {summary["trading_days"]} |
| date_start | {summary["date_start"]} |
| date_end | {summary["date_end"]} |
| candidate_roll_risk_days | {summary["candidate_roll_risk_days"]} |
| low_valid_ratio_days | {summary["low_valid_ratio_days"]} |
| max_abs_open_gap_pct | {summary["max_abs_open_gap_pct"]:.6f} |
| max_abs_oi_gap_pct | {summary["max_abs_oi_gap_pct"]:.6f} |
| max_volume_ratio | {summary["max_volume_ratio"]:.3f} |
| min_volume_ratio | {summary["min_volume_ratio"]:.3f} |

## Method

Trading days are inferred with the same `strict_reindex_futures_345` session logic used by the env.
The audit aggregates valid 1m rows by trading day, then compares consecutive trading days for:

- next-session open gap versus previous close;
- next first open interest versus previous last open interest;
- next daily volume versus previous daily volume;
- long calendar gaps and low valid-row coverage.

Because the CSV lacks a contract/symbol column, {method_note}

## Top Roll-Risk Candidates

| prev_day -> next_day | open_gap_pct | oi_gap_pct | volume_ratio | risk_score | flags |
|---|---:|---:|---:|---:|---|
{chr(10).join(top_rows) if top_rows else "| none | 0 | 0 | 0 | 0 | none |"}

## Outputs

- `{output_dir / "daily_contract_audit.csv"}`
- `{output_dir / "roll_gap_distribution.csv"}`
- `{output_dir / "roll_event_table.csv"}`
- `{output_dir / "summary.json"}`

## Training Readiness Impact

{readiness}
"""
    return summary, report


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    args.doc_path.parent.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.raw_csv)
    contract_columns = _find_contract_columns(raw_df.columns)
    raw = _ensure_input(raw_df, tz=args.timezone)
    thresholds = AuditThresholds()

    daily, phase_totals = build_daily_table(raw, tz=args.timezone)
    pairs = build_pairwise_table(daily, thresholds)
    flagged = pairs[pairs["candidate_roll_risk"]].copy() if not pairs.empty else pd.DataFrame()
    flagged = flagged.sort_values("roll_risk_score", ascending=False)

    daily.to_csv(output_dir / "daily_contract_audit.csv", index=False)
    pairs.to_csv(output_dir / "roll_gap_distribution.csv", index=False)
    flagged.to_csv(output_dir / "roll_event_table.csv", index=False)

    summary, report = build_report(
        raw_csv=args.raw_csv,
        product=args.product,
        output_dir=output_dir,
        daily=daily,
        pairs=pairs,
        contract_columns=contract_columns,
        thresholds=thresholds,
        phase_totals=phase_totals,
        accept_main_continuous=bool(args.accept_main_continuous),
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.doc_path.write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
