from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .entry_analysis import add_diagnostic_context, make_walk_forward_folds
from .entry_evaluator import (
    EntryEvalConfig,
    build_flattened_entry_features,
    build_entry_dataset,
    build_evaluation_context,
    candidate_filter_audit,
    load_entry_eval_config,
    load_market_frames,
)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: str | Path, value) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default))


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def ensure_parquet_engine() -> None:
    if (
        importlib.util.find_spec("pyarrow") is None
        and importlib.util.find_spec("fastparquet") is None
    ):
        raise RuntimeError(
            "Parquet output requires pyarrow or fastparquet. "
            "Use an environment with pyarrow, for example /home/v/miniconda3/bin/python."
        )


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    if (
        importlib.util.find_spec("pyarrow") is not None
        or importlib.util.find_spec("fastparquet") is not None
    ):
        df.to_parquet(path, index=False)
        return

    csv = path.with_suffix(".csv")
    bridge_env = os.environ.get("PARQUET_BRIDGE_PYTHON", "")
    if not bridge_env:
        df.to_csv(csv, index=False)
        return
    bridge = Path(bridge_env)
    tmp = path.with_suffix(path.suffix + ".tmp.csv")
    df.to_csv(tmp, index=False)
    try:
        subprocess.run(
            [
                str(bridge),
                "-c",
                (
                    "import pandas as pd, sys; "
                    "pd.read_csv(sys.argv[1]).to_parquet(sys.argv[2], index=False)"
                ),
                str(tmp),
                str(path),
            ],
            check=True,
        )
    finally:
        tmp.unlink(missing_ok=True)


def read_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if (
        importlib.util.find_spec("pyarrow") is not None
        or importlib.util.find_spec("fastparquet") is not None
    ):
        return pd.read_parquet(path)

    csv = path.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv)
    bridge_env = os.environ.get("PARQUET_BRIDGE_PYTHON", "")
    if not bridge_env:
        raise RuntimeError(
            f"Cannot read {path}: no parquet engine and no CSV sidecar {csv}")
    bridge = Path(bridge_env)
    tmp = path.with_suffix(path.suffix + ".tmp.csv")
    try:
        subprocess.run(
            [
                str(bridge),
                "-c",
                (
                    "import pandas as pd, sys; "
                    "pd.read_parquet(sys.argv[1]).to_csv(sys.argv[2], index=False)"
                ),
                str(path),
                str(tmp),
            ],
            check=True,
        )
        return pd.read_csv(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def outcomes_from_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    shared = [
        "dataset_version",
        "exit_policy_version",
        "product",
        "contract",
        "candidate_id",
        "decision_row",
        "decision_timestamp",
        "trading_day",
    ]
    for row in candidates.itertuples(index=False):
        base = {name: getattr(row, name) for name in shared if hasattr(row, name)}
        for direction in ("long", "short"):
            rows.append({
                **base,
                "direction": direction.upper(),
                "entry_row": int(getattr(row, f"{direction}_entry_row")),
                "entry_timestamp": getattr(row, f"{direction}_entry_timestamp"),
                "exit_row": int(getattr(row, f"{direction}_exit_row")),
                "exit_timestamp": getattr(row, f"{direction}_exit_timestamp"),
                "exit_reason": getattr(row, f"{direction}_exit_reason"),
                "holding_bars": int(getattr(row, f"{direction}_holding_bars")),
                "gross_pnl": float(getattr(row, f"{direction}_gross_pnl")),
                "spread_cost": float(getattr(row, f"{direction}_spread_cost")),
                "fee_cost": float(getattr(row, f"{direction}_fee_cost")),
                "net_pnl": float(getattr(row, f"{direction}_net_pnl")),
                "mfe_gross": float(getattr(row, f"{direction}_mfe_gross")),
                "mae_gross": float(getattr(row, f"{direction}_mae_gross")),
            })
    return pd.DataFrame.from_records(rows)


def build_dataset_artifacts(
    config_path: str | Path,
    output: str | Path,
) -> tuple[EntryEvalConfig, pd.DataFrame, np.ndarray, list[str], dict]:
    config_path = Path(config_path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    config = load_entry_eval_config(config_path)
    _, market = load_market_frames(config)
    context = build_evaluation_context(market)
    filter_audit = candidate_filter_audit(market, config, context=context)
    candidates, X, names = build_entry_dataset(market, config)
    candidates = add_diagnostic_context(candidates, X, names)
    X_flat, flat_names = build_flattened_entry_features(market, candidates, config)
    outcomes = outcomes_from_candidates(candidates)
    folds = make_walk_forward_folds(candidates["trading_day"].unique())

    shutil.copyfile(config_path, out / "config.yaml")
    write_parquet(candidates, out / "candidates.parquet")
    write_parquet(outcomes, out / "outcomes.parquet")
    np.savez_compressed(out / "features.npz", X=X, feature_names=np.asarray(names, dtype=object))
    np.savez_compressed(
        out / "flat_features.npz",
        X=X_flat,
        feature_names=np.asarray(flat_names, dtype=object),
    )
    write_json(out / "split_manifest.json", folds)

    manifest = {
        "version": config.version,
        "git_commit": git_commit(),
        "config": config.to_dict(),
        "config_sha256": file_hash(config_path),
        "data_sha256": file_hash(Path(config.data.path)),
        "rows": int(len(market)),
        "candidates": int(len(candidates)),
        "outcomes": int(len(outcomes)),
        "observation_shape": [config.data.window_size, len(names) // 4],
        "feature_transform": "latest_mean_std_delta_of_market_seq",
        "feature_count": int(X.shape[1]),
        "flat_feature_transform": "flattened_60x18_market_seq",
        "flat_feature_count": int(X_flat.shape[1]),
        "candidate_filter_audit": filter_audit,
    }
    write_json(out / "manifest.json", manifest)
    return config, candidates, X, names, manifest


def load_dataset_artifacts(
    output: str | Path,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    out = Path(output)
    candidates = read_parquet(out / "candidates.parquet")
    features = np.load(out / "features.npz", allow_pickle=True)
    X = features["X"]
    names = [str(x) for x in features["feature_names"].tolist()]
    return candidates, X, names


def load_flattened_dataset_artifacts(
    output: str | Path,
) -> tuple[np.ndarray, list[str]]:
    out = Path(output)
    features = np.load(out / "flat_features.npz", allow_pickle=True)
    X = features["X"]
    names = [str(x) for x in features["feature_names"].tolist()]
    return X, names
