from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


DATE_COL = "Date"


@dataclass(frozen=True)
class SplitDates:
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True)
class BuilderPaths:
    raw_csv: Path
    env_template: Path
    entry_eval_template: Path
    output_root: Path
    split_name: str
    raw_output_name: str
    train_output_name: str

    @property
    def raw_output(self) -> Path:
        return self.output_root / "data" / "raw" / self.raw_output_name

    @property
    def train_output(self) -> Path:
        return self.output_root / "data" / "generated" / self.train_output_name

    @property
    def train_env_config(self) -> Path:
        return self.output_root / "configs" / "env" / f"{self.split_name}_train.yaml"

    @property
    def full_env_config(self) -> Path:
        return self.output_root / "configs" / "env" / f"{self.split_name}_full.yaml"

    @property
    def entry_eval_config(self) -> Path:
        return self.output_root / "configs" / "entry_eval" / f"{self.split_name}_signal_close.yaml"

    @property
    def split_artifact_dir(self) -> Path:
        return self.output_root / "artifacts" / "walk_forward_splits" / self.split_name


def day_int(value) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError("empty date")
    date = text.split()[0].replace("/", "-")
    parts = date.split("-")
    if len(parts) == 3:
        return int(parts[0]) * 10000 + int(parts[1]) * 100 + int(parts[2])
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) < 8:
        raise ValueError(f"cannot parse date: {value!r}")
    return int(digits[:8])


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def read_market_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if DATE_COL not in df.columns:
        raise ValueError(f"raw CSV must contain {DATE_COL!r} column: {path}")
    df = df.copy()
    df["_trading_day"] = pd.to_datetime(df[DATE_COL]).dt.strftime("%Y%m%d").astype(int)
    if df.empty:
        raise ValueError(f"raw CSV is empty: {path}")
    return df


def split_days(days: Iterable[int], dates: SplitDates) -> dict[str, list[int]]:
    unique = sorted({int(x) for x in days})

    def pick(start: int, end: int) -> list[int]:
        return [day for day in unique if int(start) <= day <= int(end)]

    roles = {
        "train": pick(dates.train_start, dates.train_end),
        "validation": pick(dates.validation_start, dates.validation_end),
        "test": pick(dates.test_start, dates.test_end),
    }
    for role, role_days in roles.items():
        if not role_days:
            raise ValueError(f"{role} split has no trading days")
    if max(roles["train"]) >= min(roles["validation"]):
        raise ValueError("train must end before validation starts")
    if max(roles["validation"]) >= min(roles["test"]):
        raise ValueError("validation must end before test starts")
    if set(roles["train"]) & set(roles["validation"]):
        raise ValueError("train/validation overlap")
    if set(roles["train"]) & set(roles["test"]):
        raise ValueError("train/test overlap")
    if set(roles["validation"]) & set(roles["test"]):
        raise ValueError("validation/test overlap")
    return roles


def split_manifest(split_name: str, roles: dict[str, list[int]]) -> list[dict]:
    return [{
        "name": split_name,
        "train_days": roles["train"],
        "validation_days": roles["validation"],
        "test_days": roles["test"],
    }]


def _load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def _write_yaml(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))


def _date_range(days: list[int]) -> dict:
    return {
        "start_day": int(min(days)),
        "end_day": int(max(days)),
        "trading_days": int(len(days)),
    }


def _row_count_by_role(df: pd.DataFrame, roles: dict[str, list[int]]) -> dict[str, int]:
    out = {}
    day = df["_trading_day"].astype(int)
    for role, role_days in roles.items():
        out[role] = int(day.isin(role_days).sum())
    return out


def _write_train_csv(df: pd.DataFrame, train_days: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[df["_trading_day"].isin(train_days)].drop(columns=["_trading_day"])
    if out.empty:
        raise ValueError("generated train CSV would be empty")
    out.to_csv(path, index=False)


def _copy_raw_csv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)


def _build_env_config(template: dict, *, data_path: Path) -> dict:
    out = dict(template)
    out["trading"] = dict(out.get("trading", {}) or {})
    out["trading"]["data_path"] = str(data_path.resolve())
    out["trading"]["data_interval"] = "1m"
    return out


def _build_entry_eval_config(template: dict, *, split_name: str, raw_path: Path, split_path: Path) -> dict:
    out = dict(template)
    out["version"] = split_name
    out["data"] = dict(out.get("data", {}) or {})
    out["data"]["path"] = str(raw_path.resolve())
    out["walk_forward"] = dict(out.get("walk_forward", {}) or {})
    out["walk_forward"]["split_manifest_path"] = str(split_path.resolve())
    return out


def load_configured_split_manifest(config_path: str | Path) -> list[dict] | None:
    raw = _load_yaml(config_path)
    split_path = (raw.get("walk_forward", {}) or {}).get("split_manifest_path")
    if not split_path:
        return None
    path = Path(split_path).expanduser()
    if not path.is_absolute():
        path = Path(config_path).resolve().parent / path
    if not path.exists():
        raise FileNotFoundError(f"configured split_manifest_path does not exist: {path}")
    return json.loads(path.read_text())


def load_required_split_manifest(config_path: str | Path) -> list[dict]:
    configured = load_configured_split_manifest(config_path)
    if configured is None:
        raise ValueError(
            "entry_eval config must define walk_forward.split_manifest_path; "
            f"config={config_path}")
    return configured


def build_walk_forward_split(
    *,
    paths: BuilderPaths,
    dates: SplitDates,
) -> dict:
    raw_csv = paths.raw_csv.expanduser().resolve()
    if not raw_csv.exists():
        raise FileNotFoundError(raw_csv)
    df = read_market_csv(raw_csv)
    roles = split_days(df["_trading_day"].to_numpy(), dates)
    manifest = split_manifest(paths.split_name, roles)

    _copy_raw_csv(raw_csv, paths.raw_output)
    _write_train_csv(df, roles["train"], paths.train_output)

    paths.split_artifact_dir.mkdir(parents=True, exist_ok=True)
    split_path = paths.split_artifact_dir / "split_manifest.json"
    write_json(split_path, manifest)
    split_hash = file_sha256(split_path)
    (paths.split_artifact_dir / "split_manifest.sha256").write_text(split_hash + "\n")

    env_template = _load_yaml(paths.env_template)
    train_env = _build_env_config(env_template, data_path=paths.train_output)
    full_env = _build_env_config(env_template, data_path=paths.raw_output)
    _write_yaml(paths.train_env_config, train_env)
    _write_yaml(paths.full_env_config, full_env)

    entry_template = _load_yaml(paths.entry_eval_template)
    entry_eval = _build_entry_eval_config(
        entry_template,
        split_name=paths.split_name,
        raw_path=paths.raw_output,
        split_path=split_path,
    )
    _write_yaml(paths.entry_eval_config, entry_eval)

    rows = _row_count_by_role(df, roles)
    summary = {
        "split_name": paths.split_name,
        "raw_csv_source": str(raw_csv),
        "raw_csv": str(paths.raw_output.resolve()),
        "raw_sha256": file_sha256(paths.raw_output),
        "train_csv": str(paths.train_output.resolve()),
        "train_sha256": file_sha256(paths.train_output),
        "train_env_config": str(paths.train_env_config.resolve()),
        "full_env_config": str(paths.full_env_config.resolve()),
        "entry_eval_config": str(paths.entry_eval_config.resolve()),
        "split_manifest": str(split_path.resolve()),
        "split_manifest_hash": split_hash,
        "roles": {
            "train": {**_date_range(roles["train"]), "rows": rows["train"]},
            "validation": {**_date_range(roles["validation"]), "rows": rows["validation"]},
            "test": {**_date_range(roles["test"]), "rows": rows["test"]},
        },
        "checks": {
            "chronological": True,
            "no_overlap": True,
            "train_csv_matches_split_train": True,
            "validation_hidden_from_training": True,
            "test_hidden_from_training": True,
        },
    }
    write_json(paths.split_artifact_dir / "summary.json", summary)
    write_json(paths.split_artifact_dir / "generated_files.json", {
        "raw_csv": summary["raw_csv"],
        "train_csv": summary["train_csv"],
        "train_env_config": summary["train_env_config"],
        "full_env_config": summary["full_env_config"],
        "entry_eval_config": summary["entry_eval_config"],
        "split_manifest": summary["split_manifest"],
        "split_manifest_sha256": str((paths.split_artifact_dir / "split_manifest.sha256").resolve()),
    })
    return summary
