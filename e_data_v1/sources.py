"""Read-only adapters for existing FG TRAIN pool manifests and dated calendars."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
import yaml

from .io import sha256, stable_identity


@dataclass
class FGSource:
    contract: str
    bars: pd.DataFrame
    schedule: dict
    provenance: dict
    env_config: str


def _yaml(path, expected=None):
    record = stable_identity(path)
    if expected and record["sha256"] != expected:
        raise ValueError("Config hash mismatch: " + str(path))
    value = yaml.safe_load(Path(record["path"]).read_text())
    if sha256(path) != record["sha256"]:
        raise ValueError("Config changed while reading: " + str(path))
    return value, record


def _resolve(parent, value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(parent).parent / p).resolve()


def pool_entries(path, contracts=()):
    record = stable_identity(path)
    pool = json.loads(Path(record["path"]).read_text())
    if pool.get("product") not in ("FG", "SA") or not isinstance(pool.get("files"), list):
        raise ValueError("Expected an existing FG pool manifest")
    entries = [f for f in pool["files"] if f.get("role") == "train" and f.get("eligible_for_training")]
    if contracts:
        selected = set(contracts)
        entries = [f for f in entries if f["contract"] in selected]
        if {f["contract"] for f in entries} != selected:
            raise ValueError("Requested contract is absent from eligible TRAIN sources")
    if not entries or len({f["contract"] for f in entries}) != len(entries):
        raise ValueError("Empty or duplicate TRAIN contract selection")
    if sha256(path) != record["sha256"]:
        raise ValueError("Pool changed while reading")
    return entries, record


def read_fg_source(entry, pool_record):
    market = stable_identity(entry["path"])
    if market["sha256"] != entry["sha256"]:
        raise ValueError("Market hash mismatch: " + entry["path"])
    bars = pd.read_csv(market["path"])
    if len(bars) != entry["rows"] or set(bars.symbol) != {entry["contract"]}:
        raise ValueError("Real contract or row count disagrees with TRAIN manifest")
    if sha256(market["path"]) != market["sha256"]:
        raise ValueError("Market changed while loading")
    env, env_record = _yaml(entry["env_config"], entry["env_sha256"])
    terms, terms_record = _yaml(entry["contract_config"], entry["contract_sha256"])
    if _resolve(entry["env_config"], env["trading"]["data_path"]) != Path(market["path"]):
        raise ValueError("Env and pool refer to different market files")
    if env["entry"]["contract"] != entry["contract"] or terms["contract"] != entry["contract"]:
        raise ValueError("Env/terms real contract mismatch")
    if terms.get("source_sha256") != market["sha256"]:
        raise ValueError("Dated calendar source identity mismatch")
    if env["entry"].get("strategy_config"):
        strategy_path = _resolve(entry["env_config"], env["entry"]["strategy_config"])
        master, master_record = _yaml(strategy_path)
        clock = master["market_clock"]
    else:
        clock = env["entry"]["market_clock"]
        master_record = None
    schedule = {k: clock[k] for k in ("timezone", "minutes_per_day", "segments")}
    schedule["dated_sessions"] = terms["sessions"]
    schedule["calendar_basis"] = terms.get("calendar_basis", "explicit_supplied_calendar")
    # Hashes link the complete existing configs. No reward, direction, teacher
    # labels or account records are copied into the model data.
    provenance = dict(role="TRAIN", source_pool=pool_record, market=market,
        env=env_record, contract=terms_record, master=master_record,
        source_contract=entry["contract"], tick_size=terms["tick_size"],
        price_unit="original quoted real-contract price", roll_adjustment="none",
        calendar_basis=schedule["calendar_basis"])
    return FGSource(entry["contract"], bars, schedule, provenance, env_record["path"])
