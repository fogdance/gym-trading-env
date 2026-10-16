"""Public, CPU-only S3 E data contracts. This is not a training config."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

SCHEMA = "S3_E_DATA_V1"
CONTEXT_FIELDS = ("rel_open", "rel_high", "rel_low", "rel_close",
                  "vol_relative", "oi_change", "valid")
LEGACY_FIELDS = (
    "obs_V_t", "obs_I_t", "obs_cumVWAP_t", "obs_dC_minus_cumVWAP_t",
    "obs_cmp_C_vs_cumVWAP_t", "obs_session_high_t", "obs_session_low_t",
    "obs_range_t", "obs_open_drift_t", "obs_bar_dir_t", "obs_minute_index_t",
    "obs_session_phase_t", "obs_volatility_t", "obs_dI_from_yclose_t",
    "obs_pct_chg_from_ref_t", "obs_mask_t", "obs_weekday_sin_t", "obs_weekday_cos_t",
)
SHARED_FIELDS = ("obs_cumVWAP_t", "obs_dC_minus_cumVWAP_t",
                 "obs_session_high_t", "obs_session_low_t", "obs_range_t",
                 "obs_pct_chg_from_ref_t")
RAW_FIELDS = ("Open", "High", "Low", "Close", "Volume", "OpenInterest")


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode()


def identity(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def local_timestamp(value, timezone="Asia/Shanghai"):
    t = pd.Timestamp(value)
    if pd.isna(t):
        raise ValueError("Missing timestamp")
    return t.tz_localize(timezone) if t.tz is None else t.tz_convert(timezone)


@dataclass(frozen=True)
class EDataConfig:
    volume_clip: tuple[float, float] = (-6.0, 6.0)
    oi_clip: tuple[float, float] = (-1.0, 1.0)
    audit_1m: bool = False
    legacy_root: Path | None = field(default=None, repr=False)

    def __post_init__(self):
        for bounds in (self.volume_clip, self.oi_clip):
            if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] >= bounds[1]:
                raise ValueError("Clip bounds must be finite and increasing")

    def schema(self, legacy_hashes):
        directory = Path(__file__).resolve().parent
        from gym_trading_env.utils.macd_5m_causal import CONTRACT
        indicator = directory.parent/'src/gym_trading_env/utils/macd_5m_causal.py'
        value = {
            "macd_5m": CONTRACT, "macd_code_sha256": hashlib.sha256(indicator.read_bytes()).hexdigest(),
            "schema": SCHEMA, "context_shape": [24, 7], "minute_shape": [60, 18],
            "dtype": "float32", "context_fields": list(CONTEXT_FIELDS),
            "minute_fields": list(LEGACY_FIELDS), "shared_fields": list(SHARED_FIELDS),
            "legacy_sources_sha256": legacy_hashes,
            "implementation_sources_sha256": {"e_data_v1/"+name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                                               for name in ("core.py", "schema.py", "reference.py")},
            "price": "log(raw_price/P_anchor_at_decision); no price clip",
            "price_unit": "unchanged real-contract quoted price; no lot multiplier or roll adjustment",
            "anchor": "proven legacy ref_close, otherwise already observed legal Close",
            "volume": "log((V+1)/(mean(previous 20 compliant completed buckets)+1))",
            "oi": "log(last_OI/last_OI_of_previous_compliant_completed_bucket)",
            "volume_clip": list(self.volume_clip), "oi_clip": list(self.oi_clip),
            "warmup": "keep last 24 complete visible buckets; unready rows all zero with valid=0",
            "padding": "left zeros", "order": "oldest to newest",
            "bucket": "15 exact minute slots in one planned continuous session, (start,end]",
            "availability": "max(bucket_end, all constituent arrival times) <= visible_cutoff",
            "feature_provenance": "per-decision volume/OI baseline bucket IDs, including delayed-arrival visibility",
            "clock": "S3_MARKET_MINUTE", "roll": "reset all history on real-contract change",
        }
        return {**value, "schema_sha256": identity(value)}


@dataclass(frozen=True)
class Decision:
    source_contract: str
    visible_cutoff: object
    decision_time: object | None = None
    # Optional original Env observation. When supplied, it must match exactly.
    market_seq: np.ndarray | None = field(default=None, repr=False, compare=False)
    original_provenance: dict | str | None = field(default=None, repr=False, compare=False)


@dataclass
class EDataBundle:
    context_15m: np.ndarray
    market_seq: np.ndarray
    market_seq_e_shared: np.ndarray
    price_anchor: np.ndarray
    audit_1m_ohlc_shared: np.ndarray | None
    bucket_manifest: list[dict]
    decision_meta: list[dict]
    schema: dict
    provenance: dict
    events: list[dict]
