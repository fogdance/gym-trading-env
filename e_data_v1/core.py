"""Causal, real-contract 1m/15m snapshots. No account or training dependencies."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
import re

import numpy as np
import pandas as pd

from .reference import legacy_features, legacy_helper, load_legacy
from .schema import (Decision, EDataBundle, EDataConfig, LEGACY_FIELDS,
                     RAW_FIELDS, SHARED_FIELDS, identity, local_timestamp)


class EDataError(ValueError):
    def __init__(self, message, events=()):
        super().__init__(message)
        self.events = list(events)


def _observation_provenance(value):
    """An observation identity link cannot carry labels or account payloads."""
    if value is None or isinstance(value, str):
        return value
    allowed = {"observation_version", "observation_identity", "source_sha256",
               "schema_sha256", "clock", "clock_sha256", "source_contract",
               "source_path", "config_sha256", "gym_sha", "dreamer_sha"}
    if not isinstance(value, dict) or set(value)-allowed or not all(isinstance(x, str) for x in value.values()):
        raise EDataError("Original observation provenance accepts identity fields only; labels/account payloads refused")
    return deepcopy(value)


def _index(values, timezone):
    idx = pd.DatetimeIndex(pd.to_datetime(values, format="mixed"))
    if idx.hasnans:
        raise EDataError("Missing or invalid market timestamp")
    return legacy_helper("time_contract").ensure_feature_tz_index(idx, assume_tz=timezone)


def normalize_bars(minute_bars, schedule):
    """Copy only market fields; never propagate teacher/account columns."""
    required = set(RAW_FIELDS) | {"symbol", "trading_date", "bob"}
    if required - set(minute_bars):
        raise EDataError("Missing minute fields: " + str(sorted(required - set(minute_bars))))
    timezone = schedule["timezone"]
    if timezone != "Asia/Shanghai":
        raise EDataError("FG source timezone must be explicit Asia/Shanghai")
    idx = _index(minute_bars["Date"] if "Date" in minute_bars else minute_bars.index, timezone)
    if len(idx) == 0:
        raise EDataError("Empty minute source")
    f = minute_bars[list(RAW_FIELDS) + ["symbol", "trading_date", "bob"]].copy()
    f.index = idx
    for col in RAW_FIELDS:
        f[col] = pd.to_numeric(f[col], errors="coerce").astype(np.float64)
    f["symbol"] = f.symbol.astype(str)
    f["trading_date"] = pd.to_datetime(f.trading_date.astype(str), format="mixed").dt.strftime("%Y-%m-%d")
    if f.trading_date.isna().any():
        raise EDataError("Missing trading_date")
    f["bob"] = _index(f.bob, timezone)
    arrivals = minute_bars["available_at"] if "available_at" in minute_bars else idx
    f["available_at"] = _index(arrivals, timezone)
    # A reported early arrival cannot make an unfinished candle available.
    f["available_at"] = pd.DatetimeIndex(np.maximum(f.index.asi8, pd.DatetimeIndex(f.available_at).asi8)).tz_localize("UTC").tz_convert(timezone)
    f["_duration"] = (f.index - pd.DatetimeIndex(f.bob)).total_seconds() / 60
    f["_legal"] = f._duration.eq(1)
    # Attach all row metadata before the existing helper sorts the frame.
    # Invalid rows stay intact so duplicates/NaN VI are not silently repaired.
    if not f.index.has_duplicates and np.isfinite(f[list(RAW_FIELDS)].to_numpy()).all():
        f = legacy_helper("ohlcvi_contract").normalize_ohlcvi(f)
    return f.sort_index(kind="mergesort")


def _clock_slots(f, schedule):
    clock = {k: schedule[k] for k in ("timezone", "segments", "minutes_per_day")}
    if clock["minutes_per_day"] != 345:
        raise EDataError("S3-E V1 requires the existing FG 345-minute clock")
    start_ns = np.full(len(f), -1, np.int64)
    end_ns = np.full(len(f), -1, np.int64)
    count = 0
    minutes = f.index.hour * 60 + f.index.minute
    day = f.index.normalize()
    for a, b in clock["segments"]:
        start = sum(int(v) * m for v, m in zip(a.split(":"), (60, 1)))
        end = sum(int(v) * m for v, m in zip(b.split(":"), (60, 1)))
        if not 0 <= start < end < 1440 or (end - start) % 15:
            raise EDataError("Invalid FG continuous-session definition")
        count += end - start
        hit = (minutes > start) & (minutes <= end)
        hit |= (minutes == start) & f._duration.eq(0).to_numpy()
        start_ns[hit] = (day[hit] + pd.Timedelta(minutes=start)).asi8
        end_ns[hit] = (day[hit] + pd.Timedelta(minutes=end)).asi8
    if count != 345:
        raise EDataError("FG schedule does not contain exactly 345 minutes")
    return clock, start_ns, end_ns


def _calendar(schedule):
    if "dated_sessions" not in schedule:
        raise EDataError("Explicit dated_sessions required; do not infer holidays from future data")
    result = []
    for s in schedule["dated_sessions"]:
        a, b = (local_timestamp(s[k], schedule["timezone"]) for k in ("start", "end"))
        if a >= b or a.second or b.second or a.microsecond or b.microsecond:
            raise EDataError("Invalid dated calendar interval")
        result.extend(pd.date_range(a + pd.Timedelta(minutes=1), b, freq="min"))
    idx = pd.DatetimeIndex(result)
    if len(idx) == 0 or idx.has_duplicates:
        raise EDataError("Empty or overlapping dated calendar")
    return idx.sort_values()


def _validate(f, schedule, starts, calendar):
    values = f[list(RAW_FIELDS)].to_numpy()
    ok = np.isfinite(values).all(axis=1)
    ok &= (values[:, :4] > 0).all(axis=1) & (values[:, 4:] >= 0).all(axis=1)
    ok &= (f.High >= f[["Open", "Low", "Close"]].max(axis=1)).to_numpy()
    ok &= (f.Low <= f[["Open", "High", "Close"]].min(axis=1)).to_numpy()
    ok &= (~f._legal | (f.OpenInterest > 0)).to_numpy()
    ok &= f._duration.isin([0, 1]).to_numpy()
    ok &= (starts >= 0) & (f.index.second == 0) & (f.index.microsecond == 0)
    ok &= (f._duration.eq(0) | f.index.isin(calendar)).to_numpy()
    # Reject every conflicting copy, rather than keep a last-wins candle.
    duplicate = f.index.duplicated(keep=False)
    ok &= ~duplicate
    events = []
    for i in np.flatnonzero(~ok):
        events.append(dict(type="MINUTE_REJECTED", source_contract=str(f.symbol.iloc[i]),
            time=f.index[i].isoformat(), available_at=f.available_at.iloc[i].isoformat(),
            reason="DUPLICATE_TIMESTAMP" if duplicate[i] else "INVALID_OHLCVOI_OR_CLOCK"))
    return ok, events


def aggregate_buckets(minute_bars, schedule, config=None):
    """Inspect complete and rejected buckets, without fabricating missing bars."""
    data = MarketData(minute_bars, schedule, config or EDataConfig(), compute_features=False)
    return deepcopy(data.buckets), deepcopy(data.events)


class MarketData:
    """Prepare one real contract once, then request arbitrary causal snapshots.

    Standard CSV arrivals equal eob: the unchanged legacy calculator is reused
    once. Explicit delayed arrivals use a visible-prefix calculation per query.
    """
    def __init__(self, minute_bars, schedule, config=None, *, compute_features=True):
        self.config = config or EDataConfig()
        self.schedule = deepcopy(schedule)
        self.frame = normalize_bars(minute_bars, schedule)
        contracts = set(self.frame.symbol)
        if len(contracts) != 1:
            raise EDataError("MarketData requires one real contract; separate contracts before aggregation")
        self.contract = next(iter(contracts))
        if not re.fullmatch(r"CZCE\.FG\d{3,4}", self.contract):
            raise EDataError("S3-E V1 requires an explicit real CZCE.FG contract")
        self.clock, starts, ends = _clock_slots(self.frame, schedule)
        self.calendar = _calendar(schedule)
        self.good, self.events = _validate(self.frame, schedule, starts, self.calendar)
        self.frame["_good"] = self.good
        self.market_clock_counter = legacy_helper("s3_market_clock", self.config.legacy_root).MarketClock(
            np.asarray(self.good & self.frame._legal.to_numpy(), dtype=bool))
        self.delayed = bool((pd.DatetimeIndex(self.frame.available_at).asi8 > self.frame.index.asi8).any())
        self.buckets = self._buckets(starts, ends)
        self.complete_buckets = [x for x in self.buckets if x["complete"]]
        self.bucket_end_ns = np.array([local_timestamp(x["bucket_end"]).value for x in self.complete_buckets], dtype=np.int64)
        self.bucket_available_ns = np.array([local_timestamp(x["available_at"]).value for x in self.complete_buckets], dtype=np.int64)
        self.features = None
        if compute_features:
            self._features()

    def _features(self):
        if self.features is None:
            f = self.frame.loc[self.good & self.frame._legal.to_numpy()]
            cols = list(LEGACY_FIELDS) + ["ref_close_t", "cumVWAP_t", "session_high_t", "session_low_t", "dC_minus_cumVWAP_t"]
            self.features = (legacy_features(f, self.clock, self.config.legacy_root)[cols].copy()
                             if len(f) else pd.DataFrame(index=f.index, columns=cols))
            self.obs_array = self.features[list(LEGACY_FIELDS)].to_numpy(dtype=np.float32)
            self.aux_array = self.features[["ref_close_t", "cumVWAP_t", "session_high_t", "session_low_t", "dC_minus_cumVWAP_t"]].to_numpy(dtype=np.float64)
            self.price_array = f[list(RAW_FIELDS[:4])].to_numpy(dtype=np.float64)
            td = f.trading_date.to_numpy()
            if len(td) and not pd.DatetimeIndex(td).is_monotonic_increasing:
                raise EDataError("Trading-date history must be chronological")
            starts = np.flatnonzero(np.r_[True,td[1:] != td[:-1]]) if len(td) else np.array([],dtype=int)
            ends = np.r_[starts[1:],len(td)]
            self.ref_source = np.zeros(len(td),dtype=np.int64)
            for j,(a,b) in enumerate(zip(starts,ends)):
                self.ref_source[a:b] = int(starts[j]-1) if j else int(a)
        return self.features

    def _buckets(self, starts, ends):
        f = self.frame.copy()
        f["_session_start"] = starts
        f["_session_end"] = ends
        # Opening marks never supply a minute, volume or an OI endpoint.
        valid_minutes = f.loc[f._legal & (starts >= 0)]
        result = []
        for (start_ns, end_ns), group in valid_minutes.groupby(["_session_start", "_session_end"], sort=True):
            start = pd.Timestamp(start_ns, tz="UTC").tz_convert(self.clock["timezone"])
            end = pd.Timestamp(end_ns, tz="UTC").tz_convert(self.clock["timezone"])
            for bucket_end in pd.date_range(start + pd.Timedelta(minutes=15), end, freq="15min"):
                bucket_start = bucket_end - pd.Timedelta(minutes=15)
                rows = group.loc[(group.index > bucket_start) & (group.index <= bucket_end)]
                expected = pd.date_range(bucket_start + pd.Timedelta(minutes=1), bucket_end, freq="min")
                unique = not rows.index.has_duplicates
                complete = len(rows) == 15 and unique and rows.index.equals(expected)
                complete &= expected.isin(self.calendar).all() and rows.trading_date.nunique() == 1
                if len(rows):
                    complete &= rows._good.all()
                arrival = max([bucket_end] + rows.available_at.tolist())
                td = str(rows.trading_date.iloc[0]) if len(rows) else None
                item = dict(bucket_start=bucket_start.isoformat(), bucket_end=bucket_end.isoformat(),
                    available_at=arrival.isoformat(), source_contract=self.contract,
                    trading_date=td, complete=bool(complete), observed_minutes=len(rows),
                    bucket_id=self.contract + ":" + bucket_end.isoformat())
                if complete:
                    item.update(Open=float(rows.Open.iloc[0]), High=float(rows.High.max()),
                        Low=float(rows.Low.min()), Close=float(rows.Close.iloc[-1]),
                        Volume=float(rows.Volume.sum()), OpenInterest=float(rows.OpenInterest.iloc[-1]))
                else:
                    item["reason"] = "DUPLICATE_TIMESTAMP" if not unique else "INCOMPLETE_OR_INVALID_MINUTE_SLOTS"
                    self.events.append(dict(type="BUCKET_REJECTED", source_contract=self.contract,
                        time=bucket_end.isoformat(), available_at=arrival.isoformat(), reason=item["reason"],
                        observed_minutes=len(rows)))
                result.append(item)
        result.sort(key=lambda x: x["bucket_end"])
        history = deque(maxlen=20)
        previous = None
        for item in result:
            if not item["complete"]:
                continue
            ready = len(history) == 20 and previous is not None
            raw_vol = float(np.log((item["Volume"] + 1) / (np.mean([x["Volume"] for x in history]) + 1))) if len(history) == 20 else None
            raw_oi = float(np.log(item["OpenInterest"] / previous["OpenInterest"])) if previous is not None else None
            if any(x is not None and not np.isfinite(x) for x in (raw_vol, raw_oi)):
                raise EDataError("Nonfinite derived volume/OI feature; snapshot refused")
            item.update(feature_ready=ready, warmup_reason=None if ready else "PREVIOUS_20_BUCKETS_REQUIRED",
                vol_relative_raw=raw_vol, oi_change_raw=raw_oi,
                volume_baseline_bucket_count=len(history),
                volume_baseline_bucket_ids=[x["bucket_id"] for x in history],
                previous_OI=previous["OpenInterest"] if previous else None,
                previous_OI_bucket_id=previous["bucket_id"] if previous else None)
            history.append(item)
            previous = item
        return result

    def snapshot(self, decision):
        if decision.source_contract != self.contract:
            raise EDataError("Decision/source real contract mismatch")
        cutoff = local_timestamp(decision.visible_cutoff, self.clock["timezone"])
        if decision.decision_time is not None and local_timestamp(decision.decision_time, self.clock["timezone"]) < cutoff:
            raise EDataError("visible_cutoff exceeds decision_time")
        if self.delayed:
            visible = self.frame.loc[(self.frame.index <= cutoff) & (self.frame.available_at <= cutoff)]
            if visible.empty:
                raise EDataError("No visible market price at decision")
            view = MarketData(visible, self.schedule, self.config)
            return view._snapshot_at(decision, cutoff)
        return self._snapshot_at(decision, cutoff)

    def build(self, decisions):
        """Batch snapshots using this already prepared read-only source."""
        decisions = [d if isinstance(d, Decision) else Decision(**d) for d in decisions]
        if not decisions:
            raise EDataError("At least one decision is required")
        items = [self.snapshot(d) for d in decisions]
        return _bundle(self, decisions, items)

    def _snapshot_at(self, decision, cutoff):
        f = self.frame
        bad = [e for e in self.events if e["type"] == "MINUTE_REJECTED"
               and local_timestamp(e["time"]) <= cutoff and local_timestamp(e["available_at"]) <= cutoff]
        if bad:
            raise EDataError("Visible invalid minute data; snapshot refused", bad)
        through = int(f.index.searchsorted(cutoff, side="right"))-1
        if through < 0:
            raise EDataError("No visible market price at decision")
        self._features()
        raw_indices = self.market_clock_counter.window(through,60)
        feature_end = int(self.market_clock_counter.prefix[through+1])
        feature_start = max(0,feature_end-60)
        window = f.iloc[raw_indices]
        old = np.zeros((60, 18), np.float32)
        if len(window):
            old[-len(window):] = self.obs_array[feature_start:feature_end]
        if decision.market_seq is not None:
            supplied = np.asarray(decision.market_seq)
            if supplied.shape != (60, 18) or supplied.dtype != np.float32 or not np.array_equal(old, supplied):
                raise EDataError("Original Env market_seq does not match unchanged 18-channel reference")
        anchor, anchor_source, anchor_time, anchor_arrival = self._anchor(through,feature_end,cutoff)
        shared = old.copy()
        audit = np.zeros((60, 4), np.float32) if self.config.audit_1m else None
        if len(window):
            aux = self.aux_array[feature_start:feature_end]
            prices = self.price_array[feature_start:feature_end]
            values = {
                "obs_cumVWAP_t": np.clip(np.log(aux[:,1] / anchor), -1, 1),
                "obs_session_high_t": np.clip(np.log(aux[:,2] / anchor), -1, 1),
                "obs_session_low_t": np.clip(np.log(aux[:,3] / anchor), -1, 1),
                "obs_pct_chg_from_ref_t": np.clip(np.log(prices[:,3] / anchor), -1, 1),
                "obs_range_t": np.clip(np.log1p((aux[:,2] - aux[:,3]) / anchor), 0, 1),
                "obs_dC_minus_cumVWAP_t": np.clip(np.sign(aux[:,4]) * np.log1p(np.abs(aux[:,4]) / anchor), -1, 1),
            }
            for name in SHARED_FIELDS:
                shared[-len(window):, LEGACY_FIELDS.index(name)] = values[name].astype(np.float32)
            if audit is not None:
                audit[-len(window):] = np.log(prices / anchor).astype(np.float32)
        indices = np.flatnonzero((self.bucket_end_ns <= cutoff.value) & (self.bucket_available_ns <= cutoff.value))[-24:]
        buckets = [self.complete_buckets[i] for i in indices]
        context = np.zeros((24, 7), np.float32)
        for i, item in enumerate(buckets, 24 - len(buckets)):
            if item["feature_ready"]:
                context[i, :4] = np.log(np.array([item[k] for k in RAW_FIELDS[:4]]) / anchor)
                context[i, 4] = np.clip(item["vol_relative_raw"], *self.config.volume_clip)
                context[i, 5] = np.clip(item["oi_change_raw"], *self.config.oi_clip)
                context[i, 6] = 1
        for array in (context, old, shared, audit):
            if array is not None and not np.isfinite(array).all():
                raise EDataError("Nonfinite snapshot; no nan_to_num repair allowed")
        meta = dict(source_contract=self.contract, visible_cutoff=cutoff.isoformat(),
            decision_time=local_timestamp(decision.decision_time or cutoff).isoformat(),
            price_anchor=anchor, anchor_source=anchor_source, anchor_source_time=anchor_time,
            anchor_available_at=anchor_arrival, minute_window_times=[t.isoformat() for t in window.index],
            minute_window_sha256=hashlib.sha256(old.tobytes()).hexdigest(),
            context_bucket_ids=[x["bucket_id"] for x in buckets],
            context_bucket_ends=[x["bucket_end"] for x in buckets],
            context_available_at=[x["available_at"] for x in buckets],
            context_valid=[bool(x["feature_ready"]) for x in buckets],
            context_feature_provenance=[dict(bucket_id=x["bucket_id"], feature_ready=x["feature_ready"],
                vol_relative_raw=x["vol_relative_raw"], oi_change_raw=x["oi_change_raw"],
                volume_baseline_bucket_ids=x["volume_baseline_bucket_ids"],
                previous_OI_bucket_id=x["previous_OI_bucket_id"]) for x in buckets],
            context_padding=24 - len(buckets), minute_padding=60 - len(window))
        return context, old, shared, anchor, audit, meta

    def _anchor(self, through, feature_end, cutoff):
        if feature_end:
            source_feature = int(self.ref_source[feature_end-1])
            source_raw = int(self.market_clock_counter.rows[source_feature])
            source = self.frame.iloc[source_raw]
            source_time = self.frame.index[source_raw]
            value = float(source.Close)
            ref = float(self.aux_array[feature_end-1,0])
            if value == ref and source_time <= cutoff and source.available_at <= cutoff:
                last_raw = int(self.market_clock_counter.rows[feature_end-1])
                kind = ("legacy_first_observed_close" if source.trading_date == self.frame.trading_date.iloc[last_raw]
                        else "legacy_previous_trading_day_close")
                return value, kind, source_time.isoformat(), source.available_at.isoformat()
        # Zero-duration opening marks may expose a price, but never a future
        # first valid candle. An unknown positive price is not fabricated.
        row = self.frame.iloc[through]
        value = float(row.Close)
        if not np.isfinite(value) or value <= 0:
            raise EDataError("No legal observed price for anchor fallback")
        return value, "observed_close_fallback", self.frame.index[through].isoformat(), row.available_at.isoformat()


def build_e_data_v1(minute_bars, decisions, schedule, config=None):
    """Build the requested decisions only; full-source identity is separate.

    One invocation represents one real contract. Call separately at a roll.
    Decisions can be Decision objects or dictionaries with the same fields.
    """
    config = config or EDataConfig()
    decisions = [d if isinstance(d, Decision) else Decision(**d) for d in decisions]
    if not decisions:
        raise EDataError("At least one decision is required")
    max_cutoff = max(local_timestamp(d.visible_cutoff, schedule["timezone"]) for d in decisions)
    # Changes to values at future timestamps cannot enter validation or the
    # unchanged legacy calculator for the requested historical decisions.
    times = _index(minute_bars["Date"] if "Date" in minute_bars else minute_bars.index, schedule["timezone"])
    subset = minute_bars.loc[times <= max_cutoff].copy()
    data = MarketData(subset, schedule, config)
    return data.build(decisions)


def _bundle(data, decisions, items):
    config = data.config
    max_cutoff = max(local_timestamp(d.visible_cutoff, data.clock["timezone"]) for d in decisions)
    _, hashes = load_legacy(config.legacy_root)
    arrays = [np.stack([x[i] for x in items]) for i in (0, 1, 2)]
    manifest_data = data
    if data.delayed:
        visible = data.frame.loc[(data.frame.index <= max_cutoff) & (data.frame.available_at <= max_cutoff)]
        manifest_data = MarketData(visible, data.schedule, config, compute_features=False)
    bucket_manifest = [deepcopy(x) for x in manifest_data.buckets
                       if local_timestamp(x["bucket_end"]) <= max_cutoff
                       and local_timestamp(x["available_at"]) <= max_cutoff]
    events = [deepcopy(x) for x in manifest_data.events if local_timestamp(x["time"]) <= max_cutoff
              and local_timestamp(x["available_at"]) <= max_cutoff]
    return EDataBundle(*arrays, np.array([x[3] for x in items], dtype=np.float64),
        np.stack([x[4] for x in items]) if config.audit_1m else None,
        bucket_manifest, [x[5] for x in items], config.schema(hashes),
        dict(clock_sha256=identity(data.schedule), legacy_sources_sha256=hashes,
             original_observation_provenance=[_observation_provenance(d.original_provenance) for d in decisions],
             inputs="market OHLCVOI, explicit calendar, cutoff, original observation identity only"), events)
