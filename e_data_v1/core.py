"""Causal, real-contract 1m/15m snapshots. No account or training dependencies."""
from __future__ import annotations

from bisect import insort
from heapq import heappop, heappush
from copy import deepcopy
import hashlib
import json
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


def _index(values, timezone, tz_normalizer=None):
    idx = pd.DatetimeIndex(pd.to_datetime(values, format="mixed"))
    if idx.hasnans:
        raise EDataError("Missing or invalid market timestamp")
    normalizer = tz_normalizer or legacy_helper("time_contract").ensure_feature_tz_index
    return normalizer(idx, assume_tz=timezone)


def normalize_bars(minute_bars, schedule, *, tz_normalizer=None, ohlc_normalizer=None):
    """Copy only market fields; never propagate teacher/account columns."""
    required = set(RAW_FIELDS) | {"symbol", "trading_date", "bob"}
    if required - set(minute_bars):
        raise EDataError("Missing minute fields: " + str(sorted(required - set(minute_bars))))
    timezone = schedule["timezone"]
    if timezone != "Asia/Shanghai":
        raise EDataError("FG source timezone must be explicit Asia/Shanghai")
    idx = _index(minute_bars["Date"] if "Date" in minute_bars else minute_bars.index, timezone, tz_normalizer)
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
    f["bob"] = _index(f.bob, timezone, tz_normalizer)
    arrivals = minute_bars["available_at"] if "available_at" in minute_bars else idx
    f["available_at"] = _index(arrivals, timezone, tz_normalizer)
    # A reported early arrival cannot make an unfinished candle available.
    f["available_at"] = pd.DatetimeIndex(np.maximum(f.index.asi8, pd.DatetimeIndex(f.available_at).asi8)).tz_localize("UTC").tz_convert(timezone)
    f["_duration"] = (f.index - pd.DatetimeIndex(f.bob)).total_seconds() / 60
    f["_legal"] = f._duration.eq(1)
    # Attach all row metadata before the existing helper sorts the frame.
    # Invalid rows stay intact so duplicates/NaN VI are not silently repaired.
    if not f.index.has_duplicates and np.isfinite(f[list(RAW_FIELDS)].to_numpy()).all():
        normalizer = ohlc_normalizer or legacy_helper("ohlcvi_contract").normalize_ohlcvi
        f = normalizer(f)
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
    if 'dated_sessions' in schedule:
        start_ns[:] = -1; end_ns[:] = -1
        for session in schedule['dated_sessions']:
            a, b = (local_timestamp(session[k], timezone=clock['timezone']) for k in ('start','end'))
            hit = ((f.index > a) & (f.index <= b)) | ((f.index == a) & f._duration.eq(0).to_numpy())
            start_ns[hit], end_ns[hit] = a.value, b.value
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
    def __init__(self, minute_bars, schedule, config=None, *, compute_features=True,
                 feature_function=None, market_clock_class=None,
                 tz_normalizer=None, ohlc_normalizer=None, _env_inputs=None):
        self.config = config or EDataConfig()
        # Production Env may supply the exact existing feature/clock functions.
        # The independent CPU research path retains its original defaults.
        self._feature_function = feature_function
        self._market_clock_class = market_clock_class
        self._tz_normalizer = tz_normalizer
        self._ohlc_normalizer = ohlc_normalizer
        self.schedule = deepcopy(schedule)
        self.frame = (_env_inputs['frame'] if _env_inputs is not None else
            normalize_bars(minute_bars, schedule,
                tz_normalizer=tz_normalizer, ohlc_normalizer=ohlc_normalizer))
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
        valid = np.asarray(self.good & self.frame._legal.to_numpy(), dtype=bool)
        if _env_inputs is not None:
            self.market_clock_counter = _env_inputs['clock']
            if not np.array_equal(self.market_clock_counter.valid, valid):
                raise EDataError('E and original Env disagree on valid 1m rows')
        else:
            clock = (self._market_clock_class or
                     legacy_helper("s3_market_clock", self.config.legacy_root).MarketClock)
            self.market_clock_counter = clock(valid)
        self.delayed = bool((pd.DatetimeIndex(self.frame.available_at).asi8 > self.frame.index.asi8).any())
        self.buckets = self._buckets(starts, ends)
        self.complete_buckets = [x for x in self.buckets if x["complete"]]
        self.bucket_end_ns = np.array([local_timestamp(x["bucket_end"]).value for x in self.complete_buckets], dtype=np.int64)
        self.bucket_available_ns = np.array([local_timestamp(x["available_at"]).value for x in self.complete_buckets], dtype=np.int64)
        self.features = None
        if compute_features:
            self._features(_env_inputs['features'] if _env_inputs is not None else None)

    @classmethod
    def from_env(cls, raw, schedule, features, market_clock, bobs, **helpers):
        """Extend the Env's normalized market, original features and clock.

        Only E visibility metadata is added to a shallow market view. The
        original frame/account stay untouched; no CSV read, OHLC normalization,
        feature rebuild or second valid-minute index occurs here.
        """
        frame = raw[list(RAW_FIELDS)+['symbol','trading_date','bob']].copy(deep=False)
        frame['bob'] = bobs
        frame['trading_date'] = pd.to_datetime(frame.trading_date.astype(str),
            format='mixed').dt.strftime('%Y-%m-%d')
        arrivals = raw['available_at'] if 'available_at' in raw else raw.index
        arrivals = _index(arrivals,schedule['timezone'],helpers.get('tz_normalizer'))
        frame['available_at'] = pd.DatetimeIndex(np.maximum(raw.index.asi8,arrivals.asi8)
            ).tz_localize('UTC').tz_convert(schedule['timezone'])
        frame['_duration'] = (raw.index-bobs).total_seconds()/60
        frame['_legal'] = market_clock.valid
        return cls(raw,schedule,_env_inputs=dict(frame=frame,features=features,clock=market_clock),
                   **helpers)

    def _features(self, prepared=None):
        if self.features is None:
            f = self.frame.loc[self.good & self.frame._legal.to_numpy()]
            cols = list(LEGACY_FIELDS) + ["ref_close_t", "cumVWAP_t", "session_high_t", "session_low_t", "dC_minus_cumVWAP_t"]
            calc = (prepared if prepared is not None else
                    self._feature_function(f, self.clock) if self._feature_function
                    else legacy_features(f, self.clock, self.config.legacy_root)) if len(f) else None
            if calc is not None and not calc.index.equals(f.index):
                raise EDataError('E and Env market feature positions differ')
            self.features = (calc[cols].copy() if calc is not None
                             else pd.DataFrame(index=f.index, columns=cols))
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
        # A bucket's historic volume/OI features are fixed as of that bucket's
        # *own* end, not recomputed using previous buckets that arrive later.
        # pending is ordered by arrival; available is ordered by bucket end.
        # This also handles delayed data arriving out of chronological order.
        pending = []
        available = []
        for item in result:
            if not item["complete"]:
                continue
            end_ns = local_timestamp(item["bucket_end"]).value
            while pending and pending[0][0] <= end_ns:
                _, prior_end_ns, prior = heappop(pending)
                insort(available, (prior_end_ns, prior))
            history = [prior for _, prior in available[-20:]]
            previous = available[-1][1] if available else None
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
            arrival_ns = local_timestamp(item["available_at"]).value
            heappush(pending, (arrival_ns, end_ns, item))
        return result

    def macd_snapshot(self, cutoff):
        """One causal prefix, sharing this market's validated calendar and rows.

        Cached per-cursor outputs make arbitrary queries and cold reconstruction
        deterministic. The cache is private to each MarketData instance.
        """
        macd = legacy_helper("macd_5m_causal", self.config.legacy_root)
        CausalMACD5m, CONTRACT = macd.CausalMACD5m, macd.CONTRACT
        cutoff = local_timestamp(cutoff)
        through = int(self.frame.index.searchsorted(cutoff, side='right'))-1
        if not hasattr(self, '_macd_cache'):
            self._macd_cache = {}
            self._macd_stream = CausalMACD5m()
            self._macd_through = -1
            self._macd_anchor_close = None
            # _calendar already rejected overlapping dated minute intervals.
            # Map each row once rather than scanning every historic session
            # and constructing a mixed Pandas Series on every prefix advance.
            self._macd_sessions = sorted((local_timestamp(s['start']), local_timestamp(s['end']))
                                        for s in self.schedule['dated_sessions'])
            starts = np.array([a.value for a,b in self._macd_sessions],np.int64)
            ends = np.array([b.value for a,b in self._macd_sessions],np.int64)
            session_index = np.searchsorted(starts,self.frame.index.asi8,side='left')-1
            valid = (session_index>=0) & (self.frame.index.asi8<=ends[np.maximum(session_index,0)])
            self._macd_session_index = np.where(valid,session_index,-1)
            self._macd_close = self.frame.Close.to_numpy(copy=True)
            self._macd_good = self.frame._good.to_numpy(copy=True)
            self._macd_legal = self.frame._legal.to_numpy(copy=True)
            self._macd_duration = self.frame._duration.to_numpy(copy=True)
            self._macd_available = pd.DatetimeIndex(self.frame.available_at)
        for i in range(self._macd_through+1, through+1):
            t = self.frame.index[i]; available = self._macd_available[i]
            if available > cutoff:
                raise EDataError('MACD unavailable prefix minute')
            if not self._macd_good[i]:
                raise EDataError('MACD invalid visible minute')
            session = int(self._macd_session_index[i])
            closing_mark = (not self._macd_legal[i] and self._macd_duration[i] == 0 and
                            session>=0 and t == self._macd_sessions[session][1])
            if self._macd_legal[i] or closing_mark:
                if session<0:
                    raise EDataError('MACD untrusted dated session')
                a,b = self._macd_sessions[session]
                ordinal = int((t-a).total_seconds()/60)-1
                bucket = self.contract+':'+a.isoformat()+':'+str(ordinal//5)
                try:
                    if self._macd_legal[i]:
                        self._macd_stream.push(float(self._macd_close[i]), bucket=bucket, session=a.isoformat(),
                            slot=ordinal%5+1, time_ns=t.value, available_ns=available.value,
                            session_end_ns=b.value)
                        self._macd_anchor_close = float(self._macd_close[i])
                    else:
                        self._macd_stream.observe_interval_end(session=a.isoformat(),time_ns=t.value,
                            available_ns=available.value,session_end_ns=b.value)
                except ValueError as exc:
                    prior=self._macd_stream
                    context=dict(type='MACD_REJECTED',contract=self.contract,cursor=i,
                        trading_date=str(self.frame.trading_date.iloc[i]),time=t.isoformat(),
                        previous_time_ns=prior.last_time,previous_session=prior.session,
                        previous_interval_end_ns=prior.session_end_ns,
                        previous_bucket=prior.bucket,previous_slot=prior.slot,
                        session_start=a.isoformat(),session_end=b.isoformat(),bucket=bucket,
                        slot=ordinal%5+1,expected_slot=prior.slot+1 if bucket==prior.bucket else 1,
                        source_valid=bool(self._macd_good[i]),valid_minute=bool(self._macd_legal[i]),
                        available_at=available.isoformat())
                    raise EDataError(str(exc)+'; context='+json.dumps(context,sort_keys=True),[context]) from exc
            state = self._macd_stream.snapshot()
            value = np.asarray(self._macd_stream.output, np.float32)
            if not np.isfinite(value).all():
                raise EDataError('MACD scale overflow; no silent clipping')
            self._macd_cache[i] = (value, dict(algorithm=CONTRACT,
                raw=self._macd_stream.raw.copy(), anchor_close=self._macd_anchor_close,
                bucket_id=self._macd_stream.bucket, last_committed_bucket=self._macd_stream.last_committed,
                completed_buckets=self._macd_stream.completed, state=state,
                visible_cutoff=t.isoformat(), available_at=available.isoformat()))
            self._macd_through = i
        if through < 0:
            raise EDataError('MACD no visible source')
        value, meta = self._macd_cache[through]
        return value.copy(), deepcopy(meta)

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
            view = MarketData(visible, self.schedule, self.config,
                feature_function=self._feature_function,
                market_clock_class=self._market_clock_class,
                tz_normalizer=self._tz_normalizer,
                ohlc_normalizer=self._ohlc_normalizer)
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
        # Only the index/length are needed below. Avoid materializing a mixed
        # dtype DataFrame for every decision; feature rows already live in the
        # unchanged prepared arrays.
        window_index = f.index.take(raw_indices)
        window_size = len(raw_indices)
        old = np.zeros((60, 18), np.float32)
        if window_size:
            old[-window_size:] = self.obs_array[feature_start:feature_end]
        if decision.market_seq is not None:
            supplied = np.asarray(decision.market_seq)
            if supplied.shape != (60, 18) or supplied.dtype != np.float32 or not np.array_equal(old, supplied):
                raise EDataError("Original Env market_seq does not match unchanged 18-channel reference")
        anchor, anchor_source, anchor_time, anchor_arrival = self._anchor(through,feature_end,cutoff)
        shared = old.copy()
        audit = np.zeros((60, 4), np.float32) if self.config.audit_1m else None
        if window_size:
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
                shared[-window_size:, LEGACY_FIELDS.index(name)] = values[name].astype(np.float32)
            if audit is not None:
                audit[-window_size:] = np.log(prices / anchor).astype(np.float32)
        indices = np.flatnonzero((self.bucket_end_ns <= cutoff.value) & (self.bucket_available_ns <= cutoff.value))[-24:]
        buckets = [self.complete_buckets[i] for i in indices]
        context = np.zeros((24, 7), np.float32)
        ready = [i for i, item in enumerate(buckets) if item["feature_ready"]]
        if ready:
            selected = [buckets[i] for i in ready]
            rows = np.asarray(ready) + 24 - len(buckets)
            # Keep float64 log/clip followed by the original float32 store;
            # padding and unready buckets retain their exact all-zero rows.
            context[rows, :4] = np.log(np.array([
                [item[k] for k in RAW_FIELDS[:4]] for item in selected]) / anchor)
            context[rows, 4] = np.clip(
                [item["vol_relative_raw"] for item in selected], *self.config.volume_clip)
            context[rows, 5] = np.clip(
                [item["oi_change_raw"] for item in selected], *self.config.oi_clip)
            context[rows, 6] = 1
        for array in (context, old, shared, audit):
            if array is not None and not np.isfinite(array).all():
                raise EDataError("Nonfinite snapshot; no nan_to_num repair allowed")
        meta = dict(source_contract=self.contract, visible_cutoff=cutoff.isoformat(),
            decision_time=local_timestamp(decision.decision_time or cutoff).isoformat(),
            price_anchor=anchor, anchor_source=anchor_source, anchor_source_time=anchor_time,
            anchor_available_at=anchor_arrival, minute_window_times=[t.isoformat() for t in window_index],
            minute_window_sha256=hashlib.sha256(old.tobytes()).hexdigest(),
            context_bucket_ids=[x["bucket_id"] for x in buckets],
            context_bucket_ends=[x["bucket_end"] for x in buckets],
            context_available_at=[x["available_at"] for x in buckets],
            context_valid=[bool(x["feature_ready"]) for x in buckets],
            context_feature_provenance=[dict(bucket_id=x["bucket_id"], feature_ready=x["feature_ready"],
                vol_relative_raw=x["vol_relative_raw"], oi_change_raw=x["oi_change_raw"],
                volume_baseline_bucket_ids=x["volume_baseline_bucket_ids"],
                previous_OI_bucket_id=x["previous_OI_bucket_id"]) for x in buckets],
            context_padding=24 - len(buckets), minute_padding=60 - window_size)
        return context, old, shared, anchor, audit, meta

    def _anchor(self, through, feature_end, cutoff):
        if feature_end:
            source_feature = int(self.ref_source[feature_end-1])
            source_raw = int(self.market_clock_counter.rows[source_feature])
            source_time = self.frame.index[source_raw]
            value = float(self.frame.Close.iat[source_raw])
            arrival = self.frame.available_at.iat[source_raw]
            ref = float(self.aux_array[feature_end-1,0])
            if value == ref and source_time <= cutoff and arrival <= cutoff:
                last_raw = int(self.market_clock_counter.rows[feature_end-1])
                kind = ("legacy_first_observed_close" if self.frame.trading_date.iat[source_raw] == self.frame.trading_date.iat[last_raw]
                        else "legacy_previous_trading_day_close")
                return value, kind, source_time.isoformat(), arrival.isoformat()
        # Zero-duration opening marks may expose a price, but never a future
        # first valid candle. An unknown positive price is not fabricated.
        value = float(self.frame.Close.iat[through])
        if not np.isfinite(value) or value <= 0:
            raise EDataError("No legal observed price for anchor fallback")
        return value, "observed_close_fallback", self.frame.index[through].isoformat(), self.frame.available_at.iat[through].isoformat()


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
