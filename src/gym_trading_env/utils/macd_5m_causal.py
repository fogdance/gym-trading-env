"""Versioned EMA state machine. Calendar slots come from the native E market.

No financial execution, calendar inference, future price access or fitted scale.
"""
from __future__ import annotations
import copy
import math

SCHEMA = 'FG_S3_E_MACD5M_CAUSAL_V2'
CONTRACT = dict(schema=SCHEMA, periods=[12, 26, 9], histogram='DIF-Signal',
    initialization='first_complete_close; first_DIF_and_signal_zero',
    warmup='35_complete_buckets_strictly_before_current_bucket',
    scale='1000*raw/current_visible_1m_Close', invalid='all_zero',
    bucket='five_exact_calendar_minutes_in_one_dated_session',
    partial='derive_from_last_committed_EMAs; commit_on_fifth_visible_minute_once',
    continuity='carry_EMAs_across_sessions_and_official_days; reset_on_contract',
    errors='reject_missing_duplicate_out_of_order_or_late_minutes; never_backfill')


def ema(previous, value, period):
    return value if previous is None else previous + 2/(period+1)*(value-previous)


class CausalMACD5m:
    def __init__(self):
        self.fast = self.slow = self.signal = None
        self.completed = 0
        self.bucket = self.session = self.last_time = None
        self.slot = 0
        self.output = [0., 0., 0., 0.]
        self.raw = [0., 0., 0.]
        self.last_committed = None

    def push(self, close, *, bucket, session, slot, time_ns, available_ns):
        if not math.isfinite(close) or close <= 0 or type(slot) is not int or not 1 <= slot <= 5:
            raise ValueError('MACD invalid visible close or calendar slot')
        if available_ns > time_ns:
            raise ValueError('MACD late minute cannot revise committed EMA')
        if self.last_time is not None:
            if time_ns <= self.last_time:
                raise ValueError('MACD duplicate or out-of-order minute')
            if session == self.session and time_ns-self.last_time != 60_000_000_000:
                raise ValueError('MACD missing calendar minute')
            if bucket != self.bucket and self.slot != 5:
                raise ValueError('MACD incomplete previous bucket')
        expected = self.slot+1 if bucket == self.bucket else 1
        if slot != expected:
            raise ValueError('MACD missing bucket prefix')
        prior = self.completed
        f, s = ema(self.fast, close, 12), ema(self.slow, close, 26)
        dif = f-s
        signal = ema(self.signal, dif, 9)
        self.raw = [dif, signal, dif-signal]
        self.output = [*(1000*x/close for x in self.raw), 1.] if prior >= 35 else [0.]*4
        if slot == 5:
            self.fast, self.slow, self.signal = f, s, signal
            self.completed += 1
            self.last_committed = bucket
        self.bucket, self.session, self.slot, self.last_time = bucket, session, slot, time_ns
        return self.output.copy()

    def snapshot(self):
        return dict(schema=SCHEMA, state=copy.deepcopy(self.__dict__))

    @classmethod
    def restore(cls, snapshot):
        if snapshot.get('schema') != SCHEMA or set(snapshot.get('state', {})) != set(cls().__dict__):
            raise ValueError('MACD state schema differs')
        obj = cls(); obj.__dict__.update(copy.deepcopy(snapshot['state']))
        for k in ('fast', 'slow', 'signal'):
            if obj.__dict__[k] is not None and not math.isfinite(obj.__dict__[k]):
                raise ValueError('MACD nonfinite restored EMA')
        if type(obj.completed) is not int or obj.completed < 0 or obj.slot not in range(6):
            raise ValueError('MACD invalid restored counters')
        return obj
