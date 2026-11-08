"""Human directions and effective contract terms; never infer direction from prices."""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import pandas as pd
import yaml


def timestamp(value):
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError("Missing timestamp")
    return ts.tz_localize("Asia/Shanghai") if ts.tz is None else ts.tz_convert("Asia/Shanghai")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def read_yaml(path):
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(base, value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(base).parent / p).resolve()


def is_fg(contract):
    import re
    return bool(re.fullmatch(r"CZCE\.FG\d{3,4}", contract))


# A Chinese futures contract name, EXCHANGE.code + year-month (CZCE.FG501, SHFE.rb2505). Which
# product it is, and its own contract names, are the product profile's (envs.entry_product; user
# 2026-11-08: "后续品种可能增加 ... 不然到时候不好维护"), so a new product needs no code here.
CONTRACT_NAME = r"(CZCE|SHFE|DCE|INE|GFEX)\.[A-Za-z]{1,2}\d{3,4}"


def is_entry_contract(contract):
    import re
    return bool(re.fullmatch(CONTRACT_NAME, contract))


def read_direction_text(path):
    """Manual fg.txt dates are inclusive trading dates, not midnight timestamps."""
    import csv
    raw={"timezone":"Asia/Shanghai", "intervals":[], "date_basis":"trading_date"}
    for number, row in enumerate(csv.reader(Path(path).read_text(encoding="utf-8").splitlines()), 1):
        if not row or not any(v.strip() for v in row):
            continue
        if len(row)!=4:
            raise ValueError(f"Direction line {number}: expected contract,start,end,direction")
        symbol,start,end,side=(v.strip() for v in row)
        contract="CZCE."+symbol.upper()
        a,b=timestamp(start),timestamp(end)+pd.Timedelta(days=1)
        if not is_entry_contract(contract) or a>=b or side not in ("LONG","SHORT"):
            raise ValueError(f"Invalid direction line {number}")
        raw["intervals"].append(dict(contract=contract,start=str(a),end=str(b),direction=side))
    DirectionSchedule(raw)
    return raw


class DirectionSchedule:
    def __init__(self, raw):
        if raw.get("timezone") != "Asia/Shanghai":
            raise ValueError("Direction timezone must be Asia/Shanghai")
        self.hash = fingerprint(raw)
        self.date_basis = raw.get("date_basis", "timestamp")
        if self.date_basis not in ("timestamp", "trading_date"):
            raise ValueError("Invalid direction date basis")
        self.intervals = []
        for item in raw.get("intervals", []):
            contract = item["contract"]
            start, end = timestamp(item["start"]), timestamp(item["end"])
            direction = item["direction"]
            if not is_entry_contract(contract) or start >= end or direction not in ("LONG", "SHORT"):
                raise ValueError("Invalid FG direction interval")
            self.intervals.append((contract, start, end, direction))
        self.intervals.sort(key=lambda x: (x[0], x[1]))
        for prev, current in zip(self.intervals, self.intervals[1:]):
            if prev[0] == current[0] and prev[2] > current[1]:
                raise ValueError("Overlapping direction intervals")

    def at(self, contract, time, trading_date=None):
        if self.date_basis == "trading_date":
            if trading_date is None:
                raise ValueError("Direction lookup requires DB trading_date")
            time = trading_date
        time = timestamp(time)
        for symbol, start, end, side in self.intervals:
            if symbol == contract and start <= time < end:
                return side
        return "NO_TRADE"


class ContractTerms:
    """Explicit effective costs, deadlines and actual dated trading sessions."""
    def __init__(self, raw, contract):
        if raw.get("contract") != contract or not is_entry_contract(contract):
            raise ValueError("Contract metadata must match the real FG contract")
        self.hash = fingerprint(raw)
        self.tick = Decimal(str(raw["tick_size"]))
        self.multiplier = Decimal(str(raw["lot_size"]))
        # The env's distances are whole ticks by construction (envs.entry_product: in ticks, times the
        # tick); SP has 2 points a tick (user 2026-11-08: "sp一跳2点").
        if not self.tick.is_finite() or self.tick <= 0:
            raise ValueError("tick_size is a positive number of points")
        if not self.multiplier.is_finite() or self.multiplier <= 0:
            raise ValueError("Invalid contract multiplier")
        self.last_open = timestamp(raw["last_open_at"])
        self.liquidate = timestamp(raw["liquidate_at"])
        if self.last_open > self.liquidate:
            raise ValueError("Opening cutoff exceeds liquidation deadline")
        self.periods = []
        for item in raw["terms"]:
            start, end = timestamp(item["start"]), timestamp(item["end"])
            numbers = {k: Decimal(str(item[k])) for k in
                       ("margin_rate", "open_fee", "close_fee", "spread", "slippage")}
            numbers["close_today_fee"] = Decimal(str(item.get("close_today_fee", item["close_fee"])))
            if start >= end or any(not v.is_finite() or v < 0 for v in numbers.values()):
                raise ValueError("Invalid effective contract terms")
            if not 0 < numbers["margin_rate"] <= 1:
                raise ValueError("Invalid margin rate")
            self.periods.append((start, end, numbers))
        self.periods.sort(key=lambda x: x[0])
        if any(a[1] > b[0] for a, b in zip(self.periods, self.periods[1:])):
            raise ValueError("Overlapping contract terms")
        minutes = []
        sessions = sorted((timestamp(s["start"]), timestamp(s["end"])) for s in raw["sessions"])
        for a, b in sessions:
            if a >= b or a.second or b.second or a.microsecond or b.microsecond:
                raise ValueError("Invalid minute session")
            minutes.extend(pd.date_range(a + pd.Timedelta(minutes=1), b, freq="min"))
        self.calendar = pd.DatetimeIndex(minutes)
        if not len(self.calendar) or self.calendar.has_duplicates:
            raise ValueError("Trading calendar is empty or overlaps")

    def at(self, time):
        time = timestamp(time)
        for start, end, terms in self.periods:
            if start <= time < end:
                return terms
        raise ValueError(f"No effective contract terms at {time}")
