"""Versioned single-position reward composed from the audited V2 event scorer.

Future trade outcomes are never inputs. Only observed bars and committed fills
are accepted. This is an event reward, not an account or execution engine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
import math

from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research.fg_reward_candidates import CandidateConfig, FGRewardScorer
from gym_trading_env.utils.s3_market_clock import PROTOCOL


SCHEMA = "FG_SINGLE3_REWARD_MARKET_MINUTE_V2"
QUALITY_TP_SCHEMA = "FG_SINGLE3_REWARD_MARKET_MINUTE_TP_QUALITY_V1"
CANDIDATES = {"C0": (0., .4, 3.), "C1": (.5, .4, 3.),
              "C2": (1., .4, 3.), "C3": (1., .4, 6.)}


@dataclass(frozen=True)
class Single3RewardConfig:
    candidate: str = "C2"
    beta: float = 1.
    rho: float = .4
    L: float = 3.
    schema: str = SCHEMA
    clock_protocol: str = PROTOCOL

    def __post_init__(self):
        if (self.schema not in (SCHEMA, QUALITY_TP_SCHEMA) or self.clock_protocol != PROTOCOL or
                self.candidate not in CANDIDATES or
                (self.schema == QUALITY_TP_SCHEMA and self.candidate != 'C1') or
                any(not math.isfinite(x) for x in (self.beta, self.rho, self.L)) or
                (self.beta, self.rho, self.L) != CANDIDATES[self.candidate]):
            raise ValueError("Unknown or inconsistent frozen Single3 reward candidate")

    @classmethod
    def for_candidate(cls, name):
        return cls(name, *CANDIDATES[name])

    @property
    def hash(self):
        return fingerprint(asdict(self))


class Single3Reward:
    def __init__(self, config):
        self.config = config
        self.v2_config = CandidateConfig(beta=config.beta, rho=config.rho, clock_protocol=PROTOCOL)
        self.tp_profile = 'QUALITY_TP_V1' if config.schema == QUALITY_TP_SCHEMA else 'LEGACY_C1'
        self.scorer = FGRewardScorer(self.v2_config, tp_profile=self.tp_profile)
        self.total = 0.

    @property
    def rewarded(self):
        return set(self.scorer.closed)

    def step(self, bar, multiplier):
        bar = dict(bar, eligible_market_minute=False)  # No capacity/PBRS experiment in this protocol.
        if len(self.scorer.active) + len(bar.get("entries", ())) > 1:
            raise ValueError("Single3 reward cannot score simultaneous entry batches")
        # Validate financial attributions before advancing the V2 state machine.
        for event in bar.get("exits", ()):
            if "net_trade_pnl" not in event or not Decimal(event["net_trade_pnl"]).is_finite():
                raise ValueError("Single3 exit requires actual net trade PnL including both fees")
        result = self.scorer.step(bar)
        sl = other = 0.
        events = []
        for entry_id in result["penalized"]:
            position = self.scorer.active.get(entry_id) or self.scorer.closed[entry_id]
            events.append(dict(entry_id=entry_id, reason="FIRST_ADVERSE10",
                               time=bar["bar_end"], effective_bar=position.adverse_first_bar,
                               reward=position.early_penalty, observed=True))
        for event in bar.get("exits", ()):
            reason = event["reason"]
            if reason == "TAKE_PROFIT":
                value = self.scorer.closed[event["entry_id"]].timeboxed_tp
            elif reason == "STOP_LOSS":
                value = -self.config.L
                sl += value
            else:
                loss = max(Decimal(0), -Decimal(event["net_trade_pnl"]))
                value = -self.config.L * float(min(Decimal(1), loss / (3 * Decimal(str(multiplier)) * 30)))
                other += value
            events.append(dict(event, reward=value))
        value = result["v2"] + sl + other
        self.total += value
        return dict(tp=result["tp_timeboxed"], early=result["early"], sl=sl,
                    other=other, total=value, legacy_tp=result["v0"],
                    early_first5=result["early_first5"], early_next10=result["early_next10"],
                    penalized=result["penalized"], intrabar_ambiguous=result["intrabar_ambiguous"],
                    events=events)

    def snapshot(self):
        return dict(schema=SCHEMA, config=asdict(self.config),
                    scorer=self.scorer.snapshot(), total=self.total)

    @classmethod
    def restore(cls, state, config):
        if state["schema"] != SCHEMA or state["config"] != asdict(config):
            raise ValueError("Single3 reward checkpoint/config mismatch")
        obj = cls(config)
        obj.scorer = FGRewardScorer.restore(state["scorer"], obj.v2_config, tp_profile=obj.tp_profile)
        if len(obj.scorer.active) > 1:
            raise ValueError("Single3 checkpoint exceeds capacity")
        obj.total = float(state["total"])
        if not math.isfinite(obj.total):
            raise ValueError("Invalid Single3 reward checkpoint total")
        return obj
