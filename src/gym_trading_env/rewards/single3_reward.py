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
# V2 (V0.5, user 2026-10-31): each exit pays its realized net trade PnL (both
# fees) divided by one fixed reward_scale_cny; nothing else is rewarded.
NET_PNL_SCHEMA = "FG_SINGLE3_REWARD_NET_PNL_V1"
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
    reward_scale_cny: float | None = None

    def __post_init__(self):
        net = self.schema == NET_PNL_SCHEMA
        if (self.schema not in (SCHEMA, QUALITY_TP_SCHEMA, NET_PNL_SCHEMA) or self.clock_protocol != PROTOCOL or
                self.candidate not in CANDIDATES or
                (self.schema in (QUALITY_TP_SCHEMA, NET_PNL_SCHEMA) and self.candidate != 'C1') or
                any(not math.isfinite(x) for x in (self.beta, self.rho, self.L)) or
                (self.beta, self.rho, self.L) != CANDIDATES[self.candidate] or
                net != (self.reward_scale_cny is not None) or
                (net and not (math.isfinite(self.reward_scale_cny) and self.reward_scale_cny > 0))):
            raise ValueError("Unknown or inconsistent frozen Single3 reward candidate")

    @classmethod
    def for_candidate(cls, name):
        return cls(name, *CANDIDATES[name])

    @property
    def fields(self):
        """The set fields; an unset scale leaves the earlier identity as it was."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @property
    def hash(self):
        return fingerprint(self.fields)


class Single3Reward:
    def __init__(self, config, take_profit_points=5, stop_loss_points=30):
        # TP5 / SL30 unless the env sets other distances (paper trading, user 2026-10-30).
        self.config = config
        self.stop_loss_points = stop_loss_points
        self.v2_config = CandidateConfig(beta=config.beta, rho=config.rho, clock_protocol=PROTOCOL,
                                         take_profit_points=float(take_profit_points))
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
        if self.config.schema == NET_PNL_SCHEMA:
            scale = Decimal(str(self.config.reward_scale_cny))
            events = [dict(event, reward=float(Decimal(event["net_trade_pnl"]) / scale))
                      for event in bar.get("exits", ())]
            value = sum(e["reward"] for e in events)
            self.total += value
            return dict(tp=0., early=0., sl=0., other=0., net_pnl=value, total=value, legacy_tp=0.,
                        early_first5=0., early_next10=0., penalized=[],
                        intrabar_ambiguous=result["intrabar_ambiguous"], events=events)
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
                value = -self.config.L * float(min(Decimal(1), loss / (3 * Decimal(str(multiplier)) * self.stop_loss_points)))
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
        return dict(schema=SCHEMA, config=self.config.fields,
                    scorer=self.scorer.snapshot(), total=self.total)

    @classmethod
    def restore(cls, state, config, take_profit_points=5, stop_loss_points=30):
        if state["schema"] != SCHEMA or state["config"] != config.fields:
            raise ValueError("Single3 reward checkpoint/config mismatch")
        obj = cls(config, take_profit_points, stop_loss_points)
        obj.scorer = FGRewardScorer.restore(state["scorer"], obj.v2_config, tp_profile=obj.tp_profile)
        if len(obj.scorer.active) > 1:
            raise ValueError("Single3 checkpoint exceeds capacity")
        obj.total = float(state["total"])
        if not math.isfinite(obj.total):
            raise ValueError("Invalid Single3 reward checkpoint total")
        return obj
