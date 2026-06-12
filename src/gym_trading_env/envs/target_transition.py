from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from gym_trading_env.envs.accounting import JournalEntry
from gym_trading_env.envs.action import Action, ForexCode, TargetPos
from gym_trading_env.envs.position import Position
from gym_trading_env.envs.position_manager import CloseQuote


@dataclass(frozen=True)
class TargetExecutionQuote:
    action: Action
    action_price: Decimal
    timestamp: Any
    ledger_entry: JournalEntry
    exec_price: Decimal
    rpc_price: Decimal
    open_side: TargetPos | None = None
    open_slot: int | None = None
    open_position: Position | None = None
    open_fee: Decimal = Decimal("0")
    close_side: TargetPos | None = None
    close_slot: int | None = None
    close_quote: CloseQuote | None = None
    close_fee: Decimal = Decimal("0")


@dataclass(frozen=True)
class TargetTransitionDecision:
    requested_index: int
    current_target: TargetPos
    requested_target: TargetPos
    planned_action: Action
    allowed: bool
    result_code: ForexCode
    reason: str
    quote: TargetExecutionQuote | None = None

    def __post_init__(self) -> None:
        if self.allowed != (self.result_code == ForexCode.SUCCESS):
            raise ValueError("Decision allowed flag must match result_code")
        if not self.allowed and self.quote is not None:
            raise ValueError("Rejected decision cannot carry an execution quote")
        if self.allowed and self.planned_action != Action.HOLD and self.quote is None:
            raise ValueError("Allowed non-HOLD decision must carry an execution quote")
        if self.quote is not None and self.quote.action != self.planned_action:
            raise ValueError("Execution quote action must match planned action")


@dataclass(frozen=True)
class TargetTransitionTable:
    state_version: int
    state_key: tuple
    action_price: Decimal
    market_open: bool
    decisions: tuple[TargetTransitionDecision, ...]

    def for_index(self, action_index: int) -> TargetTransitionDecision:
        return self.decisions[int(action_index)]
