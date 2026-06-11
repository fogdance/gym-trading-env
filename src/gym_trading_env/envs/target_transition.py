from dataclasses import dataclass

from gym_trading_env.envs.action import Action, ForexCode, TargetPos


@dataclass(frozen=True)
class TargetTransitionDecision:
    requested_index: int
    current_target: TargetPos
    requested_target: TargetPos
    planned_action: Action
    allowed: bool
    result_code: ForexCode
    reason: str


@dataclass(frozen=True)
class TargetTransitionTable:
    state_version: int
    state_key: tuple
    decisions: tuple[TargetTransitionDecision, ...]

    def for_index(self, action_index: int) -> TargetTransitionDecision:
        return self.decisions[int(action_index)]
