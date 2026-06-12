# src/gym_trading_env/utils/trade_util.py

from decimal import Decimal, ROUND_HALF_UP
from gym_trading_env.envs.position import Position
from gym_trading_env.envs.action import Action, TargetPos

def calc_unrealized_pnl(current_price: Decimal,  pos: Position, lot_size: Decimal, long: bool):
    if pos is None:
        return Decimal('0')
    
    if long:
        return (current_price - pos.entry_price) * pos.size * lot_size
    else:
        return (pos.entry_price - current_price) * pos.size * lot_size

def _act_i(env, a: Action) -> int:
    if a in env.valid_actions:
        return env.valid_actions.index(a)
    if a == Action.HOLD:
        target = env._current_target()
    elif a in (Action.FLIP_SHORT_TO_LONG, Action.LONG_OPEN, Action.LONG_OPEN0, Action.LONG_OPEN1):
        target = TargetPos.LONG
    elif a in (Action.FLIP_LONG_TO_SHORT, Action.SHORT_OPEN, Action.SHORT_OPEN0, Action.SHORT_OPEN1):
        target = TargetPos.SHORT
    elif a in (
        Action.EMPTY,
        Action.LONG_CLOSE,
        Action.LONG_CLOSE0,
        Action.LONG_CLOSE1,
        Action.SHORT_CLOSE,
        Action.SHORT_CLOSE0,
        Action.SHORT_CLOSE1,
    ):
        target = TargetPos.FLAT
    else:
        raise ValueError(f"Action cannot be represented as a target position: {a}")
    return env.valid_actions.index(target)

def step_wrapper(env, a: Action):
    return env.step(_act_i(env, a))
