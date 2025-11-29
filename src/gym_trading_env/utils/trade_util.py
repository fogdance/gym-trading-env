# src/gym_trading_env/utils/trade_util.py

from decimal import Decimal, ROUND_HALF_UP
from gym_trading_env.envs.position import Position
from gym_trading_env.envs.action import Action

def calc_unrealized_pnl(current_price: Decimal,  pos: Position, lot_size: Decimal, long: bool):
    if pos is None:
        return Decimal('0')
    
    if long:
        return (current_price - pos.entry_price) * pos.size * lot_size
    else:
        return (pos.entry_price - current_price) * pos.size * lot_size

def _act_i(env, a: Action) -> int:
    # 将 Action enum 映射成 env.step 需要的离散 index（基于 env.valid_actions）
    return env.valid_actions.index(a)

def step_wrapper(env, a: Action):
    return env.step(_act_i(env, a))