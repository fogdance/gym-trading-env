# src/gym_trading_env/utils/trade_util.py

from decimal import Decimal, ROUND_HALF_UP
from gym_trading_env.envs.position import Position

def calc_unrealized_pnl(current_price: Decimal,  pos: Position, lot_size: Decimal, long: bool):
    if pos is None:
        return Decimal('0')
    
    if long:
        return (current_price - pos.entry_price) * pos.size * lot_size
    else:
        return (pos.entry_price - current_price) * pos.size * lot_size
