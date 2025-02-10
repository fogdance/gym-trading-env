# src/gym_trading_env/envs/action.py

from enum import Enum

class Action(Enum):
    HOLD = 0
    LONG_OPEN = 1
    LONG_CLOSE = 2
    SHORT_OPEN = 3
    SHORT_CLOSE = 4

class ForexCode(Enum):
    SUCCESS = 0
    ERROR_HIT_MAX_POSITION = 1
    ERROR_NO_POSITION_TO_CLOSE = 2
    ERROR_NO_ENOUGH_MONEY = 3