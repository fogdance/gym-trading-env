# src/gym_trading_env/envs/action.py

from enum import Enum

class Action(Enum):
    HOLD = 0
    EMPTY = 1
    LONG_OPEN0 = 2
    LONG_CLOSE0 = 3
    LONG_OPEN1 = 4
    LONG_CLOSE1 = 5
    SHORT_OPEN0 = 6
    SHORT_CLOSE0 = 7
    SHORT_OPEN1 = 8
    SHORT_CLOSE1 = 9
    LONG_OPEN = 10
    LONG_CLOSE = 11
    SHORT_OPEN = 12
    SHORT_CLOSE = 13
    POSITION_DOWN = 14
    POSITION_UP = 15
    

class ForexCode(Enum):
    SUCCESS = 0
    ERROR_HIT_MAX_POSITION = 1
    ERROR_NO_POSITION_TO_CLOSE = 2
    ERROR_NO_ENOUGH_MONEY = 3
    ERROR_OPEN_POSITION = 4