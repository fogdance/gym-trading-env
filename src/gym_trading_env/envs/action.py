# src/gym_trading_env/envs/action.py

from enum import Enum

class Action(Enum):
    HOLD = 0
    LONG_OPEN = 1
    LONG_CLOSE = 2
    SHORT_OPEN = 3
    SHORT_CLOSE = 4
    POSITION_DOWN = 5
    POSITION_UP = 6
    EMPTY = 7
    LONG_OPEN0 = 8
    LONG_CLOSE0 = 9
    LONG_OPEN1 = 10
    LONG_CLOSE1 = 11
    SHORT_OPEN0 = 12
    SHORT_CLOSE0 = 13
    SHORT_OPEN1 = 14
    SHORT_CLOSE1 = 15
    

class ForexCode(Enum):
    SUCCESS = 0
    ERROR_HIT_MAX_POSITION = 1
    ERROR_NO_POSITION_TO_CLOSE = 2
    ERROR_NO_ENOUGH_MONEY = 3
    ERROR_OPEN_POSITION = 4