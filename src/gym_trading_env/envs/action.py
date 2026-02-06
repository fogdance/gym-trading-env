# src/gym_trading_env/envs/action.py

from enum import Enum, IntEnum
import json
from pathlib import Path

class TargetPos(IntEnum):
    SHORT = 0
    FLAT  = 1
    LONG  = 2
    @property
    def sign(self) -> int:
        # 仅用于少数需要 -1/0/+1 的地方
        return {-1: -1, 0: 0, 1: 1}[self - TargetPos.FLAT]
        # SHORT(0)->-1, FLAT(1)->0, LONG(2)->+1

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
    # 反手 (atomic flip): 平当前仓 + 开反向仓，单步完成
    FLIP_LONG_TO_SHORT = 16  # 多翻空: 平多 + 开空
    FLIP_SHORT_TO_LONG = 17  # 空翻多: 平空 + 开多
    

# 添加错误码后，更新_is_invalid_action
class ForexCode(Enum):
    SUCCESS = 0
    ERROR_HIT_MAX_POSITION = 1
    ERROR_NO_POSITION_TO_CLOSE = 2
    ERROR_NO_ENOUGH_MONEY = 3
    ERROR_OPEN_POSITION = 4
    ERROR_MARKET_CLOSED = 5
    ERROR_BLOCKED_NEAR_EOD = 6
    ERROR_HIT_DAY_MAX_OPEN = 7

class JsonlActionLogger:
    """
    简单的 JSONL 文件 logger：
    - 每个 symbol + trading_day 写一份文件：{base_dir}/{symbol}_{trading_day}.jsonl
    - 每行是一条 JSON 记录（append-only）
    """

    def __init__(self, base_dir: str | Path, logger=None):
        self.base_dir = Path(base_dir).expanduser()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def _file_path(self, symbol: str, trading_day: int) -> Path:
        return self.base_dir / f"{symbol}_{int(trading_day)}.jsonl"

    def append(self, rec: dict):
        symbol = rec.get("symbol")
        trading_day = rec.get("trading_day")
        if symbol is None or trading_day is None:
            if self.logger:
                self.logger.warning("JsonlActionLogger: record missing symbol or trading_day; skip")
            return

        path = self._file_path(symbol, int(trading_day))
        try:
            line = json.dumps(rec, ensure_ascii=False)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            if self.logger:
                self.logger.error(f"JsonlActionLogger.append failed for {path}: {e}")

    def load_for_day(self, symbol: str, trading_day: int):
        path = self._file_path(symbol, int(trading_day))
        if not path.exists():
            return []

        records = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"JsonlActionLogger: bad line in {path}: {e}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"JsonlActionLogger.load_for_day failed for {path}: {e}")
        return records
    