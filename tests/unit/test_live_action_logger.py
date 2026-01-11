# tests/unit/test_live_action_logger.py
import json
from pathlib import Path

import pytest

from gym_trading_env.envs.trading_env import JsonlActionLogger


pytestmark = pytest.mark.unit


def test_jsonl_action_logger_roundtrip(tmp_path: Path):
    """
    JsonlActionLogger:
    - append 多条记录到 jsonl 文件
    - 文件存在 & 行数正确
    - 每行是合法 JSON
    - load_for_day 读出来的记录与写入的完全一致（顺序保持）
    """
    logger = JsonlActionLogger(base_dir=tmp_path)

    symbol = "TEST_SYMBOL"
    trading_day = 20250101

    records = [
        {
            "symbol": symbol,
            "trading_day": trading_day,
            "step": 5,
            "action": 1,
            "action_name": "LONG_OPEN0",
            "action_result": 0,
        },
        {
            "symbol": symbol,
            "trading_day": trading_day,
            "step": 6,
            "action": 0,
            "action_name": "HOLD",
            "action_result": 0,
        },
    ]

    # 写入
    for rec in records:
        logger.append(rec)

    # 文件存在且行数正确
    path = tmp_path / f"{symbol}_{trading_day}.jsonl"
    assert path.exists(), "jsonl 文件没有写出来"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records), "jsonl 行数与写入记录数不一致"

    # 每行是合法 JSON，内容等于原始记录
    parsed_lines = [json.loads(line) for line in lines]
    assert parsed_lines == records

    # load_for_day 返回的记录与原始记录一致
    loaded = logger.load_for_day(symbol, trading_day)
    assert loaded == records
