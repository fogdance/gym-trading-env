# tests/unit/test_bookkeeping_conservation.py
import numpy as np
import pandas as pd
import pytest

from gym_trading_env.envs.trading_env import CustomTradingEnv
from gym_trading_env.envs.action import ForexCode
from gym_trading_env.utils.decimal_util import D, D0

pytestmark = pytest.mark.unit


@pytest.fixture
def env():
    idx = pd.date_range("2020-01-01 21:01:00", periods=120, freq="min")
    c = 1.1 + np.linspace(0, 0.0010, 120)
    df = pd.DataFrame({"Date": idx, "Open": c, "High": c, "Low": c, "Close": c, "Volume": 1.0})
    e = CustomTradingEnv(df=df, config_path="tests/test.yaml")
    e.config.training.randomize_start = False
    e.reset()
    yield e
    e.close()


# ----------------- ledger/journal 适配层（按你工程实际字段微调） -----------------

LEDGER_ATTRS = ("ledger", "gl", "general_ledger", "book", "accounting")
ENTRY_ATTRS = ("entries", "journal", "txns", "transactions")
POSTINGS_ATTRS = ("postings", "lines", "splits")


def _get_ledger(env):
    for a in LEDGER_ATTRS:
        obj = getattr(env, a, None)
        if obj is None:
            continue
        # unwrap: env.accounting.ledger / env.book.ledger ...
        for inner in ("ledger", "gl", "general_ledger", "book"):
            if hasattr(obj, inner):
                return getattr(obj, inner)
        return obj
    raise AssertionError(f"Cannot find ledger on env. Tried attrs={LEDGER_ATTRS}")


def _get_entries(ledger):
    for a in ENTRY_ATTRS:
        x = getattr(ledger, a, None)
        if x is None:
            continue
        # journal 可能是 list，也可能是对象（带 .entries）
        if isinstance(x, list) or isinstance(x, tuple):
            return list(x)
        if hasattr(x, "entries"):
            return list(x.entries)
        return list(x)
    raise AssertionError(f"Cannot find journal entries on ledger. Tried attrs={ENTRY_ATTRS}")


def _get_postings(entry):
    for a in POSTINGS_ATTRS:
        ps = getattr(entry, a, None)
        if ps is None:
            continue
        return list(ps)
    raise AssertionError(f"Cannot find postings on entry. Tried attrs={POSTINGS_ATTRS}")


def _posting_amount(p):
    """
    统一把 posting 转成一个“带符号金额”（debit 为 +，credit 为 -）
    兼容两类实现：
      - p.amount 已经是带符号
      - p.debit / p.credit 分列
    """
    if hasattr(p, "amount"):
        return D(str(p.amount))

    if hasattr(p, "debit") and hasattr(p, "credit"):
        return D(str(p.debit)) - D(str(p.credit))

    raise AssertionError("Unknown posting shape. Expect amount or (debit, credit).")


def _assert_entry_balanced(entry):
    total = D0
    for p in _get_postings(entry):
        total += _posting_amount(p)
    assert total == D0, f"Journal entry not balanced: sum(postings)={total} entry={entry}"


def _trial_balance_sum(ledger):
    """
    如果 ledger 暴露了 accounts/balances，就做一个“全科目余额求和”
    在复式里，所有科目余额（debit-credit 口径）总和应为 0。
    """
    accounts = getattr(ledger, "accounts", None)
    if accounts is None:
        return None

    total = D0
    # accounts 可能是 dict{name->Account} 或 list[Account]
    it = accounts.values() if hasattr(accounts, "values") else accounts
    for acc in it:
        bal = getattr(acc, "balance", None)
        if callable(bal):
            bal = bal()
        if bal is None and hasattr(acc, "get_balance"):
            bal = acc.get_balance()
        if bal is None:
            continue
        total += D(str(bal))
    return total


def _account_balance_by_name_contains(ledger, needle: str):
    """
    用名字模糊找科目（你可按实际命名把 needle 换成更精确的）
    """
    accounts = getattr(ledger, "accounts", None)
    if accounts is None:
        return None

    needle = needle.lower()
    it = accounts.items() if hasattr(accounts, "items") else [(getattr(a, "name", ""), a) for a in accounts]
    for name, acc in it:
        n = str(name or getattr(acc, "name", "")).lower()
        if needle in n:
            bal = getattr(acc, "balance", None)
            if callable(bal):
                return D(str(bal()))
            if hasattr(acc, "get_balance"):
                return acc.get_balance()
            return D(str(bal))
    return None


# ----------------- 核心测试：开/平仓的复式闭合与关键流向 -----------------

def test_long_open_close_entries_balanced_and_flows(env):
    ledger = _get_ledger(env)
    entries0 = _get_entries(ledger)
    n0 = len(entries0)

    spread = env.config.trading.spread
    fee_on_close = bool(env.config.trading.is_round_turn)

    # 记录一些可能的关键科目余额（如果你 ledger 命名不同，这里 needle 改一下）
    user_cash0 = _account_balance_by_name_contains(ledger, "user:cash") or _account_balance_by_name_contains(ledger, "cash")
    broker_fee0 = _account_balance_by_name_contains(ledger, "broker:fee") or _account_balance_by_name_contains(ledger, "fee")
    margin0 = _account_balance_by_name_contains(ledger, "margin")  # 可能存在，也可能没有（memo）

    # ---- OPEN ----
    rc = env._long_open(price=D("1.1000"), spread=spread, slot=0)
    assert rc == ForexCode.SUCCESS

    entries1 = _get_entries(ledger)
    new_entries_open = entries1[n0:]
    assert len(new_entries_open) > 0, "OPEN should create at least one journal entry"

    for e in new_entries_open:
        _assert_entry_balanced(e)

    # 试算平衡表（如果暴露了 accounts）
    tb_sum = _trial_balance_sum(ledger)
    if tb_sum is not None:
        assert tb_sum == D0, f"Trial balance must sum to 0, got {tb_sum}"

    # 关键流向（弱约束：存在这些科目时才断言）
    user_cash1 = _account_balance_by_name_contains(ledger, "user:cash") or _account_balance_by_name_contains(ledger, "cash")
    broker_fee1 = _account_balance_by_name_contains(ledger, "broker:fee") or _account_balance_by_name_contains(ledger, "fee")
    margin1 = _account_balance_by_name_contains(ledger, "margin")

    # margin 可能是“入账重分类”(cash->margin) 或 “不入账memo”，两种都允许
    if margin0 is not None and margin1 is not None:
        assert margin1 >= margin0

    # fee：如果 broker fee 科目存在，则应增加；user cash（若存在）应减少（至少不增加）
    if broker_fee0 is not None and broker_fee1 is not None:
        assert broker_fee1 >= broker_fee0
    if user_cash0 is not None and user_cash1 is not None:
        assert user_cash1 <= user_cash0

    # 仓位应存在
    pos0 = env.position_manager.get_position(0, is_long=True)
    assert pos0 is not None

    # ---- CLOSE ----
    n1 = len(entries1)
    rc = env._long_close(price=D("1.1000"), spread=spread, slot=0)
    assert rc == ForexCode.SUCCESS

    entries2 = _get_entries(ledger)
    new_entries_close = entries2[n1:]
    assert len(new_entries_close) > 0, "CLOSE should create at least one journal entry"

    for e in new_entries_close:
        _assert_entry_balanced(e)

    tb_sum = _trial_balance_sum(ledger)
    if tb_sum is not None:
        assert tb_sum == D0, f"Trial balance must sum to 0, got {tb_sum}"

    # 平仓后回到 flat
    assert env.position_manager.no_position() is True

    # 如果 margin 入账，则应释放回去（趋近于 0 或回到初始）
    margin2 = _account_balance_by_name_contains(ledger, "margin")
    if margin0 is not None and margin2 is not None:
        assert margin2 == margin0 or margin2 == D0

    # 如果 fee_on_close，则 broker fee 应进一步增加（存在科目时才断言）
    broker_fee2 = _account_balance_by_name_contains(ledger, "broker:fee") or _account_balance_by_name_contains(ledger, "fee")
    if broker_fee1 is not None and broker_fee2 is not None:
        if fee_on_close:
            assert broker_fee2 >= broker_fee1
