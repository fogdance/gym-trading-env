# tests/unit/test_atomic_close_p0.py

from decimal import Decimal
import warnings

from gym_trading_env.utils.decimal_util import D
from gym_trading_env.envs.position import Position
from gym_trading_env.envs.position_manager import PositionManager
from gym_trading_env.envs.account import Account
from gym_trading_env.envs.accounting import Ledger, JournalEntry, Posting, LedgerError


def _make_ledger(cash: Decimal, margin: Decimal):
    """
    Minimal ledger setup for testing atomic close.
    """
    ledger = Ledger()

    user_cash = Account(cash)
    user_margin = Account(margin)

    broker_fee_income = Account(Decimal("0"))
    broker_pnl = Account(Decimal("0"))

    ledger.register("user_cash", user_cash, strict_nonnegative=True)
    ledger.register("user_margin", user_margin, strict_nonnegative=True)
    ledger.register("broker_fee_income", broker_fee_income, strict_nonnegative=False)
    ledger.register("broker_pnl", broker_pnl, strict_nonnegative=False)

    return ledger, user_cash, user_margin, broker_fee_income, broker_pnl


def test_quote_close_long_does_not_mutate_position():
    pm = PositionManager(long_slots=1, short_slots=1)
    p = Position(size=D("1"), entry_price=D("100"), initial_margin=D("200"), open_step=0)
    pm.add_long_position(p, slot=0)

    slot_i, q = pm.quote_close_long(closing_price=D("90"), lot_size=D("1"), slot=0)

    assert slot_i == 0
    # still there
    assert pm.long_positions[0] is not None
    assert pm.long_positions[0].entry_price == D("100")

    # pnl = (90-100)*1*1 = -10
    assert q.pnl == D("-10")
    assert q.released_margin == D("200")
    assert q.closed_size == D("1")
    assert q.entry_price == D("100")


def test_atomic_close_long_ledger_failure_does_not_remove_position():
    """
    Repro of P0:
    - quote first (no mutation)
    - ledger fails
    - MUST NOT remove position (no commit)
    """
    pm = PositionManager(long_slots=1, short_slots=1)
    p = Position(size=D("1"), entry_price=D("100"), initial_margin=D("200"), open_step=0)
    pm.add_long_position(p, slot=0)

    # margin is insufficient => ledger should fail when withdrawing released_margin
    ledger, user_cash, user_margin, broker_fee_income, broker_pnl = _make_ledger(
        cash=Decimal("0"),
        margin=Decimal("0"),
    )

    slot_i, q = pm.quote_close_long(closing_price=D("90"), lot_size=D("1"), slot=0)
    fee = D("0")

    entry = JournalEntry(
        timestamp="t0",
        memo="LONG_CLOSE",
        postings=[
            Posting("user_margin", -q.released_margin),       # will fail (withdraw 200 from 0)
            Posting("broker_pnl", -q.pnl),                    # -(-10)=+10
            Posting("broker_fee_income", +fee),
            Posting("user_cash", +(q.released_margin + q.pnl - fee)),
        ],
    )

    snap = ledger.snapshot()
    try:
        ledger.post(entry)
        assert False, "ledger.post should have raised"
    except LedgerError:
        ledger.restore(snap)

    # IMPORTANT: position must still exist (no commit happened)
    assert pm.long_positions[0] is not None
    assert pm.long_positions[0].entry_price == D("100")

    # balances restored
    assert user_cash.get_balance() == D("0")
    assert user_margin.get_balance() == D("0")


def test_atomic_close_long_success_removes_position_and_records_pnl():
    pm = PositionManager(long_slots=1, short_slots=1)
    p = Position(size=D("1"), entry_price=D("100"), initial_margin=D("200"), open_step=0)
    pm.add_long_position(p, slot=0)

    ledger, user_cash, user_margin, broker_fee_income, broker_pnl = _make_ledger(
        cash=Decimal("0"),
        margin=Decimal("200"),
    )

    slot_i, q = pm.quote_close_long(closing_price=D("90"), lot_size=D("1"), slot=0)
    fee = D("0")

    entry = JournalEntry(
        timestamp="t0",
        memo="LONG_CLOSE",
        postings=[
            Posting("user_margin", -q.released_margin),       # -200
            Posting("broker_pnl", -q.pnl),                    # -(-10)=+10
            Posting("broker_fee_income", +fee),              # +0
            Posting("user_cash", +(q.released_margin + q.pnl - fee)),  # 200-10=190
        ],
    )

    snap = ledger.snapshot()
    try:
        ledger.post(entry)
        pm.commit_close_long(slot_i, quote=q)
    except Exception:
        ledger.restore(snap)
        raise

    assert pm.long_positions[0] is None
    assert pm.closed_trade_profits[-1] == D("-10")

    # cash/margin reflect postings
    assert user_margin.get_balance() == D("0")
    assert user_cash.get_balance() == D("190")
    assert broker_pnl.get_balance() == D("10")


def test_atomic_close_short_ledger_failure_does_not_remove_position():
    pm = PositionManager(long_slots=1, short_slots=1)
    p = Position(size=D("1"), entry_price=D("100"), initial_margin=D("200"), open_step=0)
    pm.add_short_position(p, slot=0)

    ledger, user_cash, user_margin, broker_fee_income, broker_pnl = _make_ledger(
        cash=Decimal("0"),
        margin=Decimal("0"),
    )

    # short close at ask=110 => pnl = (entry - close) = (100-110) = -10
    slot_i, q = pm.quote_close_short(closing_price=D("110"), lot_size=D("1"), slot=0)
    fee = D("0")

    entry = JournalEntry(
        timestamp="t0",
        memo="SHORT_CLOSE",
        postings=[
            Posting("user_margin", -q.released_margin),       # will fail
            Posting("broker_pnl", -q.pnl),                    # -(-10)=+10
            Posting("broker_fee_income", +fee),
            Posting("user_cash", +(q.released_margin + q.pnl - fee)),  # 200-10=190
        ],
    )

    snap = ledger.snapshot()
    try:
        ledger.post(entry)
        assert False, "ledger.post should have raised"
    except LedgerError:
        ledger.restore(snap)

    # position must still exist
    assert pm.short_positions[0] is not None
    assert pm.short_positions[0].entry_price == D("100")


def test_atomic_close_short_success_removes_position_and_records_pnl():
    pm = PositionManager(long_slots=1, short_slots=1)
    p = Position(size=D("1"), entry_price=D("100"), initial_margin=D("200"), open_step=0)
    pm.add_short_position(p, slot=0)

    ledger, user_cash, user_margin, broker_fee_income, broker_pnl = _make_ledger(
        cash=Decimal("0"),
        margin=Decimal("200"),
    )

    slot_i, q = pm.quote_close_short(closing_price=D("110"), lot_size=D("1"), slot=0)
    fee = D("0")

    entry = JournalEntry(
        timestamp="t0",
        memo="SHORT_CLOSE",
        postings=[
            Posting("user_margin", -q.released_margin),
            Posting("broker_pnl", -q.pnl),                    # -(-10)=+10
            Posting("broker_fee_income", +fee),
            Posting("user_cash", +(q.released_margin + q.pnl - fee)),
        ],
    )

    snap = ledger.snapshot()
    try:
        ledger.post(entry)
        pm.commit_close_short(slot_i, quote=q)
    except Exception:
        ledger.restore(snap)
        raise

    assert pm.short_positions[0] is None
    assert pm.closed_trade_profits[-1] == D("-10")
    assert user_margin.get_balance() == D("0")
    assert user_cash.get_balance() == D("190")
    assert broker_pnl.get_balance() == D("10")


