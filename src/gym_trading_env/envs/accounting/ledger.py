# src/gym_trading_env/accounting/ledger.py
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Tuple

from gym_trading_env.envs.account import Account


class LedgerError(Exception):
    pass


@dataclass(frozen=True)
class Posting:
    """
    amount: + 增加该账户余额；- 减少该账户余额
    """
    account: str
    amount: Decimal


@dataclass
class JournalEntry:
    timestamp: Any
    memo: str
    postings: List[Posting]
    meta: Optional[dict] = None

    def validate(self) -> None:
        if not self.postings:
            raise LedgerError("JournalEntry must contain at least one posting.")
        total = sum((p.amount for p in self.postings), Decimal("0"))
        if total != Decimal("0"):
            raise LedgerError(f"Unbalanced JournalEntry: total={total} memo={self.memo}")


class _Acct:
    def __init__(self, acct: Account, strict_nonnegative: bool):
        self.acct = acct
        self.strict_nonnegative = strict_nonnegative

    def get(self) -> Decimal:
        return self.acct.get_balance()

    def set(self, v: Decimal) -> None:
        self.acct.set_balance(v)

    def apply(self, delta: Decimal) -> None:
        if delta == 0:
            return
        if self.strict_nonnegative:
            # 资产类账户：不允许透支
            if delta > 0:
                self.acct.deposit(delta)
            else:
                self.acct.withdraw(-delta)
        else:
            # 允许为负：用 deposit 直接加（可以加负数）
            self.acct.deposit(delta)


class Ledger:
    """
    一个“最小可用”的复式记账总账：
    - 每条 JournalEntry 必须 sum(postings)=0
    - 提供 snapshot/restore 用于原子回滚（交易动作跨资金流+持仓更新）
    """

    def __init__(self):
        self._accounts: Dict[str, _Acct] = {}
        self.entries: List[JournalEntry] = []

    def register(self, name: str, account: Account, *, strict_nonnegative: bool) -> None:
        if name in self._accounts:
            raise LedgerError(f"Account already registered: {name}")
        self._accounts[name] = _Acct(account, strict_nonnegative=strict_nonnegative)

    def snapshot(self) -> Dict[str, Decimal]:
        return {k: v.get() for k, v in self._accounts.items()}

    def restore(self, snap: Dict[str, Decimal]) -> None:
        for k, bal in snap.items():
            if k in self._accounts:
                self._accounts[k].set(bal)

    def post(self, entry: JournalEntry) -> None:
        entry.validate()
        # apply all postings; if any fails -> raise, caller uses snapshot/restore
        for p in entry.postings:
            if p.account not in self._accounts:
                raise LedgerError(f"Unknown account: {p.account}")
        try:
            for p in entry.postings:
                self._accounts[p.account].apply(p.amount)
        except Exception as e:
            raise LedgerError(f"Failed to post entry '{entry.memo}': {e}") from e

        self.entries.append(entry)

    def balances(self) -> Dict[str, Decimal]:
        return {k: v.get() for k, v in self._accounts.items()}

    def total_balance(self) -> Decimal:
        # 这里的“总额”是内部守恒用：注册的所有账户余额求和应保持常数（系统内转账不会改变总和）
        return sum(self.balances().values(), Decimal("0"))
