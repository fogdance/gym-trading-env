# src/gym_trading_env/envs/user_accounts.py
from decimal import Decimal
from gym_trading_env.envs.account import Account

class _ReadOnlyAccount:
    """兼容旧代码 .get_balance()，但禁止 deposit/withdraw 以防绕过 Ledger。"""
    def __init__(self, account: Account, name: str):
        self._a = account
        self._name = name

    def get_balance(self) -> Decimal:
        return self._a.get_balance()

    def deposit(self, amount: Decimal):
        raise RuntimeError(f"{self._name} is read-only. Use Ledger postings.")

    def withdraw(self, amount: Decimal):
        raise RuntimeError(f"{self._name} is read-only. Use Ledger postings.")


class UserAccounts:
    """
    单一真相：现金/保证金等“真钱”由 Ledger 驱动（通过同一份 Account 对象）。
    UserAccounts 只保留 pnl 投影 + 持仓等状态，并提供只读余额接口做兼容。
    """
    def __init__(self, ledger, position_manager, cash_account: Account, margin_account: Account, initial_balance=None):
        self.ledger = ledger
        self.position_manager = position_manager
        self._cash_account = cash_account
        self._margin_account = margin_account

        self.unrealized_pnl = Decimal("0.0")  # 投影
        self.realized_pnl = Decimal("0.0")    # 统计字段（可选）

        self.initial_balance = initial_balance

        # 兼容旧代码：还能 .get_balance()
        self._cash_view = _ReadOnlyAccount(self._cash_account, "user_cash")
        self._margin_view = _ReadOnlyAccount(self._margin_account, "user_margin")

    # ---------- positions ----------
    @property
    def long_position(self):
        return self.position_manager.total_long_position() if self.position_manager else Decimal("0.0")

    @property
    def short_position(self):
        return self.position_manager.total_short_position() if self.position_manager else Decimal("0.0")

    # ---------- legacy compatible handles (Account-like, read-only) ----------
    @property
    def cash_balance(self):
        return self._cash_view

    @property
    def used_margin(self):
        return self._margin_view

    @property
    def balance(self):
        # 老代码 env.user_accounts.balance.get_balance()
        return self._cash_view

    # ---------- numeric helpers ----------
    @property
    def cash(self) -> Decimal:
        return self._cash_account.get_balance()

    @property
    def margin(self) -> Decimal:
        return self._margin_account.get_balance()

    def equity(self) -> Decimal:
        return self.cash + self.margin + self.unrealized_pnl

    # ---------- pnl projection ----------
    def update_unrealized_pnl(self, pnl_change: Decimal):
        assert isinstance(pnl_change, Decimal)
        self.unrealized_pnl += pnl_change

    def realize_pnl(self, pnl: Decimal):
        assert isinstance(pnl, Decimal)
        self.realized_pnl += pnl

    # ---------- forbid bypassing ledger ----------
    def allocate_margin(self, amount: Decimal):
        raise RuntimeError("Do not move margin in UserAccounts. Use Ledger postings (user_cash -> user_margin).")

    def release_margin(self, amount: Decimal):
        raise RuntimeError("Do not move margin in UserAccounts. Use Ledger postings (user_margin -> user_cash).")
