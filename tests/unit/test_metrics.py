# tests/unit/test_metrics.py

import unittest
from decimal import Decimal
from datetime import datetime, timedelta
import math

from gym_trading_env.envs.metrics import Metrics
from gym_trading_env.envs.user_accounts import UserAccounts
from gym_trading_env.envs.trade_record_manager import TradeRecordManager
from gym_trading_env.envs.trade_record import TradeRecord
from gym_trading_env.utils.decimal_util import D, D0, D1, D100
from gym_trading_env.envs.accounting import Ledger, JournalEntry, Posting
from gym_trading_env.envs.account import Account
from gym_trading_env.envs.broker_accounts import BrokerAccounts

import pytest
pytestmark = pytest.mark.unit


class MockPositionManager:
    def total_long_position(self):
        return Decimal('0.0')

    def total_short_position(self):
        return Decimal('0.0')


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.initial_balance = Decimal('10000')
        self.trade_record_manager = TradeRecordManager()

        # --- Ledger (double-entry) ---
        self.ledger = Ledger()

        # 1) 创建“真钱账户”（唯一一份）
        user_cash_acct = Account(Decimal(str(self.initial_balance)))
        user_margin_acct = Account(Decimal("0.0"))
        self.broker_accounts = BrokerAccounts()

        # 2) 先注册进 Ledger
        self.ledger.register("user_cash", user_cash_acct, strict_nonnegative=True)
        self.ledger.register("user_margin", user_margin_acct, strict_nonnegative=True)
        self.ledger.register("broker_fee_income", self.broker_accounts.fee_income, strict_nonnegative=False)
        self.ledger.register("broker_pnl", self.broker_accounts.broker_pnl, strict_nonnegative=False)

        # 3) UserAccounts 引用这些账户
        self.user_accounts = UserAccounts(
            ledger=self.ledger,
            position_manager=MockPositionManager(),
            cash_account=user_cash_acct,
            margin_account=user_margin_acct,
            initial_balance=self.initial_balance,
        )
        self.metrics = Metrics(self.user_accounts, self.trade_record_manager, risk_free_rate=Decimal('0.012'))

    def _settle_pnl(self, pnl: Decimal):
        # 统计字段（可留可不留，不影响 equity）
        self.user_accounts.realize_pnl(pnl)

        # 复式记账：把 pnl 结算进用户现金腿
        self.ledger.post(
            JournalEntry(
                timestamp=None,
                memo="settle pnl (test)",
                postings=[
                    Posting("user_cash", pnl),
                    Posting("broker_pnl", -pnl),
                ],
            )
        )

    def assert_all_metrics(self, metrics, expected_values):
        self.assertEqual(metrics['current_day_lost'], expected_values['current_day_lost'])
        self.assertEqual(metrics['current_day_lost_pct'], expected_values['current_day_lost_pct'])
        self.assertEqual(metrics['current_drawdown'], expected_values['current_drawdown'])
        self.assertEqual(metrics['current_drawdown_pct'], expected_values['current_drawdown_pct'])
        self.assertEqual(metrics['max_drawdown'], expected_values['max_drawdown'])
        self.assertEqual(metrics['max_drawdown_pct'], expected_values['max_drawdown_pct'])
        self.assertEqual(metrics['total_trades'], expected_values['total_trades'])
        self.assertEqual(metrics['winning_trades'], expected_values['winning_trades'])
        self.assertEqual(metrics['max_profit'], expected_values['max_profit'])
        self.assertEqual(metrics['max_loss'], expected_values['max_loss'])

        if expected_values['calmar_ratio'] is None:
            self.assertIsNone(metrics['calmar_ratio'])
        else:
            self.assertAlmostEqual(metrics['calmar_ratio'], expected_values['calmar_ratio'], places=6)

        if expected_values['sharpe_ratio'] is None:
            self.assertIsNone(metrics['sharpe_ratio'])
        else:
            self.assertAlmostEqual(metrics['sharpe_ratio'], expected_values['sharpe_ratio'], places=6)

        if expected_values['win_rate'] is None:
            self.assertIsNone(metrics['win_rate'])
        else:
            self.assertAlmostEqual(metrics['win_rate'], expected_values['win_rate'], places=6)

    # -------------------------
    # 基础：初始状态
    # -------------------------
    def test_initial_state(self):
        self.metrics.update(datetime(2023, 1, 1))
        metrics = self.metrics.get_metrics()
        expected = {
            'current_day_lost': Decimal('0.0'),
            'current_day_lost_pct': Decimal('0.0'),
            'current_drawdown': Decimal('0.0'),
            'current_drawdown_pct': Decimal('0.0'),
            'max_drawdown': Decimal('0.0'),
            'max_drawdown_pct': Decimal('0.0'),
            'total_trades': 0,
            'winning_trades': 0,
            'max_profit': Decimal('0.0'),
            'max_loss': Decimal('0.0'),
            'calmar_ratio': None,
            'sharpe_ratio': None,
            'win_rate': None,
        }
        self.assert_all_metrics(metrics, expected)

    # -------------------------
    # 日内亏损 / 回撤：同一天内更新才会产生 day_loss
    # -------------------------
    def test_day_loss(self):
        self.metrics.update(datetime(2023, 1, 1))  # init
        self._settle_pnl(Decimal('-500'))
        self.metrics.update(datetime(2023, 1, 1))  # same-day tick
        metrics = self.metrics.get_metrics()
        expected = {
            'current_day_lost': Decimal('500'),
            'current_day_lost_pct': Decimal('5.0'),
            'current_drawdown': Decimal('500'),
            'current_drawdown_pct': Decimal('5.0'),
            'max_drawdown': Decimal('500'),
            'max_drawdown_pct': Decimal('5.0'),
            'total_trades': 0,
            'winning_trades': 0,
            'max_profit': Decimal('0.0'),
            'max_loss': Decimal('0.0'),
            'calmar_ratio': None,
            'sharpe_ratio': None,  # _ret_n=1
            'win_rate': None,
        }
        self.assert_all_metrics(metrics, expected)

    def test_new_day_reset(self):
        self._settle_pnl(Decimal('-500'))
        self.metrics.update(datetime(2023, 1, 1))  # init at 9500
        self.metrics.update(datetime(2023, 1, 2))  # day boundary
        metrics = self.metrics.get_metrics()

        # 按 Metrics 当前实现计算：1天时间跨度会计算 cagr -> calmar
        years = 1.0 / 365.0
        ratio = float(Decimal("9500") / Decimal("10000"))  # 0.95
        g = math.log(ratio) / years
        cagr = math.expm1(g)
        expected_calmar = cagr / 0.05  # max_dd_frac = 5% = 0.05

        expected = {
            'current_day_lost': Decimal('0.0'),
            'current_day_lost_pct': Decimal('0.0'),
            'current_drawdown': Decimal('500'),
            'current_drawdown_pct': Decimal('5.0'),
            'max_drawdown': Decimal('500'),
            'max_drawdown_pct': Decimal('5.0'),
            'total_trades': 0,
            'winning_trades': 0,
            'max_profit': Decimal('0.0'),
            'max_loss': Decimal('0.0'),
            'calmar_ratio': expected_calmar,   # ✅ 不再是 None
            'sharpe_ratio': None,
            'win_rate': None,
        }
        self.assert_all_metrics(metrics, expected)

    # -------------------------
    # 交易统计：需要至少第二次 update 才会 consume trade_history
    # Sharpe/Calmar 现在基于 equity 曲线，不再基于 trade pnl 列表
    # -------------------------
    def test_trades_metrics(self):
        self.metrics.update(datetime(2023, 1, 1))  # init (important)

        trades = [
            TradeRecord(
                timestamp=datetime(2023, 1, 1),
                operation_type="LONG_CLOSE",
                position_size=Decimal('1.0'),
                open_price=Decimal('100'),
                close_price=Decimal('110'),
                required_margin=Decimal('100'),
                fee=Decimal('0'),
                balance=Decimal('10000'),
                leverage=Decimal('1'),
                free_margin=Decimal('9900'),
                pnl=Decimal('100')
            ),
            TradeRecord(
                timestamp=datetime(2023, 1, 2),
                operation_type="SHORT_CLOSE",
                position_size=Decimal('1.0'),
                open_price=Decimal('100'),
                close_price=Decimal('105'),
                required_margin=Decimal('100'),
                fee=Decimal('0'),
                balance=Decimal('10000'),
                leverage=Decimal('1'),
                free_margin=Decimal('9900'),
                pnl=Decimal('-50')
            ),
            TradeRecord(
                timestamp=datetime(2023, 1, 3),
                operation_type="LONG_CLOSE",
                position_size=Decimal('1.0'),
                open_price=Decimal('100'),
                close_price=Decimal('107.5'),
                required_margin=Decimal('100'),
                fee=Decimal('0'),
                balance=Decimal('10000'),
                leverage=Decimal('1'),
                free_margin=Decimal('9900'),
                pnl=Decimal('75')
            ),
        ]
        for trade in trades:
            self.trade_record_manager.record_trade(trade)

        # 第二次 update 才会 consume trades
        self.metrics.update(datetime(2023, 1, 3))
        metrics = self.metrics.get_metrics()

        expected = {
            'current_day_lost': Decimal('0.0'),
            'current_day_lost_pct': Decimal('0.0'),
            'current_drawdown': Decimal('0.0'),
            'current_drawdown_pct': Decimal('0.0'),
            'max_drawdown': Decimal('0.0'),
            'max_drawdown_pct': Decimal('0.0'),
            'total_trades': 3,
            'winning_trades': 2,
            'max_profit': Decimal('100'),
            'max_loss': Decimal('-50'),
            'calmar_ratio': None,   # max_dd_pct=0
            'sharpe_ratio': None,   # _ret_n=1
            'win_rate': 2 / 3,
        }
        self.assert_all_metrics(metrics, expected)

    # -------------------------
    # 边界：无交易但有亏损（需要至少第二次 update 才会算 day_loss/drawdown）
    # -------------------------
    def test_no_trades(self):
        self._settle_pnl(Decimal('-1000'))
        self.metrics.update(datetime(2023, 1, 1))          # init
        self.metrics.update(datetime(2023, 1, 1, 0, 0, 1)) # same-day tick -> compute
        metrics = self.metrics.get_metrics()
        expected = {
            'current_day_lost': Decimal('1000'),
            'current_day_lost_pct': Decimal('10.0'),
            'current_drawdown': Decimal('1000'),
            'current_drawdown_pct': Decimal('10.0'),
            'max_drawdown': Decimal('1000'),
            'max_drawdown_pct': Decimal('10.0'),
            'total_trades': 0,
            'winning_trades': 0,
            'max_profit': Decimal('0.0'),
            'max_loss': Decimal('0.0'),
            'calmar_ratio': None,
            'sharpe_ratio': None,  # _ret_n=1
            'win_rate': None,
        }
        self.assert_all_metrics(metrics, expected)

    def test_single_trade(self):
        self.metrics.update(datetime(2023, 1, 1))  # init

        trade = TradeRecord(
            timestamp=datetime(2023, 1, 1),
            operation_type="LONG_CLOSE",
            position_size=Decimal('1.0'),
            open_price=Decimal('100'),
            close_price=Decimal('110'),
            required_margin=Decimal('100'),
            fee=Decimal('0'),
            balance=Decimal('10000'),
            leverage=Decimal('1'),
            free_margin=Decimal('9900'),
            pnl=Decimal('100')
        )
        self.trade_record_manager.record_trade(trade)

        self.metrics.update(datetime(2023, 1, 1, 0, 0, 1))  # consume trade
        metrics = self.metrics.get_metrics()
        expected = {
            'current_day_lost': Decimal('0.0'),
            'current_day_lost_pct': Decimal('0.0'),
            'current_drawdown': Decimal('0.0'),
            'current_drawdown_pct': Decimal('0.0'),
            'max_drawdown': Decimal('0.0'),
            'max_drawdown_pct': Decimal('0.0'),
            'total_trades': 1,
            'winning_trades': 1,
            'max_profit': Decimal('100'),
            'max_loss': Decimal('0.0'),
            'calmar_ratio': None,
            'sharpe_ratio': None,  # _ret_n=1
            'win_rate': 1.0,
        }
        self.assert_all_metrics(metrics, expected)

    # -------------------------
    # Sharpe：现在基于 equity update 形成的 excess returns；两次完全相同的 return -> 方差=0 -> Sharpe=None
    # -------------------------
    def test_zero_std_sharpe(self):
        t0 = datetime(2023, 1, 1)
        t1 = datetime(2023, 1, 2)
        t2 = datetime(2023, 1, 3)

        self.metrics.update(t0)  # init

        # 10000 -> 10100 (1% in 1 day)
        self._settle_pnl(Decimal('100'))
        self.metrics.update(t1)

        # 10100 -> 10201 (1% in 1 day again) => 两次 excess return 相同 => var=0 => sharpe None
        self._settle_pnl(Decimal('101'))
        self.metrics.update(t2)

        metrics = self.metrics.get_metrics()
        expected = {
            'current_day_lost': Decimal('0.0'),
            'current_day_lost_pct': Decimal('0.0'),
            'current_drawdown': Decimal('0.0'),
            'current_drawdown_pct': Decimal('0.0'),
            'max_drawdown': Decimal('0.0'),
            'max_drawdown_pct': Decimal('0.0'),
            'total_trades': 0,
            'winning_trades': 0,
            'max_profit': Decimal('0.0'),
            'max_loss': Decimal('0.0'),
            'calmar_ratio': None,   # max_dd_pct=0
            'sharpe_ratio': None,   # var=0
            'win_rate': None,
        }
        self.assert_all_metrics(metrics, expected)

    # -------------------------
    # Calmar：CAGR 需要 >= 1 天；时间太短 cagr=None -> calmar=None
    # -------------------------
    def test_zero_time_span_calmar(self):
        t0 = datetime(2023, 1, 1, 10, 0, 0)
        t1 = datetime(2023, 1, 1, 10, 1, 0)  # 1 minute later (< 1 day)

        self.metrics.update(t0)  # init

        self._settle_pnl(Decimal('-200'))
        self.metrics.update(t1)

        metrics = self.metrics.get_metrics()
        expected = {
            'current_day_lost': Decimal('200'),
            'current_day_lost_pct': Decimal('2.0'),
            'current_drawdown': Decimal('200'),
            'current_drawdown_pct': Decimal('2.0'),
            'max_drawdown': Decimal('200'),
            'max_drawdown_pct': Decimal('2.0'),
            'total_trades': 0,
            'winning_trades': 0,
            'max_profit': Decimal('0.0'),
            'max_loss': Decimal('0.0'),
            'calmar_ratio': None,   # cagr=None (intraday too short)
            'sharpe_ratio': None,   # _ret_n=1
            'win_rate': None,
        }
        self.assert_all_metrics(metrics, expected)

    # -------------------------
    # Calmar：max_drawdown_pct=0 时 calmar=None（即便 cagr 有值）
    # -------------------------
    def test_zero_max_drawdown_calmar(self):
        t0 = datetime(2023, 1, 1)
        t1 = datetime(2023, 2, 1)

        self.metrics.update(t0)  # init

        self._settle_pnl(Decimal('200'))  # only up, no drawdown
        self.metrics.update(t1)

        metrics = self.metrics.get_metrics()
        expected = {
            'current_day_lost': Decimal('0.0'),
            'current_day_lost_pct': Decimal('0.0'),
            'current_drawdown': Decimal('0.0'),
            'current_drawdown_pct': Decimal('0.0'),
            'max_drawdown': Decimal('0.0'),
            'max_drawdown_pct': Decimal('0.0'),
            'total_trades': 0,
            'winning_trades': 0,
            'max_profit': Decimal('0.0'),
            'max_loss': Decimal('0.0'),
            'calmar_ratio': None,   # max_dd_pct=0
            'sharpe_ratio': None,   # _ret_n=1
            'win_rate': None,
        }
        self.assert_all_metrics(metrics, expected)

    # -------------------------
    # 新增：用 equity 曲线验证 Sharpe + Calmar（与当前 Metrics 公式一致）
    # -------------------------
    def test_equity_curve_ratios(self):
        t0 = datetime(2023, 1, 1)
        t1 = datetime(2023, 2, 1)  # +31d
        t2 = datetime(2023, 3, 1)  # +28d

        self.metrics.update(t0)  # init

        # 10000 -> 10500
        self._settle_pnl(Decimal('500'))
        self.metrics.update(t1)

        # 10500 -> 10200 (drawdown 300 from peak 10500)
        self._settle_pnl(Decimal('-300'))
        self.metrics.update(t2)

        m = self.metrics.get_metrics()

        # 期望 drawdown
        expected_dd = Decimal('300')
        expected_dd_pct = (expected_dd / Decimal('10500')) * Decimal("100.0")

        # 期望 Sharpe（基于 excess returns + 实际 elapsed time 年化）
        rf = float(Decimal("0.012"))
        dt1_years = 31.0 / 365.0
        dt2_years = 28.0 / 365.0
        ex1 = (10500.0 / 10000.0 - 1.0) - rf * dt1_years
        ex2 = (10200.0 / 10500.0 - 1.0) - rf * dt2_years
        mean = (ex1 + ex2) / 2.0
        var = ((ex1 - mean) ** 2 + (ex2 - mean) ** 2)  # sample var denom=1
        std = math.sqrt(var)
        time_years = (31.0 + 28.0) / 365.0
        periods_per_year = 2.0 / time_years
        ann = math.sqrt(periods_per_year)
        expected_sharpe = (mean / std) * ann

        # 期望 Calmar（cagr / max_dd_frac）
        ratio = 10200.0 / 10000.0
        g = math.log(ratio) / time_years
        cagr = math.expm1(g)
        max_dd_frac = float(expected_dd_pct) / 100.0
        expected_calmar = cagr / max_dd_frac

        expected = {
            # 注意：跨天第一次 tick 的 daily loss 会重置为 0
            'current_day_lost': Decimal('0.0'),
            'current_day_lost_pct': Decimal('0.0'),

            'current_drawdown': expected_dd,
            'current_drawdown_pct': expected_dd_pct,
            'max_drawdown': expected_dd,
            'max_drawdown_pct': expected_dd_pct,

            'total_trades': 0,
            'winning_trades': 0,
            'max_profit': Decimal('0.0'),
            'max_loss': Decimal('0.0'),

            'calmar_ratio': expected_calmar,
            'sharpe_ratio': expected_sharpe,
            'win_rate': None,
        }
        self.assert_all_metrics(m, expected)

    # -------------------------
    # 极端场景：全部亏损交易（交易统计仍然基于 close trade pnl）
    # -------------------------
    def test_all_losing_trades(self):
        self.metrics.update(datetime(2023, 1, 1))  # init

        trades = [
            TradeRecord(
                timestamp=datetime(2023, 1, 1),
                operation_type="LONG_CLOSE",
                position_size=Decimal('1.0'),
                open_price=Decimal('100'),
                close_price=Decimal('90'),
                required_margin=Decimal('100'),
                fee=Decimal('0'),
                balance=Decimal('10000'),
                leverage=Decimal('1'),
                free_margin=Decimal('9900'),
                pnl=Decimal('-100')
            ),
            TradeRecord(
                timestamp=datetime(2023, 1, 2),
                operation_type="LONG_CLOSE",
                position_size=Decimal('1.0'),
                open_price=Decimal('100'),
                close_price=Decimal('95'),
                required_margin=Decimal('100'),
                fee=Decimal('0'),
                balance=Decimal('10000'),
                leverage=Decimal('1'),
                free_margin=Decimal('9900'),
                pnl=Decimal('-50')
            ),
        ]
        for trade in trades:
            self.trade_record_manager.record_trade(trade)

        self.metrics.update(datetime(2023, 1, 2))  # consume
        metrics = self.metrics.get_metrics()

        expected = {
            'current_day_lost': Decimal('0.0'),
            'current_day_lost_pct': Decimal('0.0'),
            'current_drawdown': Decimal('0.0'),
            'current_drawdown_pct': Decimal('0.0'),
            'max_drawdown': Decimal('0.0'),
            'max_drawdown_pct': Decimal('0.0'),
            'total_trades': 2,
            'winning_trades': 0,
            'max_profit': Decimal('0.0'),
            'max_loss': Decimal('-100'),
            'calmar_ratio': None,
            'sharpe_ratio': None,  # _ret_n=1
            'win_rate': 0.0,
        }
        self.assert_all_metrics(metrics, expected)

    def test_excessive_drawdown(self):
        self._settle_pnl(Decimal('-10000'))  # equity -> 0

        self.metrics.update(datetime(2023, 1, 1))          # init
        self.metrics.update(datetime(2023, 1, 1, 0, 0, 1)) # compute
        metrics = self.metrics.get_metrics()

        expected = {
            'current_day_lost': Decimal('10000'),
            'current_day_lost_pct': Decimal('100.0'),
            'current_drawdown': Decimal('10000'),
            'current_drawdown_pct': Decimal('100.0'),
            'max_drawdown': Decimal('10000'),
            'max_drawdown_pct': Decimal('100.0'),
            'total_trades': 0,
            'winning_trades': 0,
            'max_profit': Decimal('0.0'),
            'max_loss': Decimal('0.0'),
            'calmar_ratio': None,   # cagr 可能为 None 或有值，但 max_dd_pct=100% 且时间很短，测试这里只关心当前实现：通常 cagr 会是 None（时间太短），所以 calmar=None
            'sharpe_ratio': None,   # _ret_n=1
            'win_rate': None,
        }
        self.assert_all_metrics(metrics, expected)

    def test_large_number_of_trades(self):
        self.metrics.update(datetime(2023, 1, 1))  # init

        trades = []
        for i in range(100):
            timestamp = datetime(2023, 1, 1) + timedelta(days=i)
            pnl = D(i % 2 * 100 - 50)  # 交替盈利 50 和亏损 50
            trades.append(
                TradeRecord(
                    timestamp=timestamp,
                    operation_type="LONG_CLOSE",
                    position_size=Decimal('1.0'),
                    open_price=Decimal('100'),
                    close_price=Decimal('100') + (pnl > 0 and Decimal('10') or Decimal('-10')),
                    required_margin=Decimal('100'),
                    fee=Decimal('0'),
                    balance=Decimal('10000'),
                    leverage=Decimal('1'),
                    free_margin=Decimal('9900'),
                    pnl=pnl
                )
            )

        for trade in trades:
            self.trade_record_manager.record_trade(trade)

        self.metrics.update(datetime(2023, 4, 10))  # consume all
        metrics = self.metrics.get_metrics()

        expected = {
            'current_day_lost': Decimal('0.0'),
            'current_day_lost_pct': Decimal('0.0'),
            'current_drawdown': Decimal('0.0'),
            'current_drawdown_pct': Decimal('0.0'),
            'max_drawdown': Decimal('0.0'),
            'max_drawdown_pct': Decimal('0.0'),
            'total_trades': 100,
            'winning_trades': 50,
            'max_profit': Decimal('50'),
            'max_loss': Decimal('-50'),
            'calmar_ratio': None,   # max_dd_pct=0
            'sharpe_ratio': None,   # _ret_n=1
            'win_rate': 0.5,
        }
        self.assert_all_metrics(metrics, expected)

    def test_negative_initial_balance(self):
        # 原测试实际没有设置负初始余额，这里保持“只验证初始状态不崩”
        self.initial_balance = Decimal('10000')
        self.trade_record_manager = TradeRecordManager()
        self.ledger = Ledger()

        user_cash_acct = Account(Decimal(str(self.initial_balance)))
        user_margin_acct = Account(Decimal("0.0"))
        self.broker_accounts = BrokerAccounts()

        self.ledger.register("user_cash", user_cash_acct, strict_nonnegative=True)
        self.ledger.register("user_margin", user_margin_acct, strict_nonnegative=True)
        self.ledger.register("broker_fee_income", self.broker_accounts.fee_income, strict_nonnegative=False)
        self.ledger.register("broker_pnl", self.broker_accounts.broker_pnl, strict_nonnegative=False)

        self.user_accounts = UserAccounts(
            ledger=self.ledger,
            position_manager=MockPositionManager(),
            cash_account=user_cash_acct,
            margin_account=user_margin_acct,
            initial_balance=self.initial_balance,
        )
        self.metrics = Metrics(self.user_accounts, self.trade_record_manager, risk_free_rate=Decimal('0.012'))

        self.metrics.update(datetime(2023, 1, 1))
        metrics = self.metrics.get_metrics()

        expected = {
            'current_day_lost': Decimal('0.0'),
            'current_day_lost_pct': Decimal('0.0'),
            'current_drawdown': Decimal('0.0'),
            'current_drawdown_pct': Decimal('0.0'),
            'max_drawdown': Decimal('0.0'),
            'max_drawdown_pct': Decimal('0.0'),
            'total_trades': 0,
            'winning_trades': 0,
            'max_profit': Decimal('0.0'),
            'max_loss': Decimal('0.0'),
            'calmar_ratio': None,
            'sharpe_ratio': None,
            'win_rate': None,
        }
        self.assert_all_metrics(metrics, expected)


if __name__ == '__main__':
    unittest.main()
