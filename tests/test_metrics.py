# tests/test_metrics.py

import unittest
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np
from gym_trading_env.envs.metrics import Metrics
from gym_trading_env.envs.user_accounts import UserAccounts
from gym_trading_env.envs.trade_record_manager import TradeRecordManager
from gym_trading_env.envs.trade_record import TradeRecord
from gym_trading_env.utils.decimal_util import D, D0, D1, D100, quantize_money

class MockPositionManager:
    def total_long_position(self):
        return Decimal('0.0')
    def total_short_position(self):
        return Decimal('0.0')

class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.initial_balance = Decimal('10000')
        self.user_accounts = UserAccounts(self.initial_balance, MockPositionManager())
        self.trade_record_manager = TradeRecordManager()
        self.metrics = Metrics(self.user_accounts, self.trade_record_manager, risk_free_rate=Decimal('0.012'))

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
            self.assertAlmostEqual(metrics['calmar_ratio'], expected_values['calmar_ratio'], places=5)
        if expected_values['sharpe_ratio'] is None:
            self.assertIsNone(metrics['sharpe_ratio'])
        else:
            self.assertAlmostEqual(metrics['sharpe_ratio'], expected_values['sharpe_ratio'], places=5)
        if expected_values['win_rate'] is None:
            self.assertIsNone(metrics['win_rate'])
        else:
            self.assertAlmostEqual(metrics['win_rate'], expected_values['win_rate'], places=5)

    # 正常情况测试
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

    def test_day_loss(self):
        self.metrics.update(datetime(2023, 1, 1))
        self.user_accounts.realize_pnl(Decimal('-500'))
        self.metrics.update(datetime(2023, 1, 1))
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
            'sharpe_ratio': None,
            'win_rate': None,
        }
        self.assert_all_metrics(metrics, expected)

    def test_new_day_reset(self):
        self.user_accounts.realize_pnl(Decimal('-500'))
        self.metrics.update(datetime(2023, 1, 1))
        self.metrics.update(datetime(2023, 1, 2))
        metrics = self.metrics.get_metrics()
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
            'calmar_ratio': None,
            'sharpe_ratio': None,
            'win_rate': None,
        }
        self.assert_all_metrics(metrics, expected)

    def test_trades_metrics(self):
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
        self.metrics.update(datetime(2023, 1, 3))
        metrics = self.metrics.get_metrics()
        returns = [0.01, -0.005, 0.0075]
        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        expected_sharpe = (avg_return - 0.012 / 365) / std_return
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
            'calmar_ratio': None,
            'sharpe_ratio': expected_sharpe,
            'win_rate': 2/3,
        }
        self.assert_all_metrics(metrics, expected)

        # 模拟回撤
        self.user_accounts.realize_pnl(Decimal('-200'))
        self.metrics.update(datetime(2023, 1, 3))
        metrics = self.metrics.get_metrics()
        total_pnl = Decimal('125')
        time_span_days = 2
        annualized_return = (total_pnl / self.initial_balance) / D(time_span_days / 365.0)
        expected_calmar = float(annualized_return / Decimal('200'))
        expected.update({
            'current_day_lost': Decimal('200'),
            'current_day_lost_pct': Decimal('2.0'),
            'current_drawdown': Decimal('200'),
            'current_drawdown_pct': Decimal('2.0'),
            'max_drawdown': Decimal('200'),
            'max_drawdown_pct': Decimal('2.0'),
            'calmar_ratio': expected_calmar,
        })
        self.assert_all_metrics(metrics, expected)

    # 边界条件测试
    def test_no_trades(self):
        self.user_accounts.realize_pnl(Decimal('-1000'))
        self.metrics.update(datetime(2023, 1, 1))
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
            'sharpe_ratio': None,
            'win_rate': None,
        }
        self.assert_all_metrics(metrics, expected)

    def test_single_trade(self):
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
        self.metrics.update(datetime(2023, 1, 1))
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
            'sharpe_ratio': None,
            'win_rate': 1.0,
        }
        self.assert_all_metrics(metrics, expected)

    def test_zero_std_sharpe(self):
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
        ]
        for trade in trades:
            self.trade_record_manager.record_trade(trade)
        self.metrics.update(datetime(2023, 1, 2))
        metrics = self.metrics.get_metrics()
        expected = {
            'current_day_lost': Decimal('0.0'),
            'current_day_lost_pct': Decimal('0.0'),
            'current_drawdown': Decimal('0.0'),
            'current_drawdown_pct': Decimal('0.0'),
            'max_drawdown': Decimal('0.0'),
            'max_drawdown_pct': Decimal('0.0'),
            'total_trades': 2,
            'winning_trades': 2,
            'max_profit': Decimal('100'),
            'max_loss': Decimal('0.0'),
            'calmar_ratio': None,
            'sharpe_ratio': None,
            'win_rate': 1.0,
        }
        self.assert_all_metrics(metrics, expected)

    def test_zero_time_span_calmar(self):
        trades = [
            TradeRecord(
                timestamp=datetime(2023, 1, 1, 10, 0),
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
                timestamp=datetime(2023, 1, 1, 10, 1),
                operation_type="LONG_CLOSE",
                position_size=Decimal('1.0'),
                open_price=Decimal('100'),
                close_price=Decimal('105'),
                required_margin=Decimal('100'),
                fee=Decimal('0'),
                balance=Decimal('10000'),
                leverage=Decimal('1'),
                free_margin=Decimal('9900'),
                pnl=Decimal('50')
            ),
        ]
        for trade in trades:
            self.trade_record_manager.record_trade(trade)
        self.user_accounts.realize_pnl(Decimal('-200'))
        self.metrics.update(datetime(2023, 1, 1, 10, 1))
        metrics = self.metrics.get_metrics()
        returns = [0.01, 0.005]
        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        expected_sharpe = (avg_return - 0.012 / 365) / std_return
        expected = {
            'current_day_lost': Decimal('200'),
            'current_day_lost_pct': Decimal('2.0'),
            'current_drawdown': Decimal('200'),
            'current_drawdown_pct': Decimal('2.0'),
            'max_drawdown': Decimal('200'),
            'max_drawdown_pct': Decimal('2.0'),
            'total_trades': 2,
            'winning_trades': 2,
            'max_profit': Decimal('100'),
            'max_loss': Decimal('0.0'),
            'calmar_ratio': None,
            'sharpe_ratio': expected_sharpe, 
            'win_rate': 1.0,
        }
        self.assert_all_metrics(metrics, expected)

    def test_zero_max_drawdown_calmar(self):
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
                operation_type="LONG_CLOSE",
                position_size=Decimal('1.0'),
                open_price=Decimal('100'),
                close_price=Decimal('105'),
                required_margin=Decimal('100'),
                fee=Decimal('0'),
                balance=Decimal('10000'),
                leverage=Decimal('1'),
                free_margin=Decimal('9900'),
                pnl=Decimal('50')
            ),
        ]
        for trade in trades:
            self.trade_record_manager.record_trade(trade)
        self.metrics.update(datetime(2023, 1, 2))
        metrics = self.metrics.get_metrics()
        returns = [0.01, 0.005]
        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        expected_sharpe = (avg_return - 0.012 / 365) / std_return
        expected = {
            'current_day_lost': Decimal('0.0'),
            'current_day_lost_pct': Decimal('0.0'),
            'current_drawdown': Decimal('0.0'),
            'current_drawdown_pct': Decimal('0.0'),
            'max_drawdown': Decimal('0.0'),
            'max_drawdown_pct': Decimal('0.0'),
            'total_trades': 2,
            'winning_trades': 2,
            'max_profit': Decimal('100'),
            'max_loss': Decimal('0.0'),
            'calmar_ratio': None,
            'sharpe_ratio': expected_sharpe,
            'win_rate': 1.0,
        }
        self.assert_all_metrics(metrics, expected)

    # 极端场景测试
    def test_all_losing_trades(self):
        # 测试所有交易亏损
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
        self.metrics.update(datetime(2023, 1, 2))
        metrics = self.metrics.get_metrics()
        returns = [-0.01, -0.005]
        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        expected_sharpe = (avg_return - 0.012 / 365) / std_return
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
            'sharpe_ratio': expected_sharpe,  # 负值
            'win_rate': 0.0,
        }
        self.assert_all_metrics(metrics, expected)

    def test_excessive_drawdown(self):
        # 测试回撤超过初始资金
        self.user_accounts.realize_pnl(Decimal('-15000'))  # 净值变为 -5000
        self.metrics.update(datetime(2023, 1, 1))
        metrics = self.metrics.get_metrics()
        expected = {
            'current_day_lost': Decimal('15000'),
            'current_day_lost_pct': Decimal('150.0'),
            'current_drawdown': Decimal('15000'),
            'current_drawdown_pct': Decimal('150.0'),
            'max_drawdown': Decimal('15000'),
            'max_drawdown_pct': Decimal('150.0'),
            'total_trades': 0,
            'winning_trades': 0,
            'max_profit': Decimal('0.0'),
            'max_loss': Decimal('0.0'),
            'calmar_ratio': None,
            'sharpe_ratio': None,
            'win_rate': None,
        }
        self.assert_all_metrics(metrics, expected)

    def test_large_number_of_trades(self):
        # 测试大量交易
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
        self.metrics.update(datetime(2023, 4, 10))
        metrics = self.metrics.get_metrics()
        returns = [float(trade.pnl / self.initial_balance) for trade in trades]
        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        expected_sharpe = (avg_return - 0.012 / 365) / std_return
        total_pnl = sum(trade.pnl for trade in trades)
        time_span_days = 99
        annualized_return = (total_pnl / self.initial_balance) / D(time_span_days / 365.0)
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
            'calmar_ratio': None,  # 无回撤
            'sharpe_ratio': expected_sharpe,
            'win_rate': 0.5,
        }
        self.assert_all_metrics(metrics, expected)

    def test_negative_initial_balance(self):
        # 测试负初始余额（假设环境允许）
        self.initial_balance = Decimal('-10000')
        self.user_accounts = UserAccounts(self.initial_balance, MockPositionManager())
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