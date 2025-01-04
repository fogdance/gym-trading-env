# tests/test_plotting.py
import unittest
from datetime import datetime
import pandas as pd
from decimal import Decimal
from gym_trading_env.envs.trade_record_manager import TradeRecordManager
from gym_trading_env.envs.trade_record import TradeRecord
from gym_trading_env.envs.action import Action
from gym_trading_env.rendering.plotting import BollingerBandPlotter

class TestProcessTradeRecord(unittest.TestCase):
    def setUp(self):
        # Create a hypothetical TradeRecordManager
        self.trade_record_manager = TradeRecordManager()

        # Create a hypothetical time range
        self.df = pd.DataFrame({
            'Date': pd.date_range(start='2020-01-01', periods=10, freq='D'),
            'Open': [1.0] * 10,
            'High': [1.1] * 10,
            'Low': [0.9] * 10,
            'Close': [1.0] * 10,
            'Volume': [1000] * 10
        })
        self.df.set_index('Date', inplace=True)

    def addRecords(self):
        # Add some trade records
        self.trade_record_manager.record_trade(TradeRecord(
            timestamp=pd.Timestamp('2020-01-02 10:00:00'),
            operation_type=Action.LONG_OPEN.name,
            position_size=Decimal('0.1'),
            price=Decimal('1.0'),
            required_margin=Decimal('0.0'),
            fee=Decimal('0.0'),
            balance=Decimal('1000.0'),
            leverage=Decimal('100.0'),
            free_margin=Decimal('1000.0')
        ))

        self.trade_record_manager.record_trade(TradeRecord(
            timestamp=pd.Timestamp('2020-01-05 12:00:00'),
            operation_type=Action.LONG_CLOSE.name,
            position_size=Decimal('0.1'),
            price=Decimal('1.0'),
            required_margin=Decimal('0.0'),
            fee=Decimal('0.0'),
            balance=Decimal('1000.0'),
            leverage=Decimal('100.0'),
            free_margin=Decimal('1000.0')
        ))

        self.trade_record_manager.record_trade(TradeRecord(
            timestamp=pd.Timestamp('2020-01-07 12:00:00'),
            operation_type=Action.SHORT_OPEN.name,
            position_size=Decimal('0.1'),
            price=Decimal('1.0'),
            required_margin=Decimal('0.0'),
            fee=Decimal('0.0'),
            balance=Decimal('1000.0'),
            leverage=Decimal('100.0'),
            free_margin=Decimal('1000.0')
        ))

        self.trade_record_manager.record_trade(TradeRecord(
            timestamp=pd.Timestamp('2020-01-08 12:00:00'),
            operation_type=Action.SHORT_CLOSE.name,
            position_size=Decimal('0.1'),
            price=Decimal('1.0'),
            required_margin=Decimal('0.0'),
            fee=Decimal('0.0'),
            balance=Decimal('1000.0'),
            leverage=Decimal('100.0'),
            free_margin=Decimal('1000.0')
        ))

    def test_process_trade_record(self):
        self.addRecords()

        # Create BollingerBandPlotter object
        plotter = BollingerBandPlotter(self.df, self.trade_record_manager)

        # Get results
        long_open_records, long_close_records, short_open_records, short_close_records = plotter.process_trade_record()

        # Verify LONG_OPEN trade
        self.assertEqual(len(long_open_records), 1)
        self.assertEqual(long_open_records[0].operation_type, Action.LONG_OPEN.name)
        self.assertEqual(long_open_records[0].timestamp, pd.Timestamp('2020-01-02 10:00:00'))

        # Verify LONG_CLOSE trade
        self.assertEqual(len(long_close_records), 1)
        self.assertEqual(long_close_records[0].operation_type, Action.LONG_CLOSE.name)
        self.assertEqual(long_close_records[0].timestamp, pd.Timestamp('2020-01-05 12:00:00'))

        # Verify SHORT_OPEN trade
        self.assertEqual(len(short_open_records), 1)
        self.assertEqual(short_open_records[0].operation_type, Action.SHORT_OPEN.name)
        self.assertEqual(short_open_records[0].timestamp, pd.Timestamp('2020-01-07 12:00:00'))

        # Verify SHORT_CLOSE trade
        self.assertEqual(len(short_close_records), 1)
        self.assertEqual(short_close_records[0].operation_type, Action.SHORT_CLOSE.name)
        self.assertEqual(short_close_records[0].timestamp, pd.Timestamp('2020-01-08 12:00:00'))


    def test_filter_newest_records(self):
        # Add a trade outside the time range
        self.trade_record_manager.record_trade(TradeRecord(
            timestamp=pd.Timestamp('2010-12-01 12:00:00'),
            operation_type=Action.LONG_OPEN.name,
            position_size=Decimal('0.1'),
            price=Decimal('1.0'),
            required_margin=Decimal('0.0'),
            fee=Decimal('0.0'),
            balance=Decimal('1000.0'),
            leverage=Decimal('100.0'),
            free_margin=Decimal('1000.0')
        ))
        # Add another trade outside the time range
        self.trade_record_manager.record_trade(TradeRecord(
            timestamp=pd.Timestamp('2011-12-01 12:00:00'),
            operation_type=Action.LONG_CLOSE.name,
            position_size=Decimal('0.1'),
            price=Decimal('1.0'),
            required_margin=Decimal('0.0'),
            fee=Decimal('0.0'),
            balance=Decimal('1000.0'),
            leverage=Decimal('100.0'),
            free_margin=Decimal('1000.0')
        ))

        self.addRecords()

        plotter = BollingerBandPlotter(self.df, self.trade_record_manager)
        long_open_records, long_close_records, short_open_records, short_close_records = plotter.process_trade_record()

        # Ensure trades outside the time range are filtered out
        self.assertEqual(len(long_open_records), 1)
        self.assertEqual(long_open_records[0].operation_type, Action.LONG_OPEN.name)

        # Ensure out-of-range closing records are filtered out
        self.assertEqual(len(long_close_records), 1)
        self.assertEqual(long_close_records[0].operation_type, Action.LONG_CLOSE.name)

        # The records are correctly classified and within the time range
        self.assertEqual(len(short_open_records), 1)
        self.assertEqual(len(short_close_records), 1)


    def test_filter_half_in_records(self):
        # Add a trade outside the time range
        self.trade_record_manager.record_trade(TradeRecord(
            timestamp=pd.Timestamp('2010-12-01 12:00:00'),
            operation_type=Action.LONG_OPEN.name,
            position_size=Decimal('0.1'),
            price=Decimal('1.0'),
            required_margin=Decimal('0.0'),
            fee=Decimal('0.0'),
            balance=Decimal('1000.0'),
            leverage=Decimal('100.0'),
            free_margin=Decimal('1000.0')
        ))
        # Add a trade within the time range
        self.trade_record_manager.record_trade(TradeRecord(
            timestamp=pd.Timestamp('2020-01-01 12:00:00'),
            operation_type=Action.LONG_CLOSE.name,
            position_size=Decimal('0.1'),
            price=Decimal('1.0'),
            required_margin=Decimal('0.0'),
            fee=Decimal('0.0'),
            balance=Decimal('1000.0'),
            leverage=Decimal('100.0'),
            free_margin=Decimal('1000.0')
        ))
        # Add a trade outside the time range
        self.trade_record_manager.record_trade(TradeRecord(
            timestamp=pd.Timestamp('2010-12-01 12:00:00'),
            operation_type=Action.SHORT_OPEN.name,
            position_size=Decimal('0.1'),
            price=Decimal('1.0'),
            required_margin=Decimal('0.0'),
            fee=Decimal('0.0'),
            balance=Decimal('1000.0'),
            leverage=Decimal('100.0'),
            free_margin=Decimal('1000.0')
        ))
        # Add a trade within the time range
        self.trade_record_manager.record_trade(TradeRecord(
            timestamp=pd.Timestamp('2020-01-01 12:00:00'),
            operation_type=Action.SHORT_CLOSE.name,
            position_size=Decimal('0.1'),
            price=Decimal('1.0'),
            required_margin=Decimal('0.0'),
            fee=Decimal('0.0'),
            balance=Decimal('1000.0'),
            leverage=Decimal('100.0'),
            free_margin=Decimal('1000.0')
        ))
        self.addRecords()

        plotter = BollingerBandPlotter(self.df, self.trade_record_manager)
        long_open_records, long_close_records, short_open_records, short_close_records = plotter.process_trade_record()

        # Ensure trades outside the time range are filtered out
        self.assertEqual(len(long_open_records), 1)
        self.assertEqual(long_open_records[0].operation_type, Action.LONG_OPEN.name)

        # Ensure out-of-range closing records are filtered out
        self.assertEqual(len(long_close_records), 1)
        self.assertEqual(long_close_records[0].operation_type, Action.LONG_CLOSE.name)

        # The records are correctly classified and within the time range
        self.assertEqual(len(short_open_records), 1)
        self.assertEqual(len(short_close_records), 1)


    def test_filter_oldest_records(self):
        self.addRecords()

        # Add a trade outside the time range
        self.trade_record_manager.record_trade(TradeRecord(
            timestamp=pd.Timestamp('2020-12-01 12:00:00'),
            operation_type=Action.LONG_OPEN.name,
            position_size=Decimal('0.1'),
            price=Decimal('1.0'),
            required_margin=Decimal('0.0'),
            fee=Decimal('0.0'),
            balance=Decimal('1000.0'),
            leverage=Decimal('100.0'),
            free_margin=Decimal('1000.0')
        ))

        plotter = BollingerBandPlotter(self.df, self.trade_record_manager)
        long_open_records, long_close_records, short_open_records, short_close_records = plotter.process_trade_record()

        # Ensure trades outside the time range are filtered out
        self.assertEqual(len(long_open_records), 1)
        self.assertEqual(long_open_records[0].operation_type, Action.LONG_OPEN.name)

        # Ensure out-of-range closing records are filtered out
        self.assertEqual(len(long_close_records), 1)
        self.assertEqual(long_close_records[0].operation_type, Action.LONG_CLOSE.name)

        # The records are correctly classified and within the time range
        self.assertEqual(len(short_open_records), 1)
        self.assertEqual(len(short_close_records), 1)

    def test_matching_open_close(self):
        self.addRecords()

        # Add a trade without a closing trade
        self.trade_record_manager.record_trade(TradeRecord(
            timestamp=pd.Timestamp('2020-01-09 12:00:00'),
            operation_type=Action.LONG_OPEN.name,
            position_size=Decimal('0.1'),
            price=Decimal('1.0'),
            required_margin=Decimal('0.0'),
            fee=Decimal('0.0'),
            balance=Decimal('1000.0'),
            leverage=Decimal('100.0'),
            free_margin=Decimal('1000.0')
        ))

        plotter = BollingerBandPlotter(self.df, self.trade_record_manager)
        long_open_records, long_close_records, short_open_records, short_close_records = plotter.process_trade_record()

        # Ensure there is no corresponding closing trade for the opening trade
        self.assertEqual(len(long_open_records), 2)  # Two open trades
        self.assertEqual(len(long_close_records), 1)  # Only one close trade

    def test_hold(self):

        # Add a trade without a closing trade
        self.trade_record_manager.record_trade(TradeRecord(
            timestamp=pd.Timestamp('2010-01-09 12:00:00'),
            operation_type=Action.LONG_OPEN.name,
            position_size=Decimal('0.1'),
            price=Decimal('1.0'),
            required_margin=Decimal('0.0'),
            fee=Decimal('0.0'),
            balance=Decimal('1000.0'),
            leverage=Decimal('100.0'),
            free_margin=Decimal('1000.0')
        ))

        plotter = BollingerBandPlotter(self.df, self.trade_record_manager)
        long_open_records, long_close_records, short_open_records, short_close_records = plotter.process_trade_record()

        self.assertEqual(len(long_open_records), 0)


if __name__ == '__main__':
    unittest.main()
