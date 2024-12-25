# tests/test_position_manager.py

import unittest
from decimal import Decimal
from gym_trading_env.envs.position_manager import PositionManager
from gym_trading_env.envs.position import Position

class TestPositionManager(unittest.TestCase):
    def setUp(self):
        self.pm = PositionManager()
    
    def test_add_and_close_long_position(self):
        # Add long positions
        pos1 = Position(size=Decimal('0.01'), entry_price=Decimal('1.1000'), initial_margin=Decimal('10.0'))
        pos2 = Position(size=Decimal('0.02'), entry_price=Decimal('1.1050'), initial_margin=Decimal('20.0'))
        self.pm.add_long_position(pos1)
        self.pm.add_long_position(pos2)
        
        self.assertEqual(self.pm.total_long_position(), Decimal('0.03'))
        
        # Close the earliest long position
        pnl, released_margin, closed_size = self.pm.close_long_position(Decimal('1.1100'), Decimal('100000'))
        expected_pnl = (Decimal('1.1100') - Decimal('1.1000')) * Decimal('0.01') * Decimal('100000')
        self.assertEqual(pnl, expected_pnl)
        self.assertEqual(released_margin, Decimal('10.0'))
        self.assertEqual(closed_size, Decimal('0.01'))
        self.assertEqual(self.pm.total_long_position(), Decimal('0.02'))
        self.assertEqual(self.pm.realized_pnl, pnl)
    
    def test_close_short_position(self):
        # Add short positions
        pos1 = Position(size=Decimal('0.01'), entry_price=Decimal('1.2000'), initial_margin=Decimal('15.0'))
        pos2 = Position(size=Decimal('0.02'), entry_price=Decimal('1.1950'), initial_margin=Decimal('30.0'))
        self.pm.add_short_position(pos1)
        self.pm.add_short_position(pos2)
        
        self.assertEqual(self.pm.total_short_position(), Decimal('0.03'))
        
        # Close the earliest short position
        pnl, released_margin, closed_size = self.pm.close_short_position(Decimal('1.1900'), Decimal('100000'))
        expected_pnl = (Decimal('1.2000') - Decimal('1.1900')) * Decimal('0.01') * Decimal('100000')
        self.assertEqual(pnl, expected_pnl)
        self.assertEqual(released_margin, Decimal('15.0'))
        self.assertEqual(closed_size, Decimal('0.01'))
        self.assertEqual(self.pm.total_short_position(), Decimal('0.02'))
        self.assertEqual(self.pm.realized_pnl, pnl)
    
    def test_close_more_than_existing_long_positions(self):
        # Add a single long position
        pos = Position(size=Decimal('0.01'), entry_price=Decimal('1.1000'), initial_margin=Decimal('10.0'))
        self.pm.add_long_position(pos)
        
        # Close the existing position
        pnl, released_margin, closed_size = self.pm.close_long_position(Decimal('1.1100'), Decimal('100000'))
        expected_pnl = (Decimal('1.1100') - Decimal('1.1000')) * Decimal('0.01') * Decimal('100000')
        self.assertEqual(pnl, expected_pnl)
        self.assertEqual(released_margin, Decimal('10.0'))
        self.assertEqual(closed_size, Decimal('0.01'))
        self.assertEqual(self.pm.total_long_position(), Decimal('0.0'))
        self.assertEqual(self.pm.realized_pnl, pnl)
        
        # Attempt to close another long position, which should not exist
        with self.assertRaises(ValueError):
            self.pm.close_long_position(Decimal('1.1200'), Decimal('100000'))
    
    def test_close_more_than_existing_short_positions(self):
        # Add a single short position
        pos = Position(size=Decimal('0.01'), entry_price=Decimal('1.2000'), initial_margin=Decimal('15.0'))
        self.pm.add_short_position(pos)
        
        # Close the existing short position
        pnl, released_margin, closed_size = self.pm.close_short_position(Decimal('1.1900'), Decimal('100000'))
        expected_pnl = (Decimal('1.2000') - Decimal('1.1900')) * Decimal('0.01') * Decimal('100000')
        self.assertEqual(pnl, expected_pnl)
        self.assertEqual(released_margin, Decimal('15.0'))
        self.assertEqual(closed_size, Decimal('0.01'))
        self.assertEqual(self.pm.total_short_position(), Decimal('0.0'))
        self.assertEqual(self.pm.realized_pnl, pnl)
        
        # Attempt to close another short position, which should not exist
        with self.assertRaises(ValueError):
            self.pm.close_short_position(Decimal('1.1800'), Decimal('100000'))

if __name__ == '__main__':
    unittest.main()
