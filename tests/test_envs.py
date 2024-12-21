# tests/test_envs.py

import unittest
import pandas as pd
from gym_trading_env.envs.trading_env import CustomTradingEnv, Action
from gym_trading_env.utils.conversion import decimal_to_float
from decimal import Decimal, getcontext, ROUND_HALF_UP

# Set global decimal precision for testing
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_UP

class TestCustomTradingEnv(unittest.TestCase):

    def setUp(self):
        """
        Set up a dummy environment for testing.
        """
        # Create dummy data for testing
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = {
            'Close': [Decimal('1.1000') + Decimal('0.0001') * Decimal(str(i)) for i in range(100)]
        }
        df = pd.DataFrame(data, index=dates)
        # Define a sample configuration
        config = {
            'currency_pair': 'EURUSD',
            'initial_balance': 10000.0,
            'trading_fees': 0.001,  # 0.1% trading fee
            'spread': 0.0002,        # 2 pips spread
            'leverage': 100,         # 1:100 leverage
            'lot_size': 100000,      # Standard lot size for EUR/USD
            'trade_lot': 0.01,       # Default trade size: 0.01 lot
            'max_long_position': 0.02,     # Maximum long position size: 0.02 lot
            'max_short_position': 0.02,    # Maximum short position size: 0.02 lot
            'reward_function': 'basic_reward_function',
            'window_size': 20,
            'risk_free_rate': 0.0
        }
        self.env = CustomTradingEnv(df=df, config=config)
        self.env.reset()

    def test_reset(self):
        """
        Test the reset functionality of the environment.
        """
        obs, info = self.env.reset()
        self.assertEqual(obs['balance'], self.env.initial_balance)
        self.assertEqual(obs['equity'], self.env.initial_balance)
        self.assertEqual(obs['used_margin'], 0.0)
        self.assertEqual(obs['free_margin'], self.env.initial_balance)
        self.assertEqual(obs['long_position'], 0.0)
        self.assertEqual(obs['short_position'], 0.0)
        self.assertEqual(info, {})

    def test_step_long_open(self):
        """
        Test LONG_OPEN action in the environment.
        """
        obs, info = self.env.reset()
        action = Action.LONG_OPEN.value
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Calculate expected values
        current_price = Decimal(str(self.env.df.loc[self.env.current_step - 1, 'Close']))
        spread = Decimal(str(self.env.spread))
        ask_price = current_price + (spread)
        trading_fees = Decimal(str(self.env.trading_fees))
        trade_lot = self.env.trade_lot

        # Cost = trade_lot * lot_size * ask_price
        cost = trade_lot * self.env.lot_size * ask_price
        fee = cost * trading_fees
        total_cost = cost + fee

        # Expected balance after LONG_OPEN
        expected_balance = self.env.initial_balance - total_cost

        # Expected long_position after LONG_OPEN
        expected_long_position = trade_lot

        # Used Margin after LONG_OPEN
        used_margin_long = (trade_lot * self.env.lot_size * ask_price) / self.env.leverage
        used_margin_short = Decimal('0.0')
        total_used_margin = used_margin_long + used_margin_short

        # Equity after LONG_OPEN
        pnl = (current_price - ask_price) * trade_lot * self.env.lot_size
        equity = expected_balance + pnl + Decimal('0.0')

        # Free Margin after LONG_OPEN
        free_margin = equity - total_used_margin

        # Assertions
        self.assertAlmostEqual(obs['balance'], float(decimal_to_float(expected_balance, precision=2)), places=2,
                               msg=f"Expected balance after LONG_OPEN: {expected_balance}, but got {obs['balance']}")
        self.assertAlmostEqual(obs['equity'], float(decimal_to_float(equity, precision=2)), places=2,
                               msg=f"Expected equity after LONG_OPEN: {equity}, but got {obs['equity']}")
        self.assertAlmostEqual(obs['used_margin'], float(decimal_to_float(total_used_margin, precision=2)), places=2,
                               msg=f"Expected used_margin after LONG_OPEN: {total_used_margin}, but got {obs['used_margin']}")
        self.assertAlmostEqual(obs['free_margin'], float(decimal_to_float(free_margin, precision=2)), places=2,
                               msg=f"Expected free_margin after LONG_OPEN: {free_margin}, but got {obs['free_margin']}")
        self.assertAlmostEqual(obs['long_position'], float(expected_long_position), places=5,
                               msg=f"Expected long_position after LONG_OPEN: {expected_long_position}, but got {obs['long_position']}")
        self.assertAlmostEqual(obs['short_position'], 0.0, places=5,
                               msg=f"Expected short_position to remain 0.0, but got {obs['short_position']}")
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn('total_asset', info)

    def test_step_long_close(self):
        """
        Test LONG_CLOSE action in the environment.
        """
        obs, info = self.env.reset()
        # First, execute LONG_OPEN
        action_open = Action.LONG_OPEN.value
        obs, reward, terminated, truncated, info = self.env.step(action_open)

        # Now, execute LONG_CLOSE
        action_close = Action.LONG_CLOSE.value
        obs, reward, terminated, truncated, info = self.env.step(action_close)

        # Retrieve prices
        open_price = Decimal(str(self.env.df.loc[self.env.current_step - 2, 'Close']))
        close_price = Decimal(str(self.env.df.loc[self.env.current_step - 1, 'Close']))
        spread = Decimal(str(self.env.spread))
        trading_fees = Decimal(str(self.env.trading_fees))
        trade_lot = self.env.trade_lot

        # Ask price for LONG_OPEN
        ask_price = open_price + (spread)
        # Bid price for LONG_CLOSE
        bid_price = close_price - (spread)

        # Cost during LONG_OPEN
        cost = trade_lot * self.env.lot_size * ask_price
        fee_buy = cost * trading_fees
        total_cost = cost + fee_buy

        # P&L during LONG_CLOSE
        pnl = (bid_price - ask_price) * trade_lot * self.env.lot_size
        fee_sell = (trade_lot * self.env.lot_size * bid_price) * trading_fees
        total_revenue = pnl - fee_sell

        # Expected balance after LONG_CLOSE
        expected_balance = self.env.initial_balance - total_cost + total_revenue

        # Expected long_position after LONG_CLOSE
        expected_long_position = Decimal('0.0')

        # Used Margin after LONG_CLOSE
        used_margin_long = Decimal('0.0')
        used_margin_short = Decimal('0.0')
        total_used_margin = used_margin_long + used_margin_short

        # Equity after LONG_CLOSE
        equity = expected_balance + Decimal('0.0') + Decimal('0.0')

        # Free Margin after LONG_CLOSE
        free_margin = equity - total_used_margin

        # Assertions
        self.assertAlmostEqual(obs['balance'], float(decimal_to_float(expected_balance, precision=2)), places=2,
                               msg=f"Expected balance after LONG_CLOSE: {expected_balance}, but got {obs['balance']}")
        self.assertAlmostEqual(obs['equity'], float(decimal_to_float(equity, precision=2)), places=2,
                               msg=f"Expected equity after LONG_CLOSE: {equity}, but got {obs['equity']}")
        self.assertAlmostEqual(obs['used_margin'], float(decimal_to_float(total_used_margin, precision=2)), places=2,
                               msg=f"Expected used_margin after LONG_CLOSE: {total_used_margin}, but got {obs['used_margin']}")
        self.assertAlmostEqual(obs['free_margin'], float(decimal_to_float(free_margin, precision=2)), places=2,
                               msg=f"Expected free_margin after LONG_CLOSE: {free_margin}, but got {obs['free_margin']}")
        self.assertAlmostEqual(obs['long_position'], 0.0, places=5,
                               msg=f"Expected long_position after LONG_CLOSE: 0.0, but got {obs['long_position']}")
        self.assertAlmostEqual(obs['short_position'], 0.0, places=5,
                               msg=f"Expected short_position to remain 0.0, but got {obs['short_position']}")
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn('total_asset', info)

    def test_step_short_open(self):
        """
        Test SHORT_OPEN action in the environment.
        """
        obs, info = self.env.reset()
        action = Action.SHORT_OPEN.value
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Calculate expected values
        current_price = Decimal(str(self.env.df.loc[self.env.current_step - 1, 'Close']))
        spread = Decimal(str(self.env.spread))
        bid_price = current_price - (spread)
        trading_fees = Decimal(str(self.env.trading_fees))
        trade_lot = self.env.trade_lot

        # Revenue = trade_lot * lot_size * bid_price
        revenue = trade_lot * self.env.lot_size * bid_price
        fee = revenue * trading_fees
        total_revenue = revenue - fee

        # Expected balance after SHORT_OPEN
        expected_balance = self.env.initial_balance + total_revenue

        # Expected short_position after SHORT_OPEN
        expected_short_position = trade_lot

        # Used Margin after SHORT_OPEN
        used_margin_long = Decimal('0.0')
        used_margin_short = (trade_lot * self.env.lot_size * bid_price) / self.env.leverage
        total_used_margin = used_margin_long + used_margin_short

        # Equity after SHORT_OPEN
        pnl = (bid_price - current_price) * trade_lot * self.env.lot_size
        equity = expected_balance + pnl + Decimal('0.0')

        # Free Margin after SHORT_OPEN
        free_margin = equity - total_used_margin

        # Assertions
        self.assertAlmostEqual(obs['balance'], float(decimal_to_float(expected_balance, precision=2)), places=2,
                               msg=f"Expected balance after SHORT_OPEN: {expected_balance}, but got {obs['balance']}")
        self.assertAlmostEqual(obs['equity'], float(decimal_to_float(equity, precision=2)), places=2,
                               msg=f"Expected equity after SHORT_OPEN: {equity}, but got {obs['equity']}")
        self.assertAlmostEqual(obs['used_margin'], float(decimal_to_float(total_used_margin, precision=2)), places=2,
                               msg=f"Expected used_margin after SHORT_OPEN: {total_used_margin}, but got {obs['used_margin']}")
        self.assertAlmostEqual(obs['free_margin'], float(decimal_to_float(free_margin, precision=2)), places=2,
                               msg=f"Expected free_margin after SHORT_OPEN: {free_margin}, but got {obs['free_margin']}")
        self.assertAlmostEqual(obs['short_position'], float(expected_short_position), places=5,
                               msg=f"Expected short_position after SHORT_OPEN: {expected_short_position}, but got {obs['short_position']}")
        self.assertAlmostEqual(obs['long_position'], 0.0, places=5,
                               msg=f"Expected long_position to remain 0.0, but got {obs['long_position']}")
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn('total_asset', info)

    def test_step_short_close(self):
        """
        Test SHORT_CLOSE action in the environment.
        """
        obs, info = self.env.reset()
        # First, execute SHORT_OPEN
        action_open = Action.SHORT_OPEN.value
        obs, reward, terminated, truncated, info = self.env.step(action_open)

        # Now, execute SHORT_CLOSE
        action_close = Action.SHORT_CLOSE.value
        obs, reward, terminated, truncated, info = self.env.step(action_close)

        # Retrieve prices
        open_price = Decimal(str(self.env.df.loc[self.env.current_step - 2, 'Close']))
        close_price = Decimal(str(self.env.df.loc[self.env.current_step - 1, 'Close']))
        spread = Decimal(str(self.env.spread))
        trading_fees = Decimal(str(self.env.trading_fees))
        trade_lot = self.env.trade_lot

        # Bid price for SHORT_OPEN
        bid_price = open_price - (spread)
        # Ask price for SHORT_CLOSE
        ask_price = close_price + (spread)

        # Revenue during SHORT_OPEN
        revenue = trade_lot * self.env.lot_size * bid_price
        fee_sell = revenue * trading_fees
        total_revenue = revenue - fee_sell

        # P&L during SHORT_CLOSE
        pnl = (bid_price - ask_price) * trade_lot * self.env.lot_size
        fee_buy = (trade_lot * self.env.lot_size * ask_price) * self.env.trading_fees
        total_cost = pnl - fee_buy

        # Expected balance after SHORT_CLOSE
        expected_balance = self.env.initial_balance + total_revenue + total_cost

        # Expected short_position after SHORT_CLOSE
        expected_short_position = Decimal('0.0')

        # Used Margin after SHORT_CLOSE
        used_margin_long = Decimal('0.0')
        used_margin_short = Decimal('0.0')
        total_used_margin = used_margin_long + used_margin_short

        # Equity after SHORT_CLOSE
        equity = expected_balance + Decimal('0.0') + Decimal('0.0')

        # Free Margin after SHORT_CLOSE
        free_margin = equity - total_used_margin

        # Assertions
        self.assertAlmostEqual(obs['balance'], float(decimal_to_float(expected_balance, precision=2)), places=2,
                               msg=f"Expected balance after SHORT_CLOSE: {expected_balance}, but got {obs['balance']}")
        self.assertAlmostEqual(obs['equity'], float(decimal_to_float(equity, precision=2)), places=2,
                               msg=f"Expected equity after SHORT_CLOSE: {equity}, but got {obs['equity']}")
        self.assertAlmostEqual(obs['used_margin'], float(decimal_to_float(total_used_margin, precision=2)), places=2,
                               msg=f"Expected used_margin after SHORT_CLOSE: {total_used_margin}, but got {obs['used_margin']}")
        self.assertAlmostEqual(obs['free_margin'], float(decimal_to_float(free_margin, precision=2)), places=2,
                               msg=f"Expected free_margin after SHORT_CLOSE: {free_margin}, but got {obs['free_margin']}")
        self.assertAlmostEqual(obs['short_position'], 0.0, places=5,
                               msg=f"Expected short_position after SHORT_CLOSE: {expected_short_position}, but got {obs['short_position']}")
        self.assertAlmostEqual(obs['long_position'], 0.0, places=5,
                               msg=f"Expected long_position to remain 0.0, but got {obs['long_position']}")
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn('total_asset', info)

    def test_continuous_long_open(self):
        """
        Test continuous LONG_OPEN actions up to the maximum long position limit.
        """
        obs, info = self.env.reset()
        action = Action.LONG_OPEN.value
        for step in range(1, 4):  # Attempt to open 0.04, exceeding the 0.02 lot limit
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Assert at each step that long_position does not exceed max_long_position
            with self.subTest(step=step):
                self.assertLessEqual(self.env.long_position, self.env.max_long_position,
                                     msg=f"Long position exceeded max_long_position at step {step}: {self.env.long_position} > {self.env.max_long_position}")
                # Additionally, assert that short_position remains unchanged
                self.assertAlmostEqual(self.env.short_position, 0.0, places=5,
                                       msg=f"Expected short_position to remain 0.0 at step {step}, but got {self.env.short_position}")
        
        # After loop, assert that environment did not terminate unexpectedly
        self.assertFalse(terminated, "Environment should not terminate when max_long_position is reached.")
        self.assertFalse(truncated, "Environment should not truncate when max_long_position is reached.")

    def test_continuous_short_open(self):
        """
        Test continuous SHORT_OPEN actions up to the maximum short position limit.
        """
        obs, info = self.env.reset()
        action = Action.SHORT_OPEN.value
        for step in range(1, 4):  # Attempt to open 0.04, exceeding the 0.02 lot limit
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Assert at each step that short_position does not exceed max_short_position
            with self.subTest(step=step):
                self.assertLessEqual(self.env.short_position, self.env.max_short_position,
                                     msg=f"Short position exceeded max_short_position at step {step}: {self.env.short_position} > {self.env.max_short_position}")
                # Additionally, assert that long_position remains unchanged
                self.assertAlmostEqual(self.env.long_position, 0.0, places=5,
                                       msg=f"Expected long_position to remain 0.0 at step {step}, but got {self.env.long_position}")
        
        # After loop, assert that environment did not terminate unexpectedly
        self.assertFalse(terminated, "Environment should not terminate when max_short_position is reached.")
        self.assertFalse(truncated, "Environment should not truncate when max_short_position is reached.")

    def test_position_limit_long(self):
        """
        Test that the environment does not allow exceeding the maximum long position limit.
        """
        obs, info = self.env.reset()
        action = Action.LONG_OPEN.value
        for step in range(1, 4):  # Attempt to open 0.04, exceeding the 0.02 lot limit
            obs, reward, terminated, truncated, info = self.env.step(action)
            # Assert at each step that long_position does not exceed max_long_position
            with self.subTest(step=step):
                self.assertLessEqual(self.env.long_position, self.env.max_long_position,
                                     msg=f"Long position exceeded max_long_position at step {step}: {self.env.long_position} > {self.env.max_long_position}")
                # Additionally, assert that short_position remains unchanged
                self.assertAlmostEqual(self.env.short_position, 0.0, places=5,
                                       msg=f"Expected short_position to remain 0.0 at step {step}, but got {self.env.short_position}")
        
        # After loop, assert that environment did not terminate unexpectedly
        self.assertFalse(terminated, "Environment should not terminate when max_long_position is reached.")
        self.assertFalse(truncated, "Environment should not truncate when max_long_position is reached.")

    def test_position_limit_short(self):
        """
        Test that the environment does not allow exceeding the maximum short position limit.
        """
        obs, info = self.env.reset()
        action = Action.SHORT_OPEN.value
        for step in range(1, 4):  # Attempt to open 0.04, exceeding the 0.02 lot limit
            obs, reward, terminated, truncated, info = self.env.step(action)
            # Assert at each step that short_position does not exceed max_short_position
            with self.subTest(step=step):
                self.assertLessEqual(self.env.short_position, self.env.max_short_position,
                                     msg=f"Short position exceeded max_short_position at step {step}: {self.env.short_position} > {self.env.max_short_position}")
                # Additionally, assert that long_position remains unchanged
                self.assertAlmostEqual(self.env.long_position, 0.0, places=5,
                                       msg=f"Expected long_position to remain 0.0 at step {step}, but got {self.env.long_position}")
        
        # After loop, assert that environment did not terminate unexpectedly
        self.assertFalse(terminated, "Environment should not terminate when max_short_position is reached.")
        self.assertFalse(truncated, "Environment should not truncate when max_short_position is reached.")

    def test_invalid_action(self):
        """
        Test if the environment handles invalid actions gracefully.
        """
        obs, info = self.env.reset()
        with self.assertRaises(ValueError):
            self.env.step(5)  # Invalid action, since action_space is Discrete(5)

    def test_insufficient_balance_long_open(self):
        """
        Test LONG_OPEN action when balance is insufficient.
        """
        obs, info = self.env.reset()
        # Set balance low to prevent buy
        self.env.balance = Decimal('0.0')
        action = Action.LONG_OPEN.value
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Balance and long_position should remain unchanged
        expected_balance = Decimal('0.0')
        expected_long_position = Decimal('0.0')

        self.assertEqual(obs['balance'], float(decimal_to_float(expected_balance, precision=2)),
                         msg=f"Expected balance: {expected_balance}, but got {obs['balance']}")
        self.assertEqual(obs['long_position'], float(expected_long_position),
                         msg=f"Expected long_position: {expected_long_position}, but got {obs['long_position']}")
        self.assertAlmostEqual(obs['short_position'], 0.0, places=5,
                               msg=f"Expected short_position to remain 0.0, but got {obs['short_position']}")
        self.assertFalse(terminated, "Environment should not terminate on failed action.")
        self.assertFalse(truncated, "Environment should not truncate on failed action.")
        self.assertIn('total_asset', info)

    def test_insufficient_balance_short_open(self):
        """
        Test SHORT_OPEN action when balance is insufficient.
        """
        obs, info = self.env.reset()
        # Set balance low to prevent sell
        self.env.balance = Decimal('-1.0')  # Negative balance
        action = Action.SHORT_OPEN.value
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Balance and short_position should remain unchanged
        expected_balance = Decimal('-1.0')
        expected_short_position = Decimal('0.0')

        self.assertEqual(obs['balance'], float(decimal_to_float(expected_balance, precision=2)),
                         msg=f"Expected balance: {expected_balance}, but got {obs['balance']}")
        self.assertEqual(obs['short_position'], float(expected_short_position),
                         msg=f"Expected short_position: {expected_short_position}, but got {obs['short_position']}")
        self.assertAlmostEqual(self.env.long_position, 0.0, places=5,
                               msg=f"Expected long_position to remain 0.0, but got {self.env.long_position}")
        self.assertTrue(terminated, "Environment should terminate on failed action.")
        self.assertFalse(truncated, "Environment should not truncate on failed action.")
        self.assertIn('total_asset', info)

    def tearDown(self):
        """
        Clean up after tests.
        """
        self.env.close()

if __name__ == '__main__':
    unittest.main()
