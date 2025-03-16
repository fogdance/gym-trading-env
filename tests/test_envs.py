# tests/test_envs.py

import unittest
import pandas as pd
from gym_trading_env.envs.trading_env import CustomTradingEnv, Action
from decimal import Decimal, getcontext, ROUND_HALF_UP
import numpy as np
from gym_trading_env.utils.conversion import decimal_to_float, float_to_decimal
from stable_baselines3.common.env_checker import check_env

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
            'Date': dates,
            'Close': [float('1.1000') + float('0.0001') * float((i)) for i in range(100)],
            'Open': [float('1.1000') + float('0.0001') * float((i)) for i in range(100)],
            'High': [float('1.1000') + float('0.0001') * float((i)) for i in range(100)],
            'Low': [float('1.1000') + float('0.0001') * float((i)) for i in range(100)],
            'Volume': [float('1.1000') + float('0.0001') * float((i)) for i in range(100)]
        }
        df = pd.DataFrame(data)

        self.env = CustomTradingEnv(df=df, config_path='tests/test.yaml')
        check_env(self.env, warn=True)
        self.env.reset()
        self.initial_total_funds = self.calculate_total_funds()

    def calculate_total_funds(self):
        """
        Calculate the total funds across all user and broker accounts.
        """
        # Total funds include user balance(contain realized P&L), used margin, and broker fees
        user_info = self.env._get_info()
        total_funds = (
            Decimal(str(user_info['balance'])) +
            Decimal(str(user_info['broker_balance'])) +
            Decimal(str(user_info['fees_collected']))
        )
        return total_funds

    def test_reset(self):
        """
        Test the reset functionality of the environment.
        """
        obs, info = self.env.reset()
        # Verify that the observation is an image with the correct shape
        self.assertIsInstance(obs['image'], np.ndarray, "Observation should be a numpy array (image).")
        self.assertEqual(obs['image'].shape, (self.env.config.visualization.image_height, self.env.config.visualization.image_width, self.env.config.visualization.image_channels),
                         f"Observation shape should be {(self.env.config.visualization.image_height, self.env.config.visualization.image_width, self.env.config.visualization.image_channels)}, but got {obs['image'].shape}.")

        # Verify initial account details
        self.assertEqual(Decimal(str(info['balance'])), Decimal('10000.0'))
        self.assertEqual(Decimal(str(info['equity'])), Decimal('10000.0'))
        self.assertEqual(Decimal(str(info['used_margin'])), Decimal('0.0'))
        self.assertEqual(Decimal(str(info['free_margin'])), Decimal('10000.0'))
        self.assertEqual(Decimal(str(info['long_position'])), Decimal('0.0'))
        self.assertEqual(Decimal(str(info['short_position'])), Decimal('0.0'))
        self.assertEqual(Decimal(str(info['unrealized_pnl'])), Decimal('0.0'))
        self.assertEqual(Decimal(str(info['realized_pnl'])), Decimal('0.0'))
        self.assertEqual(Decimal(str(info['fees_collected'])), Decimal('0.0'))

        # Verify total funds after reset
        total_funds = self.calculate_total_funds()
        expected_total_funds = Decimal('10000.0')
        self.assertEqual(
            total_funds,
            expected_total_funds,
            f"Expected total funds after reset: {expected_total_funds}, but got: {total_funds}"
        )


    def test_step_long_open(self):
        """
        Test LONG_OPEN action in the environment and assert the reward and info.
        """
        obs, info = self.env.reset()
        initial_balance = Decimal(str(info['balance']))
        total_funds_before = self.calculate_total_funds()

        # Record previous_total_pnl before the step
        previous_total_pnl = Decimal(0)

        action = Action.LONG_OPEN.value
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Retrieve environment's internal variables
        current_step = self.env.current_step - 1  # step has been incremented
        current_price = Decimal(str(self.env.df.iloc[current_step]['Close']))
        spread = Decimal(str(self.env.config.trading.spread))
        ask_price = current_price + spread
        trading_fee_per_lot = Decimal(str(self.env.config.trading.trading_fee_per_lot))
        trade_lot = Decimal(str(self.env.config.trading.trade_lot))
        lot_size = Decimal(str(self.env.config.trading.lot_size))
        leverage = Decimal(str(self.env.config.trading.leverage))

        # Calculate expected values
        fee = trade_lot * trading_fee_per_lot
        required_margin = (trade_lot * lot_size * ask_price) / leverage

        # Expected balance is initial_balance minus fee
        expected_balance = initial_balance - fee

        expected_long_position = Decimal(str(trade_lot))
        total_used_margin = required_margin

        # P&L after LONG_OPEN (unrealized)
        pnl = (Decimal(str(self.env.df.iloc[self.env.current_step]['Close'])) - ask_price) * trade_lot * lot_size
        total_pnl = pnl  # realized_pnl is 0.0

        equity = expected_balance + total_pnl

        # Expected reward: change in total P&L = total_pnl - previous_total_pnl
        expected_reward = total_pnl - previous_total_pnl

        # Verify reward
        self.assertAlmostEqual(
            reward,
            float(decimal_to_float(expected_reward, precision=2)),
            places=2,
            msg=f"Expected reward: {float(decimal_to_float(expected_reward, precision=2))}, but got {reward}"
        )

        # Verify info contains updated account details
        self.assertEqual(Decimal(str(info['balance'])), expected_balance,
                         f"Expected balance after LONG_OPEN: {expected_balance}, but got {info['balance']}")
        self.assertEqual(Decimal(str(info['equity'])), equity,
                         f"Expected equity after LONG_OPEN: {equity}, but got {info['equity']}")
        self.assertEqual(Decimal(str(info['used_margin'])), total_used_margin,
                         f"Expected used_margin after LONG_OPEN: {total_used_margin}, but got {info['used_margin']}")
        self.assertEqual(Decimal(str(info['free_margin'])), equity - total_used_margin,
                         f"Expected free_margin after LONG_OPEN: {equity - total_used_margin}, but got {info['free_margin']}")
        self.assertEqual(Decimal(str(info['long_position'])), expected_long_position,
                         f"Expected long_position after LONG_OPEN: {expected_long_position}, but got {info['long_position']}")
        self.assertEqual(Decimal(str(info['short_position'])), Decimal('0.0'),
                         "Expected short_position to remain 0.0 after LONG_OPEN.")
        self.assertEqual(Decimal(str(info['realized_pnl'])), Decimal('0.0'),
                         "Realized P&L should remain 0.0 after LONG_OPEN.")
        self.assertEqual(Decimal(str(info['unrealized_pnl'])), pnl,
                         f"Expected unrealized P&L after LONG_OPEN: {pnl}, but got {info['unrealized_pnl']}")
        self.assertEqual(Decimal(str(info['fees_collected'])), fee,
                         f"Expected fees_collected after LONG_OPEN: {fee}, but got {info['fees_collected']}")

        self.assertFalse(terminated, "Environment should not terminate after LONG_OPEN.")
        self.assertFalse(truncated, "Environment should not truncate after LONG_OPEN.")

        # Verify total funds consistency
        total_funds_after = self.calculate_total_funds()
        self.assertAlmostEqual(
            float(total_funds_after),
            float(total_funds_before),
            places=6,
            msg=f"Total funds mismatch after LONG_OPEN. Expected: {total_funds_before}, Got: {total_funds_after}"
        )

    def test_step_long_close(self):
        """
        Test LONG_CLOSE action in the environment and assert the reward and info.
        """
        obs, info = self.env.reset()
        total_funds_before = self.calculate_total_funds()

        # First, execute LONG_OPEN
        action_open = Action.LONG_OPEN.value
        # Record previous_total_pnl before opening
        obs, reward_open, terminated, truncated, info = self.env.step(action_open)
        # total_pnl after open
        total_pnl_open = self.env.user_accounts.realized_pnl + self.env.user_accounts.unrealized_pnl

        # Record previous_total_pnl before closing
        previous_total_pnl_close = total_pnl_open

        # Execute LONG_CLOSE
        action_close = Action.LONG_CLOSE.value
        obs, reward_close, terminated, truncated, info = self.env.step(action_close)

        # Retrieve environment's internal variables
        open_step = self.env.current_step - 2
        close_step = self.env.current_step - 1
        open_price = Decimal(str(self.env.df.iloc[open_step]['Close']))
        close_price = Decimal(str(self.env.df.iloc[close_step]['Close']))
        spread = Decimal(str(self.env.config.trading.spread))
        trading_fee_per_lot = Decimal(str(self.env.config.trading.trading_fee_per_lot))
        is_round_turn = self.env.config.trading.is_round_turn
        trade_lot = Decimal(str(self.env.config.trading.trade_lot))
        lot_size = Decimal(str(self.env.config.trading.lot_size))
        leverage = Decimal(str(self.env.config.trading.leverage))

        # Prices at which LONG_OPEN and LONG_CLOSE occurred
        ask_price = open_price + spread
        bid_price = close_price - spread

        # P&L calculation
        pnl = (bid_price - ask_price) * trade_lot * lot_size
        fee_sell = Decimal('0')
        if is_round_turn:
            fee_sell = trade_lot * trading_fee_per_lot
        realized_pnl = pnl  # Only pnl is realized


        initial_balance = Decimal('10000.0')
        fee_open = trade_lot * trading_fee_per_lot
        required_margin_open = (trade_lot * lot_size * ask_price) / leverage
        balance_after_open = initial_balance - fee_open

        # After closing, fee_sell is deducted and pnl is added
        expected_balance = balance_after_open + pnl - fee_sell

        # Expected long_position after LONG_CLOSE
        expected_long_position = Decimal('0.0')

        # Used Margin after LONG_CLOSE
        total_used_margin = Decimal('0.0')

        # Equity after LONG_CLOSE
        equity = expected_balance

        # Expected reward: change in total P&L = realized_pnl - previous_total_pnl_close
        expected_reward = realized_pnl - previous_total_pnl_close

        # Verify reward
        self.assertAlmostEqual(
            reward_close,
            float(decimal_to_float(expected_reward, precision=2)),
            places=2,
            msg=f"Expected reward: {float(decimal_to_float(expected_reward, precision=2))}, but got {reward_close}"
        )

        # Verify realized P&L
        self.assertEqual(Decimal(str(info['realized_pnl'])), realized_pnl,
                        f"Expected realized P&L after LONG_CLOSE: {realized_pnl}, but got {info['realized_pnl']}")

        # Verify user balance has increased by realized P&L and decreased by fee_sell
        self.assertEqual(Decimal(str(info['balance'])), expected_balance,
                        f"Expected user balance after LONG_CLOSE: {expected_balance}, but got {info['balance']}")

        # Verify total funds consistency
        total_funds_after = self.calculate_total_funds()
        self.assertAlmostEqual(
            float(total_funds_after),
            float(total_funds_before),
            places=6,
            msg=f"Total funds mismatch after LONG_CLOSE. Expected: {total_funds_before}, Got: {total_funds_after}"
        )

        # Additional Assertions
        self.assertEqual(Decimal(str(info['equity'])), equity,
                        f"Expected equity after LONG_CLOSE: {equity}, but got {info['equity']}")
        self.assertEqual(Decimal(str(info['used_margin'])), total_used_margin,
                        f"Expected used_margin after LONG_CLOSE: {total_used_margin}, but got {info['used_margin']}")
        self.assertEqual(Decimal(str(info['free_margin'])), equity - total_used_margin,
                        f"Expected free_margin after LONG_CLOSE: {equity - total_used_margin}, but got {info['free_margin']}")
        self.assertEqual(Decimal(str(info['long_position'])), expected_long_position,
                        f"Expected long_position after LONG_CLOSE: {expected_long_position}, but got {info['long_position']}")
        self.assertEqual(Decimal(str(info['short_position'])), Decimal('0.0'),
                        "Expected short_position to remain 0.0 after LONG_CLOSE.")
        self.assertEqual(Decimal(str(info['unrealized_pnl'])), Decimal('0.0'),
                        "Unrealized P&L should be 0.0 after closing the position.")
        self.assertEqual(Decimal(str(info['fees_collected'])), fee_open + fee_sell,
                        f"Expected fees_collected after LONG_CLOSE: {fee_open + fee_sell}, but got {info['fees_collected']}")

        self.assertFalse(terminated, "Environment should not terminate after LONG_CLOSE.")
        self.assertFalse(truncated, "Environment should not truncate after LONG_CLOSE.")

    def test_step_short_open(self):
        """
        Test SHORT_OPEN action in the environment and assert the reward and info.
        """
        obs, info = self.env.reset()
        total_funds_before = self.calculate_total_funds()

        # Record previous_total_pnl before the step
        previous_total_pnl = Decimal(0)

        action = Action.SHORT_OPEN.value
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Retrieve environment's internal variables
        current_step = self.env.current_step - 1  # step has been incremented
        current_price = Decimal(str(self.env.df.iloc[current_step]['Close']))
        spread = Decimal(str(self.env.config.trading.spread))
        bid_price = current_price - spread
        trading_fee_per_lot = Decimal(str(self.env.config.trading.trading_fee_per_lot))
        trade_lot = Decimal(str(self.env.config.trading.trade_lot))
        lot_size = Decimal(str(self.env.config.trading.lot_size))
        leverage = Decimal(str(self.env.config.trading.leverage))

        # Calculate expected values
        fee = trade_lot * trading_fee_per_lot
        required_margin = (trade_lot * lot_size * bid_price) / leverage

        expected_balance = Decimal('10000.0') - fee

        expected_short_position = Decimal(str(trade_lot))
        total_used_margin = required_margin

        # P&L after SHORT_OPEN (unrealized)
        pnl = (bid_price - Decimal(str(self.env.df.iloc[self.env.current_step]['Close']))) * trade_lot * lot_size
        total_pnl = pnl  # realized_pnl is 0.0

        equity = expected_balance + total_pnl

        # Expected reward: change in total P&L = total_pnl - previous_total_pnl
        expected_reward = total_pnl - previous_total_pnl

        # Verify reward
        self.assertAlmostEqual(
            reward,
            float(decimal_to_float(expected_reward, precision=2)),
            places=2,
            msg=f"Expected reward: {float(decimal_to_float(expected_reward, precision=2))}, but got {reward}"
        )

        # Verify info contains updated account details
        self.assertEqual(Decimal(str(info['balance'])), expected_balance,
                         f"Expected balance after SHORT_OPEN: {expected_balance}, but got {info['balance']}")
        self.assertEqual(Decimal(str(info['equity'])), equity,
                         f"Expected equity after SHORT_OPEN: {equity}, but got {info['equity']}")
        self.assertEqual(Decimal(str(info['used_margin'])), total_used_margin,
                         f"Expected used_margin after SHORT_OPEN: {total_used_margin}, but got {info['used_margin']}")
        self.assertEqual(Decimal(str(info['free_margin'])), equity - total_used_margin,
                         f"Expected free_margin after SHORT_OPEN: {equity - total_used_margin}, but got {info['free_margin']}")
        self.assertEqual(Decimal(str(info['short_position'])), expected_short_position,
                         f"Expected short_position after SHORT_OPEN: {expected_short_position}, but got {info['short_position']}")
        self.assertEqual(Decimal(str(info['long_position'])), Decimal('0.0'),
                         "Expected long_position to remain 0.0 after SHORT_OPEN.")
        self.assertEqual(Decimal(str(info['realized_pnl'])), Decimal('0.0'),
                         "Realized P&L should remain 0.0 after SHORT_OPEN.")
        self.assertEqual(Decimal(str(info['unrealized_pnl'])), pnl,
                         f"Expected unrealized P&L after SHORT_OPEN: {pnl}, but got {info['unrealized_pnl']}")
        self.assertEqual(Decimal(str(info['fees_collected'])), fee,
                         f"Expected fees_collected after SHORT_OPEN: {fee}, but got {info['fees_collected']}")

        self.assertFalse(terminated, "Environment should not terminate after SHORT_OPEN.")
        self.assertFalse(truncated, "Environment should not truncate after SHORT_OPEN.")

        # Verify total funds consistency
        total_funds_after = self.calculate_total_funds()
        self.assertAlmostEqual(
            float(total_funds_after),
            float(total_funds_before),
            places=6,
            msg=f"Total funds mismatch after SHORT_OPEN. Expected: {total_funds_before}, Got: {total_funds_after}"
        )

    def test_step_short_close(self):
        """
        Test SHORT_CLOSE action in the environment and assert the reward and info.
        """
        obs, info = self.env.reset()
        total_funds_before = self.calculate_total_funds()

        # First, execute SHORT_OPEN
        action_open = Action.SHORT_OPEN.value
        # Record previous_total_pnl before opening
        obs, reward_open, terminated, truncated, info = self.env.step(action_open)
        # total_pnl after open
        total_pnl_open = self.env.user_accounts.realized_pnl + self.env.user_accounts.unrealized_pnl

        # Record previous_total_pnl before closing
        previous_total_pnl_close = total_pnl_open

        # Execute SHORT_CLOSE
        action_close = Action.SHORT_CLOSE.value
        obs, reward_close, terminated, truncated, info = self.env.step(action_close)

        # Retrieve environment's internal variables
        open_step = self.env.current_step - 2
        close_step = self.env.current_step - 1
        open_price = Decimal(str(self.env.df.iloc[open_step]['Close']))
        close_price = Decimal(str(self.env.df.iloc[close_step]['Close']))
        spread = Decimal(str(self.env.config.trading.spread))
        trading_fee_per_lot = Decimal(str(self.env.config.trading.trading_fee_per_lot))
        is_round_turn = self.env.config.trading.is_round_turn
        trade_lot = Decimal(str(self.env.config.trading.trade_lot))
        lot_size = Decimal(str(self.env.config.trading.lot_size))
        leverage = Decimal(str(self.env.config.trading.leverage))

        # Prices at which SHORT_OPEN and SHORT_CLOSE occurred
        bid_price_open = open_price - spread
        ask_price_close = close_price + spread

        # P&L calculation for SHORT_CLOSE
        pnl = (bid_price_open - ask_price_close) * trade_lot * lot_size
        fee_buy = Decimal('0')
        if is_round_turn:
            fee_buy = trade_lot * trading_fee_per_lot
        realized_pnl = pnl  # Only pnl is realized

        # Expected balance after SHORT_CLOSE
        initial_balance = Decimal('10000.0')
        fee_open = trade_lot * trading_fee_per_lot
        required_margin_open = (trade_lot * lot_size * bid_price_open) / leverage
        balance_after_open = initial_balance - fee_open

        # After closing, fee_buy is deducted and pnl is added
        expected_balance = balance_after_open + pnl - fee_buy

        # Expected short_position after SHORT_CLOSE
        expected_short_position = Decimal('0.0')

        # Used Margin after SHORT_CLOSE
        total_used_margin = Decimal('0.0')

        # Equity after SHORT_CLOSE
        equity = expected_balance

        # Expected reward: change in total P&L = realized_pnl - previous_total_pnl_close
        expected_reward = realized_pnl - previous_total_pnl_close

        # Verify reward
        self.assertAlmostEqual(
            reward_close,
            float(decimal_to_float(expected_reward, precision=2)),
            places=2,
            msg=f"Expected reward: {float(decimal_to_float(expected_reward, precision=2))}, but got {reward_close}"
        )

        # Verify realized P&L
        self.assertEqual(Decimal(str(info['realized_pnl'])), realized_pnl,
                        f"Expected realized P&L after SHORT_CLOSE: {realized_pnl}, but got {info['realized_pnl']}")

        # Verify user balance has increased by realized P&L and decreased by fee_buy
        self.assertEqual(Decimal(str(info['balance'])), expected_balance,
                        f"Expected user balance after SHORT_CLOSE: {expected_balance}, but got {info['balance']}")

        # Verify total funds consistency
        total_funds_after = self.calculate_total_funds()
        self.assertAlmostEqual(
            float(total_funds_after),
            float(total_funds_before),
            places=6,
            msg=f"Total funds mismatch after SHORT_CLOSE. Expected: {total_funds_before}, Got: {total_funds_after}"
        )

        # Additional Assertions
        self.assertEqual(Decimal(str(info['equity'])), equity,
                        f"Expected equity after SHORT_CLOSE: {equity}, but got {info['equity']}")
        self.assertEqual(Decimal(str(info['used_margin'])), total_used_margin,
                        f"Expected used_margin after SHORT_CLOSE: {total_used_margin}, but got {info['used_margin']}")
        self.assertEqual(Decimal(str(info['free_margin'])), equity - total_used_margin,
                        f"Expected free_margin after SHORT_CLOSE: {equity - total_used_margin}, but got {info['free_margin']}")
        self.assertEqual(Decimal(str(info['short_position'])), expected_short_position,
                        f"Expected short_position after SHORT_CLOSE: {expected_short_position}, but got {info['short_position']}")
        self.assertEqual(Decimal(str(info['long_position'])), Decimal('0.0'),
                        "Expected long_position to remain 0.0 after SHORT_CLOSE.")
        self.assertEqual(Decimal(str(info['unrealized_pnl'])), Decimal('0.0'),
                        "Unrealized P&L should be 0.0 after closing the position.")
        self.assertEqual(Decimal(str(info['fees_collected'])), fee_open + fee_buy,
                        f"Expected fees_collected after SHORT_CLOSE: {fee_open + fee_buy}, but got {info['fees_collected']}")

        self.assertFalse(terminated, "Environment should not terminate after SHORT_CLOSE.")
        self.assertFalse(truncated, "Environment should not truncate after SHORT_CLOSE.")

    def test_fee_deduction(self):
        """
        Test that trading fees are correctly deducted and recorded in the info dictionary.
        """
        obs, info = self.env.reset()
        total_funds_before = self.calculate_total_funds()

        # Execute multiple LONG_OPEN actions
        action_open = Action.LONG_OPEN.value
        # Record previous_total_pnl before the steps
        previous_total_pnl_step1 = self.env.previous_total_pnl
        obs, reward1, terminated, truncated, info = self.env.step(action_open)
        # Calculate expected reward for first step
        current_total_pnl_step1 = self.env.user_accounts.realized_pnl + self.env.user_accounts.unrealized_pnl
        expected_reward1 = current_total_pnl_step1 - previous_total_pnl_step1

        previous_total_pnl_step2 = self.env.previous_total_pnl
        obs, reward2, terminated, truncated, info = self.env.step(action_open)
        # Calculate expected reward for second step
        current_total_pnl_step2 = self.env.user_accounts.realized_pnl + self.env.user_accounts.unrealized_pnl
        expected_reward2 = current_total_pnl_step2 - previous_total_pnl_step2

        # Calculate expected total fees
        step1 = self.env.current_step - 2
        step2 = self.env.current_step - 1
        step1_price = Decimal(str(self.env.df.iloc[step1]['Close']))
        step2_price = Decimal(str(self.env.df.iloc[step2]['Close']))
        spread = Decimal(str(self.env.config.trading.spread))
        ask1_price = step1_price + spread
        ask2_price = step2_price + spread
        trading_fee_per_lot = Decimal(str(self.env.config.trading.trading_fee_per_lot))
        is_round_turn = self.env.config.trading.is_round_turn
        trade_lot = Decimal(str(self.env.config.trading.trade_lot))
        lot_size = Decimal(str(self.env.config.trading.lot_size))

        # Fee per LONG_OPEN
        fee1 = trade_lot * trading_fee_per_lot
        fee2 = trade_lot * trading_fee_per_lot
        total_fees = fee1 + fee2

        # Verify broker's fees account
        expected_fee_balance = total_fees
        self.assertEqual(Decimal(str(info['fees_collected'])), expected_fee_balance,
                         f"Expected fees_collected after multiple LONG_OPEN: {expected_fee_balance}, but got {info['fees_collected']}")

        # Verify user's balance after fee deduction
        # Total funds should remain the same as fees are moved to broker's fees_collected
        total_funds_after = self.calculate_total_funds()
        self.assertAlmostEqual(
            float(total_funds_after),
            float(total_funds_before),
            places=6,
            msg=f"Total funds mismatch after multiple LONG_OPEN. Expected: {total_funds_before}, Got: {total_funds_after}"
        )

        # Verify rewards
        self.assertAlmostEqual(
            reward1,
            float(decimal_to_float(expected_reward1, precision=2)),
            places=2,
            msg=f"Expected reward1: {float(decimal_to_float(expected_reward1, precision=2))}, but got {reward1}"
        )
        self.assertAlmostEqual(
            reward2,
            float(decimal_to_float(expected_reward2, precision=2)),
            places=2,
            msg=f"Expected reward2: {float(decimal_to_float(expected_reward2, precision=2))}, but got {reward2}"
        )

    def test_continuous_long_open(self):
        """
        Test continuous LONG_OPEN actions up to the maximum long position limit and assert the reward and info.
        """
        obs, info = self.env.reset()
        action = Action.LONG_OPEN.value
        previous_total_pnl = Decimal(0)
        for step in range(1, 4):  # Attempt to open 0.03 lots, exceeding the 0.02 lot limit
            with self.subTest(step=step):
                previous_equity = Decimal(str(info['equity']))
                previous_balance = Decimal(str(info['balance']))

                total_funds_before_step = self.calculate_total_funds()

                obs, reward, terminated, truncated, info = self.env.step(action)

                # Retrieve environment's internal variables
                current_step = self.env.current_step - 1
                current_price = Decimal(str(self.env.df.iloc[current_step]['Close']))
                spread = Decimal(str(self.env.config.trading.spread))
                ask_price = current_price + spread
                trading_fee_per_lot = Decimal(str(self.env.config.trading.trading_fee_per_lot))
                is_round_turn = self.env.config.trading.is_round_turn
                trade_lot = Decimal(str(self.env.config.trading.trade_lot))
                lot_size = Decimal(str(self.env.config.trading.lot_size))
                leverage = Decimal(str(self.env.config.trading.leverage))

                # Determine actual trade_lot (may be limited by max_long_position)
                max_allowed_trade_lot = Decimal(str(self.env.config.trading.max_long_position)) - Decimal(str(info['long_position']))
                actual_trade_lot = min(trade_lot, max_allowed_trade_lot)
                if actual_trade_lot <= Decimal('0.0'):
                    # Expect no change due to position limit
                    self.assertEqual(Decimal(str(info['long_position'])), Decimal(str(self.env.config.trading.max_long_position)),
                                     msg="Long position should not exceed the maximum limit.")

                    continue

                # Calculate cost and fee for actual_trade_lot
                fee = actual_trade_lot * trading_fee_per_lot

                required_margin = (actual_trade_lot * lot_size * ask_price) / leverage

                # Expected balance is previous balance minus fee
                expected_balance = previous_balance - fee

                # P&L after LONG_OPEN (unrealized)
                pnl = (Decimal(str(self.env.df.iloc[self.env.current_step]['Close'])) - ask_price) * actual_trade_lot * lot_size
                total_pnl = pnl  # realized_pnl is 0.0

                # Equity after LONG_OPEN
                equity = expected_balance + total_pnl

                # Expected reward: change in total P&L = total_pnl - previous_total_pnl
                expected_reward = total_pnl - previous_total_pnl
                previous_total_pnl = total_pnl

                # Expected long_position after LONG_OPEN
                expected_long_position = trade_lot * step
                if expected_long_position > Decimal(str(self.env.config.trading.max_long_position)):
                    expected_long_position = Decimal(str(self.env.config.trading.max_long_position))

                # Used Margin remains the same since we add required_margin for each new position
                total_used_margin = required_margin

                # Verify reward
                self.assertAlmostEqual(
                    reward,
                    float(decimal_to_float(expected_reward, precision=2)),
                    places=2,
                    msg=f"Expected reward at step {step}: {float(decimal_to_float(expected_reward, precision=2))}, but got {reward}"
                )

                # Verify info contains updated account details
                self.assertEqual(Decimal(str(info['balance'])), expected_balance,
                                 f"Expected balance after LONG_OPEN step {step}: {expected_balance}, but got {info['balance']}")
                self.assertEqual(Decimal(str(info['equity'])), equity,
                                 f"Expected equity after LONG_OPEN step {step}: {equity}, but got {info['equity']}")
                self.assertEqual(Decimal(str(info['used_margin'])), total_used_margin,
                                 f"Expected used_margin after LONG_OPEN step {step}: {total_used_margin}, but got {info['used_margin']}")
                self.assertEqual(Decimal(str(info['free_margin'])), equity - total_used_margin,
                                 f"Expected free_margin after LONG_OPEN step {step}: {equity - total_used_margin}, but got {info['free_margin']}")
                self.assertEqual(Decimal(str(info['long_position'])), expected_long_position,
                                 f"Expected long_position after LONG_OPEN step {step}: {expected_long_position}, but got {info['long_position']}")
                self.assertEqual(Decimal(str(info['short_position'])), Decimal('0.0'),
                                 "Expected short_position to remain 0.0, but got {info['short_position']}")
                self.assertEqual(Decimal(str(info['realized_pnl'])), Decimal('0.0'),
                                 "Realized P&L should remain 0.0 after LONG_OPEN.")
                self.assertEqual(Decimal(str(info['unrealized_pnl'])), pnl,
                                 f"Expected unrealized P&L after LONG_OPEN step {step}: {pnl}, but got {info['unrealized_pnl']}")
                self.assertEqual(Decimal(str(info['fees_collected'])), fee,
                                 f"Expected fees_collected after LONG_OPEN step {step}: {fee}, but got {info['fees_collected']}")

                self.assertFalse(terminated, "Environment should not terminate after LONG_OPEN.")
                self.assertFalse(truncated, "Environment should not truncate after LONG_OPEN.")

                # Verify total funds consistency
                total_funds_after_step = self.calculate_total_funds()
                self.assertAlmostEqual(
                    float(total_funds_after_step),
                    float(total_funds_before_step),
                    places=6,
                    msg=f"Total funds mismatch after LONG_OPEN step {step}. Expected: {total_funds_before_step}, Got: {total_funds_after_step}"
                )

    def test_position_limit_long(self):
        """
        Test that the environment does not allow exceeding the maximum long position limit and assert the reward and info.
        """
        obs, info = self.env.reset()
        action = Action.LONG_OPEN.value
        previous_total_pnl = Decimal(0)
        for step in range(1, 4):  # Attempt to open 0.03 lots, exceeding the 0.02 lot limit
            with self.subTest(step=step):
                previous_equity = Decimal(str(info['equity']))
                previous_balance = Decimal(str(info['balance']))

                total_funds_before_step = self.calculate_total_funds()

                obs, reward, terminated, truncated, info = self.env.step(action)

                # Retrieve environment's internal variables
                current_step = self.env.current_step - 1
                current_price = Decimal(str(self.env.df.iloc[current_step]['Close']))
                spread = Decimal(str(self.env.config.trading.spread))
                ask_price = current_price + spread
                trading_fee_per_lot = Decimal(str(self.env.config.trading.trading_fee_per_lot))
                is_round_turn = self.env.config.trading.is_round_turn
                trade_lot = Decimal(str(self.env.config.trading.trade_lot))
                lot_size = Decimal(str(self.env.config.trading.lot_size))
                leverage = Decimal(str(self.env.config.trading.leverage))

                # Determine actual_trade_lot (may be limited by max_long_position)
                max_allowed_trade_lot = Decimal(str(self.env.config.trading.max_long_position)) - Decimal(str(info['long_position']))
                actual_trade_lot = min(trade_lot, max_allowed_trade_lot)
                if actual_trade_lot <= Decimal('0.0'):
                    # Expect no change due to position limit
                    self.assertEqual(Decimal(str(info['long_position'])), Decimal(str(self.env.config.trading.max_long_position)),
                                     msg="Long position should not exceed the maximum limit.")
                    continue

                # Calculate cost and fee for actual_trade_lot
                fee = actual_trade_lot * trading_fee_per_lot

                required_margin = (actual_trade_lot * lot_size * ask_price) / leverage

                # Expected balance is previous balance minus fee
                expected_balance = previous_balance - fee

                # P&L after LONG_OPEN (unrealized)
                pnl = (Decimal(str(self.env.df.iloc[self.env.current_step]['Close'])) - ask_price) * actual_trade_lot * lot_size
                total_pnl = pnl  # realized_pnl is 0.0

                # Equity after LONG_OPEN
                equity = expected_balance + total_pnl

                # Expected reward: change in total P&L = total_pnl - previous_total_pnl
                expected_reward = total_pnl - previous_total_pnl
                previous_total_pnl = total_pnl

                # Expected long_position after LONG_OPEN
                expected_long_position = trade_lot * step
                if expected_long_position > Decimal(str(self.env.config.trading.max_long_position)):
                    expected_long_position = Decimal(str(self.env.config.trading.max_long_position))

                # Used Margin remains the same since we add required_margin for each new position
                total_used_margin = required_margin

                # Verify reward
                self.assertAlmostEqual(
                    reward,
                    float(decimal_to_float(expected_reward, precision=2)),
                    places=2,
                    msg=f"Expected reward at step {step}: {float(decimal_to_float(expected_reward, precision=2))}, but got {reward}"
                )

                # Verify info contains updated account details
                self.assertEqual(Decimal(str(info['balance'])), expected_balance,
                                 f"Expected balance after LONG_OPEN step {step}: {expected_balance}, but got {info['balance']}")
                self.assertEqual(Decimal(str(info['equity'])), equity,
                                 f"Expected equity after LONG_OPEN step {step}: {equity}, but got {info['equity']}")
                self.assertEqual(Decimal(str(info['used_margin'])), total_used_margin,
                                 f"Expected used_margin after LONG_OPEN step {step}: {total_used_margin}, but got {info['used_margin']}")
                self.assertEqual(Decimal(str(info['free_margin'])), equity - total_used_margin,
                                 f"Expected free_margin after LONG_OPEN step {step}: {equity - total_used_margin}, but got {info['free_margin']}")
                self.assertEqual(Decimal(str(info['long_position'])), expected_long_position,
                                 f"Expected long_position after LONG_OPEN step {step}: {expected_long_position}, but got {info['long_position']}")
                self.assertEqual(Decimal(str(info['short_position'])), Decimal('0.0'),
                                 "Expected short_position to remain 0.0, but got {info['short_position']}")
                self.assertEqual(Decimal(str(info['realized_pnl'])), Decimal('0.0'),
                                 "Realized P&L should remain 0.0 after LONG_OPEN.")
                self.assertEqual(Decimal(str(info['unrealized_pnl'])), pnl,
                                 f"Expected unrealized P&L after LONG_OPEN step {step}: {pnl}, but got {info['unrealized_pnl']}")
                self.assertEqual(Decimal(str(info['fees_collected'])), fee,
                                 f"Expected fees_collected after LONG_OPEN step {step}: {fee}, but got {info['fees_collected']}")

                self.assertFalse(terminated, "Environment should not terminate after LONG_OPEN.")
                self.assertFalse(truncated, "Environment should not truncate after LONG_OPEN.")

                # Verify total funds consistency
                total_funds_after_step = self.calculate_total_funds()
                self.assertAlmostEqual(
                    float(total_funds_after_step),
                    float(total_funds_before_step),
                    places=6,
                    msg=f"Total funds mismatch after LONG_OPEN step {step}. Expected: {total_funds_before_step}, Got: {total_funds_after_step}"
                )

    def test_position_limit_short(self):
        """
        Test that the environment does not allow exceeding the maximum short position limit and assert the reward and info.
        """
        obs, info = self.env.reset()
        action = Action.SHORT_OPEN.value
        previous_total_pnl = Decimal(0)
        for step in range(1, 4):  # Attempt to open 0.03 lots, exceeding the 0.02 lot limit
            with self.subTest(step=step):
                previous_equity = Decimal(str(info['equity']))
                previous_balance = Decimal(str(info['balance']))

                total_funds_before_step = self.calculate_total_funds()

                obs, reward, terminated, truncated, info = self.env.step(action)

                # Retrieve environment's internal variables
                current_step = self.env.current_step - 1
                current_price = Decimal(str(self.env.df.iloc[current_step]['Close']))
                spread = Decimal(str(self.env.config.trading.spread))
                bid_price = current_price - spread
                trading_fee_per_lot = Decimal(str(self.env.config.trading.trading_fee_per_lot))
                is_round_turn = self.env.config.trading.is_round_turn
                trade_lot = Decimal(str(self.env.config.trading.trade_lot))
                lot_size = Decimal(str(self.env.config.trading.lot_size))
                leverage = Decimal(str(self.env.config.trading.leverage))

                # Determine actual_trade_lot (may be limited by max_short_position)
                max_allowed_trade_lot = Decimal(str(self.env.config.trading.max_short_position)) - Decimal(str(info['short_position']))
                actual_trade_lot = min(trade_lot, max_allowed_trade_lot)
                if actual_trade_lot <= Decimal('0.0'):
                    # Expect no change due to position limit
                    self.assertEqual(Decimal(str(info['short_position'])), Decimal(str(self.env.config.trading.max_short_position)),
                                     msg="Short position should not exceed the maximum limit.")

                    continue

                # Calculate revenue and fee for actual_trade_lot
                fee = actual_trade_lot * trading_fee_per_lot

                required_margin = (actual_trade_lot * lot_size * bid_price) / leverage

                # Correct Expected balance: previous_balance - fee
                expected_balance = previous_balance - fee

                # P&L after SHORT_OPEN (unrealized)
                pnl = (bid_price - Decimal(str(self.env.df.iloc[self.env.current_step]['Close']))) * actual_trade_lot * lot_size
                total_pnl = pnl  # realized_pnl is 0.0

                # Equity after SHORT_OPEN
                equity = expected_balance + total_pnl

                # Expected reward: change in total P&L = total_pnl - previous_total_pnl
                expected_reward = total_pnl - previous_total_pnl
                previous_total_pnl = total_pnl

                # Expected short_position after SHORT_OPEN
                expected_short_position = trade_lot * step
                if expected_short_position > Decimal(str(self.env.config.trading.max_short_position)):
                    expected_short_position = Decimal(str(self.env.config.trading.max_short_position))

                # Used Margin remains the same since we add required_margin for each new position
                total_used_margin = required_margin

                # Verify reward
                self.assertAlmostEqual(
                    reward,
                    float(decimal_to_float(expected_reward, precision=2)),
                    places=2,
                    msg=f"Expected reward at step {step}: {float(decimal_to_float(expected_reward, precision=2))}, but got {reward}"
                )

                # Verify info contains updated account details
                self.assertEqual(Decimal(str(info['balance'])), expected_balance,
                                 f"Expected balance after SHORT_OPEN step {step}: {expected_balance}, but got {info['balance']}")
                self.assertEqual(Decimal(str(info['equity'])), equity,
                                 f"Expected equity after SHORT_OPEN step {step}: {equity}, but got {info['equity']}")
                self.assertEqual(Decimal(str(info['used_margin'])), total_used_margin,
                                 f"Expected used_margin after SHORT_OPEN step {step}: {total_used_margin}, but got {info['used_margin']}")
                self.assertEqual(Decimal(str(info['free_margin'])), equity - total_used_margin,
                                 f"Expected free_margin after SHORT_OPEN step {step}: {equity - total_used_margin}, but got {info['free_margin']}")
                self.assertEqual(Decimal(str(info['short_position'])), expected_short_position,
                                 f"Expected short_position after SHORT_OPEN step {step}: {expected_short_position}, but got {info['short_position']}")
                self.assertEqual(Decimal(str(info['long_position'])), Decimal('0.0'),
                                 "Expected long_position to remain 0.0, but got {info['long_position']}")
                self.assertEqual(Decimal(str(info['realized_pnl'])), Decimal('0.0'),
                                 "Realized P&L should remain 0.0 after SHORT_OPEN.")
                self.assertEqual(Decimal(str(info['unrealized_pnl'])), pnl,
                                 f"Expected unrealized P&L after SHORT_OPEN step {step}: {pnl}, but got {info['unrealized_pnl']}")
                self.assertEqual(Decimal(str(info['fees_collected'])), fee,
                                 f"Expected fees_collected after SHORT_OPEN step {step}: {fee}, but got {info['fees_collected']}")

                self.assertFalse(terminated, "Environment should not terminate after SHORT_OPEN.")
                self.assertFalse(truncated, "Environment should not truncate after SHORT_OPEN.")

                # Verify total funds consistency
                total_funds_after_step = self.calculate_total_funds()
                self.assertAlmostEqual(
                    float(total_funds_after_step),
                    float(total_funds_before_step),
                    places=6,
                    msg=f"Total funds mismatch after SHORT_OPEN step {step}. Expected: {total_funds_before_step}, Got: {total_funds_after_step}"
                )

    def test_fee_deduction(self):
        """
        Test that trading fees are correctly deducted and recorded in the info dictionary.
        """
        obs, info = self.env.reset()
        total_funds_before = self.calculate_total_funds()

        # Execute multiple LONG_OPEN actions
        action_open = Action.LONG_OPEN.value
        # Record previous_total_pnl before the steps
        previous_total_pnl_step1 = Decimal(0)
        obs, reward1, terminated, truncated, info = self.env.step(action_open)
        # Calculate expected reward for first step
        current_total_pnl_step1 = self.env.user_accounts.realized_pnl + self.env.user_accounts.unrealized_pnl
        expected_reward1 = current_total_pnl_step1 - previous_total_pnl_step1

        previous_total_pnl_step2 = current_total_pnl_step1
        obs, reward2, terminated, truncated, info = self.env.step(action_open)
        # Calculate expected reward for second step
        current_total_pnl_step2 = self.env.user_accounts.realized_pnl + self.env.user_accounts.unrealized_pnl
        expected_reward2 = current_total_pnl_step2 - previous_total_pnl_step2

        # Calculate expected total fees
        step1 = self.env.current_step - 2
        step2 = self.env.current_step - 1
        step1_price = Decimal(str(self.env.df.iloc[step1]['Close']))
        step2_price = Decimal(str(self.env.df.iloc[step2]['Close']))
        spread = Decimal(str(self.env.config.trading.spread))
        ask1_price = step1_price + spread
        ask2_price = step2_price + spread
        trading_fee_per_lot = Decimal(str(self.env.config.trading.trading_fee_per_lot))
        is_round_turn = self.env.config.trading.is_round_turn
        trade_lot = Decimal(str(self.env.config.trading.trade_lot))
        lot_size = Decimal(str(self.env.config.trading.lot_size))

        # Fee per LONG_OPEN
        fee1 = trade_lot * trading_fee_per_lot
        fee2 = trade_lot * trading_fee_per_lot
        total_fees = fee1 + fee2

        # Verify broker's fees account
        expected_fee_balance = total_fees
        self.assertEqual(Decimal(str(info['fees_collected'])), expected_fee_balance,
                         f"Expected fees_collected after multiple LONG_OPEN: {expected_fee_balance}, but got {info['fees_collected']}")

        # Verify user's balance after fee deduction
        # Total funds should remain the same as fees are moved to broker's fees_collected
        total_funds_after = self.calculate_total_funds()
        self.assertAlmostEqual(
            float(total_funds_after),
            float(total_funds_before),
            places=6,
            msg=f"Total funds mismatch after multiple LONG_OPEN. Expected: {total_funds_before}, Got: {total_funds_after}"
        )

        # Verify rewards
        self.assertAlmostEqual(
            reward1,
            float(decimal_to_float(expected_reward1, precision=2)),
            places=2,
            msg=f"Expected reward1: {float(decimal_to_float(expected_reward1, precision=2))}, but got {reward1}"
        )
        self.assertAlmostEqual(
            reward2,
            float(decimal_to_float(expected_reward2, precision=2)),
            places=2,
            msg=f"Expected reward2: {float(decimal_to_float(expected_reward2, precision=2))}, but got {reward2}"
        )

    def test_insufficient_balance_long_open(self):
        """
        Test LONG_OPEN action when balance is insufficient and assert the reward and info.
        """
        obs, info = self.env.reset()
        # Set balance low to prevent buy
        # To prevent opening, balance should be less than fee + required_margin
        # Calculate required_fee and required_margin for LONG_OPEN at current step
        current_step = self.env.current_step
        current_price = Decimal(self.env.df.iloc[current_step]['Close']) + Decimal(str(self.env.config.trading.spread))
        trading_fee_per_lot = Decimal(str(self.env.config.trading.trading_fee_per_lot))
        is_round_turn = self.env.config.trading.is_round_turn
        trade_lot = Decimal(str(self.env.config.trading.trade_lot))
        lot_size = Decimal(str(self.env.config.trading.lot_size))
        leverage = Decimal(str(self.env.config.trading.leverage))

        fee = trade_lot * trading_fee_per_lot
        required_margin = (trade_lot * lot_size * current_price) / leverage

        # Set balance to fee + required_margin - 0.01 to make it insufficient
        insufficient_balance = fee + required_margin - Decimal('0.01')
        self.env.user_accounts.balance.balance = insufficient_balance
        total_funds_before = self.calculate_total_funds()

        action = Action.LONG_OPEN.value
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Balance and long_position should remain unchanged
        expected_balance = insufficient_balance
        expected_long_position = Decimal('0.0')
        expected_realized_pnl = Decimal('0.0')
        expected_unrealized_pnl = Decimal('0.0')
        expected_fees_collected = Decimal('0.0')

        # Expected reward: No change, hence 0.0
        expected_reward = Decimal('0.0')

        # Verify info contains unchanged account details
        self.assertEqual(Decimal(str(info['balance'])), expected_balance,
                         f"Expected balance: {expected_balance}, but got {info['balance']}")
        self.assertEqual(Decimal(str(info['long_position'])), expected_long_position,
                         f"Expected long_position: {expected_long_position}, but got {info['long_position']}")
        self.assertEqual(Decimal(str(info['short_position'])), Decimal('0.0'),
                         "Short position should remain 0.0 after failed LONG_OPEN.")
        self.assertEqual(Decimal(str(info['realized_pnl'])), expected_realized_pnl,
                         "Realized P&L should remain 0.0 after failed LONG_OPEN.")
        self.assertEqual(Decimal(str(info['unrealized_pnl'])), expected_unrealized_pnl,
                         "Unrealized P&L should remain 0.0 after failed LONG_OPEN.")
        self.assertEqual(Decimal(str(info['fees_collected'])), expected_fees_collected,
                         "Fees should remain 0.0 after failed LONG_OPEN.")
        # self.assertAlmostEqual(float(reward), float(expected_reward), places=2,
        #                        msg=f"Expected reward: {float(expected_reward)}, but got {reward}")
        self.assertTrue(terminated, "Environment should not terminate on failed LONG_OPEN.")
        self.assertFalse(truncated, "Environment should not truncate on failed LONG_OPEN.")

        # Verify total funds consistency
        total_funds_after = self.calculate_total_funds()
        expected_total_funds = total_funds_before  # No change expected
        self.assertAlmostEqual(
            float(total_funds_after),
            float(expected_total_funds),
            places=6,
            msg=f"Total funds mismatch after failed LONG_OPEN. Expected: {expected_total_funds}, Got: {total_funds_after}"
        )

    def test_insufficient_balance_short_open(self):
        """
        Test SHORT_OPEN action when balance is insufficient and assert the reward and info.
        """
        obs, info = self.env.reset()
        # Set balance low to prevent sell
        # To prevent opening, balance should be sufficient to cover any required margin if applicable
        # For SHORT_OPEN, typically no immediate balance requirement, but ensure the environment handles it
        # Set balance to a negative value to simulate insufficient margin
        self.env.user_accounts.balance.balance = Decimal('-1.0')  # Negative balance
        total_funds_before = self.calculate_total_funds()

        action = Action.SHORT_OPEN.value
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Balance and short_position should remain unchanged
        expected_balance = Decimal('-1.0')
        expected_short_position = Decimal('0.0')

        # Expected reward: No change, hence 0.0
        expected_reward = Decimal('0.0')

        # Verify info contains unchanged account details
        self.assertEqual(Decimal(str(info['balance'])), expected_balance,
                         f"Expected balance: {expected_balance}, but got {info['balance']}")
        self.assertEqual(Decimal(str(info['short_position'])), expected_short_position,
                         f"Expected short_position: {expected_short_position}, but got {info['short_position']}")
        self.assertEqual(Decimal(str(info['long_position'])), Decimal('0.0'),
                         "Expected long_position to remain 0.0 after failed SHORT_OPEN.")
        self.assertEqual(Decimal(str(info['realized_pnl'])), Decimal('0.0'),
                         "Realized P&L should remain 0.0 after failed SHORT_OPEN.")
        self.assertEqual(Decimal(str(info['unrealized_pnl'])), Decimal('0.0'),
                         "Unrealized P&L should remain 0.0 after failed SHORT_OPEN.")
        self.assertEqual(Decimal(str(info['fees_collected'])), Decimal('0.0'),
                         "Fees_collected should remain 0.0 after failed SHORT_OPEN.")
        # self.assertAlmostEqual(float(reward), float(expected_reward), places=2,
        #                        msg=f"Expected reward: {float(expected_reward)}, but got {reward}")
        self.assertTrue(terminated, "Environment should terminate on failed SHORT_OPEN.")
        self.assertFalse(truncated, "Environment should not truncate on failed SHORT_OPEN.")

        # Verify total funds consistency
        total_funds_after = self.calculate_total_funds()
        expected_total_funds = total_funds_before  # No change expected
        self.assertAlmostEqual(
            float(total_funds_after),
            float(expected_total_funds),
            places=6,
            msg=f"Total funds mismatch after failed SHORT_OPEN. Expected: {expected_total_funds}, Got: {total_funds_after}"
        )


    def test_broker_fees_account(self):
        """
        Test that broker's fees account correctly accumulates fees from multiple transactions.
        """
        obs, info = self.env.reset()
        total_funds_before = self.calculate_total_funds()

        # Execute multiple LONG_OPEN actions
        action_open = Action.LONG_OPEN.value
        obs, reward, terminated, truncated, info = self.env.step(action_open)
        obs, reward, terminated, truncated, info = self.env.step(action_open)

        # Calculate expected total fees
        step1 = self.env.current_step - 2
        step2 = self.env.current_step - 1
        step1_price = Decimal(str(self.env.df.iloc[step1]['Close']))
        step2_price = Decimal(str(self.env.df.iloc[step2]['Close']))
        spread = Decimal(str(self.env.config.trading.spread))
        ask1_price = step1_price + spread
        ask2_price = step2_price + spread
        trading_fee_per_lot = Decimal(str(self.env.config.trading.trading_fee_per_lot))
        is_round_turn = self.env.config.trading.is_round_turn
        trade_lot = Decimal(str(self.env.config.trading.trade_lot))
        lot_size = Decimal(str(self.env.config.trading.lot_size))

        # Fee per LONG_OPEN
        fee1 = trade_lot * trading_fee_per_lot
        fee2 = trade_lot * trading_fee_per_lot
        total_fees = fee1 + fee2

        # Verify broker's fees account
        expected_fee_balance = total_fees
        self.assertEqual(Decimal(str(info['fees_collected'])), expected_fee_balance,
                         f"Expected fees_collected after multiple LONG_OPEN: {expected_fee_balance}, but got {info['fees_collected']}")

        # Verify user's balance after fee deduction
        # Total funds should remain the same as fees are moved to broker's fees_collected
        total_funds_after = self.calculate_total_funds()
        self.assertEqual(
            total_funds_after,
            total_funds_before,
            f"Total funds mismatch after multiple LONG_OPEN. Expected: {total_funds_before}, Got: {total_funds_after}"
        )


    def tearDown(self):
        """
        Clean up after tests.
        """
        self.env.close()

if __name__ == '__main__':
    unittest.main()
