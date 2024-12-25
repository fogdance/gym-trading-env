# tests/test_envs.py

import unittest
import pandas as pd
from gym_trading_env.envs.trading_env import CustomTradingEnv, Action
from gym_trading_env.utils.conversion import decimal_to_float
from gym_trading_env.rewards.reward_functions import total_pnl_reward_function
from decimal import Decimal, getcontext, ROUND_HALF_UP
import numpy as np

# Set global decimal precision for testing
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_UP

def get_scalar(obs, key):
    """
    Helper function to extract scalar value from observation dictionary.
    If the value is a numpy array with a single element, return the scalar.
    Otherwise, return the value as is.
    """
    value = obs.get(key, None)
    if isinstance(value, np.ndarray) and value.size == 1:
        return value[0]
    return value

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
            'use_dict_obs': True,
            'currency_pair': 'EURUSD',
            'initial_balance': 10000.0,
            'trading_fees': 0.001,  # 0.1% trading fee
            'spread': 0.0002,        # 2 pips spread
            'leverage': 100,         # 1:100 leverage
            'lot_size': 100000,      # Standard lot size for EUR/USD
            'trade_lot': 0.01,       # Default trade size: 0.01 lot
            'max_long_position': 0.02,     # Maximum long position size: 0.02 lot
            'max_short_position': 0.02,    # Maximum short position size: 0.02 lot
            'reward_function': 'total_pnl_reward_function',
            'window_size': 20,
            'risk_free_rate': 0.0
        }
        self.env = CustomTradingEnv(df=df, config=config)
        self.env.reset()
        self.initial_total_funds = self.calculate_total_funds()

    def calculate_total_funds(self):
        """
        Calculate the total funds across all user and broker accounts.
        """
        user = self.env.user_accounts
        broker = self.env.broker_accounts
        total_funds = (
            user.balance.get_balance() +
            user.margin.get_balance() +
            user.unrealized_pnl +
            user.realized_pnl +
            broker.fees.get_balance() +
            broker.balance.get_balance()
        )
        return total_funds

    def test_reset(self):
        """
        Test the reset functionality of the environment.
        """
        obs, info = self.env.reset()
        self.assertEqual(obs['balance'], float(decimal_to_float(Decimal('10000.0'), precision=2)))
        self.assertEqual(obs['equity'], float(decimal_to_float(Decimal('10000.0'), precision=2)))
        self.assertEqual(obs['used_margin'], 0.0)
        self.assertEqual(obs['free_margin'], float(decimal_to_float(Decimal('10000.0'), precision=2)))
        self.assertEqual(obs['long_position'], 0.0)
        self.assertEqual(obs['short_position'], 0.0)
        self.assertEqual(obs['realized_pnl'], 0.0)
        self.assertEqual(obs['unrealized_pnl'], 0.0)
        self.assertEqual(obs['fees'], 0.0)
        self.assertEqual(info, {})

        # Verify total funds after reset
        total_funds = self.calculate_total_funds()
        expected_total_funds = Decimal('10000.0')
        self.assertAlmostEqual(float(decimal_to_float(total_funds, precision=2)),
                               float(decimal_to_float(expected_total_funds, precision=2)),
                               places=2,
                               msg=f"Expected total funds after reset: {expected_total_funds}, but got: {total_funds}")

    def test_step_long_open(self):
        """
        Test LONG_OPEN action in the environment and assert the reward and funds.
        """
        obs, info = self.env.reset()
        initial_balance = self.env.user_accounts.balance.get_balance()
        total_funds_before = self.calculate_total_funds()
        
        action = Action.LONG_OPEN.value
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Calculate expected values
        current_price = Decimal(str(self.env.df.loc[self.env.df.index[self.env.current_step - 1], 'Close']))
        spread = Decimal(str(self.env.spread))
        ask_price = current_price + spread
        trading_fees = Decimal(str(self.env.trading_fees))
        trade_lot = Decimal(str(self.env.trade_lot))
        lot_size = Decimal(str(self.env.lot_size))
        leverage = Decimal(str(self.env.leverage))

        # Cost = trade_lot * lot_size * ask_price
        cost = trade_lot * lot_size * ask_price
        fee = cost * trading_fees
        total_cost = cost + fee

        # Required margin
        required_margin = (trade_lot * lot_size * ask_price) / leverage

        # Expected balance after LONG_OPEN
        expected_balance = initial_balance - total_cost

        # Expected long_position after LONG_OPEN
        expected_long_position = trade_lot

        # Used Margin after LONG_OPEN
        total_used_margin = required_margin

        # Equity after LONG_OPEN
        pnl = (current_price - ask_price) * trade_lot * lot_size
        equity = expected_balance + pnl

        # Free Margin after LONG_OPEN
        free_margin = equity - total_used_margin

        # Expected reward: change in equity = equity - previous_equity
        expected_reward = equity - initial_balance

        # Verify account balances and positions
        balance = get_scalar(obs, 'balance')
        equity_obs = get_scalar(obs, 'equity')
        used_margin = get_scalar(obs, 'used_margin')
        free_margin_obs = get_scalar(obs, 'free_margin')
        long_position = get_scalar(obs, 'long_position')
        short_position = get_scalar(obs, 'short_position')
        unrealized_pnl = get_scalar(obs, 'unrealized_pnl')
        realized_pnl = get_scalar(obs, 'realized_pnl')
        fees = get_scalar(obs, 'fees')
        self.assertAlmostEqual(balance, float(decimal_to_float(expected_balance, precision=2)), places=2,
                               msg=f"Expected balance after LONG_OPEN: {expected_balance}, but got {obs['balance']}")
        self.assertAlmostEqual(equity_obs, float(decimal_to_float(equity, precision=2)), places=2,
                               msg=f"Expected equity after LONG_OPEN: {equity}, but got {obs['equity']}")
        self.assertAlmostEqual(used_margin, float(decimal_to_float(total_used_margin, precision=2)), places=2,
                               msg=f"Expected used_margin after LONG_OPEN: {total_used_margin}, but got {obs['used_margin']}")
        self.assertAlmostEqual(free_margin_obs, float(decimal_to_float(free_margin, precision=2)), places=2,
                               msg=f"Expected free_margin after LONG_OPEN: {free_margin}, but got {obs['free_margin']}")
        self.assertAlmostEqual(long_position, float(expected_long_position), places=5,
                               msg=f"Expected long_position after LONG_OPEN: {expected_long_position}, but got {obs['long_position']}")
        self.assertAlmostEqual(short_position, 0.0, places=5,
                               msg=f"Expected short_position to remain 0.0, but got {obs['short_position']}")
        self.assertAlmostEqual(realized_pnl, 0.0, places=2,
                               msg=f"Realized P&L should remain 0.0 after LONG_OPEN.")
        self.assertAlmostEqual(unrealized_pnl, float(decimal_to_float(pnl, precision=2)), places=2,
                               msg=f"Expected unrealized P&L after LONG_OPEN: {pnl}, but got {obs['unrealized_pnl']}")
        self.assertAlmostEqual(fees, float(decimal_to_float(fee, precision=2)), places=2,
                               msg=f"Expected fees after LONG_OPEN: {fee}, but got {obs['fees']}")
        self.assertAlmostEqual(reward, float(decimal_to_float(expected_reward, precision=2)), places=2,
                               msg=f"Expected reward: {expected_reward}, but got {reward}")
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn('total_asset', info)

        # Verify total funds consistency
        total_funds_after = self.calculate_total_funds()
        expected_total_funds = total_funds_before - fee  # 手续费由代理商收取
        self.assertAlmostEqual(float(decimal_to_float(total_funds_after, precision=2)),
                               float(decimal_to_float(expected_total_funds, precision=2)),
                               places=2,
                               msg=f"Total funds mismatch after LONG_OPEN. Expected: {expected_total_funds}, Got: {total_funds_after}")

    def test_step_long_close(self):
        """
        Test LONG_CLOSE action in the environment and assert the reward and funds.
        """
        obs, info = self.env.reset()
        total_funds_before = self.calculate_total_funds()
        
        # First, execute LONG_OPEN
        action_open = Action.LONG_OPEN.value
        obs, reward_open, terminated, truncated, info = self.env.step(action_open)
        total_funds_after_open = self.calculate_total_funds()

        # Capture previous equity after LONG_OPEN
        previous_equity = self.env._calculate_equity()

        # Now, execute LONG_CLOSE
        action_close = Action.LONG_CLOSE.value
        obs, reward_close, terminated, truncated, info = self.env.step(action_close)

        # Retrieve prices
        open_step = self.env.current_step - 2
        close_step = self.env.current_step - 1
        open_price = Decimal(str(self.env.df.loc[self.env.df.index[open_step], 'Close']))
        close_price = Decimal(str(self.env.df.loc[self.env.df.index[close_step], 'Close']))
        spread = Decimal(str(self.env.spread))
        trading_fees = Decimal(str(self.env.trading_fees))
        trade_lot = Decimal(str(self.env.trade_lot))
        lot_size = Decimal(str(self.env.lot_size))
        leverage = Decimal(str(self.env.leverage))
    
        # Ask price for LONG_OPEN
        ask_price = open_price + spread
        # Bid price for LONG_CLOSE
        bid_price = close_price - spread
    
        # Realized P&L during LONG_CLOSE
        pnl = (bid_price - ask_price) * trade_lot * lot_size
        fee_sell = (trade_lot * lot_size * bid_price) * trading_fees
        realized_pnl = pnl - fee_sell
    
        # Expected balance after LONG_CLOSE
        cost = trade_lot * lot_size * ask_price
        fee_buy = cost * trading_fees
        total_cost = cost + fee_buy
        total_revenue = pnl - fee_sell
        expected_balance = Decimal(str(self.env.initial_balance)) - total_cost + total_revenue
    
        # Expected long_position after LONG_CLOSE
        expected_long_position = Decimal('0.0')
    
        # Used Margin after LONG_CLOSE
        released_margin = (trade_lot * lot_size * ask_price) / leverage
        total_used_margin = Decimal('0.0')
    
        # Equity after LONG_CLOSE
        equity = expected_balance + Decimal('0.0') + Decimal('0.0')  # realized_pnl already added to balance
    
        # Free Margin after LONG_CLOSE
        free_margin = equity - total_used_margin
    
        # Expected reward: change in equity = equity - previous_equity
        expected_reward = equity - previous_equity

        # Verify account balances and positions
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
        self.assertAlmostEqual(obs['realized_pnl'], float(decimal_to_float(realized_pnl, precision=2)), places=2,
                               msg=f"Expected realized P&L after LONG_CLOSE: {realized_pnl}, but got {obs['realized_pnl']}")
        self.assertAlmostEqual(obs['unrealized_pnl'], 0.0, places=2,
                               msg=f"Expected unrealized P&L after LONG_CLOSE: 0.0, but got {obs['unrealized_pnl']}")
        self.assertAlmostEqual(obs['fees'], float(decimal_to_float(fee_buy + fee_sell, precision=2)), places=2,
                               msg=f"Expected total fees after LONG_CLOSE: {fee_buy + fee_sell}, but got {obs['fees']}")
        self.assertAlmostEqual(reward_close, float(decimal_to_float(expected_reward, precision=2)), places=2,
                               msg=f"Expected reward after LONG_CLOSE: {expected_reward}, but got {reward_close}")
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn('total_asset', info)

        # Verify total funds consistency
        total_funds_after = self.calculate_total_funds()
        expected_total_funds = total_funds_before - fee_buy + realized_pnl  # 手续费已扣除
        self.assertAlmostEqual(float(decimal_to_float(total_funds_after, precision=2)),
                               float(decimal_to_float(expected_total_funds, precision=2)),
                               places=2,
                               msg=f"Total funds mismatch after LONG_CLOSE. Expected: {expected_total_funds}, Got: {total_funds_after}")

    def test_step_short_open(self):
        """
        Test SHORT_OPEN action in the environment and assert the reward and funds.
        """
        obs, info = self.env.reset()
        total_funds_before = self.calculate_total_funds()
        
        action = Action.SHORT_OPEN.value
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Calculate expected values
        current_price = Decimal(str(self.env.df.loc[self.env.df.index[self.env.current_step - 1], 'Close']))
        spread = Decimal(str(self.env.spread))
        bid_price = current_price - spread
        trading_fees = Decimal(str(self.env.trading_fees))
        trade_lot = Decimal(str(self.env.trade_lot))
        lot_size = Decimal(str(self.env.lot_size))
        leverage = Decimal(str(self.env.leverage))

        # Revenue = trade_lot * lot_size * bid_price
        revenue = trade_lot * lot_size * bid_price
        fee = revenue * trading_fees
        total_revenue = revenue - fee

        # Expected balance after SHORT_OPEN
        expected_balance = self.env.user_accounts.balance.get_balance() + total_revenue

        # Expected short_position after SHORT_OPEN
        expected_short_position = trade_lot

        # Used Margin after SHORT_OPEN
        required_margin = (trade_lot * lot_size * bid_price) / leverage
        total_used_margin = required_margin

        # P&L after SHORT_OPEN
        pnl = (bid_price - current_price) * trade_lot * lot_size
        equity = expected_balance + pnl

        # Free Margin after SHORT_OPEN
        free_margin = equity - total_used_margin

        # Expected reward: change in equity = equity - previous_equity
        expected_reward = equity - Decimal(str(self.env.initial_balance))

        # Verify account balances and positions
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
        self.assertAlmostEqual(obs['realized_pnl'], 0.0, places=2,
                               msg=f"Realized P&L should remain 0.0 after SHORT_OPEN.")
        self.assertAlmostEqual(obs['unrealized_pnl'], float(decimal_to_float(pnl, precision=2)), places=2,
                               msg=f"Expected unrealized P&L after SHORT_OPEN: {pnl}, but got {obs['unrealized_pnl']}")
        self.assertAlmostEqual(obs['fees'], float(decimal_to_float(fee, precision=2)), places=2,
                               msg=f"Expected fees after SHORT_OPEN: {fee}, but got {obs['fees']}")
        self.assertAlmostEqual(reward, float(decimal_to_float(expected_reward, precision=2)), places=2,
                               msg=f"Expected reward: {expected_reward}, but got {reward}")
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn('total_asset', info)

        # Verify total funds consistency
        total_funds_after = self.calculate_total_funds()
        expected_total_funds = total_funds_before - fee  # 手续费由代理商收取
        self.assertAlmostEqual(float(decimal_to_float(total_funds_after, precision=2)),
                               float(decimal_to_float(expected_total_funds, precision=2)),
                               places=2,
                               msg=f"Total funds mismatch after SHORT_OPEN. Expected: {expected_total_funds}, Got: {total_funds_after}")

    def test_step_short_close(self):
        """
        Test SHORT_CLOSE action in the environment and assert the reward and funds.
        """
        obs, info = self.env.reset()
        total_funds_before = self.calculate_total_funds()
        
        # First, execute SHORT_OPEN
        action_open = Action.SHORT_OPEN.value
        obs, reward_open, terminated, truncated, info = self.env.step(action_open)
        total_funds_after_open = self.calculate_total_funds()

        # Capture previous equity after SHORT_OPEN
        previous_equity = self.env._calculate_equity()

        # Now, execute SHORT_CLOSE
        action_close = Action.SHORT_CLOSE.value
        obs, reward_close, terminated, truncated, info = self.env.step(action_close)

        # Retrieve prices
        open_step = self.env.current_step - 2
        close_step = self.env.current_step - 1
        open_price = Decimal(str(self.env.df.loc[self.env.df.index[open_step], 'Close']))
        close_price = Decimal(str(self.env.df.loc[self.env.df.index[close_step], 'Close']))
        spread = Decimal(str(self.env.spread))
        trading_fees = Decimal(str(self.env.trading_fees))
        trade_lot = Decimal(str(self.env.trade_lot))
        lot_size = Decimal(str(self.env.lot_size))
        leverage = Decimal(str(self.env.leverage))
    
        # Bid price for SHORT_OPEN
        bid_price = open_price - spread
        # Ask price for SHORT_CLOSE
        ask_price = close_price + spread
    
        # Realized P&L during SHORT_CLOSE
        pnl = (bid_price - ask_price) * trade_lot * lot_size
        fee_buy = (trade_lot * lot_size * ask_price) * trading_fees
        realized_pnl = pnl - fee_buy
    
        # Expected balance after SHORT_CLOSE
        cost = trade_lot * lot_size * ask_price
        fee_sell = (trade_lot * lot_size * bid_price) * trading_fees
        total_revenue = pnl - fee_buy
        expected_balance = self.env.user_accounts.balance.get_balance() + total_revenue
    
        # Expected short_position after SHORT_CLOSE
        expected_short_position = Decimal('0.0')
    
        # Used Margin after SHORT_CLOSE
        released_margin = (trade_lot * lot_size * bid_price) / leverage
        total_used_margin = Decimal('0.0')
    
        # Equity after SHORT_CLOSE
        equity = expected_balance + Decimal('0.0') + Decimal('0.0')  # realized_pnl already added to balance
    
        # Free Margin after SHORT_CLOSE
        free_margin = equity - total_used_margin
    
        # Expected reward: change in equity = equity - previous_equity
        expected_reward = equity - previous_equity

        # Verify account balances and positions
        balance = get_scalar(obs, 'balance')
        equity_obs = get_scalar(obs, 'equity')
        used_margin = get_scalar(obs, 'used_margin')
        free_margin_obs = get_scalar(obs, 'free_margin')
        long_position = get_scalar(obs, 'long_position')
        short_position = get_scalar(obs, 'short_position')
        unrealized_pnl = get_scalar(obs, 'unrealized_pnl')
        realized_pnl = get_scalar(obs, 'realized_pnl')
        fees = get_scalar(obs, 'fees')

        self.assertAlmostEqual(balance, float(decimal_to_float(expected_balance, precision=2)), places=2,
                               msg=f"Expected balance after SHORT_CLOSE: {expected_balance}, but got {obs['balance']}")
        self.assertAlmostEqual(equity_obs, float(decimal_to_float(equity, precision=2)), places=2,
                               msg=f"Expected equity after SHORT_CLOSE: {equity}, but got {obs['equity']}")
        self.assertAlmostEqual(used_margin, float(decimal_to_float(total_used_margin, precision=2)), places=2,
                               msg=f"Expected used_margin after SHORT_CLOSE: {total_used_margin}, but got {obs['used_margin']}")
        self.assertAlmostEqual(free_margin_obs, float(decimal_to_float(free_margin, precision=2)), places=2,
                               msg=f"Expected free_margin after SHORT_CLOSE: {free_margin}, but got {obs['free_margin']}")
        self.assertAlmostEqual(short_position, 0.0, places=5,
                               msg=f"Expected short_position after SHORT_CLOSE: 0.0, but got {obs['short_position']}")
        self.assertAlmostEqual(long_position, 0.0, places=5,
                               msg=f"Expected long_position to remain 0.0, but got {obs['long_position']}")
        self.assertAlmostEqual(realized_pnl, float(decimal_to_float(realized_pnl, precision=2)), places=2,
                               msg=f"Expected realized P&L after SHORT_CLOSE: {realized_pnl}, but got {obs['realized_pnl']}")
        self.assertAlmostEqual(unrealized_pnl, 0.0, places=2,
                               msg=f"Expected unrealized P&L after SHORT_CLOSE: 0.0, but got {obs['unrealized_pnl']}")
        self.assertAlmostEqual(fees, float(decimal_to_float(fee_buy + fee_sell, precision=2)), places=2,
                               msg=f"Expected total fees after SHORT_CLOSE: {fee_buy + fee_sell}, but got {obs['fees']}")
        self.assertAlmostEqual(reward_close, float(decimal_to_float(expected_reward, precision=2)), places=2,
                               msg=f"Expected reward after SHORT_CLOSE: {expected_reward}, but got {reward_close}")
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn('total_asset', info)

        # Verify total funds consistency
        total_funds_after = self.calculate_total_funds()
        expected_total_funds = total_funds_before - fee_buy + realized_pnl  # 手续费已扣除
        self.assertAlmostEqual(float(decimal_to_float(total_funds_after, precision=2)),
                               float(decimal_to_float(expected_total_funds, precision=2)),
                               places=2,
                               msg=f"Total funds mismatch after SHORT_CLOSE. Expected: {expected_total_funds}, Got: {total_funds_after}")

    def test_fee_deduction(self):
        """
        Test that trading fees are correctly deducted and recorded.
        """
        obs, info = self.env.reset()
        total_funds_before = self.calculate_total_funds()
        
        # Execute LONG_OPEN
        action_open = Action.LONG_OPEN.value
        obs, reward, terminated, truncated, info = self.env.step(action_open)
        
        # Calculate expected fee
        current_step = self.env.current_step - 1
        current_price = Decimal(str(self.env.df.loc[self.env.df.index[current_step], 'Close']))
        spread = Decimal(str(self.env.spread))
        ask_price = current_price + spread
        trading_fees = Decimal(str(self.env.trading_fees))
        trade_lot = Decimal(str(self.env.trade_lot))
        lot_size = Decimal(str(self.env.lot_size))
        
        # Cost = trade_lot * lot_size * ask_price
        cost = trade_lot * lot_size * ask_price
        fee_buy = cost * trading_fees
        
        # Total fees collected
        expected_fee_balance = fee_buy
        self.assertAlmostEqual(obs['fees'], float(decimal_to_float(expected_fee_balance, precision=2)), places=2,
                               msg=f"Expected broker fees balance: {expected_fee_balance}, but got {obs['fees']}")
        
        # Check user's balance after fee deduction
        expected_balance = Decimal(str(self.env.initial_balance)) - (cost + fee_buy)
        self.assertAlmostEqual(obs['balance'], float(decimal_to_float(expected_balance, precision=2)), places=2,
                               msg=f"Expected user balance after fee deduction: {expected_balance}, but got {obs['balance']}")
        
        # Verify total funds consistency
        total_funds_after = self.calculate_total_funds()
        expected_total_funds = self.initial_total_funds - fee_buy
        self.assertAlmostEqual(float(decimal_to_float(total_funds_after, precision=2)),
                               float(decimal_to_float(expected_total_funds, precision=2)),
                               places=2,
                               msg=f"Total funds mismatch after fee deduction. Expected: {expected_total_funds}, Got: {total_funds_after}")



    def test_continuous_long_open(self):
        """
        Test continuous LONG_OPEN actions up to the maximum long position limit and assert the reward and funds.
        """
        obs, info = self.env.reset()
        total_funds_before = self.calculate_total_funds()
        
        action = Action.LONG_OPEN.value
        for step in range(1, 4):  # Attempt to open 0.03 lots, exceeding the 0.02 lot limit
            with self.subTest(step=step):
                previous_equity = self.env._calculate_equity()
                total_funds_before_step = self.calculate_total_funds()
                
                obs, reward, terminated, truncated, info = self.env.step(action)
    
                # Retrieve current state
                current_step = self.env.current_step - 1
                current_price = Decimal(str(self.env.df.loc[self.env.df.index[current_step], 'Close']))
                spread = Decimal(str(self.env.spread))
                ask_price = current_price + spread
                trading_fees = Decimal(str(self.env.trading_fees))
                trade_lot = Decimal(str(self.env.trade_lot))
                lot_size = Decimal(str(self.env.lot_size))
                leverage = Decimal(str(self.env.leverage))
    
                # Determine actual trade_lot (may be limited by max_long_position)
                max_allowed_trade_lot = Decimal(str(self.env.max_long_position)) - self.env.user_accounts.long_position
                actual_trade_lot = min(trade_lot, max_allowed_trade_lot)
    
                # Calculate cost and fee for actual_trade_lot
                cost = actual_trade_lot * lot_size * ask_price
                fee = cost * trading_fees
                total_cost = cost + fee
    
                # Required margin
                required_margin = (actual_trade_lot * lot_size * ask_price) / leverage
    
                # Expected balance after LONG_OPEN
                expected_balance = self.env.user_accounts.balance.get_balance() - total_cost
    
                # Expected long_position after LONG_OPEN
                expected_long_position = self.env.user_accounts.long_position + actual_trade_lot
    
                # Used Margin after LONG_OPEN
                total_used_margin = required_margin
    
                # Equity after LONG_OPEN
                pnl = (current_price - ask_price) * actual_trade_lot * lot_size
                equity = expected_balance + pnl
    
                # Free Margin after LONG_OPEN
                free_margin = equity - total_used_margin
    
                # Expected reward: change in equity = equity - previous_equity
                expected_reward = equity - previous_equity
    
                # Verify account balances and positions
                balance = get_scalar(obs, 'balance')
                equity_obs = get_scalar(obs, 'equity')
                used_margin = get_scalar(obs, 'used_margin')
                free_margin_obs = get_scalar(obs, 'free_margin')
                long_position = get_scalar(obs, 'long_position')
                short_position = get_scalar(obs, 'short_position')
                unrealized_pnl = get_scalar(obs, 'unrealized_pnl')
                realized_pnl = get_scalar(obs, 'realized_pnl')
                fees = get_scalar(obs, 'fees')

                self.assertAlmostEqual(balance, float(decimal_to_float(expected_balance, precision=2)), places=2,
                                    msg=f"Expected balance after LONG_OPEN: {expected_balance}, but got {balance}")
                self.assertAlmostEqual(equity_obs, float(decimal_to_float(equity, precision=2)), places=2,
                                    msg=f"Expected equity after LONG_OPEN: {equity}, but got {equity_obs}")
                self.assertAlmostEqual(used_margin, float(decimal_to_float(total_used_margin, precision=2)), places=2,
                                    msg=f"Expected used_margin after LONG_OPEN: {total_used_margin}, but got {used_margin}")
                self.assertAlmostEqual(free_margin_obs, float(decimal_to_float(free_margin, precision=2)), places=2,
                                    msg=f"Expected free_margin after LONG_OPEN: {free_margin}, but got {free_margin_obs}")
                self.assertAlmostEqual(long_position, float(expected_long_position), places=5,
                                    msg=f"Expected long_position after LONG_OPEN: {expected_long_position}, but got {long_position}")
                self.assertAlmostEqual(short_position, 0.0, places=5,
                                    msg=f"Expected short_position to remain 0.0, but got {short_position}")
                self.assertAlmostEqual(unrealized_pnl, float(decimal_to_float(pnl, precision=2)), places=2,
                                    msg=f"Expected unrealized P&L after LONG_OPEN: {pnl}, but got {unrealized_pnl}")
                self.assertAlmostEqual(fees, float(decimal_to_float(fee, precision=2)), places=2,
                                    msg=f"Expected fees after LONG_OPEN: {fee}, but got {fees}")
                self.assertAlmostEqual(reward, float(decimal_to_float(expected_reward, precision=2)), places=2,
                                       msg=f"Expected reward at step {step}: {expected_reward}, but got {reward}")
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertIn('total_asset', info)
    
                # Verify total funds consistency
                total_funds_after_step = self.calculate_total_funds()
                expected_total_funds = total_funds_before_step - fee  # 手续费由代理商收取
                self.assertAlmostEqual(float(decimal_to_float(total_funds_after_step, precision=2)),
                                       float(decimal_to_float(expected_total_funds, precision=2)),
                                       places=2,
                                       msg=f"Total funds mismatch after LONG_OPEN step {step}. Expected: {expected_total_funds}, Got: {total_funds_after_step}")

    def test_position_limit_long(self):
        """
        Test that the environment does not allow exceeding the maximum long position limit and assert the reward.
        """
        obs, info = self.env.reset()
        action = Action.LONG_OPEN.value
        for step in range(1, 4):  # Attempt to open 0.03 lots, exceeding the 0.02 lot limit
            with self.subTest(step=step):
                previous_equity = self.env._calculate_equity()
                total_funds_before_step = self.calculate_total_funds()
                
                obs, reward, terminated, truncated, info = self.env.step(action)
    
                # Retrieve current state
                current_step = self.env.current_step - 1
                current_price = Decimal(str(self.env.df.loc[self.env.df.index[current_step], 'Close']))
                spread = Decimal(str(self.env.spread))
                ask_price = current_price + spread
                trading_fees = Decimal(str(self.env.trading_fees))
                trade_lot = Decimal(str(self.env.trade_lot))
                lot_size = Decimal(str(self.env.lot_size))
                leverage = Decimal(str(self.env.leverage))
    
                # Determine actual trade_lot (may be limited by max_long_position)
                max_allowed_trade_lot = Decimal(str(self.env.max_long_position)) - self.env.user_accounts.long_position
                actual_trade_lot = min(trade_lot, max_allowed_trade_lot)
    
                # Calculate cost and fee for actual_trade_lot
                cost = actual_trade_lot * lot_size * ask_price
                fee = cost * trading_fees
                total_cost = cost + fee
    
                # Required margin
                required_margin = (actual_trade_lot * lot_size * ask_price) / leverage
    
                # Expected balance after LONG_OPEN
                expected_balance = self.env.user_accounts.balance.get_balance() - total_cost
    
                # Expected long_position after LONG_OPEN
                expected_long_position = self.env.user_accounts.long_position + actual_trade_lot
    
                # Used Margin after LONG_OPEN
                total_used_margin = required_margin
    
                # Equity after LONG_OPEN
                pnl = (current_price - ask_price) * actual_trade_lot * lot_size
                equity = expected_balance + pnl
    
                # Free Margin after LONG_OPEN
                free_margin = equity - total_used_margin
    
                # Expected reward: change in equity = equity - previous_equity
                expected_reward = equity - previous_equity
    
                # Verify account balances and positions
                self.assertLessEqual(self.env.user_accounts.long_position, self.env.max_long_position,
                                     msg=f"Long position exceeded max_long_position at step {step}: {self.env.user_accounts.long_position} > {self.env.max_long_position}")
                self.assertAlmostEqual(obs['balance'], float(decimal_to_float(expected_balance, precision=2)), places=2,
                                       msg=f"Expected balance after LONG_OPEN step {step}: {expected_balance}, but got {obs['balance']}")
                self.assertAlmostEqual(obs['long_position'], float(decimal_to_float(expected_long_position, precision=5)), places=5,
                                       msg=f"Expected long_position after LONG_OPEN step {step}: {expected_long_position}, but got {obs['long_position']}")
                self.assertAlmostEqual(obs['used_margin'], float(decimal_to_float(total_used_margin, precision=2)), places=2,
                                       msg=f"Expected used_margin after LONG_OPEN step {step}: {total_used_margin}, but got {obs['used_margin']}")
                self.assertAlmostEqual(obs['free_margin'], float(decimal_to_float(free_margin, precision=2)), places=2,
                                       msg=f"Expected free_margin after LONG_OPEN step {step}: {free_margin}, but got {obs['free_margin']}")
                self.assertAlmostEqual(obs['short_position'], 0.0, places=5,
                                       msg=f"Expected short_position to remain 0.0 at step {step}, but got {obs['short_position']}")
                self.assertAlmostEqual(obs['realized_pnl'], 0.0, places=2,
                                       msg=f"Realized P&L should remain 0.0 after LONG_OPEN step {step}.")
                self.assertAlmostEqual(obs['unrealized_pnl'], float(decimal_to_float(pnl, precision=2)), places=2,
                                       msg=f"Expected unrealized P&L after LONG_OPEN step {step}: {pnl}, but got {obs['unrealized_pnl']}")
                self.assertAlmostEqual(obs['fees'], float(decimal_to_float(self.env.broker_accounts.fees.get_balance(), precision=2)), places=2,
                                       msg=f"Expected fees after LONG_OPEN step {step}: {fee}, but got {obs['fees']}")
                self.assertAlmostEqual(reward, float(decimal_to_float(expected_reward, precision=2)), places=2,
                                       msg=f"Expected reward at step {step}: {expected_reward}, but got {reward}")
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertIn('total_asset', info)
    
                # Verify total funds consistency
                total_funds_after_step = self.calculate_total_funds()
                expected_total_funds = total_funds_before_step - fee  # 手续费由代理商收取
                self.assertAlmostEqual(float(decimal_to_float(total_funds_after_step, precision=2)),
                                       float(decimal_to_float(expected_total_funds, precision=2)),
                                       places=2,
                                       msg=f"Total funds mismatch after LONG_OPEN step {step}. Expected: {expected_total_funds}, Got: {total_funds_after_step}")

    def test_continuous_short_open(self):
        """
        Test continuous SHORT_OPEN actions up to the maximum short position limit and assert the reward and funds.
        """
        obs, info = self.env.reset()
        total_funds_before = self.calculate_total_funds()
        
        action = Action.SHORT_OPEN.value
        for step in range(1, 4):  # Attempt to open 0.03 lots, exceeding the 0.02 lot limit
            with self.subTest(step=step):
                previous_equity = self.env._calculate_equity()
                total_funds_before_step = self.calculate_total_funds()
                
                obs, reward, terminated, truncated, info = self.env.step(action)
    
                # Retrieve current state
                current_step = self.env.current_step - 1
                current_price = Decimal(str(self.env.df.loc[self.env.df.index[current_step], 'Close']))
                spread = Decimal(str(self.env.spread))
                bid_price = current_price - spread
                trading_fees = Decimal(str(self.env.trading_fees))
                trade_lot = Decimal(str(self.env.trade_lot))
                lot_size = Decimal(str(self.env.lot_size))
                leverage = Decimal(str(self.env.leverage))
    
                # Determine actual trade_lot (may be limited by max_short_position)
                max_allowed_trade_lot = Decimal(str(self.env.max_short_position)) - self.env.user_accounts.short_position
                actual_trade_lot = min(trade_lot, max_allowed_trade_lot)
    
                # Calculate revenue and fee for actual_trade_lot
                revenue = actual_trade_lot * lot_size * bid_price
                fee = revenue * trading_fees
                total_revenue = revenue - fee
    
                # Expected balance after SHORT_OPEN
                expected_balance = self.env.user_accounts.balance.get_balance() + total_revenue
    
                # Expected short_position after SHORT_OPEN
                expected_short_position = self.env.user_accounts.short_position + actual_trade_lot
    
                # Used Margin after SHORT_OPEN
                total_used_margin = (actual_trade_lot * lot_size * bid_price) / leverage
    
                # P&L after SHORT_OPEN
                pnl = (bid_price - current_price) * actual_trade_lot * lot_size
                equity = expected_balance + pnl
    
                # Free Margin after SHORT_OPEN
                free_margin = equity - total_used_margin
    
                # Expected reward: change in equity = equity - previous_equity
                expected_reward = equity - previous_equity
    
                # Verify account balances and positions
                self.assertLessEqual(self.env.user_accounts.short_position, self.env.max_short_position,
                                     msg=f"Short position exceeded max_short_position at step {step}: {self.env.user_accounts.short_position} > {self.env.max_short_position}")
                self.assertAlmostEqual(obs['balance'], float(decimal_to_float(expected_balance, precision=2)), places=2,
                                       msg=f"Expected balance after SHORT_OPEN step {step}: {expected_balance}, but got {obs['balance']}")
                self.assertAlmostEqual(obs['short_position'], float(decimal_to_float(expected_short_position, precision=5)), places=5,
                                       msg=f"Expected short_position after SHORT_OPEN step {step}: {expected_short_position}, but got {obs['short_position']}")
                self.assertAlmostEqual(obs['used_margin'], float(decimal_to_float(total_used_margin, precision=2)), places=2,
                                       msg=f"Expected used_margin after SHORT_OPEN step {step}: {total_used_margin}, but got {obs['used_margin']}")
                self.assertAlmostEqual(obs['free_margin'], float(decimal_to_float(free_margin, precision=2)), places=2,
                                       msg=f"Expected free_margin after SHORT_OPEN step {step}: {free_margin}, but got {obs['free_margin']}")
                self.assertAlmostEqual(obs['long_position'], 0.0, places=5,
                                       msg=f"Expected long_position to remain 0.0 at step {step}, but got {obs['long_position']}")
                self.assertAlmostEqual(obs['realized_pnl'], 0.0, places=2,
                                       msg=f"Realized P&L should remain 0.0 after SHORT_OPEN step {step}.")
                self.assertAlmostEqual(obs['unrealized_pnl'], float(decimal_to_float(pnl, precision=2)), places=2,
                                       msg=f"Expected unrealized P&L after SHORT_OPEN step {step}: {pnl}, but got {obs['unrealized_pnl']}")
                self.assertAlmostEqual(obs['fees'], float(decimal_to_float(fee, precision=2)), places=2,
                                       msg=f"Expected fees after SHORT_OPEN step {step}: {fee}, but got {obs['fees']}")
                self.assertAlmostEqual(reward, float(decimal_to_float(expected_reward, precision=2)), places=2,
                                       msg=f"Expected reward at step {step}: {expected_reward}, but got {reward}")
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertIn('total_asset', info)
    
                # Verify total funds consistency
                total_funds_after_step = self.calculate_total_funds()
                expected_total_funds = total_funds_before_step - fee  # 手续费由代理商收取
                self.assertAlmostEqual(float(decimal_to_float(total_funds_after_step, precision=2)),
                                       float(decimal_to_float(expected_total_funds, precision=2)),
                                       places=2,
                                       msg=f"Total funds mismatch after SHORT_OPEN step {step}. Expected: {expected_total_funds}, Got: {total_funds_after_step}")

    def test_position_limit_short(self):
        """
        Test that the environment does not allow exceeding the maximum short position limit and assert the reward and funds.
        """
        obs, info = self.env.reset()
        action = Action.SHORT_OPEN.value
        for step in range(1, 4):  # Attempt to open 0.03 lots, exceeding the 0.02 lot limit
            with self.subTest(step=step):
                previous_equity = self.env._calculate_equity()
                total_funds_before_step = self.calculate_total_funds()
                
                obs, reward, terminated, truncated, info = self.env.step(action)
    
                # Retrieve current state
                current_step = self.env.current_step - 1
                current_price = Decimal(str(self.env.df.loc[self.env.df.index[current_step], 'Close']))
                spread = Decimal(str(self.env.spread))
                bid_price = current_price - spread
                trading_fees = Decimal(str(self.env.trading_fees))
                trade_lot = Decimal(str(self.env.trade_lot))
                lot_size = Decimal(str(self.env.lot_size))
                leverage = Decimal(str(self.env.leverage))
    
                # Determine actual trade_lot (may be limited by max_short_position)
                max_allowed_trade_lot = Decimal(str(self.env.max_short_position)) - self.env.user_accounts.short_position
                actual_trade_lot = min(trade_lot, max_allowed_trade_lot)
    
                # Calculate revenue and fee for actual_trade_lot
                revenue = actual_trade_lot * lot_size * bid_price
                fee = revenue * trading_fees
                total_revenue = revenue - fee
    
                # Expected balance after SHORT_OPEN
                expected_balance = self.env.user_accounts.balance.get_balance() + total_revenue
    
                # Expected short_position after SHORT_OPEN
                expected_short_position = self.env.user_accounts.short_position + actual_trade_lot
    
                # Used Margin after SHORT_OPEN
                total_used_margin = (actual_trade_lot * lot_size * bid_price) / leverage
    
                # P&L after SHORT_OPEN
                pnl = (bid_price - current_price) * actual_trade_lot * lot_size
                equity = expected_balance + pnl
    
                # Free Margin after SHORT_OPEN
                free_margin = equity - total_used_margin
    
                # Expected reward: change in equity = equity - previous_equity
                expected_reward = equity - previous_equity
    
                # Verify account balances and positions
                self.assertLessEqual(self.env.user_accounts.short_position, self.env.max_short_position,
                                     msg=f"Short position exceeded max_short_position at step {step}: {self.env.user_accounts.short_position} > {self.env.max_short_position}")
                self.assertAlmostEqual(obs['balance'], float(decimal_to_float(expected_balance, precision=2)), places=2,
                                       msg=f"Expected balance after SHORT_OPEN step {step}: {expected_balance}, but got {obs['balance']}")
                self.assertAlmostEqual(obs['short_position'], float(decimal_to_float(expected_short_position, precision=5)), places=5,
                                       msg=f"Expected short_position after SHORT_OPEN step {step}: {expected_short_position}, but got {obs['short_position']}")
                self.assertAlmostEqual(obs['used_margin'], float(decimal_to_float(total_used_margin, precision=2)), places=2,
                                       msg=f"Expected used_margin after SHORT_OPEN step {step}: {total_used_margin}, but got {obs['used_margin']}")
                self.assertAlmostEqual(obs['free_margin'], float(decimal_to_float(free_margin, precision=2)), places=2,
                                       msg=f"Expected free_margin after SHORT_OPEN step {step}: {free_margin}, but got {obs['free_margin']}")
                self.assertAlmostEqual(obs['long_position'], 0.0, places=5,
                                       msg=f"Expected long_position to remain 0.0 at step {step}, but got {obs['long_position']}")
                self.assertAlmostEqual(obs['realized_pnl'], 0.0, places=2,
                                       msg=f"Realized P&L should remain 0.0 after SHORT_OPEN step {step}.")
                self.assertAlmostEqual(obs['unrealized_pnl'], float(decimal_to_float(pnl, precision=2)), places=2,
                                       msg=f"Expected unrealized P&L after SHORT_OPEN step {step}: {pnl}, but got {obs['unrealized_pnl']}")
                self.assertAlmostEqual(obs['fees'], float(decimal_to_float(fee, precision=2)), places=2,
                                       msg=f"Expected fees after SHORT_OPEN step {step}: {fee}, but got {obs['fees']}")
                self.assertAlmostEqual(reward, float(decimal_to_float(expected_reward, precision=2)), places=2,
                                       msg=f"Expected reward at step {step}: {expected_reward}, but got {reward}")
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertIn('total_asset', info)
    
                # Verify total funds consistency
                total_funds_after_step = self.calculate_total_funds()
                expected_total_funds = total_funds_before_step - fee  # 手续费由代理商收取
                self.assertAlmostEqual(float(decimal_to_float(total_funds_after_step, precision=2)),
                                       float(decimal_to_float(expected_total_funds, precision=2)),
                                       places=2,
                                       msg=f"Total funds mismatch after SHORT_OPEN step {step}. Expected: {expected_total_funds}, Got: {total_funds_after_step}")

    def test_position_limit_long(self):
        """
        Test that the environment does not allow exceeding the maximum long position limit and assert the reward and funds.
        """
        obs, info = self.env.reset()
        action = Action.LONG_OPEN.value
        for step in range(1, 4):  # Attempt to open 0.03 lots, exceeding the 0.02 lot limit
            with self.subTest(step=step):
                previous_equity = self.env._calculate_equity()
                total_funds_before_step = self.calculate_total_funds()
                
                obs, reward, terminated, truncated, info = self.env.step(action)
    
                # Retrieve current state
                current_step = self.env.current_step - 1
                current_price = Decimal(str(self.env.df.loc[self.env.df.index[current_step], 'Close']))
                spread = Decimal(str(self.env.spread))
                ask_price = current_price + spread
                trading_fees = Decimal(str(self.env.trading_fees))
                trade_lot = Decimal(str(self.env.trade_lot))
                lot_size = Decimal(str(self.env.lot_size))
                leverage = Decimal(str(self.env.leverage))
    
                # Determine actual trade_lot (may be limited by max_long_position)
                max_allowed_trade_lot = Decimal(str(self.env.max_long_position)) - self.env.user_accounts.long_position
                actual_trade_lot = min(trade_lot, max_allowed_trade_lot)
    
                # Calculate cost and fee for actual_trade_lot
                cost = actual_trade_lot * lot_size * ask_price
                fee = cost * trading_fees
                total_cost = cost + fee
    
                # Required margin
                required_margin = (actual_trade_lot * lot_size * ask_price) / leverage
    
                # Expected balance after LONG_OPEN
                expected_balance = self.env.user_accounts.balance.get_balance() - total_cost
    
                # Expected long_position after LONG_OPEN
                expected_long_position = self.env.user_accounts.long_position + actual_trade_lot
    
                # Used Margin after LONG_OPEN
                total_used_margin = required_margin
    
                # Equity after LONG_OPEN
                pnl = (current_price - ask_price) * actual_trade_lot * lot_size
                equity = expected_balance + pnl
    
                # Free Margin after LONG_OPEN
                free_margin = equity - total_used_margin
    
                # Expected reward: change in equity = equity - previous_equity
                expected_reward = equity - previous_equity
    
                # Verify account balances and positions
                self.assertLessEqual(self.env.user_accounts.long_position, self.env.max_long_position,
                                     msg=f"Long position exceeded max_long_position at step {step}: {self.env.user_accounts.long_position} > {self.env.max_long_position}")
                self.assertAlmostEqual(obs['balance'], float(decimal_to_float(expected_balance, precision=2)), places=2,
                                       msg=f"Expected balance after LONG_OPEN step {step}: {expected_balance}, but got {obs['balance']}")
                self.assertAlmostEqual(obs['long_position'], float(decimal_to_float(expected_long_position, precision=5)), places=5,
                                       msg=f"Expected long_position after LONG_OPEN step {step}: {expected_long_position}, but got {obs['long_position']}")
                self.assertAlmostEqual(obs['used_margin'], float(decimal_to_float(total_used_margin, precision=2)), places=2,
                                       msg=f"Expected used_margin after LONG_OPEN step {step}: {total_used_margin}, but got {obs['used_margin']}")
                self.assertAlmostEqual(obs['free_margin'], float(decimal_to_float(free_margin, precision=2)), places=2,
                                       msg=f"Expected free_margin after LONG_OPEN step {step}: {free_margin}, but got {obs['free_margin']}")
                self.assertAlmostEqual(obs['short_position'], 0.0, places=5,
                                       msg=f"Expected short_position to remain 0.0 at step {step}, but got {obs['short_position']}")
                self.assertAlmostEqual(obs['realized_pnl'], 0.0, places=2,
                                       msg=f"Realized P&L should remain 0.0 after LONG_OPEN step {step}.")
                self.assertAlmostEqual(obs['unrealized_pnl'], float(decimal_to_float(pnl, precision=2)), places=2,
                                       msg=f"Expected unrealized P&L after LONG_OPEN step {step}: {pnl}, but got {obs['unrealized_pnl']}")
                self.assertAlmostEqual(obs['fees'], float(decimal_to_float(self.env.broker_accounts.fees.get_balance(), precision=2)), places=2,
                                       msg=f"Expected fees after LONG_OPEN step {step}: {fee}, but got {obs['fees']}")
                self.assertAlmostEqual(reward, float(decimal_to_float(expected_reward, precision=2)), places=2,
                                       msg=f"Expected reward at step {step}: {expected_reward}, but got {reward}")
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertIn('total_asset', info)
    
                # Verify total funds consistency
                total_funds_after_step = self.calculate_total_funds()
                expected_total_funds = total_funds_before_step - fee  # 手续费由代理商收取
                self.assertAlmostEqual(float(decimal_to_float(total_funds_after_step, precision=2)),
                                       float(decimal_to_float(expected_total_funds, precision=2)),
                                       places=2,
                                       msg=f"Total funds mismatch after LONG_OPEN step {step}. Expected: {expected_total_funds}, Got: {total_funds_after_step}")

    def test_position_limit_short(self):
        """
        Test that the environment does not allow exceeding the maximum short position limit and assert the reward and funds.
        """
        obs, info = self.env.reset()
        action = Action.SHORT_OPEN.value
        for step in range(1, 4):  # Attempt to open 0.03 lots, exceeding the 0.02 lot limit
            with self.subTest(step=step):
                previous_equity = self.env._calculate_equity()
                total_funds_before_step = self.calculate_total_funds()
                
                obs, reward, terminated, truncated, info = self.env.step(action)
    
                # Retrieve current state
                current_step = self.env.current_step - 1
                current_price = Decimal(str(self.env.df.loc[self.env.df.index[current_step], 'Close']))
                spread = Decimal(str(self.env.spread))
                bid_price = current_price - spread
                trading_fees = Decimal(str(self.env.trading_fees))
                trade_lot = Decimal(str(self.env.trade_lot))
                lot_size = Decimal(str(self.env.lot_size))
                leverage = Decimal(str(self.env.leverage))
    
                # Determine actual trade_lot (may be limited by max_short_position)
                max_allowed_trade_lot = Decimal(str(self.env.max_short_position)) - self.env.user_accounts.short_position
                actual_trade_lot = min(trade_lot, max_allowed_trade_lot)
    
                # Calculate revenue and fee for actual_trade_lot
                revenue = actual_trade_lot * lot_size * bid_price
                fee = revenue * trading_fees
                total_revenue = revenue - fee
    
                # Expected balance after SHORT_OPEN
                expected_balance = self.env.user_accounts.balance.get_balance() + total_revenue
    
                # Expected short_position after SHORT_OPEN
                expected_short_position = self.env.user_accounts.short_position + actual_trade_lot
    
                # Used Margin after SHORT_OPEN
                total_used_margin = (actual_trade_lot * lot_size * bid_price) / leverage
    
                # P&L after SHORT_OPEN
                pnl = (bid_price - current_price) * actual_trade_lot * lot_size
                equity = expected_balance + pnl
    
                # Free Margin after SHORT_OPEN
                free_margin = equity - total_used_margin
    
                # Expected reward: change in equity = equity - previous_equity
                expected_reward = equity - previous_equity
    
                # Verify account balances and positions
                self.assertLessEqual(self.env.user_accounts.short_position, self.env.max_short_position,
                                     msg=f"Short position exceeded max_short_position at step {step}: {self.env.user_accounts.short_position} > {self.env.max_short_position}")
                self.assertAlmostEqual(obs['balance'], float(decimal_to_float(expected_balance, precision=2)), places=2,
                                       msg=f"Expected balance after SHORT_OPEN step {step}: {expected_balance}, but got {obs['balance']}")
                self.assertAlmostEqual(obs['short_position'], float(decimal_to_float(expected_short_position, precision=5)), places=5,
                                       msg=f"Expected short_position after SHORT_OPEN step {step}: {expected_short_position}, but got {obs['short_position']}")
                self.assertAlmostEqual(obs['used_margin'], float(decimal_to_float(total_used_margin, precision=2)), places=2,
                                       msg=f"Expected used_margin after SHORT_OPEN step {step}: {total_used_margin}, but got {obs['used_margin']}")
                self.assertAlmostEqual(obs['free_margin'], float(decimal_to_float(free_margin, precision=2)), places=2,
                                       msg=f"Expected free_margin after SHORT_OPEN step {step}: {free_margin}, but got {obs['free_margin']}")
                self.assertAlmostEqual(obs['long_position'], 0.0, places=5,
                                       msg=f"Expected long_position to remain 0.0 at step {step}, but got {obs['long_position']}")
                self.assertAlmostEqual(obs['realized_pnl'], 0.0, places=2,
                                       msg=f"Realized P&L should remain 0.0 after SHORT_OPEN step {step}.")
                self.assertAlmostEqual(obs['unrealized_pnl'], float(decimal_to_float(pnl, precision=2)), places=2,
                                       msg=f"Expected unrealized P&L after SHORT_OPEN step {step}: {pnl}, but got {obs['unrealized_pnl']}")
                self.assertAlmostEqual(obs['fees'], float(decimal_to_float(fee, precision=2)), places=2,
                                       msg=f"Expected fees after SHORT_OPEN step {step}: {fee}, but got {obs['fees']}")
                self.assertAlmostEqual(reward, float(decimal_to_float(expected_reward, precision=2)), places=2,
                                       msg=f"Expected reward at step {step}: {expected_reward}, but got {reward}")
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertIn('total_asset', info)
    
                # Verify total funds consistency
                total_funds_after_step = self.calculate_total_funds()
                expected_total_funds = total_funds_before_step - fee  # 手续费由代理商收取
                self.assertAlmostEqual(float(decimal_to_float(total_funds_after_step, precision=2)),
                                       float(decimal_to_float(expected_total_funds, precision=2)),
                                       places=2,
                                       msg=f"Total funds mismatch after SHORT_OPEN step {step}. Expected: {expected_total_funds}, Got: {total_funds_after_step}")

    def test_position_limit_long(self):
        """
        Test that the environment does not allow exceeding the maximum long position limit and assert the reward and funds.
        """
        obs, info = self.env.reset()
        action = Action.LONG_OPEN.value
        for step in range(1, 4):  # Attempt to open 0.03 lots, exceeding the 0.02 lot limit
            with self.subTest(step=step):
                previous_equity = self.env._calculate_equity()
                total_funds_before_step = self.calculate_total_funds()
                
                obs, reward, terminated, truncated, info = self.env.step(action)
    
                # Retrieve current state
                current_step = self.env.current_step - 1
                current_price = Decimal(str(self.env.df.loc[self.env.df.index[current_step], 'Close']))
                spread = Decimal(str(self.env.spread))
                ask_price = current_price + spread
                trading_fees = Decimal(str(self.env.trading_fees))
                trade_lot = Decimal(str(self.env.trade_lot))
                lot_size = Decimal(str(self.env.lot_size))
                leverage = Decimal(str(self.env.leverage))
    
                # Determine actual trade_lot (may be limited by max_long_position)
                max_allowed_trade_lot = Decimal(str(self.env.max_long_position)) - self.env.user_accounts.long_position
                actual_trade_lot = min(trade_lot, max_allowed_trade_lot)
    
                # Calculate cost and fee for actual_trade_lot
                cost = actual_trade_lot * lot_size * ask_price
                fee = cost * trading_fees
                total_cost = cost + fee
    
                # Required margin
                required_margin = (actual_trade_lot * lot_size * ask_price) / leverage
    
                # Expected balance after LONG_OPEN
                expected_balance = self.env.user_accounts.balance.get_balance() - total_cost
    
                # Expected long_position after LONG_OPEN
                expected_long_position = self.env.user_accounts.long_position + actual_trade_lot
    
                # Used Margin after LONG_OPEN
                total_used_margin = required_margin
    
                # Equity after LONG_OPEN
                pnl = (current_price - ask_price) * actual_trade_lot * lot_size
                equity = expected_balance + pnl
    
                # Free Margin after LONG_OPEN
                free_margin = equity - total_used_margin
    
                # Expected reward: change in equity = equity - previous_equity
                expected_reward = equity - previous_equity
    
                # Verify account balances and positions
                self.assertLessEqual(self.env.user_accounts.long_position, self.env.max_long_position,
                                     msg=f"Long position exceeded max_long_position at step {step}: {self.env.user_accounts.long_position} > {self.env.max_long_position}")
                self.assertAlmostEqual(obs['balance'], float(decimal_to_float(expected_balance, precision=2)), places=2,
                                       msg=f"Expected balance after LONG_OPEN step {step}: {expected_balance}, but got {obs['balance']}")
                self.assertAlmostEqual(obs['long_position'], float(decimal_to_float(expected_long_position, precision=5)), places=5,
                                       msg=f"Expected long_position after LONG_OPEN step {step}: {expected_long_position}, but got {obs['long_position']}")
                self.assertAlmostEqual(obs['used_margin'], float(decimal_to_float(total_used_margin, precision=2)), places=2,
                                       msg=f"Expected used_margin after LONG_OPEN step {step}: {total_used_margin}, but got {obs['used_margin']}")
                self.assertAlmostEqual(obs['free_margin'], float(decimal_to_float(free_margin, precision=2)), places=2,
                                       msg=f"Expected free_margin after LONG_OPEN step {step}: {free_margin}, but got {obs['free_margin']}")
                self.assertAlmostEqual(obs['short_position'], 0.0, places=5,
                                       msg=f"Expected short_position to remain 0.0 at step {step}, but got {obs['short_position']}")
                self.assertAlmostEqual(obs['realized_pnl'], 0.0, places=2,
                                       msg=f"Realized P&L should remain 0.0 after LONG_OPEN step {step}.")
                self.assertAlmostEqual(obs['unrealized_pnl'], float(decimal_to_float(pnl, precision=2)), places=2,
                                       msg=f"Expected unrealized P&L after LONG_OPEN step {step}: {pnl}, but got {obs['unrealized_pnl']}")
                self.assertAlmostEqual(obs['fees'], float(decimal_to_float(self.env.broker_accounts.fees.get_balance(), precision=2)), places=2,
                                       msg=f"Expected fees after LONG_OPEN step {step}: {fee}, but got {obs['fees']}")
                self.assertAlmostEqual(reward, float(decimal_to_float(expected_reward, precision=2)), places=2,
                                       msg=f"Expected reward at step {step}: {expected_reward}, but got {reward}")
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertIn('total_asset', info)
    
                # Verify total funds consistency
                total_funds_after_step = self.calculate_total_funds()
                expected_total_funds = total_funds_before_step - fee  # 手续费由代理商收取
                self.assertAlmostEqual(float(decimal_to_float(total_funds_after_step, precision=2)),
                                       float(decimal_to_float(expected_total_funds, precision=2)),
                                       places=2,
                                       msg=f"Total funds mismatch after LONG_OPEN step {step}. Expected: {expected_total_funds}, Got: {total_funds_after_step}")

    def test_continuous_short_open(self):
        """
        Test continuous SHORT_OPEN actions up to the maximum short position limit and assert the reward and funds.
        """
        obs, info = self.env.reset()
        total_funds_before = self.calculate_total_funds()
        
        action = Action.SHORT_OPEN.value
        for step in range(1, 4):  # Attempt to open 0.03 lots, exceeding the 0.02 lot limit
            with self.subTest(step=step):
                previous_equity = self.env._calculate_equity()
                total_funds_before_step = self.calculate_total_funds()
                
                obs, reward, terminated, truncated, info = self.env.step(action)
    
                # Retrieve current state
                current_step = self.env.current_step - 1
                current_price = Decimal(str(self.env.df.loc[self.env.df.index[current_step], 'Close']))
                spread = Decimal(str(self.env.spread))
                bid_price = current_price - spread
                trading_fees = Decimal(str(self.env.trading_fees))
                trade_lot = Decimal(str(self.env.trade_lot))
                lot_size = Decimal(str(self.env.lot_size))
                leverage = Decimal(str(self.env.leverage))
    
                # Determine actual trade_lot (may be limited by max_short_position)
                max_allowed_trade_lot = Decimal(str(self.env.max_short_position)) - self.env.user_accounts.short_position
                actual_trade_lot = min(trade_lot, max_allowed_trade_lot)
    
                # Calculate revenue and fee for actual_trade_lot
                revenue = actual_trade_lot * lot_size * bid_price
                fee = revenue * trading_fees
                total_revenue = revenue - fee
    
                # Expected balance after SHORT_OPEN
                expected_balance = self.env.user_accounts.balance.get_balance() + total_revenue
    
                # Expected short_position after SHORT_OPEN
                expected_short_position = self.env.user_accounts.short_position + actual_trade_lot
    
                # Used Margin after SHORT_OPEN
                total_used_margin = (actual_trade_lot * lot_size * bid_price) / leverage
    
                # P&L after SHORT_OPEN
                pnl = (bid_price - current_price) * actual_trade_lot * lot_size
                equity = expected_balance + pnl
    
                # Free Margin after SHORT_OPEN
                free_margin = equity - total_used_margin
    
                # Expected reward: change in equity = equity - previous_equity
                expected_reward = equity - previous_equity
    
                # Verify account balances and positions
                self.assertLessEqual(self.env.user_accounts.short_position, self.env.max_short_position,
                                     msg=f"Short position exceeded max_short_position at step {step}: {self.env.user_accounts.short_position} > {self.env.max_short_position}")
                self.assertAlmostEqual(obs['balance'], float(decimal_to_float(expected_balance, precision=2)), places=2,
                                       msg=f"Expected balance after SHORT_OPEN step {step}: {expected_balance}, but got {obs['balance']}")
                self.assertAlmostEqual(obs['short_position'], float(decimal_to_float(expected_short_position, precision=5)), places=5,
                                       msg=f"Expected short_position after SHORT_OPEN step {step}: {expected_short_position}, but got {obs['short_position']}")
                self.assertAlmostEqual(obs['used_margin'], float(decimal_to_float(total_used_margin, precision=2)), places=2,
                                       msg=f"Expected used_margin after SHORT_OPEN step {step}: {total_used_margin}, but got {obs['used_margin']}")
                self.assertAlmostEqual(obs['free_margin'], float(decimal_to_float(free_margin, precision=2)), places=2,
                                       msg=f"Expected free_margin after SHORT_OPEN step {step}: {free_margin}, but got {obs['free_margin']}")
                self.assertAlmostEqual(obs['long_position'], 0.0, places=5,
                                       msg=f"Expected long_position to remain 0.0 at step {step}, but got {obs['long_position']}")
                self.assertAlmostEqual(obs['realized_pnl'], 0.0, places=2,
                                       msg=f"Realized P&L should remain 0.0 after SHORT_OPEN step {step}.")
                self.assertAlmostEqual(obs['unrealized_pnl'], float(decimal_to_float(pnl, precision=2)), places=2,
                                       msg=f"Expected unrealized P&L after SHORT_OPEN step {step}: {pnl}, but got {obs['unrealized_pnl']}")
                self.assertAlmostEqual(obs['fees'], float(decimal_to_float(fee, precision=2)), places=2,
                                       msg=f"Expected fees after SHORT_OPEN step {step}: {fee}, but got {obs['fees']}")
                self.assertAlmostEqual(reward, float(decimal_to_float(expected_reward, precision=2)), places=2,
                                       msg=f"Expected reward at step {step}: {expected_reward}, but got {reward}")
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertIn('total_asset', info)
    
                # Verify total funds consistency
                total_funds_after_step = self.calculate_total_funds()
                expected_total_funds = total_funds_before_step - fee  # 手续费由代理商收取
                self.assertAlmostEqual(float(decimal_to_float(total_funds_after_step, precision=2)),
                                       float(decimal_to_float(expected_total_funds, precision=2)),
                                       places=2,
                                       msg=f"Total funds mismatch after SHORT_OPEN step {step}. Expected: {expected_total_funds}, Got: {total_funds_after_step}")

    def test_position_limit_long(self):
        """
        Test that the environment does not allow exceeding the maximum long position limit and assert the reward and funds.
        """
        obs, info = self.env.reset()
        action = Action.LONG_OPEN.value
        for step in range(1, 4):  # Attempt to open 0.03 lots, exceeding the 0.02 lot limit
            with self.subTest(step=step):
                previous_equity = self.env._calculate_equity()
                total_funds_before_step = self.calculate_total_funds()
                
                obs, reward, terminated, truncated, info = self.env.step(action)
    
                # Retrieve current state
                current_step = self.env.current_step - 1
                current_price = Decimal(str(self.env.df.loc[self.env.df.index[current_step], 'Close']))
                spread = Decimal(str(self.env.spread))
                ask_price = current_price + spread
                trading_fees = Decimal(str(self.env.trading_fees))
                trade_lot = Decimal(str(self.env.trade_lot))
                lot_size = Decimal(str(self.env.lot_size))
                leverage = Decimal(str(self.env.leverage))
    
                # Determine actual trade_lot (may be limited by max_long_position)
                max_allowed_trade_lot = Decimal(str(self.env.max_long_position)) - self.env.user_accounts.long_position
                actual_trade_lot = min(trade_lot, max_allowed_trade_lot)
    
                # Calculate cost and fee for actual_trade_lot
                cost = actual_trade_lot * lot_size * ask_price
                fee = cost * trading_fees
                total_cost = cost + fee
    
                # Required margin
                required_margin = (actual_trade_lot * lot_size * ask_price) / leverage
    
                # Expected balance after LONG_OPEN
                expected_balance = self.env.user_accounts.balance.get_balance() - total_cost
    
                # Expected long_position after LONG_OPEN
                expected_long_position = self.env.user_accounts.long_position + actual_trade_lot
    
                # Used Margin after LONG_OPEN
                total_used_margin = required_margin
    
                # Equity after LONG_OPEN
                pnl = (current_price - ask_price) * actual_trade_lot * lot_size
                equity = expected_balance + pnl
    
                # Free Margin after LONG_OPEN
                free_margin = equity - total_used_margin
    
                # Expected reward: change in equity = equity - previous_equity
                expected_reward = equity - previous_equity
    
                # Verify account balances and positions
                balance = get_scalar(obs, 'balance')
                equity_obs = get_scalar(obs, 'equity')
                used_margin = get_scalar(obs, 'used_margin')
                free_margin_obs = get_scalar(obs, 'free_margin')
                long_position = get_scalar(obs, 'long_position')
                short_position = get_scalar(obs, 'short_position')
                unrealized_pnl = get_scalar(obs, 'unrealized_pnl')
                realized_pnl = get_scalar(obs, 'realized_pnl')
                fees = get_scalar(obs, 'fees')

                self.assertLessEqual(self.env.user_accounts.long_position, self.env.max_long_position,
                                     msg=f"Long position exceeded max_long_position at step {step}: {self.env.user_accounts.long_position} > {self.env.max_long_position}")
                self.assertAlmostEqual(balance, float(decimal_to_float(expected_balance, precision=2)), places=2,
                                       msg=f"Expected balance after LONG_OPEN step {step}: {expected_balance}, but got {obs['balance']}")
                self.assertAlmostEqual(long_position, float(decimal_to_float(expected_long_position, precision=5)), places=5,
                                       msg=f"Expected long_position after LONG_OPEN step {step}: {expected_long_position}, but got {obs['long_position']}")
                self.assertAlmostEqual(used_margin, float(decimal_to_float(total_used_margin, precision=2)), places=2,
                                       msg=f"Expected used_margin after LONG_OPEN step {step}: {total_used_margin}, but got {obs['used_margin']}")
                self.assertAlmostEqual(free_margin, float(decimal_to_float(free_margin, precision=2)), places=2,
                                       msg=f"Expected free_margin after LONG_OPEN step {step}: {free_margin}, but got {obs['free_margin']}")
                self.assertAlmostEqual(short_position, 0.0, places=5,
                                       msg=f"Expected short_position to remain 0.0 at step {step}, but got {obs['short_position']}")
                self.assertAlmostEqual(realized_pnl, 0.0, places=2,
                                       msg=f"Realized P&L should remain 0.0 after LONG_OPEN step {step}.")
                self.assertAlmostEqual(unrealized_pnl, float(decimal_to_float(pnl, precision=2)), places=2,
                                       msg=f"Expected unrealized P&L after LONG_OPEN step {step}: {pnl}, but got {obs['unrealized_pnl']}")
                self.assertAlmostEqual(realized_pnl, float(decimal_to_float(self.env.broker_accounts.fees.get_balance(), precision=2)), places=2,
                                       msg=f"Expected fees after LONG_OPEN step {step}: {fee}, but got {obs['fees']}")
                self.assertAlmostEqual(reward, float(decimal_to_float(expected_reward, precision=2)), places=2,
                                       msg=f"Expected reward at step {step}: {expected_reward}, but got {reward}")
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertIn('total_asset', info)
    
                # Verify total funds consistency
                total_funds_after_step = self.calculate_total_funds()
                expected_total_funds = total_funds_before_step - fee  # 手续费由代理商收取
                self.assertAlmostEqual(float(decimal_to_float(total_funds_after_step, precision=2)),
                                       float(decimal_to_float(expected_total_funds, precision=2)),
                                       places=2,
                                       msg=f"Total funds mismatch after LONG_OPEN step {step}. Expected: {expected_total_funds}, Got: {total_funds_after_step}")

    def test_insufficient_balance_long_open(self):
        """
        Test LONG_OPEN action when balance is insufficient and assert the reward and funds.
        """
        obs, info = self.env.reset()
        # Set balance low to prevent buy
        self.env.user_accounts.balance.balance = Decimal('0.0')
        total_funds_before = self.calculate_total_funds()
        
        action = Action.LONG_OPEN.value
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Balance and long_position should remain unchanged
        expected_balance = Decimal('0.0')
        expected_long_position = Decimal('0.0')
        expected_realized_pnl = Decimal('0.0')
        expected_unrealized_pnl = Decimal('0.0')
        expected_fees = Decimal('0.0')
        
        # Expected reward: change in equity = equity - initial_balance
        # Equity = balance + realized_pnl + unrealized_pnl = 0 + 0 + 0 = 0
        expected_reward = Decimal('0.0') - Decimal(str(self.env.initial_balance))  # 0 - initial_balance
        
        self.assertEqual(obs['balance'], float(decimal_to_float(expected_balance, precision=2)),
                         msg=f"Expected balance: {expected_balance}, but got {obs['balance']}")
        self.assertEqual(obs['long_position'], float(expected_long_position),
                         msg=f"Expected long_position: {expected_long_position}, but got {obs['long_position']}")
        self.assertEqual(obs['short_position'], 0.0,
                         msg=f"Short position should remain 0.0 after failed LONG_OPEN.")
        self.assertEqual(obs['realized_pnl'], 0.0,
                         msg=f"Realized P&L should remain 0.0 after failed LONG_OPEN.")
        self.assertEqual(obs['unrealized_pnl'], 0.0,
                         msg=f"Unrealized P&L should remain 0.0 after failed LONG_OPEN.")
        self.assertEqual(obs['fees'], 0.0,
                         msg=f"Fees should remain 0.0 after failed LONG_OPEN.")
        self.assertAlmostEqual(reward, float(decimal_to_float(expected_reward, precision=2)), places=2,
                               msg=f"Expected reward: {expected_reward}, but got {reward}")
        self.assertFalse(terminated, "Environment should not terminate on failed action.")
        self.assertFalse(truncated, "Environment should not truncate on failed action.")
        self.assertIn('total_asset', info)
    
        # Verify total funds consistency
        total_funds_after = self.calculate_total_funds()
        expected_total_funds = total_funds_before  # No change expected
        self.assertAlmostEqual(float(decimal_to_float(total_funds_after, precision=2)),
                               float(decimal_to_float(expected_total_funds, precision=2)),
                               places=2,
                               msg=f"Total funds mismatch after failed LONG_OPEN. Expected: {expected_total_funds}, Got: {total_funds_after}")

    def test_insufficient_balance_short_open(self):
        """
        Test SHORT_OPEN action when balance is insufficient and assert the reward and funds.
        """
        obs, info = self.env.reset()
        # Set balance low to prevent sell
        self.env.user_accounts.balance.balance = Decimal('-1.0')  # Negative balance
        total_funds_before = self.calculate_total_funds()
        
        action = Action.SHORT_OPEN.value
        obs, reward, terminated, truncated, info = self.env.step(action)
    
        # Balance and short_position should remain unchanged
        expected_balance = Decimal('-1.0')
        expected_short_position = Decimal('0.0')
    
        # Expected reward: change in equity = equity - initial_balance
        expected_reward = Decimal('-1.0') - Decimal(str(self.env.initial_balance))
    
        self.assertEqual(obs['balance'], float(decimal_to_float(expected_balance, precision=2)),
                         msg=f"Expected balance: {expected_balance}, but got {obs['balance']}")
        self.assertEqual(obs['short_position'], float(expected_short_position),
                         msg=f"Expected short_position: {expected_short_position}, but got {obs['short_position']}")
        self.assertAlmostEqual(self.env.user_accounts.long_position, 0.0, places=5,
                               msg=f"Expected long_position to remain 0.0, but got {self.env.long_position}")
        self.assertAlmostEqual(obs['realized_pnl'], 0.0, places=2,
                               msg=f"Realized P&L should remain 0.0 after failed SHORT_OPEN.")
        self.assertAlmostEqual(obs['unrealized_pnl'], 0.0, places=2,
                               msg=f"Unrealized P&L should remain 0.0 after failed SHORT_OPEN.")
        self.assertAlmostEqual(obs['fees'], 0.0, places=2,
                               msg=f"Fees should remain 0.0 after failed SHORT_OPEN.")
        self.assertAlmostEqual(reward, float(decimal_to_float(expected_reward, precision=2)), places=2,
                               msg=f"Expected reward: {expected_reward}, but got {reward}")
        self.assertTrue(terminated, "Environment should terminate on failed action.")
        self.assertFalse(truncated, "Environment should not truncate on failed action.")
        self.assertIn('total_asset', info)
    
        # Verify total funds consistency
        total_funds_after = self.calculate_total_funds()
        expected_total_funds = total_funds_before  # No change expected
        self.assertAlmostEqual(float(decimal_to_float(total_funds_after, precision=2)),
                               float(decimal_to_float(expected_total_funds, precision=2)),
                               places=2,
                               msg=f"Total funds mismatch after failed SHORT_OPEN. Expected: {expected_total_funds}, Got: {total_funds_after}")

    def test_broker_fees_account(self):
        """
        Test that broker's fees account correctly accumulates fees from multiple transactions.
        """
        obs, info = self.env.reset()
        total_funds_before = self.calculate_total_funds()
        
        # Execute multiple LONG_OPEN actions
        action_open = Action.LONG_OPEN.value
        for _ in range(2):
            obs, reward, terminated, truncated, info = self.env.step(action_open)
        
        # Calculate expected total fees
        current_step = self.env.current_step - 1
        current_price = Decimal(str(self.env.df.loc[self.env.df.index[current_step], 'Close']))
        spread = Decimal(str(self.env.spread))
        ask_price = current_price + spread
        trading_fees = Decimal(str(self.env.trading_fees))
        trade_lot = Decimal(str(self.env.trade_lot))
        lot_size = Decimal(str(self.env.lot_size))
        
        # Fee per LONG_OPEN
        cost = trade_lot * lot_size * ask_price
        fee = cost * trading_fees
        total_fees = fee * 2  # Two transactions
        
        # Verify broker's fees account
        expected_fee_balance = total_fees
        self.assertAlmostEqual(obs['fees'], float(decimal_to_float(expected_fee_balance, precision=2)), places=2,
                            msg=f"Expected broker fees balance: {expected_fee_balance}, but got {obs['fees']}")
        
        # Verify total funds consistency
        total_funds_after = self.calculate_total_funds()
        expected_total_funds = self.initial_total_funds - (fee * 2)
        self.assertAlmostEqual(float(decimal_to_float(total_funds_after, precision=2)),
                            float(decimal_to_float(expected_total_funds, precision=2)),
                            places=2,
                            msg=f"Total funds mismatch after multiple LONG_OPEN. Expected: {expected_total_funds}, Got: {total_funds_after}")


    def test_realized_pnl_after_long_close(self):
        """
        Test that realized P&L is correctly added to user balance after LONG_CLOSE.
        """
        obs, info = self.env.reset()
        total_funds_before = self.calculate_total_funds()
        
        # Execute LONG_OPEN
        action_open = Action.LONG_OPEN.value
        obs, reward_open, terminated, truncated, info = self.env.step(action_open)
        
        # Execute LONG_CLOSE
        action_close = Action.LONG_CLOSE.value
        obs, reward_close, terminated, truncated, info = self.env.step(action_close)
        
        # Calculate expected realized P&L
        open_step = self.env.current_step - 2
        close_step = self.env.current_step - 1
        open_price = Decimal(str(self.env.df.loc[self.env.df.index[open_step], 'Close']))
        close_price = Decimal(str(self.env.df.loc[self.env.df.index[close_step], 'Close']))
        spread = Decimal(str(self.env.spread))
        trading_fees = Decimal(str(self.env.trading_fees))
        trade_lot = Decimal(str(self.env.trade_lot))
        lot_size = Decimal(str(self.env.lot_size))
        leverage = Decimal(str(self.env.leverage))
        
        ask_price = open_price + spread
        bid_price = close_price - spread
        
        # Realized P&L
        pnl = (bid_price - ask_price) * trade_lot * lot_size
        fee_sell = (trade_lot * lot_size * bid_price) * trading_fees
        realized_pnl = pnl - fee_sell
        
        # Expected balance after LONG_CLOSE
        cost = trade_lot * lot_size * ask_price
        fee_buy = cost * trading_fees
        total_cost = cost + fee_buy
        total_revenue = pnl - fee_sell
        expected_balance = Decimal(str(self.env.initial_balance)) - total_cost + total_revenue
        
        # Verify realized P&L
        self.assertAlmostEqual(obs['realized_pnl'], float(decimal_to_float(realized_pnl, precision=2)), places=2,
                            msg=f"Expected realized P&L after LONG_CLOSE: {realized_pnl}, but got {obs['realized_pnl']}")
        
        # Verify user balance has increased by realized P&L
        expected_balance = self.env.user_accounts.balance.get_balance()
        self.assertAlmostEqual(obs['balance'], float(decimal_to_float(expected_balance, precision=2)), places=2,
                            msg=f"Expected user balance after LONG_CLOSE: {expected_balance}, but got {obs['balance']}")
        
        # Verify total funds consistency
        total_funds_after = self.calculate_total_funds()
        expected_total_funds = total_funds_before - fee_buy + realized_pnl  # 手续费已扣除
        self.assertAlmostEqual(float(decimal_to_float(total_funds_after, precision=2)),
                            float(decimal_to_float(expected_total_funds, precision=2)),
                            places=2,
                            msg=f"Total funds mismatch after LONG_CLOSE. Expected: {expected_total_funds}, Got: {total_funds_after}")

    def tearDown(self):
        """
        Clean up after tests.
        """
        self.env.close()

if __name__ == '__main__':
    unittest.main()
