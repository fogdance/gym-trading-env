# tests/test_reward.py

import unittest
import pandas as pd
from decimal import Decimal, getcontext, ROUND_HALF_UP
import numpy as np


from gym_trading_env.envs.trading_env import CustomTradingEnv, Action
from stable_baselines3.common.env_checker import check_env
from gym_trading_env.utils.conversion import decimal_to_float
from unittest.mock import MagicMock
from unittest.mock import patch
from gym_trading_env.rewards.reward_functions import fast_car_racing_likely_reward_function

# Set global decimal precision for testing
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_UP


class TestRewardFunction(unittest.TestCase):

    def setUp(self):
        """
        Set up a dummy environment with the 'fast_car_racing_likely_reward_function'.
        """
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        data = {
            'Date': dates,
            # We'll keep the close price small so we can easily manipulate PnL
            'Close': [Decimal('1.1000') + Decimal('0.0001') * Decimal(str(i)) for i in range(5)]
        }
        df = pd.DataFrame(data)

        config = {
            'initial_balance': 10000.0,
            'trading_fees': 0.001,
            'spread': 0.0002,
            'leverage': 100,
            'lot_size': 100000,
            'trade_lot': 0.01,
            'max_long_position': 0.02,
            'max_short_position': 0.02,
            'reward_function': 'fast_car_racing_likely_reward_function',
            'window_size': 1,       # minimal
            'is_unittest': True,
        }

        self.env = CustomTradingEnv(df=df, config=config)
        check_env(self.env, warn=True)
        self.env.reset()

    def tearDown(self):
        self.env.close()

    def calculate_total_funds(self):
        """
        Helper to track total funds across user/broker accounts.
        """
        info = self.env._get_info()
        total_funds = (
            Decimal(str(info['balance'])) +
            Decimal(str(info['used_margin'])) +
            Decimal(str(info['broker_balance'])) +
            Decimal(str(info['fees_collected']))
        )
        return total_funds

    def test_delta_pnl_scaling(self):
        """
        Test that step-wise delta PnL is scaled by 0.0001, 
        and added to the reward in the step.
        """
        # Reset environment
        obs, info = self.env.reset()

        # Suppose we forcibly set a small unrealized PnL 
        # so we can see if the reward is scaled
        self.env.previous_total_pnl = Decimal('0.0')
        self.env.user_accounts.unrealized_pnl = Decimal('200.0')  # big float profit

        # Step with HOLD => no big changes except the reward from delta PnL
        # delta_pnl = 200.0 / (10000+200) * 100 = 1.96
        reward = fast_car_racing_likely_reward_function(self.env)

        self.assertAlmostEqual(
            reward, 
            1.96,
            places=2,
            msg=f"Expected scaled delta PnL reward of 1.96, got {reward}"
        )

    def test_close_reward_positive(self):
        """
        If we close a trade with positive PnL, we should get +1 
        in addition to the scaled delta PnL.
        """
        obs, info = self.env.reset()
        
        # Step 1: LONG_OPEN => no immediate close, so no "just_closed_trade"
        self.env.user_accounts.unrealized_pnl = Decimal('0.0')
        obs, reward_open, done, truncated, info = self.env.step(Action.LONG_OPEN.value)
        self.assertNotEqual(reward_open, 0.0, "Should get some scaled PnL delta, might be small negative/positive.")
        
        # Step 2: artificially set just_closed_trade = +50 
        # to simulate we closed with +50 PnL
        # We'll call LONG_CLOSE, and inside _long_close env sets self.just_closed_trade = actual PnL
        # But let's do a direct approach for test
        self.env.just_closed_trade = Decimal('50.0')
        
        # Step with HOLD => no open/close action, but "just_closed_trade" won't typically 
        # be set if we do HOLD. Let's pretend we intercept the logic. 
        # Alternatively, we do a real LONG_CLOSE scenario:
        obs, reward_close, done, truncated, info = self.env.step(Action.HOLD.value)
        
        # The scaled delta PnL in this step is presumably 0, 
        # but close_reward = +1 since closed_pnl > 0
        # total => +1
        self.assertAlmostEqual(
            reward_close,
            1.0,
            places=2,
            msg=f"Expected reward of +1 for profitable close, got {reward_close}"
        )
        self.assertFalse(done)

    def test_close_reward_negative(self):
        """
        If we close a trade with negative PnL, we should get -0.5 
        in addition to the scaled delta PnL.
        """
        obs, info = self.env.reset()

        # Step 1: SHORT_OPEN => no immediate close
        obs, reward_open, done, truncated, info = self.env.step(Action.SHORT_OPEN.value)
        
        # Step 2: artificially set just_closed_trade = -20 
        self.env.just_closed_trade = Decimal('-20.0')

        obs, reward_close, done, truncated, info = self.env.step(Action.HOLD.value)

        # The scaled delta PnL might be 0, but close_reward = -0.5
        self.assertAlmostEqual(
            reward_close,
            -0.5,
            places=2,
            msg=f"Expected reward of -0.5 for losing close, got {reward_close}"
        )
        self.assertFalse(done)

    def test_final_bonus_natural_end(self):
        """
        Test we get final stats-based bonus if the episode ends 'naturally' 
        (not forced). Suppose we set a 100% win rate => bonus=5.0
        """
        obs, info = self.env.reset()

        # Suppose the environment tracks forced_termination = False by default
        # We'll let the data run out or artificially do so

        # We'll mock the trade_record_manager to say total=2 trades, all winning
        with patch.object(self.env.trade_record_manager, 'get_total_trades', return_value=2), \
             patch.object(self.env.trade_record_manager, 'get_winning_trades_count', return_value=2):
            

            self.env.current_step = 1
            self.env.end_idx =1
            obs, reward, done, truncated, info = self.env.step(Action.HOLD.value)

            
            self.assertTrue(done, "Expected environment to finish after data exhausted.")
            # The last step reward should contain final bonus => +5.0
            self.assertGreaterEqual(reward, 5.0, "Expected final bonus at least +5.0 for 100% win rate.")
            self.assertFalse(getattr(self.env, "forced_termination", False), 
                            "Should be a natural termination, forced_termination = False.")
    
    def test_final_bonus_forced_end(self):
        """
        Test that if forced_termination is True, the final bonus is 0 
        (or no bonus).
        """
        obs, info = self.env.reset()

        # 1) Set environment's max_drawdown_ratio = 0.3
        self.env.max_drawdown_ratio = Decimal('0.3')

        # 2) Next, force a big negative PnL so that equity < 7000 => 30% drawdown from 10000
        # Could do this by big LONG_OPEN then drastically lowering price
        self.env.user_accounts.balance.balance = Decimal('10000.0')
        # big open
        action_open = Action.LONG_OPEN.value
        obs, reward_open, terminated, truncated, info = self.env.step(action_open)
        self.assertFalse(terminated, "Should not terminate yet.")

        # artificially drop price to create large float loss
        # e.g. self.env.df.iloc[self.env.current_step]['Close'] = something
        # or directly manipulate user_accounts.unrealized_pnl
        self.env.user_accounts.unrealized_pnl = Decimal('-4000')  # 40% drawdown

        # 3) Next step => triggers _check_drawdown
        obs, reward, terminated, truncated, info = self.env.step(Action.HOLD.value)

        self.assertTrue(terminated, "Should end immediately.")
        # Because forced_termination => final bonus=0
        self.assertLessEqual(reward, 0.0, "Expected reward <= 0 if forced termination. Possibly negative penalty?")

        self.assertTrue(getattr(self.env, "forced_termination", False))


if __name__ == '__main__':
    unittest.main()
