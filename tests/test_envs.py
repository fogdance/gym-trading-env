# tests/test_envs.py

import unittest
import pandas as pd
from gym_trading_env.envs.trading_env import CustomTradingEnv, Action

class TestCustomTradingEnv(unittest.TestCase):

    def setUp(self):
        """
        Set up a dummy environment for testing.
        """
        # Create dummy data for testing
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = {
            'Close': [100 + i for i in range(100)]
        }
        df = pd.DataFrame(data, index=dates)
        # Define a sample configuration
        config = {
            'initial_balance': 1000.0,
            'trading_fees': 0.001,
            'reward_function': 'basic_reward_function',
            'window_size': 20,
            'risk_free_rate': 0.0
        }
        self.env = CustomTradingEnv(df=df, config=config)

    def test_reset(self):
        """
        Test the reset functionality of the environment.
        """
        obs, info = self.env.reset()
        self.assertEqual(obs['balance'], self.env.initial_balance)
        self.assertEqual(obs['position'], 0.0)
        self.assertEqual(info, {})

    def test_step_buy(self):
        """
        Test buying action in the environment.
        """
        obs, info = self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(Action.BUY.value)  # Buy using Enum
        self.assertGreater(obs['position'], 0)
        self.assertLess(obs['balance'], self.env.initial_balance)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn('total_asset', info)

    def test_step_sell(self):
        """
        Test selling action in the environment.
        """
        obs, info = self.env.reset()
        # Buy first
        obs, reward, terminated, truncated, info = self.env.step(Action.BUY.value)  # Buy using Enum
        # Then sell
        obs, reward, terminated, truncated, info = self.env.step(Action.SELL.value)  # Sell using Enum
        # Due to trading fees, position is not exactly 0.0
        self.assertAlmostEqual(obs['position'], 0.008, places=3)
        self.assertGreater(obs['balance'], self.env.initial_balance)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn('total_asset', info)

    def test_step_hold(self):
        """
        Test hold action in the environment.
        """
        obs, info = self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(Action.HOLD.value)  # Hold using Enum
        self.assertEqual(obs['position'], 0.0)
        self.assertEqual(obs['balance'], self.env.initial_balance)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn('total_asset', info)

    def test_done(self):
        """
        Test if the environment correctly identifies the end of the episode.
        """
        obs, info = self.env.reset()
        total_steps = len(self.env.df) - self.env.window_size
        for step in range(total_steps):
            action = Action.HOLD.value  # Hold using Enum
            obs, reward, terminated, truncated, info = self.env.step(action)
            if step == total_steps - 1:
                self.assertTrue(terminated, f"Episode should be terminated at step {step}, but it was not.")
                self.assertFalse(truncated, "Truncated flag should be False when episode is terminated naturally.")
            else:
                self.assertFalse(terminated, f"Episode should not be terminated at step {step}, but it was.")
                self.assertFalse(truncated, f"Truncated flag should be False at step {step}, but it was True.")

    def test_invalid_action(self):
        """
        Test if the environment handles invalid actions gracefully.
        """
        obs, info = self.env.reset()
        with self.assertRaises(ValueError):
            self.env.step(300000)  # Invalid action

    def tearDown(self):
        """
        Clean up after tests.
        """
        self.env.close()

if __name__ == '__main__':
    unittest.main()
