# src/gym_trading_env/envs/trading_env.py

from enum import Enum
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import pandas as pd
import logging
from gym_trading_env.utils.feature_engineering import FeatureEngineer
from gym_trading_env.utils.portfolio_management import Portfolio
from gym_trading_env.rewards.reward_functions import reward_functions  # Import the mapping

class Action(Enum):
    SELL = 0
    HOLD = 1
    BUY = 2

class CustomTradingEnv(gym.Env):
    """
    A custom trading environment for OpenAI Gymnasium.

    Attributes:
        action_space (spaces.Discrete): Action space consisting of Buy, Hold, Sell.
        observation_space (spaces.Box): Observation space including balance, position, and technical indicators.
        df (pd.DataFrame): Historical price data.
        initial_balance (float): Starting balance for the portfolio.
        balance (float): Current balance.
        position (float): Current position (number of assets held).
        portfolio (Portfolio): Portfolio management instance.
        feature_engineer (FeatureEngineer): Feature engineering instance.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, config: dict = None):
        """
        Initializes the trading environment.

        Args:
            df (pd.DataFrame): Historical price data.
            config (dict, optional): Configuration parameters. If None, default configurations are used.
        """
        super(CustomTradingEnv, self).__init__()

        # Configuration management
        if config is None:
            config = {}
        self.initial_balance = config.get('initial_balance', 1000.0)
        self.trading_fees = config.get('trading_fees', 0.001)
        reward_function_name = config.get('reward_function', 'basic_reward_function')
        self.reward_function = reward_functions.get(reward_function_name, reward_functions['basic_reward_function'])
        self.window_size = config.get('window_size', 20)
        self.risk_free_rate = config.get('risk_free_rate', 0.0)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Data and features
        self.df = df.reset_index(drop=True)
        self.feature_engineer = FeatureEngineer(window_size=self.window_size)
        self.features = self.feature_engineer.compute_features(self.df)

        # Action space: dynamically based on Action Enum
        self.action_space = spaces.Discrete(len(Action))

        # Observation space: Dict with named keys
        tech_indicator_size = self.features.shape[1] * self.window_size
        self.observation_space = spaces.Dict({
            'balance': spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
            'position': spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
            'technical_indicators': spaces.Box(low=0, high=np.inf, shape=(tech_indicator_size,), dtype=np.float32)
        })

        # Initialize state
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.

        Args:
            seed (int, optional): Seed for the environment's random number generator.
            options (dict, optional): Additional options for resetting the environment.

        Returns:
            tuple: A tuple containing the initial observation dictionary and info dictionary.
        """
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.position = 0.0
        self.current_step = self.window_size - 1  # Start after window_size to have enough data
        self.portfolio = Portfolio(asset=self.position, fiat=self.balance)
        self.terminated = False
        self.truncated = False
        self.logger.info("Environment reset.")
        return self._get_obs(), {}

    def _get_obs(self):
        """
        Retrieves the current observation as a dictionary.

        Returns:
            dict: The current observation.
        """
        obs = {
            'balance': self.balance,
            'position': self.position,
            'technical_indicators': self.features.iloc[self.current_step - self.window_size:self.current_step].values.flatten().astype(np.float32)
        }
        return obs

    def step(self, action):
        """
        Executes one time step within the environment.

        Args:
            action (int): The action taken by the agent.

        Returns:
            tuple: A tuple containing the next observation dict, reward, terminated flag, truncated flag, and info dictionary.
        """
        if self.terminated or self.truncated:
            return self._get_obs(), 0.0, self.terminated, self.truncated, {}

        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Map action integer to Action Enum
        action_enum = Action(action)

        # Current price
        current_price = self.df.loc[self.current_step, 'Close']

        # Execute action based on Enum
        if action_enum == Action.BUY:
            self._buy(current_price)
            self.logger.info(f"Action: Buy at {current_price}, Position: {self.position}, Balance: {self.balance}")
        elif action_enum == Action.SELL:
            self._sell(current_price)
            self.logger.info(f"Action: Sell at {current_price}, Position: {self.position}, Balance: {self.balance}")
        elif action_enum == Action.HOLD:
            self.logger.info(f"Action: Hold at {current_price}, Position: {self.position}, Balance: {self.balance}")

        # Move to next step
        self.current_step += 1

        # Calculate reward
        previous_total = self.portfolio.valorisation(current_price)
        # Update portfolio valuation with new price
        if self.current_step < len(self.df):
            new_price = self.df.loc[self.current_step, 'Close']
        else:
            new_price = current_price  # If at the end, use current price
        current_total = self.portfolio.valorisation(new_price)
        reward = self.reward_function(previous_total, current_total)

        # Check if done
        if self.current_step >= len(self.df) - 1:
            self.terminated = True  # Episode ends naturally
            self.logger.info("Reached the end of data. Episode terminated.")

        # Get next observation
        obs = self._get_obs()

        # Info dictionary
        info = {'total_asset': current_total}

        return obs, reward, self.terminated, self.truncated, info

    def _buy(self, price):
        """
        Executes a buy action.

        Args:
            price (float): The current price of the asset.
        """
        if self.balance > 0:
            # Calculate amount to buy considering trading fees
            asset_bought = (self.balance / price) * (1 - self.trading_fees)
            cost = asset_bought * price
            fee = cost * self.trading_fees
            total_cost = cost + fee
            self.position += asset_bought
            self.balance -= total_cost
            self.portfolio.update(asset=self.position, fiat=self.balance)

    def _sell(self, price):
        """
        Executes a sell action.

        Args:
            price (float): The current price of the asset.
        """
        if self.position > 0:
            # Calculate amount to sell considering trading fees
            asset_sold = self.position * (1 - self.trading_fees)
            revenue = asset_sold * price
            fee = revenue * self.trading_fees
            total_revenue = revenue - fee
            self.balance += total_revenue
            self.position -= asset_sold
            self.portfolio.update(asset=self.position, fiat=self.balance)

    def render(self, mode='human'):
        """
        Renders the current state of the environment.

        Args:
            mode (str): The mode in which to render.
        """
        if mode == 'human':
            total_asset = self.portfolio.valorisation(self.df.loc[self.current_step, 'Close'])
            print(f'Step: {self.current_step}')
            print(f'Balance: {self.balance:.2f}')
            print(f'Position: {self.position:.4f}')
            print(f'Total Asset: {total_asset:.2f}')

    def close(self):
        """
        Performs any necessary cleanup.
        """
        self.logger.info("Environment closed.")
        pass
