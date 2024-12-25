# src/gym_trading_env/envs/trading_env.py

from decimal import Decimal, getcontext, ROUND_HALF_UP
import logging
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque

from gym_trading_env.utils.feature_engineering import FeatureEngineer
from gym_trading_env.envs.position import Position
from gym_trading_env.envs.user_accounts import UserAccounts
from gym_trading_env.envs.broker_accounts import BrokerAccounts
from gym_trading_env.envs.position_manager import PositionManager
from gym_trading_env.rewards.reward_functions import basic_reward_function, total_pnl_reward_function, reward_functions
from gym_trading_env.utils.conversion import decimal_to_float

# Set global decimal precision
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_UP

class Action(Enum):
    HOLD = 0
    LONG_OPEN = 1
    LONG_CLOSE = 2
    SHORT_OPEN = 3
    SHORT_CLOSE = 4

class CustomTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, config: dict = None):
        super(CustomTradingEnv, self).__init__()

        # Configuration management
        if config is None:
            config = {}
        self.currency_pair = config.get('currency_pair', 'EURUSD')
        self.initial_balance = Decimal(str(config.get('initial_balance', 10000.0)))
        self.broker_accounts = BrokerAccounts()
        self.trading_fees = Decimal(str(config.get('trading_fees', 0.001)))  # 0.1% trading fee
        self.spread = Decimal(str(config.get('spread', 0.0002)))  # Spread in pips
        self.leverage = Decimal(str(config.get('leverage', 100)))  # 1:100 leverage
        self.lot_size = Decimal(str(config.get('lot_size', 100000)))  # Standard lot size
        self.trade_lot = Decimal(str(config.get('trade_lot', 0.01)))  # Default trade size: 0.01 lot
        self.max_long_position = Decimal(str(config.get('max_long_position', 0.1)))  # Max long position: 0.1 lot
        self.max_short_position = Decimal(str(config.get('max_short_position', 0.1)))  # Max short position: 0.1 lot
        reward_function_name = config.get('reward_function', 'basic_reward_function')
        self.reward_function = reward_functions.get(reward_function_name, reward_functions['basic_reward_function'])
        self.reward_function = total_pnl_reward_function  # Assign the new reward function
        self.window_size = config.get('window_size', 20)
        self.risk_free_rate = Decimal(str(config.get('risk_free_rate', 0.0)))

        # Whether to use a dict observation space or flatten everything into a 1D array
        self.use_dict_obs = config.get('use_dict_obs', False)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.ERROR)

        # Data and features
        self.df = df.reset_index(drop=True)
        self.feature_engineer = FeatureEngineer(window_size=self.window_size)
        self.features = self.feature_engineer.compute_features(self.df)

        # Action space: dynamically based on Action Enum
        self.action_space = spaces.Discrete(len(Action))

        # Prepare shapes
        self.tech_indicator_size = self.features.shape[1] * self.window_size

        # If using Dict obs or 1D obs
        if self.use_dict_obs:
            # Dict observation space
            self.observation_space = spaces.Dict({
                'balance': spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                'equity': spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                'used_margin': spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                'free_margin': spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                'long_position': spaces.Box(low=0.0, high=float(self.max_long_position), shape=(1,), dtype=np.float32),
                'short_position': spaces.Box(low=0.0, high=float(self.max_short_position), shape=(1,), dtype=np.float32),
                'unrealized_pnl': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                'realized_pnl': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                'fees': spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                'technical_indicators': spaces.Box(low=-np.inf, high=np.inf, shape=(self.tech_indicator_size,), dtype=np.float32)
            })
        else:
            # Flattened 1D observation space
            # We have 8 scalar features + self.tech_indicator_size
            # The 8 scalars are:
            #   [balance, equity, used_margin, free_margin, long_position, short_position, unrealized_pnl, realized_pnl]
            # Then we append the technical indicators
            obs_size = 8 + self.tech_indicator_size
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
            )

        # Initialize state
        self.position_manager = PositionManager()
        self.user_accounts = UserAccounts(initial_balance=self.initial_balance, position_manager=self.position_manager)

        # Other state variables
        self.current_step = 0
        self.terminated = False

        # Reset previous total P&L
        self.previous_total_pnl = Decimal('0.0')

        # Initialize previous balance and price for reward calculation
        self.previous_balance = self.user_accounts.balance.get_balance()
        self.previous_price = Decimal(str(self.df.loc[self.current_step, 'Close']))

        self.reset()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns an initial observation.
        """
        super().reset(seed=seed)
        # Reset positions
        self.position_manager = PositionManager()
        # Reset user accounts
        self.user_accounts = UserAccounts(initial_balance=self.initial_balance, position_manager=self.position_manager)
        # Reset broker accounts
        self.broker_accounts = BrokerAccounts()
        # Reset step
        self.current_step = self.window_size
        self.terminated = False
        # Reset previous total P&L
        self.previous_total_pnl = Decimal('0.0')
        # Reset previous balance and price
        self.previous_balance = self.initial_balance
        self.previous_price = Decimal(str(self.df.loc[self.current_step, 'Close']))
        return self._get_obs(), {}

    def step(self, action):
        """
        Executes one time step within the environment.

        Args:
            action (int): The action to take.

        Returns:
            Tuple: observation, reward, terminated, truncated, info
        """
        if self.terminated:
            return self._get_obs(), 0.0, self.terminated, False, {}

        # Get current price
        self.current_price = Decimal(str(self.df.loc[self.current_step, 'Close']))

        # Execute action
        try:
            action = Action(action)
        except ValueError:
            self.logger.error(f"Invalid action: {action}")
            raise

        if action == Action.HOLD:
            pass  # Do nothing
        elif action == Action.LONG_OPEN:
            self._long_open(self.current_price + self.spread)
        elif action == Action.LONG_CLOSE:
            self._long_close(self.current_price - self.spread)
        elif action == Action.SHORT_OPEN:
            self._short_open(self.current_price - self.spread)
        elif action == Action.SHORT_CLOSE:
            self._short_close(self.current_price + self.spread)

        # Calculate reward
        reward = self.reward_function(self)  # Already a float

        # Check termination conditions (e.g., last time step)
        if self.current_step >= len(self.df) - 1:
            self.terminated = True
            self.logger.info(f"Episode terminated. current_step: {self.current_step}, df_len: {len(self.df)}")

        if self.current_step + 1 >= len(self.features):
            # End immediately so that the next step will not be accessed again
            self.terminated = True
            self.logger.info(f"Episode terminated. current_step: {self.current_step}, features_len: {len(self.features)}")

        # Update step
        self.current_step += 1

        # Calculate new equity
        equity = self._calculate_equity()

        # Update free margin
        free_margin = equity - self.user_accounts.margin.get_balance()

        # Update unrealized P&L
        self._update_unrealized_pnl()

        # Check margin requirements
        self._check_margin(equity)

        # Construct observation
        obs = self._get_obs()

        # Include total_asset in info
        info = {
            'total_asset': float(decimal_to_float(equity, precision=2)),
            'realized_pnl': float(decimal_to_float(self.user_accounts.realized_pnl, precision=2)),
            'unrealized_pnl': float(decimal_to_float(self.user_accounts.unrealized_pnl, precision=2)),
            'fees_collected': float(decimal_to_float(self.broker_accounts.fees.get_balance(), precision=2))
        }

        # Return the observation, reward (float), termination flags, and info
        return obs, reward, self.terminated, False, info

    def _calculate_equity(self):
        """
        Calculates the current equity.

        Returns:
            Decimal: The current equity.
        """
        equity = self.user_accounts.balance.get_balance() + self.user_accounts.realized_pnl + self.user_accounts.unrealized_pnl
        return equity

    def _update_unrealized_pnl(self):
        """
        Updates the user's unrealized P&L based on current prices.
        """
        # Calculate unrealized P&L for long positions
        unrealized_pnl_long = sum(
            (self.current_price - pos.entry_price) * pos.size * self.lot_size
            for pos in self.position_manager.long_positions
        )
        # Calculate unrealized P&L for short positions
        unrealized_pnl_short = sum(
            (pos.entry_price - self.current_price) * pos.size * self.lot_size
            for pos in self.position_manager.short_positions
        )
        # Update user's unrealized P&L
        self.user_accounts.unrealized_pnl = unrealized_pnl_long + unrealized_pnl_short

    def _check_margin(self, equity: Decimal):
        """
        Checks margin requirements and performs liquidation if necessary.
        """
        if equity < self.user_accounts.margin.get_balance():
            # Liquidate all positions
            self.logger.info("Equity below margin requirement. Liquidating all positions.")
            while self.user_accounts.long_position > Decimal('0.0'):
                self._long_close(self.df.loc[self.current_step, 'Close'] - self.spread)
            while self.user_accounts.short_position > Decimal('0.0'):
                self._short_close(self.df.loc[self.current_step, 'Close'] + self.spread)
            self.terminated = True
            self.logger.info("Margin requirement not met. Episode terminated.")

    def _long_open(self, ask_price: Decimal):
        """
        Executes a LONG_OPEN action.

        Args:
            ask_price (Decimal): The ask price at which the long position is opened.
        """
        max_additional_long = self.max_long_position - self.user_accounts.long_position
        if max_additional_long <= Decimal('0.0'):
            self.logger.warning("Reached maximum long position limit.")
            return

        position_size = min(self.trade_lot, max_additional_long)

        # Calculate required margin
        required_margin = (position_size * self.lot_size * ask_price) / self.leverage

        # Check if user has sufficient free margin
        free_margin = self._calculate_equity() - self.user_accounts.margin.get_balance()
        if required_margin > free_margin:
            self.logger.warning("Insufficient free margin to execute LONG_OPEN.")
            return

        # Calculate cost (including fees)
        cost = position_size * self.lot_size * ask_price
        fee = cost * self.trading_fees
        total_cost = cost + fee

        # Deduct cost and fees from user balance
        try:
            self.user_accounts.balance.withdraw(total_cost)
        except ValueError:
            self.logger.warning("Insufficient balance to execute LONG_OPEN.")
            return

        # Allocate margin
        self.user_accounts.allocate_margin(required_margin)

        # Collect fees
        self.broker_accounts.collect_fee(fee)

        # Create and add new position
        new_position = Position(size=position_size, entry_price=ask_price)
        self.position_manager.add_long_position(new_position)

        self.logger.debug(f"Opened LONG position: {new_position}")
        self.logger.debug(f"New balance: {self.user_accounts.balance.get_balance()}, "
                          f"Long position: {self.user_accounts.long_position}, "
                          f"Used margin: {self.user_accounts.margin.get_balance()}")

    def _long_close(self, bid_price: Decimal):
        """
        Executes a LONG_CLOSE action.

        Args:
            bid_price (Decimal): The bid price at which the long position is closed.
        """
        position_size = min(self.trade_lot, self.user_accounts.long_position)
        if position_size <= Decimal('0.0'):
            self.logger.warning("No long position to close.")
            return

        # Close position and calculate realized P&L
        pnl = self.position_manager.close_long_position(position_size, closing_price=bid_price)

        # Calculate fees
        fee = (position_size * self.lot_size * bid_price) * self.trading_fees

        # Deduct fees from broker
        self.broker_accounts.collect_fee(fee)

        # Realize P&L (add to realized P&L account)
        self.user_accounts.realize_pnl(pnl - fee)

        # Release margin back to user balance
        entry_price = bid_price - self.spread  # Assuming entry price was bid_price - spread
        released_margin = (position_size * self.lot_size * entry_price) / self.leverage
        self.user_accounts.release_margin(released_margin)

        self.logger.debug(f"Closed LONG position of size {position_size} at price {bid_price}")
        self.logger.debug(f"P&L: {pnl - fee}, New balance: {self.user_accounts.balance.get_balance()}, "
                          f"Long position: {self.user_accounts.long_position}, "
                          f"Used margin: {self.user_accounts.margin.get_balance()}")

    def _short_open(self, bid_price: Decimal):
        """
        Executes a SHORT_OPEN action.

        Args:
            bid_price (Decimal): The bid price at which the short position is opened.
        """
        max_additional_short = self.max_short_position - self.user_accounts.short_position
        if max_additional_short <= Decimal('0.0'):
            self.logger.warning("Reached maximum short position limit.")
            return

        position_size = min(self.trade_lot, max_additional_short)

        # Calculate required margin
        required_margin = (position_size * self.lot_size * bid_price) / self.leverage

        # Check if user has sufficient free margin
        free_margin = self._calculate_equity() - self.user_accounts.margin.get_balance()
        if required_margin > free_margin:
            self.logger.warning("Insufficient free margin to execute SHORT_OPEN.")
            return

        # Calculate revenue and fees
        revenue = position_size * self.lot_size * bid_price
        fee = revenue * self.trading_fees
        total_revenue = revenue - fee

        # Add revenue to user balance
        self.user_accounts.balance.deposit(total_revenue)

        # Allocate margin
        self.user_accounts.allocate_margin(required_margin)

        # Collect fees
        self.broker_accounts.collect_fee(fee)

        # Create and add new position
        new_position = Position(size=position_size, entry_price=bid_price)
        self.position_manager.add_short_position(new_position)

        self.logger.debug(f"Opened SHORT position: {new_position}")
        self.logger.debug(f"New balance: {self.user_accounts.balance.get_balance()}, "
                          f"Short position: {self.user_accounts.short_position}, "
                          f"Used margin: {self.user_accounts.margin.get_balance()}")

    def _short_close(self, ask_price: Decimal):
        """
        Executes a SHORT_CLOSE action.

        Args:
            ask_price (Decimal): The ask price at which the short position is closed.
        """
        position_size = min(self.trade_lot, self.user_accounts.short_position)
        if position_size <= Decimal('0.0'):
            self.logger.warning("No short position to close.")
            return

        # Close position and calculate realized P&L
        pnl = self.position_manager.close_short_position(position_size, closing_price=ask_price)

        # Calculate fees
        fee = (position_size * self.lot_size * ask_price) * self.trading_fees

        # Deduct fees from broker
        self.broker_accounts.collect_fee(fee)

        # Realize P&L (add to realized P&L account)
        self.user_accounts.realize_pnl(pnl - fee)

        # Release margin back to user balance
        entry_price = ask_price + self.spread  # Assuming entry price was ask_price + spread
        released_margin = (position_size * self.lot_size * entry_price) / self.leverage
        self.user_accounts.release_margin(released_margin)

        self.logger.debug(f"Closed SHORT position of size {position_size} at price {ask_price}")
        self.logger.debug(f"P&L: {pnl - fee}, New balance: {self.user_accounts.balance.get_balance()}, "
                          f"Short position: {self.user_accounts.short_position}, "
                          f"Used margin: {self.user_accounts.margin.get_balance()}")

    def _get_obs(self):
        """
        Constructs the observation, either as a Dict or a flattened 1D array.
        """
        equity = self._calculate_equity()
        free_margin = equity - self.user_accounts.margin.get_balance()

        balance_val = float(decimal_to_float(self.user_accounts.balance.get_balance(), precision=2))
        equity_val = float(decimal_to_float(equity, precision=2))
        used_margin_val = float(decimal_to_float(self.user_accounts.margin.get_balance(), precision=2))
        free_margin_val = float(decimal_to_float(free_margin, precision=2))
        long_pos_val = float(self.user_accounts.long_position)
        short_pos_val = float(self.user_accounts.short_position)
        unrealized_pnl_val = float(decimal_to_float(self.user_accounts.unrealized_pnl, precision=2))
        realized_pnl_val = float(decimal_to_float(self.user_accounts.realized_pnl, precision=2))

        tech_data = self.features.iloc[self.current_step - self.window_size : self.current_step].values.flatten()
        tech_data = tech_data.astype(np.float32)

        if self.use_dict_obs:
            # Return as Dict
            obs = {
                'balance': np.array([balance_val], dtype=np.float32),
                'equity': np.array([equity_val], dtype=np.float32),
                'used_margin': np.array([used_margin_val], dtype=np.float32),
                'free_margin': np.array([free_margin_val], dtype=np.float32),
                'long_position': np.array([long_pos_val], dtype=np.float32),
                'short_position': np.array([short_pos_val], dtype=np.float32),
                'unrealized_pnl': np.array([unrealized_pnl_val], dtype=np.float32),
                'realized_pnl': np.array([realized_pnl_val], dtype=np.float32),
                'fees': np.array([float(decimal_to_float(self.broker_accounts.fees.get_balance(), precision=2))], dtype=np.float32),
                'technical_indicators': tech_data
            }
            return obs
        else:
            # Return as flattened 1D array
            # We'll stack:
            # [balance, equity, used_margin, free_margin, long_position, short_position, unrealized_pnl, realized_pnl]
            # + technical_indicators
            scalar_part = np.array([
                balance_val,
                equity_val,
                used_margin_val,
                free_margin_val,
                long_pos_val,
                short_pos_val,
                unrealized_pnl_val,
                realized_pnl_val
            ], dtype=np.float32)

            obs = np.concatenate([scalar_part, tech_data], axis=0)
            return obs

    def render(self, mode=None):
        """
        Renders the current state of the environment.
        """
        if mode == 'human':
            equity = self._calculate_equity()
            free_margin = equity - self.user_accounts.margin.get_balance()
            total_asset = float(decimal_to_float(equity, precision=2))
            realized_pnl = float(decimal_to_float(self.user_accounts.realized_pnl, precision=2))
            unrealized_pnl = float(decimal_to_float(self.user_accounts.unrealized_pnl, precision=2))
            fees_collected = float(decimal_to_float(self.broker_accounts.fees.get_balance(), precision=2))

            print(f'Step: {self.current_step}')
            print(f'Currency Pair: {self.currency_pair}')
            print(f'Balance: {self.user_accounts.balance.get_balance():.2f}')
            print(f'Equity: {equity:.2f}')
            print(f'Used Margin: {self.user_accounts.margin.get_balance():.2f}')
            print(f'Free Margin: {free_margin:.2f}')
            print(f'Long Position: {self.user_accounts.long_position:.4f} lots')
            print(f'Short Position: {self.user_accounts.short_position:.4f} lots')
            print(f'Realized P&L: {realized_pnl:.2f}')
            print(f'Unrealized P&L: {unrealized_pnl:.2f}')
            print(f'Fees Collected: {fees_collected:.2f}')
            print(f'Total Asset: {total_asset:.2f}')
            print(f'Long Positions: {list(self.position_manager.long_positions)}')
            print(f'Short Positions: {list(self.position_manager.short_positions)}')

    def close(self):
        """
        Performs any necessary cleanup.
        """
        self.logger.info("Environment closed.")
        pass
