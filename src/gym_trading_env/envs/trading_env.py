# src/gym_trading_env/envs/trading_env.py

from decimal import Decimal, getcontext, ROUND_HALF_UP
import logging
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque
from typing import Tuple
import os

from gym_trading_env.utils.feature_engineering import FeatureEngineer
from gym_trading_env.envs.position import Position
from gym_trading_env.envs.user_accounts import UserAccounts
from gym_trading_env.envs.broker_accounts import BrokerAccounts
from gym_trading_env.envs.position_manager import PositionManager
from gym_trading_env.rewards.reward_functions import total_pnl_reward_function, reward_functions
from gym_trading_env.utils.conversion import decimal_to_float, float_to_decimal
from gym_trading_env.rendering.plotting import BollingerBandPlotter  # Import plotting utility
from gym_trading_env.envs.trade_record import TradeRecord
from gym_trading_env.envs.trade_record_manager import TradeRecordManager
from gym_trading_env.envs.action import Action

# Set global decimal precision
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_UP


class CustomTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, config: dict = None):
        super(CustomTradingEnv, self).__init__()

        # Configuration management
        if config is None:
            config = {}
        self.currency_pair = config.get('currency_pair', 'EURUSD')
        self.initial_balance = Decimal(str(config.get('initial_balance', 10000.0)))
        self.broker_accounts = BrokerAccounts()  # Initialize broker accounts with balance and fees
        self.trading_fees = Decimal(str(config.get('trading_fees', 0.001)))  # 0.1% trading fee
        self.spread = Decimal(str(config.get('spread', 0.0002)))  # Spread in pips
        self.leverage = Decimal(str(config.get('leverage', 100)))  # 1:100 leverage
        self.lot_size = Decimal(str(config.get('lot_size', 100000)))  # Standard lot size
        self.trade_lot = Decimal(str(config.get('trade_lot', 0.01)))  # Default trade size: 0.01 lot
        self.max_long_position = Decimal(str(config.get('max_long_position', 0.1)))  # Max long position: 0.1 lot
        self.max_short_position = Decimal(str(config.get('max_short_position', 0.1)))  # Max short position: 0.1 lot
        reward_function_name = config.get('reward_function', 'total_pnl_reward_function')
        self.reward_function = reward_functions.get(reward_function_name, total_pnl_reward_function)
        self.window_size = config.get('window_size', 20)
        self.risk_free_rate = Decimal(str(config.get('risk_free_rate', 0.0)))
        self.is_unittest = config.get('is_unittest', False)
        self.debug_enabled = True
        self.trade_record_manager = TradeRecordManager()

        # Set up logging
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.ERROR)

        # Data
        self.df = df.copy()
        # Ensure 'Date' is datetime and set as index
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df.set_index('Date', inplace=True)
        elif not isinstance(self.df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame must have a 'Date' column or a DatetimeIndex.")


        # Action space: dynamically based on Action Enum
        self.action_space = spaces.Discrete(len(Action))

        # Define image dimensions
        self.image_height = config.get('image_height', 300)
        self.image_width = config.get('image_width', 400)
        self.channels = 3  # RGB

        # Update observation space to image
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(self.image_height, self.image_width, self.channels), 
            dtype=np.uint8
        )

        # Initialize state
        self.position_manager = PositionManager()
        self.user_accounts = UserAccounts(initial_balance=self.initial_balance, position_manager=self.position_manager)

        # Other state variables
        self.current_step = self.window_size
        self.terminated = False

        # Reset previous total P&L
        self.previous_total_pnl = Decimal('0.0')
        self.previous_equity = Decimal(self.initial_balance)

        self.reset()

    def record_trade(self, trade_record: TradeRecord):
        """
        Record a trade in the trade record manager.

        Args:
            trade_record (TradeRecord): The trade record to record.
        """
        self.trade_record_manager.record_trade(trade_record)

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns an initial observation.

        Args:
            seed (int, optional): Seed for the environment's random number generator.
            options (dict, optional): Additional options for resetting the environment.

        Returns:
            Tuple: (observation, info)
        """
        super().reset(seed=seed)
        self.trade_record_manager = TradeRecordManager()
        # Reset positions
        self.position_manager = PositionManager()
        # Reset user accounts
        self.user_accounts = UserAccounts(initial_balance=self.initial_balance, position_manager=self.position_manager)
        # Reset broker accounts
        self.broker_accounts = BrokerAccounts()  # Re-initialize broker accounts with both balance and fees
        # Reset step
        self.current_step = self.window_size
        self.terminated = False
        # Reset previous total P&L
        self.previous_total_pnl = Decimal('0.0')
        self.previous_equity = Decimal(self.initial_balance)
        return self._get_obs(), self._get_info(self.initial_balance)

    def step(self, action):
        """
        Executes one time step within the environment.

        Args:
            action (int): The action to take.

        Returns:
            Tuple: (observation, reward, terminated, truncated, info)
        """
        if self.terminated:
            return self._get_obs(), 0.0, self.terminated, False, {}

        current_kline = self.df.iloc[self.current_step]
        timestamp = current_kline.name

        # Get current price
        try:
            self.current_price = Decimal(str(current_kline['Close']))
        except IndexError:
            self.logger.error(f"Current step {self.current_step} is out of bounds for DataFrame with length {len(self.df)}.")
            self.terminated = True
            return self._get_obs(), 0.0, self.terminated, False, {}

        # Execute action
        try:
            action_enum = Action(action)
        except ValueError:
            self.logger.error(f"Invalid action: {action}. Action must be one of {list(Action)}.")
            self.terminated = True
            return self._get_obs(), 0.0, self.terminated, False, {}

        if action_enum == Action.HOLD:
            pass  # Do nothing
        elif action_enum == Action.LONG_OPEN:
            self._long_open(timestamp, self.current_price + self.spread)
        elif action_enum == Action.LONG_CLOSE:
            self._long_close(timestamp, self.current_price - self.spread)
        elif action_enum == Action.SHORT_OPEN:
            self._short_open(timestamp, self.current_price - self.spread)
        elif action_enum == Action.SHORT_CLOSE:
            self._short_close(timestamp, self.current_price + self.spread)

        # Check termination conditions (e.g., last time step)
        if self.current_step >= len(self.df) - 1:
            self.terminated = True
            self.logger.info(f"Episode terminated. current_step: {self.current_step}, df_len: {len(self.df)}")

        # Update step
        self.current_step += 1

        # Calculate new equity
        equity = self._calculate_equity()

        # Update unrealized P&L
        self._update_unrealized_pnl()

        # Calculate reward
        reward = self.reward_function(self)  # Already a float

        # Check margin requirements
        self._check_margin(equity)

        # Construct observation
        obs = self._get_obs()

        # Update info
        info = self._get_info(equity)

        if self.terminated and self.debug_enabled:
            self.trade_record_manager.dump_to_json(f"output/trade_records_{self.current_step}.json")

        # Return the observation, reward (float), termination flags, and info
        return obs, reward, self.terminated, False, info

    def _get_info(self, equity: Decimal):
        """
        Retrieve information about the current state.

        Args:
            equity (Decimal): Current equity.

        Returns:
            dict: Information dictionary containing various account details.
        """
        info = {
            'total_asset': equity,
            'realized_pnl': self.user_accounts.realized_pnl,
            'unrealized_pnl': self.user_accounts.unrealized_pnl,
            'fees_collected': self.broker_accounts.fees.get_balance(),
            'broker_balance': self.broker_accounts.balance.get_balance(),  # Added broker balance
            'balance': self.user_accounts.balance.get_balance(),
            'equity': self._calculate_equity(),
            'used_margin': self.user_accounts.margin.get_balance(),
            'free_margin': self._calculate_equity() - self.user_accounts.margin.get_balance(),
            'long_position': self.user_accounts.long_position,
            'short_position': self.user_accounts.short_position,
        }
        return info

    def _calculate_equity(self) -> Decimal:
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
            (
                (self.current_price - pos.entry_price) * pos.size * self.lot_size
                for pos in self.position_manager.long_positions
            ),
            Decimal('0.0')  # Specify Decimal start value
        )
        
        # Calculate unrealized P&L for short positions
        unrealized_pnl_short = sum(
            (
                (pos.entry_price - self.current_price) * pos.size * self.lot_size
                for pos in self.position_manager.short_positions
            ),
            Decimal('0.0')  # Specify Decimal start value
        )
        
        assert isinstance(unrealized_pnl_long, Decimal), "unrealized_pnl_long must be a Decimal."
        assert isinstance(unrealized_pnl_short, Decimal), "unrealized_pnl_short must be a Decimal."

        # Update user's unrealized P&L
        self.user_accounts.unrealized_pnl = unrealized_pnl_long + unrealized_pnl_short

    def _check_margin(self, equity: Decimal):
        """
        Checks margin requirements and performs liquidation if necessary.

        Args:
            equity (Decimal): The current equity.
        """
        if equity < self.user_accounts.margin.get_balance():
            # Liquidate all positions
            self.logger.info("Equity below margin requirement. Liquidating all positions.")
            while self.user_accounts.long_position > Decimal('0.0'):
                self._long_close(self.current_price - self.spread)
            while self.user_accounts.short_position > Decimal('0.0'):
                self._short_close(self.current_price + self.spread)
            self.terminated = True
            self.logger.info("Margin requirement not met. Episode terminated.")

    def _long_open(self, timestamp, ask_price: Decimal):
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

        # Calculate trading fee based on notional value
        notional_value = position_size * self.lot_size * ask_price
        fee = notional_value * self.trading_fees

        # Total deduction from balance: required_margin + fee
        total_deduction = required_margin + fee

        # Check if user has sufficient free margin
        free_margin = self._calculate_equity() - self.user_accounts.margin.get_balance()
        if total_deduction > free_margin:
            self.logger.warning("Insufficient free margin to execute LONG_OPEN.")
            return

        # Deduct only the fee from user balance
        try:
            self.user_accounts.balance.withdraw(fee)
        except ValueError:
            self.logger.warning("Insufficient balance to execute LONG_OPEN.")
            return

        # Deduct the required margin from user balance and allocate to margin account
        try:
            self.user_accounts.allocate_margin(required_margin)
        except ValueError as e:
            self.logger.warning(f"Failed to allocate margin: {e}")
            return

        # Collect fees to broker's fees account
        self.broker_accounts.collect_fee(fee)

        # Create and add new position with initial_margin
        new_position = Position(size=position_size, entry_price=ask_price, initial_margin=required_margin)
        self.position_manager.add_long_position(new_position)

        # Record trade
        trade_record = TradeRecord(
            timestamp=timestamp,
            operation_type=Action.LONG_OPEN.name,
            position_size=position_size,
            price=ask_price,
            required_margin=required_margin,
            fee=fee,
            balance=self.user_accounts.balance.get_balance(),
            leverage=self.leverage,
            free_margin=free_margin
        )
        self.record_trade(trade_record)

        self.logger.debug(f"Opened LONG position: {new_position}")
        self.logger.debug(f"New balance: {self.user_accounts.balance.get_balance()}, "
                          f"Long position: {self.user_accounts.long_position}, "
                          f"Used margin: {self.user_accounts.margin.get_balance()}")

    def _long_close(self, timestamp, bid_price: Decimal):
        """
        Executes a LONG_CLOSE action.

        Args:
            bid_price (Decimal): The bid price at which the long position is closed.
        """
        if self.user_accounts.long_position <= Decimal('0.0'):
            self.logger.warning("No long position to close.")
            return

        # Close the earliest long position
        try:
            pnl, released_margin, closed_size = self.position_manager.close_long_position(bid_price, self.lot_size)
        except ValueError as e:
            self.logger.error(f"Error closing long position: {e}")
            self.terminated = True
            return

        # Calculate fees based on closed size
        fee = (closed_size * self.lot_size * bid_price) * self.trading_fees

        # Deduct fees from user balance
        try:
            self.user_accounts.balance.withdraw(fee)
        except ValueError:
            self.logger.warning("Insufficient balance to pay fees on LONG_CLOSE.")
            self.terminated = True
            return

        # Collect fees to broker's fees account
        self.broker_accounts.collect_fee(fee)

        # Realize P&L (add to realized P&L account)
        self.user_accounts.realize_pnl(pnl)

        # Adjust broker's balance based on user's P&L to maintain funds conservation
        self.broker_accounts.adjust_balance(-pnl)

        # Release initial margin back to user balance
        try:
            self.user_accounts.release_margin(released_margin)
        except ValueError as e:
            self.logger.error(f"Error releasing margin: {e}")
            self.terminated = True
            return
        
        trade_record = TradeRecord(
            timestamp=timestamp,
            operation_type=Action.LONG_CLOSE.name,
            position_size=closed_size,
            price=bid_price,
            required_margin=0,
            fee=fee,
            balance=self.user_accounts.balance.get_balance(),
            leverage=self.leverage,
            free_margin=self._calculate_equity() - self.user_accounts.margin.get_balance(),
            pnl=pnl,
            closed_size=closed_size,
            released_margin=released_margin
        )
        self.record_trade(trade_record)

        self.logger.debug(f"Closed LONG position at price {bid_price}")
        self.logger.debug(f"P&L: {pnl}, New balance: {self.user_accounts.balance.get_balance()}, "
                          f"Long position: {self.user_accounts.long_position}, "
                          f"Used margin: {self.user_accounts.margin.get_balance()}")

    def _short_open(self, timestamp, bid_price: Decimal):
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

        # Calculate trading fee based on notional value
        notional_value = position_size * self.lot_size * bid_price
        fee = notional_value * self.trading_fees

        # Total deduction from balance: required_margin + fee
        total_deduction = required_margin + fee

        # Check if user has sufficient free margin
        free_margin = self._calculate_equity() - self.user_accounts.margin.get_balance()
        if total_deduction > free_margin:
            self.logger.warning("Insufficient free margin to execute SHORT_OPEN.")
            return

        # Deduct only the fee from user balance
        try:
            self.user_accounts.balance.withdraw(fee)
        except ValueError:
            self.logger.warning("Insufficient balance to execute SHORT_OPEN.")
            return

        # Deduct the required margin from user balance and allocate to margin account
        try:
            self.user_accounts.allocate_margin(required_margin)
        except ValueError as e:
            self.logger.warning(f"Failed to allocate margin: {e}")
            return

        # Collect fees to broker's fees account
        self.broker_accounts.collect_fee(fee)

        # Create and add new position with initial_margin
        new_position = Position(size=position_size, entry_price=bid_price, initial_margin=required_margin)
        self.position_manager.add_short_position(new_position)

        
        # Record trade
        trade_record = TradeRecord(
            timestamp=timestamp,
            operation_type=Action.SHORT_OPEN.name,
            position_size=position_size,
            price=bid_price,
            required_margin=required_margin,
            fee=fee,
            balance=self.user_accounts.balance.get_balance(),
            leverage=self.leverage,
            free_margin=free_margin
        )
        self.record_trade(trade_record)

        self.logger.debug(f"Opened SHORT position: {new_position}")
        self.logger.debug(f"New balance: {self.user_accounts.balance.get_balance()}, "
                          f"Short position: {self.user_accounts.short_position}, "
                          f"Used margin: {self.user_accounts.margin.get_balance()}")

    def _short_close(self, timestamp, ask_price: Decimal):
        """
        Executes a SHORT_CLOSE action.

        Args:
            ask_price (Decimal): The ask price at which the short position is closed.
        """
        if self.user_accounts.short_position <= Decimal('0.0'):
            self.logger.warning("No short position to close.")
            return

        # Close the earliest short position
        try:
            pnl, released_margin, closed_size = self.position_manager.close_short_position(ask_price, self.lot_size)
        except ValueError as e:
            self.logger.error(f"Error closing short position: {e}")
            self.terminated = True
            return

        # Calculate fees based on closed size
        fee = (closed_size * self.lot_size * ask_price) * self.trading_fees

        # Deduct fees from user balance
        try:
            self.user_accounts.balance.withdraw(fee)
        except ValueError:
            self.logger.warning("Insufficient balance to pay fees on SHORT_CLOSE.")
            self.terminated = True
            return

        # Collect fees to broker's fees account
        self.broker_accounts.collect_fee(fee)

        # Realize P&L (add to realized P&L account)
        self.user_accounts.realize_pnl(pnl)

        # Adjust broker's balance based on user's P&L to maintain funds conservation
        self.broker_accounts.adjust_balance(-pnl)

        # Release initial margin back to user balance
        try:
            self.user_accounts.release_margin(released_margin)
        except ValueError as e:
            self.logger.error(f"Error releasing margin: {e}")
            self.terminated = True
            return

        trade_record = TradeRecord(
            timestamp=timestamp,
            operation_type=Action.SHORT_CLOSE.name,
            position_size=closed_size,
            price=ask_price,
            required_margin=0,
            fee=fee,
            balance=self.user_accounts.balance.get_balance(),
            leverage=self.leverage,
            free_margin=self._calculate_equity() - self.user_accounts.margin.get_balance(),
            pnl=pnl,
            closed_size=closed_size,
            released_margin=released_margin
        )
        self.record_trade(trade_record)

        self.logger.debug(f"Closed SHORT position at price {ask_price}")
        self.logger.debug(f"P&L: {pnl}, New balance: {self.user_accounts.balance.get_balance()}, "
                          f"Short position: {self.user_accounts.short_position}, "
                          f"Used margin: {self.user_accounts.margin.get_balance()}")

    def _get_obs(self):
        """
        Constructs the observation as an image with K-line and technical indicators.

        Returns:
            np.ndarray: The observation image.
        """
        if self.is_unittest:
            # Ugly hack to return random image for unit tests
            return np.random.randint(0, 256, size=(self.image_height, self.image_width, self.channels), dtype=np.uint8)
        else:
            # Slice the dataframe for the current window
            window_start = max(0, self.current_step - self.window_size)
            window_end = self.current_step
            df_window = self.df.iloc[window_start:window_end]


            output_filepath = None
            if self.debug_enabled:
                os.makedirs('output', exist_ok=True)
                output_filepath = os.path.join('output', f'{self.currency_pair}_candlestick_{self.current_step}.png')

            # timestamp_at_window_end = df_window.index[-1] if len(df_window) > 0 else None
            # print(f'{timestamp_at_window_end} {self.currency_pair}_candlestick_{self.current_step}.png')
            # Draw the candlestick chart with indicators and return as numpy array
            plotter = BollingerBandPlotter(
                df=df_window,
                trade_record_manager=self.trade_record_manager,
                balance=self.user_accounts.balance.get_balance(),
                fig_width=self.image_width,
                fig_height=self.image_height,
            )

            return plotter.plot(filename=output_filepath)

    def render(self, mode=None):
        """
        Renders the current state of the environment.

        Args:
            mode (str): The mode to render with.

        Returns:
            None or np.ndarray: Returns an image array if mode is 'rgb_array'.
        """
        if mode == 'human':
            equity = self._calculate_equity()
            free_margin = equity - self.user_accounts.margin.get_balance()
            total_asset = float(decimal_to_float(equity, precision=2))
            realized_pnl = float(decimal_to_float(self.user_accounts.realized_pnl, precision=2))
            unrealized_pnl = float(decimal_to_float(self.user_accounts.unrealized_pnl, precision=2))
            fees_collected = float(decimal_to_float(self.broker_accounts.fees.get_balance(), precision=2))
            broker_balance = float(decimal_to_float(self.broker_accounts.balance.get_balance(), precision=2))

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
            print(f'Broker Balance: {broker_balance:.2f}')
            print(f'Total Asset: {total_asset:.2f}')
            print(f'Long Positions: {list(self.position_manager.long_positions)}')
            print(f'Short Positions: {list(self.position_manager.short_positions)}')

    def close(self):
        """
        Performs any necessary cleanup.
        """
        self.logger.info("Environment closed.")
        pass
