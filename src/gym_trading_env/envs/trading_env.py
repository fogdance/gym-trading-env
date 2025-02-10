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
from gym_trading_env.envs.action import Action, ForexCode

# Set global decimal precision
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_UP


class CustomTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, render_mode: str = 'human', config: dict = None):
        super(CustomTradingEnv, self).__init__()

        self.render_mode = render_mode

        # Configuration management
        if config is None:
            config = {}
        self.is_unittest = config.get('is_unittest', False)
        self.debug_enabled = False
    

        # Overdue positions
        self.max_holding_bars = config.get('max_holding_bars', 0) 
        # Disallow hedging
        self.allow_hedging = config.get('allow_hedging', False)
        # Max inactivity
        self.max_no_trade_bars = config.get('max_no_trade_bars', 30)
        
        # Penalty for violation
        self.violation_penalty = float(config.get('violation_penalty', 50.0))

        self.forced_termination = False
        self.just_closed_trade = None

        # Track last trade step
        self.last_trade_step = None
        self.out_of_boundary_penalty = float(config.get('out_of_boundary_penalty', 100.0))
        self.max_drawdown_ratio = Decimal(str(config.get('max_drawdown_ratio', 0.3)))

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
 
        self.trade_record_manager = TradeRecordManager()

        # Set up logging
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

        # Data
        self.df = df.copy()
        # Ensure 'Date' is datetime and set as index
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df.set_index('Date', inplace=True)
        elif not isinstance(self.df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame must have a 'Date' column or a DatetimeIndex.")


        self.max_episode_steps = config.get('max_episode_steps', 1000)  # e.g. limit episode length
        self.randomize_start = config.get('randomize_start', True)     # random start index
        self.episode_length = config.get('episode_length', None)       # if not None, fix length of each episode

        # Initialize step counters
        self.episode_step_count = 0
        self.start_idx = 0
        self.end_idx = len(self.df)  # default to entire dataset

        # We'll store self.np_random for picking random start
        # if you're using gym>=0.26, you can do self.np_random = np.random.default_rng(seed)
        self.np_random = np.random.RandomState(seed=42)

        # Check basic feasibility right away
        self._check_data_sufficiency()


        # Action space: dynamically based on Action Enum
        self.action_space = spaces.Discrete(len(Action))

        # Define image dimensions
        self.image_height = config.get('image_height', 256)
        self.image_width = config.get('image_width', 256)
        self.channels = config.get('image_channels', 1)

        # Update observation space to image
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(self.image_height, self.image_width, self.channels), dtype=np.uint8),
            # 'realized_pnl': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            # 'balance': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'positions': spaces.Box(low=-np.inf, high=np.inf, shape=(4, 3), dtype=np.float32),
        })

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

    def _check_data_sufficiency(self):
        """
        Checks if the DataFrame is large enough given window_size and episode_length.
        If not sufficient, raise ValueError or adapt the config as fallback.
        """
        df_len = len(self.df)
        if df_len < self.window_size:
            raise ValueError(f"Data has only {df_len} rows, smaller than window_size={self.window_size}. Not feasible.")
        
        if self.episode_length is not None:
            # If we do random start, the maximum start index is (df_len - window_size - episode_length)
            max_start = df_len - self.window_size - self.episode_length
            if max_start < 0:
                self.logger.warning(
                    f"Data length={df_len} is insufficient to support window_size={self.window_size} "
                    f"and episode_length={self.episode_length} in randomize_start. "
                    f"Falling back to episode_length={df_len - self.window_size}."
                )
                # fallback: reduce episode_length
                self.episode_length = df_len - self.window_size
                if self.episode_length < 1:
                    raise ValueError("Even after fallback, there's no feasible episode_length. Please provide more data.")


    def position_to_features(self, position: Position, is_long: bool) -> np.ndarray:
        entry_price = position.entry_price
        position_size = position.size
        current_price = self.current_price

        if is_long:
            current_unrealized_pnl = (current_price - entry_price) * position_size * self.lot_size
            position_type = Action.LONG_OPEN.value
        else:
            current_unrealized_pnl = (entry_price - current_price) * position_size * self.lot_size
            position_type = Action.SHORT_OPEN.value

        features = [
            float(position_size),
            float(position_type),
            float(current_unrealized_pnl),
        ]
        return np.array(features, dtype=np.float32)
    
    def get_current_positions(self):
        positions = []
        for pos in self.position_manager.long_positions:
            positions.append(self.position_to_features(pos, is_long=True))

        for pos in self.position_manager.short_positions:
            positions.append(self.position_to_features(pos, is_long=False))

        positions = positions[:4]

        while len(positions) < 4:
            positions.append(np.zeros(3, dtype=np.float32))
        return np.array(positions, dtype=np.float32)

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
        self.terminated = False
        self.forced_termination = False
        self.episode_step_count = 0
        df_len = len(self.df)

        # 1) Decide start_idx
        if self.randomize_start and self.episode_length is not None:
            # max possible start
            max_start = df_len - self.window_size - self.episode_length
            max_start = max(max_start, 0)  # ensure not negative
            self.start_idx = self.np_random.randint(low=0, high=max_start+1)
        else:
            # simple scenario: start at 0
            self.start_idx = 0

        # 2) Decide end_idx
        if self.episode_length is not None:
            self.end_idx = min(self.start_idx + self.episode_length, df_len)
        else:
            # use entire data
            self.end_idx = df_len

        # 3) current_step starts after window_size to ensure we have enough hist data
        self.current_step = self.start_idx + self.window_size
        if self.current_step >= self.end_idx:
            # if that happens, it means there's no valid range
            self.logger.warning(
                f"current_step={self.current_step} >= end_idx={self.end_idx}. "
                f"Data might be too short. Forcing ended episode."
            )
            self.terminated = True


        # Reset previous total P&L
        self.previous_total_pnl = Decimal('0.0')
        self.previous_equity = Decimal(self.initial_balance)
        self.last_trade_step = None
        return self._get_obs(), self._get_info()

    def _check_drawdown(self, equity: Decimal) -> bool:
        """
        Checks if equity < initial_balance*(1 - max_drawdown_ratio).
        Returns True if the boundary is crossed, else False.
        """
        drawdown_threshold = self.initial_balance * (Decimal('1.0') - self.max_drawdown_ratio)
        if equity < drawdown_threshold:
            self.logger.info(
                f"[CRASH] Max drawdown triggered! equity={equity:.2f} < threshold={drawdown_threshold:.2f}"
            )
            return True
        return False


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


        # Get current price
        try:
            self.current_price = Decimal(str(self.df.iloc[self.current_step]['Close']))
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

        # If the action is a real trade (not HOLD), update last_trade_step
        if action_enum != Action.HOLD:
            self.last_trade_step = self.current_step

        result = ForexCode.SUCCESS
        if action_enum == Action.HOLD:
            pass  # Do nothing
        elif action_enum == Action.LONG_OPEN:
            result = self._long_open(self.current_price + self.spread)
        elif action_enum == Action.LONG_CLOSE:
            result = self._long_close(self.current_price - self.spread)
        elif action_enum == Action.SHORT_OPEN:
            result = self._short_open(self.current_price - self.spread)
        elif action_enum == Action.SHORT_CLOSE:
            result = self._short_close(self.current_price + self.spread)

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

        # Check margin requirements
        self._check_margin(equity)

        reward = 0

        # Then check drawdown
        is_drawdown_crash = self._check_drawdown(equity)
        if is_drawdown_crash:
            self.terminated = True
            reward -= self.out_of_boundary_penalty

        # violation checks
        is_violation = self._check_violations()
        if is_violation or result == ForexCode.ERROR_HIT_MAX_POSITION or result == ForexCode.ERROR_NO_POSITION_TO_CLOSE:
            self.terminated = True
            reward -= self.violation_penalty            

        self.episode_step_count += 1

        if self.terminated:
            self.forced_termination = True
            self.position_manager.close_all_position(self.current_price, self.lot_size)
            self._update_unrealized_pnl()

        # check if we run out of data
        if self.current_step >= self.end_idx:
            self.logger.debug(
                f"Reached end_idx={self.end_idx}, current_step={self.current_step}. Episode done."
            )
            self.terminated = True

        # or if we exceed max_episode_steps
        if self.max_episode_steps > 0 and self.episode_step_count >= self.max_episode_steps:
            self.logger.debug(
                f"Reached max_episode_steps={self.max_episode_steps}. Episode done."
            )
            self.terminated = True

        # Calculate reward
        reward += self.reward_function(self)

        # Construct observation
        obs = self._get_obs()

        # Update info
        info = self._get_info()

        if self.terminated and self.debug_enabled:
            self.trade_record_manager.dump_to_json(f"output/trade_records_{self.current_step}.json")

        # Return the observation, reward (float), termination flags, and info
        return obs, reward, self.terminated, False, info
    
    def _check_violations(self) -> bool:
        """
        Returns True if any violation is detected:
        1) Overdue positions
        2) Hedging not allowed
        3) Inactivity
        """
        if self._check_overdue_positions():
            self.logger.warning("[VIOLATION] Overdue position.")
            return True
        
        if not self.allow_hedging:
            if self.position_manager.total_long_position() > Decimal('0.0') and self.position_manager.total_short_position > Decimal('0.0'):
                self.logger.warning("[VIOLATION] Hedging not allowed, but both long and short exist.")
                return True
        
        if self._check_inactivity():
            self.logger.warning("[VIOLATION] Agent is inactive for too many bars.")
            return True
        
        return False

    def _check_overdue_positions(self) -> bool:
        # skip if max_holding_bars <= 0
        if self.max_holding_bars <= 0:
            return False
        
        for pos in self.position_manager.long_positions:
            if pos.open_step is not None:
                if (self.current_step - pos.open_step) > self.max_holding_bars:
                    return True
        
        for pos in self.position_manager.short_positions:
            if pos.open_step is not None:
                if (self.current_step - pos.open_step) > self.max_holding_bars:
                    return True
        
        return False

    def _check_inactivity(self) -> bool:
        if self.max_no_trade_bars <= 0:
            return False
        if self.last_trade_step is None:
            return False
        
        if (self.current_step - self.last_trade_step) > self.max_no_trade_bars:
            return True
        
        return False

    def _get_info(self):
        """
        Retrieve information about the current state.

        Returns:
            dict: Information dictionary containing various account details.
        """
        info = {
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

    def _long_open(self, ask_price: Decimal):
        """
        Executes a LONG_OPEN action.

        Args:
            ask_price (Decimal): The ask price at which the long position is opened.
        """
        max_additional_long = self.max_long_position - self.user_accounts.long_position
        if max_additional_long <= Decimal('0.0'):
            self.logger.warning("Reached maximum long position limit.")
            return ForexCode.ERROR_HIT_MAX_POSITION

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
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        # Deduct only the fee from user balance
        try:
            self.user_accounts.balance.withdraw(fee)
        except ValueError:
            self.logger.warning("Insufficient balance to execute LONG_OPEN.")
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        # Deduct the required margin from user balance and allocate to margin account
        try:
            self.user_accounts.allocate_margin(required_margin)
        except ValueError as e:
            self.logger.warning(f"Failed to allocate margin: {e}")
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        # Collect fees to broker's fees account
        self.broker_accounts.collect_fee(fee)

        # Create and add new position with initial_margin
        new_position = Position(size=position_size, entry_price=ask_price, initial_margin=required_margin, open_step=self.current_step)
        self.position_manager.add_long_position(new_position)

        # Record trade
        trade_record = TradeRecord(
            timestamp=self.df.iloc[self.current_step].name,
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
        return ForexCode.SUCCESS

    def _long_close(self, bid_price: Decimal):
        """
        Executes a LONG_CLOSE action.

        Args:
            bid_price (Decimal): The bid price at which the long position is closed.
        """
        if self.user_accounts.long_position <= Decimal('0.0'):
            self.logger.warning("No long position to close.")
            return ForexCode.ERROR_NO_POSITION_TO_CLOSE

        # Close the earliest long position
        try:
            pnl, released_margin, closed_size = self.position_manager.close_long_position(bid_price, self.lot_size)
        except ValueError as e:
            self.logger.error(f"Error closing long position: {e}")
            self.terminated = True
            return ForexCode.ERROR_NO_POSITION_TO_CLOSE

        # Calculate fees based on closed size
        fee = (closed_size * self.lot_size * bid_price) * self.trading_fees

        # Deduct fees from user balance
        try:
            self.user_accounts.balance.withdraw(fee)
        except ValueError:
            self.logger.warning("Insufficient balance to pay fees on LONG_CLOSE.")
            self.terminated = True
            return ForexCode.ERROR_NO_ENOUGH_MONEY
        
        self.just_closed_trade = pnl

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
            return ForexCode.ERROR_NO_ENOUGH_MONEY
        
        trade_record = TradeRecord(
            timestamp=self.df.iloc[self.current_step].name,
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
        
        return ForexCode.SUCCESS

    def _short_open(self, bid_price: Decimal):
        """
        Executes a SHORT_OPEN action.

        Args:
            bid_price (Decimal): The bid price at which the short position is opened.
        """
        max_additional_short = self.max_short_position - self.user_accounts.short_position
        if max_additional_short <= Decimal('0.0'):
            self.logger.warning("Reached maximum short position limit.")
            return ForexCode.ERROR_HIT_MAX_POSITION

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
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        # Deduct only the fee from user balance
        try:
            self.user_accounts.balance.withdraw(fee)
        except ValueError:
            self.logger.warning("Insufficient balance to execute SHORT_OPEN.")
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        # Deduct the required margin from user balance and allocate to margin account
        try:
            self.user_accounts.allocate_margin(required_margin)
        except ValueError as e:
            self.logger.warning(f"Failed to allocate margin: {e}")
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        # Collect fees to broker's fees account
        self.broker_accounts.collect_fee(fee)

        # Create and add new position with initial_margin
        new_position = Position(size=position_size, entry_price=bid_price, initial_margin=required_margin, open_step=self.current_step)
        self.position_manager.add_short_position(new_position)

        
        # Record trade
        trade_record = TradeRecord(
            timestamp=self.df.iloc[self.current_step].name,
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
        return ForexCode.SUCCESS

    def _short_close(self, ask_price: Decimal):
        """
        Executes a SHORT_CLOSE action.

        Args:
            ask_price (Decimal): The ask price at which the short position is closed.
        """
        if self.user_accounts.short_position <= Decimal('0.0'):
            self.logger.warning("No short position to close.")
            return ForexCode.ERROR_NO_POSITION_TO_CLOSE

        # Close the earliest short position
        try:
            pnl, released_margin, closed_size = self.position_manager.close_short_position(ask_price, self.lot_size)
        except ValueError as e:
            self.logger.error(f"Error closing short position: {e}")
            self.terminated = True
            return ForexCode.ERROR_NO_POSITION_TO_CLOSE

        # Calculate fees based on closed size
        fee = (closed_size * self.lot_size * ask_price) * self.trading_fees

        # Deduct fees from user balance
        try:
            self.user_accounts.balance.withdraw(fee)
        except ValueError:
            self.logger.warning("Insufficient balance to pay fees on SHORT_CLOSE.")
            self.terminated = True
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        self.just_closed_trade = pnl

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
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        trade_record = TradeRecord(
            timestamp=self.df.iloc[self.current_step].name,
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
        
        return ForexCode.SUCCESS

    def _get_obs(self):
        """
        Constructs the observation as an image with K-line and technical indicators.

        Returns:
            np.ndarray: The observation image.
        """
        if self.is_unittest:
            # Ugly hack to return random image for unit tests
            image = np.random.randint(0, 256, size=(self.image_height, self.image_width, self.channels), dtype=np.uint8)
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
                channels=self.channels,
                trade_record_manager=self.trade_record_manager,
                balance=self.user_accounts.balance.get_balance(),
                fig_width=self.image_width,
                fig_height=self.image_height,
            )

            image = plotter.plot(filename=output_filepath)
        
        positions = self.get_current_positions()

        obs =  {
            'image': image,
            # 'realized_pnl': np.array([float(decimal_to_float(self.user_accounts.realized_pnl, precision=2))], dtype=np.float32),
            # 'balance': np.array([float(decimal_to_float(self.user_accounts.balance.get_balance()))], dtype=np.float32),
            'positions': positions,
        }

        return obs


    def render(self):
        if self.render_mode == 'human':
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
