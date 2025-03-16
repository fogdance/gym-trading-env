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
from gym_trading_env.envs.metrics import Metrics
from gym_trading_env.rewards.reward_functions import TotalPnlReward, reward_classes
from gym_trading_env.utils.conversion import decimal_to_float, float_to_decimal
from gym_trading_env.rendering.plotting import BollingerBandPlotter  # Import plotting utility
from gym_trading_env.rendering.game.game import Game  # Import plotting utility
from gym_trading_env.envs.trade_record import TradeRecord
from gym_trading_env.envs.trade_record_manager import TradeRecordManager
from gym_trading_env.envs.action import Action, ForexCode
from gym_trading_env.envs.config import TradingConfig
from gym_trading_env.utils.data_processing import load_data

# Set global decimal precision
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_UP


class CustomTradingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, df: pd.DataFrame = None, config_path: str = None):
        super(CustomTradingEnv, self).__init__()

        self.config = self._config(config_path=config_path)

        self._data(df=df, config=self.config)

        self.valid_actions = [
            Action.HOLD,
            Action.LONG_OPEN0,
            Action.LONG_CLOSE0,
            Action.LONG_OPEN1,
            Action.LONG_CLOSE1,
            Action.SHORT_OPEN0,
            Action.SHORT_CLOSE0,
            Action.SHORT_OPEN1,
            Action.SHORT_CLOSE1,
            Action.EMPTY
        ]
        
        self.action_space = spaces.Discrete(len(self.valid_actions))

        self.game = None
        if self.game_mode:
            self.game = Game((self.image_width, self.image_height), self.window_size, self.daily_lost_ratio, self.max_drawdown_ratio,
                            decimal_to_float(self.risk_reward_ratio, 2), 
                            decimal_to_float(self.trade_lot, 2), 
                            decimal_to_float(self.max_long_position, 2), 
                            decimal_to_float(self.max_short_position, 2),
                            self.render_mode)

        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(self.image_height, self.image_width, self.channels), dtype=np.uint8),
            'positions': spaces.Box(low=-np.inf, high=np.inf, shape=(12, ), dtype=np.float32),
            'trade_history': spaces.Box(low=-np.inf, high=np.inf, shape=(20, ), dtype=np.float32),
            'indicators': spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32),
            'account': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            'risk': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })

        # Initialize state
        self.position_manager = PositionManager(logger=self.logger)
        self.user_accounts = UserAccounts(initial_balance=self.initial_balance, position_manager=self.position_manager)

        self.broker_accounts = BrokerAccounts()  # Initialize broker accounts with balance and fees
        self.trade_record_manager = TradeRecordManager()
        self.metrics = Metrics(self.user_accounts, self.trade_record_manager)

        # Other state variables
        self.current_step = self.window_size
        self.terminated = False
        self.action_result = None

        # Reset previous total P&L
        self.previous_total_pnl = Decimal('0.0')
        self.previous_equity = Decimal(self.initial_balance)

        self.last_close_position = None

        self.reset()

    def _config(self, config_path):

        # Configuration management
        if config_path is None:
            raise ValueError("config_path is None")
        
        config = TradingConfig.from_yaml(config_path)

        # Validate config
        config.validate()

        # Direct access to nested configs
        self.debug_enabled = config.debug.debug_enabled

        # Trading-specific
        self.currency_pair = config.trading.currency_pair
        self.initial_balance = config.trading.initial_balance
        self.trading_fee_per_lot = config.trading.trading_fee_per_lot
        self.is_round_turn = config.trading.is_round_turn
        self.spread = config.trading.spread
        self.leverage = config.trading.leverage
        self.lot_size = config.trading.lot_size
        self.trade_lot = config.trading.trade_lot
        self.max_long_position = config.trading.max_long_position
        self.max_short_position = config.trading.max_short_position

        # Risk management
        self.max_drawdown_ratio = config.risk.max_drawdown_ratio
        self.daily_lost_ratio = config.risk.daily_lost_ratio
        self.risk_reward_ratio = config.risk.risk_reward_ratio
        self.risk_reward_ratio_enable = config.risk.risk_reward_ratio_enable

        # Training-specific
        reward_class = reward_classes.get(
            config.training.reward_function,
            TotalPnlReward  # 默认使用 TotalPnlReward
        )
        self.reward_function = reward_class(self)
        self.window_size = config.training.window_size
        self.max_episode_steps = config.training.max_episode_steps
        self.randomize_start = config.training.randomize_start
        self.episode_length = config.training.episode_length
        self.game_mode = config.training.game_mode
        self.render_mode = config.training.render_mode
        self.data_window_size = 400

        # Visualization
        self.image_height = config.visualization.image_height
        self.image_width = config.visualization.image_width
        self.channels = config.visualization.image_channels

        # Set up logging
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        log_level = getattr(logging, config.debug.log_level.upper())
        self.logger.setLevel(log_level)

        return config


    def _data(self, df, config):
        # Data
        if df is None:
            df = load_data(config.trading.data_path, config.trading.data_interval)
        self.df = df.copy()
        # Ensure 'Date' is datetime and set as index
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df.set_index('Date', inplace=True)
        elif not isinstance(self.df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame must have a 'Date' column or a DatetimeIndex.")
        
        # Initialize step counters
        self.episode_step_count = 0
        self.start_idx = 0
        self.end_idx = len(self.df)  # default to entire dataset

        # We'll store self.np_random for picking random start
        self.np_random = np.random.default_rng(seed=42)

        # Check basic feasibility right away
        self._check_data_sufficiency()

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
        self.logger.info("REST env")

        super().reset(seed=seed)
        self.position_manager = PositionManager(logger=self.logger)
        self.user_accounts = UserAccounts(initial_balance=self.initial_balance, position_manager=self.position_manager)

        self.broker_accounts = BrokerAccounts()  # Initialize broker accounts with balance and fees
        self.trade_record_manager = TradeRecordManager()
        self.metrics = Metrics(self.user_accounts, self.trade_record_manager)

        # Other state variables
        self.current_step = self.window_size
        self.terminated = False
        self.action_result = None

        # Reset previous total P&L
        self.previous_total_pnl = Decimal('0.0')
        self.previous_equity = Decimal(self.initial_balance)
        
        self.last_close_position = None
        

        df_len = len(self.df)

        # 1) Decide start_idx
        if self.randomize_start and self.episode_length is not None:
            # max possible start
            max_start = df_len - self.data_window_size - self.episode_length
            max_start = max(max_start, 0)  # ensure not negative
            self.start_idx = self.np_random.integers(low=0, high=max_start+1)
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

        return self._get_obs(), self._get_info()



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

        #
        # 10:00
        #

        # Get action price
        action_price = None
        try:
            action_price = Decimal(str(self.df.iloc[self.current_step]['Close']))
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

        self.action_result = ForexCode.SUCCESS
        if action_enum == Action.HOLD:
            pass  # Do nothing
        elif action_enum == Action.LONG_OPEN:
            self.action_result = self._long_open(action_price, self.spread)
        elif action_enum == Action.LONG_CLOSE:
            self.action_result = self._long_close(action_price, self.spread)
        elif action_enum == Action.SHORT_OPEN:
            self.action_result = self._short_open(action_price, self.spread)
        elif action_enum == Action.SHORT_CLOSE:
            self.action_result = self._short_close(action_price, self.spread)
        elif action_enum == Action.POSITION_UP:
            self.action_result = self._position_up(action_price, self.spread)
        elif action_enum == Action.POSITION_DOWN:
            self.action_result = self._position_down(action_price, self.spread)
        elif action_enum == Action.EMPTY:
            self.action_result = self._empty_position(action_price, self.spread)
        elif action_enum == Action.LONG_OPEN0:
            self.action_result = self._long_open(action_price, self.spread, slot=0)
        elif action_enum == Action.LONG_CLOSE0:
            self.action_result = self._long_close(action_price, self.spread, slot=0)
        elif action_enum == Action.SHORT_OPEN0:
            self.action_result = self._short_open(action_price, self.spread, slot=0)
        elif action_enum == Action.SHORT_CLOSE0:
            self.action_result = self._short_close(action_price, self.spread, slot=0)
        elif action_enum == Action.LONG_OPEN1:
            self.action_result = self._long_open(action_price, self.spread, slot=1)
        elif action_enum == Action.LONG_CLOSE1:
            self.action_result = self._long_close(action_price, self.spread, slot=1)
        elif action_enum == Action.SHORT_OPEN1:
            self.action_result = self._short_open(action_price, self.spread, slot=1)
        elif action_enum == Action.SHORT_CLOSE1:
            self.action_result = self._short_close(action_price, self.spread, slot=1)

        #
        # wait until 10:05
        #

        # Update step, now is 10:05
        self.current_step += 1
        self.episode_step_count += 1
        self.current_price = Decimal(str(self.df.iloc[self.current_step]['Close']))

        # Update unrealized P&L
        self._update_unrealized_pnl()

        self.metrics.update(self.df.index[self.current_step])

        if self._should_terminated():
            self.terminated = True
            self._empty_position(self.current_price, self.spread)
            self._update_unrealized_pnl()

        # Calculate reward
        reward = self.reward_function()

        # Construct observation
        obs = self._get_obs()

        # Update info
        info = self._get_info()

        if self.terminated and self.debug_enabled:
            self.trade_record_manager.dump_to_json(f"output/trade_records_{self.current_step}.json")

        # Return the observation, reward (float), termination flags, and info
        return obs, reward, self.terminated, False, info

    def _should_terminated(self):
        # Check termination conditions (e.g., last time step)
        if self.current_step >= len(self.df) - 1:
            self.logger.error(f"Episode terminated. current_step: {self.current_step}, df_len: {len(self.df)}")
            return True

        # Check margin requirements
        if self._check_margin():
            return True
    
        metrics = self.metrics.get_metrics()
        # 检查风险限制（使用百分比形式）
        daily_lost_pct = decimal_to_float(metrics['current_day_lost_pct'] / Decimal('100.0'))
        drawdown_pct = decimal_to_float(metrics['current_drawdown_pct'] / Decimal('100.0')) 
        if daily_lost_pct > self.daily_lost_ratio or drawdown_pct > self.max_drawdown_ratio:
            self.logger.error(f"Terminated: Daily Loss {daily_lost_pct:.4f} > {self.daily_lost_ratio} "
                           f"or Drawdown {drawdown_pct:.4f} > {self.max_drawdown_ratio}")
            return True

        current_rrr = self.position_manager.calc_profit_factor()
        if self.risk_reward_ratio_enable and current_rrr is not None and current_rrr < self.risk_reward_ratio:
            self.logger.error(f"Terminated: RRR {current_rrr:.4f} < {self.risk_reward_ratio}")
            return True



        # check if we run out of data
        if self.current_step >= self.end_idx:
            self.logger.error(f"Reached end_idx={self.end_idx}, current_step={self.current_step}. Episode done.")
            return True

        # or if we exceed max_episode_steps
        if self.max_episode_steps > 0 and self.episode_step_count >= self.max_episode_steps:
            self.logger.error(f"Reached max_episode_steps={self.max_episode_steps}. Episode done.")
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
        return self.user_accounts.equity()

    def _update_unrealized_pnl(self):
        """
        Updates the user's unrealized P&L based on current prices.
        """
        # Calculate unrealized P&L for long positions
        unrealized_pnl_long = sum(
            (
                self._calc_unrealized_pnl(self.current_price, pos=pos, lot_size=self.lot_size, long=True)
                for pos in self.position_manager.long_positions if pos is not None
            ),
            Decimal('0.0')  # Specify Decimal start value
        )
        
        # Calculate unrealized P&L for short positions
        unrealized_pnl_short = sum(
            (
                self._calc_unrealized_pnl(self.current_price, pos=pos, lot_size=self.lot_size, long=False)
                for pos in self.position_manager.short_positions  if pos is not None
            ),
            Decimal('0.0')  # Specify Decimal start value
        )
        
        assert isinstance(unrealized_pnl_long, Decimal), "unrealized_pnl_long must be a Decimal."
        assert isinstance(unrealized_pnl_short, Decimal), "unrealized_pnl_short must be a Decimal."

        # Update user's unrealized P&L
        self.user_accounts.unrealized_pnl = unrealized_pnl_long + unrealized_pnl_short

    def _calc_unrealized_pnl(self, current_price: Decimal,  pos: Position, lot_size: Decimal, long: bool):
        if pos is None:
            return Decimal('0')
        
        if long:
            return (current_price - pos.entry_price) * pos.size * self.lot_size
        else:
            return (pos.entry_price - current_price) * pos.size * self.lot_size


    def _check_margin(self):
        """
        Checks margin requirements and performs liquidation if necessary.

        """
        equity = self._calculate_equity()
        if equity < self.user_accounts.margin.get_balance():
            # Liquidate all positions
            self.logger.info("Equity below margin requirement. Liquidating all positions.")
            while self.user_accounts.long_position > Decimal('0.0'):
                self._long_close(self.current_price - self.spread)
            while self.user_accounts.short_position > Decimal('0.0'):
                self._short_close(self.current_price + self.spread)
            self.logger.error("Margin requirement not met. Episode terminated.")
            return True
        
        return False


    def _long_open(self, price: Decimal, spread: Decimal, slot: int = None):
        """
        Executes a LONG_OPEN action with manual rollback.
        """
        ask_price = price + spread
        max_additional_long = self.max_long_position - self.user_accounts.long_position
        if max_additional_long <= Decimal('0.0'):
            self.logger.warning("Reached maximum long position limit.")
            return ForexCode.ERROR_HIT_MAX_POSITION

        position_size = min(self.trade_lot, max_additional_long)
        required_margin = (position_size * self.lot_size * ask_price) / self.leverage
        fee = self.trading_fee_per_lot * position_size
        total_deduction = required_margin + fee

        free_margin = self._calculate_equity() - self.user_accounts.margin.get_balance()
        if total_deduction > free_margin:
            self.logger.warning("Insufficient free margin to execute LONG_OPEN.")
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        new_position = Position(size=position_size, entry_price=ask_price, initial_margin=required_margin, open_step=self.current_step)

        # Step 1: Deduct fee from balance
        try:
            self.user_accounts.balance.withdraw(fee)
        except ValueError:
            self.logger.warning("Insufficient balance to execute LONG_OPEN.")
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        # Step 2: Allocate margin
        try:
            self.user_accounts.allocate_margin(required_margin)
        except ValueError as e:
            self.logger.warning(f"Failed to allocate margin: {e}")
            self.user_accounts.balance.deposit(fee)  # Rollback Step 1
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        # Step 3: Collect fee to broker
        try:
            self.broker_accounts.collect_fee(fee)
        except ValueError as e:
            self.logger.error(f"Error collecting fee to broker: {e}")
            self.user_accounts.release_margin(required_margin)   # Rollback Step 2
            self.user_accounts.balance.deposit(fee)              # Rollback Step 1
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        # Step 4: Add long position
        try:
            self.position_manager.add_long_position(new_position, slot=slot)
        except ValueError as e:
            self.logger.warning(f"Error opening long position: {e}")
            self.broker_accounts.fees.withdraw(fee)              # Rollback Step 3
            self.user_accounts.release_margin(required_margin)   # Rollback Step 2
            self.user_accounts.balance.deposit(fee)              # Rollback Step 1
            return ForexCode.ERROR_OPEN_POSITION

        # All operations successful
        trade_record = TradeRecord(
            timestamp=self.df.iloc[self.current_step].name,
            operation_type=Action.LONG_OPEN.name,
            position_size=position_size,
            open_price=ask_price,
            close_price=Decimal(0),
            required_margin=required_margin,
            fee=fee,
            balance=self.user_accounts.balance.get_balance(),
            leverage=self.leverage,
            free_margin=self._calculate_equity() - self.user_accounts.margin.get_balance()
        )
        self.record_trade(trade_record)

        self.logger.debug(f"Opened LONG position: {new_position}")
        self.logger.debug(f"New balance: {self.user_accounts.balance.get_balance()}, "
                          f"Long position: {self.user_accounts.long_position}, "
                          f"Used margin: {self.user_accounts.margin.get_balance()}, "
                          f"Fee: {fee}")
        return ForexCode.SUCCESS

    def _long_close(self, price: Decimal, spread: Decimal, slot: int = None):
        """
        Executes a LONG_CLOSE action with manual rollback.
        """
        bid_price = price - spread
        if self.user_accounts.long_position <= Decimal('0.0'):
            self.logger.warning("No long position to close.")
            return ForexCode.ERROR_NO_POSITION_TO_CLOSE

        # Close position to get PNL and margin
        try:
            pnl, released_margin, closed_size, open_price = self.position_manager.close_long_position(bid_price, self.lot_size, slot=slot)
        except ValueError as e:
            self.logger.warning(f"Error closing long position: {e}")
            return ForexCode.ERROR_NO_POSITION_TO_CLOSE

        fee = Decimal('0') if not self.is_round_turn else self.trading_fee_per_lot * self.trade_lot

        # Step 1: Deduct fee from balance
        try:
            self.user_accounts.balance.withdraw(fee)
        except ValueError:
            self.logger.warning("Insufficient balance to pay fees on LONG_CLOSE.")
            self.terminated = True
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        # Step 2: Collect fee to broker
        try:
            self.broker_accounts.collect_fee(fee)
        except ValueError as e:
            self.logger.error(f"Error collecting fee to broker: {e}")
            self.user_accounts.balance.deposit(fee)  # Rollback Step 1
            self.terminated = True
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        # Step 3: Realize PNL
        try:
            self.user_accounts.realize_pnl(pnl)
        except ValueError as e:
            self.logger.error(f"Error realizing PNL: {e}")
            self.broker_accounts.fees.withdraw(fee)  # Rollback Step 2
            self.user_accounts.balance.deposit(fee)  # Rollback Step 1
            self.terminated = True
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        # Step 4: Adjust broker balance
        try:
            self.broker_accounts.adjust_balance(-pnl)
        except ValueError as e:
            self.logger.error(f"Error adjusting broker balance: {e}")
            self.user_accounts.realize_pnl(-pnl)     # Rollback Step 3
            self.broker_accounts.fees.withdraw(fee)  # Rollback Step 2
            self.user_accounts.balance.deposit(fee)  # Rollback Step 1
            self.terminated = True
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        # Step 5: Release margin
        try:
            self.user_accounts.release_margin(released_margin)
        except ValueError as e:
            self.logger.error(f"Error releasing margin: {e}")
            self.broker_accounts.adjust_balance(pnl)  # Rollback Step 4
            self.user_accounts.realize_pnl(-pnl)      # Rollback Step 3
            self.broker_accounts.fees.withdraw(fee)   # Rollback Step 2
            self.user_accounts.balance.deposit(fee)   # Rollback Step 1
            self.terminated = True
            return ForexCode.ERROR_NO_ENOUGH_MONEY
        
        self.last_close_position = {'pnl': pnl, 'margin': released_margin}

        # All operations successful
        trade_record = TradeRecord(
            timestamp=self.df.iloc[self.current_step].name,
            operation_type=Action.LONG_CLOSE.name,
            position_size=closed_size,
            open_price=open_price,
            close_price=bid_price,
            required_margin=Decimal('0'),
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
                          f"Used margin: {self.user_accounts.margin.get_balance()}, "
                          f"Fee: {fee}")
        
        return ForexCode.SUCCESS

    def _short_open(self, price: Decimal, spread: Decimal, slot: int = None):
        """
        Executes a SHORT_OPEN action with manual rollback.
        """
        bid_price = price - spread
        max_additional_short = self.max_short_position - self.user_accounts.short_position
        if max_additional_short <= Decimal('0.0'):
            self.logger.warning("Reached maximum short position limit.")
            return ForexCode.ERROR_HIT_MAX_POSITION

        position_size = min(self.trade_lot, max_additional_short)
        required_margin = (position_size * self.lot_size * bid_price) / self.leverage
        fee = self.trading_fee_per_lot * position_size
        total_deduction = required_margin + fee

        free_margin = self._calculate_equity() - self.user_accounts.margin.get_balance()
        if total_deduction > free_margin:
            self.logger.warning("Insufficient free margin to execute SHORT_OPEN.")
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        new_position = Position(size=position_size, entry_price=bid_price, initial_margin=required_margin, open_step=self.current_step)

        # Step 1: Deduct fee from balance
        try:
            self.user_accounts.balance.withdraw(fee)
        except ValueError:
            self.logger.warning("Insufficient balance to execute SHORT_OPEN.")
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        # Step 2: Allocate margin
        try:
            self.user_accounts.allocate_margin(required_margin)
        except ValueError as e:
            self.logger.warning(f"Failed to allocate margin: {e}")
            self.user_accounts.balance.deposit(fee)  # Rollback Step 1
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        # Step 3: Collect fee to broker
        try:
            self.broker_accounts.collect_fee(fee)
        except ValueError as e:
            self.logger.error(f"Error collecting fee to broker: {e}")
            self.user_accounts.release_margin(required_margin)   # Rollback Step 2
            self.user_accounts.balance.deposit(fee)              # Rollback Step 1
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        # Step 4: Add short position
        try:
            self.position_manager.add_short_position(new_position, slot=slot)
        except ValueError as e:
            self.logger.warning(f"Error opening short position: {e}")
            self.broker_accounts.fees.withdraw(fee)              # Rollback Step 3
            self.user_accounts.release_margin(required_margin)   # Rollback Step 2
            self.user_accounts.balance.deposit(fee)              # Rollback Step 1
            return ForexCode.ERROR_OPEN_POSITION

        # All operations successful
        trade_record = TradeRecord(
            timestamp=self.df.iloc[self.current_step].name,
            operation_type=Action.SHORT_OPEN.name,
            position_size=position_size,
            open_price=bid_price,
            close_price=Decimal(0),
            required_margin=required_margin,
            fee=fee,
            balance=self.user_accounts.balance.get_balance(),
            leverage=self.leverage,
            free_margin=self._calculate_equity() - self.user_accounts.margin.get_balance()
        )
        self.record_trade(trade_record)

        self.logger.debug(f"Opened SHORT position: {new_position}")
        self.logger.debug(f"New balance: {self.user_accounts.balance.get_balance()}, "
                          f"Short position: {self.user_accounts.short_position}, "
                          f"Used margin: {self.user_accounts.margin.get_balance()}, "
                          f"Fee: {fee}")
        return ForexCode.SUCCESS

    def _short_close(self, price: Decimal, spread: Decimal, slot: int = None):
        """
        Executes a SHORT_CLOSE action with manual rollback.
        """
        ask_price = price + spread
        if self.user_accounts.short_position <= Decimal('0.0'):
            self.logger.warning("No short position to close.")
            return ForexCode.ERROR_NO_POSITION_TO_CLOSE

        # Close position to get PNL and margin
        try:
            pnl, released_margin, closed_size, open_price = self.position_manager.close_short_position(ask_price, self.lot_size, slot=slot)
        except ValueError as e:
            self.logger.warning(f"Error closing short position: {e}")
            return ForexCode.ERROR_NO_POSITION_TO_CLOSE

        fee = Decimal('0') if not self.is_round_turn else self.trading_fee_per_lot * self.trade_lot

        # Step 1: Deduct fee from balance
        try:
            self.user_accounts.balance.withdraw(fee)
        except ValueError:
            self.logger.warning("Insufficient balance to pay fees on SHORT_CLOSE.")
            self.terminated = True
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        # Step 2: Collect fee to broker
        try:
            self.broker_accounts.collect_fee(fee)
        except ValueError as e:
            self.logger.error(f"Error collecting fee to broker: {e}")
            self.user_accounts.balance.deposit(fee)  # Rollback Step 1
            self.terminated = True
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        # Step 3: Realize PNL
        try:
            self.user_accounts.realize_pnl(pnl)
        except ValueError as e:
            self.logger.error(f"Error realizing PNL: {e}")
            self.broker_accounts.fees.withdraw(fee)  # Rollback Step 2
            self.user_accounts.balance.deposit(fee)  # Rollback Step 1
            self.terminated = True
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        # Step 4: Adjust broker balance
        try:
            self.broker_accounts.adjust_balance(-pnl)
        except ValueError as e:
            self.logger.error(f"Error adjusting broker balance: {e}")
            self.user_accounts.realize_pnl(-pnl)     # Rollback Step 3
            self.broker_accounts.fees.withdraw(fee)  # Rollback Step 2
            self.user_accounts.balance.deposit(fee)  # Rollback Step 1
            self.terminated = True
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        # Step 5: Release margin
        try:
            self.user_accounts.release_margin(released_margin)
        except ValueError as e:
            self.logger.error(f"Error releasing margin: {e}")
            self.broker_accounts.adjust_balance(pnl)  # Rollback Step 4
            self.user_accounts.realize_pnl(-pnl)      # Rollback Step 3
            self.broker_accounts.fees.withdraw(fee)   # Rollback Step 2
            self.user_accounts.balance.deposit(fee)   # Rollback Step 1
            self.terminated = True
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        self.last_close_position = {'pnl': pnl, 'margin': released_margin}

        # All operations successful
        trade_record = TradeRecord(
            timestamp=self.df.iloc[self.current_step].name,
            operation_type=Action.SHORT_CLOSE.name,
            position_size=closed_size,
            open_price=open_price,
            close_price=ask_price,
            required_margin=Decimal('0'),
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
                          f"Used margin: {self.user_accounts.margin.get_balance()}, "
                          f"Fee: {fee}")
        
        return ForexCode.SUCCESS

    def _position_up(self, price: Decimal, spread: Decimal):
        for pos in self.position_manager.long_positions:
            if pos is not None:
                return self._long_open(price=price, spread=spread)

        for pos in self.position_manager.short_positions:
            if pos is not None:
                return self._short_open(price=price, spread=spread)
            
        return ForexCode.SUCCESS
        
    def _position_down(self, price: Decimal, spread: Decimal):
        for pos in self.position_manager.long_positions:
            if pos is not None:
                return self._long_close(price=price, spread=spread)

        for pos in self.position_manager.short_positions:
            if pos is not None:
                return self._short_close(price=price, spread=spread)
        
        return ForexCode.SUCCESS

    def _empty_position(self, price: Decimal, spread: Decimal):
        # Close all long positions
        for pos in self.position_manager.long_positions:
            if pos is not None:
                self._long_close(price=price, spread=spread)

        # Close all short positions
        for pos in self.position_manager.short_positions:
            if pos is not None:
                self._short_close(price=price, spread=spread)
        
        return ForexCode.SUCCESS

    def _get_obs(self):
        """
        Constructs the observation as an image with K-line and technical indicators.

        Returns:
            np.ndarray: The observation image.
        """

        # Slice the dataframe for the current window
        window_start = max(0, self.current_step - self.data_window_size)
        window_end = self.current_step
        df_window = self.df.iloc[window_start:window_end]
        
        # 1. 当前持仓
        positions = np.zeros((4, 3), dtype=np.float32)
        for i, pos in enumerate(self.position_manager.long_positions[:2]):
            if pos is None:
                continue
            pnl = self._calc_unrealized_pnl(self.current_price, pos=pos, lot_size=self.lot_size, long=True)
            positions[i] = [ decimal_to_float(pnl), decimal_to_float(pos.size), decimal_to_float(pos.entry_price, 5)]

        for i, pos in enumerate(self.position_manager.short_positions[:2]):
            i += 2
            if pos is None:
                continue
            pnl = self._calc_unrealized_pnl(self.current_price, pos=pos, lot_size=self.lot_size, long=False)
            positions[i] = [decimal_to_float(pnl), -decimal_to_float(pos.size), decimal_to_float(pos.entry_price, 5)]                


        # 2. 历史交易（优化版）
        num_records = 5
        trade_history = np.zeros((num_records, 4), dtype=np.float32)
        trades = self.trade_record_manager.trade_history[-20:]
        count = 0

        # 遍历最近 20 条交易记录（倒序遍历）
        for trade in reversed(trades):
            if trade.pnl is None:
                continue

            position_size = 0
            if trade.operation_type == Action.LONG_CLOSE.name:
                position_size = decimal_to_float(trade.position_size)
            elif trade.operation_type == Action.SHORT_CLOSE.name:
                position_size = -decimal_to_float(trade.position_size)

            trade_history[count] = [
                position_size,
                decimal_to_float(trade.pnl),
                decimal_to_float(trade.open_price, 5),
                decimal_to_float(trade.close_price, 5)
            ]
            count += 1
            if count >= num_records:
                break

        # 3. K线图
        image = self._render(render_mode='rgb_array', df=df_window)

        # 4. 技术指标
        indicators = self._calculate_indicators(df=df_window)

        metrics = self.metrics.get_metrics()

        # 5. 账户信息
        account = np.array([
            float(self.user_accounts.balance.get_balance()),
            float(self.user_accounts.equity()),
            float(self.user_accounts.margin.get_balance()),
            float(self._calculate_equity() - self.user_accounts.margin.get_balance()),
            float(metrics["max_profit"]),
            float(metrics["max_loss"]),
        ], dtype=np.float32)

        # 6. 风险管理
        risk = np.array([
            float(0 if metrics["sharpe_ratio"] is None else metrics["sharpe_ratio"]),
            float(0 if metrics["calmar_ratio"] is None else metrics["calmar_ratio"]),
            float(self.daily_lost_ratio),
            float(self.max_drawdown_ratio),
            float(metrics["current_day_lost_pct"]),
            float(metrics["current_drawdown_pct"]),
        ], dtype=np.float32)

        return {
            'image': image,
            'positions': positions.flatten(),
            'trade_history': trade_history.flatten(),
            'indicators': indicators,
            'account': account,
            'risk': risk
        }



    def _render(self, render_mode, df):
        if self.game_mode:
            if df is not None:
                self.game.step(df)
            return self.game.render(decimal_to_float(self.position_manager.total_long_position(), precision=2),
                                    decimal_to_float(self.position_manager.total_short_position(), precision=2),
                                        0 if self.user_accounts.margin.get_balance() == Decimal('0') else decimal_to_float(self.user_accounts.unrealized_pnl/self.user_accounts.margin.get_balance(), precision=4), 
                                        decimal_to_float(self.user_accounts.current_day_lost_pct / Decimal('100.0')),
                                        decimal_to_float(self.user_accounts.current_drawdown_pct / Decimal('100.0')),
                                        None if self.position_manager.calc_profit_factor() is None else decimal_to_float(self.position_manager.calc_profit_factor(), precision=2), 
                                        render_mode=render_mode)
        else:
            if render_mode == 'rgb_array':
                output_filepath = None
                if self.debug_enabled:
                    os.makedirs('output', exist_ok=True)
                    output_filepath = os.path.join('output', f'{self.currency_pair}_candlestick_{self.current_step}.png')

                df_15m = df.resample('15min').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                render_df = df_15m[-self.window_size:]

                # timestamp_at_window_end = df_window.index[-1] if len(df_window) > 0 else None
                # print(f'{timestamp_at_window_end} {self.currency_pair}_candlestick_{self.current_step}.png')
                # Draw the candlestick chart with indicators and return as numpy array
                plotter = BollingerBandPlotter(
                    df=render_df,
                    channels=self.channels,
                    trade_record_manager=self.trade_record_manager,
                    balance=self.user_accounts.balance.get_balance(),
                    fig_width=self.image_width,
                    fig_height=self.image_height,
                )

                return plotter.plotOnlyCandle(filename=output_filepath)


    def render(self):
        equity = self._calculate_equity()
        free_margin = equity - self.user_accounts.margin.get_balance()

        print(f'Step: {self.current_step} Balance: {self.user_accounts.balance.get_balance():.2f} Equity: {equity:.2f} Margin: {self.user_accounts.margin.get_balance():.2f} Free Margin: {free_margin:.2f}')
        
        # self._text_render()

        if self.render_mode == 'human':
            self._render(render_mode='human', df=None)


    def _text_render(self):
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


    def _calculate_indicators(self, df):
        # 计算 h1, l1, h2, l2
        def get_hl(df, up_thresh, down_thresh):
            engineer = FeatureEngineer()
            features = engineer.get_zigzag_features(df=df, up_thresh=up_thresh, down_thresh=down_thresh, debug=self.debug_enabled)
            h1 = features['Prev_High'].iloc[-1]  # 最近高
            l1 = features['Prev_Low'].iloc[-1]   # 最近低
            h2 = features['Prev_Prev_High'].iloc[-1]  # 前前高
            l2 = features['Prev_Prev_Low'].iloc[-1]   # 前前低
            return h1,l1,h2,l2
        
        df_5m = df.copy()

        df_15m = df_5m.resample('15min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        df_1h = df_5m.resample('60min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        indicators = np.array([
            *get_hl(df_5m[-30:], self.config.trading.up_thresh_5m, self.config.trading.down_thresh_5m),   # 5分钟
            *get_hl(df_15m[-30:], self.config.trading.up_thresh_15m, self.config.trading.down_thresh_15m),  # 15分钟
            *get_hl(df_1h[-30:], self.config.trading.up_thresh_1h, self.config.trading.down_thresh_1h),   # 1小时
            self.df.iloc[self.current_step]['Close']
        ], dtype=np.float32)
        return indicators

