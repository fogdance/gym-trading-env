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
from gym_trading_env.utils.decimal_util import decimal_to_float, float_to_decimal
from gym_trading_env.utils.trade_util import calc_unrealized_pnl
from gym_trading_env.envs.trade_record import TradeRecord
from gym_trading_env.envs.trade_record_manager import TradeRecordManager
from gym_trading_env.envs.action import Action, ForexCode
from gym_trading_env.envs.config import TradingConfig
from gym_trading_env.utils.data_processing import load_data
from gym_trading_env.utils.decimal_util import D, D0, D1, D100, quantize_money
from gym_trading_env.utils.build_xt import FEATURES_MARKET, FEATURES_AGENT, build_market_features


class CustomTradingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, df: pd.DataFrame = None, config_path: str = None):
        super(CustomTradingEnv, self).__init__()

        self._config(config_path=config_path)

        self._data(df=df, config=self.config)

        self.valid_actions = [
            Action.HOLD,
            Action.LONG_OPEN0,
            Action.LONG_CLOSE0,
            Action.SHORT_OPEN0,
            Action.SHORT_CLOSE0
        ]
        

        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.df_market = build_market_features(self.df, tz="Asia/Singapore", rollover_hour_local=5)


        # ---- constants ----
        self.DAY_LEN = 1440
        self._F_MARKET = len(FEATURES_MARKET)
        self._F_AGENT  = len(FEATURES_AGENT)

        # Enforce column order and dtype
        dfm = self.df_market.copy()
        dfm = dfm.astype(np.float32)

        # Required session columns
        required_cols = {"day_id", "minute_index_t", "mask_t"}
        missing = required_cols - set(dfm.columns)
        if missing:
            raise ValueError(f"df_market missing columns: {missing}")

        # Numpy views
        minute_idx = dfm["minute_index_t"].to_numpy(dtype=np.int32, copy=False)
        mask_np    = dfm["mask_t"].to_numpy(dtype=np.float32, copy=False)
        X_all      = dfm[FEATURES_MARKET].to_numpy(dtype=np.float32, copy=False)
        day_ids    = dfm["day_id"].to_numpy(copy=False)

        # Discover day segments (contiguous blocks per day_id in order)
        # Assumes df_market is sorted by time.
        unique_days, first_idx = np.unique(day_ids, return_index=True)
        order = np.argsort(first_idx)
        self._days = unique_days[order]
        self._sid_to_dayi = {str(sid): i for i, sid in enumerate(self._days)}
        self._day_ranges = []  # list[(start,end)] half-open
        for d in self._days:
            sel = (day_ids == d)
            start = int(np.argmax(sel))                       # first True
            end   = int(start + sel.sum())                    # first index of next day
            self._day_ranges.append((start, end))

        # Prebuild “full-day” tensors (zeros, then fill rows where mask==1)
        # Shape: [num_days, 1440, F_MARKET]
        num_days = len(self._days)
        self._daily_X = np.zeros((num_days, self.DAY_LEN, self._F_MARKET), dtype=np.float32)
        self._daily_mask = np.zeros((num_days, self.DAY_LEN), dtype=np.float32)

        for di, (s, e) in enumerate(self._day_ranges):
            m_idx = minute_idx[s:e]        # [0..1439]
            msk   = mask_np[s:e]           # 0/1
            rows  = (msk >= 0.5)
            if rows.any():
                self._daily_X[di, m_idx[rows], :] = X_all[s:e, :][rows]
                self._daily_mask[di, m_idx] = msk


        self.observation_space = spaces.Dict({
            "market_seq": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.DAY_LEN, self._F_MARKET),
                dtype=np.float32
            ),
            "agent_state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._F_AGENT,),
                dtype=np.float32
            ),
        })


        self.reset()



    def _config(self, config_path):
        # Configuration management
        if config_path is None:
            raise ValueError("config_path is None")
        
        self.config = TradingConfig.from_yaml(config_path)

        # Validate config
        self.config.validate()


        # Training-specific
        reward_class = reward_classes.get(
            self.config.training.reward_function,
            TotalPnlReward  # 默认使用 TotalPnlReward
        )
        self.reward_function = reward_class(self)
        self.data_window_size = 400

        # Set up logging
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        log_level = getattr(logging, self.config.debug.log_level.upper())
        self.logger.setLevel(log_level)



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
        if df_len < self.config.training.window_size:
            raise ValueError(f"Data has only {df_len} rows, smaller than window_size={self.config.training.window_size}. Not feasible.")
        
        if self.config.training.episode_length is not None:
            # If we do random start, the maximum start index is (df_len - window_size - episode_length)
            max_start = df_len - self.config.training.window_size - self.config.training.episode_length
            if max_start < 0:
                self.logger.warning(
                    f"Data length={df_len} is insufficient to support window_size={self.config.training.window_size} "
                    f"and episode_length={self.config.training.episode_length} in randomize_start. "
                    f"Falling back to episode_length={df_len - self.config.training.window_size}."
                )
                # fallback: reduce episode_length
                self.config.training.episode_length = df_len - self.config.training.window_size
                if self.config.training.episode_length < 1:
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
        self.position_manager = PositionManager()
        self.user_accounts = UserAccounts(initial_balance=self.config.trading.initial_balance, position_manager=self.position_manager)

        self.broker_accounts = BrokerAccounts()  # Initialize broker accounts with balance and fees
        self.trade_record_manager = TradeRecordManager()
        self.metrics = Metrics(self.user_accounts, self.trade_record_manager)

        # Other state variables
        self.current_step = 0
        self.terminated = False
        self.action_result = None
        self.df_window = None
        self.last_close_position = None
        self.action = None

        reward_class = reward_classes.get(
            self.config.training.reward_function,
            TotalPnlReward
        )
        self.reward_function = reward_class(self)
        
        usable = np.where(self._daily_mask.sum(axis=1) > 0)[0]
        if usable.size == 0:
            raise RuntimeError("No days contain any valid minutes (mask_t).")

        if getattr(self.config.training, "randomize_start", True):
            self._day_i = int(self.np_random.choice(usable))
        else:
            self._day_i = int(usable[0])

        
        ts0 = self.df.index[self.current_step]
        try:
            self._start_minute = int(self.df_market.loc[ts0, "minute_index_t"])
        except KeyError:
            self._start_minute = 0

        self.current_minute = self._start_minute


        self.episode_step_count = 0
        self.terminated = False
        self.truncated = False

        # init agent accounting
        self.position = 0
        self.entry_price = D0
        self.holding_minutes = 0
        self.upnl = D0
        self.realized_step = D0
        self.realized_cum  = D0
        self.fee_step = D0
        self.fee_cum  = D0
        self.equity = D(self.config.trading.initial_balance)
        self.max_equity = self.equity
        self.drawdown = D0
        self.sigma_entry = D0
        self.sl_ticks = D0
        self.tp_ticks = D0
        self.sl_price = D0
        self.tp_price = D0
        self.minutes_to_timeout = 0
        
        self._prev_realized_pnl_cum = D0
        self._prev_fee_cum = D0

        # price cache at the *current* minute (if you want: derive price from df at day/minute)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info



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
            action_price = D(self.df.iloc[self.current_step]['Close'])
        except IndexError:
            self.logger.error(f"Current step {self.current_step} is out of bounds for DataFrame with length {len(self.df)}.")
            self.terminated = True
            return self._get_obs(), 0.0, self.terminated, False, {}

        # Execute action
        try:
            self.action = Action(action)
        except ValueError:
            self.logger.error(f"Invalid action: {action}. Action must be one of {list(Action)}.")
            self.terminated = True
            return self._get_obs(), 0.0, self.terminated, False, {}

        self.action_result = ForexCode.SUCCESS
        if self.action == Action.HOLD:
            pass  # Do nothing
        elif self.action == Action.LONG_OPEN:
            self.action_result = self._long_open(action_price, self.config.trading.spread)
        elif self.action == Action.LONG_CLOSE:
            self.action_result = self._long_close(action_price, self.config.trading.spread)
        elif self.action == Action.SHORT_OPEN:
            self.action_result = self._short_open(action_price, self.config.trading.spread)
        elif self.action == Action.SHORT_CLOSE:
            self.action_result = self._short_close(action_price, self.config.trading.spread)
        elif self.action == Action.POSITION_UP:
            self.action_result = self._position_up(action_price, self.config.trading.spread)
        elif self.action == Action.POSITION_DOWN:
            self.action_result = self._position_down(action_price, self.config.trading.spread)
        elif self.action == Action.EMPTY:
            self.action_result = self._empty_position(action_price, self.config.trading.spread)
        elif self.action == Action.LONG_OPEN0:
            self.action_result = self._long_open(action_price, self.config.trading.spread, slot=0)
        elif self.action == Action.LONG_CLOSE0:
            self.action_result = self._long_close(action_price, self.config.trading.spread, slot=0)
        elif self.action == Action.SHORT_OPEN0:
            self.action_result = self._short_open(action_price, self.config.trading.spread, slot=0)
        elif self.action == Action.SHORT_CLOSE0:
            self.action_result = self._short_close(action_price, self.config.trading.spread, slot=0)
        elif self.action == Action.LONG_OPEN1:
            self.action_result = self._long_open(action_price, self.config.trading.spread, slot=1)
        elif self.action == Action.LONG_CLOSE1:
            self.action_result = self._long_close(action_price, self.config.trading.spread, slot=1)
        elif self.action == Action.SHORT_OPEN1:
            self.action_result = self._short_open(action_price, self.config.trading.spread, slot=1)
        elif self.action == Action.SHORT_CLOSE1:
            self.action_result = self._short_close(action_price, self.config.trading.spread, slot=1)

        #
        # wait until 10:05
        #

        # Update step, now is 10:05
        self.current_step += 1
        self.episode_step_count += 1

        # If we are past the last valid row, terminate BEFORE reading df
        if self.current_step >= len(self.df):
            self.terminated = True
            return self._get_obs(), float(0.0), self.terminated, False, self._get_info()

        self.current_minute = min(self.current_minute + 1, self.DAY_LEN - 1)

        self.current_price = D(self.df.iloc[self.current_step]['Close'])

        # Update unrealized P&L
        self._update_unrealized_pnl()

        self.metrics.update(self.df.index[self.current_step])

        if self._should_terminated():
            self.terminated = True
            self._empty_position(self.current_price, self.config.trading.spread)
            self._update_unrealized_pnl()

        # Construct observation
        obs = self._get_obs()

        # Update info
        info = self._get_info()

        # Calculate reward
        reward = self.reward_function(obs)

        if self.terminated and self.config.debug.debug_enabled:
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
        if daily_lost_pct > self.config.risk.daily_lost_ratio or drawdown_pct > self.config.risk.max_drawdown_ratio:
            self.logger.error(f"Terminated: Daily Loss {daily_lost_pct:.4f} > {self.config.risk.daily_lost_ratio} "
                           f"or Drawdown {drawdown_pct:.4f} > {self.config.risk.max_drawdown_ratio}")
            return True

        current_rrr = self.position_manager.calc_profit_factor()
        if self.config.risk.risk_reward_ratio_enable and current_rrr is not None and current_rrr < self.config.risk.risk_reward_ratio:
            self.logger.error(f"Terminated: RRR {current_rrr:.4f} < {self.config.risk.risk_reward_ratio}")
            return True



        # check if we run out of data
        if self.current_step >= self.end_idx:
            self.logger.error(f"Reached end_idx={self.end_idx}, start_idx={self.start_idx}, episode_length={self.config.training.episode_length}, current_step={self.current_step}. Episode done.")
            return True

        # or if we exceed max_episode_steps
        if self.config.training.max_episode_steps > 0 and self.episode_step_count >= self.config.training.max_episode_steps:
            self.logger.error(f"Reached max_episode_steps={self.config.training.max_episode_steps}. Episode done.")
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
                calc_unrealized_pnl(self.current_price, pos=pos, lot_size=self.config.trading.lot_size, long=True)
                for pos in self.position_manager.long_positions if pos is not None
            ),
            Decimal('0.0')  # Specify Decimal start value
        )
        
        # Calculate unrealized P&L for short positions
        unrealized_pnl_short = sum(
            (
                calc_unrealized_pnl(self.current_price, pos=pos, lot_size=self.config.trading.lot_size, long=False)
                for pos in self.position_manager.short_positions  if pos is not None
            ),
            Decimal('0.0')  # Specify Decimal start value
        )
        
        assert isinstance(unrealized_pnl_long, Decimal), "unrealized_pnl_long must be a Decimal."
        assert isinstance(unrealized_pnl_short, Decimal), "unrealized_pnl_short must be a Decimal."

        # Update user's unrealized P&L
        self.user_accounts.unrealized_pnl = unrealized_pnl_long + unrealized_pnl_short



    def _check_margin(self):
        """
        Checks margin requirements and performs liquidation if necessary.

        """
        equity = self._calculate_equity()
        if equity < self.user_accounts.margin.get_balance():
            # Liquidate all positions
            self.logger.info("Equity below margin requirement. Liquidating all positions.")
            while self.user_accounts.long_position > Decimal('0.0'):
                self._long_close(self.current_price, self.config.trading.spread)
            while self.user_accounts.short_position > Decimal('0.0'):
                self._short_close(self.current_price, self.config.trading.spread)
            self.logger.error("Margin requirement not met. Episode terminated.")
            return True
        
        return False


    def _long_open(self, price: Decimal, spread: Decimal, slot: int = None):
        """
        Executes a LONG_OPEN action with manual rollback.
        """
        ask_price = price + spread
        max_additional_long = self.config.trading.max_long_position - self.user_accounts.long_position
        if max_additional_long <= Decimal('0.0'):
            self.logger.warning("Reached maximum long position limit.")
            return ForexCode.ERROR_HIT_MAX_POSITION

        position_size = min(self.config.trading.trade_lot, max_additional_long)
        required_margin = (position_size * self.config.trading.lot_size * ask_price) / self.config.trading.leverage
        fee = self.config.trading.trading_fee_per_lot * position_size
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
            leverage=self.config.trading.leverage,
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
            pnl, released_margin, closed_size, open_price = self.position_manager.close_long_position(bid_price, self.config.trading.lot_size, slot=slot)
        except ValueError as e:
            self.logger.warning(f"Error closing long position: {e}")
            return ForexCode.ERROR_NO_POSITION_TO_CLOSE

        fee = Decimal('0') if not self.config.trading.is_round_turn else self.config.trading.trading_fee_per_lot * self.config.trading.trade_lot

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
            leverage=self.config.trading.leverage,
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
        max_additional_short = self.config.trading.max_short_position - self.user_accounts.short_position
        if max_additional_short <= Decimal('0.0'):
            self.logger.warning("Reached maximum short position limit.")
            return ForexCode.ERROR_HIT_MAX_POSITION

        position_size = min(self.config.trading.trade_lot, max_additional_short)
        required_margin = (position_size * self.config.trading.lot_size * bid_price) / self.config.trading.leverage
        fee = self.config.trading.trading_fee_per_lot * position_size
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
            leverage=self.config.trading.leverage,
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
            pnl, released_margin, closed_size, open_price = self.position_manager.close_short_position(ask_price, self.config.trading.lot_size, slot=slot)
        except ValueError as e:
            self.logger.warning(f"Error closing short position: {e}")
            return ForexCode.ERROR_NO_POSITION_TO_CLOSE

        fee = Decimal('0') if not self.config.trading.is_round_turn else self.config.trading.trading_fee_per_lot * self.config.trading.trade_lot

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
            leverage=self.config.trading.leverage,
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
        Returns the observation dict. This version focuses on the market_seq part.
        - market_seq: (1440, F_MARKET)
            * Past & current minutes (<= current_minute) → data as built (already includes data-valid mask_t).
            * Future minutes (> current_minute)         → all zeros, including mask_t column.
        - agent_state: filled elsewhere; omitted here if you are implementing in two steps.
        """
        # --- Market sequence with temporal censoring ---
        # Base full-day tensor for this episode day: already 0 where data-invalid (mask_t==0).
        X_day = self._daily_X[self._day_i]        # shape: (1440, F_MARKET), dtype float32

        # Build a visibility mask that hides the future minutes strictly.
        # Example: at 10:00, current_minute = 600, we allow indices [0..600].
        end = int(min(self.current_minute, self.DAY_LEN - 1))
        vis = np.zeros((self.DAY_LEN, 1), dtype=np.float32)
        vis[:end + 1, 0] = 1.0

        # Apply temporal visibility (no copy necessary; multiplication produces a new array)
        market_seq = X_day * vis  # future minutes become all-zero rows (incl. mask_t col)

        # --- Agent vector to be implemented in your next step ---
        agent_state = self._get_agent_state_vector()  # stub you’ll implement next

        return {"market_seq": market_seq, "agent_state": agent_state}


    def _get_agent_state_vector(self) -> np.ndarray:
        """
        Build FEATURES_AGENT strictly from existing system state.
        All internal math stays in Decimal; convert with decimal_to_float at the edge.
        """
        # Net position in lots (Decimal), have_long/short flags
        long_lots  = self.user_accounts.long_position             # Decimal
        short_lots = self.user_accounts.short_position            # Decimal
        net_lots   = long_lots - short_lots                       # Decimal

        have_long = 1.0 if long_lots > D0 else 0.0
        have_short= 1.0 if short_lots > D0 else 0.0

        # Pick active side snapshot for entry price & age
        snap = self._active_side_snapshot()
        if snap["side"] is None:
            entry_price = D0
            holding_minutes = D0
        else:
            entry_price = snap["vwap"]          # Decimal VWAP of active side
            holding_minutes = snap["age_min"]   # Decimal minutes (size-weighted)

        # Unrealized & realized (cum) PnL, equity
        upnl_dec = self.user_accounts.unrealized_pnl                      # Decimal
        realized_cum_dec = self.user_accounts.realized_pnl                # Decimal
        equity_dec = self._calculate_equity()                              # Decimal

        # Per-step deltas (safe even if no trade this step)
        realized_step_dec, fee_step_dec = self._step_deltas_from_cum()     # Decimal, Decimal
        fee_cum_dec = self.broker_accounts.fees.get_balance()              # Decimal

        # Peak equity & drawdown from Metrics (already Decimal)
        peak_equity_dec = self.metrics.peak_equity
        drawdown_dec = self.metrics.metrics.get('current_drawdown', D0)

        # Not implemented in your system yet → keep zeros
        sigma_entry_dec = D0
        sl_ticks_dec = D0
        tp_ticks_dec = D0
        sl_price_dec = D0
        tp_price_dec = D0
        minutes_to_timeout_dec = D0

        # Convert to float32 at the very edge (use your helper)
        vec = np.array([
            decimal_to_float(net_lots),             # pos_t (signed lots)
            float(have_long),                       # have_long_t
            float(have_short),                      # have_short_t
            decimal_to_float(entry_price),          # entry_price_t
            decimal_to_float(holding_minutes),      # holding_minutes_t
            decimal_to_float(upnl_dec),             # upnl_t
            decimal_to_float(realized_step_dec),    # realized_pnl_step_t
            decimal_to_float(realized_cum_dec),     # realized_pnl_cum_t
            decimal_to_float(fee_step_dec),         # fee_step_t
            decimal_to_float(fee_cum_dec),          # fee_cum_t
            decimal_to_float(equity_dec),           # equity_t
            decimal_to_float(peak_equity_dec),      # max_equity_t
            decimal_to_float(drawdown_dec),         # drawdown_t
            decimal_to_float(sigma_entry_dec),      # sigma_entry_t
            decimal_to_float(sl_ticks_dec),         # sl_ticks_t
            decimal_to_float(tp_ticks_dec),         # tp_ticks_t
            decimal_to_float(sl_price_dec),         # sl_price_t
            decimal_to_float(tp_price_dec),         # tp_price_t
            decimal_to_float(minutes_to_timeout_dec)# minutes_to_timeout_t
        ], dtype=np.float32)

        return vec


    def _sum_sizes_and_vwap(self, positions, is_long: bool):
        """
        Returns (total_size, vwap_entry_price, size_weighted_avg_age_minutes).
        - total_size: Decimal (sum of lots)
        - vwap_entry_price: Decimal (0 if no positions)
        - age_minutes: Decimal (size-weighted mean of (current_step - open_step))
        """
        total_size = D0
        vwap_num = D0     # sum(size * entry_price)
        age_num  = D0     # sum(size * age_minutes)
        for pos in positions:
            if pos is None:
                continue
            sz = pos.size
            total_size += sz
            vwap_num += (sz * pos.entry_price)
            # open_step is int; convert to Decimal only at the end
            age_minutes = Decimal(self.current_step - pos.open_step)
            age_num += (sz * age_minutes)

        if total_size == D0:
            return D0, D0, D0

        vwap = (vwap_num / total_size)
        age  = (age_num  / total_size)
        return total_size, vwap, age

    def _active_side_snapshot(self):
        """
        Decide the 'active side' to report entry/holding for:
        - If only long: use long side
        - If only short: use short side
        - If both: choose the side with larger total lots; if equal, choose the side whose most recent open is later.
        Returns: dict with keys:
            side ('long'|'short'|None), size (Decimal), vwap (Decimal), age_min (Decimal)
        """
        long_size, long_vwap, long_age = self._sum_sizes_and_vwap(self.position_manager.long_positions, is_long=True)
        short_size, short_vwap, short_age = self._sum_sizes_and_vwap(self.position_manager.short_positions, is_long=False)

        if long_size > D0 and short_size == D0:
            return {"side":"long", "size": long_size, "vwap": long_vwap, "age_min": long_age}
        if short_size > D0 and long_size == D0:
            return {"side":"short","size": short_size,"vwap": short_vwap,"age_min": short_age}
        if long_size == D0 and short_size == D0:
            return {"side":None, "size": D0, "vwap": D0, "age_min": D0}

        # both present → choose larger size; if tie, choose more recent open
        if long_size > short_size:
            return {"side":"long", "size": long_size, "vwap": long_vwap, "age_min": long_age}
        if short_size > long_size:
            return {"side":"short","size": short_size,"vwap": short_vwap,"age_min": short_age}

        # equal size → pick side with latest open_step (most recent)
        latest_long_open = max((p.open_step for p in self.position_manager.long_positions if p is not None), default=-1)
        latest_short_open= max((p.open_step for p in self.position_manager.short_positions if p is not None), default=-1)
        if latest_long_open >= latest_short_open:
            return {"side":"long", "size": long_size, "vwap": long_vwap, "age_min": long_age}
        else:
            return {"side":"short","size": short_size,"vwap": short_vwap,"age_min": short_age}

    def _step_deltas_from_cum(self):
        """
        Compute per-step deltas for realized PnL and fees from cumulatives, then update caches.
        Returns (realized_step: Decimal, fee_step: Decimal)
        """
        realized_cum_now = self.user_accounts.realized_pnl                       # Decimal
        fee_cum_now = self.broker_accounts.fees.get_balance()                    # Decimal

        realized_step = realized_cum_now - self._prev_realized_pnl_cum
        fee_step = fee_cum_now - self._prev_fee_cum

        # update caches for next call
        self._prev_realized_pnl_cum = realized_cum_now
        self._prev_fee_cum = fee_cum_now

        return realized_step, fee_step



    def render(self):
        equity = self._calculate_equity()
        free_margin = equity - self.user_accounts.margin.get_balance()

        print(f'Step: {self.current_step} Balance: {self.user_accounts.balance.get_balance():.2f} Equity: {equity:.2f} Margin: {self.user_accounts.margin.get_balance():.2f} Free Margin: {free_margin:.2f}')
        
        # self._text_render()



    def _text_render(self):
        equity = self._calculate_equity()
        free_margin = equity - self.user_accounts.margin.get_balance()
        total_asset = float(decimal_to_float(equity, precision=2))
        realized_pnl = float(decimal_to_float(self.user_accounts.realized_pnl, precision=2))
        unrealized_pnl = float(decimal_to_float(self.user_accounts.unrealized_pnl, precision=2))
        fees_collected = float(decimal_to_float(self.broker_accounts.fees.get_balance(), precision=2))
        broker_balance = float(decimal_to_float(self.broker_accounts.balance.get_balance(), precision=2))

        print(f'Step: {self.current_step}')
        print(f'Currency Pair: {self.config.trading.currency_pair}')
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



