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
from pathlib import Path

from gym_trading_env.envs.accounting import Ledger, JournalEntry, Posting, LedgerError
from gym_trading_env.utils.feature_engineering import FeatureEngineer
from gym_trading_env.envs.position import Position
from gym_trading_env.envs.user_accounts import UserAccounts
from gym_trading_env.envs.broker_accounts import BrokerAccounts
from gym_trading_env.envs.position_manager import PositionManager
from gym_trading_env.envs.metrics import Metrics
from gym_trading_env.rewards.reward_functions import EquityDeltaReward, reward_classes
from gym_trading_env.utils.decimal_util import decimal_to_float, float_to_decimal
from gym_trading_env.utils.trade_util import calc_unrealized_pnl
from gym_trading_env.envs.trade_record import TradeRecord
from gym_trading_env.envs.trade_record_manager import TradeRecordManager
from gym_trading_env.envs.action import Action, ForexCode
from gym_trading_env.envs.config import TradingConfig
from gym_trading_env.utils.data_processing import load_data
from gym_trading_env.utils.decimal_util import D, D0, D1, D100, quantize_money, number_to_float
from gym_trading_env.utils.market_features import FEATURES_MARKET, build_market_features
from gym_trading_env.utils.agent_features import FEATURES_AGENT
from gym_trading_env.utils.session_futures_strict import DEFAULT_TZ
from gym_trading_env.utils.plot_intraday import save_intraday_html
from gym_trading_env.envs.account import Account

from gym_trading_env.utils.agent_features import (
    AgentFeatureInput, compute_agent_features, agent_feature_vector, compute_unrealized_pnl
)


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
        self.df_market = build_market_features(self.df, rollover_hour_local=5, is_future=self.config.trading.is_future)

        # ---- constants ----

        self._F_MARKET = len(FEATURES_MARKET)
        self._F_AGENT  = len(FEATURES_AGENT)

        # Enforce column order and dtype
        dfm = self.df_market.copy()
        dfm = dfm.astype(np.float32)

        # Required session columns
        required_cols = {"day_id", "minute_index_t", "mask_t"}
        if getattr(self.config.trading, "stop_loss_enabled", False):
            required_cols |= {"H_t", "L_t"}
        missing = required_cols - set(dfm.columns)
        if missing:
            raise ValueError(f"df_market missing columns: {missing}")
        
            
        # Numpy views
        minute_idx = dfm["minute_index_t"].to_numpy(dtype=np.int32, copy=False)
        mask_np    = dfm["mask_t"].to_numpy(dtype=np.float32, copy=False)
        X_all      = dfm[FEATURES_MARKET].to_numpy(dtype=np.float32, copy=False)
        day_ids    = dfm["day_id"].to_numpy(copy=False)

        # Normalize all day ids once
        day_ids_norm = np.array([self._day_key(x) for x in day_ids], dtype=object)

        # Compute unique days in order of first appearance
        unique_days, first_idx = np.unique(day_ids_norm, return_index=True)
        order = np.argsort(first_idx)
        self._days = unique_days[order]

        # Map day_key -> day_index
        self._sid_to_dayi = {dk: i for i, dk in enumerate(self._days)}

        # Day ranges (half-open)
        self._day_ranges = []
        for dk in self._days:
            sel = (day_ids_norm == dk)
            start = int(np.argmax(sel))           # first True
            end   = int(start + sel.sum())        # first index of next day
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
                shape=(self.window_size, self._F_MARKET),
                dtype=np.float32
            ),
            "agent_state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._F_AGENT,),
                dtype=np.float32
            ),
        })


        self.reset()

        if self.config.debug.debug_enabled:
            i = self.current_step
            print("df        :", self.df.index[i])
            print("df_market :", self.df_market.index[i])             # 如果这里不是同一个时间，就是 iloc 错位
            ts = self.df.index[i]
            print("df_m.loc  :", None if ts not in self.df_market.index else ts)
            print("len(df)=", len(self.df), "len(df_market)=", len(self.df_market))

            save_intraday_html(
                df_market=self.df_market,
                title = (f"{self.config.trading.currency_pair} "f"{self.df_market.index[self.current_step]}"),
                out_path= ("/tmp/"f"{self.config.trading.currency_pair} "f"{self.df_market.index[self.current_step]}"".html"),
                start_pos=self.current_step,
                end_pos=self.end_idx)



    def _config(self, config_path):
        # Configuration management
        if config_path is None:
            raise ValueError("config_path is None")
        
        self.config = TradingConfig.from_yaml(config_path)

        # Validate config
        self.config.validate()

        # DAY_LEN 只依赖 is_future，提前定好
        self.DAY_LEN = 345 if self.config.trading.is_future else 1440

        # window_size：不允许超过 DAY_LEN（否则你永远无法从“单日 full tensor”里切出更长窗口）
        ws = int(getattr(self.config.training, "window_size", self.DAY_LEN))
        if ws <= 0:
            raise ValueError(f"window_size must be > 0, got {ws}")
        if ws > self.DAY_LEN:
            self.logger.warning(f"window_size={ws} > DAY_LEN={self.DAY_LEN}, clamp to {self.DAY_LEN}")
            ws = self.DAY_LEN

        self.window_size = ws
        self.config.training.window_size = ws 

        # Training-specific
        reward_class = reward_classes.get(
            self.config.training.reward_function,
            EquityDeltaReward
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

        self.logger.info(f"config_path (absolute): {Path(config_path).resolve()}")

    def _data(self, df, config):
        # Data
        if df is None:
            df = load_data(config.trading.data_path, config.trading.data_interval)
        # Ensure 'Date' is datetime and set as index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame must have a 'Date' column or a DatetimeIndex.")
        
        # Initialize step counters
        self.episode_step_count = 0
        self.start_idx = 0
        self.end_idx = len(df)  # default to entire dataset

        # We'll store self.np_random for picking random start
        self.np_random = np.random.default_rng(seed=42)

        # Check basic feasibility right away
        self._check_data_sufficiency(df)

        self.df = df.copy()


    def _check_data_sufficiency(self, df):
        """
        Checks if the DataFrame is large enough given window_size and episode_length.
        If not sufficient, raise ValueError or adapt the config as fallback.
        """
        df_len = len(df)
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
        Picks a random valid start row if training.randomize_start is True.
        At reset, the visible market window shows all minutes from the session open
        up to the chosen start minute (inclusive).
        """
        self.logger.info("REST env")
        super().reset(seed=seed)

        self.position_manager = PositionManager()
        self.broker_accounts = BrokerAccounts()
        self.trade_record_manager = TradeRecordManager()

        # --- Ledger (double-entry) ---
        self.ledger = Ledger()

        # 1) 创建“真钱账户”（唯一一份）
        user_cash_acct = Account(Decimal(str(self.config.trading.initial_balance)))
        user_margin_acct = Account(Decimal("0.0"))

        # 2) 先注册进 Ledger（Ledger 以后会操作这些账户）
        self.ledger.register("user_cash", user_cash_acct, strict_nonnegative=True)
        self.ledger.register("user_margin", user_margin_acct, strict_nonnegative=True)
        self.ledger.register("broker_fee_income", self.broker_accounts.fee_income, strict_nonnegative=False)
        self.ledger.register("broker_pnl", self.broker_accounts.broker_pnl, strict_nonnegative=False)

        # 3) UserAccounts 只引用这些账户（读余额、算 equity、存 pnl 投影）
        self.user_accounts = UserAccounts(
            ledger=self.ledger,
            position_manager=self.position_manager,
            cash_account=user_cash_acct,
            margin_account=user_margin_acct,
            initial_balance=self.config.trading.initial_balance,
        )

        self._ledger_total0 = self.ledger.total_balance()

        self.metrics = Metrics(self.user_accounts, self.trade_record_manager)

        # --- Housekeeping ---
        self.terminated = False
        self.truncated = False
        self.action_result = None
        self.df_window = None
        self.last_close_position = None
        self.action = None

        reward_class = reward_classes.get(
            self.config.training.reward_function, EquityDeltaReward
        )
        self.reward_function = reward_class(self)

        # -------------------------------
        # Choose a valid start row by clock anchor
        # -------------------------------
        start_policy = getattr(self.config.training, "start_clock", "random_9_or_21")

        # 先根据时钟锚点生成候选行
        candidate_rows = self._candidate_start_rows_by_clock(start_policy)

        # 约束 episode_length（需要给定窗口足够）
        df_len = len(self.df_market)
        episode_len = self.config.training.episode_length
        if episode_len is not None:
            max_start = df_len - int(episode_len)
            candidate_rows = candidate_rows[candidate_rows <= max_start]

        # 若锚点集合为空，退回到原有的“任意有效行”
        if candidate_rows.size == 0:
            mask_np = self.df_market["mask_t"].to_numpy(dtype=np.float32, copy=False)
            candidate_rows = np.flatnonzero(mask_np >= 0.5)
            if episode_len is not None:
                max_start = df_len - int(episode_len)
                candidate_rows = candidate_rows[candidate_rows <= max_start]
            if candidate_rows.size == 0:
                raise RuntimeError("No valid start rows found (after applying start_clock and episode_length).")

        # 随机挑选：如策略为 random_9_or_21，并且既有 09:00 也有 21:00，会混合在 candidate_rows 再随机
        if getattr(self.config.training, "randomize_start", True):
            start_row = int(self.np_random.choice(candidate_rows))
        else:
            start_row = int(candidate_rows[0])


        # --- Align all counters/indexes to this chosen row ---
        self.current_step = start_row
        ts0 = self.df_market.index[self.current_step]

        # Resolve the episode's day index using normalized day_id
        try:
            day_id_raw = self.df_market.loc[ts0, "day_id"]
            self._day_i = int(self._sid_to_dayi[self._day_key(day_id_raw)])
        except Exception:
            # Fallback: find the day range that contains current_step
            self._day_i = 0
            for i, (s, e) in enumerate(self._day_ranges):
                if s <= self.current_step < e:
                    self._day_i = i
                    break

        # Set the current visible minute using minute_index_t (not +1 arithmetic)
        try:
            self._start_minute = int(self.df_market.loc[ts0, "minute_index_t"])
        except KeyError:
            self._start_minute = 0
        self.current_minute = self._start_minute

        # Bound the episode if episode_length is provided
        self.start_idx = self.current_step
        if episode_len is not None:
            self.end_idx = min(df_len, self.start_idx + int(episode_len))
        else:
            self.end_idx = df_len

        if self.config.trading.is_future:
            end_idx_15 = self._compute_end_idx_at_15(start_row, tz="Asia/Shanghai")
            self.end_idx = min(self.end_idx, end_idx_15)

        end_ts = self.df_market.index[self.end_idx - 1] if self.end_idx > self.start_idx else ts0
        self.logger.info(f"{ts0} -> {end_ts} (end_idx={self.end_idx})")

        # Per-episode counters
        self.episode_step_count = 0

        # --- Accounting zeros ---
        self.position = 0
        self.entry_price = D0
        self.holding_minutes = 0
        self.upnl = D0
        self.realized_step = D0
        self.realized_cum = D0
        self.fee_step = D0
        self.fee_cum = D0
        self.equity = D(self.config.trading.initial_balance)
        self.max_equity = self.equity
        self.drawdown = D0
        self.sigma_entry = D0
        self.sl_ticks = D0
        self.tp_ticks = D0
        self.sl_price = D0
        self.tp_price = D0
        self.minutes_to_timeout = 0

        # Cumulative caches (used to compute per-step deltas) — initialize from current totals
        self._prev_realized_pnl_cum = self.user_accounts.realized_pnl
        self._prev_fee_cum = self.broker_accounts.fee_income.get_balance()

        # Make sure deltas start at 0 for the first obs
        self.realized_step = D0
        self.fee_step = D0
        self.stop_loss_fired = 0

        self._last_valid_price = D(self.df_market.iloc[self.current_step]["C_t"])
        self._refresh_agent_state()

        # First observation
        obs = self._get_obs()
        info = self._get_info()
        return obs, info




    def _day_key(self, val):
        """Normalize day_id to a single canonical string form.
        Handles numpy scalars and float vs int (e.g. 0, 0.0) uniformly."""
        import numpy as _np
        if isinstance(val, _np.generic):
            val = val.item()
        if isinstance(val, (int, _np.integer)):
            return str(int(val))
        if isinstance(val, (float, _np.floating)):
            # ':g' turns 0.0 -> '0', 20200101.0 -> '20200101'
            return f"{float(val):g}"
        return str(val)

    def step(self, action):
        """
        Executes one time step within the environment.

        Args:
            action (int): The action to take (index into self.valid_actions).

        Returns:
            Tuple: (observation, reward, terminated, truncated, info)
        """
        if self.terminated:
            return self._get_obs(), 0.0, self.terminated, False, {}

        # Map discrete action index -> Action enum (MUST use valid_actions)
        try:
            a = int(action)
            if a < 0 or a >= len(self.valid_actions):
                raise ValueError(f"Action index out of range: {a}")
            requested_action = self.valid_actions[a]
            self.action = requested_action
        except Exception:
            self.logger.error(f"Invalid action: {action}. Must be an int in [0, {len(self.valid_actions)-1}]")
            self.terminated = True
            return self._get_obs(), 0.0, self.terminated, False, {}

        if self.config.debug.debug_enabled:
            self.logger.info(f"{self.df_market.index[self.current_step]}, {action} -> {self.action}")


        # --- Price / market-closed gate at CURRENT step (t) ---
        try:
            mask_now = float(self.df_market.iloc[self.current_step]["mask_t"])
            market_open = (mask_now >= 0.5)

            if market_open:
                action_price = D(self.df_market.iloc[self.current_step]["C_t"])
                self._last_valid_price = action_price
                market_code = ForexCode.SUCCESS
            else:
                action_price = self._last_valid_price
                # 若 agent 在闭市时尝试非 HOLD，则记为 market closed；否则仍算 SUCCESS
                market_code = ForexCode.SUCCESS if (self.action == Action.HOLD) else ForexCode.ERROR_MARKET_CLOSED
                self.action = Action.HOLD  # 强制不交易
        except Exception as e:
            self.logger.error(
                f"Failed to read action price at step={self.current_step} "
                f"(len={len(self.df_market)}): {e}"
            )
            self.terminated = True
            return self._get_obs(), 0.0, self.terminated, False, {}

        # --- Execute action ---
        self.action_result = market_code

        if self.action == Action.HOLD:
            pass
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

        # Behavior counters
        try:
            in_market = (self.user_accounts.long_position > D0) or (self.user_accounts.short_position > D0)
        except Exception:
            in_market = False
        self.metrics.on_step(self.action, self.action_result == ForexCode.SUCCESS, in_market=in_market)

        #
        # 推进到 t+1 → 设置 current_price → 先跑止损 → 再 update_unrealized → metrics.update
        #

        # --- Advance time to NEXT step (t+1) ---
        self.current_step += 1
        self.episode_step_count += 1

        # Bound check BEFORE sync/index access
        if self.current_step >= len(self.df_market):
            self.terminated = True
            # Update deltas once for this final transition
            self._update_step_deltas()
            return self._get_obs(), 0.0, self.terminated, False, self._get_info()

        # Sync day/minute based on minute_index_t (no +1 drift)
        self._sync_day_and_minute()

        # Mark-to-market price at NEXT step
        mask_next = float(self.df_market.iloc[self.current_step]["mask_t"])
        if mask_next < 0.5:
            self.current_price = self._last_valid_price
        else:
            self.current_price = D(self.df_market.iloc[self.current_step]["C_t"])
            self._last_valid_price = self.current_price

        # 先止损（可能会自动平仓，改变仓位/保证金/现金）
        self.stop_loss_fired += self._apply_stop_losses()

        # Update unrealized P&L
        self._update_unrealized_pnl()

        # Update metrics (equity/drawdown etc.)
        self.metrics.update(self.df_market.index[self.current_step])

        # Termination rules
        if self._should_terminated():
            self.terminated = True
            self._empty_position(self.current_price, self.config.trading.spread)
            self._update_unrealized_pnl()

        # IMPORTANT: update per-step deltas ONCE here (no side effects in obs)
        self._update_step_deltas()

        # refresh agent state vector ONCE
        self._refresh_agent_state()

        # Construct observation & info
        obs = self._get_obs()
        info = self._get_info()

        # Calculate reward
        reward = self.reward_function(obs)

        if self.terminated and self.config.debug.debug_enabled:
            self.trade_record_manager.dump_to_json(f"output/trade_records_{self.current_step}.json")

        return obs, reward, self.terminated, False, info


    def _sync_day_and_minute(self):
        ts = self.df_market.index[self.current_step]
        day_key = self._day_key(self.df_market.loc[ts, "day_id"])
        self._day_i = int(self._sid_to_dayi[day_key])
        self.current_minute = int(self.df_market.loc[ts, "minute_index_t"])


    def _update_step_deltas(self):
        """
        Update per-step deltas from cumulative realized pnl & fees.
        IMPORTANT: This must be called exactly once per env.step(),
        and NEVER inside _get_obs() .
        """
        realized_cum_now = self.user_accounts.realized_pnl            # Decimal
        fee_cum_now = self.broker_accounts.fee_income.get_balance()         # Decimal

        # Per-step deltas
        self.realized_step = realized_cum_now - self._prev_realized_pnl_cum
        self.fee_step = fee_cum_now - self._prev_fee_cum

        # Update caches
        self._prev_realized_pnl_cum = realized_cum_now
        self._prev_fee_cum = fee_cum_now

    def _should_terminated(self):
        # Check termination conditions (e.g., last time step)
        if self.current_step >= len(self.df_market) - 1:
            self.logger.error(f"Episode terminated. current_step: {self.current_step}, df_len: {len(self.df_market)}")
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
            'fees_collected': self.broker_accounts.fee_income.get_balance(),
            'broker_balance': self.broker_accounts.broker_pnl.get_balance(),  # Added broker balance
            'balance': self.user_accounts.cash_balance .get_balance(),
            'equity': self._calculate_equity(),
            'used_margin': self.user_accounts.used_margin .get_balance(),
            'free_margin': self._calculate_equity() - self.user_accounts.used_margin .get_balance(),
            'long_position': self.user_accounts.long_position,
            'short_position': self.user_accounts.short_position,
            'stop_loss_fired': self.stop_loss_fired,
        }


        # Add the whole metrics bag under 'log/env/*'
        env_metrics = self.metrics.get_metrics()
        for k, v in env_metrics.items():
            vv = np.nan if v is None else number_to_float(v)
            info[f'log/env/{k}'] = np.asarray(vv, dtype=np.float32).reshape(())

        # Handy real-time signals
        info['log/env/equity'] = number_to_float(info['equity'])
        info['log/env/used_margin'] = number_to_float(info['used_margin'])
        info['log/env/free_margin'] = number_to_float(info['free_margin'])
        info['log/env/long_position'] = number_to_float(info['long_position'])
        info['log/env/short_position'] = number_to_float(info['short_position'])
        info['log/env/stop_loss_fired'] = number_to_float(info['stop_loss_fired'])

        return info

    def _calculate_equity(self) -> Decimal:
        return self.user_accounts.equity()

    def _update_unrealized_pnl(self):
        self.user_accounts.unrealized_pnl = compute_unrealized_pnl(
            self.position_manager.long_positions,
            self.position_manager.short_positions,
            self.current_price,
            self.config.trading.lot_size,
        )

    def _refresh_agent_state(self):
        # prev_max_equity 用 env 里的 max_equity 做缓存（只在 step() 的固定位置更新）
        prev_max = getattr(self, "max_equity", D(self.config.trading.initial_balance))

        inp = AgentFeatureInput(
            long_positions=self.position_manager.long_positions,
            short_positions=self.position_manager.short_positions,
            current_step=self.current_step,
            current_price=getattr(self, "current_price", self._last_valid_price),
            lot_size=self.config.trading.lot_size,

            realized_pnl_step=getattr(self, "realized_step", D0),
            realized_pnl_cum=self.user_accounts.realized_pnl,
            fee_step=getattr(self, "fee_step", D0),
            fee_cum=self.broker_accounts.fee_income.get_balance(),

            cash_balance=self.user_accounts.cash_balance.get_balance(),
            used_margin=self.user_accounts.used_margin.get_balance(),

            prev_max_equity=prev_max,
        )

        feat = compute_agent_features(inp)

        # 缓存（让 state 更新发生在 step / reset，而不是 _get_obs）
        self.upnl = feat["upnl_t"]
        self.equity = feat["equity_t"]
        self.max_equity = feat["max_equity_t"]
        self.drawdown = feat["drawdown_t"]

        self._agent_state_vec = agent_feature_vector(feat)


    # --- NEW: 将索引本地化 ---
    def _localize_index(self, idx, tz: str):
        if getattr(idx, "tz", None) is None:
            return idx.tz_localize(tz)
        return idx.tz_convert(tz)
    
    # --- NEW: 根据时钟锚点生成候选起点行 ---
    def _candidate_start_rows_by_clock(self, start_clock: str) -> np.ndarray:
        """
        返回满足 start_clock 条件（'09:00'/'21:00'/'random_9_or_21'）且 mask_t==1 的行号数组。
        若 start_clock == 'any' 则返回所有 mask_t==1 的行。
        """
        mask_np = self.df_market["mask_t"].to_numpy(dtype=np.float32, copy=False)
        # 先挑 mask==1
        valid_mask_rows = np.flatnonzero(mask_np >= 0.5)

        if start_clock == "any":
            return valid_mask_rows

        # 将 df.index 本地化到期货默认时区（或按需替换为数据所在时区）
        tz = DEFAULT_TZ  # "Asia/Shanghai"
        idx_local = self._localize_index(self.df_market.index, tz)

        hours = idx_local.hour
        minutes = idx_local.minute

        def rows_at(hh, mm):
            sel = np.flatnonzero((hours == hh) & (minutes == mm))
            # 同时要求 mask==1
            return np.intersect1d(sel, valid_mask_rows, assume_unique=False)

        if start_clock == "09:00":
            return rows_at(9, 1)

        if start_clock == "21:00":
            return rows_at(21, 1)

        if start_clock == "random_9_or_21":
            r9 = rows_at(9, 1)
            r21 = rows_at(21, 1)
            # 两个集合并，随机时刻在 reset() 里用 choice 再随机
            return np.concatenate([r9, r21]) if (r9.size + r21.size) > 0 else np.array([], dtype=int)

        # 兜底
        return valid_mask_rows

    # --- NEW: 计算“当日 15:00（或夜盘起 -> 次日 15:00）”对应的 self.end_idx ---
    def _compute_end_idx_at_15(self, start_row: int, tz: str = "Asia/Shanghai") -> int:
        idx_local = self._localize_index(self.df_market.index, tz)

        df_len = len(self.df_market)

        t0 = idx_local[start_row]
        base_day = t0.normalize()
        end_day = base_day + pd.Timedelta(days=1) if t0.hour >= 18 else base_day

        # 你的数据锚点是 xx:01，这里用 15:01 更匹配
        end_ts = pd.Timestamp(end_day.date(), tz=tz) + pd.Timedelta(hours=15, minutes=1)

        arr_ns = idx_local.asi8
        # half-open: first index STRICTLY greater than end_ts
        end_idx = int(np.searchsorted(arr_ns, end_ts.value, side="right"))

        end_idx = min(df_len, max(end_idx, start_row + 1))
        return end_idx




    def _check_margin(self):
        """
        Checks margin requirements and performs liquidation if necessary.

        """
        equity = self._calculate_equity()
        if equity < self.user_accounts.used_margin .get_balance():
            # Liquidate all positions
            self.logger.info("Equity below margin requirement. Liquidating all positions.")
            while self.user_accounts.long_position > Decimal('0.0'):
                self._long_close(self.current_price, self.config.trading.spread)
            while self.user_accounts.short_position > Decimal('0.0'):
                self._short_close(self.current_price, self.config.trading.spread)
            self.logger.error("Margin requirement not met. Episode terminated.")
            return True
        
        return False

    def _post_atomic(self, entry: JournalEntry) -> None:
        """
        单条分录原子入账：失败抛 LedgerError，由调用方 snapshot/restore。
        """
        self.ledger.post(entry)

    def _assert_ledger_conservation(self):
        # 可选：debug 时确保系统内资金守恒
        if self.config.debug.debug_enabled:
            now = self.ledger.total_balance()
            if now != self._ledger_total0:
                raise RuntimeError(f"Ledger conservation broken: {now} != {self._ledger_total0}")

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

        free_margin = self._calculate_equity() - self.user_accounts.used_margin.get_balance()
        if (required_margin + fee) > free_margin:
            self.logger.warning("Insufficient free margin to execute LONG_OPEN.")
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        new_position = Position(size=position_size, entry_price=ask_price, initial_margin=required_margin, open_step=self.current_step,
                                stop_loss_price=self._compute_stop_loss_price(ask_price, side="long"))

        ts = self.df_market.iloc[self.current_step].name
        entry = JournalEntry(
            timestamp=ts,
            memo="LONG_OPEN",
            postings=[
                Posting("user_cash", -(required_margin + fee)),
                Posting("user_margin", +required_margin),
                Posting("broker_fee_income", +fee),
            ],
            meta={"side":"long","slot":slot,"price":str(ask_price)}
        )

        snap = self.ledger.snapshot()
        try:
            self._post_atomic(entry)
            self.position_manager.add_long_position(new_position, slot=slot)
            self._assert_ledger_conservation()
        except (LedgerError, ValueError) as e:
            self.ledger.restore(snap)
            self.logger.warning(f"LONG_OPEN failed and rolled back: {e}")
            return ForexCode.ERROR_OPEN_POSITION if isinstance(e, ValueError) else ForexCode.ERROR_NO_ENOUGH_MONEY

        trade_record = TradeRecord(
            timestamp=ts,
            operation_type=Action.LONG_OPEN.name,
            position_size=position_size,
            open_price=ask_price,
            close_price=Decimal(0),
            required_margin=required_margin,
            fee=fee,
            balance=self.user_accounts.cash_balance.get_balance(),
            leverage=self.config.trading.leverage,
            free_margin=self._calculate_equity() - self.user_accounts.used_margin.get_balance()
        )
        self.record_trade(trade_record)
        return ForexCode.SUCCESS


    def _long_close(self, price: Decimal, spread: Decimal, slot: int = None):
        """
        Executes a LONG_CLOSE action with manual rollback.
        """        
        bid_price = price - spread
        if self.user_accounts.long_position <= Decimal('0.0'):
            self.logger.warning("No long position to close.")
            return ForexCode.ERROR_NO_POSITION_TO_CLOSE

        try:
            pnl, released_margin, closed_size, open_price = self.position_manager.close_long_position(
                bid_price, self.config.trading.lot_size, slot=slot
            )
        except ValueError as e:
            self.logger.warning(f"Error closing long position: {e}")
            return ForexCode.ERROR_NO_POSITION_TO_CLOSE

        fee = Decimal('0') if not self.config.trading.is_round_turn else (self.config.trading.trading_fee_per_lot * closed_size)

        ts = self.df_market.iloc[self.current_step].name
        entry = JournalEntry(
            timestamp=ts,
            memo="LONG_CLOSE",
            postings=[
                Posting("user_margin", -released_margin),
                Posting("broker_pnl", -pnl),  # pnl>0 用户赚 -> broker_pnl 减少
                Posting("broker_fee_income", +fee),
                Posting("user_cash", +(released_margin + pnl - fee)),
            ],
            meta={"side":"long","slot":slot,"close_price":str(bid_price),"pnl":str(pnl)}
        )

        snap = self.ledger.snapshot()
        try:
            self._post_atomic(entry)
            self.user_accounts.realize_pnl(pnl)  # 仅统计字段
            self._assert_ledger_conservation()
        except LedgerError as e:
            self.ledger.restore(snap)
            self.logger.error(f"LONG_CLOSE failed and rolled back: {e}")
            self.terminated = True
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        self.last_close_position = {'pnl': pnl, 'margin': released_margin}

        trade_record = TradeRecord(
            timestamp=ts,
            operation_type=Action.LONG_CLOSE.name,
            position_size=closed_size,
            open_price=open_price,
            close_price=bid_price,
            required_margin=Decimal('0'),
            fee=fee,
            balance=self.user_accounts.cash_balance.get_balance(),
            leverage=self.config.trading.leverage,
            free_margin=self._calculate_equity() - self.user_accounts.used_margin.get_balance(),
            pnl=pnl,
            closed_size=closed_size,
            released_margin=released_margin
        )
        self.record_trade(trade_record)
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

        free_margin = self._calculate_equity() - self.user_accounts.used_margin.get_balance()
        if (required_margin + fee) > free_margin:
            self.logger.warning("Insufficient free margin to execute SHORT_OPEN.")
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        new_position = Position(size=position_size, entry_price=bid_price, initial_margin=required_margin, open_step=self.current_step,
                                stop_loss_price=self._compute_stop_loss_price(bid_price, side="short"))

        ts = self.df_market.iloc[self.current_step].name
        entry = JournalEntry(
            timestamp=ts,
            memo="SHORT_OPEN",
            postings=[
                Posting("user_cash", -(required_margin + fee)),
                Posting("user_margin", +required_margin),
                Posting("broker_fee_income", +fee),
            ],
            meta={"side":"short","slot":slot,"price":str(bid_price)}
        )

        snap = self.ledger.snapshot()
        try:
            self._post_atomic(entry)
            self.position_manager.add_short_position(new_position, slot=slot)
            self._assert_ledger_conservation()
        except (LedgerError, ValueError) as e:
            self.ledger.restore(snap)
            self.logger.warning(f"SHORT_OPEN failed and rolled back: {e}")
            return ForexCode.ERROR_OPEN_POSITION if isinstance(e, ValueError) else ForexCode.ERROR_NO_ENOUGH_MONEY

        trade_record = TradeRecord(
            timestamp=ts,
            operation_type=Action.SHORT_OPEN.name,
            position_size=position_size,
            open_price=bid_price,
            close_price=Decimal(0),
            required_margin=required_margin,
            fee=fee,
            balance=self.user_accounts.cash_balance.get_balance(),
            leverage=self.config.trading.leverage,
            free_margin=self._calculate_equity() - self.user_accounts.used_margin.get_balance()
        )
        self.record_trade(trade_record)
        return ForexCode.SUCCESS
   

    def _short_close(self, price: Decimal, spread: Decimal, slot: int = None):
        """
        Executes a SHORT_CLOSE action with manual rollback.
        """
        ask_price = price + spread
        if self.user_accounts.short_position <= Decimal('0.0'):
            self.logger.warning("No short position to close.")
            return ForexCode.ERROR_NO_POSITION_TO_CLOSE

        try:
            pnl, released_margin, closed_size, open_price = self.position_manager.close_short_position(
                ask_price, self.config.trading.lot_size, slot=slot
            )
        except ValueError as e:
            self.logger.warning(f"Error closing short position: {e}")
            return ForexCode.ERROR_NO_POSITION_TO_CLOSE

        fee = Decimal('0') if not self.config.trading.is_round_turn else (self.config.trading.trading_fee_per_lot * closed_size)

        ts = self.df_market.iloc[self.current_step].name
        entry = JournalEntry(
            timestamp=ts,
            memo="SHORT_CLOSE",
            postings=[
                Posting("user_margin", -released_margin),
                Posting("broker_pnl", -pnl),
                Posting("broker_fee_income", +fee),
                Posting("user_cash", +(released_margin + pnl - fee)),
            ],
            meta={"side":"short","slot":slot,"close_price":str(ask_price),"pnl":str(pnl)}
        )

        snap = self.ledger.snapshot()
        try:
            self._post_atomic(entry)
            self.user_accounts.realize_pnl(pnl)
            self._assert_ledger_conservation()
        except LedgerError as e:
            self.ledger.restore(snap)
            self.logger.error(f"SHORT_CLOSE failed and rolled back: {e}")
            self.terminated = True
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        self.last_close_position = {'pnl': pnl, 'margin': released_margin}

        trade_record = TradeRecord(
            timestamp=ts,
            operation_type=Action.SHORT_CLOSE.name,
            position_size=closed_size,
            open_price=open_price,
            close_price=ask_price,
            required_margin=Decimal('0'),
            fee=fee,
            balance=self.user_accounts.cash_balance.get_balance(),
            leverage=self.config.trading.leverage,
            free_margin=self._calculate_equity() - self.user_accounts.used_margin.get_balance(),
            pnl=pnl,
            closed_size=closed_size,
            released_margin=released_margin
        )
        self.record_trade(trade_record)
        return ForexCode.SUCCESS
    
    def _apply_stop_losses(self) -> int:
        """
        返回本 step 触发止损的平仓次数。
        触发规则：
        long:  low <= pos.stop_loss_price
        short: high >= pos.stop_loss_price
        成交价：用 stop_loss_price 作为成交价（并复用 close 的 spread 逻辑）
        """
        if self.config.trading.stop_loss_enabled == False:        
            return 0

        # 闭市不触发
        mask_next = float(self.df_market.iloc[self.current_step]["mask_t"])
        if mask_next < 0.5:
            return 0

        low, high = self._get_bar_low_high()
        if low is None or high is None:
            return 0

        fired = 0
        spr = self.config.trading.spread

        # long slots
        for slot, pos in enumerate(self.position_manager.long_positions):
            if pos is None or pos.stop_loss_price is None:
                continue
            if low <= pos.stop_loss_price:
                # 让 _long_close 的 bid_price = stop_loss_price
                # _long_close 里 bid = price - spread => 传入 price = SL + spread
                self._long_close(price=pos.stop_loss_price + spr, spread=spr, slot=slot)
                self.logger.warning(f"trigger stoploss, long {pos}")
                fired += 1

        # short slots
        for slot, pos in enumerate(self.position_manager.short_positions):
            if pos is None or pos.stop_loss_price is None:
                continue
            if high >= pos.stop_loss_price:
                # 让 _short_close 的 ask_price = stop_loss_price
                # _short_close 里 ask = price + spread => 传入 price = SL - spread
                self._short_close(price=pos.stop_loss_price - spr, spread=spr, slot=slot)
                self.logger.warning(f"trigger stoploss, short {pos}")
                fired += 1

        return fired


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
        market_seq: (window_size, F_MARKET)
        - 取最近 window_size 分钟的“已发生历史片段”
        - 当历史不足 window_size：右侧补 0
        """
        X_day = self._daily_X[self._day_i]  # (DAY_LEN, F_MARKET)

        end = int(min(self.current_minute, self.DAY_LEN - 1))
        start = max(0, end - self.window_size + 1)

        window = X_day[start:end + 1, :]   # (L, F), L<=window_size
        L = window.shape[0]

        if L < self.window_size:
            pad = np.zeros((self.window_size - L, self._F_MARKET), dtype=np.float32)
            market_seq = np.concatenate([window, pad], axis=0)  # 右侧补 0
        else:
            market_seq = window.astype(np.float32, copy=False)

        agent_state = getattr(self, "_agent_state_vec", np.zeros((self._F_AGENT,), dtype=np.float32))
        return {"market_seq": market_seq, "agent_state": agent_state}


    def _get_bar_low_high(self):
        row = self.df_market.iloc[self.current_step]
        # 兼容列名（看你 build_market_features 输出）
        if "L_t" in row and "H_t" in row:
            low = D(row["L_t"])
            high = D(row["H_t"])
        elif "Low" in row and "High" in row:
            low = D(row["Low"])
            high = D(row["High"])
        else:
            # 没有高低价就没法做“触发<=low”的止损
            self.logger.warning("StopLoss: df_market missing low/high columns (L_t/H_t or Low/High).")
            low = None
            high = None
        return low, high


    def _compute_stop_loss_price(self, entry_price: Decimal, side: str) -> Decimal | None:
        if self.config.trading.stop_loss_enabled == False:
            return None
        
        mode = self.config.trading.stop_loss_mode
        value  = self.config.trading.stop_loss_value  # Decimal

        if value <= 0:
            return None

        if mode == "pct":
            if side == "long":
                return entry_price * (Decimal("1") - value)
            else:
                return entry_price * (Decimal("1") + value)
        elif mode == "abs":
            if side == "long":
                return entry_price - value
            else:
                return entry_price + value
        else:
            self.logger.warning(f"StopLoss: unknown mode={mode}, ignored.")
            return None

    def _obs_market_df(self, market_seq: np.ndarray) -> pd.DataFrame:
        # 找到该 day 的 minute=0 的真实开盘 timestamp，用它当 1440 分钟时间轴起点
        s, e = self._day_ranges[self._day_i]
        sub = self.df_market.iloc[s:e]

        try:
            t0 = sub.index[sub["minute_index_t"].astype(int).to_numpy() == 0][0]
        except Exception:
            t0 = sub.index[0]

        idx = pd.date_range(t0, periods=self.DAY_LEN, freq="min", tz=t0.tz)

        dfp = pd.DataFrame(market_seq, index=idx, columns=FEATURES_MARKET)
        # 如果 save_intraday_html 依赖这些列，就补上
        dfp["minute_index_t"] = np.arange(self.DAY_LEN, dtype=np.int32)
        dfp["mask_t"] = self._daily_mask[self._day_i].astype(np.float32)
        return dfp

    



    def render(self):
        equity = self._calculate_equity()
        free_margin = equity - self.user_accounts.used_margin .get_balance()

        print(f'Step: {self.current_step} Balance: {self.user_accounts.cash_balance .get_balance():.2f} Equity: {equity:.2f} Margin: {self.user_accounts.used_margin .get_balance():.2f} Free Margin: {free_margin:.2f}')
        
        # self._text_render()



    def _text_render(self):
        equity = self._calculate_equity()
        free_margin = equity - self.user_accounts.used_margin .get_balance()
        total_asset = float(decimal_to_float(equity, precision=2))
        realized_pnl = float(decimal_to_float(self.user_accounts.realized_pnl, precision=2))
        unrealized_pnl = float(decimal_to_float(self.user_accounts.unrealized_pnl, precision=2))
        fees_collected = float(decimal_to_float(self.broker_accounts.fee_income.get_balance(), precision=2))
        broker_balance = float(decimal_to_float(self.broker_accounts.broker_pnl.get_balance(), precision=2))

        print(f'Step: {self.current_step}')
        print(f'Currency Pair: {self.config.trading.currency_pair}')
        print(f'Balance: {self.user_accounts.cash_balance .get_balance():.2f}')
        print(f'Equity: {equity:.2f}')
        print(f'Used Margin: {self.user_accounts.used_margin .get_balance():.2f}')
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



