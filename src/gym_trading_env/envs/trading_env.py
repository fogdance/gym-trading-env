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
from gym_trading_env.utils.market_features import FEATURES_MARKET, FEATURES_MARKET_OBS, build_market_features
from gym_trading_env.utils.session_futures_strict import DEFAULT_TZ
from gym_trading_env.utils.plot_intraday import save_intraday_html
from gym_trading_env.envs.account import Account
from gym_trading_env.utils.daily_features import (
    build_daily_context_and_seq,
    FEATURES_DAILY_CONTEXT, FEATURES_DAILY_CONTEXT_OBS,
    DAILY_SEQ_LEN,
)

from gym_trading_env.utils.agent_features import (
    AgentFeatureInput, compute_agent_features_raw, compute_agent_features_obs, compute_unrealized_pnl,
    agent_feature_vector, FEATURES_AGENT, FEATURES_AGENT_OBS
)



class CustomTradingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, df: pd.DataFrame = None, config_path: str = None, render_mode: str | None = None):
        super(CustomTradingEnv, self).__init__()

        self._config(config_path=config_path)

        self._data(df=df, config=self.config)

        self.render_mode = render_mode or getattr(self.config.training, "render_mode", "none")
        if self.render_mode is None:
            self.render_mode = "none"
        if self.render_mode != "none" and self.render_mode not in self.metadata.get("render_modes", []):
            raise ValueError(f"Unsupported render_mode={self.render_mode}, must be one of {self.metadata.get('render_modes')}")
 

        self.valid_actions = [
            Action.HOLD,
            Action.LONG_OPEN0,
            Action.LONG_CLOSE0,
            Action.SHORT_OPEN0,
            Action.SHORT_CLOSE0
        ]

        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.df_market = build_market_features(self.df, rollover_hour_local=5, is_future=self.config.trading.is_future)

        # ---- NEW: choose obs feature columns by config (default raw for backward compat) ----
        mode = getattr(self.config.trading, "obs_feature_mode", "raw")
        if mode == "obs":
            self._OBS_FEATURES_MARKET = FEATURES_MARKET_OBS
            self._OBS_FEATURES_AGENT = FEATURES_AGENT_OBS
        else:
            self._OBS_FEATURES_MARKET = FEATURES_MARKET  # default = old behavior
            self._OBS_FEATURES_AGENT = FEATURES_AGENT  # default = old behavior

        # ---- constants ----
        self._F_MARKET = len(self._OBS_FEATURES_MARKET)
        self._F_AGENT  = len(self._OBS_FEATURES_AGENT)

        dfm = self.df_market  # DON'T astype the whole df (keeps day_id dtype stable)


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
        X_all      = dfm[self._OBS_FEATURES_MARKET].to_numpy(dtype=np.float32, copy=False)
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

        # Prebuild “full-day” tensors (zeros, then fill)
        # Shape: [num_days, DAY_LEN, F_MARKET]
        num_days = len(self._days)
        self._daily_X = np.zeros((num_days, self.DAY_LEN, self._F_MARKET), dtype=np.float32)
        self._daily_mask = np.zeros((num_days, self.DAY_LEN), dtype=np.float32)

        for di, (s, e) in enumerate(self._day_ranges):
            m_idx = minute_idx[s:e]        # futures: [0..344]
            msk   = mask_np[s:e]           # 0/1

            if self.config.trading.is_future:
                # futures：把 obs_ 全量写进去（非价量的 time 特征也要保留），
                # 价量类本身在 build_market_features 里就已按 mask 抹零
                self._daily_X[di, m_idx, :] = X_all[s:e, :]
                self._daily_mask[di, m_idx] = msk
            else:
                # fx：保持旧行为（无效分钟不写入 -> 仍为 0）
                rows = (msk >= 0.5)
                if rows.any():
                    self._daily_X[di, m_idx[rows], :] = X_all[s:e, :][rows]
                self._daily_mask[di, m_idx] = msk

        # --- STEP3: daily context / daily seq ---
        self._use_daily_context = bool(getattr(self.config.trading, "use_daily_context", False))
        self._use_daily_seq_7 = bool(getattr(self.config.trading, "use_daily_seq_7", False))

        self._F_DAILY_CTX = 0
        self._daily_ctx_raw = None
        self._daily_ctx_obs = None
        self._daily_seq7_raw = None
        self._daily_seq7_obs = None

        if self._use_daily_context or self._use_daily_seq_7:
            ctx_raw, ctx_obs, seq_raw, seq_obs, _summary = build_daily_context_and_seq(
                self.df_market,
                day_key_fn=self._day_key,
                days_order=self._days,
                day_id_col="day_id",
                mask_col="mask_t",
            )
            self._daily_ctx_raw = ctx_raw
            self._daily_ctx_obs = ctx_obs
            self._daily_seq7_raw = seq_raw
            self._daily_seq7_obs = seq_obs
            self._F_DAILY_CTX = ctx_raw.shape[1]


        obs_dict = {
            "market_seq": spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self._F_MARKET), dtype=np.float32),
            "agent_state": spaces.Box(low=-np.inf, high=np.inf, shape=(self._F_AGENT,), dtype=np.float32),
        }

        if self._use_daily_context:
            obs_dict["daily_context"] = spaces.Box(low=-np.inf, high=np.inf, shape=(self._F_DAILY_CTX,), dtype=np.float32)

        if self._use_daily_seq_7:
            obs_dict["daily_seq_7"] = spaces.Box(low=-np.inf, high=np.inf, shape=(DAILY_SEQ_LEN, 3), dtype=np.float32)

        self.observation_space = spaces.Dict(obs_dict)


        self.reset()

        if self.config.debug.debug_enabled:
            i = self.current_step
            print("df        :", self.df.index[i])
            print("df_market :", self.df_market.index[i])             # 如果这里不是同一个时间，就是 iloc 错位
            ts = self.df.index[i]
            print("df_m.loc  :", None if ts not in self.df_market.index else ts)
            print("len(df)=", len(self.df), "len(df_market)=", len(self.df_market))

            # 在 debug_enabled 分支里（reset 后已有 self._day_i 和 _day_ranges）
            s, e = self._day_ranges[self._day_i]
            sub = self.df_market.iloc[s:e]

            save_intraday_html(
                df_market=sub,
                title=f"{self.config.trading.currency_pair} {sub.index[0]}",
                out_path=f"/tmp/{self.config.trading.currency_pair}_{sub.index[0]}.html",
                start_pos=0,
                end_pos=self.DAY_LEN,          # 完整 345
                focus_ts=sub.index[0],         # 或者 focus_ts=self.df_market.index[self.current_step]
                agent_raw=None,
                agent_obs=None,
            )






    def _config(self, config_path):
        # Configuration management
        if config_path is None:
            raise ValueError("config_path is None")
        
        self.config = TradingConfig.from_yaml(config_path)

        # Validate config
        self.config.validate()

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
        self.take_profit_fired = 0

        # --- intraday day caches (for agent obs normalization) ---
        self._entries_used_today = 0
        self._day_start_realized_cum = self.user_accounts.realized_pnl

        # compute EOD absolute index for this day/session
        day_start_row = self._day_ranges[self._day_i][0]
        if self.config.trading.is_future:
            self._eod_idx = self._compute_end_idx_at_15(day_start_row, tz="Asia/Shanghai")
        else:
            self._eod_idx = self._day_ranges[self._day_i][1]  # end of day slice

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
        
        # gymnasium: stop stepping after either terminated OR truncated
        if self.terminated or self.truncated:
            return self._get_obs(), 0.0, self.terminated, self.truncated, {}

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
            self.truncated = False
            return self._get_obs(), 0.0, self.terminated, self.truncated, {}

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
            self.truncated = False
            return self._get_obs(), 0.0, self.terminated, self.truncated, {}

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
            # data exhausted => truncated
            self.terminated = False
            self.truncated = True
            # Update deltas once for this final transition
            self._update_step_deltas()
            return self._get_obs(), 0.0, self.terminated, self.truncated, self._get_info()

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

        self.take_profit_fired += self._apply_take_profits()

        # Update unrealized P&L
        self._update_unrealized_pnl()

        # Update metrics (equity/drawdown etc.)
        self.metrics.update(self.df_market.index[self.current_step])

        # Termination rules
        if self._should_terminated():
            # _should_terminated() will set terminated/truncated flags
            if not (self.terminated or self.truncated):
                # safety fallback: treat as truncated
                self.truncated = True

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

        return obs, reward, self.terminated, self.truncated, info


    def _sync_day_and_minute(self):
        ts = self.df_market.index[self.current_step]
        day_key = self._day_key(self.df_market.loc[ts, "day_id"])
        new_day_i = int(self._sid_to_dayi[day_key])

        # day change hook
        if not hasattr(self, "_day_i"):
            self._day_i = new_day_i
        elif new_day_i != self._day_i:
            self._day_i = new_day_i
            self._entries_used_today = 0
            self._day_start_realized_cum = self.user_accounts.realized_pnl

            day_start_row = self._day_ranges[self._day_i][0]
            if self.config.trading.is_future:
                self._eod_idx = self._compute_end_idx_at_15(day_start_row, tz="Asia/Shanghai")
            else:
                self._eod_idx = self._day_ranges[self._day_i][1]
        else:
            self._day_i = new_day_i

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
        # Check margin requirements
        if self._check_margin():
            # margin call => terminated
            self.terminated = True
            self.truncated = False
            return True
    
        metrics = self.metrics.get_metrics()
        # 检查风险限制（使用百分比形式）
        daily_lost_pct = decimal_to_float(metrics['current_day_lost_pct'] / Decimal('100.0'))
        drawdown_pct = decimal_to_float(metrics['current_drawdown_pct'] / Decimal('100.0')) 
        if daily_lost_pct > self.config.risk.daily_lost_ratio or drawdown_pct > self.config.risk.max_drawdown_ratio:
            self.logger.error(f"Terminated: Daily Loss {daily_lost_pct:.4f} > {self.config.risk.daily_lost_ratio} "
                           f"or Drawdown {drawdown_pct:.4f} > {self.config.risk.max_drawdown_ratio}")
            self.terminated = True
            self.truncated = False
            return True

        current_rrr = self.position_manager.calc_profit_factor()
        if self.config.risk.risk_reward_ratio_enable and current_rrr is not None and current_rrr < self.config.risk.risk_reward_ratio:
            self.logger.error(f"Terminated: RRR {current_rrr:.4f} < {self.config.risk.risk_reward_ratio}")
            self.terminated = True
            self.truncated = False
            return True



        # check if we run out of data
        if self.current_step >= self.end_idx:
            self.logger.error(f"Reached end_idx={self.end_idx}, start_idx={self.start_idx}, episode_length={self.config.training.episode_length}, current_step={self.current_step}. Episode done.")
            # time/data bound => truncated
            self.terminated = False
            self.truncated = True
            return True

        # or if we exceed max_episode_steps
        if self.config.training.max_episode_steps > 0 and self.episode_step_count >= self.config.training.max_episode_steps:
            self.logger.error(f"Reached max_episode_steps={self.config.training.max_episode_steps}. Episode done.")
            # time limit => truncated
            self.terminated = False
            self.truncated = True
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
            'take_profit_fired': self.take_profit_fired,
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
        prev_max = getattr(self, "max_equity", D(self.config.trading.initial_balance))

        # --- minutes to EOD (absolute index -> remaining minutes) ---
        eod_idx = int(getattr(self, "_eod_idx", self.end_idx))
        minutes_to_eod = max(0, eod_idx - int(self.current_step))

        # --- realized today ---
        day_start_realized = getattr(self, "_day_start_realized_cum", self.user_accounts.realized_pnl)
        realized_today_cash = self.user_accounts.realized_pnl - day_start_realized

        # --- R_cash scale (1R in cash) ---
        # Use entry_price if in position else current_price as ref
        ref_price = getattr(self, "current_price", self._last_valid_price)
        try:
            if (self.user_accounts.long_position > D0) or (self.user_accounts.short_position > D0):
                # if any position exists, approximate ref with current_price (stable enough for scaling)
                ref_price = getattr(self, "current_price", self._last_valid_price)
        except Exception:
            pass

        R_cash = D0
        if bool(getattr(self.config.trading, "stop_loss_enabled", False)):
            mode = getattr(self.config.trading, "stop_loss_mode", "pct")
            slv = getattr(self.config.trading, "stop_loss_value", D0)
            if slv is None:
                slv = D0
            if mode == "pct":
                sl_dist = ref_price * slv
            else:
                sl_dist = slv
            R_cash = sl_dist * self.config.trading.lot_size * self.config.trading.trade_lot

        # fallback if SL disabled / degenerate
        if R_cash <= D0:
            B0 = D(self.config.trading.initial_balance)
            R_cash = max(B0 * Decimal("0.001"), Decimal("1"))

        inp = AgentFeatureInput(
            long_positions=self.position_manager.long_positions,
            short_positions=self.position_manager.short_positions,
            current_step=int(self.current_step),
            current_price=getattr(self, "current_price", self._last_valid_price),
            lot_size=self.config.trading.lot_size,

            realized_pnl_step=getattr(self, "realized_step", D0),
            realized_pnl_cum=self.user_accounts.realized_pnl,
            fee_step=getattr(self, "fee_step", D0),
            fee_cum=self.broker_accounts.fee_income.get_balance(),

            cash_balance=self.user_accounts.cash_balance.get_balance(),
            used_margin=self.user_accounts.used_margin.get_balance(),

            prev_max_equity=prev_max,

            # intraday extras
            entries_used_today=int(getattr(self, "_entries_used_today", 0)),
            max_entries_per_day=int(getattr(self.config.trading, "max_entries_per_day", 1)),
            minutes_to_eod=int(minutes_to_eod),
            day_len=int(self.DAY_LEN),

            initial_balance=D(self.config.trading.initial_balance),
            realized_today_cash=realized_today_cash,
            R_cash=R_cash,
        )

        raw = compute_agent_features_raw(inp)
        obs = compute_agent_features_obs(inp, raw)

        # env caches (unchanged semantics)
        self.upnl = raw["upnl_t"]
        self.equity = raw["equity_t"]
        self.max_equity = raw["max_equity_t"]
        self.drawdown = raw["drawdown_t"]

        # vectors
        self._agent_state_raw_vec = agent_feature_vector(raw, FEATURES_AGENT)
        self._agent_state_obs_vec = agent_feature_vector(obs, FEATURES_AGENT_OBS)

        # debug dicts (labels never mismatch)
        self._agent_raw_debug = {k: float(decimal_to_float(raw.get(k, D0))) for k in FEATURES_AGENT}
        self._agent_obs_debug = {k: float(decimal_to_float(obs.get(k, D0))) for k in FEATURES_AGENT_OBS}

        mode = getattr(self.config.trading, "obs_feature_mode", "raw")
        self._agent_state_vec = self._agent_state_obs_vec if mode == "obs" else self._agent_state_raw_vec




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
        if equity < self.user_accounts.used_margin.get_balance():
            self.logger.info("Equity below margin requirement. Liquidating all positions.")
            self._empty_position(self.current_price, self.config.trading.spread, close_reason="MARGIN_CALL")
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
        # ---- intraday constraints ----
        if bool(getattr(self.config.trading, "intraday_single_position", True)):
            # already in any position => reject (no add, no simultaneous long/short)
            if (self.user_accounts.long_position > D0) or (self.user_accounts.short_position > D0):
                self.logger.warning("Intraday rule: cannot open while already in position (no add / no flip).")
                return ForexCode.ERROR_OPEN_POSITION

        max_entries = int(getattr(self.config.trading, "max_entries_per_day", 1))
        if max_entries > 0 and int(getattr(self, "_entries_used_today", 0)) >= max_entries:
            self.logger.warning("Intraday rule: hit max_entries_per_day, cannot open new position.")
            return ForexCode.ERROR_OPEN_POSITION

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

        sl = self._compute_stop_loss_price(ask_price, side="long")
        tp = self._compute_take_profit_price(entry_exec_price=ask_price, sl_exec_price=sl, side="long")

        new_position = Position(
            size=position_size,
            entry_price=ask_price,
            initial_margin=required_margin,
            open_step=self.current_step,
            stop_loss_price=sl,
            take_profit_price=tp,
        )


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
        self._entries_used_today = int(getattr(self, "_entries_used_today", 0)) + 1

        return ForexCode.SUCCESS


    def _long_close(self, price: Decimal, spread: Decimal, slot: int = None, close_reason: str | None = None):
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
            released_margin=released_margin,
            meta={
                "side": "long",
                "slot": slot,
                "reason": close_reason or "MANUAL",
            },
        )
        self.record_trade(trade_record)
        return ForexCode.SUCCESS
   

    def _short_open(self, price: Decimal, spread: Decimal, slot: int = None):
        """
        Executes a SHORT_OPEN action with manual rollback.
        """
        # ---- intraday constraints ----
        if bool(getattr(self.config.trading, "intraday_single_position", True)):
            # already in any position => reject (no add, no simultaneous long/short)
            if (self.user_accounts.long_position > D0) or (self.user_accounts.short_position > D0):
                self.logger.warning("Intraday rule: cannot open while already in position (no add / no flip).")
                return ForexCode.ERROR_OPEN_POSITION

        max_entries = int(getattr(self.config.trading, "max_entries_per_day", 1))
        if max_entries > 0 and int(getattr(self, "_entries_used_today", 0)) >= max_entries:
            self.logger.warning("Intraday rule: hit max_entries_per_day, cannot open new position.")
            return ForexCode.ERROR_OPEN_POSITION
                
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

        sl = self._compute_stop_loss_price(bid_price, side="short")
        tp = self._compute_take_profit_price(entry_exec_price=bid_price, sl_exec_price=sl, side="short")

        new_position = Position(
            size=position_size,
            entry_price=bid_price,
            initial_margin=required_margin,
            open_step=self.current_step,
            stop_loss_price=sl,
            take_profit_price=tp,
        )


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
        self._entries_used_today = int(getattr(self, "_entries_used_today", 0)) + 1

        return ForexCode.SUCCESS
   

    def _short_close(self, price: Decimal, spread: Decimal, slot: int = None, close_reason: str | None = None):
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
            released_margin=released_margin,
            meta={
                "side": "short",
                "slot": slot,
                "reason": close_reason or "MANUAL",
            },
        )
        self.record_trade(trade_record)
        return ForexCode.SUCCESS
    
    def _apply_take_profits(self) -> int:
        if not bool(getattr(self.config.trading, "take_profit_enabled", False)):
            return 0
        if float(self.df_market.iloc[self.current_step]["mask_t"]) < 0.5:
            return 0

        low, high = self._get_bar_low_high()
        if low is None or high is None:
            return 0

        spr = self.config.trading.spread
        fired = 0

        long_hits = []
        for slot, pos in enumerate(self.position_manager.long_positions):
            if pos is not None and pos.take_profit_price is not None and high >= pos.take_profit_price:
                long_hits.append((slot, pos, pos.take_profit_price))

        short_hits = []
        for slot, pos in enumerate(self.position_manager.short_positions):
            if pos is not None and pos.take_profit_price is not None and low <= pos.take_profit_price:
                short_hits.append((slot, pos, pos.take_profit_price))

        for slot, pos, tp in long_hits:
            self.logger.info(f"trigger takeprofit, long {pos}")
            self._long_close(price=tp + spr, spread=spr, slot=slot, close_reason="TAKE_PROFIT")
            fired += 1

        for slot, pos, tp in short_hits:
            self.logger.info(f"trigger takeprofit, short {pos}")
            self._short_close(price=tp - spr, spread=spr, slot=slot, close_reason="TAKE_PROFIT")
            fired += 1

        return fired


    def _apply_stop_losses(self) -> int:
        if self.config.trading.stop_loss_enabled == False:
            return 0
        if float(self.df_market.iloc[self.current_step]["mask_t"]) < 0.5:
            return 0

        low, high = self._get_bar_low_high()
        if low is None or high is None:
            return 0

        spr = self.config.trading.spread
        fired = 0

        long_hits = []
        for slot, pos in enumerate(self.position_manager.long_positions):
            if pos is not None and pos.stop_loss_price is not None and low <= pos.stop_loss_price:
                long_hits.append((slot, pos, pos.stop_loss_price))

        short_hits = []
        for slot, pos in enumerate(self.position_manager.short_positions):
            if pos is not None and pos.stop_loss_price is not None and high >= pos.stop_loss_price:
                short_hits.append((slot, pos, pos.stop_loss_price))

        for slot, pos, sl in long_hits:
            self.logger.warning(f"trigger stoploss, long {pos}")
            self._long_close(price=sl + spr, spread=spr, slot=slot, close_reason="STOP_LOSS")
            fired += 1

        for slot, pos, sl in short_hits:
            self.logger.warning(f"trigger stoploss, short {pos}")
            self._short_close(price=sl - spr, spread=spr, slot=slot, close_reason="STOP_LOSS")
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

    def _empty_position(self, price: Decimal, spread: Decimal, close_reason: str | None = None):
        # close all long by slot
        for slot in range(len(self.position_manager.long_positions)):
            if self.position_manager.long_positions[slot] is not None:
                self._long_close(price=price, spread=spread, slot=slot, close_reason=close_reason)

        for slot in range(len(self.position_manager.short_positions)):
            if self.position_manager.short_positions[slot] is not None:
                self._short_close(price=price, spread=spread, slot=slot, close_reason=close_reason)

        return ForexCode.SUCCESS



    def _get_obs(self):
        """
        market_seq: (window_size, F_MARKET_SELECTED)
        - 取最近 window_size 分钟的“已发生历史片段”
        - 当历史不足 window_size：右侧补 0
        """
        X_day = self._daily_X[self._day_i]  # (DAY_LEN, F)

        end = int(min(self.current_minute, self.DAY_LEN - 1))
        start = max(0, end - self.window_size + 1)

        window = X_day[start:end + 1, :]
        L = window.shape[0]

        if L < self.window_size:
            pad = np.zeros((self.window_size - L, self._F_MARKET), dtype=np.float32)
            market_seq = np.concatenate([window, pad], axis=0)
        else:
            market_seq = window.astype(np.float32, copy=False)

        agent_state = getattr(self, "_agent_state_vec", np.zeros((self._F_AGENT,), dtype=np.float32))

        if self.config.debug.debug_enabled:
            # 1) 校验 daily_X 与 df_market 对齐
            ts = self.df_market.index[self.current_step]
            mi = int(self.df_market.loc[ts, "minute_index_t"])
            row_df = self.df_market.loc[ts, self._OBS_FEATURES_MARKET].to_numpy(np.float32)
            row_X = self._daily_X[self._day_i, mi, :]
            if not np.allclose(row_df, row_X, atol=1e-6, rtol=0):
                raise RuntimeError(f"daily_X build mismatch at {ts} mi={mi}")

            # 2) 造 dfp：带 raw+obs（obs 窗口来自 agent）
            dfp = self._obs_market_df(market_seq)

            # 3) agent raw/obs dict（obs 预留接口）
            raw_vec = getattr(self, "_agent_state_raw_vec", None)
            obs_vec = getattr(self, "_agent_state_obs_vec", None)

            agent_raw = getattr(self, "_agent_raw_debug", None)
            if agent_raw is None and raw_vec is not None:
                agent_raw = {k: float(v) for k, v in zip(FEATURES_AGENT, raw_vec)}

            agent_obs = getattr(self, "_agent_obs_debug", None)
            if agent_obs is None and obs_vec is not None:
                agent_obs = {k: float(v) for k, v in zip(FEATURES_AGENT_OBS, obs_vec)}

            # 4) 只画 window_size（start..end）
            save_intraday_html(
                df_market=dfp,
                title=f"{self.config.trading.currency_pair} obs {self.df_market.index[self.current_step]}",
                out_path=f"/tmp/obs_{self.df_market.index[self.current_step]}.html",
                start_pos=start,
                end_pos=end + 1,
                focus_pos=end,
                agent_raw=agent_raw,
                agent_obs=agent_obs,
            )

        out = {"market_seq": market_seq, "agent_state": agent_state}

        mode = getattr(self.config.trading, "obs_feature_mode", "raw")
        use_obs = (mode == "obs")

        if self._use_daily_context:
            out["daily_context"] = (self._daily_ctx_obs[self._day_i] if use_obs else self._daily_ctx_raw[self._day_i]).astype(np.float32, copy=False)


        if self._use_daily_seq_7:
            out["daily_seq_7"] = (self._daily_seq7_obs[self._day_i] if use_obs else self._daily_seq7_raw[self._day_i]).astype(np.float32, copy=False)


        # Debug-only: assert all outputs are finite (NaN/Inf will break Dreamer-style training fast)
        if self.config.debug.debug_enabled:
            for k, v in out.items():
                if isinstance(v, np.ndarray):
                    ok = np.isfinite(v)
                    if not np.all(ok):
                        bad = np.where(~ok)
                        # show up to first 5 bad indices for quick定位
                        bad_idx = list(zip(*(b[:5] for b in bad)))
                        raise RuntimeError(
                            f"Non-finite values in obs[{k}] at step={self.current_step} "
                            f"ts={self.df_market.index[self.current_step]} bad_idx={bad_idx}"
                        )
        else:
            # --- NEW: harden against NaN/Inf (Dreamer hates them) ---
            out["market_seq"] = np.nan_to_num(out["market_seq"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            out["agent_state"] = np.nan_to_num(out["agent_state"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            if "daily_context" in out:
                out["daily_context"] = np.nan_to_num(out["daily_context"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            if "daily_seq_7" in out:
                out["daily_seq_7"] = np.nan_to_num(out["daily_seq_7"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        return out





    def _obs_market_df(self, market_seq: np.ndarray) -> pd.DataFrame:
        from gym_trading_env.utils.market_features import FEATURES_MARKET, FEATURES_MARKET_OBS

        s, e = self._day_ranges[self._day_i]
        sub = self.df_market.iloc[s:e]
        idx = sub.index

        if len(idx) != self.DAY_LEN:
            raise RuntimeError(f"day slice length != DAY_LEN: {len(idx)} vs {self.DAY_LEN}")

        # 画布：先把 raw 全量铺进去（用于对照），obs 用 NaN 先占位（未来分钟保持空白）
        cols = list(dict.fromkeys(list(FEATURES_MARKET) + list(FEATURES_MARKET_OBS)))
        dfp = pd.DataFrame(np.nan, index=idx, columns=cols, dtype=np.float32)

        # raw：直接从 df_market 取（整天 345）
        for c in FEATURES_MARKET:
            if c in sub.columns:
                dfp[c] = sub[c].to_numpy(dtype=np.float32, copy=False)

        # obs：把 agent “可见窗口”映射回当天对应分钟
        end = int(min(self.current_minute, self.DAY_LEN - 1))
        start = max(0, end - self.window_size + 1)
        L = end - start + 1  # 真实历史长度（<= window_size）

        # market_seq 的列顺序严格对应 self._OBS_FEATURES_MARKET
        obs_cols = list(self._OBS_FEATURES_MARKET)

        # 只把真实历史段写入；market_seq 右侧 pad 不写入未来分钟
        dfp.iloc[start:end + 1, dfp.columns.get_indexer(obs_cols)] = market_seq[:L, :].astype(np.float32, copy=False)

        # 辅助列：两套都补（方便 plot_intraday 正常工作）
        dfp["minute_index_t"] = np.arange(self.DAY_LEN, dtype=np.int32)
        dfp["mask_t"] = self._daily_mask[self._day_i].astype(np.float32)

        # 如果 sub 里已有 obs_minute_index_t / obs_mask_t，顺便带上（表格/调试更完整）
        if "obs_minute_index_t" in sub.columns:
            dfp["obs_minute_index_t"] = sub["obs_minute_index_t"].to_numpy(dtype=np.float32, copy=False)
        else:
            dfp["obs_minute_index_t"] = dfp["minute_index_t"].astype(np.float32)

        if "obs_mask_t" in sub.columns:
            dfp["obs_mask_t"] = sub["obs_mask_t"].to_numpy(dtype=np.float32, copy=False)
        else:
            # obs_mask_t 的语义：agent 可见历史段为 1，未来/空白为 0
            obs_mask = np.zeros((self.DAY_LEN,), dtype=np.float32)
            obs_mask[start:end + 1] = 1.0
            dfp["obs_mask_t"] = obs_mask

        return dfp

    def _compute_take_profit_price(
        self,
        entry_exec_price: Decimal,
        sl_exec_price: Decimal | None,
        side: str,
    ) -> Decimal | None:
        if not bool(getattr(self.config.trading, "take_profit_enabled", False)):
            return None

        mode = getattr(self.config.trading, "take_profit_mode", "rr")
        if mode != "rr":
            self.logger.warning(f"TakeProfit: unknown mode={mode}, ignored.")
            return None

        rr = getattr(self.config.trading, "take_profit_rr", None)
        if rr is None:
            return None
        rr = Decimal(rr)

        # RR needs a valid SL
        if sl_exec_price is None:
            return None

        if rr <= D0:
            return None

        if side == "long":
            R = entry_exec_price - sl_exec_price
            if R <= D0:
                return None
            return entry_exec_price + rr * R
        else:
            R = sl_exec_price - entry_exec_price
            if R <= D0:
                return None
            return entry_exec_price - rr * R


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



    def render(self):
        if self.render_mode != "human":
            return None
        if self.current_step % 200 != 0:
            return None
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



