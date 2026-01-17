# src/gym_trading_env/envs/trading_env.py

from decimal import Decimal, getcontext, ROUND_HALF_UP
import logging
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque
from typing import Tuple
import os, shutil
from pathlib import Path

from gym_trading_env.utils.rpc_protocol import TradeSignal
from gym_trading_env.utils.rpc_client import LanOrderClient
from gym_trading_env.config.settings import RPC_ORDER_CONFIG

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
from gym_trading_env.envs.action import Action, ForexCode, JsonlActionLogger
from gym_trading_env.envs.config import TradingConfig
from gym_trading_env.utils.decimal_util import D, D0, D1, D100, quantize_money, number_to_float
from gym_trading_env.utils.market_features import FEATURES_MARKET, FEATURES_MARKET_OBS
from gym_trading_env.utils.bar_source import CsvBarSource, JuejinBarSource
from gym_trading_env.utils.timebase import FEATURE_TZ as DEFAULT_TZ, ts_to_naive_str, yyyymmdd_int
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

CORE_LOG_ENV_KEYS = [
    "equity",
    "return_pct",
    "current_drawdown_pct",
    "max_drawdown_pct",
    "trades_opened",
    "trades_closed",
    "total_trades",
    "winning_trades",
    "win_rate",
    "opens_per_1000_steps",
    "fee_total",
    "fee_drag_ratio",
    "profit_factor",
    "expectancy",
    "invalid_action",
    "invalid_action_total",
    "invalid_action_ratio",
]



class CustomTradingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, df: pd.DataFrame = None, config_path: str = None, render_mode: str | None = None):
        super(CustomTradingEnv, self).__init__()

        self._config(config_path=config_path)
        self._data(df=df, config=self.config)

        # ---- LIVE 下单信号 client（默认关闭）----
        self._order_client = None

        if self._live_mode and RPC_ORDER_CONFIG.order_enabled:
            self._order_client = LanOrderClient(
                endpoint=RPC_ORDER_CONFIG.order_endpoint,
                token=RPC_ORDER_CONFIG.order_token,
                timeout_sec=RPC_ORDER_CONFIG.order_timeout_sec,
                logger=self.logger,
            )
            self.logger.info(f"LAN order client enabled: {RPC_ORDER_CONFIG.order_endpoint}")



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

        # ---- NEW: choose obs feature columns by config (default raw for backward compat) ----
        mode = getattr(self.config.trading, "obs_feature_mode", "raw")
        if mode == "obs":
            self._OBS_FEATURES_MARKET = FEATURES_MARKET_OBS
            self._OBS_FEATURES_AGENT = FEATURES_AGENT_OBS
        else:
            self._OBS_FEATURES_MARKET = FEATURES_MARKET  # default = old behavior
            self._OBS_FEATURES_AGENT = FEATURES_AGENT    # default = old behavior

        self._F_MARKET = len(self._OBS_FEATURES_MARKET)
        self._F_AGENT  = len(self._OBS_FEATURES_AGENT)

        # ---- sanity: store day_len should match env DAY_LEN policy ----
        if int(getattr(self.bar_source.store, "day_len", self.DAY_LEN)) != int(self.DAY_LEN):
            self.logger.warning(f"store.day_len={self.bar_source.store.day_len} != env.DAY_LEN={self.DAY_LEN} (check store build policy)")

        # ---- daily extras flags (store should already have them if enabled) ----
        self._use_daily_context = bool(getattr(self.config.trading, "use_daily_context", False))
        self._use_daily_seq_7 = bool(getattr(self.config.trading, "use_daily_seq_7", False))

        self._F_DAILY_CTX = 0
        if self._use_daily_context:
            if getattr(self.bar_source.store, "daily_ctx_raw", None) is None:
                raise RuntimeError("use_daily_context=True but store.daily_ctx_raw is None (store not built with daily ctx)")
            self._F_DAILY_CTX = int(self.bar_source.store.daily_ctx_raw.shape[1])

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
            i = int(self.current_step)

            ts_store = self.bar_source.store.index[i]
            ts_raw = self.bar_source.df_raw.index[i] if i < len(self.bar_source.df_raw) else None
            ts_mkt = self.bar_source.df_market.index[i] if i < len(self.bar_source.df_market) else None

            print("store     :", ts_store)
            print("df_raw    :", ts_raw)
            print("df_market :", ts_mkt)
            print("df_m.loc  :", ts_store if ts_store in self.bar_source.df_market.index else None)
            print("len(df)=", len(self.bar_source.df_raw), "len(df_market)=", len(self.bar_source.df_market))

            # 用 store.day_ranges + store.index 做“当日切片锚点”，避免 df_market iloc 错位导致画错日
            s, e = self.bar_source.store.day_ranges[self._day_i]
            idx_day = self.bar_source.store.index[s:e]

            dfm = self.bar_source.df_market
            sub = dfm.iloc[s:e]

            # 文件名避免 ":"（某些系统/工具不喜欢）
            ts0_str = str(idx_day[0]).replace(":", "-") if len(idx_day) > 0 else "NA"

            save_intraday_html(
                df_market=sub,
                title=f"{self.config.trading.currency_pair} {idx_day[0] if len(idx_day) > 0 else ts_store}",
                out_path=f"/tmp/{self.config.trading.currency_pair}/{self.config.trading.currency_pair}_{ts0_str}.html",
                start_pos=0,
                end_pos=self.DAY_LEN,
                focus_ts=(idx_day[0] if len(idx_day) > 0 else ts_store),
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

        abs_cfg = Path(config_path).expanduser().resolve()
        self.logger.info(f"config_path (absolute): {abs_cfg}")

        run_dir = os.environ.get("DREAMER_RUN_DIR")
        if run_dir:
            run_dir = Path(run_dir).expanduser().resolve()
            run_dir.mkdir(parents=True, exist_ok=True)

            # 自定义名字，避免跟 dreamer 的 config.yaml 混淆
            dst = run_dir / f"env_{abs_cfg.name}"
            try:
                if not dst.exists():                # 避免多进程重复覆盖
                    shutil.copy2(abs_cfg, dst)
                    self.logger.info(f"Backed up env config -> {dst}")
                else:
                    self.logger.info(f"Env config already exists, skip: {dst}")
            except Exception as e:
                self.logger.warning(f"Failed to backup env config: {e}")
        else:
            self.logger.warning("DREAMER_RUN_DIR not set; skip env config backup.")

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

        # LIVE 模式 & replay 相关
        self._live_mode = getattr(self.config.trading, "live_mode", False)
        self._live_replay_on_reset = bool(getattr(self.config.trading, "live_replay_on_reset", False))
        self._replaying = False
        self._action_logger = None

        # 默认 JSONL logger：优先 config.trading.live_action_log_dir，
        # 否则用 DREAMER_RUN_DIR/live_actions 或 /tmp/live_actions
        if self._live_mode:
            base_dir = getattr(self.config.trading, "live_action_log_dir", None)
            if base_dir is None:
                run_dir_env = os.environ.get("DREAMER_RUN_DIR", None)
                if run_dir_env:
                    base_dir = Path(run_dir_env) / "live_actions"
                else:
                    base_dir = "/tmp/live_actions"
            self._action_logger = JsonlActionLogger(base_dir=base_dir, logger=self.logger)

    def set_action_logger(self, logger):
        """
        替换默认的 JSONL action logger；

        logger 需要实现两个方法：
          - append(rec: dict)
          - load_for_day(symbol: str, trading_day: int) -> list[dict]
        """
        self._action_logger = logger


    def _data(self, df, config):
        # Initialize episode counters & bounds
        self.episode_step_count = 0
        self.start_idx = 0
        self.end_idx = 0

        src = getattr(config.training, "bar_source", "csv")

        if src == "csv":
            self.bar_source = CsvBarSource(config=config, df=df)
        elif src == "juejin":
            self.bar_source = JuejinBarSource(config=config, df=df)  # v1 会 NotImplemented
        else:
            raise ValueError(f"Unknown bar_source={src}")


        # Compatibility: keep your existing sufficiency check, but check df_market (not raw df)
        self._check_data_sufficiency(self.bar_source.df_market)

        # default to full dataset (reset() will pick start/end)
        self.end_idx = int(self.bar_source.store.n_rows)






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

        Live mode:
        - start_row MUST satisfy row_mask==1 (data exists).或来自 action 日志
        - if no bar yet for today's trading_day, block until first bar arrives.
        - default start_row = latest available bar for the trading_day.
        """
        self.logger.info("RESET env")
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)

        # 每次 reset 都重置 replay 标记
        self._replaying = False

        self.position_manager = PositionManager()
        self.broker_accounts = BrokerAccounts()
        self.trade_record_manager = TradeRecordManager()

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

        reward_class = reward_classes.get(self.config.training.reward_function, EquityDeltaReward)
        self.reward_function = reward_class(self)

        df_len = int(self.bar_source.store.n_rows)
        episode_len = self.config.training.episode_length

        # -------------------------------
        # Choose start_row
        # -------------------------------
        if self._live_mode:
            # today trading_day from bar_source (resolved at build)
            td_int = yyyymmdd_int(self.bar_source._trading_date)
            day_sel = np.flatnonzero(self.bar_source.store.row_trading_day == td_int)

            if day_sel.size == 0:
                raise RuntimeError(f"No rows for trading_day={td_int} in store window")

            # only rows where data exists
            valid_day = day_sel[self.bar_source.store.row_mask[day_sel] >= 0.5]

            # if no bar yet, wait until first bar arrives
            while valid_day.size == 0:
                self.logger.info("[LIVE reset] no bar yet, waiting first kline...")
                self.bar_source.wait_kline_block()
                valid_day = day_sel[self.bar_source.store.row_mask[day_sel] >= 0.5]

            # LIVE + replay: 尝试从 action 日志中找“第一条动作所在行”作为起点
            if self._live_replay_on_reset and self._action_logger is not None:
                actions = self._load_actions_for_trading_day(td_int)
                if actions:
                    first_step = min(int(r.get("step", 0)) for r in actions)
                    start_row = first_step
                    self.logger.info(f"[LIVE reset] replay enabled -> start_row from action log: {start_row}")
                else:
                    # 没有历史 action，从当天最后一根已有 bar 开始
                    start_row = int(valid_day.min())
                    self.logger.info(f"[LIVE reset] replay enabled but no actions; start_row={start_row} (last valid bar)")
            else:
                # 原有行为：从当天最后一根已有 bar 开始
                start_row = int(valid_day.max())

        else:
            # backtest/train: your original anchor policy
            start_policy = getattr(self.config.training, "start_clock", "future_night")
            candidate_rows = self._candidate_start_rows_by_clock(start_policy)

            if episode_len is not None:
                max_start = df_len - int(episode_len)
                candidate_rows = candidate_rows[candidate_rows <= max_start]

            if candidate_rows.size == 0:
                candidate_rows = np.flatnonzero(self.bar_source.store.row_mask >= 0.5)
                if episode_len is not None:
                    max_start = df_len - int(episode_len)
                    candidate_rows = candidate_rows[candidate_rows <= max_start]
                if candidate_rows.size == 0:
                    raise RuntimeError("No valid start rows found (after applying start_clock and episode_length).")

            if getattr(self.config.training, "randomize_start", True):
                start_row = int(self.np_random.choice(candidate_rows))
            else:
                start_row = int(candidate_rows[0])

        # --- Align all counters/indexes to chosen row ---
        self.current_step = int(start_row)
        ts0 = self.bar_source.store.index[self.current_step]

        self._day_i = int(self.bar_source.store.row_day_i[self.current_step])
        self._start_minute = int(self.bar_source.store.row_minute[self.current_step])
        self.current_minute = self._start_minute

        # Bound the episode if episode_length is provided
        self.start_idx = self.current_step
        if episode_len is not None:
            self.end_idx = min(df_len, self.start_idx + int(episode_len))
        else:
            self.end_idx = df_len

        # --- EpisodePolicy: should we truncate the episode at every session end? ---
        truncate_on_session_end = bool(self.config.training.episode_policy.truncate_on_session_end)

        # Futures: if truncate_on_session_end=True, clamp episode end to current session/day end.
        # If False (MC monthly), DO NOT clamp here; episode can span multiple sessions.
        if self.config.trading.is_future and self.config.trading.intraday_mode and truncate_on_session_end:
            eod_end = int(self.bar_source.store.day_ranges[self._day_i][1])  # half-open
            self.end_idx = min(self.end_idx, eod_end)


        end_ts = self.bar_source.store.index[self.end_idx - 1] if self.end_idx > self.start_idx else ts0
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

        # NEW: compute EOD absolute index for this day/session from store
        self._eod_idx = int(self.bar_source.store.day_ranges[self._day_i][1])

        # NEW: last valid price from store (not df_market)
        self._last_valid_price = D(self.bar_source.store.row_C[self.current_step])

        self._refresh_agent_state()

        # NEW: live + replay => 在 reset 后立刻按历史 action 重放，恢复状态
        if self._live_mode and self._live_replay_on_reset:
            self._replay_from_action_log()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info




    def _near_eod(self) -> bool:
        """
        SessionPolicy: define "near EOD" window inside current session/day.

        Default behavior (compatible with your old comment “14:59 及之后”):
        - near_eod_bars defaults to 2 => last 2 bars of the session.
        If last bar is 15:00, then 14:59 and 15:00 are "near EOD".
        """
        near_eod_bars = int(self.config.trading.session_policy.near_eod_bars)

        # current session/day last bar idx (inclusive)
        eod_end = int(self.bar_source.store.day_ranges[self._day_i][1])  # half-open
        last_bar_idx = eod_end - 1

        threshold = last_bar_idx - (near_eod_bars - 1)
        return int(self.current_step) >= int(threshold)



    def step(self, action):
        """
        Executes one time step within the environment。

        Args:
            action (int): The action to take (index into self.valid_actions)。

        Returns:
            Tuple: (observation, reward, terminated, truncated, info)
        """

        # gymnasium: stop stepping after either terminated OR truncated
        if self.terminated or self.truncated:
            info = self._get_info()
            return self._get_obs(), 0.0, self.terminated, self.truncated, info

        # 记录本次 step 开始时所处的行号（action 作用在这一 bar）
        step_before = int(self.current_step)

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
            info = self._get_info()
            return self._get_obs(), 0.0, self.terminated, self.truncated, info

        if self.config.debug.debug_enabled:
            # NEW: use store index (env runtime does not rely on df_market)
            self.logger.info(f"bob {self.bar_source.store.index[self.current_step]}, {action} -> {self.action}")

        sp = self.config.trading.session_policy

        block_open = bool(sp.block_open_near_eod)
        force_flatten = bool(sp.force_flatten_eod)
        forced_code = None
        if block_open and self.config.trading.intraday_mode:
            if self._near_eod():
                if self.action in (Action.LONG_OPEN0, Action.SHORT_OPEN0, Action.LONG_OPEN, Action.SHORT_OPEN,
                                        Action.LONG_OPEN1, Action.SHORT_OPEN1):
                    ts_now = self.bar_source.store.index[self.current_step]
                    self.logger.info(f"{ts_now} near_eod -> force {self.action} to HOLD")
                    self.action = Action.HOLD
                    forced_code = ForexCode.ERROR_BLOCKED_NEAR_EOD

        # --- Price / market-closed gate at CURRENT step (t) ---
        try:
            # NEW: read mask/price from store
            mask_now = float(self.bar_source.store.row_mask[self.current_step])
            market_open = (mask_now >= 0.5)

            if market_open:
                action_price = D(self.bar_source.store.row_C[self.current_step])
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
                f"(len={int(self.bar_source.store.n_rows)}): {e}"
            )
            self.terminated = True
            self.truncated = False
            info = self._get_info()
            return self._get_obs(), 0.0, self.terminated, self.truncated, info

        if forced_code is not None:
            market_code = forced_code
            
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


        # --- OPTIONAL: force flatten at EOD (default True) ---
        if force_flatten and self.config.trading.intraday_mode:
            if self._near_eod():
                try:
                    in_market_now = (self.user_accounts.long_position > D0) or (self.user_accounts.short_position > D0)
                except Exception:
                    in_market_now = False
                if in_market_now:
                    self.logger.info(f"time is 14:59, makrket will close, force empty all position.")
                    # 用当前 action_price 平（close 函数内部会用 spread 算 bid/ask）
                    self._empty_position(price=action_price, spread=self.config.trading.spread, close_reason="EOD")

        # Behavior counters
        try:
            in_market = (self.user_accounts.long_position > D0) or (self.user_accounts.short_position > D0)
        except Exception:
            in_market = False
        invalid_action = ((self.action_result == ForexCode.ERROR_HIT_MAX_POSITION)
                        or (self.action_result == ForexCode.ERROR_OPEN_POSITION)
                        or (self.action_result == ForexCode.ERROR_NO_POSITION_TO_CLOSE)
                        )

        self.metrics.on_step(self.action, self.action_result == ForexCode.SUCCESS, in_market=in_market, invalid_action=invalid_action)

        # ---------------------------------------------------------
        #   分界线：action 已经在 t 执行完了
        # ---------------------------------------------------------


        #
        # 推进到 t+1 → 设置 current_price → 先跑止损 → 再 update_unrealized → metrics.update
        #
        next_i = self._advance_next_step(live=self._live_mode)
        if next_i is None:
            # 1) 如果需要，强制平仓（避免带仓结束 episode）
            if bool(self.config.trading.session_policy.force_flatten_eod) and self.config.trading.intraday_mode:
                self._force_flatten_if_any("TRUNCATE_NO_NEXT_BAR")

            # 2) 用 last_valid_price 做一次 mark-to-market（可选，但建议）
            self.current_price = getattr(self, "_last_valid_price", D0)
            self._update_unrealized_pnl()

            # 3) 更新 deltas & agent_state（保证最后一步 info/obs 一致）
            self._update_step_deltas()
            self._refresh_agent_state()

            self.terminated = False
            self.truncated = True

            obs = self._get_obs()
            info = self._get_info()
            reward = self.reward_function(obs)  # 或者给 0，但更推荐一致结算

            # 在 episode 走到头时也记录一下最后一个 action
            if self._live_mode and (not getattr(self, "_replaying", False)):
                self._log_live_action(
                    step=step_before,
                    action_index=int(a),
                    action_enum=self.action,
                    action_result=self.action_result,
                )

            return obs, reward, self.terminated, self.truncated, info


        # --- Advance time to NEXT step (t+1) ---
        self.current_step = next_i
        self.episode_step_count += 1

        # Sync day/minute from store
        self._sync_day_and_minute()

        # Mark-to-market price at NEXT step
        mask_next = float(self.bar_source.store.row_mask[self.current_step])
        if mask_next < 0.5:
            self.current_price = self._last_valid_price
        else:
            self.current_price = D(self.bar_source.store.row_C[self.current_step])
            self._last_valid_price = self.current_price

        # 先止损（可能会自动平仓，改变仓位/保证金/现金）
        self.stop_loss_fired += self._apply_stop_losses()
        self.take_profit_fired += self._apply_take_profits()

        # Update unrealized P&L
        self._update_unrealized_pnl()

        # metrics update (use store index/day_id; env runtime does not rely on df_market)
        ts = self.bar_source.store.index[self.current_step]
        day_id = int(self.bar_source.store.row_day_id[self.current_step])
        self.metrics.update(ts, day_id=day_id)

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

        if self.config.debug.debug_enabled and (self.truncated or self.terminated):
            self.trade_record_manager.dump_to_json(f"output/trade_records_{self.current_step}.json")

        # 真实 live step 结束后，记录 action（replay 时不记）
        if self._live_mode and (not getattr(self, "_replaying", False)):
            self._log_live_action(
                step=step_before,
                action_index=int(a),
                action_enum=self.action,
                action_result=self.action_result,
            )

        return obs, reward, self.terminated, self.truncated, info

    def _advance_next_step(self, *, live: bool) -> int | None:
        """
        Live mode (fixed store index):
        - Only block when next bar's mask==0 (data not arrived yet).
        - If next bar already exists (mask==1), do NOT block (catch-up).
        """
        next_i = int(self.current_step) + 1

        # bounds: window/episode
        if next_i >= int(self.bar_source.store.n_rows):
            return None
        if next_i >= int(self.end_idx):
            return None

        if not live:
            return next_i

        #  mask_t is "data exists" by design (strict_reindex_futures_345 computes it before fillna)
        if float(self.bar_source.store.row_mask[next_i]) >= 0.5:
            return next_i

        #  otherwise block until the next bar arrives (mask flips to 1)
        while float(self.bar_source.store.row_mask[next_i]) < 0.5:

            self.bar_source.wait_kline_block()  # blocks until update/correction applied
            # store is updated in-place, so row_mask will eventually change

        return next_i


    def _rpc_send_after_execute(self, ts, action, result, price):
        # 只在 live 且非 replay 场景下真正发单
        if not getattr(self, "_live_mode", False):
            return
        if getattr(self, "_replaying", False):
            # 重放历史 action 时，不再发真实订单，只做本地状态恢复
            return
        if self._order_client is None:
            return

        eob_naive_str = ts_to_naive_str(ts, tz=DEFAULT_TZ)

        symbol = getattr(self.config.trading, "future_symbol", self.config.trading.currency_pair)

        action_index = int(self.valid_actions.index(action))  # 0..len-1

        sig = TradeSignal(
            signal_id=f"{symbol}|{eob_naive_str}|{action.name}",
            symbol=symbol,
            eob=eob_naive_str,
            action_index=action_index,
            action_name=action.name,
            volume=int(getattr(self.config.trading, "trade_lot", 1)),
            price=float(price),
            meta={
                "result": int(getattr(result, "value", -1)),
                "equity": float(self._calculate_equity()),
            },
        )
        
        result = self._order_client.send(sig)

        if result.get("success"):
            self.logger.info(f"[RPC] send success signal_id={sig.signal_id} action={sig.action_name} msg={result['msg']}")
        else:
            # 现在可以区分是网络错误还是被拒绝（如重复）
            if result["msg"] in ("duplicate", "queue_full"):
                self.logger.warning(f"[RPC] signal rejected: {result['msg']} signal_id={sig.signal_id}")
            else:
                self.logger.error(f"[RPC] send failed signal_id={sig.signal_id} reason={result.get('msg')} {result.get('error')}")


    def _log_live_action(self, step: int, action_index: int, action_enum: Action, action_result: ForexCode):
        """
        live 模式下，每次真实 env.step() 结束时调用，写一条 JSONL 记录。
        """
        if self._action_logger is None:
            return

        try:
            ts = self.bar_source.store.index[step]
            ts_str = ts_to_naive_str(ts, tz=DEFAULT_TZ)
            trading_day = int(self.bar_source.store.row_trading_day[step])
        except Exception as e:
            self.logger.error(f"failed to prepare live action log at step={step}: {e}")
            return

        symbol = getattr(self.config.trading, "future_symbol", self.config.trading.currency_pair)

        rec = {
            "ts": ts_str,
            "step": int(step),
            "trading_day": int(trading_day),
            "symbol": symbol,
            "action": int(action_index),
            "action_name": getattr(action_enum, "name", None),
            "action_result": int(getattr(action_result, "value", -1)),
        }

        try:
            self._action_logger.append(rec)
        except Exception as e:
            self.logger.error(f"failed to append live action log: {e}")


    def _load_actions_for_trading_day(self, trading_day: int):
        """
        从 action logger 读取某个交易日的全部动作记录，并做 env 侧强过滤：
        - trading_day 必须匹配（如果记录里有 trading_day 字段）
        - step 必须落在 store 里该 trading_day 的范围内
        - 若有 ts 字段，则必须与 store.index[step] 对齐（防止窗口漂移导致 step 映射错位）
        """
        if self._action_logger is None:
            return []

        symbol = getattr(self.config.trading, "future_symbol", self.config.trading.currency_pair)

        # 先算该 trading_day 在 store 里的 step 范围（硬约束）
        try:
            td = int(trading_day)
            day_sel = np.flatnonzero(self.bar_source.store.row_trading_day == td)
            if day_sel.size == 0:
                self.logger.warning(f"[LIVE replay] no rows in store for trading_day={td}, skip load actions")
                return []
            day_min_step = int(day_sel.min())
            day_max_step = int(day_sel.max())
        except Exception as e:
            self.logger.error(f"[LIVE replay] failed to compute day range for trading_day={trading_day}: {e}")
            return []

        # 读取 logger
        try:
            raw = list(self._action_logger.load_for_day(symbol=symbol, trading_day=int(trading_day)))
        except Exception as e:
            self.logger.error(f"failed to load actions for trading_day={trading_day}: {e}")
            return []

        if not raw:
            return []

        def _ts_matches(step_i: int, ts_str: str) -> bool:
            try:
                # 日志里 ts 是 ts_to_naive_str() 产物（本地时间 naive string）
                ts_log = pd.Timestamp(ts_str)
                ts_log = ts_log.tz_localize(DEFAULT_TZ)

                ts_store = self.bar_source.store.index[int(step_i)]
                # 允许 0~59 秒误差（字符串格式/对齐造成的小抖动）
                return abs((ts_store - ts_log).total_seconds()) < 60
            except Exception:
                # ts 解析失败就不拿 ts 当约束（不因日志脏就直接全挂）
                return True

        filtered = []
        for rec in raw:
            try:
                step_i = int(rec.get("step", -1))
            except Exception:
                continue

            # step 范围硬过滤
            if step_i < day_min_step or step_i > day_max_step:
                continue

            # trading_day 字段存在时必须匹配
            if "trading_day" in rec:
                try:
                    if int(rec.get("trading_day")) != int(trading_day):
                        continue
                except Exception:
                    continue

            # ts 对齐过滤（如果有 ts）
            ts_str = rec.get("ts", None)
            if ts_str:
                if not _ts_matches(step_i, ts_str):
                    continue

            filtered.append(rec)

        if not filtered:
            return []

        # 同一个 step 多条记录 -> 保留最后一条（按 ts 或输入顺序）
        # 先按 (step, ts) 排序，ts 为空时放最后
        def _sort_key(r):
            s = int(r.get("step", 0))
            t = r.get("ts", "")
            return (s, t)

        filtered.sort(key=_sort_key)

        dedup = {}
        for r in filtered:
            dedup[int(r.get("step", 0))] = r  # 覆盖 => 保留最后一条

        # 返回按 step 升序的列表
        return [dedup[k] for k in sorted(dedup.keys())]


    def _replay_from_action_log(self):
        """
        在 live reset 后，根据历史 action 序列重放一遍 env.step()，
        恢复到崩溃前的内部状态（position / ledger / metrics / agent_state / reward 内部状态等）。

        重要：多日运行可能存在“非当日记录 / step 映射错位 / 重复 step”。
        这里依赖 _load_actions_for_trading_day() 的强过滤与去重保证 replay 只吃当日有效轨迹。
        """
        td_int = yyyymmdd_int(self.bar_source._trading_date)
        actions_sorted = self._load_actions_for_trading_day(td_int)

        if not actions_sorted:
            self.logger.info("[LIVE replay] no valid actions for today; skip replay.")
            return

        self.logger.info(
            f"[LIVE replay] start replay {len(actions_sorted)} actions, "
            f"from step={int(actions_sorted[0].get('step', 0))}"
        )

        hold_idx = self.valid_actions.index(Action.HOLD)

        self._replaying = True
        try:
            for rec in actions_sorted:
                target_step = int(rec.get("step", 0))
                act_idx = int(rec.get("action", 0))

                # action index 越界直接跳过（防脏数据）
                if act_idx < 0 or act_idx >= len(self.valid_actions):
                    self.logger.warning(f"[LIVE replay] skip invalid action index={act_idx} at step={target_step}")
                    continue

                # 如果日志 step 比当前还小（比如 reset 起点没按 first_step），无法回退，直接跳过
                if int(self.current_step) > target_step:
                    self.logger.warning(
                        f"[LIVE replay] skip past action: current_step={int(self.current_step)} > target_step={target_step}"
                    )
                    continue

                # 防御性：如果 current_step 落后，用 HOLD 补齐
                while int(self.current_step) < target_step:
                    _, _, term, trunc, _ = self.step(hold_idx)
                    if term or trunc:
                        self.logger.error(
                            f"[LIVE replay] terminated/truncated while padding HOLD to target_step={target_step}, "
                            f"current_step={int(self.current_step)} term={term} trunc={trunc}"
                        )
                        return

                _, _, term, trunc, _ = self.step(act_idx)
                if term or trunc:
                    self.logger.error(
                        f"[LIVE replay] env terminated/truncated during replay at "
                        f"step={int(self.current_step)}, term={term}, trunc={trunc}"
                    )
                    return
        finally:
            self._replaying = False

        # 打印 replay 后的仓位状态
        try:
            ua = self.user_accounts
            long_pos = getattr(ua, "long_position", None)
            short_pos = getattr(ua, "short_position", None)
            eq = getattr(ua, "equity", lambda: None)()
            self.logger.info(
                f"[LIVE replay] finished at current_step={int(self.current_step)} | "
                f"long={long_pos}, short={short_pos}, equity={eq}"
            )
        except Exception as e:
            self.logger.warning(f"[LIVE replay] finished but failed to log positions: {e}")


    def _sync_day_and_minute(self):
        new_day_i = int(self.bar_source.store.row_day_i[self.current_step])

        if not hasattr(self, "_day_i"):
            self._day_i = new_day_i
        elif new_day_i != self._day_i:
            self._day_i = new_day_i
            self._entries_used_today = 0
            self._day_start_realized_cum = self.user_accounts.realized_pnl

            # NEW: day/session end from store
            self._eod_idx = int(self.bar_source.store.day_ranges[self._day_i][1])
        else:
            self._day_i = new_day_i

        self.current_minute = int(self.bar_source.store.row_minute[self.current_step])




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

    def _force_flatten_if_any(self, reason: str):
        try:
            in_market = (self.user_accounts.long_position > D0) or (self.user_accounts.short_position > D0)
        except Exception:
            in_market = False
        if in_market:
            px = getattr(self, "current_price", getattr(self, "_last_valid_price", D0))
            self._empty_position(price=px, spread=self.config.trading.spread, close_reason=reason)

    def _episode_last_bar_idx(self) -> int:
        """Return the last valid bar index for this episode (inclusive)."""
        # end_idx 是 half-open
        last_idx = int(self.end_idx) - 1

        if self.config.trading.intraday_mode:
            # _eod_idx 也是 half-open（当日/session 的 end）
            eod_idx = int(getattr(self, "_eod_idx", self.end_idx))
            last_idx = min(last_idx, eod_idx - 1)

        # 保险：不越界
        last_idx = max(0, min(last_idx, int(self.bar_source.store.n_rows) - 1))
        return last_idx


    def _should_terminated(self):
        """
        Episode termination logic is governed by EpisodePolicy.
        SessionPolicy (flatten EOD / block open near EOD) stays elsewhere.
        """
        truncate_on_session_end = bool(self.config.training.episode_policy.truncate_on_session_end)


        # --- 1) Episode boundary: session end (optional) ---
        if truncate_on_session_end and self.config.trading.intraday_mode:
            # End at the end of CURRENT session/day
            eod_end = int(self.bar_source.store.day_ranges[self._day_i][1])  # half-open
            last_bar_idx = eod_end - 1
            if int(self.current_step) >= int(last_bar_idx):
                self.logger.warning(
                    f"Reached session end. day_i={self._day_i} "
                    f"current_step={self.current_step} last_bar_idx={last_bar_idx} end_idx={self.end_idx}"
                )
                self._force_flatten_if_any("TRUNCATE_SESSION_END")
                self.terminated = False
                self.truncated = True
                return True

        # --- 2) Episode boundary: end_idx (dataset / episode_length) ---
        if int(self.current_step) >= int(self.end_idx) - 1:
            self.logger.warning(
                f"Reached end_idx boundary. start_idx={self.start_idx} end_idx={self.end_idx} "
                f"episode_length={self.config.training.episode_length} current_step={self.current_step}."
            )
            self._force_flatten_if_any("TRUNCATE_END_IDX")
            self.terminated = False
            self.truncated = True
            return True

        # --- 3) Hard cap on steps ---
        if self.config.training.max_episode_steps > 0 and self.episode_step_count >= self.config.training.max_episode_steps:
            self.logger.error(f"Reached max_episode_steps={self.config.training.max_episode_steps}. Episode done.")
            self._force_flatten_if_any("TRUNCATE_MAX_STEPS")
            self.terminated = False
            self.truncated = True
            return True

        # --- 4) Margin call => terminated ---
        if self._check_margin():
            self.terminated = True
            self.truncated = False
            return True

        # --- 5) Risk guardrails (still terminate, not truncate) ---
        metrics = self.metrics.get_metrics()
        daily_lost_pct = decimal_to_float(metrics['current_day_lost_pct'] / Decimal('100.0'))
        drawdown_pct = decimal_to_float(metrics['current_drawdown_pct'] / Decimal('100.0'))
        if daily_lost_pct > self.config.risk.daily_lost_ratio or drawdown_pct > self.config.risk.max_drawdown_ratio:
            self.logger.error(
                f"Terminated: Daily Loss {daily_lost_pct:.4f} > {self.config.risk.daily_lost_ratio} "
                f"or Drawdown {drawdown_pct:.4f} > {self.config.risk.max_drawdown_ratio}"
            )
            self.terminated = True
            self.truncated = False
            return True

        
        pf = self.metrics.get_metrics().get("profit_factor", None)   # float or None
        if self.config.risk.risk_reward_ratio_enable and pf is not None:
            thr = float(self.config.risk.risk_reward_ratio)
            if pf < thr:
                self.logger.error(f"Terminated: RRR {pf:.4f} < {self.config.risk.risk_reward_ratio}")

                self.terminated = True
                self.truncated = False
                return True

        
        return False



    def get_oracle_snapshot(self) -> dict:
        """
        Oracle snapshot for integration tests.
        - MUST be pure "data extraction": NO feature computation here.
        - Keys are stable; tests should only depend on this dict.
        """

        # --------- meta ----------
        i = int(getattr(self, "current_step", 0))
        ts = None
        try:
            ts = self.bar_source.store.index[i]
        except Exception:
            ts = None

        mode = getattr(self.config.trading, "obs_feature_mode", "raw")
        use_obs = (mode == "obs")

        # --------- market window (same logic as _get_obs, but no nan_to_num side effects) ----------
        end_i = i
        start_i = end_i - int(self.window_size) + 1
        X_all = self.bar_source.store.X_market_obs if use_obs else self.bar_source.store.X_market_raw

        if start_i >= 0:
            market_seq = X_all[start_i:end_i + 1, :].astype(np.float32, copy=False)
            pad_len = 0
        else:
            pad_len = -start_i
            window = X_all[0:end_i + 1, :].astype(np.float32, copy=False)
            pad = np.zeros((pad_len, X_all.shape[1]), dtype=np.float32)
            market_seq = np.concatenate([pad, window], axis=0)

        # --------- positions ----------
        pm = getattr(self, "position_manager", None)
        if pm is None:
            long_positions = []
            short_positions = []
        else:
            long_positions = getattr(pm, "long_positions", [])
            short_positions = getattr(pm, "short_positions", [])

        def _pos_to_dict(p):
            if p is None:
                return None
            if hasattr(p, "to_dict"):
                return p.to_dict()
            # fallback: minimal
            return {
                "size": str(getattr(p, "size", "")),
                "entry_price": str(getattr(p, "entry_price", "")),
                "initial_margin": str(getattr(p, "initial_margin", "")),
                "open_step": int(getattr(p, "open_step", 0)),
                "stop_loss_price": None if getattr(p, "stop_loss_price", None) is None else str(getattr(p, "stop_loss_price")),
                "take_profit_price": None if getattr(p, "take_profit_price", None) is None else str(getattr(p, "take_profit_price")),
            }

        # --------- accounting ----------
        ua = getattr(self, "user_accounts", None)
        ba = getattr(self, "broker_accounts", None)

        cash_balance = ua.cash_balance.get_balance() if ua is not None else Decimal("0")
        used_margin  = ua.used_margin.get_balance() if ua is not None else Decimal("0")
        realized_cum = ua.realized_pnl if ua is not None else Decimal("0")
        unrealized   = ua.unrealized_pnl if ua is not None else Decimal("0")
        fee_cum      = ba.fee_income.get_balance() if ba is not None else Decimal("0")

        realized_step = getattr(self, "realized_step", Decimal("0"))
        fee_step      = getattr(self, "fee_step", Decimal("0"))

        # --------- gates / derived inputs (match _refresh_agent_state semantics) ----------
        try:
            market_open = 1 if float(self.bar_source.store.row_mask[i]) >= 0.5 else 0
        except Exception:
            market_open = 0

        # minutes_to_eod: same as _refresh_agent_state
        eod_idx = int(getattr(self, "_eod_idx", getattr(self, "end_idx", i + 1)))
        minutes_to_eod = max(0, eod_idx - i)

        # entries
        entries_used_today = int(getattr(self, "_entries_used_today", 0))
        max_entries_per_day = int(getattr(self.config.trading, "max_entries_per_day", 1))
        if max_entries_per_day <= 0:
            max_entries_per_day = 1

        # --- have_long / have_short / is_flat ---
        try:
            have_long = bool(ua.long_position > Decimal("0"))
            have_short = bool(ua.short_position > Decimal("0"))
        except Exception:
            have_long = False
            have_short = False

        is_flat = not (have_long or have_short)

        # near_eod gating
        sp = self.config.trading.session_policy
        block_open = bool(getattr(sp, "block_open_near_eod", False))
        near_eod = False
        if self.config.trading.intraday_mode and block_open:
            try:
                near_eod = bool(self._near_eod())
            except Exception:
                near_eod = False

        entries_left = max(0, max_entries_per_day - max(0, entries_used_today))
        # --- base open permission ---
        base_can_open = (
            (market_open == 1)
            and is_flat
            and (entries_left > 0)
            and (not (self.config.trading.intraday_mode and block_open and near_eod))
        )

        can_long_open = 1 if base_can_open else 0
        can_short_open = 1 if base_can_open else 0

        # --- close permission: 按 side 拆开，避免 NO_POSITION_TO_CLOSE ---
        can_long_close = 1 if ((market_open == 1) and have_long) else 0
        can_short_close = 1 if ((market_open == 1) and have_short) else 0

        # last action_result_code
        ar = getattr(self, "action_result", None)
        action_result_code = int(getattr(ar, "value", 0))

        # current_price / lot_size
        current_price = getattr(self, "current_price", getattr(self, "_last_valid_price", Decimal("0")))
        lot_size = getattr(self.config.trading, "lot_size", Decimal("1"))

        # prev_max_equity
        prev_max_equity = getattr(self, "max_equity", Decimal(str(getattr(self.config.trading, "initial_balance", 0))))

        # realized_today_cash / R_cash: reuse cached values from _refresh_agent_state if exists
        day_start_realized = getattr(self, "_day_start_realized_cum", realized_cum)
        realized_today_cash = realized_cum - day_start_realized
        R_cash = getattr(self, "_R_cash_last", Decimal("0"))
        if R_cash == Decimal("0"):
            # fallback (kept simple; still “取数”优先)
            R_cash = Decimal("1")

        snap = {
            "meta": {
                "schema_version": 1,
                "step": i,
                "ts": ts,
                "mode": mode,
                "day_i": int(getattr(self, "_day_i", 0)),
                "start_idx": int(getattr(self, "start_idx", 0)),
                "end_idx": int(getattr(self, "end_idx", 0)),
                "window_size": int(getattr(self, "window_size", 0)),
                "day_len": int(getattr(self, "DAY_LEN", 0)),
            },
            "market": {
                "market_seq": market_seq,     # np.float32, shape=(window_size, F)
                "pad_len": int(pad_len),
                "features": list(self._OBS_FEATURES_MARKET),  # stable names for spec mapping
            },
            "positions": {
                "long": [ _pos_to_dict(p) for p in long_positions ],
                "short": [ _pos_to_dict(p) for p in short_positions ],
            },
            "accounting": {
                "cash_balance": cash_balance,
                "used_margin": used_margin,
                "realized_pnl_step": realized_step,
                "realized_pnl_cum": realized_cum,
                "unrealized_pnl": unrealized,
                "fee_step": fee_step,
                "fee_cum": fee_cum,
                "initial_balance": Decimal(str(getattr(self.config.trading, "initial_balance", "0"))),
            },
            "agent_input": {
                "current_step": i,
                "current_price": current_price,
                "lot_size": Decimal(str(lot_size)),
                "prev_max_equity": prev_max_equity,

                "entries_used_today": entries_used_today,
                "max_entries_per_day": max_entries_per_day,
                "minutes_to_eod": int(minutes_to_eod),
                "day_len": int(getattr(self, "DAY_LEN", 0)),

                "realized_today_cash": realized_today_cash,
                "R_cash": R_cash,

                "market_open": int(market_open),
                "can_long_open": int(can_long_open),
                "can_short_open": int(can_short_open),
                "can_long_close": int(can_long_close),
                "can_short_close": int(can_short_close),
                "action_result_code": int(action_result_code),
            },
            "daily": {},
        }

        # optional daily extras (only extraction)
        if getattr(self, "_use_daily_context", False):
            snap["daily"]["daily_context"] = (
                self.bar_source.store.daily_ctx_obs[int(getattr(self, "_day_i", 0))] if use_obs
                else self.bar_source.store.daily_ctx_raw[int(getattr(self, "_day_i", 0))]
            ).astype(np.float32, copy=False)

        if getattr(self, "_use_daily_seq_7", False):
            snap["daily"]["daily_seq_7"] = (
                self.bar_source.store.daily_seq7_obs[int(getattr(self, "_day_i", 0))] if use_obs
                else self.bar_source.store.daily_seq7_raw[int(getattr(self, "_day_i", 0))]
            ).astype(np.float32, copy=False)

        return snap


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


        # Core log/env metrics: fixed set, always present
        env_metrics = self.metrics.get_metrics()
        for k in CORE_LOG_ENV_KEYS:
            v = env_metrics.get(k, None)
            vv = np.nan if v is None else number_to_float(v)
            info[f'log/env/{k}'] = np.asarray(vv, dtype=np.float32).reshape(())


        info[f'log/env/is_truncated'] = np.bool_(self.truncated)
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

        # --- market_open (from store mask) ---
        try:
            market_open = 1 if float(self.bar_source.store.row_mask[int(self.current_step)]) >= 0.5 else 0
        except Exception:
            market_open = 0

        # --- position status (flat?) ---
        try:
            have_long = (self.user_accounts.long_position > D0)
            have_short = (self.user_accounts.short_position > D0)
        except Exception:
            have_long = False
            have_short = False

        is_flat = (not have_long) and (not have_short)

        # --- entries left ---
        used = int(getattr(self, "_entries_used_today", 0))
        max_e = int(getattr(self.config.trading, "max_entries_per_day", 1))
        if max_e <= 0:
            max_e = 1
        entries_left = max(0, max_e - max(0, used))

        # --- near_eod open block policy folded into can_open ---
        sp = self.config.trading.session_policy
        block_open = bool(sp.block_open_near_eod)
        near_eod = False
        if self.config.trading.intraday_mode and block_open:
            try:
                near_eod = bool(self._near_eod())
            except Exception:
                near_eod = False
        open_blocked = (self.config.trading.intraday_mode and block_open and near_eod)

        can_long_open  = (market_open == 1) and is_flat and (entries_left > 0) and (not open_blocked)
        can_short_open = (market_open == 1) and is_flat and (entries_left > 0) and (not open_blocked)

        can_long_close  = (market_open == 1) and have_long
        can_short_close = (market_open == 1) and have_short

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

        self._R_cash_last = R_cash
        self._minutes_to_eod_last = int(minutes_to_eod)

        # env 里 self.action_result 是 ForexCode 或 None
        ar = getattr(self, "action_result", None)
        action_result_code = int(getattr(ar, "value", 0))  # None -> 0 (当作 SUCCESS/初始态)

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

            # v2 gates
            market_open=int(market_open),
            can_long_open=1 if can_long_open else 0,
            can_short_open=1 if can_short_open else 0,
            can_long_close=1 if can_long_close else 0,
            can_short_close=1 if can_short_close else 0,

            action_result_code=action_result_code,
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



    
    # --- NEW: 根据时钟锚点生成候选起点行 ---
    def _candidate_start_rows_by_clock(self, start_clock: str) -> np.ndarray:
        """
        返回满足 start_clock 条件且 mask_t==1 的行号数组。
        现在统一由 store 维护（env 不再关心 index 本地化 / 交集逻辑）。
        """
        return self.bar_source.store.candidate_start_rows_by_clock(start_clock)




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
            return ForexCode.ERROR_HIT_DAY_MAX_OPEN

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


        ts = self.bar_source.store.index[self.current_step]
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

        if self._live_mode:
            self._rpc_send_after_execute(
                ts=self.bar_source.store.index[self.current_step],
                action=Action.LONG_OPEN0,
                result=ForexCode.SUCCESS,
                price=ask_price,
            )
        return ForexCode.SUCCESS


    def _long_close(self, price: Decimal, spread: Decimal, slot: int = None, close_reason: str | None = None):
        """
        Executes a LONG_CLOSE action atomically:
        1) quote (no mutation)
        2) ledger post (may fail)
        3) commit position removal (mutates only after ledger success)
        """
        bid_price = price - spread
        if self.user_accounts.long_position <= Decimal('0.0'):
            self.logger.warning("No long position to close.")
            return ForexCode.ERROR_NO_POSITION_TO_CLOSE

        # 1) quote (NO mutation)
        try:
            slot_i, q = self.position_manager.quote_close_long(
                closing_price=bid_price,
                lot_size=self.config.trading.lot_size,
                slot=slot,
            )
        except ValueError as e:
            self.logger.warning(f"Error quoting long close: {e}")
            return ForexCode.ERROR_NO_POSITION_TO_CLOSE

        # fee: round-turn means charge on close as well
        fee = Decimal('0') if not self.config.trading.is_round_turn else (self.config.trading.trading_fee_per_lot * q.closed_size)

        ts = self.bar_source.store.index[self.current_step]
        entry = JournalEntry(
            timestamp=ts,
            memo="LONG_CLOSE",
            postings=[
                Posting("user_margin", -q.released_margin),
                Posting("broker_pnl", -q.pnl),  # pnl>0 user wins => broker_pnl decreases
                Posting("broker_fee_income", +fee),
                Posting("user_cash", +(q.released_margin + q.pnl - fee)),
            ],
            meta={"side": "long", "slot": slot_i, "close_price": str(bid_price), "pnl": str(q.pnl)}
        )

        snap = self.ledger.snapshot()
        try:
            # 2) ledger post
            self._post_atomic(entry)

            # 3) commit position removal ONLY after ledger success
            self.position_manager.commit_close_long(slot_i, quote=q)

            # stats-only projection
            self.user_accounts.realize_pnl(q.pnl)

            self._assert_ledger_conservation()
        except (LedgerError, ValueError) as e:
            self.ledger.restore(snap)
            self.logger.error(f"LONG_CLOSE failed and rolled back (position NOT removed): {e}")
            self.terminated = True
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        self.last_close_position = {'pnl': q.pnl, 'margin': q.released_margin}

        trade_record = TradeRecord(
            timestamp=ts,
            operation_type=Action.LONG_CLOSE.name,
            position_size=q.closed_size,
            open_price=q.entry_price,
            close_price=bid_price,
            required_margin=Decimal('0'),
            fee=fee,
            balance=self.user_accounts.cash_balance.get_balance(),
            leverage=self.config.trading.leverage,
            free_margin=self._calculate_equity() - self.user_accounts.used_margin.get_balance(),
            pnl=q.pnl,
            closed_size=q.closed_size,
            released_margin=q.released_margin,
            meta={
                "side": "long",
                "slot": slot_i,
                "reason": close_reason or "MANUAL",
            },
        )
        self.record_trade(trade_record)

        if self._live_mode:
            self._rpc_send_after_execute(
                ts=self.bar_source.store.index[self.current_step],
                action=Action.LONG_CLOSE0,
                result=ForexCode.SUCCESS,
                price=bid_price,
            )
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
            return ForexCode.ERROR_HIT_DAY_MAX_OPEN
                
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


        ts = self.bar_source.store.index[self.current_step]
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

        if self._live_mode:
            self._rpc_send_after_execute(
                ts=self.bar_source.store.index[self.current_step],
                action=Action.SHORT_OPEN0,
                result=ForexCode.SUCCESS,
                price=bid_price,
            )
        return ForexCode.SUCCESS
   

    def _short_close(self, price: Decimal, spread: Decimal, slot: int = None, close_reason: str | None = None):
        """
        Executes a SHORT_CLOSE action atomically:
        1) quote (no mutation)
        2) ledger post (may fail)
        3) commit position removal (mutates only after ledger success)
        """
        ask_price = price + spread
        if self.user_accounts.short_position <= Decimal('0.0'):
            self.logger.warning("No short position to close.")
            return ForexCode.ERROR_NO_POSITION_TO_CLOSE

        # 1) quote (NO mutation)
        try:
            slot_i, q = self.position_manager.quote_close_short(
                closing_price=ask_price,
                lot_size=self.config.trading.lot_size,
                slot=slot,
            )
        except ValueError as e:
            self.logger.warning(f"Error quoting short close: {e}")
            return ForexCode.ERROR_NO_POSITION_TO_CLOSE

        fee = Decimal('0') if not self.config.trading.is_round_turn else (self.config.trading.trading_fee_per_lot * q.closed_size)

        ts = self.bar_source.store.index[self.current_step]
        entry = JournalEntry(
            timestamp=ts,
            memo="SHORT_CLOSE",
            postings=[
                Posting("user_margin", -q.released_margin),
                Posting("broker_pnl", -q.pnl),
                Posting("broker_fee_income", +fee),
                Posting("user_cash", +(q.released_margin + q.pnl - fee)),
            ],
            meta={"side": "short", "slot": slot_i, "close_price": str(ask_price), "pnl": str(q.pnl)}
        )

        snap = self.ledger.snapshot()
        try:
            # 2) ledger post
            self._post_atomic(entry)

            # 3) commit position removal ONLY after ledger success
            self.position_manager.commit_close_short(slot_i, quote=q)

            self.user_accounts.realize_pnl(q.pnl)

            self._assert_ledger_conservation()
        except (LedgerError, ValueError) as e:
            self.ledger.restore(snap)
            self.logger.error(f"SHORT_CLOSE failed and rolled back (position NOT removed): {e}")
            self.terminated = True
            return ForexCode.ERROR_NO_ENOUGH_MONEY

        self.last_close_position = {'pnl': q.pnl, 'margin': q.released_margin}

        trade_record = TradeRecord(
            timestamp=ts,
            operation_type=Action.SHORT_CLOSE.name,
            position_size=q.closed_size,
            open_price=q.entry_price,
            close_price=ask_price,
            required_margin=Decimal('0'),
            fee=fee,
            balance=self.user_accounts.cash_balance.get_balance(),
            leverage=self.config.trading.leverage,
            free_margin=self._calculate_equity() - self.user_accounts.used_margin.get_balance(),
            pnl=q.pnl,
            closed_size=q.closed_size,
            released_margin=q.released_margin,
            meta={
                "side": "short",
                "slot": slot_i,
                "reason": close_reason or "MANUAL",
            },
        )
        self.record_trade(trade_record)

        if self._live_mode:
            self._rpc_send_after_execute(
                ts=self.bar_source.store.index[self.current_step],
                action=Action.SHORT_CLOSE0,
                result=ForexCode.SUCCESS,
                price=ask_price,
            )
        return ForexCode.SUCCESS


    def _apply_take_profits(self) -> int:
        if not bool(getattr(self.config.trading, "take_profit_enabled", False)):
            return 0
        if float(self.bar_source.store.row_mask[self.current_step]) < 0.5:
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
        if float(self.bar_source.store.row_mask[self.current_step]) < 0.5:
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
        - 取全局 [t0-window_size+1, t0] 的数据（按 step 回看，跨天也允许）
        - 只有到达数据开头不够 window_size，才左侧补 0
        """
        end_i = int(self.current_step)
        start_i = end_i - self.window_size + 1

        mode = getattr(self.config.trading, "obs_feature_mode", "raw")
        use_obs = (mode == "obs")

        # NEW: 从 store 取底层数组（行序与 df_market 完全对齐）
        X_all = self.bar_source.store.X_market_obs if use_obs else self.bar_source.store.X_market_raw  # shape=(n_rows, F)

        if start_i >= 0:
            window = X_all[start_i:end_i + 1, :]  # view
            market_seq = window.astype(np.float32, copy=False)
            pad_len = 0
            L = self.window_size
        else:
            pad_len = -start_i
            window = X_all[0:end_i + 1, :]
            L = int(window.shape[0])
            pad = np.zeros((pad_len, X_all.shape[1]), dtype=np.float32)
            market_seq = np.concatenate([pad, window.astype(np.float32, copy=False)], axis=0)

        agent_state = getattr(self, "_agent_state_vec", np.zeros((self._F_AGENT,), dtype=np.float32))

        if self.config.debug.debug_enabled:
            ts = self.bar_source.store.index[end_i]

            # 1) 最后一行必须对齐当前 step 的底层行
            row_x = X_all[end_i, :].astype(np.float32, copy=False)
            if not np.allclose(market_seq[-1], row_x, atol=1e-6, rtol=0):
                raise RuntimeError(f"market_seq last row mismatch at step={end_i} ts={ts}")

            # 2) HTML：画的必须和 agent 输入一致（包括 padding）
            dfw = self._obs_window_df(end_i=end_i, pad_len=pad_len, market_seq=market_seq)

            ts_str = str(ts).replace(":", "-")
            save_intraday_html(
                df_market=dfw,
                title=f"{self.config.trading.currency_pair} obs {ts}",
                out_path=f"/tmp/{self.config.trading.currency_pair}/{self.config.trading.currency_pair}_{ts_str}.html",
                start_pos=0,
                end_pos=len(dfw),
                focus_pos=len(dfw) - 1,
                agent_raw=getattr(self, "_agent_raw_debug", None),
                agent_obs=getattr(self, "_agent_obs_debug", None),
            )


        out = {"market_seq": market_seq, "agent_state": agent_state}

        # daily_context / daily_seq_7：不再由 env 自己维护，改为 store 提供（语义不变）
        if self._use_daily_context:
            out["daily_context"] = (
                self.bar_source.store.daily_ctx_obs[self._day_i] if use_obs else self.bar_source.store.daily_ctx_raw[self._day_i]
            ).astype(np.float32, copy=False)

        if self._use_daily_seq_7:
            out["daily_seq_7"] = (
                self.bar_source.store.daily_seq7_obs[self._day_i] if use_obs else self.bar_source.store.daily_seq7_raw[self._day_i]
            ).astype(np.float32, copy=False)

        # Harden against NaN/Inf (不改变语义，只做数值安全)
        if self.config.debug.debug_enabled:
            for k, v in out.items():
                if isinstance(v, np.ndarray):
                    ok = np.isfinite(v)
                    if not np.all(ok):
                        bad = np.where(~ok)
                        bad_idx = list(zip(*(b[:5] for b in bad)))
                        raise RuntimeError(
                            f"Non-finite values in obs[{k}] at step={self.current_step} "
                            f"ts={self.bar_source.df_market.index[self.current_step]} bad_idx={bad_idx}"
                        )
        else:
            out["market_seq"] = np.nan_to_num(out["market_seq"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            out["agent_state"] = np.nan_to_num(out["agent_state"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            if "daily_context" in out:
                out["daily_context"] = np.nan_to_num(out["daily_context"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            if "daily_seq_7" in out:
                out["daily_seq_7"] = np.nan_to_num(out["daily_seq_7"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        return out


    def _obs_window_df(self, end_i: int, pad_len: int, market_seq: np.ndarray) -> pd.DataFrame:
        """
        构造一个用于 debug 绘图的 dfw：
        - 行数恒等于 window_size
        - 最后 L 行对应 store.index 的真实时间戳（避免 df_market iloc 错位）
        - 前 pad_len 行为 padding（obs_mask_t=0），但列值与 agent 输入一致（通常是0）
        """
        cols = list(dict.fromkeys(list(FEATURES_MARKET) + list(FEATURES_MARKET_OBS)))

        # 真实段长度
        L = int(self.window_size - int(pad_len))

        # 真实时间索引：从 store 取，作为唯一锚点
        if L > 0:
            idx_real = self.bar_source.store.index[end_i - L + 1:end_i + 1]
        else:
            idx_real = pd.DatetimeIndex([])

        # padding 时间索引：用 1min 倒推合成（仅用于可视化对齐）
        if pad_len > 0:
            ts0 = idx_real[0] if L > 0 else self.bar_source.store.index[end_i]
            idx_pad = pd.date_range(
                end=ts0 - pd.Timedelta(minutes=1),
                periods=int(pad_len),
                freq="1min",
                tz=getattr(ts0, "tz", None),
            )
            idx = idx_pad.append(idx_real)
        else:
            idx = idx_real

        dfw = pd.DataFrame(np.nan, index=idx, columns=cols, dtype=np.float32)

        # raw：只填真实段，且按 idx_real 用 reindex 对齐（避免 iloc 错位）
        if L > 0:
            start_i = end_i - L + 1
            dfm = self.bar_source.df_market
            sub = dfm.iloc[start_i:end_i+1]
            assert sub.index.equals(idx_real)
            for c in FEATURES_MARKET:
                if c in sub.columns and c in dfw.columns:
                    dfw.iloc[-L:, dfw.columns.get_loc(c)] = sub[c].to_numpy(dtype=np.float32, copy=False)


        # obs：严格等于 agent 输入（包含 pad）
        obs_cols = list(self._OBS_FEATURES_MARKET)
        dfw.iloc[:, dfw.columns.get_indexer(obs_cols)] = market_seq.astype(np.float32, copy=False)

        # mask：pad 段为 0，真实段为 1
        dfw["obs_mask_t"] = 0.0
        if L > 0:
            dfw.iloc[-L:, dfw.columns.get_loc("obs_mask_t")] = 1.0

        # 给 plot_intraday 一些常用辅助列（不强依赖真实 minute_index_t）
        dfw["minute_index_t"] = np.arange(self.window_size, dtype=np.int32)
        dfw["mask_t"] = dfw["obs_mask_t"].astype(np.float32)
        dfw["obs_minute_index_t"] = dfw["minute_index_t"].astype(np.float32)

        return dfw



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
        # store 已经保证缺失时用 0.0（或你 build_market_features 里保证列存在）
        low = D(self.bar_source.store.row_L[self.current_step])
        high = D(self.bar_source.store.row_H[self.current_step])
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



