# src/gym_trading_env/config.py

from dataclasses import dataclass, field, asdict, is_dataclass
import yaml
from typing import Optional, Dict, Any
from decimal import Decimal
from gym_trading_env.utils.decimal_util import D, D0, D1, D100, quantize_money


# Helper to convert float/int to Decimal
def _to_decimal(value):
    """Convert a numeric value to Decimal for precise financial calculations."""
    try:
        return D(value) if value is not None else None
    except Exception as e:
        print(f"[DEBUG] Cannot convert value={value} (type={type(value)}) to Decimal")
        raise e

TRADING_DECIMAL_KEYS = {
    "initial_balance", "trading_fee_per_lot", "spread", "leverage", "lot_size",
    "trade_lot", "max_long_position", "max_short_position",
    "stop_loss_value", "take_profit_rr",
}
RISK_DECIMAL_KEYS = {"risk_reward_ratio"}

def _cast_decimals(d: dict, keys: set[str]) -> dict:
    out = {}
    for k, v in d.items():
        if k in keys and isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = _to_decimal(v)
        else:
            out[k] = v
    return out

# ============================================================
# Policy blocks: strongly-typed configs
# ============================================================

@dataclass
class SessionPolicy:
    """
    SessionPolicy（会话内规则）：
    只描述“单个交易日/单个交易时段”内部的行为约束。

    典型用途：
    - near EOD 的定义：最后 N 根bar视为 near_eod
    - near_eod 时是否禁止开仓（防止尾盘开仓）
    - 是否在 EOD 强制平仓（force flatten）
    """
    near_eod_bars: int = 2
    """
    near_eod_bars=2 的语义（期货 1min 例子）：
    - session 最后一根bar=15:00
    - near_eod_bars=2 -> 14:59 和 15:00 都算 near_eod
    """

    block_open_near_eod: bool = True
    """near_eod 时禁止开新仓（open），一般建议 True。"""

    force_flatten_eod: bool = True
    """EOD 强制平仓。注意：这是 session 内规则，不等于 episode 截断。"""

    def validate(self) -> None:
        assert int(self.near_eod_bars) >= 1, "SessionPolicy.near_eod_bars must be >= 1"
        assert isinstance(self.block_open_near_eod, bool), "SessionPolicy.block_open_near_eod must be bool"
        assert isinstance(self.force_flatten_eod, bool), "SessionPolicy.force_flatten_eod must be bool"


@dataclass
class EpisodePolicy:
    """
    EpisodePolicy（episode 边界规则）：
    只描述“一个 episode 什么时候结束/截断”。

    典型用途：
    - 训练 intraday：每个 session 末尾 truncate（= 只跑 1 天）
    - MC 月度 warmup/eval：跨多个 session，不在 session 末尾 truncate（= 跑满月）
    """
    truncate_on_session_end: bool = True
    """True: 每个 session/day 末尾就 truncate；False: episode 可跨天跨 session。"""

    def validate(self) -> None:
        assert isinstance(self.truncate_on_session_end, bool), "EpisodePolicy.truncate_on_session_end must be bool"


# ============================================================
# Main config blocks
# ============================================================

@dataclass
class TradingParams:
    """Trading-specific parameters for CFD (Contract for Difference) trading environment.

    These parameters define the financial and operational aspects of trading a currency pair.
    All monetary values use Decimal for precision in financial calculations.
    """
    currency_pair: str = "EURUSD"
    """The currency pair to trade (e.g., 'EURUSD', 'XAUUSD')."""

    initial_balance: Decimal = _to_decimal(10000.0)
    """Initial account balance in the base currency (e.g., USD). Must be positive."""

    trading_fee_per_lot: Decimal = _to_decimal(5)
    """Fee charged per lot traded, in the base currency (e.g., $5 per lot)."""

    is_round_turn: bool = False
    """Fee structure: False for one-way fees (per trade), True for round-turn fees (open + close)."""

    spread: Decimal = _to_decimal(0.0002)
    """Bid-ask spread in pips (e.g., 0.0002 for 2 pips on EURUSD)."""

    leverage: Decimal = _to_decimal(100)
    """Leverage ratio (e.g., 100 for 1:100 leverage). Must be positive."""

    lot_size: Decimal = _to_decimal(100000)
    """Standard lot size in base currency units (e.g., 100,000 for forex)."""

    trade_lot: Decimal = _to_decimal(0.01)
    """Default trade size in lots (e.g., 0.01 for a micro lot). Must not exceed max positions."""

    max_long_position: Decimal = _to_decimal(0.1)
    """Maximum long position size in lots (e.g., 0.1 lot). Limits total open long trades."""

    max_short_position: Decimal = _to_decimal(0.1)
    """Maximum short position size in lots (e.g., 0.1 lot). Limits total open short trades."""

    data_path: str = ""
    data_interval: str = "5m"
    data_tz: Optional[str] = "Asia/Shanghai"  # "UTC" / "Asia/Shanghai"
    " csv的时区"

    down_thresh_5m: float = -0.0001
    up_thresh_5m: float = 0.0001
    down_thresh_15m: float = -0.0001
    up_thresh_15m: float = 0.0001
    down_thresh_1h: float = -0.0001
    up_thresh_1h: float = 0.0001
    """"ZigZag"""

    is_future: bool = True
    """"是否期货"""

    stop_loss_enabled: bool = False
    """"是否开启止损"""
    stop_loss_mode: str = "pct"  # "pct" | "abs"
    """"止损是百分比pct还是固定点差abs"""
    stop_loss_value: Decimal = _to_decimal(0.02)  # pct=0.02; abs=0.0010
    """"止损值"""

    obs_feature_mode: str = "raw"  # "raw" | "obs"
    """"obs 使用哪套特征（默认 raw：兼容旧测试；obs：用归一化 obs_ 列）"""

    use_daily_context: bool = False
    """"日内k线信息"""

    use_daily_seq_7: bool = False
    """"最近7天的k线信息"""

    max_entries_per_day: int = 3
    """"每天最大可开仓数量"""

    intraday_single_position: bool = True
    """"日内交易是否单向持仓"""

    take_profit_enabled: bool = False
    """"启用止盈"""

    take_profit_mode: str = "rr"  # currently only "rr"
    """"止盈模式"""

    take_profit_rr: Decimal = _to_decimal(2.0)  # e.g. 2R
    """"止盈值"""

    future_symbol: Optional[str] = None  # "DCE.jm2601"
    """"期货symbol, 交易所.主力合约"""

    trading_date: str = None  # "2025-12-28"
    """交易日期， 不填默认今天"""

    live_mode: bool = False
    """"实盘模式"""

    live_replay_on_reset: bool = False
    """"实盘模式下恢复agent状态"""

    intraday_mode: bool = True
    """True: session end 相关逻辑启用（配合 policy）"""

    invalid_action_punish: float = 0.0 # 0.02
    """重复无效动作的基础惩罚强度"""

    invalid_time_cost_total: float = 0.0 # 0.2
    """每次无效动作的“时间/操作成本”，按 episode 步数均摊（不改经济账）"""

    invalid_streak_cap = 10
    """streak 上限"""

    # -----------------------------
    # NEW: SessionPolicy (strong type)
    # -----------------------------
    session_policy: SessionPolicy = field(default_factory=SessionPolicy)
    """
    SessionPolicy（会话内规则）：
    - near_eod_bars / block_open_near_eod / force_flatten_eod 等
    """

    def validate(self):
        """Validate trading parameters to ensure they are feasible."""
        assert self.initial_balance > 0, "Initial balance must be positive"
        assert self.leverage > 0, "Leverage must be positive"
        assert self.trade_lot <= self.max_long_position, "Trade lot exceeds max long position"
        assert self.trade_lot <= self.max_short_position, "Trade lot exceeds max short position"

        allowed = {"pct", "abs"}
        assert self.stop_loss_mode in allowed, f"stop_loss_mode must be one of {allowed}"
        if self.stop_loss_enabled:
            assert self.stop_loss_value is not None, "stop_loss_value must not be None"
            assert self.stop_loss_value > 0, "stop_loss_value must be > 0"
            if self.stop_loss_mode == "pct":
                assert self.stop_loss_value < 1, "stop_loss_value (pct) must be < 1"

        allowed_obs = {"raw", "obs"}
        assert self.obs_feature_mode in allowed_obs, f"obs_feature_mode must be one of {allowed_obs}"

        assert int(self.max_entries_per_day) > 0, "max_entries_per_day must be > 0"

        allowed_tp = {"rr"}
        assert self.take_profit_mode in allowed_tp, f"take_profit_mode must be one of {allowed_tp}"

        if self.take_profit_enabled:
            # RR TP requires SL to define 1R
            assert self.stop_loss_enabled, "take_profit_enabled=True requires stop_loss_enabled=True (RR needs SL)"
            assert self.take_profit_rr is not None, "take_profit_rr must not be None"
            assert self.take_profit_rr > 0, "take_profit_rr must be > 0"

        # NEW: policy validation
        if self.session_policy is not None:
            self.session_policy.validate()


@dataclass
class RiskParams:
    """Risk management parameters for the trading environment.

    These parameters control loss limits and risk-adjusted metrics.
    """
    max_drawdown_ratio: float = 0.1
    """Maximum allowable drawdown as a ratio of initial balance (e.g., 0.1 = 10%). Range: [0, 1]."""

    daily_lost_ratio: float = 0.05
    """Maximum allowable daily loss as a ratio of balance (e.g., 0.05 = 5%). Range: [0, 1]."""

    risk_reward_ratio: Decimal = _to_decimal(0.0)
    """盈亏比"""

    risk_reward_ratio_enable: bool = False
    """启用盈亏比"""

    def validate(self):
        """Validate risk parameters to ensure they are within acceptable bounds."""
        assert 0 <= self.max_drawdown_ratio <= 1, "Max drawdown ratio must be between 0 and 1"
        assert 0 <= self.daily_lost_ratio <= 1, "Daily lost ratio must be between 0 and 1"


@dataclass
class TrainingParams:
    """Training and environment parameters for reinforcement learning.

    These settings control the RL agent's interaction with the trading environment.
    """
    reward_function: str = "total_pnl_reward_function"
    """Name of the reward function to use (e.g., 'total_pnl_reward_function'). Must match a defined function."""

    window_size: int = 20
    """Number of past time steps in the observation window. Must be positive."""

    max_episode_steps: int = 1_000_000
    """Maximum steps per episode to prevent infinite loops. Must be positive."""

    randomize_start: bool = True
    """Whether to randomize the starting index in the dataset for each episode."""

    episode_length: Optional[int] = None
    """Fixed length of each episode in steps, or None for variable length. Must be positive if set."""

    render_mode: str = "rgb_array"
    """Rendering mode: 'human' for visual display, 'rgb_array' for array output."""

    game_mode: bool = False
    """ 游戏模式 """

    start_clock: str = "future_night"
    """ 开盘时间 """

    bar_source: str = "csv"  # "csv" | "juejin"
    """ 数据来源 """

    # -----------------------------
    # NEW: EpisodePolicy (strong type)
    # -----------------------------
    episode_policy: EpisodePolicy = field(default_factory=EpisodePolicy)
    """
    EpisodePolicy（episode 边界规则）：
    - truncate_on_session_end 等
    """

    def validate(self):
        """Validate training parameters to ensure they are feasible."""
        assert self.window_size > 0, "Window size must be positive"
        assert self.max_episode_steps > 0, "Max episode steps must be positive"
        if self.episode_length is not None:
            assert self.episode_length > 0, "Episode length must be positive"
        allowed = {"random_9_or_21", "future_day", "future_night", "any"}
        assert self.start_clock in allowed, f"start_clock must be one of {allowed}"

        allowed_src = {"csv", "juejin"}
        assert self.bar_source in allowed_src, f"bar_source must be one of {allowed_src}"

        # NEW: policy validation
        if self.episode_policy is not None:
            self.episode_policy.validate()


@dataclass
class VisualizationParams:
    """Visualization and observation space parameters.

    These settings define how the environment's state is represented visually.
    """
    image_height: int = 256
    """Height of the observation image in pixels. Must be positive."""

    image_width: int = 256
    """Width of the observation image in pixels. Must be positive."""

    image_channels: int = 1
    """Number of color channels: 1 (grayscale), 3 (RGB), or 4 (RGBA)."""

    def validate(self):
        """Validate visualization parameters to ensure they are feasible."""
        assert self.image_height > 0, "Image height must be positive"
        assert self.image_width > 0, "Image width must be positive"
        assert self.image_channels in [1, 3, 4], "Image channels must be 1 (grayscale), 3 (RGB), or 4 (RGBA)"


@dataclass
class DebugParams:
    """Debugging and testing parameters."""
    log_level: str = "ERROR"
    debug_enabled: bool = False

    def validate(self):
        """Validate debug parameters."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert self.log_level.upper() in valid_levels, f"Log level must be one of {valid_levels}"


@dataclass
class TradingConfig:
    """Top-level configuration class for the trading environment."""

    trading: TradingParams = field(default_factory=TradingParams)
    """Trading-specific settings (e.g., currency pair, fees, leverage)."""

    risk: RiskParams = field(default_factory=RiskParams)
    """Risk management settings (e.g., max drawdown, daily loss limits)."""

    training: TrainingParams = field(default_factory=TrainingParams)
    """Training and environment settings for RL (e.g., reward function, episode length)."""

    visualization: VisualizationParams = field(default_factory=VisualizationParams)
    """Visualization settings (e.g., image dimensions)."""

    debug: DebugParams = field(default_factory=DebugParams)
    """Debugging and testing settings."""

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TradingConfig":
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        # ---- trading ----
        trading_dict = dict(config_dict.get("trading", {}) or {})
        sp_dict = dict(trading_dict.pop("session_policy", {}) or {})

        trading_kwargs = _cast_decimals(trading_dict, TRADING_DECIMAL_KEYS)

        trading_kwargs["session_policy"] = SessionPolicy(**sp_dict)

        # ---- risk ----
        risk_dict = dict(config_dict.get("risk", {}) or {})
        risk_kwargs    = _cast_decimals(risk_dict, RISK_DECIMAL_KEYS)


        # ---- training ----
        training_dict = dict(config_dict.get("training", {}) or {})
        ep_dict = dict(training_dict.pop("episode_policy", {}) or {})

        training_kwargs = dict(training_dict)
        training_kwargs["episode_policy"] = EpisodePolicy(**ep_dict)

        return cls(
            trading=TradingParams(**trading_kwargs),
            risk=RiskParams(**risk_kwargs),
            training=TrainingParams(**training_kwargs),
            visualization=VisualizationParams(**(config_dict.get("visualization", {}) or {})),
            debug=DebugParams(**(config_dict.get("debug", {}) or {})),
        )

    def validate(self):
        """Validate all nested configurations to ensure consistency."""
        self.trading.validate()
        self.risk.validate()
        self.training.validate()
        self.visualization.validate()
        self.debug.validate()

    def to_dict(self) -> dict:
        out = {}
        out.update(asdict(self.trading))
        out.update(asdict(self.risk))
        out.update(asdict(self.training))
        out.update(asdict(self.visualization))
        out.update(asdict(self.debug))
        return out