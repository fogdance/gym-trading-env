# src/gym_trading_env/config.py

from dataclasses import dataclass, field
import yaml
from typing import Optional
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
    stop_loss_mode: str = "pct"          # "pct" | "abs"
    """"止损是百分比pct还是固定点差abs"""
    stop_loss_value: Decimal = _to_decimal(0.02)  # pct=0.02; abs=0.0010
    """"止损值"""

    obs_feature_mode: str = "raw"   # "raw" | "obs"
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

    take_profit_mode: str = "rr"              # currently only "rr"
    """"止盈模式"""

    take_profit_rr: Decimal = _to_decimal(2.0)  # e.g. 2R
    """"止盈值"""

    future_symbol: str = None #"DCE.jm2601"
    """"期货symbol, 交易所.主力合约"""

    trading_date: str = None # "2025-12-28"
    """交易日期， 不填默认今天"""

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
                
        # 新增校验
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

    bar_source: str = "csv"   # "csv" | "juejin"
    """ 数据来源 """


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
        """Load configuration from a YAML file.

        Args:
            yaml_path (str): Path to the YAML configuration file.

        Returns:
            TradingConfig: An instance with values from the YAML file or defaults.
        """
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Convert nested dictionaries to sub-configs
        return cls(
            trading=TradingParams(**{
                k: _to_decimal(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v 
                for k, v in config_dict.get("trading", {}).items()
            }),
            risk=RiskParams(**{
                k: _to_decimal(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v 
                for k, v in config_dict.get("risk", {}).items()
            }),
            training=TrainingParams(**config_dict.get("training", {})),
            visualization=VisualizationParams(**config_dict.get("visualization", {})),
            debug=DebugParams(**config_dict.get("debug", {}))
        )

    def validate(self):
        """Validate all nested configurations to ensure consistency."""
        self.trading.validate()
        self.risk.validate()
        self.training.validate()
        self.visualization.validate()

    def to_dict(self) -> dict:
        """Convert config to a flat dictionary for compatibility with legacy code.

        Returns:
            dict: A flat dictionary of all configuration parameters.
        """
        return {
            **self.trading.__dict__,
            **self.risk.__dict__,
            **self.training.__dict__,
            **self.visualization.__dict__,
            **self.debug.__dict__
        }