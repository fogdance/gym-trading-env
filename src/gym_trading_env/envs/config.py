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

    def validate(self):
        """Validate trading parameters to ensure they are feasible."""
        assert self.initial_balance > 0, "Initial balance must be positive"
        assert self.leverage > 0, "Leverage must be positive"
        assert self.trade_lot <= self.max_long_position, "Trade lot exceeds max long position"
        assert self.trade_lot <= self.max_short_position, "Trade lot exceeds max short position"

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

    risk_reward_ratio_enable: bool = False

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
    
    episode_length: Optional[int] = 500_000
    """Fixed length of each episode in steps, or None for variable length. Must be positive if set."""
    
    render_mode: str = "rgb_array"
    """Rendering mode: 'human' for visual display, 'rgb_array' for array output."""

    game_mode: bool = False

    def validate(self):
        """Validate training parameters to ensure they are feasible."""
        assert self.window_size > 0, "Window size must be positive"
        assert self.max_episode_steps > 0, "Max episode steps must be positive"
        if self.episode_length is not None:
            assert self.episode_length > 0, "Episode length must be positive"

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