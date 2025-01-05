# src/gym_trading_env.envs.__init__.py

from gymnasium.envs.registration import register
import pandas as pd
from gym_trading_env.envs.trading_env import CustomTradingEnv
from gym_trading_env.utils.data_processing import load_data

df = load_data('EURUSD', interval = '5m')

config = {
    'currency_pair': 'EURUSD',
    'initial_balance': 1000.0,
    'trading_fees': 0.001,  # 0.1% trading fee
    'spread': 0.0002,        # 2 pips spread
    'leverage': 100,         # 1:100 leverage
    'lot_size': 100000,      # Standard lot size for EUR/USD
    'trade_lot': 0.01,       # Default trade size: 0.01 lot
    'max_long_position': 0.02,     # Maximum long position size: 0.02 lot
    'max_short_position': 0.02,    # Maximum short position size: 0.02 lot
    'reward_function': 'total_pnl_reward_function',
    'window_size': 60,
    'risk_free_rate': 0.0,
    'image_height': 96,
    'image_width': 192,
    'image_channels': 1,
}

# Factory function to create the environment with the necessary arguments
def make_custom_trading_env(df: pd.DataFrame, config: dict = None):
    def _env_creator():
        return CustomTradingEnv(df, config)
    
    return _env_creator

# Register your custom trading environment with the factory function
register(
    id='CustomTradingEnv-v0',  # The environment ID
    entry_point=make_custom_trading_env(df, config),  # Pass the factory function
    max_episode_steps=1000  # Set any other environment parameters
)

