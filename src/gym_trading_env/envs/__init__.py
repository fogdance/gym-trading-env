# src/gym_trading_env.envs.__init__.py

from gymnasium.envs.registration import register
from gym_trading_env.envs.trading_env import CustomTradingEnv

def make_custom_trading_env():
    def _env_creator(config_path=None):
        return CustomTradingEnv(config_path=config_path)
    return _env_creator

# Register your custom trading environment with the factory function
register(
    id='CustomTradingEnv-v0',  # The environment ID
    entry_point=make_custom_trading_env(),  # Pass the factory function
)

