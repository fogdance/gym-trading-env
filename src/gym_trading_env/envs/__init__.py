# src/gym_trading_env.envs/__init__.py
from gymnasium.envs.registration import register

register(
    id='CustomTradingEnv-v0',
    entry_point='gym_trading_env.envs.trading_env:CustomTradingEnv',
    kwargs={'config_path': None},
)