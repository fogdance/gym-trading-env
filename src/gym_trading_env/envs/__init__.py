from gymnasium.envs.registration import register

register(
    id='CustomTradingEnv-v0',
    entry_point='gym_trading_env.envs.trading_env:CustomTradingEnv',
    max_episode_steps=1000,
)
