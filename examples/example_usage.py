# examples/example_usage.py

import gymnasium as gym
from gym_trading_env.envs.trading_env import CustomTradingEnv
from gym_trading_env.utils.data_processing import load_data
from gym_trading_env.rewards.reward_functions import reward_functions

def main():
    # Load data
    df = load_data('USDJPY', interval = '1d')
    
    # Define configuration
    config = {
        'currency_pair': 'USDJPY',
        'initial_balance': 10000.0,
        'trading_fees': 0.001,  # 0.1% trading fee
        'spread': 0.0002,        # 2 pips spread
        'leverage': 100,         # 1:100 leverage
        'lot_size': 100000,      # Standard lot size for EUR/USD
        'trade_lot': 0.01,       # Default trade size: 0.01 lot
        'max_long_position': 0.02,     # Maximum long position size: 0.02 lot
        'max_short_position': 0.02,    # Maximum short position size: 0.02 lot
        'reward_function': 'total_pnl_reward_function',
        'window_size': 20,
        'risk_free_rate': 0.0
    }
    
    # Initialize environment with configuration
    env = CustomTradingEnv(df=df, config=config)
    obs, info = env.reset()
    
    done = False
    total_reward = 0.0
    
    while not done:
        action = env.action_space.sample()  # Replace with your agent's action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        done = terminated or truncated
    
    print(f'Total Reward: {total_reward:.2f}')
    env.close()

if __name__ == "__main__":
    main()
