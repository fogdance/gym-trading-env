# examples/example_usage.py

import gymnasium as gym
from gym_trading_env.envs.trading_env import CustomTradingEnv
from gym_trading_env.utils.data_processing import load_data
from gym_trading_env.rewards.reward_functions import reward_functions

def main():
    # Load data
    df = load_data('data/binance_BTCUSDT_1d.csv')
    
    # Define configuration
    config = {
        'initial_balance': 1000.0,
        'trading_fees': 0.001,
        'reward_function': 'basic_reward_function',
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
