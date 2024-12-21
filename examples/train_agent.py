# examples/train_agent.py

import gymnasium as gym
from gym_trading_env.envs.trading_env import CustomTradingEnv, Action
from gym_trading_env.utils.data_processing import load_data
from gym_trading_env.rewards.reward_functions import reward_functions
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

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
    
    # Check environment compatibility
    check_env(env, warn=True)
    
    # Initialize PPO model
    model = PPO('MlpPolicy', env, verbose=1)
    
    # Train the model
    model.learn(total_timesteps=10000)
    
    # Save the model
    model.save("ppo_custom_trading_env")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
