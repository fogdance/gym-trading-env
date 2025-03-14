# examples/train_agent.py

import gymnasium as gym
from gym_trading_env.envs.trading_env import CustomTradingEnv, Action
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def main():
    env = CustomTradingEnv(config_path='data/config.yaml')
    
    # Check environment compatibility
    check_env(env, warn=True)
    
    # Initialize PPO model
    model = PPO('CnnPolicy', env, verbose=0)
    
    # Train the model
    model.learn(total_timesteps=10000)
    
    # Save the model
    model.save("ppo_custom_trading_env")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
