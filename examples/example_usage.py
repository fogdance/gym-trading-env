# examples/example_usage.py

import gymnasium as gym
from gym_trading_env.envs.trading_env import CustomTradingEnv
from stable_baselines3.common.env_checker import check_env

def main():
    env = CustomTradingEnv(config_path='data/config.yaml')

    # Check environment compatibility
    check_env(env, warn=True)
    
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
