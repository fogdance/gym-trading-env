# examples/example_usage.py

import gymnasium as gym
from gym_trading_env.envs.trading_env import CustomTradingEnv
from gym_trading_env.utils.data_processing import load_data
from gym_trading_env.rewards.reward_functions import reward_functions
from stable_baselines3.common.env_checker import check_env
from gym_trading_env.envs.action import Action

import pygame

def register_input():
    action = Action.HOLD  # 默认动作
    done = False
    
    while True:  # 等待有效输入
        event = pygame.event.wait()  # 阻塞等待事件
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                action = Action.LONG_OPEN
                break
            elif event.key == pygame.K_RIGHT:
                action = Action.SHORT_OPEN
                break
            elif event.key == pygame.K_UP:
                action = Action.POSITION_UP
                break
            elif event.key == pygame.K_DOWN:
                action = Action.POSITION_DOWN
                break
            elif event.key == pygame.K_SPACE:
                action = Action.HOLD
                break
            elif event.key == pygame.K_RETURN:
                action = Action.EMPTY
                break
            elif event.key == pygame.K_ESCAPE:
                done = True
                break
        elif event.type == pygame.QUIT:
            done = True
            break

    return action, done


def main():
    # Initialize environment with configuration
    env = CustomTradingEnv(config_path='data/config.yaml')

    # Check environment compatibility
    check_env(env, warn=True)
    
    obs, info = env.reset()
    
    done = False
    total_reward = 0.0
    
    while not done:
        action, cancel = register_input()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        done = terminated or truncated or cancel
    
    print(f'Total Reward: {total_reward:.2f}')
    env.close()

if __name__ == "__main__":
    main()
