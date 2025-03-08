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
                done = True
                break
            elif event.key == pygame.K_ESCAPE:
                done = True
                break
        elif event.type == pygame.QUIT:
            done = True
            break

    return action, done


def main():
    # Load data
    df = load_data('XAUUSD', interval = '5m')
    
    # Define configuration
    config = {
        'currency_pair': 'XAUUSD',
        'initial_balance': 1000.0,
        'trading_fees': 0.001,  # 0.1% trading fee
        'spread': 0.07,        # 2 pips spread
        'leverage': 100,         # 1:100 leverage
        'lot_size': 100,      # Standard lot size for EUR/USD
        'trade_lot': 0.01,       # Default trade size: 0.01 lot
        'max_long_position': 0.05,     # Maximum long position size: 0.02 lot
        'max_short_position': 0.05,    # Maximum short position size: 0.02 lot
        'reward_function': 'total_pnl_reward_function',
        'window_size': 60,
        'risk_free_rate': 0.0,
        'image_height': 96,
        'image_width': 96,
        'image_channels': 1,
    }
    
    # Initialize environment with configuration
    env = CustomTradingEnv(df=df, render_mode='human', config=config)

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
