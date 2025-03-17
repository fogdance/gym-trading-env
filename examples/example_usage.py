# examples/example_usage.py

import gymnasium as gym
from gym_trading_env.envs.trading_env import CustomTradingEnv
from stable_baselines3.common.env_checker import check_env
import numpy as np
import matplotlib.pyplot as plt

def run_and_analyze_rewards(env, num_episodes=1, steps_per_episode=1000):
    """
    运行环境，统计并分析 reward 分布
    Args:
        env: TradingEnv 实例
        num_episodes: 运行的回合数
        steps_per_episode: 每回合的最大步数
    """
    all_rewards = []  # 存储所有步的 reward
    episode_rewards = []  # 存储每回合的总 reward

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done and step_count < steps_per_episode:
            action = env.action_space.sample()  # 随机动作，可替换为你的策略
            obs, reward, terminated, truncated, info = env.step(action)
            all_rewards.append(reward)  # 记录每步 reward
            total_reward += reward
            step_count += 1
            # env.render()  # 可选，注释掉以加快运行
            done = terminated or truncated
        
        episode_rewards.append(total_reward)
        print(f'Episode {episode + 1} - Total Reward: {total_reward:.2f}, Steps: {step_count}')

    env.close()

    # 统计分析
    rewards_array = np.array(all_rewards)
    
    # 基本统计量
    print("\nReward Distribution Analysis:")
    print(f"Total Steps: {len(all_rewards)}")
    print(f"Mean Reward: {np.mean(rewards_array):.5f}")
    print(f"Std Dev: {np.std(rewards_array):.5f}")
    print(f"Min Reward: {np.min(rewards_array):.5f}")
    print(f"Max Reward: {np.max(rewards_array):.5f}")
    print(f"Positive Rewards: {np.sum(rewards_array > 0) / len(rewards_array):.2%}")
    print(f"Negative Rewards: {np.sum(rewards_array < 0) / len(rewards_array):.2%}")
    print(f"Zero Rewards: {np.sum(rewards_array == 0) / len(rewards_array):.2%}")
    
    # 分位数
    percentiles = [25, 50, 75, 95]
    percentile_values = np.percentile(rewards_array, percentiles)
    for p, v in zip(percentiles, percentile_values):
        print(f"{p}th Percentile: {v:.5f}")

    # 每回合统计
    print(f"\nEpisode Total Rewards: Mean = {np.mean(episode_rewards):.2f}, "
          f"Std = {np.std(episode_rewards):.2f}")

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(rewards_array, bins=50, edgecolor='black')
    plt.title('Reward Distribution')
    plt.xlabel('Reward Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    env = CustomTradingEnv(config_path='data/config.yaml')
    check_env(env, warn=True)

    run_and_analyze_rewards(env, num_episodes=5, steps_per_episode=1000)

