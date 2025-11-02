# examples/example_usage.py

import gymnasium as gym
from gym_trading_env.envs.trading_env import CustomTradingEnv
from stable_baselines3.common.env_checker import check_env
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ---- Add to examples/example_usage.py ----
import numpy as np
import pandas as pd
from gym_trading_env.utils.build_xt import FEATURES_MARKET, FEATURES_AGENT

def show_obs_readable(env, obs, rows=10, around=5):
    """
    人类可读地展示 observation：
    1) 概览：形状/可见分钟数/当前分钟
    2) agent_state：映射为命名向量
    3) market_seq：只展示“当前分钟附近”的若干行（或最早可见的若干行）

    Args:
        env: 传入当前 env（用于拿 current_minute）
        obs: env 返回的 observation dict
        rows: 市场特征表最多显示多少行
        around: 以 current_minute 为中心左右取多少行
    """
    market_seq = obs["market_seq"]          # (1440, F_MARKET)
    agent_vec  = obs["agent_state"]         # (F_AGENT,)

    # 0) 快速概览
    mask_col = FEATURES_MARKET.index("mask_t")
    visible_idx = np.where(market_seq[:, mask_col] != 0.0)[0]
    current_minute = getattr(env, "current_minute", (visible_idx.max() if visible_idx.size else 0))

    print("\n=== OBS OVERVIEW ===")
    print(f"market_seq shape: {market_seq.shape}  | agent_state shape: {agent_vec.shape}")
    print(f"current_minute: {current_minute}  | visible_rows: {len(visible_idx)} / {market_seq.shape[0]}")
    if visible_idx.size:
        print(f"visible range: [{visible_idx.min()} .. {visible_idx.max()}]")

    # 1) agent_state → 命名向量
    agent_map = {name: float(agent_vec[i]) for i, name in enumerate(FEATURES_AGENT)}
    agent_df = pd.DataFrame(agent_map, index=["value"]).T
    print("\n=== AGENT STATE (named) ===")
    # 只打印最关心的前若干行，可按需删掉这一行
    print(agent_df.round(6).to_string())

    # 2) market_seq（只看当前分钟附近）
    print("\n=== MARKET SEQ (around current_minute) ===")
    if visible_idx.size:
        lo = max(0, current_minute - around)
        hi = min(market_seq.shape[0] - 1, current_minute + around)
        # 仅选择可见且在窗口内的行
        window_idx = [i for i in range(lo, hi + 1) if market_seq[i, mask_col] != 0.0]

        # 如果当前还只揭示到很少的分钟，就从头部截取
        if not window_idx:
            window_idx = visible_idx.tolist()

        # 对显示的行数做统一裁剪
        if rows is not None:
            window_idx_display = window_idx[:rows]
        else:
            window_idx_display = window_idx

        if window_idx_display:
            data = market_seq[window_idx_display, :]  # 和 index 一样多的行
            mdf = pd.DataFrame(data, columns=FEATURES_MARKET, index=window_idx_display)

            # 重点列放前面（可按需调整）
            front = [
                "minute_index_t", "mask_t", "Close", "Open", "High", "Low",
                "cumVWAP_t", "ref_close_t", "session_high_t", "session_low_t", "bar_dir_t"
            ]
            ordered_cols = [c for c in front if c in mdf.columns] + [c for c in mdf.columns if c not in front]
            print(mdf[ordered_cols].round(6).to_string())
        else:
            print("(no visible minutes in window)")
    else:
        print("(no visible minutes yet)")


def demo_one_step_readable(env):
    """重置 → 打印初始 obs → HOLD 一步 → 再打印一次，验证『只推进一格』的可视化。"""
    obs0, info0 = env.reset()
    print("\n##### RESET #####")
    show_obs_readable(env, obs0, rows=12, around=3)

    # 执行一步 HOLD
    from gym_trading_env.envs.action import Action
    obs1, reward, terminated, truncated, info1 = env.step(Action.HOLD.value)
    print("\n##### AFTER ONE HOLD STEP #####")
    show_obs_readable(env, obs1, rows=12, around=3)
    print(f"\nreward={reward:.6f}, terminated={terminated}, truncated={truncated}")


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
        show_obs_readable(env, obs, rows=120, around=300)
        done = False
        total_reward = 0.0
        step_count = 0

        while not done and step_count < steps_per_episode:
            action = env.action_space.sample()  # 随机动作，可替换为你的策略
            obs, reward, terminated, truncated, info = env.step(action)
            show_obs_readable(env, obs, rows=12, around=3)
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

