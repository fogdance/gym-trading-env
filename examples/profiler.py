# examples/example_usage.py

from PyQt5.QtWidgets import QApplication, QWidget
import sys
# app = QApplication.instance()
# if not app:
#     # app = QApplication(['', '-platform', 'offscreen'])
#     app = QApplication([])
    
import gymnasium as gym
from gym_trading_env.envs.trading_env import CustomTradingEnv
from stable_baselines3.common.env_checker import check_env
import cProfile
import pstats
import time

def main():

    env = CustomTradingEnv(config_path='data/config.yaml')

    # Check environment compatibility
    # check_env(env, warn=True)
    
    obs, info = env.reset()
    
    done = False
    total_reward = 0.0
    
    # 创建 cProfile 对象
    profiler = cProfile.Profile()
    
    # 运行环境循环，收集性能数据
    profiler.enable()
    steps = 0
    max_steps = 1000  # 限制步数，避免过长运行
    
    while not done and steps < max_steps:
        action = env.action_space.sample()  # 随机动作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        done = terminated or truncated
        steps += 1
    
    profiler.disable()
    
    # 输出性能统计
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')  # 按累计时间排序
    stats.print_stats('step')  # 只显示与 step 相关的内容
    stats.print_stats(10)  # 前 10 个耗时函数
    stats.dump_stats('profile_results.prof')  # 保存结果文件，可用 snakeviz 可视化
    
    print(f'Total Reward: {total_reward:.2f}')
    print(f'Total Steps: {steps}')
    env.close()

if __name__ == "__main__":

    main()