# Gym Trading Env

A custom trading environment for OpenAI Gymnasium, designed for reinforcement learning research and applications in trading.

## Features

- **Modular Design**: Separate modules for environment, data processing, feature engineering, portfolio management, and rendering.
- **Multiple Data Support**: Easily switch between different datasets to enhance model generalization.
- **Technical Indicators**: Built-in technical indicators like SMA, EMA, RSI, and MACD.
- **Customizable Rewards**: Define your own reward functions to suit different trading strategies.
- **Real-time Rendering**: Visualize trading performance with integrated rendering tools.

## Installation

```bash
git clone https://github.com/fogdance/gym_trading_env.git
cd gym_trading_env
pip install -e .

python -m unittest discover tests

python -m unittest tests.test_position_manager.TestPositionManager


python examples/example_usage.py
