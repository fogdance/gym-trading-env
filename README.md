# Gym Forex Trading Env

A custom trading environment for OpenAI Gymnasium, designed for reinforcement learning (RL) research and applications in Forex trading.

## Core Concepts

This environment is designed specifically for Forex (foreign exchange) trading and aims to facilitate RL research in the context of financial markets. It offers a modular and extensible framework, where users can experiment with different trading strategies and models.

### Current Features:

- [x] **Complete Forex Trading Environment**: Supports Forex-specific parameters like spread, standard lot size, transaction fees, leverage, and default lot size.
- [x] **Account-based Asset Management**: Uses an accounting system to manage assets and track trades.
- [x] **Unit Testing**: Comprehensive unit tests for the Forex environment to ensure robustness and reliability.
- [x] **WandB & TensorBoard Integration**: Visualization tools like WandB and TensorBoard for tracking trading performance and model training.
- [x] **Customizable Reward Functions**: Design your own reward functions to align with various trading strategies.
- [x] **Multiple Data Support**: Easily switch between datasets to test model performance across different market conditions.
- [ ] **API Services**: External API services for strategy deployment and integration.
- [ ] **MT5 Integration**: Support for custom Expert Advisors (EAs) in MetaTrader 5 for strategy execution and API communication.

### Future Features:

- [ ] **TODO**

## Installation

### Prerequisites:

- Python 3.11 (Recommended)
- Conda environment (for managing dependencies)
- WandB (Weights and Biases) for tracking experiments and visualizing training performance.

### Steps to Install:

1. Clone the repository:
    ```bash
    git clone https://github.com/fogdance/gym_trading_env.git
    cd gym_trading_env
    ```

2. Set up the Conda environment (Python 3.11):
    ```bash
    conda create -n gym_trading_env python=3.11
    conda activate gym_trading_env
    ```

3. Install dependencies:
    ```bash
    poetry install
    ```

4. Run unit tests to verify installation:
    ```bash
    python -m unittest discover tests
    ```

5. Optionally, run a specific test:
    ```bash
    python -m unittest tests.test_position_manager.TestPositionManager
    ```
[x]窗口左pading
[x] 节假日
[x] 周五-周一
[x] 无夜盘
[x] 归一化
[x] window_size，而不是返回345分时图
[x] 默认带止损，可配置
[x]训练数据
[x]指标全部输出到html
[x] live卡住
[] live中间断掉，恢复持仓信息
[x] 15:00收盘后，疯狂reset env
[x] live加载ckpk后，好像还是乱开仓，使用train试试

优化
[x] obs，agent增加action的操作结果
[-] obs剩余开仓次数, 已有
[x] 日内高低波动幅度
[x] 开盘价到当前价格的波动幅度
[x] 备份dreamerv3，env共2份配置
[] 窗口60,增加到120
[] 模型参数50M，增加到100M
[x] 没仓位去平仓
    [x] 当前持仓的情况下还会去开仓
[x] 加1T ssd
[x] 期货品种焦煤换螺纹钢/甲醇

[x] 蒙特卡洛
[x] 限价->市价
[x] market/agent全raw/obs单元测试，集成测试——正确性

先把这 3 条曲线跑顺：
log/env/episode_return_pct（越高越好）

log/env/max_drawdown_pct（越低越好）

log/env/profit_factor（>1 且稳定）
for file in $(find  . -name "*.py"); do echo -e "\n\n### File: $file ###\n"; cat "$file"; done | xclip -sel clip
for file in $(find  tests -name "*.py"); do echo -e "\n\n### File: $file ###\n"; cat "$file"; done | xclip -sel clip


## How to Use

### Download Forex Data:
To download Forex data, run the following script:
```bash
python examples/download_forex_data.py
```

### Train a Model:
Once the data is ready, you can start training models using the provided script:
```bash
python examples/train_multiple_agents.py
```

### Example Usage:
You can also run an example usage script to understand how to interact with the environment:
```bash
python examples/example_usage.py
```

### Visualizing with TensorBoard:

```bash
tensorboard --logdir=tensorboard_logs
```
---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
