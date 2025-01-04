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
