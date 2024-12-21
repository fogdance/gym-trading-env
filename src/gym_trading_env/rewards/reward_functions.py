# src/gym_trading_env/rewards/reward_functions.py

import numpy as np

def basic_reward_function(previous_total: float, current_total: float) -> float:
    """
    Computes the reward based on the change in portfolio value.

    Args:
        previous_total (float): Portfolio value before the action.
        current_total (float): Portfolio value after the action.

    Returns:
        float: The calculated reward.
    """
    return np.log(current_total / previous_total) if previous_total > 0 else 0

def sharpe_ratio_reward(previous_total: float, current_total: float, risk_free_rate: float = 0.0) -> float:
    """
    Computes the reward based on the Sharpe ratio.

    Args:
        previous_total (float): Portfolio value before the action.
        current_total (float): Portfolio value after the action.
        risk_free_rate (float, optional): Risk-free rate. Defaults to 0.0.

    Returns:
        float: The calculated reward.
    """
    return (current_total - previous_total) / previous_total - risk_free_rate

# Mapping of reward function names to actual functions
reward_functions = {
    'basic_reward_function': basic_reward_function,
    'sharpe_ratio_reward': sharpe_ratio_reward,
}
