# src/gym_trading_env/rewards/reward_functions.py

from decimal import Decimal
from gym_trading_env.utils.conversion import decimal_to_float


def total_pnl_reward_function(env):
    """
    Reward function based on total profit and loss (realized + unrealized P&L).
    
    Args:
        env (CustomTradingEnv): The trading environment instance.
    
    Returns:
        float: The reward calculated as the change in total P&L.
    """
    # Calculate total P&L (realized + unrealized)
    total_pnl = env.user_accounts.realized_pnl + env.user_accounts.unrealized_pnl

    # Calculate reward as the change in total P&L since the last step
    reward = total_pnl - env.previous_total_pnl

    # Log the relevant information for debugging
    env.logger.debug(f"Previous Total P&L: {env.previous_total_pnl}")
    env.logger.debug(f"Current Total P&L: {total_pnl}")
    env.logger.debug(f"Reward: {reward}")

    # Update previous total P&L for the next step
    env.previous_total_pnl = total_pnl

    # Convert to float with precision
    return float(decimal_to_float(reward, precision=2))

# Mapping of reward function names to actual functions
reward_functions = {
    'total_pnl_reward_function': total_pnl_reward_function,
}