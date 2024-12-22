# src/gym_trading_env/rewards/reward_functions.py

from decimal import Decimal
from gym_trading_env.utils.conversion import decimal_to_float

def basic_reward_function(env):
    """
    Basic reward function that calculates the change in total assets.
    
    Args:
        env (CustomTradingEnv): The trading environment instance.
    
    Returns:
        float: The reward calculated as the change in total assets.
    """
    # Calculate current equity
    current_equity = env._calculate_equity()

    # Calculate previous equity
    previous_equity = env.previous_balance + sum(
        (pos.entry_price - env.previous_price) * pos.size * env.lot_size 
        for pos in env.position_manager.long_positions
    ) + sum(
        (env.previous_price - pos.entry_price) * pos.size * env.lot_size 
        for pos in env.position_manager.short_positions
    )

    # Update previous balance and price for the next step
    env.previous_balance = env.balance
    env.previous_price = env.current_price

    # Reward is the change in equity
    reward = current_equity - previous_equity

    # Convert to float with precision
    return decimal_to_float(reward, precision=2)



def sharpe_ratio_reward(previous_total: Decimal, current_total: Decimal, risk_free_rate: Decimal = Decimal('0.0')) -> Decimal:
    """
    Computes the reward based on the Sharpe ratio.

    Args:
        previous_total (Decimal): Portfolio value before the action.
        current_total (Decimal): Portfolio value after the action.
        risk_free_rate (Decimal, optional): Risk-free rate. Defaults to Decimal('0.0').

    Returns:
        Decimal: The calculated reward.
    """
    return (current_total - previous_total) / previous_total - risk_free_rate

def max_drawdown_reward(previous_total: Decimal, current_total: Decimal) -> Decimal:
    """
    Computes the reward based on maximum drawdown.

    Args:
        previous_total (Decimal): Portfolio value before the action.
        current_total (Decimal): Portfolio value after the action.

    Returns:
        Decimal: The calculated reward.
    """
    drawdown = (previous_total - current_total) / previous_total
    return -drawdown if drawdown > Decimal('0') else Decimal('0.0')

# Mapping of reward function names to actual functions
reward_functions = {
    'basic_reward_function': basic_reward_function,
    'sharpe_ratio_reward': sharpe_ratio_reward,
    'max_drawdown_reward': max_drawdown_reward,
}
