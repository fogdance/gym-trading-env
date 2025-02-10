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

def current_balance_reward_function(env):

    # Convert to float with precision
    return float(decimal_to_float(env.user_accounts.balance.get_balance(), precision=2))

def fast_car_racing_likely_reward_function(env):
    """
    A more advanced combined reward function that:
      1) Gives step-wise PnL difference with scaling.
      2) Adds immediate reward for profitable close actions, and small penalty for losing close.
      3) Adds final statistics-based bonus ONLY if the episode ended naturally (not forced).
    """

    # ----------------------------
    #  Part A: Step-wise delta PnL with scaling
    # ----------------------------
    current_pnl = env.user_accounts.realized_pnl + env.user_accounts.unrealized_pnl
    delta_pnl = current_pnl - env.previous_total_pnl

    # Get the current equity (total balance in the account)
    equity = env._calculate_equity()

    # Scale the PnL based on the equity (this way, it's a relative reward)
    if equity > 0:
        scaled_reward = (delta_pnl / equity) * 100  # For example, reward as percentage of equity
    else:
        scaled_reward = 0  # Avoid division by zero

    # Update previous total PnL for the next step
    env.previous_total_pnl = current_pnl

    step_reward = float(scaled_reward)

    # ----------------------------
    #  Part B: Instant close reward
    # ----------------------------
    # Suppose we stored "env.just_closed_trade" in step() or in the close methods.
    # For example, if we track: env.just_closed_trade = (pnl) or None if no close
    # Let's do a simple logic:
    #   if user closed a trade this step:
    #       if pnl>0 => +1
    #       if pnl<0 => -0.5
    # We'll use a small scale to encourage profitable trades but not overshadow the main PnL reward

    close_reward = 0.0
    if getattr(env, "just_closed_trade", None) is not None:
        closed_pnl = env.just_closed_trade  # a Decimal or float representing the closed trade PnL
        if closed_pnl > 0:
            close_reward = 1.0
        elif closed_pnl < 0:
            close_reward = -0.5

    step_reward += close_reward

    # Clear the just_closed_trade flag for next step
    env.just_closed_trade = None

    # ----------------------------
    #  Part C: Final bonus if this is the last step and it's a natural termination
    # ----------------------------
    final_bonus = 0.0
    if env.terminated:
        # Check if forced_termination or not
        if not getattr(env, "forced_termination", False):
            # If it's a normal finish, give a win-rate based bonus
            final_bonus = _episode_end_bonus(env)
        else:
            # If forced termination, maybe 0 or even negative
            # final_bonus = -2.0  # optional
            final_bonus = 0.0

    step_reward += final_bonus

    # Debug info
    env.logger.debug(f"Prev PnL: {env.previous_total_pnl}, Current PnL: {current_pnl}, Delta: {delta_pnl},"
                     f" Step Reward(Scaled): {float(scaled_reward)}, Close Reward: {close_reward}, Final Bonus: {final_bonus}")

    # Update env.previous_total_pnl
    env.previous_total_pnl = current_pnl

    return float(step_reward)


def _episode_end_bonus(env):
    """
    Example final bonus: map win rate to [0..5].
    Only called if the episode ends 'naturally'.
    """
    total_trades = env.trade_record_manager.get_total_trades()
    winning_trades = env.trade_record_manager.get_winning_trades_count()
    if total_trades > 0:
        win_rate = winning_trades / total_trades
    else:
        win_rate = 0.0

    bonus = 5.0 * win_rate
    env.logger.debug(f"Episode End Bonus => total_trades={total_trades}, winning={winning_trades},"
                     f" win_rate={win_rate}, bonus={bonus}")
    return bonus



# Mapping of reward function names to actual functions
reward_functions = {
    'current_balance_reward_function': current_balance_reward_function,
    'total_pnl_reward_function': total_pnl_reward_function,
    'fast_car_racing_likely_reward_function': fast_car_racing_likely_reward_function,
}