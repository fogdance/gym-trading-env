# src/gym_trading_env/callbacks.py

from stable_baselines3.common.callbacks import BaseCallback
import wandb
from torch.utils.tensorboard import SummaryWriter
from decimal import Decimal

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

class InfoLoggingCallback(BaseCallback):
    """
    Custom callback to extract the info dictionary from the callback's locals
    and log it to WandB and TensorBoard.
    """

    def __init__(self, log_dir, verbose=0):
        super(InfoLoggingCallback, self).__init__(verbose)
        self.writer = SummaryWriter(log_dir=log_dir)
        # Optionally log key metrics to avoid performance issues with TensorBoard
        self.important_keys = ['balance', 'realized_pnl', 'broker_balance', 'fees_collected', 'used_margin']

    def _on_step(self) -> bool:
        """
        Executed after each training step to access the current step's info.
        """
        # Get the info from the current step
        infos = self.locals.get('infos', [])
        if infos:
            for info in infos:
                # Convert Decimal to float
                info_float = {k: float(v) if isinstance(v, Decimal) else v for k, v in info.items()}

                for key, value in info_float.items():
                    # Log to TensorBoard (optionally filter by important keys)
                    if key in self.important_keys:
                        self.writer.add_scalar(key, value, self.num_timesteps)

        return True  # Continue training

    def _on_training_end(self):
        # Close the SummaryWriter
        self.writer.close()
        print(f"Training ended. Total steps: {self.num_timesteps}.")

class EarlyStoppingCallback(BaseCallback):
    """
    Custom Early Stopping callback that stops training when the evaluation metric 
    does not improve for a specified number of steps.
    
    :param eval_env: The environment used for evaluation
    :param eval_freq: Frequency of evaluation (every how many steps to evaluate)
    :param n_eval_episodes: Number of episodes to run for each evaluation
    :param early_stopping_patience: The number of evaluations with no improvement 
                                     before stopping training
    :param verbose: Verbosity level for logging
    """
    def __init__(
        self, 
        eval_env, 
        eval_freq: int = 10000, 
        n_eval_episodes: int = 5, 
        early_stopping_patience: int = 10, 
        verbose: int = 1
    ):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.eval_env = eval_env  # The environment used for evaluation
        self.eval_freq = eval_freq  # Frequency of evaluation (steps)
        self.n_eval_episodes = n_eval_episodes  # Number of evaluation episodes
        self.early_stopping_patience = early_stopping_patience  # Patience for early stopping
        self.best_mean_reward = -np.inf  # Initialize the best mean reward (negative infinity to allow improvement)
        self.patience_counter = 0  # Counter to track number of evaluations without improvement

    def _on_step(self) -> bool:
        """
        This function is called at each step during training. It checks whether to stop training 
        based on the performance of the model in the evaluation environment.
        
        :return: True if training should continue, False if early stopping is triggered
        """
        if self.num_timesteps % self.eval_freq == 0:  # Perform evaluation every 'eval_freq' steps
            print(f"Step {self.num_timesteps}: Evaluating model...")
            # Evaluate the model on the evaluation environment
            mean_reward, _ = evaluate_policy(
                self.model, 
                self.eval_env, 
                n_eval_episodes=self.n_eval_episodes, 
                deterministic=True,  # Ensure deterministic actions during evaluation
                render=False  # Disable rendering for evaluation
            )
            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: Mean Reward = {mean_reward:.2f}")
            
            if mean_reward > self.best_mean_reward:  # Check if the new mean reward is better
                self.best_mean_reward = mean_reward  # Update the best mean reward
                self.patience_counter = 0  # Reset the patience counter
                if self.verbose > 0:
                    print("New best mean reward!")
            else:
                self.patience_counter += 1  # No improvement, increase patience counter
                if self.verbose > 0:
                    print(f"No improvement in mean reward for {self.patience_counter} evaluations.")
                
                if self.patience_counter >= self.early_stopping_patience:  # Early stopping condition met
                    if self.verbose > 0:
                        print("Early stopping triggered.")
                    return False  # Stop training if no improvement for a specified number of evaluations
        
        return True  # Continue training if early stopping is not triggered
