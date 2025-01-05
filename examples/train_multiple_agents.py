# examples/train_multiple_agents.py

import os
from typing import List, Tuple, Dict, Type
import uuid
from pathlib import Path

import pandas as pd
import numpy as np
import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback

# Replace these imports with your actual modules
from gym_trading_env.envs.trading_env import CustomTradingEnv
from gym_trading_env.utils.data_processing import load_data
from gym_trading_env.envs.callbacks import InfoLoggingCallback, EarlyStoppingCallback


class TradingRLTrainer:
    """
    A trainer class for multiple RL algorithms in a Forex environment.
    This class handles:
      1. Data loading and splitting into train/test sets.
      2. Environment creation (with optional vectorization).
      3. Training of different SB3 algorithms.
      4. Evaluation and result comparison.
    """

    def __init__(
        self,
        symbol: str,
        interval: str,
        config: Dict,
        train_ratio: float = 0.8,
        train_timesteps: int = 50000,
        n_eval_episodes: int = 5,
        results_dir: str = "results",
        tensorboard_log_dir: str = "tensorboard_logs",
        seed: int = 42,
    ):
        """
        Initialize the TradingRLTrainer.

        :param symbol: The trading symbol (e.g., "EURUSD").
        :param interval: The time interval for the data (e.g., "5m").
        :param config: Configuration dictionary for CustomTradingEnv.
        :param train_ratio: The ratio of data to be used for training (remainder for testing).
        :param train_timesteps: Number of total timesteps for training each model.
        :param n_eval_episodes: Number of episodes to run in evaluation.
        :param results_dir: Directory path where models and results will be saved.
        :param tensorboard_log_dir: Directory path for TensorBoard logs.
        :param seed: Random seed for reproducibility (if the algorithms support it).
        """
        self.symbol = symbol
        self.interval = interval
        self.config = config
        self.train_ratio = train_ratio
        self.train_timesteps = train_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.results_dir = results_dir
        self.tensorboard_log_dir = tensorboard_log_dir  # Store TensorBoard log path
        self.seed = seed

        # Load and split the dataset
        self.train_df, self.test_df = self._load_and_split_data()

        # Perform basic sanity checks
        if len(self.train_df) < self.config.get("window_size", 20):
            raise ValueError(
                "Train dataset has fewer rows than the required window_size."
            )
        if len(self.test_df) < self.config.get("window_size", 20):
            raise ValueError(
                "Test dataset has fewer rows than the required window_size."
            )

        # Create train and test environments (non-vectorized)
        self.train_env = self._make_env(self.train_df)
        self.test_env = self._make_env(self.test_df)

        # Check environments to ensure they comply with SB3 interface
        check_env(self.train_env, warn=True)
        check_env(self.test_env, warn=True)

        # Ensure the results and TensorBoard directories exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)

    def _load_and_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads raw data from file and splits it into train/test sets.

        :return: (train_df, test_df) dataframes.
        """
        df = load_data(self.symbol, self.interval)

        if len(df) == 0:
            raise ValueError("Loaded dataset is empty.")

        split_index = int(len(df) * self.train_ratio)
        if split_index <= 0 or split_index >= len(df):
            raise ValueError("Invalid train_ratio leads to empty train or test set.")

        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]

        return train_df, test_df

    def _make_env(self, df: pd.DataFrame) -> CustomTradingEnv:
        """
        Creates a CustomTradingEnv instance for a given dataframe.

        :param df: The sliced dataframe (train or test).
        :return: CustomTradingEnv instance.
        """
        env = CustomTradingEnv(df=df, config=self.config)
        check_env(env, warn=True)
        env = Monitor(env)
        return env

    def _make_vec_env(self, df: pd.DataFrame, n_envs: int = 4) -> SubprocVecEnv:
        """
        Creates a vectorized SubprocVecEnv for better parallelism.

        :param df: The sliced dataframe (train or test).
        :param n_envs: Number of parallel environments.
        :return: A SubprocVecEnv wrapping multiple CustomTradingEnv instances.
        """
        def env_fn():
            env = CustomTradingEnv(df=df, config=self.config)
            check_env(env, warn=True)
            env = Monitor(env)
            return env

        vec_env = SubprocVecEnv([env_fn for _ in range(n_envs)])
        return vec_env

    def train_and_evaluate(
        self,
        model_classes: List[Type[BaseAlgorithm]],
        n_envs: int = 4,  # New parameter for number of environments
    ) -> List[Tuple[str, float, float]]:
        """
        Trains and evaluates each model class on the provided data.

        :param model_classes: A list of Stable-Baselines3 algorithm classes.
        :param n_envs: Number of parallel environments.
        :return: A list of tuples: (model_name, mean_reward, std_reward).
        """
        results = []

        for algo_cls in model_classes:
            algo_name = algo_cls.__name__
            print(f"Starting training for: {algo_name}")

            # Determine whether to use a vectorized environment
            off_policy_algos = {DQN, DDPG, TD3, SAC}
            if algo_cls in off_policy_algos:
                train_env = self._make_vec_env(self.train_df, n_envs=n_envs)
                test_env = self._make_vec_env(self.test_df, n_envs=1)
            else:
                # On-policy algorithms can also use multiple environments
                train_env = self._make_vec_env(self.train_df, n_envs=n_envs)
                test_env = self._make_vec_env(self.test_df, n_envs=1)

            # Generate a unique session ID for each training session
            sess_id = str(uuid.uuid4())[:8]

            # Patch the tensorboard log directory for this session
            wandb.tensorboard.patch(root_logdir=self.tensorboard_log_dir)

            tensorboard_log_dir = os.path.join(self.tensorboard_log_dir, f"{algo_name}_{sess_id}")
            # Instantiate the model and add the tensorboard_log parameter
            model = algo_cls(
                policy="CnnPolicy",
                env=train_env,
                verbose=1,
                tensorboard_log=tensorboard_log_dir,  # Use the specific TensorBoard log directory
                seed=self.seed
            )

            print(f"Model is using device: {model.device}")
            print(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")

            # Initialize wandb for this training session
            run = wandb.init(
                project="forex-train",
                id=sess_id,
                sync_tensorboard=True,  # Sync TensorBoard logs to WandB
                monitor_gym=True,
                save_code=True,
                config={**{
                    "algorithm": algo_name,
                    "env": "CustomTradingEnv",
                    "total_timesteps": self.train_timesteps,
                }, **self.config},  # Merge both configs
                reinit=True  # Allow multiple runs in the same script
            )

            # Configure WandbCallback
            wandb_callback = WandbCallback(
                gradient_save_freq=100,
                verbose=2
            )

            checkpoint_callback = CheckpointCallback(
                save_freq=1000,
                save_path='checkpoints/',
                name_prefix=self.symbol + "_" + sess_id
            )

            early_stopping_callback = EarlyStoppingCallback(
                eval_env=test_env,
                eval_freq=100000,                # evaluate every {eval_freq} steps
                n_eval_episodes=self.n_eval_episodes,  # number of episodes to evaluate
                early_stopping_patience=5,     # stop training if performance hasn't improved in 10 evals
                verbose=1
            )


            info_logging_callback = InfoLoggingCallback(verbose=1, log_dir=tensorboard_log_dir) 

            # Combine all callbacks
            callback_list = CallbackList([checkpoint_callback, wandb_callback, info_logging_callback, early_stopping_callback])

            # Train the model
            model.learn(
                total_timesteps=self.train_timesteps,
                progress_bar=True,
                callback=callback_list,
                # log_interval=10,  # Set logging interval
            )

            # Save the model
            model_path = os.path.join(self.results_dir, f"{algo_name}_{sess_id}_model.zip")
            model.save(model_path)
            print(f"Model saved to: {model_path}")

            # Evaluate the model on the test environment
            mean_reward, std_reward = evaluate_policy(
                model,
                test_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                render=False,  # Disable rendering
            )

            print(f"{algo_name} -> Mean reward: {mean_reward:.2f}, Std: {std_reward:.2f}")
            results.append((algo_name, mean_reward, std_reward))

            # Cleanup environments
            train_env.close()
            test_env.close()

            # Finish the wandb run
            run.finish()

        return results


def main():
    """
    Main entry point for training and evaluating multiple RL algorithms in a Forex environment.
    Adjust config and model classes as needed for your specific use case.
    """
    symbol = "EURUSD"
    interval = "5m"
    config = {
        'currency_pair': 'EURUSD',
        'initial_balance': 1000.0,
        'trading_fees': 0.001,  # 0.1% trading fee
        'spread': 0.0002,        # 2 pips spread
        'leverage': 100,         # 1:100 leverage
        'lot_size': 100000,      # Standard lot size for EUR/USD
        'trade_lot': 0.01,       # Default trade size: 0.01 lot
        'max_long_position': 0.02,     # Maximum long position size: 0.02 lot
        'max_short_position': 0.02,    # Maximum short position size: 0.02 lot
        'reward_function': 'total_pnl_reward_function',
        'window_size': 60,
        'risk_free_rate': 0.0,
        'image_height': 96,
        'image_width': 192,
        'image_channels': 3,
    }

    trainer = TradingRLTrainer(
        symbol=symbol,
        interval=interval,
        config=config,
        train_ratio=0.99,
        train_timesteps=300_000, # default n_steps 2048
        n_eval_episodes=2,
        results_dir="models",  # Specify model directory
        tensorboard_log_dir="tensorboard_logs",  # Specify TensorBoard log directory
        seed=42
    )

    n_envs = 10

    # Adjust or reduce the list below to the algorithms you actually need.
    model_classes = [PPO]  # You can add more algorithms like A2C, DDPG, etc.

    results = trainer.train_and_evaluate(model_classes, n_envs=n_envs)

    # Print a final comparison
    print("\n==== Comparison of Algorithm Performance ====")
    for algo_name, mean_r, std_r in results:
        print(f"{algo_name} : mean_reward={mean_r:.2f}, std={std_r:.2f}")


if __name__ == "__main__":
    main()
