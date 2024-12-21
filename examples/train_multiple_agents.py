# examples/train_multiple_agents.py

import os
from typing import List, Tuple, Dict, Type

import pandas as pd
import numpy as np
import gymnasium as gym

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

# Replace these imports with your actual modules
from gym_trading_env.envs.trading_env import CustomTradingEnv
from gym_trading_env.utils.data_processing import load_data


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
        data_path: str,
        config: Dict,
        train_ratio: float = 0.8,
        train_timesteps: int = 50000,
        n_eval_episodes: int = 5,
        results_dir: str = "results",
        seed: int = 42,
    ):
        """
        Initialize the TradingRLTrainer.

        :param data_path: Path to the raw data file, e.g. 'data/USDJPY_Daily.csv'.
        :param config: Configuration dictionary for CustomTradingEnv (e.g., window_size, fees, etc.).
        :param train_ratio: The ratio of data to be used for training (remainder for testing).
        :param train_timesteps: Number of total timesteps for training each model.
        :param n_eval_episodes: Number of episodes to run in evaluation.
        :param results_dir: Directory path where models and results will be saved.
        :param seed: Random seed for reproducibility (if the algorithms support it).
        """
        self.data_path = data_path
        self.config = config
        self.train_ratio = train_ratio
        self.train_timesteps = train_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.results_dir = results_dir
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

        # Ensure the result directory exists
        os.makedirs(self.results_dir, exist_ok=True)

    def _load_and_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads raw data from file and splits it into train/test sets.

        :return: (train_df, test_df) dataframes.
        """
        df = load_data("USDJPY")  # Replace with direct read if needed: pd.read_csv(self.data_path)
        df = df.sort_values(by="Date").reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("Loaded dataset is empty.")

        split_index = int(len(df) * self.train_ratio)
        if split_index <= 0 or split_index >= len(df):
            raise ValueError("Invalid train_ratio leads to empty train or test set.")

        train_df = df.iloc[:split_index].reset_index(drop=True)
        test_df = df.iloc[split_index:].reset_index(drop=True)

        return train_df, test_df

    def _make_env(self, df: pd.DataFrame) -> CustomTradingEnv:
        """
        Creates a CustomTradingEnv instance for a given dataframe.

        :param df: The sliced dataframe (train or test).
        :return: CustomTradingEnv instance.
        """
        env = CustomTradingEnv(df=df, config=self.config)
        return env

    def _make_vec_env(self, df: pd.DataFrame) -> DummyVecEnv:
        """
        Creates a vectorized DummyVecEnv for off-policy algorithms.

        :param df: The sliced dataframe (train or test).
        :return: A DummyVecEnv wrapping the CustomTradingEnv.
        """
        def env_fn():
            return CustomTradingEnv(df=df, config=self.config)

        vec_env = DummyVecEnv([env_fn])
        return vec_env

    def train_and_evaluate(
        self,
        model_classes: List[Type[BaseAlgorithm]],
    ) -> List[Tuple[str, float, float]]:
        """
        Trains and evaluates each model class on the provided data.

        :param model_classes: A list of Stable-Baselines3 algorithm classes.
        :return: A list of tuples: (model_name, mean_reward, std_reward).
        """
        results = []

        for algo_cls in model_classes:
            algo_name = algo_cls.__name__
            print(f"Starting training for: {algo_name}")

            # Decide whether to use vectorized environment
            # Off-policy algorithms typically use a vec env for better data collection
            off_policy_algos = {DQN, DDPG, TD3, SAC}
            if algo_cls in off_policy_algos:
                train_env = self._make_vec_env(self.train_df)
                test_env = self._make_vec_env(self.test_df)
            else:
                # on-policy env can be used directly, or also in vec form if you prefer
                train_env = self._make_env(self.train_df)
                test_env = self._make_env(self.test_df)

            # Instantiate the model
            model = algo_cls(
                policy="MultiInputPolicy",
                env=train_env,
                verbose=1,
                seed=self.seed,
            )

            # Train the model
            model.learn(total_timesteps=self.train_timesteps, progress_bar=True, log_interval=None)

            # Save the model
            model_path = os.path.join(self.results_dir, f"{algo_name}_model.zip")
            model.save(model_path)
            print(f"Model saved to: {model_path}")

            # Evaluate the model on the test environment
            mean_reward, std_reward = evaluate_policy(
                model,
                test_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )

            print(
                f"{algo_name} -> Mean reward: {mean_reward:.2f}, Std: {std_reward:.2f}"
            )
            results.append((algo_name, mean_reward, std_reward))

        return results


def main():
    """
    Main entry point for training and evaluating multiple RL algorithms in a forex environment.
    Adjust config and model classes as needed for your specific use case.
    """

    config = {
        'use_dict_obs': True,
        "currency_pair": "USDJPY",
        "initial_balance": 10000.0,
        "trading_fees": 0.001,
        "spread": 0.0002,
        "leverage": 100,
        "lot_size": 100000,
        "trade_lot": 0.01,
        "max_long_position": 0.02,
        "max_short_position": 0.02,
        "reward_function": "basic_reward_function",
        "window_size": 20,
        "risk_free_rate": 0.0,
    }

    trainer = TradingRLTrainer(
        data_path="data/USDJPY_Daily.csv",
        config=config,
        train_ratio=0.8,
        train_timesteps=50000,
        n_eval_episodes=5,
        results_dir="results",
        seed=42
    )

    # You can adjust or reduce the list below to the algorithms you actually need.
    model_classes = [A2C, DQN, PPO]  # omit DDPG, TD3, SAC if action space is Discrete

    results = trainer.train_and_evaluate(model_classes)

    # Print a final comparison
    print("\n==== Comparison of Algorithm Performance ====")
    for algo_name, mean_r, std_r in results:
        print(f"{algo_name} : mean_reward={mean_r:.2f}, std={std_r:.2f}")


if __name__ == "__main__":
    main()
