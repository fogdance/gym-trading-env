# src/gym_trading_env/utils/feature_engineering.py

import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes technical indicators for the given DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing price data.

        Returns:
            pd.DataFrame: DataFrame with additional technical indicator columns.
        """
        df = df.copy()
        # Example: Simple Moving Average (SMA)
        df['SMA'] = df['Close'].rolling(window=self.window_size).mean()
        # Example: Relative Strength Index (RSI)
        df['RSI'] = self._calculate_rsi(df['Close'], window=self.window_size)
        # Drop rows with NaN values
        df.dropna(inplace=True)
        return df

    def _calculate_rsi(self, series: pd.Series, window: int) -> pd.Series:
        """
        Calculates the Relative Strength Index (RSI).

        Args:
            series (pd.Series): Series of prices.
            window (int): Window size for RSI calculation.

        Returns:
            pd.Series: RSI values.
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
