# src/gym_trading_env/utils/feature_engineering.py

import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    A class for computing technical indicators and engineering features.
    """
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes technical indicators for the given DataFrame.
        
        Args:
            df (pd.DataFrame): The input data containing at least 'Close' price.
        
        Returns:
            pd.DataFrame: DataFrame with additional technical indicators.
        """
        features = pd.DataFrame(index=df.index)
        features['close'] = df['Close']
        
        # Simple Moving Average
        features['SMA'] = df['Close'].rolling(window=self.window_size).mean()
        
        # Exponential Moving Average
        features['EMA'] = df['Close'].ewm(span=self.window_size, adjust=False).mean()
        
        # Relative Strength Index
        features['RSI'] = self._compute_rsi(df['Close'], window=14)
        
        # Moving Average Convergence Divergence
        features['MACD'], features['MACD_signal'], features['MACD_hist'] = self._compute_macd(df['Close'])
        
        # Fill NaN values
        features.fillna(0, inplace=True)
        
        return features
    
    def _compute_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """
        Computes the Relative Strength Index (RSI) for a given series.
        
        Args:
            series (pd.Series): The price series.
            window (int): The window size for RSI calculation.
        
        Returns:
            pd.Series: RSI values.
        """
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        gain = up.rolling(window=window).mean()
        loss = down.rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _compute_macd(self, series: pd.Series):
        """
        Computes the Moving Average Convergence Divergence (MACD) for a given series.
        
        Args:
            series (pd.Series): The price series.
        
        Returns:
            tuple: MACD line, Signal line, and Histogram.
        """
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist
