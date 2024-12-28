# src/gym_trading_env/utils/feature_engineering.py

import pandas as pd
import talib

class FeatureEngineer:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes technical indicators and adds them to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing price data.

        Returns:
            pd.DataFrame: DataFrame with additional technical indicator columns.
        """
        df = df.copy()
        
        # Simple Moving Average (SMA)
        df['SMA'] = talib.SMA(df['Close'], timeperiod=self.window_size)
        
        # Relative Strength Index (RSI)
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        
        # Moving Average Convergence Divergence (MACD)
        macd, macd_signal, macd_hist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        
        # Bollinger Bands
        upper_bb, middle_bb, lower_bb = talib.BBANDS(df['Close'], timeperiod=self.window_size, nbdevup=2, nbdevdn=2, matype=0)
        df['Upper_BB'] = upper_bb
        df['Middle_BB'] = middle_bb
        df['Lower_BB'] = lower_bb
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        return df
