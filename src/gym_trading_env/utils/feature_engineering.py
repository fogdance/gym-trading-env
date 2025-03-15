# src/gym_trading_env/utils/feature_engineering.py

import pandas as pd
import numpy as np
import talib
from zigzag import peak_valley_pivots
import matplotlib.pyplot as plt


class FeatureEngineer:

    def compute_features(self, df: pd.DataFrame, window_size: int = 20) -> pd.DataFrame:
        """
        Computes technical indicators and adds them to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing price data with 'Close' column.

        Returns:
            pd.DataFrame: DataFrame with additional technical indicator columns.
        """
        df = df.copy()
        
        # Simple Moving Average (SMA)
        df['SMA'] = talib.SMA(df['Close'], timeperiod=window_size)
        
        # Relative Strength Index (RSI)
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        
        # Moving Average Convergence Divergence (MACD)
        macd, macd_signal, macd_hist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        
        # Bollinger Bands
        upper_bb, middle_bb, lower_bb = talib.BBANDS(df['Close'], timeperiod=window_size, nbdevup=2, nbdevdn=2, matype=0)
        df['Upper_BB'] = upper_bb
        df['Middle_BB'] = middle_bb
        df['Lower_BB'] = lower_bb
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        return df

    def compute_zigzag_features(self, df: pd.DataFrame, up_thresh: float = 0.05, down_thresh: float = -0.05) -> pd.DataFrame:
        """
        Computes ZigZag-based features: previous high, previous previous high,
        previous low, and previous previous low for each row.

        Args:
            df (pd.DataFrame): DataFrame containing price data with 'Close' column.
                               Must have a datetime index.

        Returns:
            pd.DataFrame: DataFrame with additional columns:
                          - Prev_High: Previous peak value.
                          - Prev_Prev_High: Previous previous peak value.
                          - Prev_Low: Previous valley value.
                          - Prev_Prev_Low: Previous previous valley value.
        """
        df = df.copy()
        
        # Ensure 'Close' column exists and has no NaN values
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column")
        if df['Close'].isna().any():
            raise ValueError("'Close' column contains NaN values")

        # Compute ZigZag pivots
        pivots = peak_valley_pivots(df['Close'].values, up_thresh, down_thresh)
        
        # Create a Series of pivot points (peaks and valleys)
        pivot_series = pd.Series(pivots, index=df.index)
        pivot_points = pivot_series[pivot_series != 0]  # Non-zero values are peaks (1) or valleys (-1)
        
        # Separate peaks (1) and valleys (-1)
        peaks = pivot_points[pivot_points == 1]  # Peak indices
        valleys = pivot_points[pivot_points == -1]  # Valley indices
        
        # Initialize columns for features
        df['Prev_High'] = np.nan
        df['Prev_Prev_High'] = np.nan
        df['Prev_Low'] = np.nan
        df['Prev_Prev_Low'] = np.nan
        
        # For each row, find the previous peaks and valleys
        for idx in df.index:
            # Find peaks before the current index
            prev_peaks = peaks[peaks.index < idx]
            if len(prev_peaks) >= 1:
                prev_high_idx = prev_peaks.index[-1]
                df.at[idx, 'Prev_High'] = df.at[prev_high_idx, 'Close']
            if len(prev_peaks) >= 2:
                prev_prev_high_idx = prev_peaks.index[-2]
                df.at[idx, 'Prev_Prev_High'] = df.at[prev_prev_high_idx, 'Close']
            
            # Find valleys before the current index
            prev_valleys = valleys[valleys.index < idx]
            if len(prev_valleys) >= 1:
                prev_low_idx = prev_valleys.index[-1]
                df.at[idx, 'Prev_Low'] = df.at[prev_low_idx, 'Close']
            if len(prev_valleys) >= 2:
                prev_prev_low_idx = prev_valleys.index[-2]
                df.at[idx, 'Prev_Prev_Low'] = df.at[prev_prev_low_idx, 'Close']
        
        # Drop rows where any of the new features are NaN (i.e., not enough pivots)
        df.dropna(subset=['Prev_High', 'Prev_Prev_High', 'Prev_Low', 'Prev_Prev_Low'], inplace=True)
        
        return df

    def get_zigzag_features(self, df: pd.DataFrame, up_thresh: float = 0.05, down_thresh: float = -0.05, debug: bool = False) -> pd.DataFrame:
        """
        Extracts and returns ZigZag features (Prev_High, Prev_Prev_High, Prev_Low, Prev_Prev_Low),
        replacing NaN with 0. Optionally saves a debug plot.

        Args:
            df (pd.DataFrame): DataFrame containing price data with ZigZag features.
            debug (bool): If True, saves a plot of Close price with ZigZag pivots for verification.

        Returns:
            pd.DataFrame: DataFrame with only ['Prev_High', 'Prev_Prev_High', 'Prev_Low', 'Prev_Prev_Low'],
                          where NaN values are replaced with 0.
        """
        # Ensure ZigZag features are computed
        if all(col not in df.columns for col in ['Prev_High', 'Prev_Prev_High', 'Prev_Low', 'Prev_Prev_Low']):
            df = self.compute_zigzag_features(df, up_thresh, down_thresh)

        # Extract the required columns
        features_df = df[['Prev_High', 'Prev_Prev_High', 'Prev_Low', 'Prev_Prev_Low']].copy()

        # Replace NaN with 0
        features_df.fillna(0, inplace=True)

        # Debug plot if enabled
        if debug:
            pivots = peak_valley_pivots(df['Close'].values, up_thresh, down_thresh)
            ts_pivots = pd.Series(df['Close'], index=df.index)
            ts_pivots = ts_pivots[pivots != 0]
            df['Close'].plot(label='Close Price')
            ts_pivots.plot(style='g-o', label='ZigZag Pivots')
            plt.title('EURUSD 5m with ZigZag Pivots (Debug)')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.savefig('data/zigzag_features_plot.png')  # Save plot
            plt.close()  # Close the plot to avoid display

        return features_df

