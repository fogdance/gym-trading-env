# src/gym_trading_env/rendering/renderer.py

import matplotlib.pyplot as plt
import pandas as pd

class Renderer:
    """
    A simple renderer for the trading environment using matplotlib.
    """
    
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
    
    def render(self, df: pd.DataFrame, portfolio_history: pd.DataFrame):
        """
        Renders the portfolio performance over time.
        
        Args:
            df (pd.DataFrame): The historical price data.
            portfolio_history (pd.DataFrame): The history of portfolio valuations.
        """
        self.ax.clear()
        self.ax.plot(df.index, df['Close'], label='Close Price')
        self.ax.plot(portfolio_history.index, portfolio_history['total_asset'], label='Portfolio Value')
        self.ax.legend()
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Value')
        plt.pause(0.001)
    
    def close(self):
        """
        Closes the rendering window.
        """
        plt.close(self.fig)
