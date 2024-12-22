# src/gym_trading_env/utils/data_downloader.py

import yfinance as yf
import pandas as pd
from typing import Optional
import logging

class ForexDataDownloader:
    def __init__(self, proxy: Optional[dict] = None):
        """
        Initializes the ForexDataDownloader with the given proxy settings.

        Args:
            proxy (dict, optional): Dictionary of proxies to use for HTTP requests. Defaults to None.
        """
        self.proxy = proxy
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def download_forex_data(self, symbol: str, interval: str = '1d', start_date: str = '2023-01-01', end_date: str = '2023-12-31') -> pd.DataFrame:
        """
        Downloads forex data for the specified symbol and interval.

        Args:
            symbol (str): Forex pair symbol (e.g., 'EURUSD=X').
            interval (str, optional): Time interval between data points. Options include '1m', '5m', '1d', etc. Defaults to '1d'.
            start_date (str, optional): Start date for the data download. Defaults to '2023-01-01'.
            end_date (str, optional): End date for the data download. Defaults to '2023-12-31'.

        Returns:
            pd.DataFrame: DataFrame containing the forex data.
        """
        symbol = symbol + "=X"
        
        self.logger.info(f"Downloading forex data for {symbol} with interval {interval} from {start_date} to {end_date}.")

        try:
            # Download data using yfinance
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval, proxy=self.proxy)
            if data.empty:
                self.logger.error(f"No data returned for {symbol} during {start_date} ~ {end_date}.")
                raise ValueError(f"No data returned for {symbol} during {start_date} ~ {end_date}.")
            

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            data.reset_index(inplace=True)

            if 'Datetime' in data.columns:
                data["Date"] = pd.to_datetime(data["Datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            elif 'Date' in data.columns:
                data["Date"] = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                raise ValueError("No valid date column found in DataFrame.")

            needed_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            data = data[needed_cols]

            return data

        except Exception as e:
            self.logger.error(f"Error occurred while downloading data: {e}")
            raise e
