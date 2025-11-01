# src/gym_trading_env/utils/data_processing.py

import os
import pandas as pd
from gym_trading_env.utils.data_downloader import ForexDataDownloader
import logging

def load_data(symbol: str, interval: str, proxy: str = None) -> pd.DataFrame:
    """
    Loads forex K-line data from a CSV file or downloads it if not present.

    Args:
        symbol (str): Forex pair symbol (e.g., 'USDJPY').
        interval (str, optional): {'1m','2m','5m','15m','30m','60m','90m','1h','1d'}. Defaults to '1d'.
        proxy (dict, optional): Proxy settings for downloading data. Defaults to None.
        api_key (str, optional): API key for data provider. Required if downloading data. Defaults to ''.

    Returns:
        pd.DataFrame: DataFrame containing the K-line data.
    """
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    filename = f"{symbol}_{interval}.csv"
    filepath = os.path.join(data_dir, filename)

    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    if os.path.exists(filepath):
        logger.info(f"Loading existing data from {filepath}.")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    else:
        downloader = ForexDataDownloader(proxy=proxy)
        df = downloader.download_forex_data(symbol=symbol, interval=interval)
        df.to_csv(filepath)
        logger.info(f"Data for {symbol} downloaded and saved to {filepath}.")

    return df
