# src/gym_trading_env/utils/data_downloader.py

import requests
import pandas as pd
from decimal import Decimal
from typing import Optional
import logging

class ForexDataDownloader:
    def __init__(self, api_key: str, proxy: Optional[dict] = None):
        """
        Initializes the ForexDataDownloader with the given API key and proxy settings.

        Args:
            api_key (str): API key for the data provider.
            proxy (dict, optional): Dictionary of proxies to use for HTTP requests. Defaults to None.
        """
        self.api_key = api_key
        self.proxy = proxy
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def download_forex_data(self, symbol: str, interval: str = 'Daily', outputsize: str = 'full') -> pd.DataFrame:
        """
        Downloads forex K-line data for the specified symbol and interval.

        Args:
            symbol (str): Forex pair symbol (e.g., 'USDJPY').
            interval (str, optional): Time interval between data points. Options include 'Intraday', 'Daily', etc. Defaults to 'Daily'.
            outputsize (str, optional): The amount of data to retrieve. 'compact' returns the latest 100 data points; 'full' returns the full-length time series. Defaults to 'full'.

        Returns:
            pd.DataFrame: DataFrame containing the K-line data.
        """
        self.logger.info(f"Downloading forex data for {symbol} with interval {interval} and outputsize {outputsize}.")

        if interval.lower() == 'daily':
            function = 'FX_DAILY'
            datatype = 'json'
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': function,
                'from_symbol': symbol[:3],
                'to_symbol': symbol[3:],
                'outputsize': outputsize,
                'apikey': self.api_key,
                'datatype': datatype
            }
        elif interval.lower() == 'intraday':
            function = 'FX_INTRADAY'
            datatype = 'json'
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': function,
                'from_symbol': symbol[:3],
                'to_symbol': symbol[3:],
                'interval': '60min',  # Modify as needed
                'outputsize': outputsize,
                'apikey': self.api_key,
                'datatype': datatype
            }
        else:
            raise ValueError("Unsupported interval. Choose 'Daily' or 'Intraday'.")

        try:
            response = requests.get(url, params=params, proxies=self.proxy, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'Error Message' in data:
                self.logger.error(f"Error fetching data: {data['Error Message']}")
                raise ValueError(data['Error Message'])
            if 'Note' in data:
                self.logger.error(f"API call frequency exceeded: {data['Note']}")
                raise ValueError(data['Note'])

            time_series_key = ''
            if function == 'FX_DAILY':
                time_series_key = 'Time Series FX (Daily)'
            elif function == 'FX_INTRADAY':
                time_series_key = 'Time Series FX (60min)'  # Modify based on interval

            if time_series_key not in data:
                self.logger.error("Unexpected data format received from API.")
                raise ValueError("Unexpected data format received from API.")

            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df = df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close'
            })
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df.astype(float)
            self.logger.info(f"Successfully downloaded {len(df)} data points for {symbol}.")
            return df

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise e
        except ValueError as ve:
            self.logger.error(f"Value error: {ve}")
            raise ve
