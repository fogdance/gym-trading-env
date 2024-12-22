# examples/download_forex_data.py

from gym_trading_env.utils.data_downloader import ForexDataDownloader
import pandas as pd
import os

def main():
    os.mkdir('data') if not os.path.exists('data') else None

    proxy = None

    proxy = {
        'http': 'http://127.0.0.1:10809',
        'https': 'http://127.0.0.1:10809',
    }  # Replace with your actual proxy settings or set to None

    downloader = ForexDataDownloader(proxy=proxy)

    symbol = "EURUSD"  # Forex pair symbol
    interval = '60m'

    df = downloader.download_forex_data(symbol=symbol, interval=interval, start_date="2023-01-01", end_date="2023-12-31")

    # Save to CSV for future use
    df.to_csv(f'data/{symbol}_{interval}.csv', index=False)
    print(f"Data for {symbol} downloaded and saved successfully.")

if __name__ == '__main__':
    main()
