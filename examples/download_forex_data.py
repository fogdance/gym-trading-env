# examples/download_forex_data.py

from gym_trading_env.utils.data_downloader import ForexDataDownloader
import pandas as pd

def main():
    api_key = 'YOUR_ALPHA_VANTAGE_API_KEY'  # Replace with your actual API key
    proxy = {
        'http': 'http://your_proxy:port',
        'https': 'https://your_proxy:port',
    }  # Replace with your actual proxy settings or set to None

    downloader = ForexDataDownloader(api_key=api_key, proxy=proxy)

    symbol = 'USDJPY'  # Forex pair symbol
    interval = 'Daily'  # Options: 'Daily', 'Intraday'
    outputsize = 'full'  # 'compact' or 'full'

    df = downloader.download_forex_data(symbol=symbol, interval=interval, outputsize=outputsize)

    # Save to CSV for future use
    df.to_csv(f'data/{symbol}_{interval}.csv')
    print(f"Data for {symbol} downloaded and saved successfully.")

if __name__ == '__main__':
    main()
