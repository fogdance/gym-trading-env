# src/gym_trading_env/utils/data_processing.py

import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads historical price data from a CSV file.
    
    Args:
        file_path (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: The loaded data.
    """
    df = pd.read_csv(file_path, parse_dates=True, index_col='Date')
    df = df.sort_index()
    return df
