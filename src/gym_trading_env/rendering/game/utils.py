import pandas as pd
from math import atan

def load_csv_data(path):
    df = pd.read_csv(path, parse_dates=['Date'])
    return df

