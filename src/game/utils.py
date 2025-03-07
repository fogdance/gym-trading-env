import pandas as pd
from math import atan

def load_csv_data(path):
    df = pd.read_csv(path, parse_dates=['Date'])
    return df

def calculate_road_angle(open_price, close_price):
    return atan((close_price - open_price)/open_price * 100)

def calculate_road_width(high, low, open_price):
    return base_width - (high - low)/open_price * scale_factor
