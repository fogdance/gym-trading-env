# examples/example_plot_usdjpy.py

import os
import pandas as pd
import logging
from gym_trading_env.utils.feature_engineering import FeatureEngineer
from gym_trading_env.rendering.plotting import draw_candlestick_with_indicators, draw_basic_candlestick
from gym_trading_env.utils.data_processing import load_data



def main():
    """
    Main function to plot USDJPY candlestick chart with technical indicators.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Define the path to the USDJPY data CSV
    data_filepath = os.path.join('data', 'USDJPY_1d.csv')

    # Load OHLC data
    logger.info(f"Loading data from {data_filepath}...")
    df = load_data('USDJPY', interval = '1d')
    print(df)
    logger.info("Data loaded successfully.")
    draw_basic_candlestick(df)

    # Initialize FeatureEngineer
    feature_engineer = FeatureEngineer(window_size=20)
    logger.info("Computing technical indicators...")
    df_features = feature_engineer.compute_features(df)
    logger.info("Technical indicators computed successfully.")

    # Plot and save the candlestick chart with indicators
    output_filepath = os.path.join('output', 'USDJPY_candlestick.png')
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    logger.info(f"Plotting candlestick chart and saving to {output_filepath}...")
    draw_candlestick_with_indicators(
        df=df_features,
        width=400,
        height=300,
        dpi=100,          # DPI corresponding to figsize_inches = (8,6)
        filename=output_filepath
    )
    logger.info("Candlestick chart plotted and saved successfully.")


if __name__ == "__main__":
    main()
