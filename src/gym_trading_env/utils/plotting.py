import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from typing import Optional

def draw_basic_candlestick(df: pd.DataFrame):
    # 定义图表尺寸
    figsize_inches = (8, 6)  # 800x600 pixels at 100 dpi

    # 定义样式
    style = mpf.make_mpf_style(base_mpf_style='classic')

    # 创建图表
    fig, ax = mpf.plot(
        df,
        type='ohlc',
        # style=style,
        volume=False,
        returnfig=True,
        figsize=figsize_inches,
    )

    plt.show()
    plt.close(fig)

def draw_candlestick_with_indicators(
    df: pd.DataFrame,
    width: int = 800,  # Desired width in pixels
    height: int = 600,  # Desired height in pixels
    dpi: int = 100,     # Dots per inch
    filename: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Draws a candlestick chart with SMA, Bollinger Bands, MACD, and RSI indicators.

    Args:
        df (pd.DataFrame): DataFrame containing OHLC data with DateTime index.
                           Required columns: ['Open', 'High', 'Low', 'Close', 'SMA', 'RSI', 'MACD', 'MACD_Signal', 'Upper_BB', 'Middle_BB', 'Lower_BB']
        width (int, optional): Width of the output image in pixels. Defaults to 800.
        height (int, optional): Height of the output image in pixels. Defaults to 600.
        dpi (int, optional): Dots per inch for the figure. Defaults to 100.
        filename (str, optional): If provided, saves the plot to the specified file path.
                                  Supported formats: PNG, JPEG, etc.

    Returns:
        Optional[np.ndarray]: Numpy array representation of the image if filename is not provided.
    """
    logger = logging.getLogger(__name__)

    # Validate DataFrame contains all required columns
    required_columns = [
        'Open', 'High', 'Low', 'Close',
        'SMA', 'RSI', 'MACD', 'MACD_Signal',
        'Upper_BB', 'Middle_BB', 'Lower_BB'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"DataFrame must contain columns: {', '.join(missing_columns)} for plotting.")

    # Define additional plots for technical indicators without y-axis labels
    apds = [
        mpf.make_addplot(df['SMA'], color='blue', width=1, panel=0),
        mpf.make_addplot(df['Upper_BB'], color='grey', linestyle='--', width=1, panel=0),
        mpf.make_addplot(df['Lower_BB'], color='grey', linestyle='--', width=1, panel=0),
        mpf.make_addplot(df['MACD'], color='purple', panel=1),
        mpf.make_addplot(df['MACD_Signal'], color='orange', panel=1),
        mpf.make_addplot(df['RSI'], color='green', panel=2),
    ]

    # Define panel ratios (main chart, MACD, RSI)
    panel_ratios = (3, 1, 1)

    # Calculate figsize in inches based on desired pixel dimensions and dpi
    figsize_inches = (width / dpi, height / dpi)

    # Define style
    style = mpf.make_mpf_style(
        base_mpf_style='classic',
        rc={'figure.figsize': figsize_inches}  # Set figure size in inches
    )

    # Create the plot
    fig, axes = mpf.plot(
        df,
        type='candle',  # Ensures a candlestick chart
        style=style,
        addplot=apds,
        panel_ratios=panel_ratios,
        volume=False,
        returnfig=True,
        figsize=figsize_inches,  # Ensure figsize is consistent
        tight_layout=False        # Disable tight_layout to manage manually
    )

    # Adjust subplot parameters to prevent clipping and reduce white space
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, hspace=0.1)

    # Loop through all axes and remove text, ticks, and labels
    for ax in axes:
        if ax:
            ax.set_xlabel('')  # Remove x-axis label
            ax.set_ylabel('')  # Remove y-axis label
            ax.set_xticks([])  # Remove x-axis ticks
            ax.set_yticks([])  # Remove y-axis ticks
            ax.set_xticklabels([])  # Remove x-axis tick labels
            ax.set_yticklabels([])  # Remove y-axis tick labels

    if filename:
        # Save the figure with specified dpi without bbox_inches='tight'
        fig.savefig(filename, dpi=dpi)
        plt.close(fig)
        logger.info(f"Plot saved to {filename} with size {width}x{height} pixels.")

    # Ensure the canvas is fully rendered
    fig.canvas.draw()

    # Convert the figure to a numpy array with the exact pixel dimensions
    buf = fig.canvas.get_renderer().buffer_rgba()
    img = np.asarray(buf, dtype=np.uint8)
    img = img[:, :, :3]  # Remove the alpha channel to get RGB
    img = img.reshape(int(height), int(width), 3)  # Ensure correct shape

    plt.close(fig)
    logger.info(f"Plot generated as numpy array with size {width}x{height} pixels.")
    return img

