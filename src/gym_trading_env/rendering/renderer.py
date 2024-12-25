# src/gym_trading_env/rendering/renderer.py

import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
import mplfinance as mpf
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class Renderer:
    """
    Renderer class to generate visual representations of the trading environment's state.
    """
    def __init__(self, image_width=640, image_height=480, channels=3):
        """
        Initialize the Renderer with specified image dimensions.

        Args:
            image_width (int): Width of the generated image.
            image_height (int): Height of the generated image.
        """
        self.image_height = image_height
        self.image_width = image_width
        self.fig, self.ax = plt.subplots(figsize=(self.image_width / 100, self.image_height / 100), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.channels = channels

    def render(self, state_info):
        """
        Generate an image representing the current state of the trading environment.

        Args:
            state_info (dict): Dictionary containing state information such as balance, equity, positions, etc.

        Returns:
            np.ndarray: An image array representing the current state.
        """
        self.ax.clear()  # Clear previous contents

        # Example visualization: Display account balances and positions as text
        text_str = (
            f"Balance: ${state_info['balance']:.2f}\n"
            f"Equity: ${state_info['equity']:.2f}\n"
            f"Used Margin: ${state_info['used_margin']:.2f}\n"
            f"Free Margin: ${state_info['free_margin']:.2f}\n"
            f"Long Position: {state_info['long_position']} lots\n"
            f"Short Position: {state_info['short_position']} lots\n"
            f"Unrealized P&L: ${state_info['unrealized_pnl']:.2f}\n"
            f"Realized P&L: ${state_info['realized_pnl']:.2f}\n"
            f"Fees Collected: ${state_info['fees_collected']:.2f}"
        )

        # Add text to the plot
        self.ax.text(0.05, 0.95, text_str, fontsize=12, verticalalignment='top', transform=self.ax.transAxes)

        # Remove axes for a cleaner image
        self.ax.axis('off')

        # Render the plot to a PNG image in memory
        buf = BytesIO()
        self.fig.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)

        # Convert the image to RGB and resize if necessary
        image = image.convert('RGB')
        image = image.resize((self.image_width, self.image_height))

        # Convert the image to a NumPy array
        image_array = np.array(image)

        buf.close()
        return image_array
    
    def render_to_array(self, df: pd.DataFrame, technical_indicators: np.ndarray):
        """
        Renders the K-lines and technical indicators to an image array using mplfinance.

        Args:
            df (pd.DataFrame): The historical price data.
            technical_indicators (np.ndarray): Technical indicators data.

        Returns:
            np.ndarray: The rendered image as a (H, W, C) array.
        """
        self.ax.clear()

        # Prepare additional plots for technical indicators
        add_plots = []
        if 'SMA' in df.columns:
            add_plots.append(mpf.make_addplot(df['SMA'], color='orange'))
        if 'EMA' in df.columns:
            add_plots.append(mpf.make_addplot(df['EMA'], color='green'))

        # Plot candlestick chart with technical indicators
        mpf.plot(
            df,
            type='candle',
            ax=self.ax,
            addplot=add_plots,
            volume=False,
            show_nontrading=True,
            style='charles',
            datetime_format='%Y-%m-%d',
            xrotation=45
        )

        self.ax.set_title('Forex Trading Environment')

        # Render the plot to a buffer
        self.canvas.draw()
        buf = BytesIO()
        self.fig.savefig(buf, format='rgba')
        buf.seek(0)
        img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()

        # Convert to image array
        img = img.reshape((self.image_height, self.image_width, self.channels))
        return img