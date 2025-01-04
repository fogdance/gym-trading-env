# src/gym_trading_env/rendering/plotting.py

import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from decimal import Decimal
import talib
from matplotlib.patches import Rectangle

from gym_trading_env.envs.trade_record import TradeRecord
from gym_trading_env.envs.trade_record_manager import TradeRecordManager
from gym_trading_env.envs.action import Action
from gym_trading_env.rendering.health_bar import HealthBar

class BollingerBandPlotter:
    def __init__(self, df, trade_record_manager, balance, window=20, fig_width=1000, fig_height=600, dpi=100):
        plt.rcParams['toolbar'] = 'none'  # Disable toolbar
        self.window = window
        self.balance = balance
        self.df = df
        self.trade_record_manager = trade_record_manager
        self.show_volume = False

        self.fig_width = fig_width  # Figure width (in pixels)
        self.fig_height = fig_height  # Figure height (in pixels)
        self.dpi = dpi  # DPI (dots per inch) of the figure
        self.verify_data(self.df)
    
    def verify_data(self, df):
        self.df.columns = self.df.columns.str.strip()  # Remove leading/trailing spaces from column names

        # If the index is a Date, ensure it's in datetime format
        if isinstance(self.df.index, pd.DatetimeIndex):
            # If the index is already in datetime type, skip conversion
            pass
        else:
            # If the index is not in datetime type, try to convert it
            self.df.index = pd.to_datetime(self.df.index, errors='raise')

        # Ensure required columns are present
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure data is sorted by date
        self.df.sort_index(inplace=True)
        
        # Convert all required columns to numeric type, handling potential non-numeric data
        # for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        #     self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Drop missing values
        # self.df.dropna(inplace=True)
        
        # If Volume has non-zero values, ensure it is of integer type
        if self.show_volume:
            self.df['Volume'] = self.df['Volume'].astype(int)            

    def format_yaxis_labels(self, ax):
        """Format the y-axis labels to 4 decimal places"""
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

    def process_trade_record(self):
        # Get the time range of the data
        data_start_time = self.df.index[0]
        data_end_time = self.df.index[-1]
        
        # Separate the trade records by open and close types
        long_open_records = []
        long_close_records = []
        short_open_records = []
        short_close_records = []

        # Categorize the trade records
        for trade_record in self.trade_record_manager.trade_history:
            if trade_record.timestamp < data_start_time or trade_record.timestamp > data_end_time:
                # Skip trade records that are outside the data time range
                continue
            # Only process open records within the data time range
            if trade_record.operation_type == Action.LONG_OPEN.name:
                long_open_records.append(trade_record)
            elif trade_record.operation_type == Action.LONG_CLOSE.name:
                if len(long_open_records) == 0:
                    # If there is no matching open record, skip this close record
                    continue
                long_close_records.append(trade_record)
            elif trade_record.operation_type == Action.SHORT_OPEN.name:
                short_open_records.append(trade_record)
            elif trade_record.operation_type == Action.SHORT_CLOSE.name:
                if len(short_open_records) == 0:
                    # If there is no matching open record, skip this close record
                    continue
                short_close_records.append(trade_record)
        
        return long_open_records, long_close_records, short_open_records, short_close_records

    def plot_trades(self, ax, open_records, close_records, trade_type, show_close=False):
        """
        Plot the open and close trades for a specific trade type (long or short)
        """
        adds = []
        for open_record in open_records:
            # For each open record, find the earliest matching close record
            matching_close_record = self.find_matching_close(open_record, close_records)
            
            if matching_close_record:
                if show_close:
                    # Plot entry and exit markers and connecting line
                    addplot = self.plot_trade(ax, open_record, matching_close_record, trade_type)
                    adds.extend(addplot)
                # Remove the matched close record from the pending close records
                close_records.remove(matching_close_record)
            else:
                # If no matching close record, it means the trade is still open
                addplot = self.plot_unclosed_trade(ax, open_record, trade_type)
                adds.extend(addplot)
        return adds

    def find_matching_close(self, open_record, close_records):
        """
        Find the close record that matches the open record
        """
        for close_record in close_records:
            # Ensure that the close record's timestamp is after the open record's timestamp
            if close_record.timestamp > open_record.timestamp:
                return close_record
        return None

    def plot_trade(self, ax, open_record, close_record, trade_type):        
        # Get the entry and exit timestamps and prices
        entry_date = open_record.timestamp
        entry_price = float(open_record.price)
        exit_date = close_record.timestamp
        exit_price = float(close_record.price)

        return self.plot_trade_and_line(ax, entry_date, entry_price, exit_date, exit_price, trade_type, linestyle='-')

    def plot_unclosed_trade(self, ax, open_record, trade_type):        
        # Get the entry timestamp and price
        entry_date = open_record.timestamp
        entry_price = float(open_record.price)
        exit_date = self.df.index[-1]  # Use the last available timestamp for exit
        exit_price = self.df['Close'].iloc[-1]  # Use the last closing price for exit

        return self.plot_trade_and_line(ax, entry_date, entry_price, exit_date, exit_price, trade_type, linestyle=':')

    def plot_trade_and_line(self, ax, entry_date, entry_price, exit_date, exit_price, trade_type, linestyle):
        # Define marker properties based on the trade type and action (entry or exit)
        if trade_type == 'Long':
            entry_marker = {'marker': 'o', 'color': 'green', 'markersize': 75}
            exit_marker = {'marker': 'o', 'color': 'green', 'markersize': 75}
            profit_condition = exit_price > entry_price
        else:
            entry_marker = {'marker': '^', 'color': 'red', 'markersize': 75}
            exit_marker = {'marker': '^', 'color': 'red', 'markersize': 75}
            profit_condition = exit_price < entry_price

        # Determine the line color based on profit or loss
        color = 'green' if profit_condition else 'red'

        # Ensure that the entry and exit dates exist in the dataframe index
        if entry_date not in self.df.index or exit_date not in self.df.index:
            raise ValueError(f"Time points {entry_date} or {exit_date} are not in the data's index")

        # Initialize marker data for entry and exit
        entry_data = pd.Series([np.nan] * len(self.df), index=self.df.index)
        exit_data = pd.Series([np.nan] * len(self.df), index=self.df.index)
        entry_data.loc[entry_date] = entry_price
        exit_data.loc[exit_date] = exit_price

        # Calculate the line data using linear interpolation between entry and exit
        x1, y1 = self.df.index.get_loc(entry_date), entry_price
        x2, y2 = self.df.index.get_loc(exit_date), exit_price

        line_data = pd.Series([np.nan] * len(self.df), index=self.df.index)
        if x1 < x2:
            line_data.iloc[x1:x2+1] = np.linspace(y1, y2, x2 - x1 + 1)
        elif x1 > x2:
            line_data.iloc[x2:x1+1] = np.linspace(y2, y1, x1 - x2 + 1)
        else:
            line_data.iloc[x1] = y1  # Same point

        # Create addplot objects for entry and exit markers
        addplot_entry = mpf.make_addplot(entry_data, ax=ax, type='scatter',
                                        markersize=entry_marker['markersize'],
                                        marker=entry_marker['marker'],
                                        color=entry_marker['color'])
        
        addplot_exit = None
        if exit_date != entry_date:
            addplot_exit = mpf.make_addplot(exit_data, ax=ax, type='scatter',
                                            markersize=exit_marker['markersize'],
                                            marker=exit_marker['marker'],
                                            color=exit_marker['color'])
        
        # Create addplot object for the connecting line
        addplot_line = mpf.make_addplot(line_data, ax=ax, type='line',
                                        color=color, width=1.5, linestyle=linestyle)

        return [addplot_entry, addplot_exit, addplot_line] if addplot_exit else [addplot_entry, addplot_line]

    def plot_candlestick_chart(self, ax, df, show_bollinger=True, show_entry_exit=True, small=False):
        """Plot the candlestick chart (large or small)"""
        apds = []
        if show_bollinger:
            upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=self.window, nbdevup=2, nbdevdn=2, matype=0)

            if small:  # For small chart, only show the middle band
                apds = [
                    mpf.make_addplot(middle, ax=ax, color='blue', width=1.5)  # Only show middle band
                ]
            else:  # For large chart, show the full Bollinger Bands
                apds = [
                    mpf.make_addplot(upper, ax=ax, linestyle='--', width=1.5),  # Upper band
                    mpf.make_addplot(lower, ax=ax, linestyle='--', width=1.5),  # Lower band
                    mpf.make_addplot(middle, ax=ax, color='blue', width=1.5)  # Middle band
                ]

        # Plot entry and exit markers
        addplots = []
        if show_entry_exit:
            long_open_records, long_close_records, short_open_records, short_close_records = self.process_trade_record()

            # Plot long trades
            adds1 = self.plot_trades(ax, long_open_records, long_close_records, 'Long')

            # Plot short trades
            adds2 = self.plot_trades(ax, short_open_records, short_close_records, 'Short')
            addplots = adds1 + adds2

        all_addplots = apds + addplots

        # Plot the candlestick chart on the given ax
        mpf.plot(df, type='candle', ax=ax, style='default', addplot=all_addplots, volume=self.show_volume)

        # Remove the right "Price" label
        ax.set_ylabel('')
        self.format_yaxis_labels(ax)  # Format y-axis labels

        # Hide the x-axis and y-axis ticks
        ax.set_xticks([])  # Hide x-axis ticks
        ax.set_yticks([])  # Hide y-axis ticks

    def plot_progress_bar(self, ax, balance, unit=1000):
        """
        Draw a progress bar to represent the balance and display the latest time label
        
        Parameters:
        - ax: Matplotlib Axes object
        - balance: Current balance
        - unit: Progress bar unit, default is 1000
        """
        
        # Get the latest time label
        latest_time = self.df.index[-1].strftime('%H:%M')
        
        # Create a HealthBar instance
        health_bar = HealthBar(initial_health=balance, 
                            unit_health=unit,
                           color_A='red', 
                           color_B='orange', 
                           color_N='cyan', 
                           line_width=10)
        
        # Draw the progress bar and time label
        health_bar.draw_on_ax(ax, latest_time=latest_time)

    def plot(self, filename=None, show=False):
        """Plot the candlestick chart and Bollinger Bands"""
        try:
            # Calculate the figure size in inches based on pixels and DPI
            fig_width_inch = self.fig_width / self.dpi
            fig_height_inch = self.fig_height / self.dpi
            
            # Create the figure and set subplot size ratios
            fig = plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=self.dpi)  # Use the provided width and height (pixels to inches)
            gs = fig.add_gridspec(3, 1, height_ratios=[30, 65, 5])  # Set the ratio for 3 subplots
            
            # Plot the small chart at the top
            ax_small = fig.add_subplot(gs[0])
            self.plot_candlestick_chart(ax_small, self.df, show_bollinger=True, show_entry_exit=True, small=True)  # Small chart

            # Plot the large chart in the middle
            ax_large = fig.add_subplot(gs[1])
            self.plot_candlestick_chart(ax_large, self.df, show_bollinger=True, show_entry_exit=True, small=False)  # Large chart

            # Plot the progress bar at the bottom
            ax_progress = fig.add_subplot(gs[2])
            self.plot_progress_bar(ax_progress, self.balance)

            plt.tight_layout()
            if show:
                plt.show()

            if filename:
                # Save the figure with specified dpi without bbox_inches='tight'
                fig.savefig(filename, dpi=self.dpi)

            # Ensure the canvas is fully rendered
            fig.canvas.draw()

            # Convert the figure to a numpy array with the exact pixel dimensions
            buf = fig.canvas.get_renderer().buffer_rgba()
            img = np.asarray(buf, dtype=np.uint8)
            img = img[:, :, :3]  # Remove the alpha channel to get RGB
            img = img.reshape(int(self.fig_height), int(self.fig_width), 3)  # Ensure correct shape
            plt.close(fig)
            
            return img
        except Exception as e:
            print(f"Error: {e}")
            white_image = np.ones((self.fig_height, self.fig_width, 3), dtype=np.uint8) * 255
            return white_image
