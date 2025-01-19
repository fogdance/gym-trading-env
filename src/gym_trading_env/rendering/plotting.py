# src/gym_trading_env/rendering/plotting.py

import pandas as pd
import mplfinance as mpf
import matplotlib
matplotlib.use('Agg')  # Use the Anti-Grain Geometry backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from decimal import Decimal
import talib
from matplotlib.patches import Rectangle
from PIL import Image
import io
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from gym_trading_env.envs.trade_record import TradeRecord
from gym_trading_env.envs.trade_record_manager import TradeRecordManager
from gym_trading_env.envs.action import Action
from gym_trading_env.rendering.health_bar import HealthBar

class BollingerBandPlotter:
    def __init__(self, df, trade_record_manager, balance, window=20, channels=1, fig_width=1000, fig_height=600, dpi=100):
        plt.rcParams['toolbar'] = 'none'  # Disable toolbar
        self.window = window
        self.balance = balance
        self.df = df
        self.channels = channels
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
            entry_marker = {'marker': 'o', 'color': 'green', 'markersize': 15}
            exit_marker = {'marker': 'o', 'color': 'green', 'markersize': 15}
            profit_condition = exit_price > entry_price
        else:
            entry_marker = {'marker': '^', 'color': 'red', 'markersize': 15}
            exit_marker = {'marker': '^', 'color': 'red', 'markersize': 15}
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

    def plot_candlestick_chart(self, ax, df, show_bollinger=True, show_entry_exit=True, show_macd=False):
        """Plot the candlestick chart"""
        apds = []
        if show_bollinger:
            upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=self.window, nbdevup=2, nbdevdn=2, matype=0)
            apds = [
                mpf.make_addplot(upper, ax=ax, linestyle='--', width=1.5),  # Upper band
                mpf.make_addplot(lower, ax=ax, linestyle='--', width=1.5),  # Lower band
                mpf.make_addplot(middle, ax=ax, color='blue', width=1.5)  # Middle band
            ]

        apds_macd = []    
        if show_macd:
            macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
            
            # Plot MACD line and Signal line
            apds_macd = ([
                mpf.make_addplot(macd, ax=ax, color='orange', width=1.5, panel=1),  # MACD line
                mpf.make_addplot(macdsignal, ax=ax, color='blue', width=1.5, panel=1),  # Signal line
            ])
            

        # Plot entry and exit markers
        addplots = []
        if show_entry_exit:
            long_open_records, long_close_records, short_open_records, short_close_records = self.process_trade_record()

            # Plot long trades
            adds1 = self.plot_trades(ax, long_open_records, long_close_records, 'Long')

            # Plot short trades
            adds2 = self.plot_trades(ax, short_open_records, short_close_records, 'Short')
            addplots = adds1 + adds2

        all_addplots = apds + addplots + apds_macd

        # Plot the candlestick chart on the given ax
        mpf.plot(df, type='candle', ax=ax, style='default', addplot=all_addplots, volume=self.show_volume)

        # Remove the right "Price" label
        ax.set_ylabel('')
        self.format_yaxis_labels(ax)  # Format y-axis labels

        # Hide the x-axis and y-axis ticks
        ax.set_xticks([])  # Hide x-axis ticks
        ax.set_yticks([])  # Hide y-axis ticks

    def plot_progress_bar(self, ax_progress, ax_progress_text, unit=1000):
        """
        Draw a progress bar to represent the balance and display the latest time label
        
        Parameters:
        - ax: Matplotlib Axes object
        - balance: Current balance
        - unit: Progress bar unit, default is 1000
        """
        
        # Create a HealthBar instance
        health_bar = HealthBar(initial_health=self.balance, 
                            unit_health=unit,
                           color_A='red', 
                           color_B='orange', 
                           color_N='cyan', 
                           line_width=10)
        
        health_bar.draw_on_ax(ax_progress, max_health=unit*3)

        multiplier = health_bar.get_multiplier()
        ax_progress_text.text(0, 0, f'x{multiplier}', 
                       ha='center', va='bottom', fontsize=8, color='black')

    def plot(self, filename=None):
        """Plot the candlestick chart and Bollinger Bands"""
        try:
            # Calculate the figure size in inches based on pixels and DPI
            fig_width_inch = self.fig_width / self.dpi
            fig_height_inch = self.fig_height / self.dpi

            fig = plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=self.dpi)

            # Use GridSpec to define a total of 5 rows (3 subplots + 2 separators)
            gs = GridSpec(nrows=3, ncols=1, figure=fig, height_ratios=[0.82, 0.02, 0.2], hspace=0, wspace=0)


            # Large plot (third row)
            ax_large = fig.add_subplot(gs[0, 0])
            # Plot the large chart
            self.plot_candlestick_chart(ax_large, self.df, show_bollinger=True, show_entry_exit=True, show_macd=False)

            # Separator line 2 (fourth row)
            ax_sep2 = fig.add_subplot(gs[1, 0])
            ax_sep2.axis('off')  # Hide the axes
            ax_sep2.hlines(0.5, 0, 1, colors='black', linewidth=1)  # Draw a horizontal line

            # Bottom subplot (fifth row)
            # The bottom subplot (fifth row) is divided into 3 columns using GridSpecFromSubplotSpec, with a width ratio of 6:2:2
            gs_bottom = GridSpecFromSubplotSpec(nrows=1, ncols=3, 
                                                subplot_spec=gs[2, 0], 
                                                width_ratios=[6, 2, 2], 
                                                wspace=0, hspace=0)

            # Progress bar (first column)
            ax_progress = fig.add_subplot(gs_bottom[0, 0])
            ax_progress_text = fig.add_subplot(gs_bottom[0, 1])

            self.plot_progress_bar(ax_progress, ax_progress_text)

            # Latest time label (third column)
            ax_time = fig.add_subplot(gs_bottom[0, 2])
            latest_time = self.df.index[-1].strftime('%H:%M')  # Use the latest time from the actual data
            ax_time.text(0.5, 0.5, latest_time, ha='center', va='center', fontsize=8, color='black')
            ax_time.axis('off')  # Hide the axes

            # Hide the axes, ticks, and borders of the main subplots
            for ax in [ax_large, ax_progress, ax_progress_text, ax_time]:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            # Adjust the overall margins of the figure to ensure subplots are tightly arranged
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

            # Ensure the canvas is fully rendered
            fig.canvas.draw()

            img = self.to_binary(fig, filename)

            plt.close(fig)

            return img

            
        except Exception as e:
            print(f"Error: {e}")
            white_image = np.ones((self.fig_height, self.fig_width, self.channels), dtype=np.uint8) * 255  # White image
            return white_image
        

    
    def to_binary(self, fig, filename=None):
        if self.channels == 1:
            return self._to_gray(fig, filename)
        
        if filename:
            # Save the figure with specified dpi without bbox_inches='tight'
            fig.savefig(filename, dpi=self.dpi)

        # Convert the figure to a numpy array with the exact pixel dimensions
        buf = fig.canvas.get_renderer().buffer_rgba()
        img = np.asarray(buf, dtype=np.uint8)
        img = img[:, :, :3]  # Remove the alpha channel to get RGB
        img = img.reshape(int(self.fig_height), int(self.fig_width), self.channels)
        return img
    
    def _to_gray(self, fig, filename=None):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)

        buf.seek(0)
        pil_image = Image.open(buf)
        
        # Convert the image to grayscale
        gray_pil_image = pil_image.convert('L')
        
        # Resize the image to the desired dimensions
        gray_pil_image = gray_pil_image.resize((self.fig_width, self.fig_height))
        
        if filename:
            gray_pil_image.save(filename, format='PNG')
        
        # Convert the image to binary data
        img_array = np.array(gray_pil_image)
        img_array = img_array.reshape(int(self.fig_height), int(self.fig_width), self.channels)

        return img_array    