# examples/plot.py

import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from decimal import Decimal
import os

from gym_trading_env.envs.trade_record import TradeRecord
from gym_trading_env.envs.trade_record_manager import TradeRecordManager
from gym_trading_env.rendering.plotting import BollingerBandPlotter
from gym_trading_env.envs.action import Action

def main():
    # Load sample data
    df = pd.read_csv('data/EURUSD_test5m.csv', parse_dates=['Date'])

    # Create an instance of TradeRecordManager
    trade_record_manager = TradeRecordManager()

    # Select every 10th data point from the dataframe
    sampled_df = df.iloc[::10]  # Take every 10th data point

    # Used to mark if the current operation is an opening trade or closing trade
    is_long_open = True

    # Generate trade records and add them sequentially
    for idx, row in sampled_df.iterrows():
        # Opening trade (LONG_OPEN)
        if is_long_open:
            trade_record = TradeRecord(
                timestamp=row['Date'],
                operation_type=Action.SHORT_OPEN.name,  # Use Action.LONG_OPEN
                position_size=Decimal('0.1'),  # Hypothetical position size
                price=Decimal(str(row['Close'])),  # Use Close price
                required_margin=Decimal('0.0'),  # Hypothetical required margin
                fee=Decimal('0.0'),  # Hypothetical fee
                balance=Decimal('1000.0'),  # Hypothetical balance
                leverage=Decimal('100.0'),  # Hypothetical leverage
                free_margin=Decimal('1000.0')  # Hypothetical free margin
            )
            trade_record_manager.record_trade(trade_record)
            
            # Switch to closing operation
            is_long_open = False

        # Closing trade (LONG_CLOSE)
        else:
            trade_record = TradeRecord(
                timestamp=row['Date'],
                operation_type=Action.SHORT_CLOSE.name,  # Use Action.LONG_CLOSE
                position_size=Decimal('0.1'),  # Hypothetical position size
                price=Decimal(str(row['Close'])),  # Use Close price
                required_margin=Decimal('0.0'),  # Hypothetical required margin
                fee=Decimal('0.0'),  # Hypothetical fee
                balance=Decimal('1000.0'),  # Hypothetical balance
                leverage=Decimal('100.0'),  # Hypothetical leverage
                free_margin=Decimal('1000.0')  # Hypothetical free margin
            )
            trade_record_manager.record_trade(trade_record)
            
            # Switch to next opening trade
            is_long_open = True

    # Final trade record (LONG_OPEN)
    trade_record = TradeRecord(
        timestamp=df.iloc[-7]['Date'], 
        operation_type=Action.LONG_OPEN.name,  
        position_size=Decimal('0.1'),  # Hypothetical position size
        price=Decimal(str(df.iloc[-7]['Close'])),  # Use Close price
        required_margin=Decimal('0.0'),  # Hypothetical required margin
        fee=Decimal('0.0'),  # Hypothetical fee
        balance=Decimal('1000.0'),  # Hypothetical balance
        leverage=Decimal('100.0'),  # Hypothetical leverage
        free_margin=Decimal('1000.0')  # Hypothetical free margin
    )

    df.set_index('Date', inplace=True)

    # Record the final trade
    trade_record_manager.record_trade(trade_record)

    # Create a plotter for Bollinger Bands
    plotter = BollingerBandPlotter(df, trade_record_manager, balance=11100, window=20, fig_width=700, fig_height=500, dpi=100)
    
    # Create output directory if not exists
    os.makedirs('output', exist_ok=True)

    # Plot the Bollinger Bands and save to file
    plotter.plot(filename='output/bollinger_bands.png', show=True)

# Call the main function
if __name__ == '__main__':
    main()
