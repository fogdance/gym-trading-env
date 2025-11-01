# utils/build_xt.py
import numpy as np
import pandas as pd
from typing import List
from .session_fx import compute_session_meta

# Market-side features (sequence)
FEATURES_MARKET: List[str] = [
    "C_t",                    # Closing price at time t
    "V_t",                    # Volume at time t
    "I_t",                    # Imbalance at time t
    "cumVWAP_t",              # Cumulative VWAP up to time t
    "ref_close_t",            # Reference closing price (e.g., previous day close)
    "session_high_t",         # Session high price up to time t
    "session_low_t",          # Session low price up to time t
    "bar_dir_t",              # Direction of the current bar (1 = up, -1 = down, 0 = flat)
    "turnover_t",             # Turnover (cumulative volume * price) at time t
    "minute_index_t",         # Minute index within the trading session
    "limit_up_price_t",       # Upper price limit at time t
    "limit_down_price_t",     # Lower price limit at time t
    "dI_from_yclose_t",       # Change in imbalance from yesterday's close
    "dP_from_ref_t",          # Price change from reference price
    "pct_chg_from_ref_t",     # Percentage change from reference price
    "mask_t",                 # Mask flag (e.g., valid data or trading halt)
    "roll_flag_t",            # Contract rollover flag
    "weekday_sin_t",          # Sine-encoded weekday (for cyclical time feature)
    "weekday_cos_t",          # Cosine-encoded weekday (for cyclical time feature)
]

# Agent-side features (single vector, emitted by env at runtime)
FEATURES_AGENT: List[str] = [
    "pos_t",                  # Current position size (positive = long, negative = short)
    "have_long_t",            # 1 if holding long position, 0 otherwise
    "have_short_t",           # 1 if holding short position, 0 otherwise
    "entry_price_t",          # Entry price of the current position
    "holding_minutes_t",      # Number of minutes the position has been held
    "upnl_t",                 # Unrealized PnL at time t
    "realized_pnl_step_t",    # Realized PnL in the current step
    "realized_pnl_cum_t",     # Cumulative realized PnL
    "fee_step_t",             # Trading fee in the current step
    "fee_cum_t",              # Cumulative trading fees
    "equity_t",               # Current equity (cash + unrealized PnL)
    "max_equity_t",           # Historical maximum equity (for drawdown calc)
    "drawdown_t",             # Current drawdown from peak equity
    "sigma_entry_t",          # Volatility estimate at entry time
    "sl_ticks_t",             # Stop-loss distance in ticks
    "tp_ticks_t",             # Take-profit distance in ticks
    "sl_price_t",             # Stop-loss price level
    "tp_price_t",             # Take-profit price level
    "minutes_to_timeout_t",   # Minutes remaining until position timeout
]

def build_market_features(df_1m: pd.DataFrame,
                          tz: str = "Asia/Singapore",
                          rollover_hour_local: int = 5) -> pd.DataFrame:
    """
    Convert 1-minute OHLCV into the market-side feature matrix.
    Assumes df_1m has columns: ['Open','High','Low','Close','Volume'] and a DatetimeIndex.
    """
    if not set(["Open","High","Low","Close"]).issubset(df_1m.columns):
        raise ValueError("df_1m must contain ['Open','High','Low','Close'] columns")
    df = df_1m.copy()

    # Compute session/meta
    meta = compute_session_meta(df, tz=tz, rollover_hour_local=rollover_hour_local)
    sid = meta["session_id"]

    # Base series
    df["C_t"] = df["Close"].astype(float)
    df["V_t"] = df.get("Volume", pd.Series(0, index=df.index)).fillna(0).astype(float)
    df["I_t"] = 0.0  # FX has no open interest

    # cumVWAP within session, fallback to cumulative mean if V == 0
    cv = (df["C_t"] * df["V_t"]).astype(float)
    csum_cv = cv.groupby(sid).cumsum()
    csum_v  = df["V_t"].groupby(sid).cumsum()
    cum_vwap = csum_cv / csum_v.replace(0, np.nan)
    cum_mean_c = df["C_t"].groupby(sid).expanding().mean().reset_index(level=0, drop=True)
    df["cumVWAP_t"] = cum_vwap.fillna(cum_mean_c)

    # ref close (prev session close)
    df["ref_close_t"] = meta["prev_session_close"].astype(float)

    # Intraday running high/low
    df["session_high_t"] = df["High"].groupby(sid).cummax().astype(float)
    df["session_low_t"]  = df["Low"].groupby(sid).cummin().astype(float)

    # Bar direction relative to previous minute
    prev_close = df["Close"].shift(1)
    df["bar_dir_t"] = np.sign(df["Close"] - prev_close).fillna(0).astype(int)

    # FX specifics
    df["turnover_t"] = 0.0
    df["limit_up_price_t"] = 0.0
    df["limit_down_price_t"] = 0.0
    df["dI_from_yclose_t"] = 0.0

    # Deltas vs reference
    dP = df["Close"] - df["ref_close_t"]
    df["dP_from_ref_t"] = dP.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = df["Close"] / df["ref_close_t"] - 1.0
    df["pct_chg_from_ref_t"] = pct.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    # Indices & flags
    df["minute_index_t"] = meta["minute_index"].astype(int)
    df["weekday_sin_t"] = meta["weekday_sin"].astype(float)
    df["weekday_cos_t"] = meta["weekday_cos"].astype(float)
    df["roll_flag_t"] = 0  # no contract roll for spot FX
    df["day_id"] = meta["day_id"].astype(np.int32)

    # mask_t: 1 if OHLC present and no hard gap; else 0
    has_ohlc = df[["Open","High","Low","Close"]].notna().all(axis=1).astype(int)
    no_gap = (1 - meta["is_hard_gap"]).astype(int)
    df["mask_t"] = (has_ohlc * no_gap).astype(int)

    # Fill NaNs only on feature columns (avoid touching day_id)
    for col in FEATURES_MARKET:
        if col in df.columns:
            df[col] = (
                df[col].astype(float)
                      .replace([np.inf, -np.inf], np.nan)
                      .fillna(0.0)
            )

    # Return features + auxiliary day_id needed by the env (NOT a learnable feature)
    out = df[FEATURES_MARKET + ["day_id"]]

    return out
