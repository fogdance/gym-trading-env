# utils/market_features.py
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from gym_trading_env.utils.session_fx import compute_session_meta
from gym_trading_env.utils.session_futures_strict import strict_reindex_futures_345, DEFAULT_TZ

# Market-side features (sequence)
FEATURES_MARKET: List[str] = [
    "C_t",                    # Closing price at time t
    "V_t",                    # Volume at time t
    "I_t",                    # Open Interest at time t
    "cumVWAP_t",              # Cumulative VWAP up to time t
    "dC_minus_cumVWAP_t",     # C_t - cumVWAP_t
    "cmp_C_vs_cumVWAP_t",     # sign(C_t - cumVWAP_t) -> {-1,0,1}
    "ref_close_t",            # Reference closing price (e.g., previous day close)
    "session_high_t",         # Session high price up to time t
    "session_low_t",          # Session low price up to time t
    "bar_dir_t",              # Direction of the current bar (1 = up, -1 = down, 0 = flat)
    "turnover_t",             # cumulative sum of (price * volume) within session
    "minute_index_t",         # Minute index within the trading session
    "limit_up_price_t",       # Upper price limit at time t
    "limit_down_price_t",     # Lower price limit at time t
    "dI_from_yclose_t",       # Change in imbalance from yesterday's close
    "dP_from_ref_t",          # Price change from reference price
    "pct_chg_from_ref_t",     # Percentage change from reference price
    "mask_t",                 # Mask flag (e.g., valid data or trading halt)
    "weekday_sin_t",          # Sine-encoded weekday (for cyclical time feature)
    "weekday_cos_t",          # Cosine-encoded weekday (for cyclical time feature)
]

# 仅用于环境内部逻辑（止损/滑点/撮合等），不进 obs
AUX_MARKET_COLS = ["H_t", "L_t"]

# env 里会强依赖的列
REQUIRED_MARKET_COLS = ["day_id"] + AUX_MARKET_COLS + FEATURES_MARKET

def _weekday_cyc_from_sid(session_id: pd.Series) -> pd.DataFrame:
    # 将 session_id 统一转为字符串再解析，避免 dtype 干扰
    d = pd.to_datetime(session_id.astype(str), format="%Y%m%d", errors="coerce")
    wd = d.dt.weekday
    return pd.DataFrame({
        "weekday_sin_t": np.sin(2*np.pi*(wd/7.0)),
        "weekday_cos_t": np.cos(2*np.pi*(wd/7.0)),
    }, index=session_id.index)


def build_market_features(df_1m: pd.DataFrame,
                          tz: str = "Asia/Singapore",
                          rollover_hour_local: int = 5,
                          df_prev_session: Optional[pd.DataFrame] = None,
                          is_future: bool = False,
                          limit_up_pct: Optional[float] = None,
                          limit_down_pct: Optional[float] = None):
    need_cols = {"Open","High","Low","Close"}
    if not need_cols.issubset(df_1m.columns):
        raise ValueError(f"df_1m must contain {need_cols}")
    
    if is_future:
        return _build_market_future(df_1m, tz, df_prev_session, limit_up_pct, limit_down_pct)
    else:
        return _build_market_fx(df_1m, tz, rollover_hour_local)

def _build_market_fx(df_1m: pd.DataFrame,
                     tz: str = "Asia/Singapore",
                     rollover_hour_local: int = 5):
    df = df_1m.copy()
    # 兼容老测试：若存在 Date 列则提为索引
    if not isinstance(df.index, pd.DatetimeIndex) and "Date" in df.columns:
        df.index = pd.to_datetime(df["Date"])
        df.drop(columns=["Date"], inplace=True)

    meta = compute_session_meta(df, tz=tz, rollover_hour_local=rollover_hour_local)
    sid = meta["session_id"]

    df["L_t"] = df["Low"].astype(float)
    df["H_t"] = df["High"].astype(float)
    df["C_t"] = df["Close"].astype(float)
    df["V_t"] = df.get("Volume", pd.Series(0, index=df.index)).fillna(0).astype(float)
    df["I_t"] = 0.0

    cv = (df["C_t"] * df["V_t"]).astype(float)
    csum_cv = cv.groupby(sid).cumsum()
    csum_v  = df["V_t"].groupby(sid).cumsum()
    cum_vwap = csum_cv / csum_v.replace(0, np.nan)
    cum_mean_c = df["C_t"].groupby(sid).expanding().mean().reset_index(level=0, drop=True)
    df["cumVWAP_t"] = cum_vwap.fillna(cum_mean_c)

    # cumVWAP_t 已计算完毕
    df["dC_minus_cumVWAP_t"] = (df["C_t"] - df["cumVWAP_t"]).astype(float)
    df["cmp_C_vs_cumVWAP_t"] = np.sign(df["dC_minus_cumVWAP_t"]).astype(int)

    df["ref_close_t"] = meta["prev_session_close"].astype(float)
    df["session_high_t"] = df["High"].groupby(sid).cummax().astype(float)
    df["session_low_t"]  = df["Low"].groupby(sid).cummin().astype(float)

    prev_close = df["Close"].shift(1)
    df["bar_dir_t"] = np.sign(df["Close"] - prev_close).fillna(0).astype(int)

    # turnover/amount: cumsum(C*V) within session
    cv = (df["C_t"] * df["V_t"]).astype(float)
    df["turnover_t"] = cv.groupby(sid).cumsum().astype(float)

    df["minute_index_t"] = meta["minute_index"].astype(int)

    df["limit_up_price_t"] = 0.0
    df["limit_down_price_t"] = 0.0
    df["dI_from_yclose_t"] = 0.0

    dP = df["C_t"] - df["ref_close_t"]
    df["dP_from_ref_t"] = dP.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = df["C_t"] / df["ref_close_t"] - 1.0
    df["pct_chg_from_ref_t"] = pd.Series(pct).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    df["weekday_sin_t"] = meta["weekday_sin"].astype(float)
    df["weekday_cos_t"] = meta["weekday_cos"].astype(float)

    has_ohlc = df[["Open","High","Low","Close"]].notna().all(axis=1).astype(int)
    no_gap = (1 - meta["is_hard_gap"]).astype(int)
    df["mask_t"] = (has_ohlc * no_gap).astype(float)
    df["day_id"] = meta["day_id"].astype(np.int32)

    for col in FEATURES_MARKET:
        df[col] = df[col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df[REQUIRED_MARKET_COLS]

def _build_market_future(df_1m: pd.DataFrame,
                         tz: str = "Asia/Singapore",
                         df_prev_session: Optional[pd.DataFrame] = None,
                         limit_up_pct: Optional[float] = None,
                         limit_down_pct: Optional[float] = None):

    # 1) 严格 345 对齐（含 canonical 索引）
    base = strict_reindex_futures_345(df_1m, tz=tz)
    X = base["aligned"].copy()
    mask_t = base["mask"].astype(int)
    session_id = base["session_id"]
    minute_index = base["minute_index"]
    day_id = base["day_id"]

    # 2) 基础价量持仓
    X["C_t"] = X["Close"].astype(float)
    X["H_t"] = X["High"].astype(float)
    X["L_t"] = X["Low"].astype(float)
    X["V_t"] = X.get("Volume", 0.0).astype(float)
    X["I_t"] = X.get("OpenInterest", 0.0).astype(float)

    # 3) 参照价（上一交易日收盘）
    if df_prev_session is not None and len(df_prev_session) > 0:
        prev = strict_reindex_futures_345(df_prev_session, tz=tz)
        prev_last_close = float(prev["aligned"]["Close"].tail(1).iloc[0])
        ref_close_t = pd.Series(np.nan, index=X.index, dtype=float)

        first_sid = session_id.iloc[0]
        first_mask = (session_id == first_sid)
        ref_close_t.loc[first_mask] = prev_last_close

        sid_last_close = X.loc[minute_index == 344, ["Close"]].copy()
        sid_last_close["session_id"] = session_id[minute_index == 344]
        sid_last_close = sid_last_close.set_index("session_id").rename(columns={"Close": "prev_close"})
        prev_map = sid_last_close["prev_close"].shift(1)
        ref_close_rest = session_id.map(prev_map).astype(float)
        ref_close_t = ref_close_t.fillna(ref_close_rest)
    else:
        sid_last_close = X.loc[minute_index == 344, ["Close"]].copy()
        sid_last_close["session_id"] = session_id[minute_index == 344]
        sid_last_close = sid_last_close.set_index("session_id").rename(columns={"Close": "prev_close"})
        prev_map = sid_last_close["prev_close"].shift(1)
        ref_close_t = session_id.map(prev_map).astype(float)

        first_sid = session_id.iloc[0]
        first_val = float(X.loc[session_id == first_sid, "Close"].iloc[0])
        ref_close_t.loc[session_id == first_sid] = ref_close_t.loc[session_id == first_sid].fillna(first_val)

    # 使用 ffill/bfill 避免 FutureWarning
    X["ref_close_t"] = ref_close_t.ffill().bfill().astype(float)

    # 4) cumVWAP（用 mask 过滤）
    valid_V = X["V_t"] * mask_t
    valid_C = X["C_t"]
    csum_cv = (valid_C * valid_V).groupby(session_id).cumsum()
    csum_v  = valid_V.groupby(session_id).cumsum()
    with np.errstate(divide="ignore", invalid="ignore"):
        cum_vwap = csum_cv / csum_v
    c_valid = X["C_t"].where(mask_t == 1, np.nan)
    cum_mean_c = c_valid.groupby(session_id).expanding().mean().reset_index(level=0, drop=True)
    X["cumVWAP_t"] = pd.Series(cum_vwap).fillna(cum_mean_c).fillna(0.0).astype(float)

    # cumVWAP_t 已计算完毕
    X["dC_minus_cumVWAP_t"] = (X["C_t"] - X["cumVWAP_t"]).astype(float)
    X["cmp_C_vs_cumVWAP_t"] = np.sign(X["dC_minus_cumVWAP_t"]).astype(int)

    # 5) session 高/低（只用有效分钟更新）
    H_valid = X["High"].where(mask_t == 1, np.nan)
    L_valid = X["Low"].where(mask_t == 1, np.nan)
    session_high = H_valid.groupby(session_id).cummax()
    session_low  = L_valid.groupby(session_id).cummin()
    X["session_high_t"] = session_high.groupby(session_id).ffill().fillna(0.0).astype(float)
    X["session_low_t"]  = session_low.groupby(session_id).ffill().fillna(0.0).astype(float)

    # 6) bar_dir（相邻有效收盘方向；跨日/无效置 0）
    prev_close = X["C_t"].shift(1)
    prev_mask = mask_t.shift(1).fillna(0)
    raw_dir = np.sign(X["C_t"] - prev_close).astype(float)
    bar_dir = np.where((mask_t == 1) & (prev_mask == 1), raw_dir, 0.0)
    bar_dir = np.where(minute_index == 0, 0.0, bar_dir)
    X["bar_dir_t"] = bar_dir.astype(int)


    # 7) turnover/amount: cumsum(C*V) within session (invalid minutes contribute 0 via mask)
    valid_V = X["V_t"] * mask_t
    cv = (X["C_t"] * valid_V).astype(float)
    X["turnover_t"] = cv.groupby(session_id).cumsum().astype(float)


    # 8) 分钟索引、weekday
    X["minute_index_t"] = minute_index.astype(int)
    wd = _weekday_cyc_from_sid(session_id)
    X["weekday_sin_t"] = wd["weekday_sin_t"].astype(float)
    X["weekday_cos_t"] = wd["weekday_cos_t"].astype(float)

    # 9) 涨跌停（如无参数则置 0）
    if (limit_up_pct is not None) and (limit_down_pct is not None):
        up = float(limit_up_pct)
        down = float(limit_down_pct)

        # Contract: both must be positive magnitudes in [0, 1]
        if not (0.0 <= up <= 1.0):
            raise ValueError(f"limit_up_pct must be in [0, 1], got {limit_up_pct}")
        if not (0.0 <= down <= 1.0):
            raise ValueError(f"limit_down_pct must be in [0, 1], got {limit_down_pct}")

        X["limit_up_price_t"] = X["ref_close_t"] * (1.0 + up)
        X["limit_down_price_t"] = X["ref_close_t"] * (1.0 - down)
    else:
        X["limit_up_price_t"] = 0.0
        X["limit_down_price_t"] = 0.0


    # 10) dI_from_yclose（上一交易日最后 OI）
    oi_last = X.loc[minute_index == 344, ["I_t"]].copy()
    oi_last["session_id"] = session_id[minute_index == 344]
    oi_last = oi_last.set_index("session_id").rename(columns={"I_t":"prev_oi"})
    prev_oi_map = oi_last["prev_oi"].shift(1)
    I_yclose = session_id.map(prev_oi_map).astype(float)
    first_sid = session_id.iloc[0]
    first_I0 = float(X.loc[session_id == first_sid, "I_t"].iloc[0])
    I_yclose.loc[session_id == first_sid] = I_yclose.loc[session_id == first_sid].fillna(first_I0)
    X["dI_from_yclose_t"] = (X["I_t"] - I_yclose).fillna(0.0).astype(float)

    # 11) dP / pct_chg
    X["dP_from_ref_t"] = (X["C_t"] - X["ref_close_t"]).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = X["C_t"] / X["ref_close_t"] - 1.0
    X["pct_chg_from_ref_t"] = pd.Series(pct).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    # 12) mask / day_id
    X["mask_t"] = mask_t.astype(float)
    X["day_id"] = day_id.astype(np.int32)

    # === 12.5) 统一按 mask 抹零（无效分钟所有特征都为 0；保留 mask_t 自身）===
    m = X["mask_t"].astype(float)
    for col in FEATURES_MARKET + AUX_MARKET_COLS:
        if col == "mask_t":
            continue
        X[col] = (X[col].astype(float) * m).astype(float)

    # 13) 清洗并返回
    for col in FEATURES_MARKET:
        if col not in X.columns:
            X[col] = 0.0
        X[col] = X[col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X[REQUIRED_MARKET_COLS]