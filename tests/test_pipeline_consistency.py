#!/usr/bin/env python3
"""
Pipeline Consistency Test
Compare CSV vs Juejin-style pipeline outputs for the same input data.

This test:
1. Loads CSV data (training pipeline)
2. Simulates Juejin pipeline by adding trading_day column
3. Compares the resulting obs features
"""

import sys
sys.path.insert(0, "/home/v/Documents/work/gym-trading-env/src")

import pandas as pd
import numpy as np
from gym_trading_env.utils.market_features import (
    build_market_features,
    FEATURES_MARKET_OBS,
    FEATURES_MARKET,
)
from gym_trading_env.utils.session_futures_strict import strict_reindex_futures_345
from gym_trading_env.utils.ohlcvi_contract import normalize_ohlcvi
from gym_trading_env.utils.time_contract import ensure_feature_tz_index

TZ = "Asia/Shanghai"


def load_csv_data(csv_path: str, nrows: int = 5000) -> pd.DataFrame:
    """Load CSV like CsvBarSource does"""
    df = pd.read_csv(csv_path, nrows=nrows)
    df = normalize_ohlcvi(df, date_col="Date")
    df.index = ensure_feature_tz_index(df.index, assume_tz=TZ)
    return df


def add_trading_day_from_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate what strict_reindex_futures_345 would infer for trading_day.
    This is what CSV pipeline does (no trading_day column).
    """
    # Don't add trading_day column - let strict_reindex infer it
    return df.copy()


def add_trading_day_from_db_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate what DB provides for trading_day.
    Night session (21:00-23:59) -> next business day
    Day session -> same day
    """
    df = df.copy()
    
    trading_days = []
    for ts in df.index:
        h = ts.hour
        d = ts.normalize()
        
        if h >= 21:
            # Night session -> next trading day
            # Simple: add 1 day, skip weekend
            next_d = d + pd.Timedelta(days=1)
            if next_d.weekday() == 5:  # Saturday
                next_d += pd.Timedelta(days=2)
            elif next_d.weekday() == 6:  # Sunday
                next_d += pd.Timedelta(days=1)
            trading_days.append(int(next_d.strftime("%Y%m%d")))
        else:
            # Day session -> same day
            trading_days.append(int(d.strftime("%Y%m%d")))
    
    df["trading_day"] = trading_days
    return df


def run_csv_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Run CSV pipeline (no trading_day, no limit prices)"""
    return build_market_features(
        df_raw,
        tz=TZ,
        rollover_hour_local=5,  # This is ignored for futures
        is_future=True,
        # limit_up_pct=None,  # CSV doesn't pass this
        # limit_down_pct=None,
    )


def run_juejin_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Run Juejin pipeline (with trading_day, with limit prices)"""
    # Add trading_day column like DB would
    df_with_td = add_trading_day_from_db_logic(df_raw)
    
    return build_market_features(
        df_with_td,
        tz=TZ,
        is_future=True,
        limit_up_pct=0.08,  # JM limit is 8%
        limit_down_pct=0.08,
    )


def compare_features(df_csv: pd.DataFrame, df_juejin: pd.DataFrame, tol: float = 1e-6):
    """Compare obs features between two pipelines"""
    
    print("=" * 60)
    print("PIPELINE CONSISTENCY COMPARISON")
    print("=" * 60)
    
    # Check index alignment
    if not df_csv.index.equals(df_juejin.index):
        print("\n❌ INDEX MISMATCH!")
        print(f"CSV index: {len(df_csv)} rows, {df_csv.index.min()} ~ {df_csv.index.max()}")
        print(f"Juejin index: {len(df_juejin)} rows, {df_juejin.index.min()} ~ {df_juejin.index.max()}")
        
        # Find differences
        only_csv = df_csv.index.difference(df_juejin.index)
        only_juejin = df_juejin.index.difference(df_csv.index)
        print(f"Only in CSV: {len(only_csv)}")
        print(f"Only in Juejin: {len(only_juejin)}")
        
        # Align for comparison
        common_idx = df_csv.index.intersection(df_juejin.index)
        df_csv = df_csv.loc[common_idx]
        df_juejin = df_juejin.loc[common_idx]
        print(f"Comparing common rows: {len(common_idx)}")
    else:
        print(f"\n✅ Index aligned: {len(df_csv)} rows")
    
    # Check trading_day
    print("\n--- trading_day ---")
    td_csv = df_csv["trading_day"].astype(int)
    td_juejin = df_juejin["trading_day"].astype(int)
    td_diff = (td_csv != td_juejin).sum()
    if td_diff > 0:
        print(f"❌ trading_day mismatch: {td_diff} rows differ")
        mismatch_idx = df_csv.index[td_csv != td_juejin]
        for idx in mismatch_idx[:5]:
            print(f"  {idx}: CSV={td_csv.loc[idx]}, Juejin={td_juejin.loc[idx]}")
    else:
        print(f"✅ trading_day identical")
    
    # Check session_id
    print("\n--- session_id ---")
    sid_csv = df_csv["session_id"].astype(str)
    sid_juejin = df_juejin["session_id"].astype(str)
    sid_diff = (sid_csv != sid_juejin).sum()
    if sid_diff > 0:
        print(f"❌ session_id mismatch: {sid_diff} rows differ")
    else:
        print(f"✅ session_id identical")
    
    # Check mask
    print("\n--- mask_t ---")
    mask_csv = df_csv["mask_t"].astype(float)
    mask_juejin = df_juejin["mask_t"].astype(float)
    mask_diff = (mask_csv != mask_juejin).sum()
    if mask_diff > 0:
        print(f"❌ mask_t mismatch: {mask_diff} rows differ")
    else:
        print(f"✅ mask_t identical")
        print(f"   mask=1: {(mask_csv == 1).sum()}, mask=0: {(mask_csv == 0).sum()}")
    
    # Check OBS features
    print("\n--- OBS Features ---")
    obs_cols = [c for c in FEATURES_MARKET_OBS if c in df_csv.columns and c in df_juejin.columns]
    
    max_diffs = {}
    for col in obs_cols:
        a = df_csv[col].astype(float).fillna(0).to_numpy()
        b = df_juejin[col].astype(float).fillna(0).to_numpy()
        diff = np.abs(a - b)
        max_diff = diff.max()
        max_diffs[col] = max_diff
        
        if max_diff > tol:
            n_diff = (diff > tol).sum()
            print(f"❌ {col}: max_diff={max_diff:.6f}, n_diff={n_diff}")
        else:
            print(f"✅ {col}: max_diff={max_diff:.2e}")
    
    # Check RAW features (limit prices expected to differ)
    print("\n--- RAW Features (expected differences) ---")
    raw_cols = ["limit_up_price_t", "limit_down_price_t"]
    for col in raw_cols:
        if col in df_csv.columns and col in df_juejin.columns:
            a = df_csv[col].astype(float).fillna(0).to_numpy()
            b = df_juejin[col].astype(float).fillna(0).to_numpy()
            diff = np.abs(a - b)
            max_diff = diff.max()
            
            csv_nonzero = (a != 0).sum()
            juejin_nonzero = (b != 0).sum()
            print(f"  {col}: CSV nonzero={csv_nonzero}, Juejin nonzero={juejin_nonzero}, max_diff={max_diff:.2f}")
    
    # Summary
    print("\n" + "=" * 60)
    obs_all_match = all(d <= tol for d in max_diffs.values())
    if obs_all_match and td_diff == 0 and sid_diff == 0 and mask_diff == 0:
        print("✅ PIPELINES ARE CONSISTENT!")
    else:
        print("❌ PIPELINES HAVE DIFFERENCES!")
        if not obs_all_match:
            problematic = [k for k, v in max_diffs.items() if v > tol]
            print(f"   Problematic OBS columns: {problematic}")
    print("=" * 60)
    
    return obs_all_match and td_diff == 0


if __name__ == "__main__":
    csv_path = os.environ.get(
        "GYM_TRADING_PIPELINE_CONSISTENCY_CSV",
        "/data/logdir/trading_contracts/jm_walk_forward_20240603_20251202/"
        "data/raw/jm2601_18m_1m.csv",
    )
    
    print(f"Loading CSV: {csv_path}")
    df_raw = load_csv_data(csv_path, nrows=5000)
    print(f"Raw data: {len(df_raw)} rows, {df_raw.index.min()} ~ {df_raw.index.max()}")
    
    print("\n--- Running CSV Pipeline ---")
    df_csv = run_csv_pipeline(df_raw)
    print(f"CSV pipeline output: {len(df_csv)} rows")
    
    print("\n--- Running Juejin Pipeline ---")
    df_juejin = run_juejin_pipeline(df_raw)
    print(f"Juejin pipeline output: {len(df_juejin)} rows")
    
    compare_features(df_csv, df_juejin)
