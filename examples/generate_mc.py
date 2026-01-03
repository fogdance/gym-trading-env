from __future__ import annotations

from pathlib import Path
import pandas as pd

from gym_trading_env.utils.timebase import FEATURE_TZ
from gym_trading_env.utils.ohlcvi_contract import normalize_ohlcvi

from gym_trading_env.monte_carlo import (
    build_session_library,
    DayBlockBootstrapConfig,
    generate_synth_path_day_block,
    save_session_library_npz,
    load_session_library_npz,
    write_synth_csv,
)


RAW_CSV = Path("data/DCE_JM2601_1m.csv")
CACHE_NPZ = Path("cache/DCE_JM2601_sessions_345.npz")
OUT_CSV = Path("data/MC_DCE_JM2601_1m.csv")


def load_futures_1m_csv(path: Path) -> pd.DataFrame:
    """
    兼容两种格式：
      A) 有 header: Date,Open,High,Low,Close,Volume,OpenInterest(可选),trading_day(可选)
      B) 无 header: 7 列 (Date,Open,High,Low,Close,Volume,OpenInterest)
    返回：
      - index: tz-aware (FEATURE_TZ)
      - cols: Open/High/Low/Close/Volume/OpenInterest (+ 其他列保留)
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    # 先按“有 header”读
    df = pd.read_csv(path)

    need = {"Open", "High", "Low", "Close"}
    if not need.issubset(set(df.columns)):
        # fallback：按无 header 读（你贴的样例就是这种）
        df = pd.read_csv(path, header=None)
        if df.shape[1] < 6:
            raise ValueError(f"CSV columns too few: {df.shape[1]} in {path}")

        # 典型：Date,O,H,L,C,V,OI
        if df.shape[1] >= 7:
            df.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInterest"] + [
                f"extra_{i}" for i in range(df.shape[1] - 7)
            ]
        else:
            # 没有 OI 的情况（不推荐，但兼容）
            df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"] + [
                f"extra_{i}" for i in range(df.shape[1] - 6)
            ]

    # Date 列兼容：有些文件第一列叫 Unnamed: 0
    if "Date" not in df.columns:
        if "Unnamed: 0" in df.columns:
            df.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
        else:
            # 最保守：把第一列当 Date
            first = df.columns[0]
            df.rename(columns={first: "Date"}, inplace=True)

    # 建 index + tz
    idx = pd.to_datetime(df["Date"], errors="coerce")
    if idx.isna().any():
        bad_n = int(idx.isna().sum())
        raise ValueError(f"Date parse failed rows={bad_n}, path={path}")

    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize(FEATURE_TZ)
    else:
        idx = idx.tz_convert(FEATURE_TZ)

    df = df.drop(columns=["Date"], errors="ignore").copy()
    df.index = idx
    df = df.sort_index()

    # 标准化 OHLCVI（缺 Volume/OI 自动补 0，但你们 futures 不建议缺）
    df = normalize_ohlcvi(df)

    return df


def main():
    CACHE_NPZ.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # 0) 读 8 年原始 1m OHLCVI
    df_8y_raw = load_futures_1m_csv(RAW_CSV)

    # 1) 构建/加载 session library（只保留“完整 345 分钟且 OHLCVI 全有”的 session）
    if CACHE_NPZ.exists():
        lib = load_session_library_npz(str(CACHE_NPZ))
    else:
        lib = build_session_library(df_8y_raw, min_valid_ratio=1.0)  # 1.0 = 必须满 345
        save_session_library_npz(lib, str(CACHE_NPZ))

    # 2) 生成 1 条“1个月”路径（含 warmup）
    cfg = DayBlockBootstrapConfig(
        warmup_days=7,
        eval_days=20,
        block_size_days=5,
        seed=123,
        relink_prices=True,  # 只缩放 OHLC，让跨日 ref_close 等正确
    )
    path = generate_synth_path_day_block(lib, cfg)

    # 3) 写出到 data/MC_1m.csv（CsvBarSource 读 data/{symbol}_{interval}.csv）
    #    => config.trading.data_path = "MC"; config.trading.data_interval = "1m"
    write_synth_csv(path.df_raw, str(OUT_CSV))

    print(f"[OK] wrote synth csv: {OUT_CSV}")
    print("Hint: set config.trading.data_path='MC', data_interval='1m', data_tz='Asia/Shanghai' then run env.")


if __name__ == "__main__":
    main()
