# tests/test_futures_strict_345.py
import unittest
import numpy as np
import pandas as pd

from gym_trading_env.utils.build_xt import build_market_features, FEATURES_MARKET
import pytest
pytestmark = pytest.mark.unit

TZ = "Asia/Shanghai"

def _mk_df(rows):
    return pd.DataFrame(rows).set_index("Date")

def make_intraday_rows(start_ts: str, minutes: int, price_base=2300.0, vol=10.0, oi=1000.0):
    """
    生成从 start_ts 开始、连续 `minutes` 个“分钟收盘时刻”的行。
    注意：为了与严格 345 的“收盘时刻”对齐，夜盘首个分钟应从 21:01，日盘从 09:01 开始。
    """
    idx = pd.date_range(start=start_ts, periods=minutes, freq="1min", tz=TZ)
    c = price_base + np.arange(minutes).astype(float) * 1.0  # 每分钟+1，便于断言
    rows = {
        "Date": idx,
        "Open": c,
        "High": c,
        "Low":  c,
        "Close": c,
        "Volume": np.full(minutes, vol, dtype=float),
        "OpenInterest": np.full(minutes, oi, dtype=float),
    }
    return _mk_df(rows)


def _canonicalize_futures_index_like(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """
    把 df 的索引按期货“交易日”规则规范化（夜盘21-23→次日；凌晨0-5→前一日），
    返回一个新 df，索引 = canonical_index（tz-aware），列保留原始。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be DatetimeIndex")
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    else:
        idx = idx.tz_convert(tz)

    # 交易日映射
    d = idx.normalize()
    h = idx.hour
    trading_day = d.where(~((h>=21)&(h<=23)), d + pd.Timedelta(days=1))
    trading_day = trading_day.where(~((h>=0)&(h<=5)), d - pd.Timedelta(days=1))

    # canonical: 用“交易日日期00:00”+ 原时分秒
    time_of_day = idx - idx.normalize()
    canonical = trading_day.normalize() + time_of_day

    out = df.copy()
    out.index = canonical
    return out

def _assert_input_equals_output_on_data_minutes(df_1m: pd.DataFrame, out: pd.DataFrame, tz: str):
    """
    断言：在 df_1m 有数据的分钟，out 的各列与之完全一致：
      C_t == Close, V_t == Volume(缺省0), I_t == OpenInterest(缺省0), mask_t == 1
    out 来自 build_market_features(..., is_future=True) 的结果。
    """
    # 1) 规范化输入索引为 canonical
    df_can = _canonicalize_futures_index_like(df_1m, tz)

    # 2) 取出 out 里我们要核对的列（和 df_can 对齐）
    cols_in = {
        "C_t": "Close",
        "V_t": "Volume",
        "I_t": "OpenInterest",
    }
    # 若输入里缺失 Volume / OpenInterest，则当 0 处理以匹配产线逻辑
    tmp_in = pd.DataFrame(index=df_can.index)
    tmp_in["Close"] = df_can["Close"].astype(float)
    tmp_in["Volume"] = df_can["Volume"].astype(float) if "Volume" in df_can.columns else 0.0
    tmp_in["OpenInterest"] = df_can["OpenInterest"].astype(float) if "OpenInterest" in df_can.columns else 0.0

    tmp_out = out[["C_t", "V_t", "I_t", "mask_t"]].copy()

    # 3) 对齐到公共索引（输入的 canonical 分钟应该都在 out 的 345 时钟上）
    common_idx = tmp_in.index.intersection(tmp_out.index)
    assert len(common_idx) == len(tmp_in.index), (
        f"有 {len(tmp_in.index) - len(common_idx)} 个输入分钟不在 345 时钟上，请检查索引或时区/夜盘映射。"
    )

    # 4) 数值一致性检查
    for out_col, in_col in cols_in.items():
        a = tmp_out.loc[common_idx, out_col].to_numpy(dtype=float)
        b = tmp_in.loc[common_idx, in_col].to_numpy(dtype=float)
        np.testing.assert_allclose(a, b, rtol=0, atol=0, err_msg=f"{out_col} != {in_col} on data minutes")

    # 5) mask 必须为 1
    m = tmp_out.loc[common_idx, "mask_t"].to_numpy(dtype=float)
    assert np.all(m == 1.0), "mask_t 必须在有数据的分钟为 1"


class TestFuturesStrict345(unittest.TestCase):

    def test_case1_day_start_5min_only(self):
        """
        1）仅有日盘数据：从 09:01 开始的 5 分钟。
           期望：输出 345 行；夜盘全部 0 且 mask=0；
                09:01..09:05 这 5 分钟 mask=1，其余日盘未到分钟为 0 且 mask=0。
        """
        df_1m = make_intraday_rows("2024-05-20 09:01:00", minutes=5)  # 09:01..09:05
        out = build_market_features(df_1m, tz=TZ, is_future=True)
        X = out[FEATURES_MARKET].to_numpy()
        mask_idx = FEATURES_MARKET.index("mask_t")
        minute_idx = out["minute_index_t"].to_numpy().astype(int)

        self.assertEqual(out.shape[0], 345, "每个交易日必须输出 345 行")

        # 夜盘段（minute 0..119）应全部 mask=0
        night_mask = X[0:120, mask_idx]
        self.assertTrue(np.all(night_mask == 0.0))

        # 日盘第一段 minute 范围：120..(120 + 75) 对应 09:01..10:15
        day1_start, day1_end = 120, 120 + 75
        day1_mask = X[day1_start:day1_end, mask_idx]
        # 前 5 分钟有数据，其余为 0
        self.assertTrue(np.all(day1_mask[:5] == 1.0))
        self.assertTrue(np.all(day1_mask[5:] == 0.0))

        # 其余两个日盘段（10:30..11:30 以及 13:30..15:00）也应全 0
        day2_start, day2_end = day1_end + 0, day1_end + 60
        day3_start, day3_end = day2_end + 0, day2_end + 90
        self.assertTrue(np.all(X[day2_start:day2_end, mask_idx] == 0.0))
        self.assertTrue(np.all(X[day3_start:day3_end, mask_idx] == 0.0))
        _assert_input_equals_output_on_data_minutes(df_1m, out, tz=TZ)

    def test_case2_night_first_10min_only(self):
        """
        2）仅有夜盘首 10 分钟：21:01..21:10。
           期望：minute_index 0..9 的 mask=1，其余 0；总行数仍为 345。
        """
        df_1m = make_intraday_rows("2024-05-20 21:01:00", minutes=10)  # 夜盘属于“次日交易日”
        out = build_market_features(df_1m, tz=TZ, is_future=True)
        X = out[FEATURES_MARKET].to_numpy()
        mask_idx = FEATURES_MARKET.index("mask_t")

        self.assertEqual(out.shape[0], 345)

        # 0..9 -> 有数据
        self.assertTrue(np.all(X[0:10, mask_idx] == 1.0))
        # 其余 -> 无数据
        self.assertTrue(np.all(X[10:, mask_idx] == 0.0))
        _assert_input_equals_output_on_data_minutes(df_1m, out, tz=TZ)

    def test_case3_night_present_plus_first_10min_day(self):
        """
        3）既有夜盘部分（21:01..21:10），又有日盘前 10 分钟（09:01..09:10）。
           期望：0..9 与 120..129 的 mask=1，其余 0。
        """
        df_night = make_intraday_rows("2024-05-20 21:01:00", minutes=10)  # 归到 2024-05-21 交易日
        df_day   = make_intraday_rows("2024-05-21 09:01:00", minutes=10, price_base=3000.0, oi=1200.0)
        df_1m = pd.concat([df_night, df_day]).sort_index()

        out = build_market_features(df_1m, tz=TZ, is_future=True)
        X = out[FEATURES_MARKET].to_numpy()
        mask_idx = FEATURES_MARKET.index("mask_t")

        self.assertEqual(out.shape[0], 345)

        # 0..9（夜盘首 10 分钟）=1；10..119=0
        self.assertTrue(np.all(X[0:10, mask_idx] == 1.0))
        self.assertTrue(np.all(X[10:120, mask_idx] == 0.0))

        # 120..129（日盘首 10 分钟）=1；其余 0
        self.assertTrue(np.all(X[120:130, mask_idx] == 1.0))
        self.assertTrue(np.all(X[130:, mask_idx] == 0.0))

        # 价格/量不应被“补零分钟”污染：检查 cumVWAP 在有效段单调合理
        cumvwap = out["cumVWAP_t"].to_numpy()
        self.assertGreater(cumvwap[9], 0.0)      # 夜盘有成交
        self.assertEqual(cumvwap[10], cumvwap[11])  # 10..119 为 0 → cumVWAP 前向保持
        self.assertGreater(cumvwap[129], cumvwap[120])  # 日盘前 10 分钟也应更新
        _assert_input_equals_output_on_data_minutes(df_1m, out, tz=TZ)

if __name__ == "__main__":
    unittest.main()
