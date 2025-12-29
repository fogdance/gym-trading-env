# src/gym_trading_env/utils/bar_source.py
from __future__ import annotations

from typing import Optional, Literal

from datetime import datetime, time as dt_time, timedelta

import pandas as pd
import numpy as np
from gym_trading_env.envs.config import TradingConfig  # 保持你原 import 口径
from gym_trading_env.utils.data_processing import load_data
from gym_trading_env.utils.market_features import build_market_features
from gym_trading_env.utils.market_store import MarketStore
from gym_trading_env.utils.timebase import FEATURE_TZ as DEFAULT_TZ, yyyymmdd_int

from gym_trading_env.config.settings import DB_CONFIG
from gym_trading_env.utils.timebase import to_local_ts_strict as _to_local_ts, localize_index_explicit, ensure_index_tz_strict, FEATURE_TZ
from gym_trading_env.utils.timebase import DB_TZ
from gym_trading_env.utils.ohlcvi_contract import normalize_ohlcvi
from gym_trading_env.utils.time_contract import ensure_feature_tz_index
from gym_trading_env.utils.time_contract import from_db_naive_dt, from_db_naive_series, to_db_naive_dt
from gym_trading_env.utils.ohlcvi_contract import normalize_ohlcvi

try:
    import pymysql
except ImportError as e:  # pragma: no cover
    # 不在 import 时就 crash，只有真正用 JuejinBarSource 时才会报错
    pymysql = None


BarSourceKind = Literal["csv", "juejin"]

# 掘金期货 1m bar 表名
JUEJIN_FUT_BAR_TABLE = "fut_bar_1m_v2"
JUEJIN_CONT_MAP_TABLE = "fut_continuous_map_v2"


def _normalize_ohlcvi_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible wrapper:
    - accepts Date column or DatetimeIndex
    - ensures OHLCVI schema (Open/High/Low/Close/Volume/OpenInterest)
    """
    return normalize_ohlcvi(df, date_col="Date")


class BaseBarSource:
    """
    v2: env reads ONLY from self.store (numpy stable arrays).
    df_raw/df_market remain for debug/visualization.

    Must provide:
      - df_raw
      - df_market
      - store : MarketStore
    """

    kind: BarSourceKind = "csv"

    def __init__(self, config: TradingConfig, df: Optional[pd.DataFrame] = None):
        self.config = config
        self.df_raw: pd.DataFrame = pd.DataFrame()
        self.df_market: pd.DataFrame = pd.DataFrame()
        self.store: Optional[MarketStore] = None

        self._build(df=df)
        self._validate_basic()

    def _build(self, df: Optional[pd.DataFrame]):
        raise NotImplementedError

    def _validate_basic(self):
        if self.store is None:
            raise RuntimeError("BarSource.store is None (build failed)")

        ws = int(getattr(self.config.training, "window_size", 1))
        if ws <= 0:
            raise ValueError(f"window_size must be > 0, got {ws}")

        is_future = bool(getattr(self.config.trading, "is_future", False))
        if is_future:
            if ws > int(self.store.day_len):
                raise ValueError(f"window_size={ws} > day_len={self.store.day_len} (futures). Clamp in env or config.")

        if len(self.store.days) < 1:
            raise ValueError("store has no days")

        # Use your canonical constant for env dependency
        from gym_trading_env.utils.market_features import REQUIRED_MARKET_COLS

        missing = set(REQUIRED_MARKET_COLS) - set(self.df_market.columns)
        if missing:
            raise ValueError(f"df_market missing REQUIRED_MARKET_COLS: {missing}")



class CsvBarSource(BaseBarSource):
    kind: BarSourceKind = "csv"

    def _build(self, df: Optional[pd.DataFrame]):
        # 1) Load raw OHLCVI
        if df is None:
            sym = self.config.trading.data_path
            interval = self.config.trading.data_interval
            df = load_data(sym, interval)

        # 2) normalize columns + index
        df = normalize_ohlcvi(df, date_col="Date")  # ensures Open/High/Low/Close/Volume/OpenInterest
        raw_tz = getattr(self.config.trading, "data_tz", None)

        df.index = ensure_feature_tz_index(df.index, assume_tz=raw_tz)

        self.df_raw = df

        # 3) Feature engineering
        self.df_market = build_market_features(
            self.df_raw,
            rollover_hour_local=5,
            is_future=bool(getattr(self.config.trading, "is_future", False)),
        )

        # 4) Build stable store (numpy arrays)
        self.store = MarketStore.from_frames(
            df_raw=self.df_raw,
            df_market=self.df_market,
            is_future=bool(getattr(self.config.trading, "is_future", False)),
            tz=DEFAULT_TZ,
            build_daily=bool(
                getattr(self.config.trading, "use_daily_context", False)
                or getattr(self.config.trading, "use_daily_seq_7", False)
            ),
        )

class JuejinBarSource(BaseBarSource):
    """
    期货实盘/回测专用 BarSource，从 MySQL 表 market_data.fut_bar_1m_v2 读取数据。

    约定：
      - 只支持期货（config.trading.is_future == True）
      - symbol 默认从 config.trading.symbol 或 config.trading.currency_pair 读取
      - 时间窗口：
          t1 = trading_date 15:01（当地时间，无时区）
          t0 = t1 - 14 天
        其中 trading_date：
          - 优先使用 config.trading.trading_date（可为 str/date/datetime）
          - 否则用当前本地日期（服务器时间）
      - 实盘时你可以在开盘前构造一次 env / BarSource，这个窗口固定，
        数据全部从 DB 读（HISTORY/EOD/LIVE 都写在一张表里），
    """

    kind: BarSourceKind = "juejin"

    def _build(self, df: Optional[pd.DataFrame]):
        is_future = bool(getattr(self.config.trading, "is_future", False))
        if not is_future:
            raise ValueError("JuejinBarSource currently only supports futures (config.trading.is_future must be True)")

        # symbol: ONLY future_symbol
        symbol = getattr(self.config.trading, "future_symbol", None)
        if not symbol:
            raise ValueError("JuejinBarSource requires config.trading.future_symbol")

        if df is not None:
            raise ValueError("JuejinBarSource does not accept df injection; use CsvBarSource or mock DB.")
        else:
            df_raw = self._load_from_db_initial_window()
        last_real_eob = _to_local_ts(df_raw.index.max())

        # 2) build df_market (strict futures 345 + canonical index)
        limit_up_pct = getattr(self.config.trading, "limit_up_pct", None)
        limit_down_pct = getattr(self.config.trading, "limit_down_pct", None)

        df_market = build_market_features(
            df_raw,
            tz=DEFAULT_TZ,
            is_future=True,
            limit_up_pct=limit_up_pct,
            limit_down_pct=limit_down_pct,
        )

        # 3) IMPORTANT: make df_raw canonical-grid aligned to df_market.index (fixed window)
        #    So live updates won't INSERT new rows and change index.
        df_raw = df_raw.reindex(df_market.index)

        # Optional: fill non-OHLC columns for convenience
        if "Volume" in df_raw.columns:
            df_raw["Volume"] = pd.to_numeric(df_raw["Volume"], errors="coerce").fillna(0.0).astype(float)
        else:
            df_raw["Volume"] = 0.0
        if "OpenInterest" in df_raw.columns:
            df_raw["OpenInterest"] = pd.to_numeric(df_raw["OpenInterest"], errors="coerce").fillna(0.0).astype(float)
        else:
            df_raw["OpenInterest"] = 0.0
            
        df_raw["trading_day"] = df_market["trading_day"].astype(np.int32)

        # Keep
        self.df_raw = df_raw
        self.df_market = df_market

        # 4) build stable store (single store for whole day)
        self.store = MarketStore.from_frames(
            df_raw=self.df_raw,
            df_market=self.df_market,
            is_future=True,
            tz=DEFAULT_TZ,
            build_daily=bool(
                getattr(self.config.trading, "use_daily_context", False)
                or getattr(self.config.trading, "use_daily_seq_7", False)
            ),
        )

        # ---- compute expected EOD eob for this trading_date (from canonical grid) ----
        if not hasattr(self, "_trading_date"):
            raise RuntimeError("missing _trading_date; ensure _load_from_db_initial_window sets it")


        td = yyyymmdd_int(self._trading_date)
        pos_td = np.flatnonzero(self.store.row_trading_day == td)

        if pos_td.size > 0:
            self._expected_eod_eob = _to_local_ts(self.store.index[int(pos_td.max())])
        else:
            self._expected_eod_eob = self._t1 - timedelta(minutes=1)

        # init _last_eob = REAL DB last bar, aligned to canonical store.index
        # do NOT use row_mask to init _last_eob; row_mask is "data exists" per-minute, not "latest DB waterline"
        idx = self.store.index
        # align/pad in case last_real_eob not exactly on the canonical grid
        pos = idx.get_indexer([last_real_eob], method="pad")
        if pos.size == 0 or int(pos[0]) < 0:
            self._last_eob = _to_local_ts(idx.min())
        else:
            self._last_eob = _to_local_ts(idx[int(pos[0])])

        if getattr(self.config.debug, "debug_enabled", False):
            # mask==0 的地方，raw Close 必须是 NaN（否则说明有人把缺失填成 0 了）
            idx0 = self.store.index[self.store.row_mask < 0.5]
            if len(idx0) > 0:
                bad = self.df_raw.loc[idx0, "Close"].notna().any()
                if bad:
                    raise RuntimeError("Contract broken: mask_t==0 but df_raw Close is not NaN (missing bars must stay NaN in df_raw)")



    # ------------------------------------------------------------------ #
    #   内部：从 fut_bar_1m_v2 读取 [t0, t1] 窗口的原始 OHLCVI
    # ------------------------------------------------------------------ #
    def _load_from_db_initial_window(self) -> pd.DataFrame:
            """
            Load raw 1m bars from MySQL within fixed window [t0, t1]:
            - t1 = trading_date 15:01 (FEATURE_TZ, tz-aware)
            - t0 = t1 - 14 days
            Returns: OHLCVI dataframe (index tz-aware FEATURE_TZ)
            """
            symbol = getattr(self.config.trading, "future_symbol", None)
            if not symbol:
                raise ValueError("JuejinBarSource requires config.trading.future_symbol")

            conn = self._connect()
            try:
                td = self._resolve_trading_date(conn, symbol)
                self._trading_date = td

                # ✅ MUST be tz-aware (FEATURE_TZ / DEFAULT_TZ)
                t1 = pd.Timestamp(datetime.combine(td, dt_time(15, 1, 0)), tz=DEFAULT_TZ)
                t0 = t1 - pd.Timedelta(days=14)

                # cache window for wait_kline_block
                self._t0 = t0
                self._t1 = t1

                t0_db = to_db_naive_dt(t0)
                t1_db = to_db_naive_dt(t1)

                with conn.cursor() as cursor:
                    sql = f"""
                        SELECT
                            trading_date,
                            symbol,
                            underlying,
                            bob,
                            eob,
                            `open`,
                            `high`,
                            `low`,
                            `close`,
                            volume,
                            `position`,
                            `source`,
                            provider,
                            created_at,
                            updated_at
                        FROM {JUEJIN_FUT_BAR_TABLE}
                        WHERE symbol = %s
                        AND eob >= %s
                        AND eob <= %s
                        ORDER BY eob ASC
                    """
                    cursor.execute(sql, (symbol, t0_db, t1_db))
                    rows = cursor.fetchall()
            finally:
                conn.close()

            if not rows:
                raise ValueError(
                    f"No rows found in {JUEJIN_FUT_BAR_TABLE} for symbol={symbol} "
                    f"between eob>={t0!s} and eob<={t1!s}"
                )

            df = pd.DataFrame(rows)

            # --- DB datetime(naive) -> tz-aware FEATURE_TZ ---
            df["eob"] = from_db_naive_series(df["eob"])
            df["bob"] = from_db_naive_series(df["bob"])

            df = df.dropna(subset=["eob"]).copy()
            df = df.sort_values("eob").set_index("eob")

            # 现在 index 是 eob，先把 trading_day 变成和 index 对齐的 Series
            td_series = pd.to_datetime(df["trading_date"]).dt.strftime("%Y%m%d").astype(np.int32)
            td_series.index = df.index

            # --- rename to standard OHLCVI ---
            df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                    "position": "OpenInterest",
                },
                inplace=True,
            )

            # Keep only rows with valid OHLC
            need_cols = ["Open", "High", "Low", "Close"]
            mask_valid = df[need_cols].notna().all(axis=1)
            if not mask_valid.any():
                raise ValueError(
                    f"All OHLC rows from {JUEJIN_FUT_BAR_TABLE} are NaN for symbol={symbol} between {t0} and {t1}"
                )
            df = df[mask_valid].copy()
            td_series = td_series.loc[df.index]

            # Enforce OHLCVI contract (float, missing VI -> 0)
            df = normalize_ohlcvi(df, date_col="Date")  # Date col ignored since index already set

            df["trading_day"] = td_series.astype(np.int32)
            

            return df

    def _connect(self):
        if pymysql is None:  # pragma: no cover
            raise ImportError(
                "pymysql is required for JuejinBarSource but is not installed. Please `pip install pymysql`."
            )

        return pymysql.connect(
            host=DB_CONFIG.host,
            port=DB_CONFIG.port,
            user=DB_CONFIG.user,
            password=DB_CONFIG.password,
            database=DB_CONFIG.db,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )

    def _resolve_trading_date(self, conn, symbol: str) -> datetime.date:
        """
        Resolve trading_date.

        Priority:
        1) config.trading.trading_date if provided
        2) live_mode=True: choose "target trading date" (today or next day after 15:01),
            then try to find the earliest trading_date >= target in fut_continuous_map_v2.
            If not found, RETURN target (do NOT fallback to past date).
        3) live_mode=False (backtest/offline): can fallback to latest available in map table.
        """
        trading_date_attr = getattr(self.config.trading, "trading_date", None)
        if trading_date_attr is not None:
            return pd.to_datetime(trading_date_attr).date()

        live_mode = bool(getattr(self.config.trading, "live_mode", False))

        now_local = pd.Timestamp.now(tz=DEFAULT_TZ)
        today = now_local.date()

        # 你们定义 t1=15:01，所以 15:01 后默认目标切到“下一天”
        target = today + timedelta(days=1) if now_local.time() >= dt_time(15, 1) else today

        with conn.cursor() as cursor:
            # 先找 >= target 的最小交易日（如果表里已提前写入，就能拿到真实 trading_date）
            cursor.execute(
                f"""
                SELECT DISTINCT trading_date
                FROM {JUEJIN_CONT_MAP_TABLE}
                WHERE symbol = %s
                AND trading_date >= %s
                ORDER BY trading_date ASC
                LIMIT 1
                """,
                (symbol, target),
            )
            row = cursor.fetchone()

            if row and row.get("trading_date") is not None:
                return pd.to_datetime(row["trading_date"]).date()

            if live_mode:
                # live 模式：宁可等，不回退到过去（避免你现在这种“周末跑周五整天历史数据”）
                return target

            # offline/backtest: fallback to latest available
            cursor.execute(
                f"""
                SELECT DISTINCT trading_date
                FROM {JUEJIN_CONT_MAP_TABLE}
                WHERE symbol = %s
                ORDER BY trading_date DESC
                LIMIT 1
                """,
                (symbol,),
            )
            row2 = cursor.fetchone()

        if row2 and row2.get("trading_date") is not None:
            return pd.to_datetime(row2["trading_date"]).date()

        return target

    def wait_kline_block(self, poll_interval: float = 1.0, lookback_minutes: int = 2) -> bool:
        """
        Block until DB has:
        - a newer bar (mx > _last_eob), OR
        - a real correction in lookback window (<= _last_eob)

        IMPORTANT (per your requirement):
        - This function NEVER returns "finished/end-of-day".
        - Exit logic must be handled by env (end_idx / eod_idx / episode rules).
        - Therefore this function returns True whenever it applied update/correction,
            and otherwise keeps blocking (sleep + poll).

        Returns:
        True -> applied update (new bar or correction)
        """
        if self.store is None:
            raise RuntimeError("wait_kline_block called before initial build (store is None)")

        symbol = getattr(self.config.trading, "future_symbol", None)
        if not symbol:
            raise ValueError("JuejinBarSource requires config.trading.future_symbol")

        if not hasattr(self, "_trading_date") or not hasattr(self, "_t1"):
            raise RuntimeError("Missing _trading_date/_t1. Ensure initial window build ran.")
        if not hasattr(self, "_expected_eod_eob"):
            raise RuntimeError("Missing _expected_eod_eob. Ensure _build computed expected EOD timestamp.")
        if not hasattr(self, "_last_eob"):
            self._last_eob = _to_local_ts(self.store.index.min())

        td = self._trading_date
        lookback = timedelta(minutes=int(lookback_minutes))

        limit_up_pct = getattr(self.config.trading, "limit_up_pct", None)
        limit_down_pct = getattr(self.config.trading, "limit_down_pct", None)

        cols_raw = ["Open", "High", "Low", "Close", "Volume", "OpenInterest"]

        def _db_max_eob(conn) -> Optional[pd.Timestamp]:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT MAX(eob) AS max_eob
                    FROM {JUEJIN_FUT_BAR_TABLE}
                    WHERE symbol=%s AND trading_date=%s AND eob <= %s
                    """,
                    (symbol, td, to_db_naive_dt(self._t1)),
                )
                row = cursor.fetchone()
            mx = row.get("max_eob") if row else None
            return from_db_naive_dt(mx) if mx is not None else None

        def _fetch_rows(conn, start_eob: pd.Timestamp, end_eob: pd.Timestamp) -> pd.DataFrame:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT eob, `open`, `high`, `low`, `close`, volume, `position`
                    FROM {JUEJIN_FUT_BAR_TABLE}
                    WHERE symbol=%s AND trading_date=%s AND eob >= %s AND eob <= %s
                    ORDER BY eob ASC
                    """,
                    (symbol, td, to_db_naive_dt(start_eob), to_db_naive_dt(end_eob)),
                )
                rows = cursor.fetchall()

            if not rows:
                return pd.DataFrame()

            d = pd.DataFrame(rows)
            d["eob"] = from_db_naive_series(d["eob"])
            d = d.dropna(subset=["eob"]).copy()
            d = d.set_index("eob").sort_index()
            if d.index.has_duplicates:
                d = d[~d.index.duplicated(keep="last")]

            d.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                    "position": "OpenInterest",
                },
                inplace=True,
            )

            d = normalize_ohlcvi(d)
            return d[cols_raw]

        def _has_real_change(idx: pd.DatetimeIndex, incoming: pd.DataFrame, eps: float = 1e-9) -> bool:
            if len(idx) == 0:
                return False
            cur = self.df_raw.loc[idx, cols_raw]
            a = cur.to_numpy(dtype=float, copy=False)
            b = incoming.loc[idx, cols_raw].to_numpy(dtype=float, copy=False)

            a_nan = np.isnan(a)
            b_nan = np.isnan(b)
            if np.any(a_nan != b_nan):
                return True

            da = np.abs(np.nan_to_num(a) - np.nan_to_num(b))
            return bool(np.any(da > eps))

        def _overwrite_df_inplace(dst: pd.DataFrame, src: pd.DataFrame):
            if (not dst.index.equals(src.index)) or (list(dst.columns) != list(src.columns)):
                raise RuntimeError("df_market schema/index changed; cannot inplace overwrite")
            # 覆写全部数据，不改变对象 id
            dst.iloc[:, :] = src.to_numpy(copy=False)

        def _apply_update(incoming: pd.DataFrame, idx_hit: pd.DatetimeIndex) -> None:
            # update df_raw (no index change)
            self.df_raw.loc[idx_hit, cols_raw] = incoming.loc[idx_hit, cols_raw].to_numpy()

            # full recompute df_market (reliable)
            new_df_market = build_market_features(
                self.df_raw,
                tz=DEFAULT_TZ,
                is_future=True,
                limit_up_pct=limit_up_pct,
                limit_down_pct=limit_down_pct,
            )
            if not new_df_market.index.equals(self.store.index):
                raise RuntimeError(
                    "df_market index changed; cannot in-place update store. "
                    "Ensure fixed window + strict_reindex_futures_345 is stable."
                )

            # impacted day blocks
            pos = self.store.index.get_indexer(idx_hit)
            pos = pos[pos >= 0]
            if pos.size == 0:
                raise RuntimeError("idx_hit exists but none found in store.index (index mismatch)")

            day_is = np.unique(self.store.row_day_i[pos]).astype(int)
            for di in day_is:
                s, e = self.store.day_ranges[int(di)]
                self.store.inplace_overwrite_day_from_df_market(int(di), new_df_market.iloc[s:e])

            _overwrite_df_inplace(self.df_market, new_df_market) 

        import time as _time

        conn = self._connect()
        try:
            while True:
                mx = _db_max_eob(conn)

                # DB has no bar yet for this trading_date -> keep waiting
                if mx is None:
                    _time.sleep(float(poll_interval))
                    continue

                # 1) New bars arrived: mx > _last_eob
                if mx > self._last_eob:
                    start = max(self._last_eob - lookback, self._t0)
                    incoming = _fetch_rows(conn, start, mx)
                    if incoming.empty:
                        _time.sleep(float(poll_interval))
                        continue

                    idx_hit = incoming.index.intersection(self.df_raw.index)
                    if len(idx_hit) == 0:
                        # This is serious: DB timestamps not on our canonical grid.
                        raise RuntimeError(
                            f"Incoming bars [{incoming.index.min()}..{incoming.index.max()}] not in df_raw.index. "
                            f"Check tz conversion / strict futures grid alignment."
                        )

                    _apply_update(incoming, idx_hit)

                    # ✅ Update DB-last cursor (latest arrived bar)
                    self._last_eob = mx
                    return True

                # 2) No new bar: check corrections within lookback
                start = max(self._last_eob - lookback, self._t0)
                incoming = _fetch_rows(conn, start, self._last_eob)
                if incoming.empty:
                    _time.sleep(float(poll_interval))
                    continue

                idx_hit = incoming.index.intersection(self.df_raw.index)
                if len(idx_hit) > 0 and _has_real_change(idx_hit, incoming):
                    _apply_update(incoming, idx_hit)
                    # ✅ correction only: _last_eob unchanged
                    return True

                # 3) Nothing changed -> keep blocking
                _time.sleep(float(poll_interval))
        finally:
            conn.close()
