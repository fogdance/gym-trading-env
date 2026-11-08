"""Thin FG entry adapter over the existing accounting and atomic execution engine.

All fills are conservative minute simulations, not broker execution confirmations.
"""
from __future__ import annotations

import copy
import json
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from .action import ForexCode, TargetPos
from .accounting import JournalEntry, Posting
from .position import Position
from .trade_record import TradeRecord
from .trading_env import CustomTradingEnv
from .entry_contract import (PRODUCTS, ContractTerms, DirectionSchedule, fingerprint,
                             read_yaml, resolve_path, timestamp)
from gym_trading_env.utils.bar_source import CsvBarSource, normalize_csv_frame
from gym_trading_env.rewards.fast_take_profit_reward import FastTakeProfitReward
from gym_trading_env.utils.s3_market_clock import MarketClock, PROTOCOL as TIME_PROTOCOL


POSITION_OBS_FIELDS = (
    "direction", "position_ratio", "pending_ratio",
    "pos1_size", "pos1_tp_distance", "pos1_holding_age",
    "pos2_size", "pos2_tp_distance", "pos2_holding_age",
)
SINGLE3_POSITION_OBS_FIELDS = ("direction", "position_ratio_3", "sl_pending",
    *POSITION_OBS_FIELDS[3:])
SINGLE3_PROTOCOL = "fg_single3_v1"
SINGLE3_FILL_MODEL = "minute_tp5_sl30_risk_first_v1"
# The same minute fills (risk first) with the V2 structural SL / target-R TP.
STRUCTURAL_FILL_MODEL = "minute_structural_risk_first_v1"
# Fixed input scale of the TP distance in position_state (the default TP5),
# not the env's rule: a policy sees the same inputs under any TP / SL.
POSITION_TP_SCALE = 5
PRIOR_HIGH_KEYS = ('pivot_left', 'pivot_right', 'lookback_buckets', 'departure_points')
# entry.stop_rule values: the prior 15m high of a SHORT (V0.5), and the prior
# 15m extreme of either direction (V2.1, user 2026-11-04: the high of a SHORT,
# the low of a LONG).
STOP_RULES = ('prior_15m_high', 'prior_15m_extreme')


def _e_market_features(frame, schedule):
    """Keep the original visible-prefix calculator importable by spawn."""
    from gym_trading_env.utils.market_features import build_market_features
    return build_market_features(frame, is_future=True, real_futures_clock=True,
                                 futures_schedule=schedule)


class FixedDirectionEntryEnv(CustomTradingEnv):
    """WAIT/ADD_3 strategy; does not call or rewrite legacy target-position step."""
    metadata = {"render_modes": []}
    # TP5 / SL30 points; also the default of envs pickled before the keys existed.
    take_profit_points = 5
    stop_loss_points = 30
    # Structural SL / target-R TP (V2, user 2026-10-31; LONG too in V2.1): None
    # keeps the fixed distances above. Complete 15m buckets in the E context.
    stop_rule = None
    context_15m_buckets = 24
    _lines = None
    # V2.3 rule gate (user 2026-11-05): {SHORT, LONG} rules of the V4
    # material; None opens an ADD wherever the account allows it.
    rule_gate = None
    _gates = None
    # V3 long trend exit (user 2026-11-06, research.swing_trail): no TP, the
    # stop moves with the 15m troughs (peaks) after the fill; None keeps the TP.
    exit_rule = None
    no_reentry_after_stop = False
    _swings = None

    def __init__(self, df=None, config_path=None, render_mode=None, *, market_cache=None, e_profile=None):
        gym.Env.__init__(self)
        if config_path is None:
            raise ValueError("Entry config_path is required")
        raw = read_yaml(config_path)
        if raw.get("strategy_mode") != "entry_only":
            raise ValueError("Explicit strategy_mode=entry_only is required")
        self.entry_config = raw["entry"]
        self.protocol = self.entry_config.get("protocol", "fg_entry_v1")
        if self.protocol not in ("fg_entry_v1", SINGLE3_PROTOCOL):
            raise ValueError("Unknown FG entry protocol")
        self.single3 = self.protocol == SINGLE3_PROTOCOL
        if e_profile is not None and not self.single3:
            raise ValueError('S3 E observation is Single3-only')
        if e_profile is not None and self.entry_config['contract'] not in e_profile.contracts:
            raise ValueError('E observation profile refuses an unadmitted contract')
        self._e_profile = e_profile
        self._e_market = None
        self.max_lots, self.max_entries = (3, 1) if self.single3 else (6, 2)
        self.fill_model = ((STRUCTURAL_FILL_MODEL if self.entry_config.get('stop_rule') else SINGLE3_FILL_MODEL)
                           if self.single3 else "minute_close_confirm_v1")
        self.contract = self.entry_config["contract"]
        self.strategy_path=None
        if self.entry_config.get('strategy_config'):
            self.strategy_path=resolve_path(config_path,self.entry_config['strategy_config'])
            master=read_yaml(self.strategy_path)
            expected_reward = master.get('protocol') == SINGLE3_PROTOCOL if self.single3 else master['reward']=={
                    'formula':'60/(60+T_minutes)', 'complete_quantity':3,'take_profit_points':5}
            if (master['product'] not in PRODUCTS or not expected_reward or master['expiry']['mode']!='main_qualification'):
                raise ValueError('Unified FG configuration changes the frozen business contract')
            for key,value in master['environment']['trading'].items():
                if raw['trading'].get(key)!=value:
                    raise ValueError(f'Stale derived FG snapshot: trading.{key}; rebuild from fg.yaml')
            if raw['entry'].get('chunk_steps')!=master['environment']['entry']['chunk_steps']:
                raise ValueError('Stale derived FG chunk configuration')
            self.directions=DirectionSchedule(master['directions'])
        else:
            master=None
            self.directions = DirectionSchedule(read_yaml(resolve_path(config_path, self.entry_config["direction_config"])))
        self.market_clock = master.get('market_clock') if master is not None else self.entry_config.get('market_clock')
        terms_raw=read_yaml(resolve_path(config_path, self.entry_config["contract_config"]))
        if master is not None:
            terms_raw['lot_size']=master['costs']['lot_size']
            for period in terms_raw['terms']:
                period.update({k:master['costs'][k] for k in ('margin_rate','open_fee','close_fee',
                    'close_today_fee','spread','slippage')})
        self.terms = ContractTerms(terms_raw, self.contract)
        if e_profile is not None or self.entry_config.get('stop_rule'):
            self._e_schedule = dict(self.market_clock or {}, dated_sessions=terms_raw['sessions'])
        identity = {"env": raw, "strategy":master,"direction": self.directions.hash,
                    "contract": self.terms.hash,
                    "position_obs": SINGLE3_POSITION_OBS_FIELDS if self.single3 else POSITION_OBS_FIELDS,
                    "market_clock_version": "planned_345_v1"}
        if self.single3:
            from gym_trading_env.rewards.single3_reward import Single3RewardConfig
            self.reward_config = Single3RewardConfig(**(master['reward'] if master else raw['entry']['reward']))
            identity.update(protocol=self.protocol, fill_model=self.fill_model,
                            reward_hash=self.reward_config.hash, clock_protocol=TIME_PROTOCOL,
                            market_window='last_60_valid_bars', age_normalization='log1p_market_bars/43200')
        self.config_hash = fingerprint(identity)
        self.observation_identity = fingerprint(dict(protocol=self.protocol,
            fields=identity['position_obs'], reward=self.reward_config.hash if self.single3 else 'V1',
            **({'clock_protocol': TIME_PROTOCOL, 'market_window': 'last_60_valid_bars'} if self.single3 else {})))
        self._config(config_path)
        tr = self.config.trading
        if (tr.live_mode and not self.single3) or not tr.is_future or tr.future_symbol != self.contract:
            raise ValueError("Entry v1 requires a matching FG futures config (live: Single3 only)")
        # The base env's own order hook stays off: entry_live sends Single3 fills.
        self._live, self._order_client = None, None
        if tr.live_mode:
            # Live mode (user 2026-10-30): DB bars, blocking step, order RPC.
            from .entry_live import LiveFeed
            self._live = LiveFeed(self.contract, self.entry_config.get('live', {}),
                                  lambda m: print(m, flush=True))
        if (tr.initial_balance, tr.trade_lot, tr.max_long_position, tr.max_short_position) != (300000, 3, self.max_lots, self.max_lots):
            raise ValueError(f"Entry contract requires 300000 CNY, 3 lots per ADD and max {self.max_lots}")
        if self.single3 and raw['trading'].get('stop_loss_enabled') is not True:
            raise ValueError("Single3 snapshot must explicitly enable SL30")
        # These are strategy properties; do not inherit daily caps or strategy exits.
        tr.intraday_single_position = False
        tr.max_entries_per_day = 2 ** 63 - 1
        tr.stop_loss_enabled = self.single3
        tr.take_profit_enabled = True
        tr.lot_size = self.terms.multiplier
        self.main_exit_at = timestamp(self.entry_config["main_exit_at"]) if self.entry_config.get("main_exit_at") else None
        # TP5 / SL30 by default; a paper-trading env may set other distances (user 2026-10-30).
        for key in ("take_profit_points", "stop_loss_points"):
            if key in self.entry_config:
                value = int(self.entry_config[key])
                if value <= 0 or not self.single3:
                    raise ValueError(f"entry.{key} is a positive Single3 distance in points")
                setattr(self, key, value)
        self._stop_rule_config()
        self._rule_gate_config(config_path, master)
        self._exit_rule_config()
        self.chunk_steps = int(self.entry_config.get("chunk_steps", 0))
        if self.chunk_steps < 0:
            raise ValueError("chunk_steps must be nonnegative")
        if market_cache is None or df is not None:
            self._load_market(df)
        else:
            market_cache.load(self)
        if self.window_size != 60 or self._features.shape[1] != 18:
            raise ValueError("FG entry market observation requires exactly (60, 18)")
        if self.main_exit_at is not None and self.main_exit_at not in self._raw.index:
            raise ValueError("Main roll exit must be an actual old-contract minute")
        self.action_space = spaces.Discrete(2)
        obs_spaces = {
            "market_seq": spaces.Box(-np.inf, np.inf, (self.window_size, self._features.shape[1]), np.float32),
            "position_state": spaces.Box(-np.inf, np.inf, (9,), np.float32),
            "action_mask": spaces.Box(0, 1, (2,), np.float32),
        }
        if self._e_profile is not None:
            obs_spaces.update({
                'market_seq_e_shared': spaces.Box(-np.inf, np.inf, (60, 18), np.float32),
                'market_context_15m': spaces.Box(-np.inf, np.inf, (self.context_15m_buckets, 7), np.float32),
                'market_macd_5m': spaces.Box(-np.inf, np.inf, (4,), np.float32),
            })
        self.observation_space = spaces.Dict(obs_spaces)
        self.completed_accounts = []
        self.account_sequence = 0
        self._initialized = False
        self.reset()

    def _load_market(self, df):
        self._lines = None  # the stop lines of the market loaded before
        self._gates = None  # and its rule positions
        self._swings = None  # and its 15m swings
        if df is None:
            live = getattr(self, '_live', None)
            df = live.read(floor=self.terms.calendar[0]) if live else pd.read_csv(Path(self.config.trading.data_path).expanduser())
        if getattr(self, '_live', None):
            self._live_frame = df
        input_index = pd.DatetimeIndex(pd.to_datetime(df["Date"])) if "Date" in df else df.index
        if not isinstance(input_index, pd.DatetimeIndex) or input_index.hasnans or input_index.has_duplicates or not input_index.is_monotonic_increasing:
            raise ValueError("Entry market timestamps must be unique, valid and ascending")
        required = ["Open", "High", "Low", "Close", "Volume", "OpenInterest"]
        if set(required)-set(df) or not np.isfinite(df[required].apply(pd.to_numeric, errors="coerce").to_numpy()).all():
            raise ValueError("Entry requires complete numeric OHLCVOI before normalization")
        if 'trading_date' in df:
            # Real-clock features below replace the legacy CSV features. Keep
            # their exact normalization without building the discarded store.
            raw = normalize_csv_frame(df, data_tz=getattr(self.config.trading, 'data_tz', None))
            raw = raw.sort_index().copy()
        else:
            source = CsvBarSource(self.config, df=df)
            raw = source.df_raw.sort_index().copy()
        if raw.empty or raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
            raise ValueError("Market must be nonempty and have unique ascending timestamps")
        cols = ["Open", "High", "Low", "Close", "Volume", "OpenInterest"]
        values = raw[cols].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4:] < 0).any():
            raise ValueError("Invalid OHLCVOI values")
        if ((raw.High < raw[["Open", "Close", "Low"]].max(axis=1)) |
                (raw.Low > raw[["Open", "Close", "High"]].min(axis=1))).any():
            raise ValueError("Inconsistent OHLC")
        if "symbol" in raw and set(raw.symbol) != {self.contract}:
            raise ValueError("A trading trajectory cannot mix contracts")
        if not raw.index.isin(self.terms.calendar).all():
            raise ValueError("Market timestamps fall outside the supplied trading calendar")
        for time in raw.index:
            self.terms.at(time - pd.Timedelta(minutes=1))
            self.terms.at(time)
        bobs = pd.DatetimeIndex([timestamp(x) for x in raw.bob]) if 'bob' in raw else raw.index - pd.Timedelta(minutes=1)
        durations = raw.index - bobs
        valid = np.asarray(durations == pd.Timedelta(minutes=1))
        if not ((durations == pd.Timedelta(minutes=1)) | (durations == pd.Timedelta(0))).all():
            raise ValueError('Unsupported minute bob/eob interval')
        feature_raw = raw.loc[valid] if self.single3 else raw
        if feature_raw.empty:
            raise ValueError('Market has no valid one-minute bars')
        if 'trading_date' in raw:
            from gym_trading_env.utils.market_features import build_market_features, FEATURES_MARKET_OBS_OUTPUT
            feature_frame=build_market_features(feature_raw,is_future=True,real_futures_clock=True,futures_schedule=self.market_clock)
            features = feature_frame.reindex(raw.index).ffill().fillna(0)
            self._features=features[FEATURES_MARKET_OBS_OUTPUT].to_numpy(dtype=np.float32)
        else:
            if self.single3 and not valid.all():
                source = CsvBarSource(self.config, df=feature_raw)
            mapping = source.store.index.get_indexer(raw.index)
            if (mapping[valid if self.single3 else np.ones(len(raw), bool)] < 0).any():
                raise ValueError("Existing market feature clock cannot represent these FG timestamps")
            if self.single3:
                features = pd.DataFrame(source.store.X_market_obs, index=source.store.index)
                self._features = features.reindex(raw.index).ffill().fillna(0).to_numpy(dtype=np.float32)
            else:
                self._features = source.store.X_market_obs[mapping].astype(np.float32)
        self._raw = raw
        self.bar_source = SimpleNamespace(store=SimpleNamespace(index=raw.index, n_rows=len(raw)))
        self.bobs, self.valid_fill_clock = bobs, valid
        self.market_clock_counter = MarketClock(valid)
        if self._e_profile is not None or self.stop_rule:
            if 'trading_date' not in raw or self.market_clock is None:
                raise ValueError('E requires the admitted real-clock TRAIN source')
            from e_data_v1.core import MarketData
            from e_data_v1.schema import EDataConfig
            from gym_trading_env.utils.time_contract import ensure_feature_tz_index
            from gym_trading_env.utils.ohlcvi_contract import normalize_ohlcvi
            # Reuse the unpadded original feature frame including its auxiliary
            # price columns. Prefix calculations for delayed data still call
            # the same original feature builder on the visible prefix only.
            self._e_market = MarketData.from_env(raw,self._e_schedule,
                feature_frame,self.market_clock_counter,bobs,
                config=EDataConfig(context_buckets=self.context_15m_buckets),
                feature_function=_e_market_features,
                market_clock_class=MarketClock,tz_normalizer=ensure_feature_tz_index,
                ohlc_normalizer=normalize_ohlcvi)
        # Existing DB contains zero-duration metadata rows: mark-only, never fills.

    @property
    def net_lots(self):
        return self.user_accounts.long_position + self.user_accounts.short_position

    def _positions(self):
        side = "LONG" if self.user_accounts.long_position else "SHORT"
        positions = self.position_manager.long_positions if side == "LONG" else self.position_manager.short_positions
        return side, positions

    def _set_terms(self, time, *, closing=False, entry=None):
        terms = self._terms_at(time)
        tr = self.config.trading
        tr.leverage = Decimal(1) / terms["margin_rate"]
        same_day = entry is not None and self._trading_day(time) == entry["trading_day"]
        tr.trading_fee_per_lot = terms["close_today_fee" if same_day else "close_fee"] if closing else terms["open_fee"]
        tr.is_round_turn = True
        # Adapter has already converted to the execution-price convention.
        tr.spread = Decimal(0)
        return terms

    def _trading_day(self, time):
        i = self._raw.index.searchsorted(timestamp(time), side="left")
        i = min(i, len(self._raw) - 1)
        column = "trading_date" if "trading_date" in self._raw else "trading_day"
        return str(self._market_row(i)[column]) if column in self._raw else str(timestamp(time).date())

    def _market_row(self, cursor):
        cache = getattr(self, '_native_execution_cache', None)
        return self._raw.iloc[cursor] if cache is None else cache.row(self._raw, cursor)

    def _terms_at(self, time):
        cache = getattr(self, '_native_execution_cache', None)
        return self.terms.at(time) if cache is None else cache.terms_at(self.terms, time)

    def _price(self, raw_price, side, time, *, opening=False):
        terms = self._terms_at(time)
        cache = getattr(self, '_native_execution_cache', None)
        if cache is not None:
            key = cache.quote_key(raw_price, side, opening, terms, self.terms.tick)
            cached = cache.quote(key)
            if cached is not None:
                return cached
        buy = (side == "LONG") == opening
        price = Decimal(str(raw_price)) + (1 if buy else -1) * (terms["spread"] + terms["slippage"])
        rounding = ROUND_CEILING if buy else ROUND_FLOOR
        result = (price / self.terms.tick).to_integral_value(rounding=rounding) * self.terms.tick
        if cache is not None:
            cache.put_quote(key, result)
        return result

    def _mark(self, price, time):
        side, _ = self._positions()
        self.current_price = self._price(price, side, time) if self.net_lots else Decimal(str(price))
        self._update_unrealized_pnl()

    def _compute_stop_loss_price(self, entry_exec_price, side):
        if self.stop_rule:
            # The ADD was decided at the close of the previous bar.
            self._plan = self.planned_exit(self.current_step - 1, entry_exec_price, side)
            return self._plan['stop_loss_price']
        d = self.stop_loss_points
        return entry_exec_price + (-d if side == "long" else d) if self.single3 else None

    def _compute_take_profit_price(self, entry_exec_price, sl_exec_price, side):
        if self.stop_rule:
            return self._plan['take_profit_price']
        d = self.take_profit_points
        return entry_exec_price + (d if side == "long" else -d)

    def _stop_rule_config(self):
        """entry.stop_rule (V0.5): `prior_15m_high` sets SL = H* + stop_above
        and TP = fill - target_r * R, R = SL - fill, with H* the lowest line
        (causal 15m pivot high, ts_v2_positive.pivot_levels) live at the
        decision, already departed (ts_v2_positive.departed_at) and at or
        above the fill. A stateless rule: no touch or event state is kept,
        so a line confirmed during a material touch may differ (the teacher
        drops those as STOP_LINE_DIFFERS). R is capped at max_risk, and with
        no such line R = max_risk; `stop_source` names which. No ADD mask or
        gate depends on it, unless entry.rule_gate sets one
        (_rule_gate_config: then the line is the touch's own). `prior_15m_high` is SHORT only; `prior_15m_extreme`
        (V2.1) is the same for a SHORT and its mirror for a LONG: L* the
        highest departed prior 15m pivot low at or below the fill, SL =
        L* - stop_above, TP = fill + target_r * R (ts_v2_positive short space).
        entry.context_15m_buckets sets the E context."""
        keys = ('stop_rule', 'stop_above_points', 'max_risk_points', 'target_r', 'prior_high')
        given = {k: self.entry_config[k] for k in keys if k in self.entry_config}
        if 'context_15m_buckets' in self.entry_config:
            value = self.entry_config['context_15m_buckets']
            if type(value) is not int or value < 1 or not self.single3:
                raise ValueError('entry.context_15m_buckets is a positive Single3 bucket count')
            self.context_15m_buckets = value
        if not given:
            return
        if given.keys() != set(keys) or given['stop_rule'] not in STOP_RULES or not self.single3:
            raise ValueError(f'entry.stop_rule {STOP_RULES} (Single3) needs exactly {keys}')
        p = given['prior_high']
        ints = (given['stop_above_points'], given['max_risk_points'], *(p.get(k) for k in PRIOR_HIGH_KEYS))
        if (set(p) != set(PRIOR_HIGH_KEYS) or any(type(x) is not int or x < 1 for x in ints)
                or given['max_risk_points'] < given['stop_above_points']
                or not (isinstance(given['target_r'], (int, float)) and given['target_r'] > 0)):
            raise ValueError(f'entry.stop_rule: positive integer points, max_risk_points >= '
                             f'stop_above_points, prior_high {PRIOR_HIGH_KEYS}, target_r > 0')
        self.stop_rule = dict(given)

    def _rule_gate_config(self, config_path, master):
        """entry.rule_gate (V2.3, user 2026-11-05: "规则决定学生能在哪里交易；
        大脑决定在这些位置中什么时候交易"): {SHORT, LONG} rules files of the V4
        material (ts_v2_positive.load_config). An ADD is legal only at their
        rule positions (_gate: rule_cursors, the material's own pass), and its
        stop is the touch's own line (planned_exit). The rules must share the
        stop rule's lines, stop and target, keep the planned risk of a rule
        position within max_risk_points, and carry this env's costs. Rules V5
        (user 2026-11-05, "止损固定5点"): entry.fixed_stop_points sets R from the
        fill (FIXED), the same on both sides and within max_risk_points.
        entry.cap_rule_risk (paper only, user 2026-11-08: "止损上限5点"; default false): the
        strategy's max_risk_points may be below the rules' zone risk; the rule positions stay
        and a rule stop beyond it is CAPPED at max_risk_points from the fill (planned_exit)."""
        paths = self.entry_config.get('rule_gate')
        if paths is None:
            return
        from gym_trading_env.research.ts_v2_positive import load_config
        r = self.stop_rule
        if (not r or r['stop_rule'] != 'prior_15m_extreme' or not isinstance(paths, dict)
                or set(paths) != {'SHORT', 'LONG'} or master is None):
            raise ValueError('entry.rule_gate {SHORT, LONG} needs stop_rule prior_15m_extreme and a strategy_config')
        # The rules' costs must agree with the env's only where they place the rule positions: the
        # spread of the decision quote. Point value, fees and margin are the strategy's alone, so one
        # rules file serves every product (user 2026-11-06: "应该只需要改1个地方才对，这样配置冗余了").
        values = ('spread',)
        capped = self.entry_config.get('cap_rule_risk', False)
        if type(capped) is not bool:
            raise ValueError('entry.cap_rule_risk is true or false')
        # entry.no_reentry_after_stop (paper only, user 2026-11-08: "止损出场后，同一次触线不再开仓"; default
        # false): once an entry of a touch exits at its stop, the touch's later rule positions open no ADD.
        self.no_reentry_after_stop = self.entry_config.get('no_reentry_after_stop', False)
        if type(self.no_reentry_after_stop) is not bool:
            raise ValueError('entry.no_reentry_after_stop is true or false')
        gate = {}
        for side, path in paths.items():
            cfg = load_config(resolve_path(config_path, path))
            e, costs = cfg['entry'], cfg.get('costs')
            if (cfg['direction'] != side or e['stop_above_points'] != r['stop_above_points']
                    or e['target_r'] != r['target_r'] or cfg['prior_high'] != r['prior_high']
                    or (not capped and e['stop_above_points'] + e['zone_points'] > r['max_risk_points'])):
                raise ValueError(f'entry.rule_gate {side}: rules of another side, line, stop, target or risk')
            fixed = e.get('fixed_stop_points')
            if fixed is not None and (capped or fixed > r['max_risk_points']):
                raise ValueError(f'entry.rule_gate {side}: fixed_stop_points above max_risk_points or with cap_rule_risk')
            if costs is None or any(float(costs[k]) != float(master['costs'][k]) for k in values):
                raise ValueError(f'entry.rule_gate {side}: its costs are not this env\'s (spread)')
            gate[side] = cfg
        if len({c['entry'].get('fixed_stop_points') for c in gate.values()}) != 1:
            raise ValueError('entry.rule_gate: SHORT and LONG have different stops')
        self.rule_gate = gate

    def _exit_rule_config(self):
        """entry.exit_rule (V3, user 2026-11-06: "这个是新的策略，做长线，而不是短线";
        "入场方式没变"): research.swing_trail. Only with the rule gate, whose stop
        (FIXED or the touch's H + 5) is the entry's initial stop; the plan keeps
        its TP as the observation's reference only, and no TP exit is taken."""
        cfg = self.entry_config.get('exit_rule')
        if cfg is None:
            return
        from gym_trading_env.research.swing_trail import check_config
        if self.rule_gate is None:
            raise ValueError('entry.exit_rule needs entry.rule_gate')
        self.exit_rule = check_config(cfg)

    def _swing_chain(self):
        """The 15m swings of the market loaded (research.swing_trail.Swings), once per market."""
        if self._swings is None:
            from gym_trading_env.research.swing_trail import Swings
            r = self.exit_rule
            self._swings = Swings(self._e_market.complete_buckets, r['pivot_left'], r['pivot_right'])
        return self._swings

    def _trail(self, item, pos, row, bob, time):
        """After the close of bar `time`: the stop of an open entry under
        entry.exit_rule, effective from the next bar. best is the most
        favourable price since the fill (the entry bar: its close only, as the
        TP rule's same-bar quote)."""
        from gym_trading_env.research.swing_trail import trail_stop
        s = -1 if item['side'] == 'LONG' else 1  # short space
        same_bar = timestamp(item['entry_time']) == bob
        quote = Decimal(str(row.Close if same_bar else (row.Low if s > 0 else row.High)))
        best = min(Decimal(item['best_price']), s * quote) if 'best_price' in item else s * quote
        item['best_price'] = str(best)
        fill_ns = timestamp(item['entry_time']).value
        kind = 'L' if s > 0 else 'H'
        swings = [s * Decimal(str(x)) for x in self._swing_chain().after(kind, fill_ns, timestamp(time).value)]
        stop, stage = trail_stop(s * Decimal(item['entry_price']), s * pos.stop_loss_price,
                                 item.get('stop_stage', 'INITIAL'), best, swings, self.exit_rule,
                                 initial=s * Decimal(item['planned_exit']['stop_loss_price']))
        if s * stop != pos.stop_loss_price:
            pos.stop_loss_price = s * stop
            item.setdefault('stop_history', []).append(dict(time=str(time), stop=str(s * stop), stage=stage,
                                                             swings=len(swings)))
        item['stop_stage'], item['swings_after_fill'] = stage, len(swings)

    def _stopped_touches(self):
        """(side, rule_event) of the touches whose entry exited at its stop
        (entry.no_reentry_after_stop), from the account's own fills and
        attributions, so a restored state needs nothing more."""
        key = (id(self.attributions), len(self.attributions))
        if getattr(self, '_stopped_key', None) != key:
            plans = {f['entry_id']: f for f in self.fills if f['reason'] == 'ENTRY'}
            self._stopped = {(a['side'], plans[a['entry_id']]['rule_event']) for a in self.attributions
                             if a['reason'] == 'STOP_LOSS'}
            self._stopped_key = key
        return self._stopped

    def _gate(self, side):
        """The rule positions of `side` in the market loaded
        (ts_v2_positive.rule_cursors), computed once per market."""
        if self._gates is None:
            self._gates = {}
        if side not in self._gates:
            from gym_trading_env.research.ts_v2_positive import rule_cursors
            if self._e_market is None:
                raise ValueError('entry.rule_gate needs the complete 15m buckets of the market')
            self._gates[side] = rule_cursors(self, self.rule_gate[side])
        return self._gates[side]

    def _stop_lines(self, side):
        """H, confirm, broken and expire cursors of every line of `side` in
        short space (a LONG's H is minus its prior low), computed once."""
        if self._lines is None:
            self._lines = {}
        if side not in self._lines:
            from gym_trading_env.research.ts_v2_positive import departed_at, pivot_levels, side_buckets, sign
            if self._e_market is None:
                raise ValueError('entry.stop_rule needs the complete 15m buckets of the market')
            raw, r, p, s = self._raw, self.stop_rule, self.stop_rule['prior_high'], sign(side)
            # The quote the position's SL checks: a SHORT buys at the high, a LONG sells at the low.
            stop_quote = np.array([float(self._price(v, side, t)) for v, t in
                                   zip(raw.High if s > 0 else raw.Low, raw.index)])
            a = dict(n=len(raw), times=raw.index, times_ns=raw.index.asi8,
                     bob_ns=pd.DatetimeIndex(self.bobs).asi8, valid=np.asarray(self.valid_fill_clock, bool),
                     close=s * raw.Close.to_numpy(float), buy_high=s * stop_quote)
            levels = pivot_levels(side_buckets(self._e_market.complete_buckets, s), a, p['pivot_left'],
                                  p['pivot_right'], p['lookback_buckets'], r['stop_above_points'])
            for x in levels:
                x['departed'] = departed_at(a, x, p['departure_points'])
            self._lines[side] = {k: np.array([x[k] for x in levels], float if k == 'H' else np.int64)
                                 for k in ('H', 'confirm', 'departed', 'broken', 'expire')}
        return self._lines[side]

    def planned_exit(self, cursor, fill, side):
        """SL / TP prices, R and stop_source of a SHORT (or, under
        prior_15m_extreme, a LONG) decided at `cursor` and filled at `fill`
        under entry.stop_rule."""
        side = str(side).upper()
        if side != 'SHORT' and (side != 'LONG' or self.stop_rule['stop_rule'] == 'prior_15m_high'):
            raise ValueError('entry.stop_rule prior_15m_high is SHORT only; prior_15m_extreme is SHORT or LONG')
        s = 1 if side == 'SHORT' else -1
        r, lines, fill = self.stop_rule, self._stop_lines(side), Decimal(str(fill))
        # In short space (a LONG's prices negated): the lines known at the
        # decision, not broken or expired, that the price has already left (a
        # close at H - departure_points): the lowest one at or above the fill
        # is the stop's line.
        f = s * fill
        cap = Decimal(r['max_risk_points'])
        rule = {}
        if self.rule_gate is not None:
            # V2.3: the line of the rule position's touch; a fill gap beyond
            # max_risk caps R (CAPPED), a fill at or through the stop takes
            # R = max_risk (GAP).
            g = self._gate(side)
            if not g['open'][cursor]:
                raise ValueError(f'{self.contract} cursor {cursor}: not a rule position, no gated ADD')
            i = int(g['event'][cursor])
            line = Decimal(str(g['H'][i]))
            fixed = self.rule_gate[side]['entry'].get('fixed_stop_points')
            if fixed is not None:  # V5: R from the fill, whatever the line
                risk, source = Decimal(fixed), 'FIXED'
            else:
                risk = line + r['stop_above_points'] - f
                source = 'RULE' if 0 < risk <= cap else 'CAPPED' if risk > cap else 'GAP'
                risk = risk if source == 'RULE' else cap
            rule = dict(rule_event=g['event_id'][i], touch=int(g['touch'][i]))
            tick = self.terms.tick
            target = (Decimal(str(r['target_r'])) * risk / tick).to_integral_value(rounding=ROUND_CEILING) * tick
            plan = dict(line=str(s * line), risk_points=str(risk), stop_source=source,
                        stop_loss_price=fill + s * risk, take_profit_price=fill - s * target, **rule)
            if self.exit_rule is not None:
                # V3: no TP exit; the rule's TP stays the observation's reference.
                plan.update(take_profit_price=None, reference_tp_price=plan['take_profit_price'],
                            exit_rule=self.exit_rule['kind'])
            return plan
        live = ((lines['confirm'] <= cursor) & (lines['departed'] <= cursor) & (cursor < lines['broken'])
                & (cursor < lines['expire']) & (lines['H'] >= float(f)))
        if live.any():
            line = Decimal(str(lines['H'][live].min()))
            risk = line + r['stop_above_points'] - f
            source = 'STRUCTURAL' if risk <= cap else 'CAPPED'
            risk = min(risk, cap)
        else:
            line, risk, source = None, cap, 'NO_LINE'
        tick = self.terms.tick
        target = (Decimal(str(r['target_r'])) * risk / tick).to_integral_value(rounding=ROUND_CEILING) * tick
        return dict(line=None if line is None else str(s * line), risk_points=str(risk), stop_source=source,
                    stop_loss_price=fill + s * risk, take_profit_price=fill - s * target)

    def _assert_ledger_conservation(self):
        if self.ledger.total_balance() != self._ledger_total0:
            raise RuntimeError("Entry Ledger conservation failed")

    def _preflight_open(self, side, price):
        if self.net_lots + 3 > self.max_lots or len(self.entries) >= self.max_entries:
            return ForexCode.ERROR_HIT_MAX_POSITION
        held, _ = self._positions()
        if self.net_lots and held != side.name:
            return ForexCode.ERROR_OPEN_POSITION
        time = self._raw.index[self.current_step]
        required = 3 * self.terms.multiplier * price / self.config.trading.leverage
        available = self._calculate_equity() - max(self.user_accounts.margin, self._required_margin(price, time))
        if required + 3*self.config.trading.trading_fee_per_lot > available:
            return ForexCode.ERROR_NO_ENOUGH_MONEY
        return super()._preflight_open(side, price)

    def _required_margin(self, price, time):
        return self.net_lots * self.terms.multiplier * Decimal(str(price)) * self._terms_at(time)["margin_rate"]

    def direction_at(self, time, row_index=None):
        if self.directions.date_basis == "trading_date":
            row = self._market_row(self.current_step if row_index is None else row_index)
            if "trading_date" not in row:
                raise ValueError("Manual direction dates require DB trading_date")
            day = str(row.trading_date)
            if len(day) == 8 and day.isdigit():
                day = pd.to_datetime(day, format="%Y%m%d")
            return self.directions.at(self.contract, time, day)
        return self.directions.at(self.contract, time)

    def _can_add(self, *, continuing=False):
        time = self._raw.index[self.current_step]
        direction = self.direction_at(time)
        status = 'ACTIVE' if continuing and self.status=='CHUNK_END' else self.status
        if status != "ACTIVE" or direction == "NO_TRADE" or time >= self.terms.last_open or (self.main_exit_at is not None and time >= self.main_exit_at):
            return False
        if self.rule_gate is not None:
            g = self._gate(direction)
            if not g['open'][self.current_step]:
                return False
            if (self.no_reentry_after_stop and (direction, g['event_id'][g['event'][self.current_step]])
                    in self._stopped_touches()):
                return False
        self._set_terms(time)
        price = self._price(self._market_row(self.current_step).Close, direction, time, opening=True)
        return self._preflight_open(TargetPos[direction], price) == ForexCode.SUCCESS

    def _get_obs(self):
        i = self.current_step
        time = self._raw.index[i]
        direction = self.direction_at(time)
        start = max(0, i - self.window_size + 1)
        market = (self._features[self.market_clock_counter.window(i, self.window_size)]
                  if self.single3 else self._features[start:i + 1])
        market = np.pad(market, ((self.window_size-len(market), 0), (0, 0)))
        position = np.zeros(9, np.float32)
        position[0] = {"SHORT": -1, "NO_TRADE": 0, "LONG": 1}[direction]
        # Opens commit atomically or are rejected; there is no resting opening order.
        position[2] = float(any(e.get("sl_pending", False) for e in self.entries.values())) if self.single3 else 0.0
        held_quantity = Decimal(0)
        for item in self.entries.values():
            positions = self.position_manager.long_positions if item["side"] == "LONG" else self.position_manager.short_positions
            pos = positions[item["slot"]]
            if pos is None:
                raise RuntimeError("Entry identity has no actual PositionManager position")
            held_quantity += pos.size
            age = (self.market_clock_counter.held(item['entry_cursor'], i) if self.single3 else
                   (time - timestamp(item['entry_time'])).total_seconds() / 60)
            if age < 0:
                raise RuntimeError("Position observation precedes the actual entry fill")
            sign = 1 if item["side"] == "LONG" else -1
            tp = (pos.take_profit_price if pos.take_profit_price is not None
                  else Decimal(item['planned_exit']['reference_tp_price']))
            distance = sign * (tp - self.current_price)
            offset = 3 + 3 * item["obs_slot"]
            position[offset:offset+3] = [float(pos.size / 3), float(distance / POSITION_TP_SCALE),
                                       np.log1p(age) / np.log1p(43200)]
        position[1] = float(held_quantity / self.max_lots)
        if held_quantity != self.net_lots:
            raise RuntimeError("PositionManager/account quantities lack a matching entry identity")
        mask = np.array([1, self._can_add()], np.float32)
        self._published_mask = mask.copy()
        obs = {
            # np.pad above owns this window; normalize/cast in place when
            # possible instead of allocating two more identical float32 windows.
            "market_seq": np.nan_to_num(market, copy=False).astype(np.float32, copy=False),
            "position_state": position, "action_mask": mask,
        }
        if self._e_profile is not None:
            from e_data_v1.schema import Decision
            # A delayed raw candle cannot be used by the original Env either;
            # MarketData's exact-old-observation check fails closed here.
            cutoff = self._raw.index[self.current_step]
            context, old, shared, _, _, meta = self._e_market.snapshot(
                Decision(self.contract, visible_cutoff=cutoff, decision_time=cutoff,
                         market_seq=obs['market_seq']))
            if not np.array_equal(old, obs['market_seq']):
                raise ValueError('E observation changed the original 18 channels')
            macd, _ = self._e_market.macd_snapshot(cutoff)
            arrays = dict(market_seq_e_shared=shared.copy(), market_context_15m=context.copy(),
                          market_macd_5m=macd)
            if (arrays['market_seq_e_shared'].shape != (60, 18)
                    or arrays['market_context_15m'].shape != (self.context_15m_buckets, 7)
                    or any(x.dtype != np.float32 or not np.isfinite(x).all()
                           for x in arrays.values())
                    or not np.isin(context[:, 6], (0., 1.)).all()):
                raise ValueError('Native E Env Observation differs from signed schema')
            if (any(pd.Timestamp(t) > cutoff for t in meta['context_available_at'])
                    or pd.Timestamp(meta['anchor_available_at']) > cutoff
                    or pd.Timestamp(meta['anchor_source_time']) > cutoff):
                raise ValueError('Native E Env cannot publish a future-visible feature')
            obs.update(arrays)
        return obs

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if options and "resume" in options:
            self.import_state(options["resume"])
            return self._get_obs(), self._get_info()
        if self._initialized and not self.terminated:
            if self.status == "CHUNK_END":
                self.status, self.truncated, self.chunk_count = "ACTIVE", False, 0
            elif not self.entries and not self.fills and self.current_step == (
                    len(self._raw) - 1 if getattr(self, '_live', None) else 0):
                pass  # a fresh account (live: at its latest bar)
            else:
                raise RuntimeError("Reset would discard an unfinished account; continue or restore it")
            return self._get_obs(), self._get_info()
        if self._initialized:
            completed = self.export_state()
            completed["completed_accounts"] = []
            self.completed_accounts.append(completed)
        self.account_sequence += 1
        self._initialize_accounts(allow_negative_cash=True)
        self.last_close_position = None
        self._entries_used_today = 0
        self.current_step = 0
        self.episode_step_count = self.chunk_count = self.entry_sequence = 0
        self.current_price = Decimal(str(self._raw.iloc[0].Close))
        self.entries = {}
        self.fills, self.attributions, self.reward_events, self.equity_history = [], [], [], []
        self.processed_fills = set()
        self.reward_function = self._new_reward()
        if self.single3:
            self.last_reward_components = None
        self.total_reward = self.max_floating_loss = 0.0
        self.legal_add_decisions = self.voluntary_waits = 0
        self.status, self.terminated, self.truncated = "ACTIVE", False, False
        self.pending_reason = None
        if self.single3:
            self._tp_info_cache = (0, 0, 0.)
        if getattr(self, '_live', None):
            # A live account starts empty at the latest bar; earlier bars
            # only fill the observation window.
            self.current_step = len(self._raw) - 1
            self.current_price = Decimal(str(self._raw.iloc[-1].Close))
        self._initialized = True
        self._sample_equity()
        self._ensure_private_history()
        return self._get_obs(), self._get_info()

    def _open_entry(self, side, price, time):
        self._set_terms(time)
        execution = self._price(price, side, time, opening=True)
        code = self._preflight_open(TargetPos[side], execution)
        if code != ForexCode.SUCCESS:
            return code
        positions = self.position_manager.long_positions if side == "LONG" else self.position_manager.short_positions
        slot = next(i for i, p in enumerate(positions) if p is None)
        fn = self._long_open if side == "LONG" else self._short_open
        code = fn(execution, Decimal(0), slot=slot, fill_time=time)
        if code != ForexCode.SUCCESS:
            return code
        self.entry_sequence += 1
        entry_id = f"{self.contract}:{self.account_sequence}:{self.entry_sequence}"
        used_obs_slots = {entry["obs_slot"] for entry in self.entries.values()}
        obs_slot = next(i for i in range(self.max_entries) if i not in used_obs_slots)
        item = dict(entry_id=entry_id, side=side, slot=slot, entry_time=str(time),
                    obs_slot=obs_slot,
                    entry_price=str(execution), closed_quantity="0", exit_notional="0",
                    exit_pending=False, trading_day=self._trading_day(time), all_tp=True)
        self.entries[entry_id] = item
        if self.single3:
            item.update(sl_pending=False, entry_cursor=self.current_step,
                        open_fee=str(self.trade_record_manager.trade_history[-1].fee))
        self.trade_record_manager.trade_history[-1].meta.update(entry_id=entry_id, fill_model=self.fill_model)
        self.fills.append(dict(entry_id=entry_id, time=str(time), side=side, quantity=3,
                               price=str(execution), reason="ENTRY", committed=True))
        if self.stop_rule:
            plan = {k: str(v) if isinstance(v, Decimal) else v for k, v in self._plan.items()}
            self.fills[-1].update(plan)
            item['planned_exit'] = plan
        return code

    def settle_exit(self, entry_id, price, time, reason, *, quantity=None, fill_id=None):
        """Commit an explicit exit fill through the existing atomic close path.

        Partial fills retain the original entry reservation until the whole entry exits.
        Duplicate external fill IDs are rejected before bookkeeping.
        """
        if self.single3 and fill_id is None:
            fill_id = f"{entry_id}|{timestamp(time)}|{reason}|3"
        if fill_id is not None and fill_id in self.processed_fills:
            raise ValueError("Duplicate exit fill")
        item = self.entries[entry_id]
        if self.single3 and quantity is not None and Decimal(str(quantity)) != 3:
            raise ValueError("Single3 requires an atomic full three-lot exit")
        time = timestamp(time)
        if self.single3 and time != self._raw.index[self.current_step]:
            raise ValueError('Exit must belong to the currently observed market bar')
        if time < timestamp(item["entry_time"]):
            raise ValueError("Exit precedes entry")
        side = item["side"]
        price = Decimal(str(price))
        # The TP distance of this entry (its planned TP: below a SHORT's fill, above a LONG's).
        if reason == "TAKE_PROFIT" and item.get('planned_exit', {}).get('take_profit_price', '') is None:
            raise ValueError('entry.exit_rule takes no TP exit')
        distance = (abs(Decimal(item['entry_price']) - Decimal(item['planned_exit']['take_profit_price']))
                    if reason == "TAKE_PROFIT" and 'planned_exit' in item else self.take_profit_points)
        if reason == "TAKE_PROFIT" and (1 if side == "LONG" else -1) * (price - Decimal(item["entry_price"])) < distance:
            raise ValueError("TP execution price does not reach the TP distance")
        self._set_terms(time, closing=True, entry=item)
        fn = self._long_close if side == "LONG" else self._short_close
        code = fn(price, Decimal(0), slot=item["slot"], close_reason=reason,
                  fill_time=time, quantity=quantity)
        if code != ForexCode.SUCCESS:
            item["exit_pending"] = True
            return code, None
        record = self.trade_record_manager.trade_history[-1]
        record.meta.update(entry_id=entry_id, fill_model=self.fill_model)
        closed = record.closed_size
        item["closed_quantity"] = str(Decimal(item["closed_quantity"]) + closed)
        item["exit_notional"] = str(Decimal(item["exit_notional"]) + closed * price)
        item["all_tp"] = item["all_tp"] and reason == "TAKE_PROFIT"
        if reason != "TAKE_PROFIT":
            item.setdefault("non_tp_reason", reason)
        if fill_id is not None:
            self.processed_fills.add(fill_id)
        self.fills.append(dict(entry_id=entry_id, time=str(time), side=side, quantity=str(closed),
                               price=str(price), reason=reason, committed=True, fill_id=fill_id))
        if Decimal(item["closed_quantity"]) != 3:
            return code, None
        event = dict(entry_id=entry_id, side=side, entry_time=item["entry_time"], exit_time=str(time),
                     entry_price=item["entry_price"], exit_price=str(Decimal(item["exit_notional"])/3),
                     closed_quantity=3, reason="TAKE_PROFIT" if item["all_tp"] else item["non_tp_reason"], committed=True)
        if self.single3:
            event.update(gross_trade_pnl=str(record.pnl), open_fee=item['open_fee'],
                         close_fee=str(record.fee),
                         net_trade_pnl=str(record.pnl - record.fee - Decimal(item['open_fee'])),
                         **item.get('exit_metadata', {}))
            if self.exit_rule is not None:
                event.update(stop_stage=item.get('stop_stage', 'INITIAL'), final_stop_price=str(
                    item.get('stop_history', [{}])[-1].get('stop', item['planned_exit']['stop_loss_price'])),
                    swings_after_fill=item.get('swings_after_fill', 0), stop_history=item.get('stop_history', []))
            event.update(entry_cursor=item['entry_cursor'], exit_cursor=self.current_step,
                         held_market_bars=self.market_clock_counter.held(item['entry_cursor'], self.current_step),
                         clock_protocol=TIME_PROTOCOL)
            if event['reason'] == 'TAKE_PROFIT':
                event['tp_market_minutes'] = event['held_market_bars']
        self.attributions.append(event)
        del self.entries[entry_id]
        return code, event

    def _new_reward(self):
        if not self.single3:
            return FastTakeProfitReward()
        from gym_trading_env.rewards.single3_reward import Single3Reward
        return Single3Reward(self.reward_config, self.take_profit_points, self.stop_loss_points)

    def _exit_request(self, item, pos, row, bob, time):
        """Shared executable quote rules; financial mutation stays in settle_exit.

        Called even without liquidity so a touched SL can remain pending.
        """
        from .entry_execution import single3_exit_decision
        same_bar = timestamp(item['entry_time']) == bob
        sign = 1 if item['side'] == 'LONG' else -1
        favorable = row.Close if same_bar else (row.High if sign == 1 else row.Low)
        decision = single3_exit_decision(side=item['side'], entry_time=timestamp(item['entry_time']),
            stop_price=pos.stop_loss_price, tp_price=pos.take_profit_price,
            opening_quote=self._price(row.Open, item['side'], bob),
            adverse_quote=self._price(row.Low if sign==1 else row.High, item['side'], time),
            favorable_quote=self._price(favorable, item['side'], time),
            close_quote=self._price(row.Close, item['side'], time), bob=bob, time=time,
            tick=self.terms.tick, valid_bar=bool(self.valid_fill_clock[self.current_step]),
            was_pending=item.get('sl_pending',False), trigger_time=item.get('sl_trigger_time'),
            global_reason=self.pending_reason)
        item['sl_pending'] = decision.sl_pending
        if decision.sl_trigger_time:
            item['sl_trigger_time'] = decision.sl_trigger_time
        # Metadata belongs to this attempt, never a stale pending attempt.
        item['exit_metadata'] = decision.metadata
        return (decision.reason, decision.price) if decision.reason else None

    def step(self, action):
        live = getattr(self, '_live', None)
        if live is None:
            return self._native_step(action)
        if self.current_step + 1 == len(self._raw) and not (self.terminated or self.truncated):
            rows = live.wait(self._raw.index[-1])
            self.status, self.truncated = "DATA_END", True
            self.continue_data(pd.concat([self._live_frame, rows], ignore_index=True))
        decision_time, fills = self._raw.index[self.current_step], len(self.fills)
        cursor, legal = self.current_step, bool(self._published_mask[1])
        result = self._native_step(action)
        live.send(self, decision_time, self.fills[fills:])
        live.notify_fills(self, decision_time, self.fills[fills:])
        live.bar_line(self, cursor, legal, int(action), self.fills[fills:])
        return result

    def _native_step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid Entry action: {action}")
        if self.terminated or self.truncated:
            raise RuntimeError("Episode paused/ended; continuation or explicit reset required")
        allowed = bool(self._published_mask[1])
        if allowed:
            self.legal_add_decisions += 1
            self.voluntary_waits += int(action == 0)
        old = self.current_step
        fills_before = len(self.fills)
        if old + 1 == len(self._raw):
            self.status, self.truncated = "DATA_END", True
            return self._get_obs(), 0.0, False, True, self._get_info()
        next_time = self._raw.index[old + 1]
        cal_i = self.terms.calendar.searchsorted(self._raw.index[old]) + 1
        if cal_i >= len(self.terms.calendar) or self.terms.calendar[cal_i] != next_time:
            self.status, self.truncated = "DATA_GAP", True
            return self._get_obs(), 0.0, False, True, self._get_info()
        side = self.direction_at(self._raw.index[old], old)
        self.current_step += 1
        self.episode_step_count += 1
        self.chunk_count += 1
        row = self._market_row(self.current_step)
        bob = self.bobs[self.current_step]
        self._mark(row.Open, bob)
        # Recheck actual execution-time direction, deadlines, price and funding.
        self.reject_reason = None
        opened_quantity = Decimal(0)
        if action == 1:
            if not allowed:
                self.reject_reason = "MASKED_ADD"
            elif (self.direction_at(bob) != side or bob >= self.terms.last_open):
                self.reject_reason = "DIRECTION_OR_DEADLINE_CHANGED"
            elif not self.valid_fill_clock[self.current_step]:
                self.reject_reason = "UNVERIFIED_BAR_TIME"
            elif row.Volume < 3:
                self.reject_reason = "INSUFFICIENT_BAR_VOLUME"
            elif self._calculate_equity() < self._required_margin(row.Open, bob):
                self.reject_reason = "MARGIN_CALL"
            else:
                result = self._open_entry(side, row.Open, bob)
                if result != ForexCode.SUCCESS:
                    self.reject_reason = result.name
                else:
                    opened_quantity = Decimal(3)
        time = next_time
        held, _ = self._positions()
        adverse = row.Low if held == "LONG" else row.High
        self._mark(adverse, time)
        if self.net_lots and self._calculate_equity() < self._required_margin(adverse, time):
            self.pending_reason = "MARGIN_CALL"
        if self.main_exit_at is not None and time >= self.main_exit_at and not self.pending_reason:
            self.pending_reason = "MAIN_ROLL"
        if time >= self.terms.liquidate and not self.pending_reason:
            self.pending_reason = "CONTRACT_EXPIRY"
        # With OHLC, liquidity and intrabar order cannot be proven. Risk wins.
        self._sample_equity(sample_time=time, intrabar=True)
        events = []
        remaining_volume = Decimal(str(row.Volume)) - opened_quantity if self.valid_fill_clock[self.current_step] else Decimal(0)
        for entry_id, item in list(self.entries.items()):
            positions = self.position_manager.long_positions if item["side"] == "LONG" else self.position_manager.short_positions
            pos = positions[item["slot"]]
            request = self._exit_request(item, pos, row, bob, time) if self.single3 else None
            if remaining_volume < pos.size:
                continue
            reason = self.pending_reason
            if self.single3:
                if request is None:
                    continue
                reason, price = request
            elif reason:
                raw_exit = row.Close if reason == "MAIN_ROLL" else (row.Low if item["side"] == "LONG" else row.High)
                price = self._price(raw_exit, item["side"], time)
            else:
                # New TP can only use the known post-entry close, never that bar's extreme.
                same_bar = timestamp(item["entry_time"]) == bob
                raw_exit = row.Close if same_bar else (row.High if item["side"] == "LONG" else row.Low)
                executable = self._price(raw_exit, item["side"], time)
                sign = 1 if item["side"] == "LONG" else -1
                if sign * (executable - pos.take_profit_price) < self.terms.tick:
                    continue
                reason, price = "TAKE_PROFIT", pos.take_profit_price
            size = pos.size
            code, event = self.settle_exit(entry_id, price, time, reason)
            if code == ForexCode.SUCCESS:
                remaining_volume -= size
                if event is not None:
                    events.append(event)
        if self.exit_rule is not None and not self.pending_reason:
            # The stops for the next bar, from this bar's close (a pending SL exits at the next open anyway).
            for item in self.entries.values():
                if not item.get('sl_pending'):
                    positions = (self.position_manager.long_positions if item['side'] == 'LONG'
                                 else self.position_manager.short_positions)
                    self._trail(item, positions[item['slot']], row, bob, time)
        if self.single3:
            bar = dict(bar_open=str(bob), bar_end=str(time),
                **{k.lower(): float(row[k]) for k in ('Open', 'High', 'Low', 'Close')},
                valid_bar=bool(self.valid_fill_clock[self.current_step]), eligible_market_minute=False,
                entries=[copy.deepcopy(f) for f in self.fills[fills_before:] if f['reason']=='ENTRY'],
                exits=events, true_terminal=bool(self.pending_reason and not self.entries))
            # Only the new, explicitly versioned TP profile reads official trading days.
            # The old C1 bar dict and signed reward identity are unchanged.
            from gym_trading_env.rewards.single3_reward import QUALITY_TP_SCHEMA
            if self.reward_config.schema == QUALITY_TP_SCHEMA:
                # No wall-calendar fallback for the quality TP day-boundary rule.
                if 'trading_date' not in self._raw and 'trading_day' not in self._raw:
                    raise ValueError('Quality TP requires exchange official trading_date')
                bar['trading_day'] = self._trading_day(time)
            self.last_reward_components = self.reward_function.step(bar, self.terms.multiplier)
            reward = self.last_reward_components['total']
            self.reward_events.extend(copy.deepcopy(self.last_reward_components['events']))
        else:
            reward = self.reward_function(events)
            self.reward_events.extend(copy.deepcopy(e) for e in events if e.get("reward") is not None)
        self.total_reward += reward
        self._mark(row.Close, time)
        self._sample_equity()
        if self.pending_reason:
            self.status = self.pending_reason
            self.terminated = not self.entries
        elif self.chunk_steps and self.chunk_count >= self.chunk_steps:
            self.status, self.truncated = "CHUNK_END", True
        return self._get_obs(), reward, self.terminated, self.truncated, self._get_info()

    def _sample_equity(self, sample_time=None, intrabar=False):
        unrealized = float(self.user_accounts.unrealized_pnl)
        self.max_floating_loss = max(self.max_floating_loss, -unrealized)
        self.equity_history.append(dict(time=str(sample_time or self._raw.index[self.current_step]),
                                       equity=float(self._calculate_equity()), unrealized_pnl=unrealized,
                                       net_lots=float(self.net_lots), intrabar=intrabar))

    def _get_info(self):
        if self.single3:
            consumed,count,total=getattr(self,'_tp_info_cache',(0,0,0.))
            if consumed>len(self.attributions):
                consumed,count,total=0,0,0.
            for event in self.attributions[consumed:]:
                if event['reason']=='TAKE_PROFIT':
                    count+=1
                    total += event['tp_market_minutes']
            self._tp_info_cache=(len(self.attributions),count,total)
            mean=total/count if count else None
        else:
            completed = [e for e in self.attributions if e["reason"] == "TAKE_PROFIT"]
            waits = [(timestamp(e["exit_time"])-timestamp(e["entry_time"])).total_seconds()/60 for e in completed]
            count=len(completed)
            mean=float(np.mean(waits)) if waits else None
        return dict(status=self.status, reject_reason=getattr(self, "reject_reason", None),
                    **({'clock_protocol': TIME_PROTOCOL, 'mean_tp_market_minutes': mean} if self.single3 else {}),
                    fill_model=self.fill_model, direction_hash=self.directions.hash,
                    tp_count=count, **({} if self.single3 else {'mean_tp_wait_minutes': mean}),
                    cumulative_speed_reward=self.total_reward, unfinished_entries=len(self.entries),
                    unfinished_lots=float(self.net_lots), max_floating_loss=self.max_floating_loss,
                    account_net_pnl=float(self._calculate_equity())-300000,
                    non_tp_exits=len(self.attributions)-count,
                    voluntary_wait_fraction=self.voluntary_waits/max(1, self.legal_add_decisions),
                    **{"log/env/account_net_pnl": float(self._calculate_equity())-300000,
                       "log/env/unfinished_entries": len(self.entries),
                       "log/env/speed_reward": self.total_reward})

    def _snapshot_fields(self):
        state = dict(schema=3 if self.single3 else 1, contract=self.contract, config_hash=self.config_hash,
                     cursor_time=str(self._raw.index[self.current_step]),
                     balances=self.ledger.snapshot(), ledger_entries=[dict(timestamp=e.timestamp, memo=e.memo,
                     postings=[dict(account=p.account, amount=p.amount) for p in e.postings], meta=e.meta) for e in self.ledger.entries],
                     long_positions=[p.to_dict() if p else None for p in self.position_manager.long_positions],
                     short_positions=[p.to_dict() if p else None for p in self.position_manager.short_positions],
                     closed_trade_profits=self.position_manager.closed_trade_profits,
                     realized_pnl=self.user_accounts.realized_pnl, unrealized_pnl=self.user_accounts.unrealized_pnl,
                     trade_history=[r.to_dict() for r in self.trade_record_manager.trade_history],
                     rng=self.np_random.bit_generator.state,
                     rewarded=sorted(self.reward_function.rewarded), processed_fills=sorted(self.processed_fills))
        for key in ("current_step", "episode_step_count", "chunk_count", "entry_sequence", "account_sequence",
                    "current_price", "entries", "fills", "attributions", "reward_events", "equity_history",
                    "total_reward", "max_floating_loss", "legal_add_decisions", "voluntary_waits", "status",
                    "terminated", "truncated", "pending_reason", "completed_accounts"):
            state[key] = getattr(self, key)
        if self.single3:
            state.update(reward_state=self.reward_function.snapshot(), last_reward_components=self.last_reward_components)
        return state

    def export_state(self):
        from gym_trading_env.utils.snapshot_json import copy_account_snapshot
        return copy_account_snapshot(self._snapshot_fields())

    def _export_private_state(self):
        from gym_trading_env.utils.native_account import NativeAccountSnapshotView,NativeHistory
        if not isinstance(self.equity_history,NativeHistory):
            raise TypeError('Private export requires a controlled native history')
        return NativeAccountSnapshotView.capture(self,self._snapshot_fields(),self.equity_history.version())

    def _enable_private_history(self):
        """Opt in only for controlled CPU Corpus accounts, never public schema."""
        self._private_history_enabled=True
        self._ensure_private_history()

    def _ensure_private_history(self):
        if not getattr(self,'_private_history_enabled',False):return
        from gym_trading_env.utils.native_account import NativeHistory,native_owner,native_source
        if not isinstance(self.equity_history,NativeHistory):
            self.equity_history=NativeHistory.from_public(self.equity_history,
                owner=native_owner(self),source=native_source(self))

    def import_state(self, state):
        # Preserve the full independently owned JSON snapshot. Native history
        # rows already consist of JSON primitives; don't reformat every float
        # in the growing history on each counterfactual account restore.
        from gym_trading_env.utils.snapshot_json import copy_account_snapshot
        state = copy_account_snapshot(state)
        self._restore_account_state(state)

    def _restore_private_state(self, view, *, source_env):
        from gym_trading_env.utils.native_account import NativeAccountSnapshotView
        if type(view) is not NativeAccountSnapshotView:
            raise TypeError('Native private snapshot required')
        view.restore(self, source_env=source_env)

    def _restore_account_state(self, state):
        # Both restore paths use these financial/clock guards.
        if state["schema"] != (3 if self.single3 else 1) or state["contract"] != self.contract or state["config_hash"] != self.config_hash:
            raise ValueError("Incompatible Entry checkpoint")
        i = int(self._raw.index.get_indexer([timestamp(state["cursor_time"])])[0])
        if i < 0:
            raise ValueError("Checkpoint cursor is not a minute of this market")
        self._initialize_accounts(allow_negative_cash=True)
        self.ledger.restore({k: Decimal(v) for k, v in state["balances"].items()})
        self.ledger.entries = [JournalEntry(timestamp(e["timestamp"]), e["memo"],
            [Posting(p["account"], Decimal(p["amount"])) for p in e["postings"]], e["meta"]) for e in state["ledger_entries"]]
        for side in ("long", "short"):
            setattr(self.position_manager, side+"_positions", [Position(**p) if p else None for p in state[side+"_positions"]])
        self.position_manager.closed_trade_profits = [Decimal(v) for v in state["closed_trade_profits"]]
        self.user_accounts.realized_pnl = Decimal(state["realized_pnl"])
        self.user_accounts.unrealized_pnl = Decimal(state["unrealized_pnl"])
        self.trade_record_manager.trade_history = []
        for raw in state["trade_history"]:
            raw["timestamp"] = timestamp(raw["timestamp"])
            for k in raw.keys() - {"timestamp", "operation_type", "meta"}:
                raw[k] = Decimal(raw[k])
            self.trade_record_manager.record_trade(TradeRecord(**raw))
        for key in ("episode_step_count", "chunk_count", "entry_sequence", "account_sequence", "entries", "fills",
                    "attributions", "reward_events", "equity_history", "total_reward", "max_floating_loss",
                    "legal_add_decisions", "voluntary_waits", "status", "terminated", "truncated", "pending_reason", "completed_accounts"):
            setattr(self, key, state[key])
        self.current_step, self.current_price = i, Decimal(state["current_price"])
        self.last_close_position, self._entries_used_today = None, self.entry_sequence
        if self.single3:
            from gym_trading_env.rewards.single3_reward import Single3Reward
            self.reward_function = Single3Reward.restore(state['reward_state'], self.reward_config,
                                                         self.take_profit_points, self.stop_loss_points)
            self.last_reward_components = state['last_reward_components']
            if self.reward_function.rewarded != set(state['rewarded']):
                raise ValueError('Single3 rewarded identity mismatch')
            if set(self.reward_function.scorer.active) != set(self.entries):
                raise ValueError('Single3 reward/financial active entry identity mismatch')
            for entry_id, item in self.entries.items():
                if self.reward_function.scorer.active[entry_id].bars_seen != self.market_clock_counter.held(item['entry_cursor'], i):
                    raise ValueError('Checkpoint holding clock differs from actual market history')
        else:
            self.reward_function = FastTakeProfitReward()
            self.reward_function.rewarded = set(state["rewarded"])
        self.processed_fills = set(state["processed_fills"])
        self.np_random.bit_generator.state = state["rng"]
        self._assert_ledger_conservation()
        self._initialized = True
        if self.single3:
            self._tp_info_cache=(0,0,0.)
        self._get_obs()
        self._ensure_private_history()

    def continue_data(self, df):
        """Append/backfill unprocessed data without changing already observed history."""
        state = self.export_state()
        old = (self._raw, self._features, self.bar_source, self.bobs, self.valid_fill_clock, self.market_clock_counter)
        try:
            self._load_market(df)
            self.import_state(state)
            if self.status not in ("DATA_END", "DATA_GAP"):
                raise ValueError("Only interrupted trajectories accept new market data")
            self.status, self.truncated = "ACTIVE", False
            return self._get_obs(), self._get_info()
        except Exception:
            self._raw, self._features, self.bar_source, self.bobs, self.valid_fill_clock, self.market_clock_counter = old
            self.import_state(state)
            raise

    def close(self):
        path = self.entry_config.get("state_path")
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            temp = Path(str(path)+".tmp")
            temp.write_text(json.dumps(self.export_state(), indent=2), encoding="utf-8")
            temp.replace(path)
