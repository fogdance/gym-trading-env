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
from .entry_contract import (ContractTerms, DirectionSchedule, fingerprint,
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


class FixedDirectionEntryEnv(CustomTradingEnv):
    """WAIT/ADD_3 strategy; does not call or rewrite legacy target-position step."""
    metadata = {"render_modes": []}

    def __init__(self, df=None, config_path=None, render_mode=None, *, market_cache=None):
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
        self.max_lots, self.max_entries = (3, 1) if self.single3 else (6, 2)
        self.fill_model = SINGLE3_FILL_MODEL if self.single3 else "minute_close_confirm_v1"
        self.contract = self.entry_config["contract"]
        self.strategy_path=None
        if self.entry_config.get('strategy_config'):
            self.strategy_path=resolve_path(config_path,self.entry_config['strategy_config'])
            master=read_yaml(self.strategy_path)
            expected_reward = master.get('protocol') == SINGLE3_PROTOCOL if self.single3 else master['reward']=={
                    'formula':'60/(60+T_minutes)', 'complete_quantity':3,'take_profit_points':5}
            if (master['product']!='FG' or not expected_reward or master['expiry']['mode']!='main_qualification'):
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
        if tr.live_mode or not tr.is_future or tr.future_symbol != self.contract:
            raise ValueError("Entry v1 requires an offline matching FG futures config")
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
        self.observation_space = spaces.Dict({
            "market_seq": spaces.Box(-np.inf, np.inf, (self.window_size, self._features.shape[1]), np.float32),
            "position_state": spaces.Box(-np.inf, np.inf, (9,), np.float32),
            "action_mask": spaces.Box(0, 1, (2,), np.float32),
        })
        self.completed_accounts = []
        self.account_sequence = 0
        self._initialized = False
        self.reset()

    def _load_market(self, df):
        if df is None:
            df = pd.read_csv(Path(self.config.trading.data_path).expanduser())
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
            features=build_market_features(feature_raw,is_future=True,real_futures_clock=True,futures_schedule=self.market_clock)
            features = features.reindex(raw.index).ffill().fillna(0)
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
        # Existing DB contains zero-duration metadata rows: mark-only, never fills.

    @property
    def net_lots(self):
        return self.user_accounts.long_position + self.user_accounts.short_position

    def _positions(self):
        side = "LONG" if self.user_accounts.long_position else "SHORT"
        positions = self.position_manager.long_positions if side == "LONG" else self.position_manager.short_positions
        return side, positions

    def _set_terms(self, time, *, closing=False, entry=None):
        terms = self.terms.at(time)
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
        return str(self._raw.iloc[i][column]) if column in self._raw else str(timestamp(time).date())

    def _price(self, raw_price, side, time, *, opening=False):
        terms = self.terms.at(time)
        buy = (side == "LONG") == opening
        price = Decimal(str(raw_price)) + (1 if buy else -1) * (terms["spread"] + terms["slippage"])
        rounding = ROUND_CEILING if buy else ROUND_FLOOR
        return (price / self.terms.tick).to_integral_value(rounding=rounding) * self.terms.tick

    def _mark(self, price, time):
        side, _ = self._positions()
        self.current_price = self._price(price, side, time) if self.net_lots else Decimal(str(price))
        self._update_unrealized_pnl()

    def _compute_stop_loss_price(self, entry_exec_price, side):
        return entry_exec_price + (-30 if side == "long" else 30) if self.single3 else None

    def _compute_take_profit_price(self, entry_exec_price, sl_exec_price, side):
        return entry_exec_price + (5 if side == "long" else -5)

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
        return self.net_lots * self.terms.multiplier * Decimal(str(price)) * self.terms.at(time)["margin_rate"]

    def direction_at(self, time, row_index=None):
        if self.directions.date_basis == "trading_date":
            row = self._raw.iloc[self.current_step if row_index is None else row_index]
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
        self._set_terms(time)
        price = self._price(self._raw.iloc[self.current_step].Close, direction, time, opening=True)
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
            distance = sign * (pos.take_profit_price - self.current_price)
            offset = 3 + 3 * item["obs_slot"]
            position[offset:offset+3] = [float(pos.size / 3), float(distance / 5),
                                       np.log1p(age) / np.log1p(43200)]
        position[1] = float(held_quantity / self.max_lots)
        if held_quantity != self.net_lots:
            raise RuntimeError("PositionManager/account quantities lack a matching entry identity")
        mask = np.array([1, self._can_add()], np.float32)
        self._published_mask = mask.copy()
        return {
            "market_seq": np.nan_to_num(market).astype(np.float32),
            "position_state": position, "action_mask": mask,
        }

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if options and "resume" in options:
            self.import_state(options["resume"])
            return self._get_obs(), self._get_info()
        if self._initialized and not self.terminated:
            if self.status == "CHUNK_END":
                self.status, self.truncated, self.chunk_count = "ACTIVE", False, 0
            elif self.current_step == 0 and not self.entries and not self.fills:
                pass
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
        self._initialized = True
        self._sample_equity()
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
        if reason == "TAKE_PROFIT" and (1 if side == "LONG" else -1) * (price - Decimal(item["entry_price"])) < 5:
            raise ValueError("TP execution price does not reach five points")
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
        return Single3Reward(self.reward_config)

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
        row = self._raw.iloc[self.current_step]
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

    def _prefix_hash(self, through):
        return fingerprint(self._raw.iloc[:through+1].to_dict(orient="split"))

    def export_state(self):
        state = dict(schema=3 if self.single3 else 1, contract=self.contract, config_hash=self.config_hash,
                     cursor_time=str(self._raw.index[self.current_step]), prefix_hash=self._prefix_hash(self.current_step),
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
        return json.loads(json.dumps(state, default=str))

    def import_state(self, state):
        state = copy.deepcopy(state)
        if state["schema"] != (3 if self.single3 else 1) or state["contract"] != self.contract or state["config_hash"] != self.config_hash:
            raise ValueError("Incompatible Entry checkpoint")
        i = int(self._raw.index.get_indexer([timestamp(state["cursor_time"])])[0])
        if i < 0 or self._prefix_hash(i) != state["prefix_hash"]:
            raise ValueError("Processed market history changed")
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
            self.reward_function = Single3Reward.restore(state['reward_state'], self.reward_config)
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
