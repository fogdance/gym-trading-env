# tests/test_actions_observation_strict.py
import math
import unittest
import numpy as np
import pandas as pd

from gym_trading_env.envs.trading_env import CustomTradingEnv, Action
from gym_trading_env.utils.market_features import FEATURES_MARKET, build_market_features
from gym_trading_env.utils.agent_features import FEATURES_AGENT

from gym_trading_env.utils.trade_util import step_wrapper
import pytest
pytestmark = pytest.mark.unit

# ================= Configuration knobs for alignment ===================


# If your env fills on the *current* visible bar (price at t) set True;
# if it fills on the *next* bar (price at t+1) set False.
FILL_ON_CURRENT_BAR = True

# ===================== Deterministic test data =========================

def make_df_linear(start="2020-01-01 21:01:00", minutes=100, p0=1.1000, dp=0.001, volume=1.0):
    """
    Build a deterministic 1-min series:
    Close/Open/High/Low drift linearly; volume constant.
    """
    idx = pd.date_range(start=start, periods=minutes, freq="min")
    close = p0 + dp * np.arange(minutes, dtype=float)
    df = pd.DataFrame(
        {
            "Date": idx,
            "Open": close,
            "High": close,
            "Low":  close,
            "Close": close,
            "Volume": np.full(minutes, volume, dtype=float),
        }
    )
    return df

def _nonzero_rows(mat):
    """Return visible row indices where any feature is non-zero."""
    return np.where(np.any(mat != 0.0, axis=1))[0].tolist()

def _assert_close(testcase, a, b, tol=1e-5, msg=""):
    """Float32-safe approximate equality."""
    testcase.assertTrue(abs(float(a) - float(b)) <= tol, msg or f"{a} != {b} (tol={tol})")

def _mask_col():
    return FEATURES_MARKET.index("mask_t")

# ======================== Expected math helpers ========================

def price_at(df, t):
    """Close at index t as float."""
    return float(df.iloc[t]["Close"])

def ask_from_close(c, spread):
    return float(c) + float(spread)

def bid_from_close(c, spread):
    return float(c) - float(spread)

def notional(lot, lot_size):
    return float(lot) * float(lot_size)

def fee_per_side(trading_fee_per_lot, trade_lot):
    return float(trade_lot) * float(trading_fee_per_lot)

def long_upnl(next_close, entry_ask, lot, lot_size):
    # Long uPnL = (next_close - entry_ask) * lot * lot_size
    return (float(next_close) - float(entry_ask)) * notional(lot, lot_size)

def short_upnl(next_close, entry_bid, lot, lot_size):
    # Short uPnL = (entry_bid - next_close) * lot * lot_size
    return (float(entry_bid) - float(next_close)) * notional(lot, lot_size)

def long_realized(exit_bid, entry_ask, lot, lot_size):
    # Close long at bid
    return (float(exit_bid) - float(entry_ask)) * notional(lot, lot_size)

def short_realized(exit_ask, entry_bid, lot, lot_size):
    # Close short at ask
    return (float(entry_bid) - float(exit_ask)) * notional(lot, lot_size)

def entry_price_bar_index(current_visible_t):
    """Which bar the env uses to *fill* entries."""
    return current_visible_t if FILL_ON_CURRENT_BAR else current_visible_t + 1

def exit_price_bar_index(current_visible_t):
    """Which bar the env uses to *fill* exits."""
    return current_visible_t if FILL_ON_CURRENT_BAR else current_visible_t + 1

# =============================== Test =================================

class TestActionObservationStrict(unittest.TestCase):
    """
    Step through: HOLD → LONG_OPEN0 → LONG_CLOSE0 → SHORT_OPEN0 → SHORT_CLOSE0.
    After each step, strictly verify observation content (market_seq + agent_state),
    and basic info consistency (fees_cum, equity).
    """

    def setUp(self):
        self.df = make_df_linear(minutes=100, p0=1.1000, dp=0.0001, volume=1.0)
        self.env = CustomTradingEnv(df=self.df, config_path="tests/test.yaml")

        self.FEE_ON_CLOSE = bool(self.env.config.trading.is_round_turn)
        self.fee_per_side = float(self.env.config.trading.trading_fee_per_lot * self.env.config.trading.trade_lot)

        # Deterministic start from first available minute.
        self.env.config.training.randomize_start = False

        # Pull constants from config for exact math
        self.cfg = self.env.config.trading
        self.spread = float(self.cfg.spread)
        self.trade_lot = float(self.cfg.trade_lot)
        self.lot_size = float(self.cfg.lot_size)
        self.trading_fee_per_lot = float(self.cfg.trading_fee_per_lot)

        # Reset
        self.obs, self.info = self.env.reset()
        self.t = 0  # visible current minute index in the window
        self.mask_col = _mask_col()

        # Track expected agent state
        self.pos = 0.0
        self.have_long = 0.0
        self.have_short = 0.0
        self.entry_price = 0.0  # ask for long, bid for short (side_spread)
        self.holding_minutes = 0.0
        self.upnl = 0.0
        self.realized_step = 0.0
        self.realized_cum = 0.0
        self.fee_step = 0.0
        self.fee_cum = 0.0
        self.equity = float(self.info["equity"])  # initial equity
        self.max_equity = self.equity

    def tearDown(self):
        try:
            self.env.close()
        except Exception:
            pass

    # ---------------- Shared assertions on observation contract ----------------

    def _assert_market_reveal(self, obs_market_seq, expected_visible_last):
        """
        Visible rows must be exactly [0..expected_visible_last],
        mask_t at the frontier == 1, future rows zero.
        """
        nz = _nonzero_rows(obs_market_seq)
        self.assertListEqual(nz, list(range(expected_visible_last + 1)),
                             f"Visible rows must be [0..{expected_visible_last}]")
        _assert_close(self, obs_market_seq[expected_visible_last, self.mask_col], 1.0,
                      msg="mask_t at frontier must be 1")
        if expected_visible_last + 1 < obs_market_seq.shape[0]:
            self.assertTrue(
                np.all(obs_market_seq[expected_visible_last + 1 :, :] == 0.0),
                "Future rows must remain zeroed",
            )

    def _assert_agent_vector(self, agent_vec):
        """Check agent vector equals our expected tracker values."""
        idx = {name: FEATURES_AGENT.index(name) for name in FEATURES_AGENT}
        _assert_close(self, agent_vec[idx["pos_t"]], self.pos)
        _assert_close(self, agent_vec[idx["have_long_t"]], self.have_long)
        _assert_close(self, agent_vec[idx["have_short_t"]], self.have_short)
        _assert_close(self, agent_vec[idx["entry_price_t"]], self.entry_price)
        _assert_close(self, agent_vec[idx["holding_minutes_t"]], self.holding_minutes)
        _assert_close(self, agent_vec[idx["upnl_t"]], self.upnl, tol=1e-5)
        _assert_close(self, agent_vec[idx["realized_pnl_step_t"]], self.realized_step, tol=1e-5)
        _assert_close(self, agent_vec[idx["realized_pnl_cum_t"]], self.realized_cum, tol=1e-5)
        _assert_close(self, agent_vec[idx["fee_step_t"]], self.fee_step, tol=1e-5)
        _assert_close(self, agent_vec[idx["fee_cum_t"]], self.fee_cum, tol=1e-5)
        _assert_close(self, agent_vec[idx["equity_t"]], self.equity, tol=1e-2)

        # Peak equity & drawdown
        self.max_equity = max(self.max_equity, self.equity)
        _assert_close(self, agent_vec[idx["max_equity_t"]], self.max_equity, tol=1e-5)
        dd = self.max_equity - self.equity
        _assert_close(self, agent_vec[idx["drawdown_t"]], dd, tol=1e-5)

    def _assert_info_consistency(self, info):
        """Basic info cross-checks with our trackers."""
        self.assertIn("fees_collected", info)
        self.assertIn("equity", info)
        _assert_close(self, float(info["fees_collected"]), self.fee_cum, tol=1e-8, msg="fees_collected mismatch")
        _assert_close(self, float(info["equity"]), self.equity, tol=1e-5, msg="equity mismatch")

    # --------------------------------- Test -----------------------------------

    def test_action_sequence_observation_strict(self):
        """
        Execute the sequence:
          1) HOLD
          2) LONG_OPEN0
          3) LONG_CLOSE0
          4) SHORT_OPEN0
          5) SHORT_CLOSE0
        After each step, assert market_seq/agent_state are strictly correct,
        and info has consistent cumulative fees and equity.
        """

        # --- 1) HOLD ---
        obs, reward, terminated, truncated, info = step_wrapper(self.env,Action.HOLD)
        self.t += 1  # frontier advances by one row

        market_seq = obs["market_seq"]
        agent_vec = obs["agent_state"].astype(float)
        self._assert_market_reveal(market_seq, expected_visible_last=self.t)

        # No position → no changes
        self.holding_minutes = 0.0
        self.upnl = 0.0
        self.realized_step = 0.0
        self.fee_step = 0.0
        # Equity should not change on HOLD with no position
        self._assert_agent_vector(agent_vec)
        self._assert_info_consistency(info)

        # --- 2) LONG_OPEN0 ---
        # Entry filled using side_spread at chosen timing bar
        entry_bar = entry_price_bar_index(self.t)
        pre_close = price_at(self.df, entry_bar)
        entry_ask = ask_from_close(pre_close, self.spread)
        fee_open = fee_per_side(self.trading_fee_per_lot, self.trade_lot)

        obs, reward, terminated, truncated, info = step_wrapper(self.env,Action.LONG_OPEN0)
        self.t += 1
        market_seq = obs["market_seq"]
        agent_vec = obs["agent_state"].astype(float)
        self._assert_market_reveal(market_seq, expected_visible_last=self.t)

        # Update expected agent state
        self.pos = self.trade_lot
        self.have_long, self.have_short = 1.0, 0.0
        self.entry_price = entry_ask
        # If your env increments holding minutes immediately on entry, set = 1.0
        self.holding_minutes = 1

        next_close = price_at(self.df, self.t)  # newly revealed bar after step
        self.upnl = long_upnl(next_close, self.entry_price, self.trade_lot, self.lot_size)
        self.realized_step = 0.0
        self.realized_cum += self.realized_step
        self.fee_step = fee_open
        self.fee_cum += fee_open
        # Equity is env-defined; we trust it and verify consistency
        self.equity = float(info["equity"])
        self._assert_agent_vector(agent_vec)
        self._assert_info_consistency(info)

        # --- 3) LONG_CLOSE0 ---
        exit_bar = exit_price_bar_index(self.t)
        pre_close = price_at(self.df, exit_bar)
        exit_bid = bid_from_close(pre_close, self.spread)
        fee_close = fee_per_side(self.trading_fee_per_lot, self.trade_lot) if self.FEE_ON_CLOSE else 0.0
        realized = long_realized(exit_bid, self.entry_price, self.trade_lot, self.lot_size)

        obs, reward, terminated, truncated, info = step_wrapper(self.env,Action.LONG_CLOSE0)
        self.t += 1
        market_seq = obs["market_seq"]
        agent_vec = obs["agent_state"].astype(float)
        self._assert_market_reveal(market_seq, expected_visible_last=self.t)

        # Reset to flat
        self.pos = 0.0
        self.have_long, self.have_short = 0.0, 0.0
        self.entry_price = 0.0
        self.holding_minutes = 0.0
        self.upnl = 0.0
        self.realized_step = realized
        self.realized_cum += self.realized_step
        self.fee_step = fee_close
        self.fee_cum += fee_close
        self.equity = float(info["equity"])
        self._assert_agent_vector(agent_vec)
        self._assert_info_consistency(info)

        # --- 4) SHORT_OPEN0 ---
        entry_bar = entry_price_bar_index(self.t)
        pre_close = price_at(self.df, entry_bar)
        entry_bid = bid_from_close(pre_close, self.spread)
        fee_open_s = fee_per_side(self.trading_fee_per_lot, self.trade_lot)

        obs, reward, terminated, truncated, info = step_wrapper(self.env,Action.SHORT_OPEN0)
        self.t += 1
        market_seq = obs["market_seq"]
        agent_vec = obs["agent_state"].astype(float)
        self._assert_market_reveal(market_seq, expected_visible_last=self.t)

        self.pos = -self.trade_lot
        self.have_long, self.have_short = 0.0, 1.0
        self.entry_price = entry_bid
        self.holding_minutes = 1
        next_close = price_at(self.df, self.t)
        self.upnl = short_upnl(next_close, self.entry_price, self.trade_lot, self.lot_size)
        self.realized_step = 0.0
        self.fee_step = fee_open_s
        self.fee_cum += fee_open_s
        self.equity = float(info["equity"])
        self._assert_agent_vector(agent_vec)
        self._assert_info_consistency(info)

        # --- 5) SHORT_CLOSE0 ---
        exit_bar = exit_price_bar_index(self.t)
        pre_close = price_at(self.df, exit_bar)
        exit_ask = ask_from_close(pre_close, self.spread)
        fee_close_s = fee_per_side(self.trading_fee_per_lot, self.trade_lot) if self.FEE_ON_CLOSE else 0.0
        realized_s = short_realized(exit_ask, self.entry_price, self.trade_lot, self.lot_size)

        obs, reward, terminated, truncated, info = step_wrapper(self.env,Action.SHORT_CLOSE0)
        self.t += 1
        market_seq = obs["market_seq"]
        agent_vec = obs["agent_state"].astype(float)
        self._assert_market_reveal(market_seq, expected_visible_last=self.t)

        self.pos = 0.0
        self.have_long, self.have_short = 0.0, 0.0
        self.entry_price = 0.0
        self.holding_minutes = 0.0
        self.upnl = 0.0
        self.realized_step = realized_s
        self.realized_cum += self.realized_step
        self.fee_step = fee_close_s
        self.fee_cum += fee_close_s
        self.equity = float(info["equity"])
        self._assert_agent_vector(agent_vec)
        self._assert_info_consistency(info)

        # Final sanity: should not terminate/truncate in this short path.
        self.assertFalse(terminated, "Env terminated unexpectedly")
        self.assertFalse(truncated, "Env truncated unexpectedly")


if __name__ == "__main__":
    unittest.main()
