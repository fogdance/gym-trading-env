# tests/test_observation_space.py
import unittest
from unittest import mock
import numpy as np
import pandas as pd

from gym_trading_env.envs.trading_env import CustomTradingEnv, Action
from gym_trading_env.utils.build_xt import FEATURES_MARKET, FEATURES_AGENT

def make_one_session_minute_df(start="2020-01-01 05:00:00", periods=1440):
    """
    Build a clean 1-minute FX session (no gaps) so mask_t==1 for all minutes.
    Prices drift linearly to make assertions simple.
    """
    idx = pd.date_range(start=start, periods=periods, freq="min")
    base = 1.1000
    drift = np.linspace(0, 0.0020, periods)  # ~20 pips over the day
    c = base + drift
    df = pd.DataFrame(
        {
            "Date": idx,
            "Open": c,
            "High": c,
            "Low":  c,
            "Close": c,
            "Volume": np.ones(periods, dtype=float),
        }
    )
    return df

def make_df_with_gap(
    full_start="2020-01-01 05:00:00",
    hours=6,
    gap_start="2020-01-01 05:00:00",
    gap_minutes=60,
):
    """
    Create a minute-level DataFrame for `hours` hours starting at `full_start`,
    then remove the interval [gap_start, gap_start + gap_minutes) to simulate a
    leading data gap. The remaining timestamps stay strictly increasing.
    """
    periods = hours * 60
    df = make_one_session_minute_df(start=full_start, periods=periods)
    gap_index = pd.date_range(start=gap_start, periods=gap_minutes, freq="min")
    df = df[~df["Date"].isin(gap_index)].reset_index(drop=True)
    return df

class TestObservationSpace(unittest.TestCase):
    def setUp(self):
        # Deterministic, single session, all minutes valid.
        df = make_one_session_minute_df()
        self.env = CustomTradingEnv(df=df, config_path="tests/test.yaml")
        # Force deterministic start so reset always begins at the first usable minute.
        self.env.config.training.randomize_start = False
        obs, info = self.env.reset()
        self.obs0 = obs
        self.info0 = info

    def test_shapes_and_dtypes(self):
        market_seq = self.obs0["market_seq"]
        agent_state = self.obs0["agent_state"]

        self.assertEqual(market_seq.shape[0], 1440)
        self.assertEqual(market_seq.shape[1], len(FEATURES_MARKET))
        self.assertEqual(agent_state.shape[0], len(FEATURES_AGENT))
        self.assertEqual(market_seq.dtype, np.float32)
        self.assertEqual(agent_state.dtype, np.float32)

    def test_temporal_mask_at_reset(self):
        """
        At reset current_minute should be 0 in the visible window.
        => Only row 0 may be non-zero; rows [1..] must be all zeros.
        Also, mask_t column of row 0 must be 1 (valid bar).
        """
        market_seq = self.obs0["market_seq"]
        nonzero_rows = np.where(np.any(market_seq != 0.0, axis=1))[0]
        self.assertListEqual(nonzero_rows.tolist(), [0], "Only minute 0 should be visible at reset")

        mask_col = FEATURES_MARKET.index("mask_t")
        self.assertEqual(market_seq[0, mask_col], 1.0, "mask_t at minute 0 must be 1")
        self.assertTrue(np.all(market_seq[1:, :] == 0.0), "Future minutes must be zeroed")

    def test_agent_state_at_reset(self):
        """
        Agent vector is compact (e.g., 19-D). Check key fields by index using FEATURES_AGENT.
        """
        a = self.obs0["agent_state"]
        idx = {name: FEATURES_AGENT.index(name) for name in FEATURES_AGENT}

        self.assertEqual(a[idx["pos_t"]], 0.0)
        self.assertEqual(a[idx["have_long_t"]], 0.0)
        self.assertEqual(a[idx["have_short_t"]], 0.0)
        # Equity equals initial balance, pulled from config
        self.assertAlmostEqual(a[idx["equity_t"]], float(self.env.config.trading.initial_balance), places=6)
        self.assertEqual(a[idx["realized_pnl_cum_t"]], 0.0)
        self.assertEqual(a[idx["upnl_t"]], 0.0)

    def test_temporal_reveal_after_one_step(self):
        """
        After step(HOLD), minute visibility must advance by exactly one:
        non-zero rows should be [0, 1].
        """
        obs1, reward, terminated, truncated, info = self.env.step(Action.HOLD.value)
        market_seq = obs1["market_seq"]

        nonzero_rows = np.where(np.any(market_seq != 0.0, axis=1))[0]
        self.assertListEqual(nonzero_rows.tolist(), [0, 1], "Minutes 0..1 should be visible after one step")

        mask_col = FEATURES_MARKET.index("mask_t")
        self.assertEqual(market_seq[1, mask_col], 1.0, "mask_t at minute 1 must be 1")
        self.assertTrue(np.all(market_seq[2:, :] == 0.0), "Future minutes must remain zero")

    # NEW: Arbitrary start minute (environment may initialize at any minute).
    def test_reset_from_arbitrary_start(self):
        """
        The environment may start at any (valid) minute (e.g., 09:37).
        Contract stays the same:
        - At reset only row 0 is non-zero; future rows are zero.
        - mask_t at row 0 must be 1.
        - After one step, visible rows become [0, 1].
        Note: in the visible window, row 0 always denotes "current minute",
        not the absolute index in the original DataFrame.
        """
        df = make_one_session_minute_df()
        env = CustomTradingEnv(df=df, config_path="tests/test.yaml")
        env.config.training.randomize_start = True

        forced_start = 123  # any index > 0 to simulate non-zero start
        if hasattr(env, "_sample_start_index"):
            # If your env has a sampler method, patch it to force a deterministic arbitrary start.
            with mock.patch.object(env, "_sample_start_index", return_value=forced_start):
                obs, info = env.reset()
        else:
            # Fallback: try seeding reset; if unsupported, reset() without seed still validates the contract.
            try:
                obs, info = env.reset(seed=1234)
            except TypeError:
                obs, info = env.reset()

        market_seq = obs["market_seq"]
        mask_col = FEATURES_MARKET.index("mask_t")

        nonzero_rows = np.where(np.any(market_seq != 0.0, axis=1))[0]
        self.assertListEqual(nonzero_rows.tolist(), [0], "At arbitrary start, only current minute (row 0) must be visible")
        self.assertEqual(market_seq[0, mask_col], 1.0)
        self.assertTrue(np.all(market_seq[1:, :] == 0.0))

        obs1, reward, terminated, truncated, info1 = env.step(Action.HOLD.value)
        market_seq1 = obs1["market_seq"]
        nonzero_rows1 = np.where(np.any(market_seq1 != 0.0, axis=1))[0]
        self.assertListEqual(nonzero_rows1.tolist(), [0, 1])
        self.assertEqual(market_seq1[1, mask_col], 1.0)

        env.close()

    # NEW: Leading gap (e.g., 05:00–06:00 missing; data starts at 06:00).
    def test_reset_from_first_available_when_leading_gap(self):
        """
        When the earliest portion of the session is missing (e.g., 05:00–06:00),
        reset should start from the first available minute.
        Same assertions: only row 0 visible at reset, mask_t==1; after one step, rows [0, 1].
        """
        df_gap = make_df_with_gap(
            full_start="2020-01-01 05:00:00",
            hours=6,                          # 05:00–11:00
            gap_start="2020-01-01 05:00:00",  # drop the first hour
            gap_minutes=60
        )
        env = CustomTradingEnv(df=df_gap, config_path="tests/test.yaml")
        env.config.training.randomize_start = False  # start from first available
        obs, info = env.reset()

        market_seq = obs["market_seq"]
        mask_col = FEATURES_MARKET.index("mask_t")

        nonzero_rows = np.where(np.any(market_seq != 0.0, axis=1))[0]
        self.assertListEqual(nonzero_rows.tolist(), [0], "With a leading gap, reset still shows only current minute")
        self.assertEqual(market_seq[0, mask_col], 1.0)
        self.assertTrue(np.all(market_seq[1:, :] == 0.0))

        obs1, reward, terminated, truncated, info1 = env.step(Action.HOLD.value)
        market_seq1 = obs1["market_seq"]
        nonzero_rows1 = np.where(np.any(market_seq1 != 0.0, axis=1))[0]
        self.assertListEqual(nonzero_rows1.tolist(), [0, 1])
        self.assertEqual(market_seq1[1, mask_col], 1.0)

        env.close()

if __name__ == "__main__":
    unittest.main()
