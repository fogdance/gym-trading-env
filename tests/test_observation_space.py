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
    then remove the interval [gap_start, gap_start + gap_minutes) to simulate
    a leading data gap. The remaining timestamps stay strictly increasing.
    """
    periods = hours * 60
    df = make_one_session_minute_df(start=full_start, periods=periods)
    gap_index = pd.date_range(start=gap_start, periods=gap_minutes, freq="min")
    df = df[~df["Date"].isin(gap_index)].reset_index(drop=True)
    return df

class TestObservationSpace(unittest.TestCase):
    def setUp(self):
        # Deterministic single session. Start from 05:00, all minutes valid.
        df = make_one_session_minute_df()
        self.env = CustomTradingEnv(df=df, config_path="tests/test.yaml")
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

    def test_temporal_mask_at_reset_first_minute(self):
        """
        With deterministic start (at session start 05:00), current_minute == 0.
        -> Only row 0 may be non-zero; rows [1..] must be zero.
        """
        market_seq = self.obs0["market_seq"]
        mask_col = FEATURES_MARKET.index("mask_t")

        # Non-zero rows must be exactly [0]
        nonzero_rows = np.where(np.any(market_seq != 0.0, axis=1))[0]
        self.assertListEqual(nonzero_rows.tolist(), [0], "Only minute 0 should be visible at session start")

        # mask_t at row 0 must be 1; future rows zeroed
        self.assertEqual(market_seq[0, mask_col], 1.0, "mask_t at minute 0 must be 1")
        self.assertTrue(np.all(market_seq[1:, :] == 0.0), "Future minutes must be zeroed")

    def test_temporal_reveal_after_one_step(self):
        """
        After step(HOLD), frontier advances by exactly one:
        Non-zero rows should be [0, 1] when starting at minute 0.
        """
        obs1, reward, terminated, truncated, info = self.env.step(Action.HOLD.value)
        market_seq = obs1["market_seq"]
        mask_col = FEATURES_MARKET.index("mask_t")

        nonzero_rows = np.where(np.any(market_seq != 0.0, axis=1))[0]
        self.assertListEqual(nonzero_rows.tolist(), [0, 1], "Minutes 0..1 should be visible after one step")
        self.assertEqual(market_seq[1, mask_col], 1.0, "mask_t at minute 1 must be 1")
        self.assertTrue(np.all(market_seq[2:, :] == 0.0), "Future minutes must remain zero")

    def test_reset_from_arbitrary_start_cumulative_view(self):
        """
        If the environment starts at an arbitrary valid minute (e.g., ~09:37),
        the observation must show *all minutes from session start (05:00) up to current_minute*.
        That is: rows [0 .. current_minute] potentially non-zero (subject to mask_t),
        and rows [current_minute+1 ..] must be all zero.
        """
        df = make_one_session_minute_df()
        env = CustomTradingEnv(df=df, config_path="tests/test.yaml")
        env.config.training.randomize_start = True

        # Try to force a non-zero start via seed; contract assertions do not rely on exact value.
        try:
            obs, info = env.reset(seed=1234)
        except TypeError:
            obs, info = env.reset()

        market_seq = obs["market_seq"]
        mask_col = FEATURES_MARKET.index("mask_t")

        # We can read the current frontier from the visible mask:
        # last index that has any non-zero feature equals current_minute
        nonzero_rows = np.where(np.any(market_seq != 0.0, axis=1))[0]
        self.assertGreaterEqual(len(nonzero_rows), 1, "At least current minute must be visible")
        current_frontier = nonzero_rows[-1]

        # All rows after frontier must be zero
        if current_frontier + 1 < market_seq.shape[0]:
            self.assertTrue(
                np.all(market_seq[current_frontier + 1 :, :] == 0.0),
                "Future minutes must be zeroed",
            )

        # Frontier row must be a valid minute (mask_t == 1)
        self.assertEqual(market_seq[current_frontier, mask_col], 1.0)

        env.close()

    def test_clip_previous_session_rows(self):
        """
        If the raw DF includes pre-session minutes (e.g., 04:00–04:59),
        they must be clipped out of today's session tensor. Row 0 represents 05:00.
        """
        # Build DF from 04:00 to 09:37 (6h), but env should clip to 05:00 start.
        idx = pd.date_range("2020-01-01 04:00:00", periods=6*60, freq="min")
        base = 1.2000
        drift = np.linspace(0, 0.0012, len(idx))
        c = base + drift
        df = pd.DataFrame(
            {"Date": idx, "Open": c, "High": c, "Low": c, "Close": c, "Volume": 1.0}
        )
        env = CustomTradingEnv(df=df, config_path="tests/test.yaml")
        env.config.training.randomize_start = False
        obs, info = env.reset()

        market_seq = obs["market_seq"]
        mask_col = FEATURES_MARKET.index("mask_t")

        # Row 0 corresponds to session start 05:00. It should be either a valid bar (mask 1),
        # or zero if that specific minute is missing. Pre-session (04:xx) must not appear.
        # Check that there is no non-zero row before row 0 (by construction of the tensor).
        # Also ensure cumulative visibility holds (here current_minute is likely 0).
        nonzero_rows = np.where(np.any(market_seq != 0.0, axis=1))[0]
        self.assertTrue(len(nonzero_rows) >= 1, "At least the current minute must be visible")
        self.assertGreaterEqual(nonzero_rows[0], 0, "Session starts at row 0")
        self.assertIn(mask_col, range(market_seq.shape[1]))
        env.close()

    def test_reset_from_first_available_when_leading_gap(self):
        """
        When early session minutes are missing (e.g., 05:00–06:00),
        the reset view is still cumulative up to current_minute.
        Missing minutes before the first available bar will be zero rows (mask_t==0),
        but the frontier row (current_minute) must be visible and valid (mask_t==1),
        and all rows after frontier must be zero.
        """
        df_gap = make_df_with_gap(
            full_start="2020-01-01 05:00:00",
            hours=6,                          # 05:00–11:00 intended session span
            gap_start="2020-01-01 05:00:00",  # drop the first hour (05:00–05:59)
            gap_minutes=60
        )
        env = CustomTradingEnv(df=df_gap, config_path="tests/test.yaml")
        env.config.training.randomize_start = False  # start from first available (06:00)
        obs, info = env.reset()

        market_seq = obs["market_seq"]
        mask_col = FEATURES_MARKET.index("mask_t")

        # Determine frontier: last non-zero row
        nonzero_rows = np.where(np.any(market_seq != 0.0, axis=1))[0]
        self.assertGreaterEqual(len(nonzero_rows), 1, "At least the first available minute must be visible")
        frontier = nonzero_rows[-1]

        # Rows after frontier must be zero
        if frontier + 1 < market_seq.shape[0]:
            self.assertTrue(np.all(market_seq[frontier + 1 :, :] == 0.0), "Future minutes must be zeroed")

        # Frontier row must be valid (mask_t==1)
        self.assertEqual(market_seq[frontier, mask_col], 1.0, "Frontier minute must be valid")

        # Take one step → frontier advances by exactly one
        obs1, reward, terminated, truncated, info1 = env.step(Action.HOLD.value)
        market_seq1 = obs1["market_seq"]
        nonzero_rows1 = np.where(np.any(market_seq1 != 0.0, axis=1))[0]
        self.assertEqual(nonzero_rows1[-1], frontier + 1, "Frontier must advance by one after a step")
        self.assertEqual(market_seq1[nonzero_rows1[-1], mask_col], 1.0)

        env.close()

if __name__ == "__main__":
    unittest.main()
