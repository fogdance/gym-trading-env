# tests/test_observation_space.py
import unittest
import numpy as np
import pandas as pd
from decimal import Decimal

from gym_trading_env.envs.trading_env import CustomTradingEnv, Action
from gym_trading_env.utils.build_xt import FEATURES_MARKET, FEATURES_AGENT

def make_one_session_minute_df(start="2020-01-01 05:00:00", periods=1440):
    """
    Build a clean 1-minute FX session (no gaps) so mask_t==1 for all minutes.
    Prices drift linearly to make assertions simple.
    """
    idx = pd.date_range(start=start, periods=periods, freq="min")  # 1 minute
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

class TestObservationSpace(unittest.TestCase):
    def setUp(self):
        # Deterministic, single session, all bars valid.
        df = make_one_session_minute_df()
        # Use the same YAML you already use; we only force deterministic start
        self.env = CustomTradingEnv(df=df, config_path="tests/test.yaml")
        # Force no randomization so reset always starts at the first usable day/minute
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
        At reset current_minute should be 0.
        => Only row 0 may be non-zero; rows [1..] must be all zeros.
        Also, mask_t column of row 0 must be 1 (valid bar).
        """
        market_seq = self.obs0["market_seq"]
        nonzero_rows = np.where(np.any(market_seq != 0.0, axis=1))[0]
        self.assertListEqual(nonzero_rows.tolist(), [0], "Only minute 0 should be visible at reset")

        # Find mask_t column by name to avoid brittle index assumptions
        mask_col = FEATURES_MARKET.index("mask_t")
        self.assertEqual(market_seq[0, mask_col], 1.0, "mask_t at minute 0 must be 1")
        self.assertTrue(np.all(market_seq[1:, :] == 0.0), "Future minutes must be zeroed")

    def test_agent_state_at_reset(self):
        """
        Agent vector is compact (19-D). Check key fields by index using FEATURES_AGENT.
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

if __name__ == "__main__":
    unittest.main()
