"""Read-only original Env probe for CPU reference verification.

Run with the existing Gym-compatible interpreter and CUDA_VISIBLE_DEVICES=''.
Only observations, market cutoff and code identities are exported. No account
records, Oracle calls, checkpoint load or formal Run creation are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    parser.add_argument("--sequential", action="store_true", help="WAIT through the controlled boundary fixture")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "src"))
    from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
    from gym_trading_env.utils.market_features import FEATURES_MARKET_OBS_OUTPUT
    env = FixedDirectionEntryEnv(config_path=args.env)
    rows = []
    try:
        for cursor in args.indices:
            if not 0 <= cursor < len(env._raw):
                raise ValueError("Probe cursor outside source")
            if args.sequential:
                while env.current_step < cursor:
                    if env.truncated:
                        env.reset()
                    result = env.step(0)
                    if env.status in ("DATA_GAP", "DATA_END") and env.current_step < cursor:
                        raise ValueError("Controlled probe hit a market gap")
                obs = env._get_obs()
            else:
                env.current_step = cursor
                obs = env._get_obs()
            a = obs["market_seq"]
            rows.append(dict(raw_cursor=cursor, visible_cutoff=env._raw.index[cursor].isoformat(),
                market_seq=a.tolist(), sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                latest_observed_close=float(env._raw.iloc[cursor].Close)))
        print(json.dumps(dict(source_contract=env.contract, feature_order=FEATURES_MARKET_OBS_OUTPUT,
                              observation_identity=env.observation_identity, observations=rows), allow_nan=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
