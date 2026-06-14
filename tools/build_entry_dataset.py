from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gym_trading_env.research.entry_dataset import build_dataset_artifacts, json_default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/entry_eval_jm_v1.yaml")
    parser.add_argument("--output", default="artifacts/entry_eval/entry_eval_jm_v1")
    args = parser.parse_args()

    _, candidates, _, _, manifest = build_dataset_artifacts(args.config, args.output)
    print(json.dumps({
        "output": args.output,
        "candidates": int(len(candidates)),
        "manifest": {
            "version": manifest["version"],
            "rows": manifest["rows"],
            "outcomes": manifest["outcomes"],
            "feature_count": manifest["feature_count"],
        },
    }, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
