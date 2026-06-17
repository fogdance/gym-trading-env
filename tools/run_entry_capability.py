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

from gym_trading_env.research.entry_pipeline import (  # noqa: E402
    json_default,
    run_entry_capability_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/entry_eval_jm_v1.yaml")
    parser.add_argument("--output", default="artifacts/entry_eval/entry_eval_jm_v1")
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Build candidates/features/outcomes/split/manifest and stop before baselines/models.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing dataset artifacts if config/data hashes match.",
    )
    parser.add_argument(
        "--force-stage",
        choices=("none", "dataset", "models", "all"),
        default="none",
        help="Force rebuilding a pipeline stage. 'models' reuses dataset and reruns downstream stages.",
    )
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()

    result = run_entry_capability_pipeline(
        config_path=args.config,
        output=args.output,
        dataset_only=bool(args.dataset_only),
        skip_sensitivity=bool(args.skip_sensitivity),
        resume=bool(args.resume),
        force_stage=str(args.force_stage),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
