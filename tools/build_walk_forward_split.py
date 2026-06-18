#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gym_trading_env.research.walk_forward_split_builder import (
    BuilderPaths,
    SplitDates,
    build_walk_forward_split,
    day_int,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical walk-forward data/config/split assets."
    )
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--raw-csv", required=True, type=Path)
    parser.add_argument("--env-template", required=True, type=Path)
    parser.add_argument("--entry-eval-template", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        default=None,
        type=Path,
        help=(
            "Experiment contract root. Defaults to "
            "/data/logdir/trading_contracts/<split-name>."
        ),
    )
    parser.add_argument("--raw-output-name", required=True)
    parser.add_argument("--train-output-name", required=True)
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    if output_root is None:
        output_root = Path("/data/logdir/trading_contracts") / args.split_name
    paths = BuilderPaths(
        raw_csv=args.raw_csv,
        env_template=args.env_template,
        entry_eval_template=args.entry_eval_template,
        output_root=output_root,
        split_name=args.split_name,
        raw_output_name=args.raw_output_name,
        train_output_name=args.train_output_name,
    )
    dates = SplitDates(
        train_start=day_int(args.train_start),
        train_end=day_int(args.train_end),
        validation_start=day_int(args.validation_start),
        validation_end=day_int(args.validation_end),
        test_start=day_int(args.test_start),
        test_end=day_int(args.test_end),
    )
    summary = build_walk_forward_split(paths=paths, dates=dates)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
