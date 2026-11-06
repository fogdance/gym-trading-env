"""JSON command-line adapter: python -m market_structure bars.json."""

import argparse
import json
from pathlib import Path
import sys

from . import analyze_market_structure


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify OHLC swings and close-break trends; output JSON.")
    parser.add_argument("input", nargs="?", default="-", help="JSON array or {bars: [...]}; '-' reads stdin")
    parser.add_argument("--left", type=int, default=2)
    parser.add_argument("--right", type=int, default=2)
    parser.add_argument("--plateau-policy", choices=("last", "strict"), default="last")
    parser.add_argument("--ambiguous-policy", choices=("skip", "raise"), default="skip")
    parser.add_argument("--structure-tolerance-pct", type=float, default=0.0)
    parser.add_argument("--break-buffer-pct", type=float, default=0.0)
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin) if args.input == "-" else json.loads(Path(args.input).read_text(encoding="utf-8"))
        bars = payload.get("bars") if isinstance(payload, dict) else payload
        if not isinstance(bars, list):
            raise ValueError("Input must be an array of OHLC records or an object containing a bars array")
        result = analyze_market_structure(
            bars, left=args.left, right=args.right,
            plateau_policy=args.plateau_policy, ambiguous_policy=args.ambiguous_policy,
            structure_tolerance_pct=args.structure_tolerance_pct,
            break_buffer_pct=args.break_buffer_pct,
        )
    except (TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    json.dump(result, sys.stdout, ensure_ascii=False, allow_nan=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
