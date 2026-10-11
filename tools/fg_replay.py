#!/usr/bin/env python3
"""Construct/reconcile/audit CPU FG Replay; never creates a Dreamer Agent."""
import argparse
import os
from pathlib import Path
import sys

# Embodied imports JAX. Set these before importing project/Embodied modules.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gym_trading_env.research.fg_replay import (
    generate_dataset, reconcile_trace, validate_dataset, write_json, signed, digest)
from gym_trading_env.research.fg_replay_coverage import audit_dataset


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dreamer-root", default="/home/v/Documents/work/dreamerv3")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--config", default="configs/fg_entry_v1/replay_r0_r2.yaml")
    generate.add_argument("--output", required=True)
    generate.add_argument("--workers", type=int, default=1)
    generate.add_argument("--resume", action="store_true")
    generate.add_argument("--contracts", nargs="+", help="Explicit train subset for engineering verification")
    reconcile = sub.add_parser("reconcile")
    reconcile.add_argument("--dataset")
    reconcile.add_argument("--trace")
    reconcile.add_argument("--output", required=True, help="New verification directory")
    audit = sub.add_parser("audit")
    audit.add_argument("--dataset", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--batches", type=int, default=1000)
    audit.add_argument("--batch-size", type=int, default=16)
    audit.add_argument("--seed", type=int, default=202)
    args = parser.parse_args()
    if args.command == "generate":
        result = generate_dataset(args.config, args.output, args.dreamer_root,
            workers=args.workers, resume=args.resume, contracts=args.contracts)
        print(f"Complete: {len(result['traces'])} reconciled traces: {args.output}")
    elif args.command == "audit":
        audit_dataset(args.dataset, args.output, args.dreamer_root,
                      batches=args.batches, batch_size=args.batch_size, seed=args.seed)
        print(Path(args.output) / "replay_coverage.md")
    else:
        if bool(args.dataset) == bool(args.trace):
            parser.error("reconcile requires exactly one of --dataset or --trace")
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        paths = ([Path(args.trace)] if args.trace else
            [Path(args.dataset) / r["directory"] for r in validate_dataset(args.dataset)["traces"]])
        results = []
        for path in paths:
            result = reconcile_trace(path, args.dreamer_root)
            results.append(dict(trace=str(path), **result))
            print(f"PASS {path}: {result['records']} rows", flush=True)
        report = "# Replay 逐步对账\n\n" + "\n".join(
            f"- PASS `{r['trace']}`：{r['records']} 行，{r['account_restores']} 次账户恢复。" for r in results) + "\n"
        (output / "reconciliation.md").write_text(report, encoding="utf-8")
        write_json(output / "manifest.json", signed(dict(results=results,
            report_sha256=digest(output / "reconciliation.md"))))


if __name__ == "__main__":
    main()
