"""Repeatable standalone CPU CLI; creates no training Run and contacts no DB."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
import json
import os
from multiprocessing import get_context
from pathlib import Path
import sys
import time

from .core import MarketData
from .diagnostics import bucket_diagnostics, distribution, shadow_diagnostics
from .io import (git_identity, independent_output, sha256, verify_bundle, verify_collection,
                 write_bundle, write_collection_index, write_json)
from .schema import Decision, EDataConfig, SCHEMA
from .sources import pool_entries, read_fg_source


def _parser():
    p = argparse.ArgumentParser(description="S3-E Data V1: independent CPU market data")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("inspect", "build"):
        q = sub.add_parser(name)
        q.add_argument("--source", required=True, help="Existing eligible FG TRAIN pool JSON; read only")
        q.add_argument("--out", required=True, help="New independent output directory; never overwritten")
        q.add_argument("--contracts", nargs="*", default=[])
        q.add_argument("--stride", type=int, default=15, help="Snapshot every N raw decision rows; boundaries remain explicit")
        q.add_argument("--max-decisions", type=int, help="Optional diagnostic subset per contract")
        q.add_argument("--audit-1m", action="store_true")
        q.add_argument("--workers", type=int, help="CPU contract workers; default bounded by available RAM and CPUs")
        q.add_argument("--config", help="Optional EDataConfig JSON (volume_clip and oi_clip only)")
    q = sub.add_parser("verify")
    q.add_argument("--bundle", required=True)
    return p


def _config(args):
    values = json.loads(Path(args.config).read_text()) if args.config else {}
    if set(values) - {"volume_clip", "oi_clip"}:
        raise ValueError("Unknown E-data config fields")
    return EDataConfig(**{k: tuple(v) for k, v in values.items()}, audit_1m=args.audit_1m)


def _decisions(data, args):
    indices = list(range(0, len(data.frame), args.stride))
    # Include the terminal observed decision even if it misses the stride.
    if indices[-1] != len(data.frame)-1:
        indices.append(len(data.frame)-1)
    if args.max_decisions:
        indices = indices[:args.max_decisions]
    return [Decision(data.contract, data.frame.index[i]) for i in indices]


def _worker_budget(entries,args):
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:')))*1024
    cpus = len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    # Pilot peak was 207 MiB; dense JSON metadata and verification need more.
    per_worker = 768*1024*1024
    reserve = 1536*1024*1024
    cap = max(1,min(len(entries),cpus,max(1,(available-reserve)//per_worker)))
    workers = min(cap,args.workers) if args.workers is not None else cap
    return dict(workers=workers,available_bytes=available,logical_cpus=cpus,
                per_worker_budget_bytes=per_worker,reserve_bytes=reserve)


def _process_entry(entry,pool_record,args,config,out,inputs,repositories):
    started = time.monotonic()
    source = read_fg_source(entry, pool_record)
    data = MarketData(source.bars, source.schedule, config)
    summary, numeric = bucket_diagnostics(data)
    if summary["minute_rejections"]:
        write_json(out / (source.contract+"_failure.json"), summary)
        raise ValueError("Invalid source minutes: "+source.contract)
    decisions = _decisions(data, args)
    bundle = data.build(decisions)
    bundle.provenance.update(source.provenance, repositories=repositories,
                             snapshot_stride=args.stride, snapshot_limit=args.max_decisions)
    summary["shadow"] = shadow_diagnostics(bundle)
    summary["source_identity"] = source.provenance
    result = None
    if args.command == "build":
        destination = write_bundle(bundle, out / source.contract, inputs=inputs)
        validation = verify_bundle(destination)
        if validation["status"] != "PASS":
            raise ValueError("New bundle failed verification: " + json.dumps(validation))
        result = dict(path=str(destination), validation=validation)
    summary["seconds"] = round(time.monotonic()-started, 6)
    summary["raw_rows_per_second"] = len(data.frame)/summary["seconds"]
    return summary,numeric,result,bundle.schema["schema_sha256"]


def _run(args):
    if args.stride <= 0 or (args.max_decisions is not None and args.max_decisions <= 0) or (args.workers is not None and args.workers<=0):
        raise ValueError("Stride and decision limit must be positive")
    config = _config(args)
    entries, pool_record = pool_entries(args.source, args.contracts)
    inputs = [args.source] + [x[k] for x in entries for k in ("path", "env_config", "contract_config")]
    out = independent_output(args.out, inputs=inputs)
    out.mkdir(parents=True)
    project = Path(__file__).resolve().parents[1]
    repositories = {"gym": git_identity(project), "dreamer": git_identity(project.parent / "dreamerv3")}
    summaries, values, bundles = [], {"volume": [], "oi": []}, []
    budget = _worker_budget(entries,args)
    started = time.monotonic()
    schema_hashes = set()
    def collect(result):
        summary,numeric,bundle_record,schema_hash = result
        schema_hashes.add(schema_hash)
        summaries.append(summary)
        if bundle_record: bundles.append(bundle_record)
        for name in values:
            values[name].extend(numeric[name])
        print(json.dumps(dict(contract=summary['source_contract'], complete_buckets=summary["complete_buckets"],
                              decisions=summary['shadow']['decisions'], seconds=summary["seconds"])), flush=True)
    if budget['workers']==1:
        for entry in entries:
            collect(_process_entry(entry,pool_record,args,config,out,inputs,repositories))
    else:
        with ProcessPoolExecutor(max_workers=budget['workers'],mp_context=get_context('spawn')) as executor:
            tasks=[executor.submit(_process_entry,entry,pool_record,args,config,out,inputs,repositories) for entry in entries]
            for task in as_completed(tasks): collect(task.result())
    if len(schema_hashes)!=1:
        raise ValueError('Contract workers disagree on schema identity')
    summaries.sort(key=lambda x:x['source_contract'])
    bundles.sort(key=lambda x:x['path'])
    report = dict(schema=SCHEMA, source_pool=pool_record, repositories=repositories,
        dependencies={k: version(k) for k in ("numpy", "pandas", "PyYAML", "pytest")},
        contracts=summaries, total_source_rows=sum(x["rows"] for x in summaries),
        total_complete_buckets=sum(x["complete_buckets"] for x in summaries),
        total_feature_ready_buckets=sum(x["feature_ready_buckets"] for x in summaries),
        total_warmup_buckets=sum(x["warmup_buckets"] for x in summaries),
        total_rejected_buckets=sum(x["rejected_buckets"] for x in summaries),
        volume_logratio=distribution(values["volume"], config.volume_clip),
        oi_logratio=distribution(values["oi"], config.oi_clip), bundles=bundles,
        schema_sha256=next(iter(schema_hashes)),worker_budget=budget,total_seconds=time.monotonic()-started,
        clip_review="user authorized freezing these candidates after successful TRAIN verification; not optimality claim",
        numeric_status="PASS_TRAIN_NUMERICS", acceptance="REQUIRES_F01_F11_REPORT")
    if sha256(args.source) != pool_record["sha256"]:
        raise ValueError("TRAIN pool changed during command")
    write_json(out / "diagnostics.json", report)
    write_collection_index(out,report)
    print(json.dumps(dict(status=report["numeric_status"], report=str(out / "diagnostics.json"))))


def main(argv=None):
    # All CLI work is pure CPU. No GPU package is imported.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify":
            path = Path(args.bundle)
            if (path / "index_manifest.json").is_file():
                result = verify_collection(path)
            else:
                result = verify_bundle(path)
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result["status"] == "PASS" else 1
        _run(args)
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps(dict(status="FAIL", error=str(exc))), file=sys.stderr)
        return 1
