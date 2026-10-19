"""Independent immutable bundles, source identities and read-only verification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .schema import EDataBundle, LEGACY_FIELDS, SHARED_FIELDS, SCHEMA, identity, json_bytes, local_timestamp


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stable_identity(path):
    from .reference import legacy_helper
    source = Path(path)
    path = source.resolve(strict=True)
    helper = legacy_helper("entry_market_cache")
    digest, stat = helper._file_identity(path)
    if source.resolve(strict=True)!=path or helper._stat_identity(path)!=stat:
        raise ValueError("Source changed while reading stable identity: " + str(path))
    return dict(path=str(path), bytes=stat[2], sha256=digest)


def git_identity(root):
    root = Path(root).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    return dict(path=str(root), sha=git("rev-parse", "HEAD"),
                branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(git("status", "--porcelain")))


def independent_output(path, *, inputs=(), allow_existing=False):
    """Reject protected/source paths. Existing owned sessions opt in explicitly.

    New data production retains the default no-overwrite requirement. The
    caller of an existing-session read/resume also validates its own identity.
    """
    path = Path(path).expanduser()
    resolved = path.resolve()
    parts = resolved.parts
    forbidden = ("fg-s3-b2-full-train", "fg-s3-tutor-mm-train", "fg-s3-tutor-800k-freeze",
                 "fg-s3-tutor-full-train", "fg-s3-archive-dev", "fg-s3-wandb-dev", "fg-s3-next-integration")
    if any(x in parts for x in forbidden) or str(resolved).startswith("/data/logdir/fg_single3/"):
        raise ValueError("Output must be independent of formal training/archive paths")
    if path.is_symlink():
        raise ValueError("Symlink output destination refused")
    for source in inputs:
        source = Path(source).resolve()
        if resolved == source or source.is_relative_to(resolved):
            raise ValueError("Output directory contains a source file")
    if path.exists() and not (allow_existing and path.is_dir()):
        raise FileExistsError("Output already exists; choose a new directory: " + str(path))
    return resolved


def write_json(path, value):
    Path(path).write_bytes(json_bytes(value) + b"\n")


def write_bundle(bundle: EDataBundle, path, *, inputs=()):
    path = independent_output(path, inputs=inputs)
    path.mkdir(parents=True)
    arrays = dict(context_15m=bundle.context_15m, market_seq=bundle.market_seq,
                  market_seq_e_shared=bundle.market_seq_e_shared, price_anchor=bundle.price_anchor)
    if bundle.audit_1m_ohlc_shared is not None:
        arrays["audit_1m_ohlc_shared"] = bundle.audit_1m_ohlc_shared
    for name, array in arrays.items():
        np.save(path / (name + ".npy"), array, allow_pickle=False)
    for name in ("bucket_manifest", "decision_meta", "schema", "provenance", "events"):
        write_json(path / (name + ".json"), getattr(bundle, name))
    files = {p.name: dict(bytes=p.stat().st_size, sha256=sha256(p))
             for p in sorted(path.iterdir())}
    manifest = dict(schema=SCHEMA, schema_sha256=bundle.schema["schema_sha256"],
                    decisions=len(bundle.decision_meta), files=files)
    manifest["manifest_sha256"] = identity(manifest)
    write_json(path / "manifest.json", manifest)
    return path


def verify_bundle(path):
    """Verify bytes and public data contracts, without changing the bundle."""
    path = Path(path).resolve(strict=True)
    manifest = json.loads((path / "manifest.json").read_text())
    checks_count, errors = 0, []
    def check(condition, message):
        nonlocal checks_count
        checks_count += 1
        if not condition:
            errors.append(message)
    base = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    check(identity(base) == manifest.get("manifest_sha256"), "manifest identity")
    check(manifest.get("schema") == SCHEMA, "schema version")
    check(set(p.name for p in path.iterdir()) == set(manifest["files"]) | {"manifest.json"}, "complete file inventory")
    for name, record in manifest["files"].items():
        if name in ("", ".", "..") or Path(name).name != name or (path / name).is_symlink():
            raise ValueError("Unsafe bundle member")
        member = path / name
        check(member.is_file() and member.stat().st_size == record["bytes"] and sha256(member) == record["sha256"], "SHA256 " + name)
    if errors:
        return dict(status="FAIL", checks_count=checks_count, errors=errors)
    schema = json.loads((path / "schema.json").read_text())
    check(identity({k: v for k, v in schema.items() if k != "schema_sha256"}) == schema["schema_sha256"] == manifest["schema_sha256"], "schema hash")
    n = manifest["decisions"]
    arrays = {}
    expected = {"context_15m": (n, 24, 7), "market_seq": (n, 60, 18), "market_seq_e_shared": (n, 60, 18), "price_anchor": (n,)}
    if "audit_1m_ohlc_shared.npy" in manifest["files"]:
        expected["audit_1m_ohlc_shared"] = (n, 60, 4)
    for name, shape in expected.items():
        arrays[name] = a = np.load(path / (name + ".npy"), allow_pickle=False, mmap_mode="r")
        check(a.shape == shape, "shape " + name)
        check(a.dtype == (np.float64 if name == "price_anchor" else np.float32), "dtype " + name)
        check(bool(np.isfinite(a).all()), "finite " + name)
    if errors:
        return dict(status="FAIL", checks_count=checks_count, errors=errors)
    context = arrays["context_15m"]
    valid = context[:, :, 6]
    check(bool(np.isin(valid, [0, 1]).all()), "binary context mask")
    check(bool((context[valid == 0] == 0).all()), "invalid context rows all zero")
    check(bool((arrays["price_anchor"] > 0).all()), "positive anchor")
    unchanged = [i for i, name in enumerate(LEGACY_FIELDS) if name not in SHARED_FIELDS]
    check(bool(np.array_equal(arrays["market_seq"][:, :, unchanged], arrays["market_seq_e_shared"][:, :, unchanged])), "12 anchor-independent channels unchanged")
    meta = json.loads((path / "decision_meta.json").read_text())
    buckets = {x["bucket_id"]: x for x in json.loads((path / "bucket_manifest.json").read_text())}
    # Parse immutable bucket times once, as the 1m path prepares its arrays once.
    bucket_times = {key: (local_timestamp(x["bucket_end"]).value,
                          local_timestamp(x["available_at"]).value) for key, x in buckets.items()}
    raw_feature_values = {}
    check(len(meta) == n, "decision metadata count")
    for i, item in enumerate(meta):
        cutoff = local_timestamp(item["visible_cutoff"])
        cutoff_ns = cutoff.value
        check(item["price_anchor"] == float(arrays["price_anchor"][i]), "anchor metadata " + str(i))
        check(local_timestamp(item["anchor_source_time"]) <= cutoff and local_timestamp(item["anchor_available_at"]) <= cutoff, "causal anchor " + str(i))
        check(hashlib.sha256(arrays["market_seq"][i].tobytes()).hexdigest() == item["minute_window_sha256"], "legacy observation hash " + str(i))
        ids = item["context_bucket_ids"]
        check(len(ids) <= 24 and len(ids) + item["context_padding"] == 24, "context window size " + str(i))
        check(bool((context[i, :item["context_padding"]] == 0).all()), "left padding " + str(i))
        check(item["context_bucket_ends"] == sorted(item["context_bucket_ends"]), "context order " + str(i))
        for offset, bucket_id in enumerate(ids, item["context_padding"]):
            bucket = buckets.get(bucket_id)
            point = item["context_feature_provenance"][offset-item["context_padding"]]
            check(point["bucket_id"] == bucket_id, "point feature identity " + bucket_id)
            check(bucket is not None and bucket["complete"] and bucket["source_contract"] == item["source_contract"], "bucket identity " + bucket_id)
            if bucket:
                end_ns, available_ns = bucket_times[bucket_id]
                check(end_ns <= cutoff_ns and available_ns <= cutoff_ns, "causal bucket " + bucket_id)
                check(float(valid[i, offset]) == float(point["feature_ready"]), "warmup mask " + bucket_id)
                for prior in point["volume_baseline_bucket_ids"]:
                    p = buckets.get(prior)
                    check(p is not None and p["complete"] and p["source_contract"] == item["source_contract"]
                          and bucket_times[prior][0] < end_ns
                          and bucket_times[prior][1] <= cutoff_ns, "causal volume baseline " + prior)
                if point["feature_ready"]:
                    expected_price = np.log(np.array([bucket[k] for k in ("Open", "High", "Low", "Close")]) / item["price_anchor"]).astype(np.float32)
                    check(bool(np.array_equal(context[i, offset, :4], expected_price)), "common price anchor " + bucket_id)
                    baseline = [buckets[x]["Volume"] for x in point["volume_baseline_bucket_ids"]]
                    previous = buckets.get(point["previous_OI_bucket_id"])
                    check(len(baseline) == 20 and previous is not None
                          and bucket_times[previous["bucket_id"]][0] < end_ns
                          and bucket_times[previous["bucket_id"]][1] <= cutoff_ns, "volume/OI prior identity " + bucket_id)
                    if previous is None:
                        continue
                    feature_key = (bucket_id, tuple(point["volume_baseline_bucket_ids"]), previous["bucket_id"])
                    if feature_key not in raw_feature_values:
                        raw_feature_values[feature_key] = (
                            float(np.log((bucket["Volume"]+1)/(np.mean(baseline)+1))),
                            float(np.log(bucket["OpenInterest"]/previous["OpenInterest"])))
                    raw_volume, raw_oi = raw_feature_values[feature_key]
                    check(raw_volume == point["vol_relative_raw"] and raw_oi == point["oi_change_raw"], "volume/OI formula " + bucket_id)
                    expected_volume = np.float32(np.clip(raw_volume, *schema["volume_clip"]))
                    expected_oi = np.float32(np.clip(raw_oi, *schema["oi_clip"]))
                    check(context[i, offset, 4] == expected_volume and context[i, offset, 5] == expected_oi, "volume/OI clip " + bucket_id)
    return dict(status="PASS" if not errors else "FAIL", checked_files=len(manifest["files"]),
                schema_sha256=schema["schema_sha256"], manifest_sha256=manifest["manifest_sha256"],
                checks_count=checks_count, errors=errors)


def write_collection_index(path,report):
    path=Path(path)
    value=dict(schema=SCHEMA,kind='collection',schema_sha256=report['schema_sha256'],
        source_pool_sha256=report['source_pool']['sha256'],
        diagnostics_sha256=sha256(path/'diagnostics.json'),
        bundles={Path(x['path']).name:sha256(Path(x['path'])/'manifest.json') for x in report['bundles']})
    write_json(path/'index_manifest.json',{**value,'index_sha256':identity(value)})


def verify_collection(path):
    path=Path(path).resolve(strict=True)
    index=json.loads((path/'index_manifest.json').read_text())
    if identity({k:v for k,v in index.items() if k!='index_sha256'})!=index['index_sha256']:
        return dict(status='FAIL',errors=['collection index identity'])
    if sha256(path/'diagnostics.json')!=index['diagnostics_sha256']:
        return dict(status='FAIL',errors=['diagnostics SHA256'])
    report=json.loads((path/'diagnostics.json').read_text())
    if report['source_pool']['sha256']!=index['source_pool_sha256'] or report['schema_sha256']!=index['schema_sha256']:
        return dict(status='FAIL',errors=['collection source/schema identity'])
    results=[]
    for name,digest in index['bundles'].items():
        if name in ('', '.', '..') or Path(name).name!=name or (path/name).is_symlink():
            raise ValueError('Unsafe collection member')
        if sha256(path/name/'manifest.json')!=digest:
            return dict(status='FAIL',errors=['bundle manifest SHA256 '+name])
        result=verify_bundle(path/name)
        if result.get('schema_sha256')!=index['schema_sha256']:
            result=dict(status='FAIL',errors=['bundle schema does not match collection'])
        results.append(dict(source_contract=name,**result))
    return dict(status='PASS' if results and all(x['status']=='PASS' for x in results) else 'FAIL',
                index_sha256=index['index_sha256'],bundles=results)
