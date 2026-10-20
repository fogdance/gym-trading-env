# R7 state version contract

New Runs opt into READER_STREAM_STATE_V2 / PREFETCH_PENDING_STATE_V2. Corpus ABI remains 3.1.0. Public WP03 state_dict/load_state_dict and unwrapped diagnostic sources retain full snapshots.

## Local transaction

Reader marks are instance-local, stack ordered and never serialized. They capture RNG/counts plus only new coverage keys. Nested child commit merges undo into its parent. Group transactions cover all child Readers, scheduler, role/source statistics and last provenance. Source corruption remains fail-closed.

## Persistent state ownership

A lifecycle VersionedStream wraps the existing S3 source; it does not sample independently. One graph records composite small metadata and per-Reader exact new coverage deltas. A session plus stable source/index/role scope disambiguates every version. Version IDs do not consume sampler RNG.

The full graph also persists next_id, the monotonically allocated version counter. It must exceed every retained ID. After an explicit hot rewind and abandoned-branch collection, deriving it from retained nodes would reuse an already allocated ID; cold load therefore restores this counter exactly, including gaps. This is state metadata and does not change sampling or labels.

Coverage vectors use the original Reader traversal order. Parse validates the ordered static identity of every Reader leaf as well as the composite topology. Canonical JSON hashing ignores dictionary insertion order; accepting a reordered multi-source Reader map without this check could assign coverage to a different source while retaining the same canonical digest. Reordered maps are rejected before any Reader state mutation.

The Prefetch runtime member owns the complete graph (one materialized base and required immutable nodes), consumed ref and ordered pending batches/refs. course_reader.pkl uses S3_READER_CHECKPOINT_REF_V2 and contains both the source/latest ref and the complete graph_sha256. Both members must be validated together before Agent construction or mutation. Binding only the latest node digest is insufficient: compaction retains its historical digest while moving coverage into the base. The separate graph digest therefore binds the exact materialized base as well as every retained node. Ordinary standalone save_full is self-contained.

FullCheckpoint holds Prefetch.pause across both member saves. CheckpointSource exports the graph once and caches it for the immediately following runtime save; the cache is consumed once, and sampling, compaction and load invalidate it. The Checkpoint member contains only the envelope, while runtime contains the sole full graph. No external registry or prior Checkpoint is needed to restore it.

## Hot ownership and collection

Prefetch pins each delivered snapshot; moving its consumed position releases the previous ref. Latest, consumed and pending are graph roots; new version creation rejects an already active external Reader transaction. Common unreachable history is folded into the base; no pending duplicates the full coverage. Hot refs are process-local leases; durable callers use save_full. Each Reader has one exclusive VersionedStream owner. Memory is O(exact coverage + retained per-batch deltas + small states + pending arrays); extra caller-held leases retain their reachable history until released.

## Restore

Validate graph schema and full digest, node/link digests, cross-member envelope, topology/source/role, unique versions, exact new deltas, native pending array hashes, pending ordering, terminal source position, counters and journal first. Rebuild the session solely from persisted bytes, restore the source latest position, consumed version and exact pending arrays/seeds without regenerating batches. Identity and STRICT_PUBLISHED_LATEST_V2 checks remain unchanged. Missing dependencies fail before Agent restore. New Run identities include the Reader and Prefetch protocols plus the existing runtime source hashes; Corpus ABI 3.1.0 does not grant Run Checkpoint migration. Existing Runs retain their frozen code and original format.

## Release gates

The full graph must pass 0/1/2/4-pending restore, new-process cold restore and fault injection. Otherwise withdraw graph opt-in and retain the verified local transactions only. A versioned Prefetch must be unstarted and have an empty queue before load or load_full; reloading over restored pending batches would mix distinct sampling frontiers and is rejected before mutation. Legacy/non-S3 Prefetch remains on its original format. Large seeded historical coverage is DEV_ONLY pressure, separately reported from legal synthetic Corpus evidence.

## Common integration interface

The `bound` source is the existing CourseStream (F1/F2), ActorStreams (F3), MultiCorpusWindowAdapter or MultiCorpusF3Stream. Keep their sampling settings, source admission, loss and Agent.stream transform. They all expose readers, save_small, save/load/validate_snapshot and begin/commit/rollback_batch.

```python
from gym_trading_env.research.fg_s3_reader_versions import (
    VersionedStream, CheckpointSource, VersionedRuntimeState,
    validate_runtime_snapshot, STATE_PROTOCOL, PREFETCH_PROTOCOL)

source = VersionedStream(bound)
course_member = CheckpointSource(source)
# Resume: after strict bundle/identity selection, before creating the Agent:
# course_member.validate_snapshot(saved_course_member)
# validate_runtime_snapshot(source, saved_runtime, amount=expected_prefetch_amount)
# Then apply the existing pre-Agent Journal checks.

stream = agent.stream(source)  # Existing Agent seed transform and Prefetch.
# Build the original RuntimeState with this exact stream, schedule and carry.
runtime = VersionedRuntimeState.from_runtime(runtime, source)
# Register course_reader before runtime in the original FullCheckpoint.
checkpoint.course_reader = course_member
checkpoint.runtime = runtime
# Preserve its existing Agent/counters/Journal registrations, streams=[stream],
# backend='external_immutable_course_reader', strict_latest=True.
```

New Run identity must include `STATE_PROTOCOL` and `PREFETCH_PROTOCOL`, together with the existing implementation/config/source/verifier hashes. The formal Run constructors already do this and validate before Agent construction. Their `validate_checkpoint_stream` convenience helper uses their existing prefetch capacity of 1; a custom multi-source runtime uses `validate_runtime_snapshot(..., amount=its_capacity)` as above. FullCheckpoint still owns disk publication, member hashing and Journal validation. No training permission is implied by wrapping a source.

For diagnostics across independently started Runs, use `materialize_prefetch(source, snapshot)` to compare exact logical source/consumed/pending states despite distinct random session scopes. For a same-Run cold restore, compare the complete saved graph before continuation; do not normalize away its session, nodes or graph hash. This diagnostic expansion is intentionally outside the batch hot path.
