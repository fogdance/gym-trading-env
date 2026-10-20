# R7 state version contract

New Runs opt into READER_STREAM_STATE_V2 / PREFETCH_PENDING_STATE_V2. Corpus ABI remains 3.1.0. Public WP03 state_dict/load_state_dict and unwrapped diagnostic sources retain full snapshots.

## Local transaction

Reader marks are instance-local, stack ordered and never serialized. They capture RNG/counts plus only new coverage keys. Nested child commit merges undo into its parent. Group transactions cover all child Readers, scheduler, role/source statistics and last provenance. Source corruption remains fail-closed.

## Persistent state ownership

A lifecycle VersionedStream wraps the existing S3 source; it does not sample independently. One graph records composite small metadata and per-Reader exact new coverage deltas. A session plus stable source/index/role scope disambiguates every version. Version IDs do not consume sampler RNG.

The Prefetch runtime member owns the complete graph (one materialized base and required immutable nodes), consumed ref and ordered pending batches/refs. course_reader.pkl uses S3_READER_CHECKPOINT_REF_V2 and contains both the source/latest ref and the complete graph_sha256. Both members must be validated together before Agent construction or mutation. Binding only the latest node digest is insufficient: compaction retains its historical digest while moving coverage into the base. The separate graph digest therefore binds the exact materialized base as well as every retained node. Ordinary standalone save_full is self-contained.

FullCheckpoint holds Prefetch.pause across both member saves. CheckpointSource exports the graph once and caches it for the immediately following runtime save; the cache is consumed once, and sampling, compaction and load invalidate it. The Checkpoint member contains only the envelope, while runtime contains the sole full graph. No external registry or prior Checkpoint is needed to restore it.

## Hot ownership and collection

Prefetch pins each delivered snapshot; moving its consumed position releases the previous ref. Latest, consumed and pending are graph roots; new version creation rejects an already active external Reader transaction. Common unreachable history is folded into the base; no pending duplicates the full coverage. Hot refs are process-local leases; durable callers use save_full. Each Reader has one exclusive VersionedStream owner. Memory is O(exact coverage + retained per-batch deltas + small states + pending arrays); extra caller-held leases retain their reachable history until released.

## Restore

Validate graph schema and full digest, node/link digests, cross-member envelope, topology/source/role, unique versions, exact new deltas, native pending array hashes, pending ordering, terminal source position, counters and journal first. Rebuild the session solely from persisted bytes, restore the source latest position, consumed version and exact pending arrays/seeds without regenerating batches. Identity and STRICT_PUBLISHED_LATEST_V2 checks remain unchanged. Missing dependencies fail before Agent restore. New Run identities include the Reader and Prefetch protocols plus the existing runtime source hashes; Corpus ABI 3.1.0 does not grant Run Checkpoint migration. Existing Runs retain their frozen code and original format.

## Release gates

The full graph must pass new-process 0/1/2/4-pending restore and fault injection. Otherwise withdraw graph opt-in and retain the verified local transactions only. Legacy/non-S3 Prefetch remains on its original format. Large seeded historical coverage is DEV_ONLY pressure, separately reported from legal synthetic Corpus evidence.
