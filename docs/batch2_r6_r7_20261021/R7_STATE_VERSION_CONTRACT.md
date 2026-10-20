# R7 state version contract

New Runs opt into READER_STREAM_STATE_V2 / PREFETCH_PENDING_STATE_V2. Corpus ABI remains 3.1.0. Public WP03 state_dict/load_state_dict and unwrapped diagnostic sources retain full snapshots.

## Local transaction

Reader marks are instance-local, stack ordered and never serialized. They capture RNG/counts plus only new coverage keys. Nested child commit merges undo into its parent. Group transactions cover all child Readers, scheduler, role/source statistics and last provenance. Source corruption remains fail-closed.

## Persistent state ownership

A lifecycle VersionedStream wraps the existing S3 source; it does not sample independently. One graph records composite small metadata and per-Reader exact new coverage deltas. A session plus stable source/index/role scope disambiguates every version. Version IDs do not consume sampler RNG.

The Prefetch runtime member owns the complete graph (one materialized base and required immutable nodes), consumed ref and ordered pending batches/refs. course_reader.pkl contains a matching source/latest ref. Both members must be validated together before Agent construction or mutation. Ordinary standalone save_full is self-contained.

## Hot ownership and collection

Prefetch pins each delivered snapshot; moving its consumed position releases the previous ref. Latest, consumed, pending and active transactions are roots. Common unreachable history is folded into the base; no pending duplicates the full coverage. Hot refs are process-local leases; durable callers use save_full.

## Restore

Validate graph schema, base/delta digests, links, topology/source/role, unique versions, pending ordering, terminal source position, counters and journal first. Rebuild the session solely from persisted bytes, restore the source latest position, consumed version and exact pending arrays/seeds without regenerating batches. Identity and STRICT_PUBLISHED_LATEST_V2 checks remain unchanged. Missing dependencies fail before Agent restore.

## Release gates

The full graph must pass new-process 0/1/2/4-pending restore and fault injection. Otherwise withdraw graph opt-in and retain the verified local transactions only. Legacy/non-S3 Prefetch remains on its original format. Large seeded historical coverage is DEV_ONLY pressure, separately reported from legal synthetic Corpus evidence.
