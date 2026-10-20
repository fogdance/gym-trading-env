# Strict external Reader recovery

New F1/F2/F3 Runs bind `STRICT_PUBLISHED_LATEST_V2`. Use the same prepared
snapshot, frozen execution code and external Verifier pin with the original
stage's resume entry. Every formal start/resume reruns actual Native G3.

Before mutable restoration, require: exact Run/source/index identity; a regular
`latest` file naming the highest canonical `step_N`; matching Manifest step,
backend and strict marker; complete `done`/member hashes; matching Reader
identity; the committed update-log byte prefix. Then restore the original
Agent/optimizer/Reader RNG/Prefetch/counters/schedule. Never select an older node
as a successful recovery from damage to the latest published node.

For new external strict checkpoints, files and the staged directory are fsynced
before node rename, followed by the node-parent directory; the new pointer and
its parent are fsynced at publication. Final metadata is staged outside the node
and fsynced before replacement. A failed pre-publication barrier preserves the
old pointer. A failure after pointer replacement is reported; recovery must
validate the actual pointer/node and may require orphan review. The native
Replay/F4 defaults keep their existing policy.

An uncommitted update-log tail may be archived under `recovery_logs/` only after
all validation passes. The committed prefix must match byte-for-byte. Missing or
changed committed log bytes cause a failure before Agent state restoration.

## Fault handling

- `.pending_UUID`: not published, never a candidate. Leave it in place. If an
  older published node is intact and no higher `step_N` exists, it is still the
  latest committed transaction; validated crash log tails are archived on load.
- `.pending_manifest_UUID`: an interrupted final/protected metadata update,
  staged outside the published node. Leave it in place; the old published
  Manifest remains valid until its atomic replacement completes. A normal
  authorized resume can restore that transaction and finalize it again.
- Complete `step_(N+1)` with `latest=N`: report
  `UNPUBLISHED_NODE_REQUIRES_REVIEW`. Preserve both nodes, pending directories,
  update logs and the pointer. Do not relabel, delete, republish or resume N.
- Missing `done`, damaged Manifest/member, absent/symlink/external/rollback
  pointer, backend/strict/identity mismatch: stop with evidence. Never write a
  replacement `done`, overwrite a same-named node or fall back.

For an orphan, an operator must first copy the entire Run and external index
bindings for investigation, record both node hashes and pointer bytes, validate
the independent source/index and both journal/counter boundaries, and review
the interruption point. This patch intentionally has no automatic repair
command. Only a separately reviewed, explicitly authorized recovery procedure
may publish or quarantine a node. A new Run/explicit transfer is a separate
operation, not an exact resume or a reason to regenerate the Corpus.

Legacy Replay defaults, F4's existing strict policy and previously frozen Runs
retain their historical behavior. New code does not migrate or re-sign them.
Fault injection covers process interruption at the tested write/rename/publish
boundaries. Physical power loss, storage-controller durability and real host
restart were not tested.
