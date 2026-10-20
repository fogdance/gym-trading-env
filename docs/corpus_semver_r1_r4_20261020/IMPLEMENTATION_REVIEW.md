# FG-S3 Corpus SemVer + R1/R4 implementation review

## Summary

**CODE_GO for this implementation scope.** A0→A1→B1→B2→C1→C2→Z1 is
implemented on independent sibling worktrees, branch
`feat/corpus-semver-r1-r4-20261020`. Nothing was merged to integration/s3-next.

The final frozen, actually tested pair is:

| Repository | Commit |
|---|---|
| Gym | `83bc3018de9598969d9f06df39f48da57f8dbb5c` |
| Dreamer | `4c4a36636bfa1a3f6a02c6df7f7d164ca8ec7036` |

Gym's following documentation/evidence commit is the delivery HEAD; its runtime
bytes are identical to the tested pair, as checked in
`Z1_source_invariants_v2.json`. `COMMITS.json` records the full A0–Z1 chain.
The actual starting integration pair was Gym `8619f04b6f996f4c4d7f169486e4d10c25a3b2b9`
and Dreamer `b7a693b3c7b0b45eeecdec954818be42d40e67b6`.

**DATA_GO_BLOCKED / real acceptance NOT_RUN:** the active TRAIN01 final Corpus
Manifest is not published, and this task did not execute production Native G3
or formal real-model prepare/start/resume. This is separate from code acceptance.

## Evidence observed

- Two original completed TRAIN V3 assets passed the independent existing Reader
  on the final pair: 10 views / 10 exact next-batch restores each, requested-window
  support PASS, zero optimizer exposures. Their Corpus identities remain
  `dffe36e7b687e1d55ccf28e52caae02f19bb43075274b0b295f27eff94e90302`
  and `96828a02f69ed30e2d74ff0637c1138bfc709d89e7a2c71611839fb262c3c43c`.
- All **176 declared immutable files**, including original manifests, have equal
  before/after SHA256. Previously closed production behavior manifests and the
  protected frozen Producer pair also remain unchanged. No original asset was
  edited, re-signed, migrated or regenerated.
- The two E-profile full hashes differ, but their original schema.json semantic
  descriptors compare exactly equal and reproduce their recorded hashes.
  Unmodified integration rejects phase2 with E-profile identity mismatch;
  the new explicit mapping reads it. Phase3's existing source identity remains
  `42bade0da910cbf164a5dfa8e05b247ff4a58f7fd43d720b95384566b38adb52`.
- Final six-suite result: **268 distinct tests / 298 executions; 0 failures,
  errors or skips**. F1/F2/F3 actual CPU JAX cold tests compare complete structured
  state and next batches with uninterrupted runs, including F3 skip_first.

## Root cause

1. E-profile provenance mixed implementation-source hashes with the semantic
   contract. Comparing the claim only with the consumer's current full schema
   hash prevented reuse of an otherwise reviewed phase2 asset.
2. Low-level Admission already supported an external Producer pin, but F1/F2/F3
   APIs, read-checks and CLI omitted it, leaving formal verification on the old
   default pair.
3. Offline probes and runtime Checkpoints used non-strict latest selection.
   Filtering missing-done nodes allowed an older checkpoint to be selected.
   Strict API also permitted explicit old-path load, old-node republish and did
   not validate backend/node-step during probe.
4. Offline logs were reconciled after mutable restoration. Final protection also
   staged temporary metadata inside a valid node; interruption invalidated its
   file inventory. Both failures were reproduced in isolated synthetic tests.

## Changes made

- Added one read-only `CorpusCompatibility` registry for exact ABI **3.1.0**.
  Existing reviewed V3 assets resolve externally, with their original bytes and
  content identity retained. Unknown versions/contracts/encodings fail closed.
  Future Producer manifests declare the version explicitly. Internal Course,
  Model, Teacher and replay protocol version axes remain separate.
- Propagated `producer_lock` and `producer_lock_sha256` across all requested
  F1/F2/F3 Python and CLI routes. Native stage paths require explicit complete
  pins. Prepared and Run bind the actual verifier pair, pin digest, source and
  compatibility; another valid pin cannot replace the Prepared pin.
- Retained the actual frozen G3 subprocess, before/after source audit, complete
  six-behavior label checks, in-process proof and separate training authorization.
  Ephemeral log names/timings do not alter stable resume identity.
- Enabled `STRICT_PUBLISHED_LATEST_V2` at every offline probe/runtime construction,
  including F3's shared HeadsRun path. Restores validate pointer, bundle, source,
  Reader/index and committed journal prefix/chronology before Agent restoration.
  All original Agent/optimizer/Reader/RNG/Prefetch/counter/schedule state remains.
- Closed explicit-path and save-backward bypasses. Pending work is retained;
  orphan nodes require review. Final metadata stages outside published nodes.
  New external strict publication uses file/directory durability barriers.
  Legacy Replay and F4 default policy are unchanged.

## Verification performed

| Final suite | Passed | Exit |
|---|---:|---:|
| F1 checkpoint + identity cold regression | 22 | 0 |
| F2 checkpoint cold regression | 8 | 0 |
| F3 checkpoint, skip and rejection regression | 11 | 0 |
| Three mandatory shared regression files | 20 | 0 |
| Pin/Run/Checkpoint fault contracts | 67 | 0 |
| Compatibility, WP03, DATA-MULTI and causal MACD | 170 | 0 |

Counts overlap by 30 pin tests; the distinct total is 268. Exact commands, cwd,
environment, timings and exit codes are in `FINAL_COMMANDS.json`; all collected
cases and test-file hashes are in `FINAL_TEST_INDEX.json/tsv`. See TEST_REPORT.md
for commands and retained failed attempts with their diagnosis and fixes.
Cold worker receipts and checkpoint evidence are under `cold_evidence_v2/`.
Actual independent Reader commands/reports are under
`Z1_v2_real_reader_commands.json` and `real_reader_v2_phase2/phase3/`.

Self-review checked the complete paired source diffs. **61 Gym and 27 Dreamer
critical math/source files are byte-identical to baseline**, including native
financial Env/reward, L1/L2/witness, Reader/MultiCorpus and model/Loss code.
The final source patches are `CODE_DIFF_GYM.patch` / `CODE_DIFF_DREAMER.patch`.
All changed Python files parsed successfully. Original failed receipts are
retained rather than overwritten or relabeled PASS.

## Risks / not verified

- Production Native G3 and formal real-model prepare/start/resume: **NOT_RUN**.
  Synthetic routing stubs only verify wiring/identity and are not certification.
  GPU and real TRAIN optimizer updates: **0**. TEST market reads: **NOT_RUN**.
- Active production's final Manifest/guard verification remains pending. Closed
  behavior manifests observed interval 1; the final field must still be checked.
  Any interval other than 1 must fail admission, not be edited or bypassed.
- Only the exact reviewed profiles/contracts are supported. A newly reviewed
  nonsemantic profile alias can be registered without changing or regenerating
  the asset. Arbitrary V3 or unknown SemVer is not accepted.
- Fault injection exercised write/rename/publish and durability failures; no
  physical power-off, host restart or storage-controller fault was performed.
  Additional validation/fsync adds Checkpoint I/O; long-training throughput was
  not benchmarked. Every formal start/resume still reruns G3.
- Full formal F4 and remaining R2/R3/R5–R8 workflows are outside this patch.
  Previously frozen Runs require their original code; this patch does not grant
  same-Run migration across code/recovery identities.

## Next recommended step

After an operator reviews the actual published TRAIN Manifest and explicitly
authorizes a bounded real G3 scope/CPU budget, use INTERFACE.md with the reviewed
frozen Verifier pair and external pin digest. F1/F2/F3 integration requires the
two new pin arguments in existing calls/commands; config schemas, Reader,
training objectives and losses need no reconstruction. Formal training itself
still requires separate user authorization. Merge remains a separate user action.
