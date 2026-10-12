# Shared architecture and S3-C1 execution plan

## Decisions and baseline

Approved plan: AR0–AR5 before C0–C4. The user explicitly permits direct
refactoring, does not require old API/run compatibility, and permits regenerating
old development products. Latest data decision: reuse the existing immutable
inputs through `research/fg-data`, rather than copy/recompute them.

Gym baseline: `d374e93e68c09d6bd8121177709960ef2ce497da`.
Dreamer baseline: `bb1d123fbb12e9390cee0282490a604f66be10f8`.
Both have no tracked dirty diff; existing untracked files are preserved.
The user reverted the temporary `.gitignore` edit. Backup `.bz2` files are ignored.

Development worktrees: `research/fg-s3-dev/{gym-trading-env,dreamerv3}`;
branch in both repositories: `feat/fg-s3-shared-s3c1`. These are separate Git
repositories, kept as siblings. Python imports must come from these worktrees.
Environment: `/home/v/miniconda3/envs/dreamerv3/bin/python`.

Data link (created by user):
`/home/v/Documents/work/research/fg-data` →
`/home/v/Documents/work/gym-trading-env/artifacts`.
It grants read-only use of signed TRAIN artifacts, not use of external source
code. Native data, Oracle, manifests and old runs are never outputs.
The local previous Gym artifacts remain another read-only source for labels and
CPU S3-A packages. Source aliases and canonical paths are recorded in receipts.

Planning evidence: 8 Mask/runtime tests and 1 real small neural Agent test passed.
The latter uses synthetic prices; it is not production verification. Full real
P4 input reconciliation: 488629 unique rows, 402096 eligible, ADD 16171, WAIT
385925; every sparse field matched. Exclusions: held 82251, no-trade 3795,
administrative-last 469, terminal 18. This is input evidence, not new Catalog
acceptance. GPU idle is user-confirmed; host process inspection was blocked by
the tool's sandbox approval policy.

## AR0–AR5

1. AR0: isolate worktrees, prove original repository protection and actual imports;
   record inputs, code/config/Checkpoint provenance and resources.
2. AR1: shared `FG_S3_TEACHER_CATALOG_V1`, all P4 records, authenticated Natural
   chunk ownership, exact bidirectional sparse-label reconciliation, bounded
   lookup, atomic publication, pure host adapter.
3. AR2: `FG_S3_SOURCE_ADMISSION_V1`, inspect/prepare/apply/audit, actual
   direct_init/direct_full/after_bc_s3a model transfer. Preserve complete Agent,
   original joint optimizer, Gate, norms and counters; never transfer BC optimizer.
4. AR3: Catalog/source reward, observation, TRAIN and dataset cross-contract;
   fail closed on unknown IDs, missing eligible labels and foreign sources.
5. AR4: CLI, raw receipts, docs, function commits and acceptance commits.
6. AR5: complete real Catalog audit plus real CPU migration and affected A/B
   regressions before beginning dependent C development.

Shared fields: `teacher_action: int32[B,T]`, `teacher_valid: bool[B,T]`.
No labels in observations or Replay files. Report fields are zero/False without
teacher lookups. Reuse existing exclusion/alignment functions; no duplicate rules.
A generic Agent training-loss factory returns the original loss by default;
the AR phase adds no BC, lambda or Gate algorithm change.

Source transfer creates a new experiment. Source and target code identities are
distinct; source provenance is audited, never resigned to pretend exact resume.
Exact keys/shapes/dtypes/finiteness and parameter/state digests prove the handoff.
`retain_base_joint_opt` keeps pre-BC Actor moments and documents that policy.
Stronger baseline optimizer equality is NOT_VERIFIED when baseline unavailable.
External Gate controller/window state is obtained from the hash-bound baseline;
the formal runner refuses missing necessary control state. Basic Agent handoff
and complete runner admission are separately audited. `after_bc_s3b` is unsupported.

## C0–C4

Consume the accepted shared APIs. Original networks, loss scales and A/B math
remain unchanged. S3-C loss is original loss plus lambda × GateReady × mean
valid P4 masked NLL. Use current posterior repfeat and stop_gradient only on
the added BC branch. Invalid rows get temporary safe mask/label before logp.
Lambda zero statically returns original loss; one joint optimizer/update only.

Stream: Replay → Consec → TeacherJoinStream → Agent Prefetch. Teacher fields
are ext_space only. Apply replay context once, align native 65→64 and both
consec branches. Wrapper preserves Consec.current and full child state.

Dedicated runner reuses FullCheckpoint/RuntimeState/scheduling; it does not
inherit offline_pretrain's permanently closed Gate policy. Preserve source Gate
and counters; local steps begin at zero with explicit source_update_offset.
Resume restores own model/Replay/prefetch/carry/queues/Gate/aggregates/RNG and
BC counters, without re-running source transfer. Identity binds source receipt,
Catalog manifest, code/packages/config and supervision protocol.

## Verification and delivery

Entry/config clarification (2026-10-12): extend the existing S3 manager
`tools/fg_s3_experiment.py` and its implementation, with an `s3c` dispatch.
`tools/fg_single3_experiment.py` belongs to the separate pretraining/online
workflow. Existing S3 experiment YAMLs are in `configs/experiments/fg_single3/`;
the directory name does not select a training algorithm. Add the C configuration
alongside these, keep distinct preparation/FullCheckpoint semantics, and reuse
the existing launcher rather than add per-experiment training scripts. Canonical
YAML parsing and current sibling-worktree paths must be corrected before
preparation. Full inventory, actual CLI evidence and pending fixes:
[entry/config review](FG_S3_Entry_and_Config_Review.md).

Required: full real Catalog reconciliation and negative cases; three real CPU
source modes; NLL/unimix/empty batch; Gate and gradient isolation; per-source
lambda-zero equivalence; action/stepid shift rejection; 65→64 and resets;
continuous N versus K+restore+(N-K), comparing every sampled ID, teacher array,
Agent/optimizer/norm/Gate/carry/queue/counter; real TRAIN CPU end-to-end in
float32 and BF16; minimal affected A/B regressions including B global mean.

Every stage preserves commands, stdout/stderr, XML/JSON and SHA. Reports in both
repositories: `FG_S3_Shared_Architecture_Implementation_and_Verification.md`
and `FG_S3C1_Implementation_and_Verification.md`. Each records functional SHA,
report commit, PASS/FAIL/NOT_RUN and evidence level. Final review checks both
taskbooks, public interfaces, source identities, math and recovery together.

No formal GPU training, DEV/TEST, training-order decision, or modification of
active runs. Tests never count static/synthetic results as production passes.
