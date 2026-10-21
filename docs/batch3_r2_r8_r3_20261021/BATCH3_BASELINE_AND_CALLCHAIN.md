# Batch 3 baseline and call chain

Status: implementation in progress; no formal training authorization.

The actual clean integration baseline inspected on 2026-10-21 is Gym
`cadd6c87659665cc627e2275ab51ba9d14bb70cb` and Dreamer
`a6a6ad9e18196f799c4517f370dd26a4ca68aaf5`. Both contain Batch 1 and Batch 2.
Gym has additional Corpus scheduling changes after the archived `7adcf929`;
these are retained, including the current native verifier resource adapter.

The independent sibling worktrees use `feat/batch3-r2-r8-r3-20261021`.
Only the integration and this task's branches are implementation targets.

## Observed call chain and defects

`tools/fg_s3_v2_experiment.py` -> Gym `fg_s3_e_quality_online` ->
Dreamer `OnlineRun` / `OnlineGates`. F3 source inspection reuses
`inspect_f3` -> `inspect_stage` -> the R6 F1 freeze inspector.
Historical sources use the existing MultiCorpusF3Stream and WP03 admission.

* R2: the CLI rejects online Producer Pin flags (observed exit 2); `_history`
  and F3 native preflight omit log/pin/digest. Prepared lacks verifier binding.
* R8: `run_to` verifies only while COLLECTION_VALID is false; `result` infers
  completion from step/phase without final full-account reconciliation.
* R3: numeric readiness depends on an already executed live RL update;
  authorization is attempted before evidence exists; failed permission becomes
  BC and can still reach an ambiguous COMPLETE status.

## Existing checkpoint ownership

F4 uses native Replay and the original FullCheckpoint with strict latest:
replay, driver, collector, optional publisher/course_reader, agent, transaction
step, market_step, consumed_step, counters, gates, RuntimeState, logs, phase,
source_exposures. Its OnlineHistoryStream/RuntimeState are not R7 VersionedStream
or VersionedRuntimeState. This task retains those stream protocols.

## Boundaries

Keep Corpus ABI 3.1.0, all six behaviors, native Env/Reward/Teacher/Loss,
the existing Reader/Replay/Actor and strict checkpoint policy. Do not regenerate,
re-sign, edit, or launch production over formal Corpora or frozen Runs/Producers.
R5, throughput work and F1-F3 Reader migration are excluded.

R2/R8 can independently integrate after tests and review. R3 uses an explicit,
frozen BC warmup prefix, selected by the user during planning. Reaching that
boundary leads to technical checks and an auditable pause until externally
reviewed and separately authorized. Old Runs retain their frozen implementation.

Real G3, trained F3 integration, GPU and long formal training: NOT_RUN.
TEST: NOT_READ. Synthetic/stub evidence never grants formal authority.
