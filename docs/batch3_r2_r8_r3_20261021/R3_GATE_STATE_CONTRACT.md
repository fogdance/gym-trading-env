# R3 Gate state contract

Lifecycle/policy contract implemented and selected CPU contracts verified. This document is not
RL permission. Corpus ABI remains 3.1.0.

## Explicit boundary and finite work

A new controlled RL Run freezes `rl_admission` with `warmup_env_steps`,
`calibration_batches`, `calibration_horizon`, and `calibration_seed`.
The warmup prefix is a positive collect-aligned market-step boundary strictly
before the final market budget. Counts/seeds/horizon are explicit positive
integers (seed may be zero). SHADOW and ONLINE_SUPERVISED use no RL policy.
Reaching the boundary completes that cycle's BC before technical checks.
Technical failure or insufficient support is RL_BLOCKED with a safe non-final
checkpoint. A successful probe pauses at AWAIT_OPERATOR_REVIEW. No implicit
continuation as BC and no automatic weakening of the formal mask/WM policies.

## Durable state versus process authority

Durable lifecycle states: WARMUP, WM_CALIBRATING, RL_PROBING,
RL_BLOCKED, AWAIT_OPERATOR_REVIEW, RL_ACTIVE, FINAL_VERIFY, COMPLETE_VERIFIED.
Intermediate technical work is recoverable from its committed pre-probe anchor;
candidate weights/slots are never installed. Saved facts and externally reviewed
digests are persistent; process permission tokens are not checkpoint members.
Restoring RL_ACTIVE requires validating the external review again before any
further update. A saved `formal_admitted=true` is never executable permission.

The review covers the sealed technical anchor, source/verifier set, F3 lineage,
WM and mask identity, unchanged thresholds, exact Run identity, collected prefix,
and remaining frozen market/update budget. Ordinary authorized updates/counts
within that budget do not invalidate the review. Changed protected identities or
technical evidence require a new review; the same Run cannot replace its plan.

## Technical evidence

WM calibration executes the original native read-only function on certified,
causal TRAIN windows from an independent instance of the existing historical
Reader, without consuming the live stream. Unique owner-step support and the
original conservative mask thresholds remain mandatory. Teacher fields never
enter Actor observations. Full Agent state must remain unchanged.

The numeric candidate runs in a separate CPU interpreter from a byte-exact live
Agent snapshot and materialized authenticated native batch. It uses the existing
quality_actor/critic RL graph, finite gradients and legal masks. At least one
actual Actor and critic update is required; candidate parameters/slots are
discarded. The parent checks live Agent, RNG, normalizers, carry, Reader/Prefetch,
Replay, Publisher, account, counters and Gate state before accepting evidence.
External metrics dictionaries are not a probe API.

The candidate invokes the original optimizer at least once and at most twice.
The first invocation can update critic/slots before the existing optimizer
schedule changes Actor weights. No schedule or loss is modified to manufacture
an Actor change; failing to change both within the bounded probe closes the Gate.

## External review and completion

`online-rl-review-packet` exports only AWAIT_OPERATOR_REVIEW material.
The trainer cannot author APPROVED files. Resume requires external review bytes,
their externally supplied SHA256, --authorize-training and --authorize-rl.
Fresh native source certification and strict latest validation remain mandatory.

RL completion requires actual positive live RL updates and final full native
account reconciliation. Reports retain BC/RL counts, blockers, review scope,
and verified prefix separately. Old frozen Runs retain their old implementation;
new protocol checkpoints are not migrated in place.
