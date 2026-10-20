# DATA-PERF Phase 2 implementation evidence

Status: IN_PROGRESS; no phase-2 integration merge authorized or performed.
Base Gym: e328cf73e0e284e506e549a35743458029e353a6.
Base/unchanged Dreamer: b7a693b3c7b0b45eeecdec954818be42d40e67b6.

## Audit round

Restored Producer source_guard_interval=1. SHA-on-open, per-query stat identity,
source/profile/rules/mask checks and close guard remain enabled.
Native admission now matches all six PASS receipts, nonzero record counts,
verified-label counts and replay hashes to the original trace manifests.
On unchanged integration runtime the four synthetic attacks (missing receipts,
partial label count, failed behavior and foreign trace) were accepted: guard_before_01,
1 pass / 4 failures, expected regression evidence. No actual DATA admission granted.
After the fix guard_after_02 passes; exact command/XML/log/SHA receipts live in
../artifacts/guard_after_02.*. An earlier wrapper test supplied a paused account;
that illegal test setup was corrected by normal chunk continuation, with failed
receipt guard_after_01 retained. Public trade semantics were not changed.

Independent golden tests extract only the scalar _value and Observed wrapper
from fixed Git b0d5ef9af07fbdd0c553dd943ac9eb40c9a84d46. They compare exact rational
values over 365 controlled native rows, including a session closing non-minute
marker, release boundaries and two official days, and compare direct E execution
with the old wrapper after every real financial step/chunk continuation.
Existing bounded MACD/checkpoint/causal late-arrival and schema negatives run in
the same selected test set. Test pass counts are recorded in the terminal receipt,
not a claim about the whole project.

GPU / optimizer updates / formal F1 / 18-contract production: NOT_RUN.
## Clone ownership round (candidate)

Implemented private immutable 256-row blocks plus an owned append tail, with
owner/source/cursor/content commitments. Public JSON export/import and original
sorted-JSON SHA remain unchanged. Private restore calls the same complete
financial, ledger, clock and E-observation restore checks. Source full-snapshot
invariance and clone final full export/hash are retained in this first round.
clone_private_01: 24 targeted tests pass (terminal receipt carries exact count).
Covers readonly past rows, nested ownership, signed zero/bool differences,
changed/deleted/replaced/rewound public histories, cross-owner/source rejection,
Ledger/Reward/RNG/Observation and subsequent financial actions.
Guard-restored baseline query_audit_baseline_01: 107 queries, 51 native traces,
39 strict-valid, 47.169257388 CPU seconds, 367 snapshot copies / 17.772118127s;
51.298321150s worker wall, 128.059366562s including real account reconstruction.
Function times are inclusive; source/profile/clock guards remain on.
Candidate real-query and final end-to-end certification: NOT_RUN yet.
