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

## Worker/root ownership round (candidate)

Clone-only query_clone_private_01: all 107 labels / 51 real traces / 39 strict
valid match. 316 copies; CPU 49.574775507s versus guard baseline 47.169257388s
(5.1% slower); worker wall 50.223025988s versus 51.298321150s. This intermediate
commit is NOT a claimed performance improvement: prefix freezing and remaining
full root/hash materialization still dominate. Continue with shared ownership
rather than recommend this candidate for production.

Workers now freeze their initial prefix once, own ordered append tails, restore
owned hot state through the same native checks, and authenticate full reconstructed
SHA against the Driver/reconciler. A new stream resets to the original immutable
version; incorrect delta base / full SHA reject. Teacher root/final clone export
uses private views. Public JSON snapshots remain unchanged; private prefix writes
explicitly reject. Source invariance compares immutable content commitments and
exact hot JSON bytes, not object identity/length. Account hashing still materializes
and verifies every prefix in this round, before the separate immutable-hash round.
worker_private_01: 25 targeted tests PASS; exact terminal receipts in ../artifacts.
Real query and end-to-end measurements of this worker round: NOT_RUN yet.

## Immutable exact-hash round (candidate)

Private views now stream exactly the original sorted JSON history block bytes,
including separators, signed zero, bool/int and nested JSON values. The public
SHA is unchanged and consumes all canonical bytes; a private commitment is not
used as a replacement identity. Memoization applies only to a frozen view. A new
hot state or append creates a distinct view and complete hash. Public mutable
histories retain the original full-prefix pickle comparison / fallback path.
A regression forbids both pickle and full-history materialization while hashing
a private native view, checks against the original complete public JSON SHA,
and then advances real financial state and repeats exact equivalence.
immutable_hash_01: targeted native/strict/ownership/guard tests pass; exact count
and executed test list in terminal XML/command receipt. Measurements pending.

## Guard traversal round (candidate)

Immutable-hash query_immutable_hash_01: complete 107/51/39 golden labels PASS,
19.345082460 query CPU seconds versus 47.169257388 (2.44x CPU reduction).
Account-hash CPU 1.018993196s versus 10.076966720s, 209 calls; zero full snapshot
copies during worker query. Worker wall remains 50.135444917s (spawn/source/Env
initialization included), so this micro result is NOT an end-to-end wall speedup.
shared_integration_01: 91 selected tests PASS; 8 actual optimizer-update cases
explicitly deselected, NOT_RUN. Includes F1/F2/F3 MultiCorpus/Reader, native Replay,
and mandatory Gate/SourceAdmission/action-mask shared files. Exact cases in XML.

The restored per-query file guard still re-walked identical directory ancestors
for every source file. FrozenSourceStats shares plain absolute ancestor checks
before/after every query, stats EVERY source file every query, and falls back to
the original strict resolve+stat for symlinks, relative paths or namespace changes.
Open/batch SHA and close guards remain. Directed symlink/same-inode, parent change
during pass, relative cwd, real file mutation and unrelated output cases pass.
guard_traversal_02: 30 tests PASS. First receipt guard_traversal_01 is retained:
an immediate same-size write had exactly equal stat timestamps; a direct runtime
reproduction showed equal before/after original 5-field stat tuples. The test now
forces a distinct mtime without sleep. Per-query metadata never claimed to replace
batch full SHA verification. Candidate real-query measurement pending.
