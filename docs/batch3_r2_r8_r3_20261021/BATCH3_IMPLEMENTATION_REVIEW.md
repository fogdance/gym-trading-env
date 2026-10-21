# Batch3 R2 / R8 / R3 implementation review

Engineering acceptance only. Corpus ABI **3.1.0** is unchanged. No production
Corpus regeneration, formal long training, GPU execution or TEST market read.
The final paired integration HEADs are recorded in the workspace
`baseline-lock.json`, outside both repositories to avoid circular SHA updates.

## Baseline and delivery

Actual starting integration HEADs, checked before creating the independent pair:

| Repository | Actual baseline |
| --- | --- |
| Gym | `cadd6c87659665cc627e2275ab51ba9d14bb70cb` |
| Dreamer | `a6a6ad9e18196f799c4517f370dd26a4ca68aaf5` |

Both contain Batch1 and Batch2. Gym had advanced beyond the archived `7adcf929`;
that archive was not used as the development baseline. Feature branch in both
repositories: `feat/batch3-r2-r8-r3-20261021`. Worktrees are under
`fg-s3-batch3-r2-r8-r3-20261021/`. R2 and R8 were separately merged before R3.
No unrelated branch or active development worktree was modified.

## Observations, causes and changes

Baseline failures are retained in
[B3-00 evidence](receipts/B3_00_BASELINE_NEGATIVES.json), with the actual call
chain in [baseline review](BATCH3_BASELINE_AND_CALLCHAIN.md).

| Item | Confirmed baseline behavior / cause | Implemented change |
| --- | --- | --- |
| R2 | Online CLI rejected Producer Pin arguments; native source calls omitted required pin/log inputs. | Existing unified CLI forwards all native inputs; prepare freezes the ordered F3 anchor/history source bindings; start/resume rerun each native admission with unique external audit logs. |
| R8 | Early collection PASS stayed valid after more collection; budget exhaustion could publish completion without final full reconciliation. | Registered native account evidence binds exact certified prefixes; appending rows invalidates current readiness; final full trace reconciliation precedes a new protected CP and immutable completion receipt. |
| R3 | Numerical readiness depended on prior live RL, while live RL required numerical readiness; zero RL could still appear complete. | Explicit bounded BC prefix, read-only native WM calibration, separate-process actual Actor/critic probe, safe technical pause, externally reviewed resume and process-local authority. |
| Final commit | A failed final CP could leave the volatile lifecycle terminal even though publication had failed. | Roll back the uncommitted terminal stage, retain NO_GO and a non-final safe CP, allow verification retry without an extra optimizer step. |
| Completion resume | A new process has a fresh source audit and no inherited permission; including those transient values made the same final transaction produce different completion bytes. | Seal completion to its committed CP audit and explicitly exclude process permission. Exact remaining durable fields must still match on repeat publication. |

R2 uses the existing `QualitySupervisionAdmission`, Producer audit and F3
preflight. Each source has its own content/course/teacher/lock identity and log;
a common verifier is permitted only through its reviewed pin. Wrong pin, dirty
verifier, substituted source and malformed external digests fail before Run or
Agent construction. Saved audit facts retain hashes of the native receipt and
each log; restoration checks them and still requires fresh native execution.
See [R2 review](R2_NATIVE_ADMISSION_REVIEW.md).

R8 calls the original `snapshot_trace` and `reconcile_trace`; JSON PASS cannot
create executable evidence. Only the previous pending tail's two publication
flags are normalized for stable prefix comparison. The authoritative final
trace still checks the complete real account. Gzip headers omit timestamp and
filename so equivalent traces are byte-identical after cold restore. Labels
for executed non-tail decisions must be published; the final unexecuted tail
remains pending. See [R8 evidence](R8_COLLECTION_FINAL_EVIDENCE.md).

R3 retains OnlineRun/OnlineGates and the original Agent, imagination losses,
Replay, Reader, optimizers and FullCheckpoint. The online path retains its
existing Reader/snapshot implementation; it was **not migrated to R7's opt-in
versioned transaction path**. R7 is regression-tested separately.

## RL permissions and recovery contract

The lifecycle is WARMUP → WM_CALIBRATING → RL_PROBING →
AWAIT_OPERATOR_REVIEW → RL_ACTIVE → FINAL_VERIFY → COMPLETE_VERIFIED.
Failure pauses RL_BLOCKED. The explicit warmup boundary is below the full
budget; no approval means no further BC spending or silent BC completion.

Calibration reuses native WM forward on an independently seeded fork of the
same WP03/MultiCorpus Natural Reader. Source/window ownership hashes include
corpus identity, contract, behavior policy and account segment. The existing
25,000 unique owner-step / 20 passing report thresholds, ADD support and mask
guards remain intact. Reward/continue bounds must be explicitly configured.
Sample insufficiency blocks admission.

The candidate starts from the exact live Agent/slot snapshot in another CPU
interpreter. It invokes the original RL graph at most twice, proves actual
Quality Actor and critic changes, finite gradients, legal imagination and
unchanged protected owners, then discards its state. The parent compares full
live Agent, optimizer, RNG, Reader/Prefetch, Replay, Publisher, account, Gate,
counters and phase state before/after. Candidate counts are not live updates.

The packet exporter writes only AWAIT material with readable diagnostics and
evidence references. The external human review binds Run/F3 lineage, source
verifier pins, WM/Mask, technical anchor CP, verified account prefix and frozen
budgets. Resume needs both authorization flags and the review's external SHA.
Saved `formal_admitted=true` is ignored. New processes verify the external
review again; tokens are never checkpointed. Authorized budget growth retains
the review, while WM/Mask/source or technical evidence substitution closes RL.
Frozen parameter/slot validation reuses Batch2 R6's exact owner parser.

`bc_updates` and `rl_updates` count active loss components and may overlap;
`optimizer_updates` counts live optimizer invocations. Zero actual RL updates
cannot produce `RL_COMPLETE_VERIFIED`. Failure before final CP publication
cannot leave a durable verified completion. See
[state contract](R3_GATE_STATE_CONTRACT.md) and [operator commands](F4_OPERATIONS.md).

## Verification and evidence scope

Exact commands, CPU environment, actual paired SHAs/dirty states, wall times,
exit codes, full collected case lists and JUnit results are in
[TEST_EVIDENCE_INDEX.md](TEST_EVIDENCE_INDEX.md). Raw failures are preserved and
are not counted as passing runs. No selected suite is described as a whole
project test result.

The frozen R3-v2 pair was Gym
`392709a25da47e49e679107041c70700184d9d70`, Dreamer
`675ecb40ca73c204171af6e8c72e80f09086d674`. Its four completed rounds returned
exit 0: numerical probe/calibration 3 passed; true continuous versus new-process
warmup pause 1 passed; existing cold restore 5 passed; shared/interface 74 passed.
The last 21 boundary tests additionally passed the final CP failure rollback,
native-log mutation rejection and read-only packet diagnostic serialization.
Final integration reruns returned exit 0: shared/interface/R6/R7 145 passed;
controlled-prefix cold pause 1 passed; native account/Gate 6 passed; online
and R7 cold/fault cases 16 passed (**168 passed, 0 failed, 0 skipped**).
They are indexed separately with their actual HEADs. The final terminal-receipt
patch was additionally tested in 11 boundary contracts and 2 actual native
completion scenarios; it is not described as a rerun of all 168 cases.
The final completion regression also exercises repeated publication after
process audit/permission changes and rejects a changed RL update counter.

Cold cases cover actual native accounts/trace, model/optimizer, Replay, stream
RNG, prefetch/pending state and counters. The controlled RL prefix test reads
two synthetic six-behavior Corpora, does real native account reconciliation,
and safely stops with insufficient calibration support and zero RL updates.
Synthetic F3 initialization and the explicit DEV RL probe used by old cold
tests do not certify formal F3 or operator permission.

Native calibration PASS/FAIL uses the real math and unchanged support
thresholds with controlled causal tensor/head fixtures. It is not a certified
source calibration. External review tests isolate document authority with an
explicit unit-only source-authority stub; they prove rejection, binding and
nonpersistent permission, not a real formal RL approval. The separate actual
child probe proves numerical updates and isolation. These scopes must not be
combined into a claim that a formal RL Run was executed.

## Unchanged assets and limits

[Final protected-asset audit](receipts/B3_FINAL_PROTECTED_ASSETS.json) rehashes
all **176** baseline files across two existing Corpora and their file sets,
the frozen Producer pair and its external pin. All match the initial snapshot.
Running production outputs and existing Run artifacts were not written.
The source-scope audit in the final evidence index lists the actual changed
files; financial Env, TP5/SL30, Reward, Teacher/L1/L2, Actor training Loss,
shared Reader/Prefetch and FullCheckpoint implementation are unchanged.

| Acceptance layer | Evidence / status |
| --- | --- |
| B3_CODE_GO | Selected synthetic/CPU implementation tests and final integration receipts. |
| SOURCE_NATIVE_VERIFIED | Formal trained F3 + real published TRAIN source native G3: **NOT_RUN**. Routing stubs prove only argument/log orchestration. |
| WM_CALIBRATED | Actual mathematical fixture PASS/FAIL; real trained F3 calibration: **NOT_RUN**. Tiny real Reader fixture correctly blocks. |
| RL_PROBE_VERIFIED | Actual isolated CPU candidate updates and NaN rejection: PASS on synthetic model. |
| RL_OPERATOR_AUTHORIZED | Document verifier contracts tested; real operator approval: **NOT_RUN**. |
| RL_ACTUALLY_UPDATED | Candidate/explicit DEV CPU updates tested; formal live RL: **NOT_RUN**. |
| FULL_ACCOUNT_VERIFIED | Actual native complete synthetic online accounts: PASS; real formal Run: **NOT_RUN**. |
| RESEARCH_EFFECT | **NOT_RUN**; no claim about trading skill. |

Old online V1/V2 Runs remain bound to their frozen implementation; new OnlineRun
and Gate V3 are a new execution contract, not a Corpus version or an in-place
checkpoint migration. STRICT_PUBLISHED_LATEST_V2 is retained. The remaining
operational step is a separately authorized real F3/source prepare-start-resume
acceptance using the documented CLI; it is not permission for long training.

## Final integration byte identity

The final ordinary merges include the completed 168-case integration revision
and the separately verified terminal-receipt fix. The final runtime tree is
identical to the feature source tree tested for that fix; only evidence/docs
were added afterward. Final HEADs, clean states, runtime blob comparison and
evidence hashes are recorded in the external workspace baseline lock. No
additional long test, formal training or data-production task was started.
