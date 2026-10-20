# Requirements and evidence

Final pytest scope: 268 distinct cases, 298 executions, zero failures/errors/skips.
FINAL_TEST_INDEX.json/tsv enumerates the actual collected names and test hashes.
References below are relative to `docs/receipts/corpus_semver_r1_r4_20261020/`.

| Package / requirement | Evidence | Outcome |
|---|---|---|
| A0 actual integration pair, independent worktrees and protected assets | SOURCE_BASELINE.md; A0_baseline.json | PASS |
| A1 exact 3.1.0; no broad major/V3 trust | test_corpus_compatibility.py; Z1_v2_reader_multi.xml | PASS |
| Known V3 variants / E-profile math compared before mapping | A1_historical_profile_semantics.json; A1_unmodified_consumer_comparison.json | PASS |
| Unknown version, V2, missing behavior, teacher/source/MACD/byte changes reject | compatibility and WP03 Admission cases in final index | PASS |
| Consumer code changes do not modify asset identity | two real Reader reports; CORPUS_UNCHANGED_PROOF.json | PASS |
| Future Producer writes explicit course_abi_version | fg_s3_e_quality_corpus_cli.py; exact supported constant | IMPLEMENTED; real Generate NOT_RUN |
| B1 all three stages, prepare/read-check/start/resume CLI and API forwarding | test_corpus_producer_pin.py; 12 CLI dispatch cases plus three lower read-check calls | PASS |
| Formal V3 has no historical-default pin fallback | require_stage_producer_pin; missing/half-pin negatives | PASS |
| Wrong digest, Gym/Dreamer SHA, dirty/untracked runtime code reject | test_corpus_verifier_binding.py, actual isolated Git pairs | PASS |
| B2 stable pin/ABI/source in Prepared and Run; different valid pin rejects | three prepare→start→resume routing cases, changed pin/ABI negatives | PASS — synthetic routing only |
| Real G3 remains actual frozen subprocess, six complete label receipts and source audits | unchanged native_verification implementation; wrapper forwarding tests | PRESERVED; real invocation NOT_RUN |
| In-process proof and separate training authorization remain required | forged native receipt and authorization negatives; three-stage binding cases | PASS |
| Active production final source_guard_interval | protected-production metadata in CORPUS_UNCHANGED_PROOF.json | NOT_RUN — final Manifest not published |
| C1 all five Checkpoint construction sites, including shared HeadsRun runtime | three real Run assembly cases; Z1_v2_contracts.xml | PASS |
| New recovery policy; old identity cannot silently migrate | STRICT_PUBLISHED_LATEST_V2; legacy-identity negative in all three Run cases | PASS |
| C2 missing done/Manifest, bad member, identity/backend/strict/node-step mismatch | test_fg_course_strict_checkpoint.py fault matrix | PASS |
| Pointer absent, regressed, external, symlink; node symlink | same fault matrix, before any mutable restore | PASS |
| Explicit old load/save cannot bypass latest | explicit-old-load/save regression | PASS |
| Pending preserved; rename-before-pointer orphan refuses rollback | pending/orphan injection tests; RECOVERY_SOP.md | PASS |
| Interrupted final protection preserves published node | final Manifest interruption regression, failure retained then fixed | PASS |
| Durability barrier failure before publication preserves old pointer | injected fsync failure regression | PASS |
| Reader/index/log prefix and chronology before model restoration | all-stage early-rejection loops; JournalState and validate_update_log reuse | PASS |
| Actual new-interpreter full-state exact cold recovery | Z1_v2_f1/f2/f3.xml and cold_evidence_v2/ | PASS — synthetic CPU JAX |
| F3 skip/rejection: consumed batches, updates and schedule remain consistent | WP06 ordinary/skip_first cold cases and nonfinite rejection case | PASS |
| Natural/auxiliary roles, source RNG, account/Pair boundaries | DATA-MULTI / F3 sampling tests in Z1_v2_reader_multi.xml | PASS |
| Legacy Replay / F4 strict default policy preserved | guarded external-only changes; generic FullCheckpoint and WP07 core regressions | PASS for selected core; full formal F4 NOT_RUN |
| Required shared regression | Z1_v2_shared.json/xml; three AGENTS.md files | PASS — 20 cases |
| No financial / Loss / Teacher / native Reader math changes | Z1_source_invariants_v2.json; CODE_DIFF_GYM/DREAMER.patch | PASS |
| Z1 paired commits, exact commands/exits, original receipts, review | COMMITS.json; FINAL_COMMANDS.json; TEST_REPORT.md; IMPLEMENTATION_REVIEW.md | DELIVERED |

Real formal prepare/start/resume and production G3 are intentionally NOT_RUN in
this delivery. Synthetic routing PASS never stands for real G3 or model research
admission. Real TRAIN optimizer updates and GPU execution are zero. No TEST
market data was opened. Remaining R2/R3/R5–R8 workflows are outside this patch.
