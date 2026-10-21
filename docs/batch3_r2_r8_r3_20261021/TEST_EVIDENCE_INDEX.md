# Batch3 test evidence index

Each command receipt records the exact argv, cwd, CPU environment, paired actual
Git SHAs and dirty states, exit code, elapsed seconds, complete collected case
list, log SHA256 and JUnit path. Links to log/XML use the same filename prefix.
Counts below are **passed / failed-or-error / skipped**. Repeated tests are
separate runs; do not add them together as unique coverage. Early failures are
retained. Only the accepted/final rounds support delivery.

R2_FIRST had empty captured stdout after an existing LD_LIBRARY_PATH-triggered
CLI exec; R2_VERIFIED is the accepted rerun. R8 earlier failures were test
manifest/path assumptions, fixed without weakening native guards. R3 initial
failures exposed typed datetime digest and Reader ownership metadata handling;
these were fixed. Actor's first candidate call changed critic/slots but not
Actor weights under the unchanged original update schedule; a bounded second
call changes both. Old unit assumptions were updated to isolate the actual
prefix guard and separate document verification from source certification.

Numeric/math evidence is from frozen r3-v2, not relabeled as a final-HEAD rerun.
[R3_NUMERIC_BYTE_IDENTITY.json](receipts/R3_NUMERIC_BYTE_IDENTITY.json) records
the identical numeric/calibration implementation and model entry points used
for this narrow inheritance. Full Run source/recovery behavior has its own
post-merge receipts. Source routing stubs and synthetic numerical PASS do not
establish formal G3, source authority or human approval.

| Receipt (exact command and cases) | Exit | Pass/fail/skip | Wall seconds |
| --- | ---: | ---: | ---: | ---: |
| [B3_INTEGRATION_COLD_20261021T050138336947Z.command](receipts/B3_INTEGRATION_COLD_20261021T050138336947Z.command.json) | 0 | 16/0/0 | 714.10 |
| [B3_INTEGRATION_NATIVE_20261021T050138325837Z.command](receipts/B3_INTEGRATION_NATIVE_20261021T050138325837Z.command.json) | 0 | 6/0/0 | 154.04 |
| [B3_INTEGRATION_PAUSE_20261021T050138326080Z.command](receipts/B3_INTEGRATION_PAUSE_20261021T050138326080Z.command.json) | 0 | 1/0/0 | 303.09 |
| [B3_INTEGRATION_SHARED_20261021T050138336953Z.command](receipts/B3_INTEGRATION_SHARED_20261021T050138336953Z.command.json) | 0 | 145/0/0 | 586.38 |
| [R2_FIRST_20261021T034149056862Z.command](receipts/R2_FIRST_20261021T034149056862Z.command.json) | 0 | 39/0/0 | 5.17 |
| [R2_INTEGRATION_20261021T034721570980Z.command](receipts/R2_INTEGRATION_20261021T034721570980Z.command.json) | 0 | 40/0/0 | 3.11 |
| [R2_SHARED_20261021T034307544219Z.command](receipts/R2_SHARED_20261021T034307544219Z.command.json) | 0 | 23/0/0 | 173.00 |
| [R2_VERIFIED_20261021T034255536124Z.command](receipts/R2_VERIFIED_20261021T034255536124Z.command.json) | 0 | 40/0/0 | 2.96 |
| [R3_AUTH_SCOPE_VERIFIED_20261021T043716216462Z.command](receipts/R3_AUTH_SCOPE_VERIFIED_20261021T043716216462Z.command.json) | 0 | 9/0/0 | 3.16 |
| [R3_COMPLETION_IDEMPOTENCE_20261021T050548940035Z.command](receipts/R3_COMPLETION_IDEMPOTENCE_20261021T050548940035Z.command.json) | 0 | 11/0/0 | 5.39 |
| [R3_CONTRACT_FIRST_20261021T041204269024Z.command](receipts/R3_CONTRACT_FIRST_20261021T041204269024Z.command.json) | 0 | 16/0/0 | 3.30 |
| [R3_FINAL_BOUNDARIES_20261021T050041184973Z.command](receipts/R3_FINAL_BOUNDARIES_20261021T050041184973Z.command.json) | 0 | 21/0/0 | 4.90 |
| [R3_FINAL_COMPLETION_NATIVE_20261021T050619415871Z.command](receipts/R3_FINAL_COMPLETION_NATIVE_20261021T050619415871Z.command.json) | 0 | 2/0/0 | 50.31 |
| [R3_NUMERIC_FIRST_20261021T041658577680Z.command](receipts/R3_NUMERIC_FIRST_20261021T041658577680Z.command.json) | 1 | 1/1/0 | 153.30 |
| [R3_PAUSE_FIRST_20261021T041658583055Z.command](receipts/R3_PAUSE_FIRST_20261021T041658583055Z.command.json) | 1 | 0/1/0 | 269.65 |
| [R3_SCOPE_FINAL_20261021T044400829556Z.command](receipts/R3_SCOPE_FINAL_20261021T044400829556Z.command.json) | 0 | 9/0/0 | 4.03 |
| [R3_SCOPE_V2_20261021T044106852436Z.command](receipts/R3_SCOPE_V2_20261021T044106852436Z.command.json) | 1 | 6/3/0 | 3.26 |
| [R3_V1_COLD_20261021T043009834890Z.command](receipts/R3_V1_COLD_20261021T043009834890Z.command.json) | 0 | 5/0/0 | 220.60 |
| [R3_V1_CONTRACT_20261021T043009818227Z.command](receipts/R3_V1_CONTRACT_20261021T043009818227Z.command.json) | 1 | 50/1/0 | 166.96 |
| [R3_V1_NUMERIC_20261021T043009813895Z.command](receipts/R3_V1_NUMERIC_20261021T043009813895Z.command.json) | 0 | 3/0/0 | 263.63 |
| [R3_V1_PAUSE_20261021T043009814518Z.command](receipts/R3_V1_PAUSE_20261021T043009814518Z.command.json) | 1 | 0/1/0 | 307.26 |
| [R3_V2_COLD_20261021T044551910030Z.command](receipts/R3_V2_COLD_20261021T044551910030Z.command.json) | 0 | 5/0/0 | 220.58 |
| [R3_V2_NUMERIC_20261021T044551922324Z.command](receipts/R3_V2_NUMERIC_20261021T044551922324Z.command.json) | 0 | 3/0/0 | 265.22 |
| [R3_V2_PAUSE_20261021T044551915947Z.command](receipts/R3_V2_PAUSE_20261021T044551915947Z.command.json) | 0 | 1/0/0 | 308.71 |
| [R3_V2_SHARED_CONTRACT_20261021T044551919522Z.command](receipts/R3_V2_SHARED_CONTRACT_20261021T044551919522Z.command.json) | 0 | 74/0/0 | 353.70 |
| [R8_COLD_FIRST_20261021T035652567402Z.command](receipts/R8_COLD_FIRST_20261021T035652567402Z.command.json) | 1 | 0/2/0 | 89.26 |
| [R8_FROZEN_COLD_20261021T035903545289Z.command](receipts/R8_FROZEN_COLD_20261021T035903545289Z.command.json) | 1 | 0/2/0 | 95.21 |
| [R8_FROZEN_NATIVE_20261021T035903554426Z.command](receipts/R8_FROZEN_NATIVE_20261021T035903554426Z.command.json) | 1 | 4/2/0 | 131.57 |
| [R8_FROZEN_V2_COLD_20261021T040254609031Z.command](receipts/R8_FROZEN_V2_COLD_20261021T040254609031Z.command.json) | 0 | 2/0/0 | 208.25 |
| [R8_FROZEN_V2_NATIVE_20261021T040254610135Z.command](receipts/R8_FROZEN_V2_NATIVE_20261021T040254610135Z.command.json) | 0 | 6/0/0 | 162.90 |
| [R8_NATIVE_FIRST_20261021T035232112027Z.command](receipts/R8_NATIVE_FIRST_20261021T035232112027Z.command.json) | 1 | 0/4/0 | 219.16 |
| [R8_NATIVE_VERIFIED_20261021T035652566610Z.command](receipts/R8_NATIVE_VERIFIED_20261021T035652566610Z.command.json) | 0 | 4/0/0 | 97.35 |

Additional durable evidence:

- [Protected original assets](receipts/B3_FINAL_PROTECTED_ASSETS.json): 176 files,
  frozen Producer pair and external Pin unchanged.
- [Changed runtime paths and SHA256](receipts/B3_SOURCE_SCOPE_AUDIT.json).
- [R3 numerical candidate raw report](receipts/R3_V2_BATCH3_ISOLATED_CANDIDATE.json).
- [Native calibration mathematical PASS/FAIL](receipts/R3_V2_BATCH3_CALIBRATION_MATH.json).
- [Exact continuous/new-process RL prefix pause](receipts/R3_V2_BATCH3_RL_PAUSE.json).
- [Four online cold restore cases](receipts/R3_V2_G_cold_cases.json).
- [Native UUID process-hash-seed test](receipts/R3_V2_G_uuid_hash_seed.json).
- [Earlier R8 full-account/cold artifact inventory](receipts/R8_FROZEN_V2_ARTIFACT_INVENTORY.json).

Formal trained F3, real published TRAIN native G3, real source WM calibration,
external human RL approval, formal live RL, GPU, long training and research
performance: **NOT_RUN**. TEST market data: **NOT_READ**.
