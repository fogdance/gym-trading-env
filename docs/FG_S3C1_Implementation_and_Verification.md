# FG S3-C1 Implementation and Verification

> 交付后复查新增 3 项未修复 P2 问题，当前收尾状态见
> [问题清单与复现](FG_S3_Post_Delivery_Review.md)。下文 PASS 保留其历史执行范围。

## Summary

AR0–AR5 and C0–C4 CPU engineering acceptance are complete. Full real TRAIN
Catalog reconciliation, actual original neural models, equal-weight BC,
three-source transfer and exact runtime recovery have execution receipts.
Formal S3-C GPU training, DEV, sealed TEST and production convergence are NOT_RUN.

C functional commits: Dreamer `21fa6f9`, Gym `24eeb42`; final dependency callers
`b382cd9` / `dd03a9d`. Later user-authorized S3-A source work is Dreamer
`93a6fb65b493b60f7c1205b8351d7024cbfc326a` and Gym
`5293e978d3f234d5f7c3149f326f1da831012bad`.
These later changes extend public source admission; C model/loss/stream/runtime
files are byte-identical to the accepted C implementation.

## Evidence observed

Development uses sibling worktrees under `research/fg-s3-dev`; original research
HEADs `d374e93` / `bb1d123` are retained. TRAIN data is reused through the
user-created `research/fg-data` link, without importing external experiment code.

Real Catalog: 488629 unique P4 rows; 402096 legal, ADD 16171, WAIT 385925;
excluded held 82251, no-trade 3795, administrative-last 469, terminal 18.
All 2435 native C1 chunks are authenticated. Catalog SHA:
`47736742ff81e337d10585b86d0709f5fd1bff8d082b01324b267f9f1761f0de`.

Actual old S3-A20/base packages use a small original BF16 neural model and a
CLOSED Gate. Their two C updates have effective BC=0. READY fixtures are clearly
controlled CPU diagnostics; their two updates have effective BC=2. Both are
reported separately. The real TRAIN diagnostic slice links four original C1
P1/P4 chunks; arrays and labels are not regenerated.

The user subsequently selected the live online run's published 180000 node.
It has updates=21183, batches=21399, actions=180000, Gate=false. A frozen full
copy and an actual original S3-A neural CPU load preserve all 286 parameter /
state entries byte-exactly. This proves production-source transfer, without
claiming a production C joint update. Original source manifest SHA:
`4979acfb0272ffbc3c96833e5b77fccd66c5d4d0f335bfca48810f764d927a57`.

## Root cause

A/B's feature-cache and independent BC optimizer workflow cannot implement C's
current-posterior joint objective. The existing S3 manager now dispatches C to
its own preparation/runtime protocol while reusing shared teacher and source
services. `fg_single3_experiment.py` remains the other offline/online workflow.

Real execution exposed three shared defects: native Replay's synthetic first
reset needed an explicit authenticated sampling mode; Replay load required the
complete frozen filename/hash set; full-source dependency checking needed the
producer's entire inventory, including portal/cloudpickle. Each was fixed in
the public layer and reaccepted before its C caller continued.

A later user-selected online source exposed a fourth boundary: generic online
FG_FULL provenance differs from Single3 offline identity, and checkpoint step
180000 is an environment count rather than Agent updates=21183. Public admission
now verifies the frozen online source, its C1 TRAIN/offline lineage and producer
code, and reads/binds actual Agent counters. C has no private online loader.

## Changes made

- `dreamerv3/fg_s3c_agent.py` inherits original train/report/policy. Its two
  ext-space fields are teacher_action/teacher_valid. Added loss is
  `L_base + lambda * Gate * mean_valid(-logp(teacher_action))`.
- BC uses stop_gradient on current posterior repfeat through original
  feat2tensor and pol/masked distribution. Invalid rows receive only a temporary
  WAIT-safe mask/action. No new network, head, loss scale or independent optimizer.
- Lambda zero statically returns original loss: no added BC forward or PRNG.
  Original optimizer, slow-value update, counter and carry code run once.
- TeacherJoinStream sits inside original Agent Prefetch. Shared lookup remains
  outside JIT; effective 65→64/context alignment is checked. Report gets safe
  placeholders, and policy/native Replay/Env get no teacher labels.
- S3CRun uses shared SourceAdmission/transfer and native Replay admission;
  preserves model, joint optimizer, Gate/window, normalization and source count.
  It restores original FullCheckpoint/RuntimeState, both Prefetch queues,
  Consec.current, carry, sampler/chunks, aggregates, schedules, RNG and counters.
- Existing `tools/fg_s3_experiment.py` manages plan/prepare/start/resume/audit.
  Canonical YAML, official presets/Flags and immutable resolved snapshots are
  used. Start checks prepared source; resume restores its own run without source
  readmission. No private C index, feature-cache phase or model migration API.

Protocols: teacher FG_S3_TEACHER_CATALOG_V1; source FG_S3_SOURCE_ADMISSION_V1;
C preparation FG_S3C_PREPARED_V1; C run FG_SINGLE3_S3C1_CONCURRENT_BC_V3;
full state FG_FULL_V1. Broader A source preparation is FG_S3_PREPARED_V2;
A's existing Actor BC package/math stay unchanged. After BC uses the explicit
retain_base_joint_opt policy: post-BC Actor with pre-BC original joint moments.

## Verification performed

Evidence directory in Gym: `artifacts/fg_s3c1_v1/receipts/`.
Machine-readable final index: `c_final_acceptance_summary.json`.
All model tests ran on CPU with empty CUDA_VISIBLE_DEVICES; native threads=1.

| Executed receipt | Result | Scope |
| --- | --- | --- |
| c_final.xml | 17 passed, 1279.668s | Neural NLL/SG/Gate/context; streams/resume; original A/B and C manager |
| ready_lambda_resume_final.xml | 4 passed, 230.191s | READY lambda=0 for all three sources; real Gate transition/exact resume |
| final_dependency_c_regression.xml | 5 passed, 213.561s | Changed dependency callers: full restore/source inheritance/C management |
| s3a_shared_source_first.xml | 1 passed, 34.334s | New shared A admission, actual Actor BC, resume and A→C identity |
| s3a_shared_source_regression.xml | 11 passed, 304.025s | A/B/C management and exact neural recovery after this manager change |
| online_counter_contract.xml | 1 passed, 1.821s | Synthetic serialized boundary fixture, env/model domains and tampered counters |

C acceptance contains 20 distinct original cases. Repeats are not extra
coverage; all listed executions have 22 distinct names including the two added
source contracts. At the user's request, already passed suites were not rerun
for documentation or the README-only merge. Source changes got their own direct
verification; online counter fixture is not a production neural acceptance.

Actual commands (Gym cwd unless noted; Python is
`/home/v/miniconda3/envs/dreamerv3/bin/python`):

```bash
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/v/Documents/work/research/fg-s3-dev/dreamerv3:/home/v/Documents/work/research/fg-s3-dev/gym-trading-env/src
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY -m pytest -q ../dreamerv3/embodied/tests/test_fg_s3c_agent.py ../dreamerv3/embodied/tests/test_fg_s3c_stream.py ../dreamerv3/embodied/tests/test_fg_s3c_resume.py tests/entry_v1/test_fg_s3_experiment.py tests/entry_v1/test_fg_s3c_experiment.py --junitxml=artifacts/fg_s3c1_v1/receipts/c_final.xml
$PY -m pytest -q ../dreamerv3/embodied/tests/test_fg_s3c_agent.py::test_lambda_zero_real_agent_equivalence_each_source ../dreamerv3/embodied/tests/test_fg_s3c_resume.py::test_real_neural_joint_runtime_exact_resume_no_source_reload --junitxml=artifacts/fg_s3c1_v1/receipts/ready_lambda_resume_final.xml
$PY -m pytest -q ../dreamerv3/embodied/tests/test_fg_s3c_resume.py tests/entry_v1/test_fg_s3c_experiment.py --junitxml=artifacts/fg_s3c1_v1/receipts/final_dependency_c_regression.xml
$PY -m pytest -q tests/entry_v1/test_fg_s3_experiment.py tests/entry_v1/test_fg_s3c_experiment.py ../dreamerv3/embodied/tests/test_fg_s3c_resume.py --junitxml=artifacts/fg_s3c1_v1/receipts/s3a_shared_source_regression.xml
```

The final inline source/neural/payload probes have raw stdout, full signed JSON
and commands recorded in the command receipt; four actual CLI configurations
executed prepare/start/resume/audit, all **16 commands exit 0** in
`real_cpu_cli_deps_v2_commands.json`. This is real TRAIN on small original CPU
models. It predates the additional online-source extension and preserves its
actual dd03a9d/b382cd9 code identities; these old snapshots were not resigned.

Every run consumed 196 valid rows: ADD=1, WAIT=195, Always-WAIT=195/196.
float32/BF16 READY runs had 2 effective BC updates. Real base/S3-A20 CLOSED runs
had 0, while original WM still updated. All four latest saved Agent payloads,
carry, both queue/Consec states, sampler RNG/indices/keys, Gate, counters,
scheduler and reducers were compared to actual restoration:
`real_dtype_restore_deps_v2.{json,stdout}`, 4 PASS.

N=3 versus K=1+restore+2 compares actual native stepids, teacher arrays, outputs,
all Agent params/opt/policy/norm/counters, carry, sampler and Gate. The original
controller transitions after report 2; used Gate values are False, False, True,
and effective BC=1. Original source readmission/transfer is forbidden on resume.

FAIL then corrected: early fixtures referenced an incorrect metric name / Ninjax
context; the isolated NLL grad test requested unused modules; one tamper
assertion expected settings failure after runtime code had changed; initial
shared dependency inventory omitted serializers. Raw failure receipts remain,
with passing replacement runs above. No failure or unexecuted test is marked PASS.

## Risks / not verified

PASS covers CPU engineering, full real teacher reconciliation, original A/B
regression, actual old atomic A transfer, both compute dtypes, exact small-model
C recovery and the selected production-source CPU transfer. Current source
extensions and merge leave C agent/loss/stream/recovery bytes unchanged; their
historical passing evidence is retained with exact commit scope.

NOT_RUN: production-sized C gradient training / production C convergence,
formal GPU by Codex, DEV, sealed TEST, trading performance. User startup is
separate and not inferred from supplied commands. Source is still running per
user; only a published frozen node is used. No process was stopped or overwritten.
Old prepared/run code identities reject changed runtime code; use a new
preparation rather than modifying receipts. Broader online admission needs
hash-bound producer code metadata and the verified C1 Single3 lineage.

## Next recommended step

User-selected S3-A starts from the frozen online 180k node using the already
prepared recipe S3A_ORACLE_BC_FROM_ONLINE_180K in `research/fg-s3-train`.
Training commits are Gym 5293e978d3f234d5f7c3149f326f1da831012bad and
Dreamer 93a6fb65b493b60f7c1205b8351d7024cbfc326a. Source model counters are
retained; new A BC updates run from 0 to 1000. C supports direct_init,
direct_full and after_bc_s3a without choosing the next training sequence.
See the paired final review and S3-A startup record before a new run.
