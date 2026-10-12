# Shared architecture implementation and verification

## Summary

AR0–AR5 engineering acceptance and the real TRAIN Catalog audit passed.
Functional commits: Gym `47c76cd`, Dreamer `2b61ea3`. Both development worktrees
are under `/home/v/Documents/work/research/fg-s3-dev`, branch
`feat/fg-s3-shared-s3c1`. Original repositories retain `d374e93` / `bb1d123`.
User-approved scope removes old API/run compatibility requirements; source
transfer remains explicitly audited. C implementation starts after this acceptance.

## Evidence observed

`research/fg-data` is a data-only symlink to the original signed artifacts.
Python imports were actually checked and originate in the new sibling worktrees.
No source code is imported from the linked experiment. Frozen labels and the
existing CPU S3-A package are read from the original research Gym artifacts.

Real Catalog manifest SHA:
`47736742ff81e337d10585b86d0709f5fd1bff8d082b01324b267f9f1761f0de`.
488629 unique P4 rows, 402096 valid, ADD 16171, WAIT 385925. Exclusions: held
82251, no-trade 3795, administrative-last 469, terminal 18. Every sparse field
was reconciled; 2435 C1 native chunks are authenticated. Real mixed
P1/P2a/P2b/P3/P4 host lookup/join passed.

The existing 20-update S3-A CPU package was actually loaded into a fresh neural
Agent: all transferred parameter/state digests match, non-Actor and joint opt
match the actual base, policy is synchronized, independent BC opt absent.
It is a 40-feature small neural diagnostic, not the production model.
Original counters inherited: updates=0, batches=0, actions=1100; actions reflect
prior feature extraction. BC updates=20 is a separate counter.

## Root cause / design findings

A real full-build failure identified Natural's `original_training_hash` versus
P4's Oracle `training_hash`. The public ownership checker now validates the
correct identity variant. A separate original-Env Natural account regression
covers it. The failed full-build stdout is retained. A subsequent fixture setup
failure used a minimal Env config without its factory; the regression now uses
that same original factory. Failed XML/raw output remain available.

The old S3-A source predates the reviewed safe `MaskedCategorical.prob` fix.
Its producer bytes were located in local Git; AST comparison proves only that
probability diagnostic and the default loss-factory extension differ. Source and
target hashes are separately recorded. No old manifest is resigned.

## Changes made

Complete mmap P4 catalog; native chunk ownership; canonical validity/exclusion;
strict sparse set reconciliation; pure host teacher join; SourceAdmission;
complete Agent transfer and digest audits; frozen-source snapshots; real CLIs.
A default training-loss factory preserves the original loss while allowing C
without copying train/optimizer/slow-value/carry logic. A/B algorithms were not
changed. Test import paths now point to the sibling worktrees, and the stale
pre-probability-fix assertion was updated to current observed behavior.

## Verification performed

Receipts: Gym `artifacts/fg_s3_arch_v1/receipts/verification.json`, XML, stdout,
real_catalog_audit.json and complete source/transfer receipts. All commands use
`/home/v/miniconda3/envs/dreamerv3/bin/python`, explicit sibling PYTHONPATH,
`JAX_PLATFORMS=cpu`, empty CUDA_VISIBLE_DEVICES and one native thread.

- `python -m pytest -o addopts= -q -p no:cacheprovider` with shared Catalog/source,
  A Actor freeze, B batch-one/global-mean/ones/resume, Mask and replay-context:
  `architecture_final.xml`: 19 passed in 63.44s.
- Final Natural/source regressions: `natural_source_final.xml`: 7 passed in 34.40s.
- Final source snapshots/inference-disabled/missing-base contracts:
  `source_final_v2.xml`: 3 passed. Across these runs, 21 distinct test names;
  repeated tests are not added to the total.
- Real `tools/fg_s3_arch.py build-catalog`: PASS, full counts above.
- Real `scripts/fg_s3_arch.py transfer-smoke --mode after_bc_s3a` from saved
  bc_000000000020 and its hash-bound base: PASS_REAL_MODEL_TRANSFER on TFRT_CPU_0.
- Source/catalog tampering, shifted IDs, missing/invalid/duplicate labels,
  forbidden resume/unsupported S3-B, mismatched schemas/sources: expected rejection.
- AST parse and diff checks passed; own code/diffs reviewed. Raw failures with
  trailing whitespace are preserved as `.raw.gz`; readable copies only trim
  line-end whitespace.

## Risks / not verified

PASS: AR0–AR5, full real TRAIN Catalog, CPU model transfer, affected A/B contracts.
NOT_RUN: production large Agent payload transfer, formal GPU, DEV, TEST,
production training/convergence. No static or synthetic result is labelled a
production pass. External Gate/window control state is exposed by a public
transfer API; a formal C runner must reject missing necessary control state.

## Next recommended step

### C integration return review

C's actual Replay sampling exposed a public join boundary: `Replay.sample`
sets the sequence's first is_first to True. The shared adapter now takes explicit
`sampling_flags=True` and authenticates that transform only at the first row of
consec=0. All other flags/actions/masks/position fields remain strict. No native
data or teacher definition changed. Shared fix: Dreamer `e44b10f`.
Catalog + actual Replay/Prefetch/Consec regression: `shared_sampling_after.xml`,
8 passed in 49.53s. Failure and final receipts are in S3-C's receipt directory.
The plain native join still rejects sampling-modified flags, and the annotated
join rejects a missing synthetic reset. Catalog format and hash did not change.

Implement C0–C4 using these public APIs. Keep original joint optimizer, use
current posterior repfeat, add equal-weight Actor-only BC gradient, and prove
lambda-zero and exact FullCheckpoint/runtime recovery before delivery.
