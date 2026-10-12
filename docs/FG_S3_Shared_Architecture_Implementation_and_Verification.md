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

## Second shared-interface return review (2026-10-12)

Before C runner integration, authenticate the complete native Replay file set
and bytes in the shared teacher adapter. Formal admission requires all 2435 C1
chunks; diagnostic subsets require explicit admission and every included file
still matches frozen hashes/ownership. C has no private UUID or label parser.
The real Catalog metadata/view comparison found exactly 2435 expected and actual
filenames, zero unknown or missing.

SourceAdmission now accepts an explicit boolean diagnostic flag. This allows a
small CPU diagnostic model on a real TRAIN Catalog without presenting it as a
production source. It never upgrades diagnostic provenance to formal. The flag
contract is tested separately from real neural source transfer.

Functional commits: Gym `396db7aba93abff4df6cb8618b19efd7260a9bf1`,
Dreamer `4b77f11f6dcc663735a9ce0a35dfcd88747d5462`.

Actual regressions: full Catalog/pool and three-source CPU tests: **10 passed in
60.86s**. After the explicit diagnostic API refinement, repeat the three real
source cases: **3 passed in 40.10s**. This is 10 unique cases, not 13. Native
fixture prices are synthetic; models are actual original neural Agents and BC
packages. Formal GPU/production source training, DEV and sealed TEST: NOT_RUN.

Complete commands and raw XML/stdout are in Gym
`artifacts/fg_s3c1_v1/receipts/shared_return_review.json`,
`shared_pool_after.{xml,stdout}`, `shared_diagnostic_after.{xml,stdout}`.
Public changes reviewed for duplicate teacher definitions, independent BC
optimizer import and source mutation; none introduced. C may now consume these
public contracts.

## Third public-interface return review: full-source dependencies

The read-only completed original 10K checkpoint rejected admission despite
matching installed versions. Confirmed cause: SourceAdmission compared its
producer inventory against A/B's shorter package_versions() list, so portal
3.5.1 and cloudpickle 3.1.0 compared as None. All nine producer-declared installed
versions actually match. This is a shared protocol defect, not a C private
loader issue. Old producer identities/checkpoints were not edited.

Shared fix: verify every declared package via importlib.metadata, rejecting
missing packages or exact-version mismatch. Public runtime_versions also binds
stream/snapshot serialization dependencies. Functional Dreamer commit is
recorded by the final C report. Regression: **4 passed in 40.66s** in
`shared_dependencies_final.{xml,stdout}`; it includes actual three-mode neural
transfer plus matching portal/cloudpickle, mismatch and missing-package cases.

Actual real published 10K metadata and external Gate/window admission now
passes, and a corrupted copied full bundle is rejected before model load.
Evidence: `source_boundaries_final.{json,stdout}`. Agent payload bytes were
hashed for bundle integrity, not deserialized or trained. Production-sized
model transfer and training remain NOT_RUN. Source Gate/window/opening/samples
are read through the same public API, not a C bypass. Initial failure is
preserved in `source_boundaries.stdout`.

C must consume the public dependency inventory and create new immutable
diagnostic snapshots; previous run/prepared hashes remain unchanged and are
not resigned. Repeat the C restore/CLI regressions after this shared acceptance.
