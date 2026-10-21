# Corpus production / resume repair — 2026-10-21

## Result and boundary

The repair is in the actual producing Gym worktree, branch
`fix/corpus-stage-drain-20261021`, based on
`cadd6c87659665cc627e2275ab51ba9d14bb70cb`.
Dreamer stays at `a6a6ad9e18196f799c4517f370dd26a4ca68aaf5`.
Final executor SHAs, Pin and migration evidence are recorded in
`../release/RESUME_READY.json` (relative to the Gym repository).
No merge to `integration/s3-next`, formal production launch, TEST access,
GPU execution or optimizer update is part of this repair.

## Observed failure and confirmed defects

TRAIN02/CZCE.FG901 wrote CORPUS_PRODUCED with all six native account
reconciliations PASS (29,444 records per behavior), then its controller
attempt failed with `Phase left live descendant processes; refusing completion`.
TRAIN01/CZCE.FG809 Reader was subsequently terminated with -15 by global
cancellation. Its Native G3 had already succeeded. TRAIN02 has not passed
independent Native G3 or Reader.

The baseline checks live descendants once immediately after leader exit,
before recording its exit status. A finite control reproduced rejection of
an exit-0 phase whose helper exited naturally about .26 seconds later.
Historical PID 2583562's command and the leader's exit status were not saved;
neither the historical helper type nor an unrecorded leader error is asserted.
The operator's kernel-journal query returned no entries for the failure window.

Resume had four independent bindings: current executor Pin/HEAD, a recipe
including that Pin, corpus producer SHA versus executing HEAD, and current
code inventory versus historical absolute paths/hashes. Changing only one
of these checks could not restore production safely.

## Changes and review

- Wait up to five seconds for successful phases' owned groups to drain.
  Preserve rejection of persistent descendants and nonzero exits. Save
  `process_exit.json` before cleanup, including PID/start ticks/command and
  leader exit code; retain ownership checks before sending signals.
- Authenticate the complete historical code inventory against the original
  commits' Git blobs in the current repositories, without reading/executing
  a historical checkout. Current verifier and original producer Pins are
  separate. Historical Pin files are retained, not overwritten or re-signed.
- G3 uses current code plus data files for live source guards. Reconstruct
  historical provenance descriptors separately; only authenticated fields
  ending in `code_sha256` and source provenance are represented in the
  historical namespace. Financial fields are independently recomputed.
  `reconcile_trace` still compares the entire signed label for equality,
  including L1, L2, strict action, ties, funding, exact fractions and account
  ownership. No comparison was removed or replaced by approximate tolerance.
- NativeLabelPool transports this same verification context to spawned
  workers. Native Env, Reward, L1/strict Teacher, Oracle, Replay reconciliation
  and training Loss implementations are unchanged.
- Data recipe V2 binds ABI 3.1.0 and the original source/batch/seed/Reader
  contract. Execution Pins, resource allocations, verified asset locations
  and adoption mechanics are separate. Plan assets retain their SHA256.
- `prepare-resume --config NEW --previous-config ORIGINAL --apply` validates
  the old recipe, existing successful completions and recoverable Corpus;
  backs up old state, records a signed migration and changes only control
  metadata. Original attempts and completion receipts remain unchanged.
- Resume recognizes complete failed/interrupted generation outputs and
  creates a new ADOPTED_GENERATED_CORPUS attempt using the original Pin.
  It never promotes the failed attempt or treats generation as G3 PASS.
  A partial generation without final artifacts is rerun in a new directory.
- Publication records the actual original producer Pin and the verifier Pin
  from the selected G3 receipt, including when that receipt is reused.

Historical generation provenance, current execution identity and data
compatibility were reviewed as separate boundaries. `git diff --check`
passed. No existing Corpus, frozen optimized Producer or Run was edited.

## Necessary verification performed

All commands run from this Gym repository with:

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 \
  PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider
```

Append the following selections:

| Selection | Result |
| --- | --- |
| `tests/entry_v1/test_corpus_scheduler.py tests/entry_v1/test_corpus_production_controls.py -q` | exit 0, 27 passed |
| `tests/entry_v1/test_corpus_provenance.py -q` | exit 0, 3 passed |
| `tests/entry_v1/test_corpus_resume_fix.py -q` | exit 0, 4 passed |
| `tests/entry_v1/test_corpus_resume_fix.py tests/entry_v1/test_wp02_cli.py::test_E04_resigned_teacher_forgery_is_rejected_by_native_facts ../dreamerv3/embodied/tests/test_fg_s3_gate.py ../dreamerv3/embodied/tests/test_fg_s3_source_transfer.py ../dreamerv3/embodied/tests/test_action_mask_formal_guards.py --junitxml=docs/corpus_resume_fix_20261021/targeted_checks.xml -q` | exit 0, 26 passed |

The provenance test generates all six actual native behaviors using synthetic
prices, removes the original checkout paths, updates the executor commit,
then reconstructs every native row and full signed label with both sequential
and spawned parallel teachers. Original output file hashes stay identical.
Source tampering and re-signed historical code hashes are rejected.

Exit regression covers asynchronous drain, a persistent live descendant and
nonzero leader exit. Migration regression checks unchanged attempt records,
original Pin selection, idempotency and rejection of changed date ranges.

The first sandbox scheduler run could not create loopback sockets
(`PermissionError: Operation not permitted`); the authorized host retry passed.
The first new exit fixture omitted its own capacity setup; correcting the
fixture produced the four passing results above.

Actual TRAIN01/02 historical inventories have also been checked read-only:
181 Gym + 106 Dreamer files, and 186 Gym + 106 Dreamer files, respectively,
against their immutable original Git objects (exit 0).

Additional required shared-guard and label-forgery results are captured in
`targeted_checks.xml`; the completed counts and actual migration outcome are
recorded in `FINAL_EVIDENCE.json` and release `RESUME_READY.json`.

## Production continuation

The release repair script preserves the old configuration and original Pin,
copies the 18 signed batch plans and unchanged audit programs into the
release's asset directory, freezes the new executor, and invokes the explicit
control-state migration. It does not launch a production stage.

After RESUME_PREPARED:

```bash
bash /home/v/Documents/work/research/fg-s3-formal-corpus-scheduler-20261021/release/RUN_CORPUS.sh resume
```

This continues the original 18-batch plan and output root. TRAIN01 reruns its
incomplete Reader and keeps its completed G3. TRAIN02 adopts its existing
Corpus in a fresh attempt, then executes the missing independent G3/Reader.
Full real-contract G3, Reader completion and final publication after this
repair are **NOT_RUN** until the operator resumes production.
