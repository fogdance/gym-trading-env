# R6 implementation review

## Summary

One complete F1 freeze inspector now governs F1→F2, F1→F3 and F1→F2→F3. F1/F2 source proofs are V2; missing saved upstream freeze evidence is rejected. Existing Runs and Corpus bytes are unchanged.

## Evidence observed / Root cause

The unmodified consumer accepted protected Actor/Head/normalizer/momentum/schema drift on inspect_f1. The old freeze helper also accepted an unknown opt/state/3/alien/count. The new regression originally recorded 15 failures / 19 passes (R6_before); after the fix all 34 pass. These are checksummed synthetic states, not formal-training certification.

## Changes made

inspect_f1 calls source_freeze_evidence exactly once. Direct F1→F3 reuses it. F2 lineage compares protocol, both manifest/run hashes, training evidence and source_freeze. The byte inspector reuses existing state_owners/register_state_schema with the actual saved optimizer config: exact clocks, sanitized slot ownership, protected bytes, schema and finite values. At least one WM parameter may change; per-group counts do not imply six trained groups. Engineering evidence remains unadmitted. transfer_parameters and all losses are unchanged.

## Verification performed

Frozen actual tested pair: Gym `5119f1deb8fc17676b85454c20ce7793e190876d`, Dreamer `cbc1343971786caec979cead3025431690f1591b`.

- R6_transfer_real_cpu: 73 passed, exit 0 (existing F1→F2 / direct F1→F3 / F1→F2→F3 CPU-JAX handoff tests plus the new source matrix). Nonoptimizer digests, fresh optimizer/counters and policy mirror pass.
- R6_first_batch_and_shared: 83 passed, exit 0 (ABI, pins, strict CP, mandatory Gate/Transfer/Action Mask).

Original commands, environment, cwd, source pair, JUnit, failures and exit codes are retained in receipts/R6_*. No failures or skips remain in these selected suites.

## Risks / not verified

Real TRAIN optimization, production Native G3, GPU and TEST reads: NOT_RUN. No claim of trained-model quality. Source proof V2 applies to new stage preparation; old frozen Runs are not upgraded.

## Next recommended step

Merge this independent R6 change, then continue R7 local transactions and exact persistent Prefetch state.
