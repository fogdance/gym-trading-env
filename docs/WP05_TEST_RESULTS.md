# WP05 actual test results

281 distinct selected CPU cases have a final PASS, 0 FAIL, 0 SKIP. This is not the whole project. GPU model update was attempted and failed before initialization; it is not a PASS.

| Collection | Cases | Exit | Seconds |
|---|---:|---:|---:|
| A_shared | 73 | 0 | 85.82 |
| C_F1_legacy | 26 | 0 | 570.33 |
| F_contract_transfer | 27 | 0 | 66.43 |
| F_heads_diagnostics | 23 | 0 | 79.68 |
| E_checkpoint | 8 | 0 | 192.58 |
| F_cli | 5 | 0 | 43.28 |
| F_boundary_review | 3 | 0 | 3.91 |
| F_shared | 38 | 0 | 498.15 |
| F_reader_clock | 83 | 0 | 112.38 |

Exact argv, environment, HEAD/dirty state, per-source hashes and raw output are in each receipts/wp05/<label>.json/.log/.xml. Machine-readable final case inventory: receipts/wp05/F_selected_test_inventory.json.

## Reproduction commands

Run from this Gym worktree with /home/v/miniconda3/envs/dreamerv3/bin/python. The existing tools/wp03_verify.py sets CPU, PYTHONPATH, BLAS/OMP=1 and preserves logs/XML.

```bash
FG_WP_RECEIPTS=wp05 /home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp03_verify.py A_shared_REPEAT tests/entry_v1/test_wp04_admission.py::test_A06_F1_must_not_enable_teacher_or_decision_losses ../dreamerv3/dreamerv3/tests/test_fg_s3_e_quality_course_contract.py ../dreamerv3/embodied/tests/test_fg_s3_gate.py ../dreamerv3/embodied/tests/test_action_mask_formal_guards.py
```

```bash
FG_WP_RECEIPTS=wp05 /home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp03_verify.py C_F1_legacy_REPEAT tests/entry_v1/test_wp04_wm.py tests/entry_v1/test_fg_s3_quality_student_v2.py
```

```bash
FG_WP_RECEIPTS=wp05 /home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp03_verify.py F_contract_transfer_REPEAT tests/entry_v1/test_wp05_contract.py tests/entry_v1/test_wp05_transfer.py
```

```bash
FG_WP_RECEIPTS=wp05 /home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp03_verify.py F_heads_diagnostics_REPEAT tests/entry_v1/test_wp05_updates.py tests/entry_v1/test_wp05_diagnostics.py
```

```bash
FG_WP_RECEIPTS=wp05 /home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp03_verify.py E_checkpoint_REPEAT tests/entry_v1/test_wp05_checkpoint.py
```

```bash
FG_WP_RECEIPTS=wp05 /home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp03_verify.py F_cli_REPEAT tests/entry_v1/test_wp05_cli.py
```

```bash
FG_WP_RECEIPTS=wp05 /home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp03_verify.py F_boundary_review_REPEAT tests/entry_v1/test_wp05_transfer.py::test_B03_source_failures[clock] tests/entry_v1/test_wp05_cli.py::test_F05_real_read_smoke_cannot_prepare_with_synthetic_F1 tests/entry_v1/test_wp05_diagnostics.py::test_D05_L2_and_invalid_Q_values_never_enter_F2_inputs
```

```bash
FG_WP_RECEIPTS=wp05 /home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp03_verify.py F_shared_REPEAT ../dreamerv3/embodied/tests/test_fg_s3_gate.py ../dreamerv3/embodied/tests/test_fg_s3_source_transfer.py ../dreamerv3/embodied/tests/test_action_mask_formal_guards.py ../dreamerv3/embodied/tests/test_fg_full_checkpoint.py ../dreamerv3/embodied/tests/test_checkpoint_retention.py ../dreamerv3/embodied/tests/test_fg_s3c_resume.py
```

```bash
FG_WP_RECEIPTS=wp05 /home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp03_verify.py F_reader_clock_REPEAT tests/entry_v1/test_wp03_reader.py tests/entry_v1/test_wp03_join.py tests/entry_v1/test_wp03_cpu_loss.py tests/entry_v1/test_wp02_macd.py tests/entry_v1/test_s3_market_minute.py
```

## Runtime commands

F_cpu_probe: CLI f2-probe, exit 0, four original optimizer updates on built-in synthetic source; four heads each 5 changed parameter leaves, 285 frozen leaves equal.
F_real_read: CLI f2-read-check on original FG201 TRAIN corpus, exit 0, zero updates and all 349 leaves byte-equal. See F_real_read.json/.log and F_real_read_01/read_check.json.
F_gpu_probe: sandbox attempt exit 1; NVIDIA device nodes not visible. F_gpu_host: actual-host retry exit 1, cuInit CUDA_ERROR_UNKNOWN / No visible GPU devices. No GPU update occurred; driver/root cause not repaired or established.

## Preserved failures and corrections

A_contract: 13 PASS/1 FAIL; constructor mode key was missing from the F2 whitelist. Fixed, A_contract_fix 14 PASS.
C_updates: 9 PASS/1 FAIL; event invalid-placeholder test used float NaN against integer category schema. Corrected event fixture to invalid int index; risk/score NaNs retained. C_mask_fixture_fix PASS.
C allowlist self-review: intersect dynamic activation with positive four-head set; injected WM rule rejected by C_allowlist_review.
D_multiscale: wrong preserved fixture directory; corrected. Short K4/T8 15m zero ablation left realized predictions unchanged. No Encoder/RSSM change; native K32/T64 probe records nonzero Encoder/posterior and all head responses. Original short-window reason was not established as a production defect.

Research: actual trained F1 source, real F2 updates, native FORMAL G3 in this WP, independent holdout calibration, long GPU/18-contract/TEST evaluation NOT_RUN.

F_formal_evidence_final: 1 PASS, exit 0. Formal F1 requires complete update records, initial node, unchanged full schema/finite and actual WM change; engineering sources remain non-admitted. Final HEAD cold restore supplement: receipts/wp05/F_final_recovery.* (postcommit actual run).

F_formal_records_complete: 1 PASS, exit 0. Formal F1 requires complete update records, initial node, unchanged full schema/finite and actual WM change; engineering sources remain non-admitted. Final HEAD cold restore supplement: receipts/wp05/F_final_recovery_v2.* (postcommit actual run).

## Completion-audit target support fix

F_target_support_before.json reproduces real native P1/K90/T8: original view supports two windows, but all four target masks zero after burn-in. New validation rejects unsupported enabled target loss before ready receipt/model creation, without changing sampling or RNG. No lowering of labels/admission/counters.

| Actual repeat | Cases | Exit |
|---|---:|---:|
| F_heads_supported | 31 | 0 |
| F_target_support_fix | 9 | 0 |
| F_target_support_branches | 5 | 0 |
| F_target_support_admission | 1 | 0 |
| F_real_read_supported (12 windows) | command | 0 |

These overlap earlier cases; canonical distinct total is 281. Native read_01 vs read_02 has exactly equal policy reports and parameter hashes (F_target_support_sampling_equivalence.json). Old final-head v1 8 PASS remains historical after the F amendment; actual amended HEAD cold repeat is F_final_recovery_v2.*.
