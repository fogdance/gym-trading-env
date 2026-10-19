# WP08 原始测试与命令索引

以下是实际执行记录，不是全项目测试。后缀 json 记录双仓 HEAD/dirty、命令、CPU 环境及源码哈希；log 为原始输出；xml 列出实际 case。失败尝试保留，后续成功不会删掉失败。最终冻结/继承边界以实施报告为准。

| 记录 | Exit | PASS / FAIL / ERROR / SKIP | 秒 |
| --- | ---: | --- | ---: |
| [A_contract](receipts/wp08/A_contract.json) | 0 | 13 / 0 / 0 / 0 | 1.16 |
| [A_contract_final](receipts/wp08/A_contract_final.json) | 0 | 13 / 0 / 0 / 0 | 0.98 |
| [A_real_source_01](receipts/wp08/A_real_source_01.json) | 1 | 0 / 0 / 0 / 0 | 0.67 |
| [A_real_source_02](receipts/wp08/A_real_source_02.json) | 0 | 0 / 0 / 0 / 0 | 35.36 |
| [A_real_source_03](receipts/wp08/A_real_source_03.json) | 0 | 0 / 0 / 0 / 0 | 35.68 |
| [B_actor_01](receipts/wp08/B_actor_01.json) | 2 | 0 / 0 / 1 / 0 | 0.52 |
| [B_actor_02](receipts/wp08/B_actor_02.json) | 1 | 1 / 1 / 0 / 0 | 30.63 |
| [B_actor_03](receipts/wp08/B_actor_03.json) | 1 | 4 / 1 / 0 / 0 | 68.78 |
| [B_actor_04](receipts/wp08/B_actor_04.json) | 0 | 5 / 0 / 0 / 0 | 65.30 |
| [B_actor_final](receipts/wp08/B_actor_final.json) | 1 | 4 / 1 / 0 / 0 | 64.80 |
| [B_f4_native_control](receipts/wp08/B_f4_native_control.json) | 0 | 3 / 0 / 0 / 0 | 40.81 |
| [B_final](receipts/wp08/B_final.json) | 0 | 5 / 0 / 0 / 0 | 65.85 |
| [Baseline_shared](receipts/wp08/Baseline_shared.json) | 0 | 20 / 0 / 0 / 0 | 141.00 |
| [C_native_01](receipts/wp08/C_native_01.json) | 1 | 0 / 1 / 0 / 0 | 2.78 |
| [C_native_02](receipts/wp08/C_native_02.json) | 1 | 0 / 1 / 0 / 0 | 2.15 |
| [C_native_03](receipts/wp08/C_native_03.json) | 1 | 0 / 1 / 0 / 0 | 2.40 |
| [C_native_04](receipts/wp08/C_native_04.json) | 0 | 1 / 0 / 0 / 0 | 100.33 |
| [D_daily_02](receipts/wp08/D_daily_02.json) | 0 | 1 / 0 / 0 / 0 | 2.93 |
| [D_daily_final](receipts/wp08/D_daily_final.json) | 1 | 0 / 1 / 0 / 0 | 1.66 |
| [D_metrics_01](receipts/wp08/D_metrics_01.json) | 0 | 1 / 0 / 0 / 0 | 3.18 |
| [E_fair_01](receipts/wp08/E_fair_01.json) | 0 | 2 / 0 / 0 / 0 | 641.51 |
| [E_fair_shared_seed](receipts/wp08/E_fair_shared_seed.json) | 0 | 2 / 0 / 0 / 0 | 278.98 |
| [E_seed_identity](receipts/wp08/E_seed_identity.json) | 0 | 1 / 0 / 0 / 0 | 0.90 |
| [F_evidence_01](receipts/wp08/F_evidence_01.json) | 0 | 3 / 0 / 0 / 0 | 1.77 |
| [F_evidence_final](receipts/wp08/F_evidence_final.json) | 0 | 4 / 0 / 0 / 0 | 0.91 |
| [G_cold_01](receipts/wp08/G_cold_01.json) | 0 | 2 / 0 / 0 / 0 | 141.40 |
| [G_strict_boundary](receipts/wp08/G_strict_boundary.json) | 0 | 16 / 0 / 0 / 0 | 33.81 |
| [H_cli_01](receipts/wp08/H_cli_01.json) | 1 | 5 / 1 / 0 / 0 | 7.16 |
| [H_cli_02](receipts/wp08/H_cli_02.json) | 0 | 7 / 0 / 0 / 0 | 98.21 |
| [H_cli_03](receipts/wp08/H_cli_03.json) | 0 | 8 / 0 / 0 / 0 | 146.56 |
| [H_contract_risk_boundaries](receipts/wp08/H_contract_risk_boundaries.json) | 0 | 18 / 0 / 0 / 0 | 3.66 |
| [H_daily_risk_before](receipts/wp08/H_daily_risk_before.json) | 1 | 0 / 1 / 0 / 0 | 5.37 |
| [H_daily_risk_fixed](receipts/wp08/H_daily_risk_fixed.json) | 0 | 9 / 0 / 0 / 0 | 31.26 |
| [H_final_fair](receipts/wp08/H_final_fair.json) | 0 | 3 / 0 / 0 / 0 | 298.89 |
| [H_final_targeted](receipts/wp08/H_final_targeted.json) | 0 | 43 / 0 / 0 / 0 | 343.90 |
| [H_finite_train_readonly](receipts/wp08/H_finite_train_readonly.json) | 0 | 0 / 0 / 0 / 0 | 32.62 |
| [H_native_boundaries](receipts/wp08/H_native_boundaries.json) | 0 | 3 / 0 / 0 / 0 | 203.38 |
| [H_parse_01](receipts/wp08/H_parse_01.json) | 0 | 0 / 0 / 0 / 0 | 0.07 |
| [H_previous_H_correct_path](receipts/wp08/H_previous_H_correct_path.json) | 0 | 0 / 0 / 0 / 0 | 104.83 |
| [H_previous_baseline_path_repro](receipts/wp08/H_previous_baseline_path_repro.json) | 1 | 0 / 0 / 0 / 0 | 0.00 |
| [H_previous_regressions](receipts/wp08/H_previous_regressions.json) | 1 | 0 / 0 / 0 / 0 | 171.82 |
| [H_random_newprocess](receipts/wp08/H_random_newprocess.json) | 0 | 1 / 0 / 0 / 0 | 11.38 |
| [H_risk_window_math](receipts/wp08/H_risk_window_math.json) | 0 | 7 / 0 / 0 / 0 | 12.18 |
| [H_shared](receipts/wp08/H_shared.json) | 0 | 150 / 0 / 0 / 0 | 338.96 |
| [H_source_exact_reopen](receipts/wp08/H_source_exact_reopen.json) | 0 | 0 / 0 / 0 / 0 | 35.78 |
| [H_source_reopen_ready](receipts/wp08/H_source_reopen_ready.json) | 0 | 0 / 0 / 0 / 0 | 43.73 |

## 实际命令与收集清单

执行 cwd 及全部控制环境写在对应 JSON；`FG_WP08_*` 指定独立输出，不可复用旧目录。以下命令来自收据的 argv。

### A_contract (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/A_contract.xml tests/entry_v1/test_wp08_contract.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_contract::test_pure_contract_stable_hash_and_no_io`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[exam_mode-]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[exam_mode-SEALED_TEST_FINAL]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[contracts-value2]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[seed-True]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[random_seeds-value4]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[baselines-value5]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[chunk_steps-0]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[output_dir-/tmp/../run]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[end_day-2026-01-01]`
- PASS: `tests.entry_v1.test_wp08_contract::test_train_corpus_cannot_be_dev_and_empty_is_not_draft`
- PASS: `tests.entry_v1.test_wp08_contract::test_role_refused_before_selected_price_open`
- PASS: `tests.entry_v1.test_wp08_contract::test_symlink_and_forged_validation_receipt_refused`

### A_contract_final (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/A_contract_final.xml tests/entry_v1/test_wp08_contract.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_contract::test_pure_contract_stable_hash_and_no_io`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[exam_mode-]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[exam_mode-SEALED_TEST_FINAL]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[contracts-value2]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[seed-True]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[random_seeds-value4]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[baselines-value5]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[chunk_steps-0]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[output_dir-/tmp/../run]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[end_day-2026-01-01]`
- PASS: `tests.entry_v1.test_wp08_contract::test_train_corpus_cannot_be_dev_and_empty_is_not_draft`
- PASS: `tests.entry_v1.test_wp08_contract::test_role_refused_before_selected_price_open`
- PASS: `tests.entry_v1.test_wp08_contract::test_symlink_and_forged_validation_receipt_refused`

### A_real_source_01 (exit 1)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp08_source_probe.py artifacts/fg_s3_v2/e_quality_wp08/source_probe_01
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

### A_real_source_02 (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp08_source_probe.py artifacts/fg_s3_v2/e_quality_wp08/source_probe_02
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

### A_real_source_03 (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp08_source_probe.py artifacts/fg_s3_v2/e_quality_wp08/source_probe_03
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

### B_actor_01 (exit 2)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/B_actor_01.xml tests/entry_v1/test_wp08_actor.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- ERROR: `::tests.entry_v1.test_wp08_actor`

### B_actor_02 (exit 1)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/B_actor_02.xml tests/entry_v1/test_wp08_actor.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- FAIL: `tests.entry_v1.test_wp08_actor::test_actual_f3_readonly_loading_and_teacher_free_argmax`
- PASS: `tests.entry_v1.test_wp08_actor::test_policy_rng_state_and_frozen_mutation_rejection`

### B_actor_03 (exit 1)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/B_actor_03.xml tests/entry_v1/test_wp08_actor.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- FAIL: `tests.entry_v1.test_wp08_actor::test_actual_f3_readonly_loading_and_teacher_free_argmax`
- PASS: `tests.entry_v1.test_wp08_actor::test_policy_rng_state_and_frozen_mutation_rejection`
- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f4_complete_schema_and_value_preserved[SHADOW]`
- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f4_complete_schema_and_value_preserved[ONLINE_SUPERVISED]`
- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f4_complete_schema_and_value_preserved[ONLINE_RL_CONTROLLED]`

### B_actor_04 (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/B_actor_04.xml tests/entry_v1/test_wp08_actor.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f3_readonly_loading_and_teacher_free_argmax`
- PASS: `tests.entry_v1.test_wp08_actor::test_policy_rng_state_and_frozen_mutation_rejection`
- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f4_complete_schema_and_value_preserved[SHADOW]`
- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f4_complete_schema_and_value_preserved[ONLINE_SUPERVISED]`
- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f4_complete_schema_and_value_preserved[ONLINE_RL_CONTROLLED]`

### B_actor_final (exit 1)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/B_actor_final.xml tests/entry_v1/test_wp08_actor.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f3_readonly_loading_and_teacher_free_argmax`
- PASS: `tests.entry_v1.test_wp08_actor::test_policy_rng_state_and_frozen_mutation_rejection`
- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f4_complete_schema_and_value_preserved[SHADOW]`
- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f4_complete_schema_and_value_preserved[ONLINE_SUPERVISED]`
- FAIL: `tests.entry_v1.test_wp08_actor::test_actual_f4_complete_schema_and_value_preserved[ONLINE_RL_CONTROLLED]`

### B_f4_native_control (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/B_f4_native_control.xml tests/entry_v1/test_wp08_actor.py::test_actual_f4_complete_schema_and_value_preserved
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f4_complete_schema_and_value_preserved[SHADOW]`
- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f4_complete_schema_and_value_preserved[ONLINE_SUPERVISED]`
- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f4_complete_schema_and_value_preserved[ONLINE_RL_CONTROLLED]`

### B_final (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/B_final.xml tests/entry_v1/test_wp08_actor.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f3_readonly_loading_and_teacher_free_argmax`
- PASS: `tests.entry_v1.test_wp08_actor::test_policy_rng_state_and_frozen_mutation_rejection`
- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f4_complete_schema_and_value_preserved[SHADOW]`
- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f4_complete_schema_and_value_preserved[ONLINE_SUPERVISED]`
- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f4_complete_schema_and_value_preserved[ONLINE_RL_CONTROLLED]`

### Baseline_shared (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/Baseline_shared.xml ../dreamerv3/embodied/tests/test_fg_s3_gate.py ../dreamerv3/embodied/tests/test_fg_s3_source_transfer.py ../dreamerv3/embodied/tests/test_action_mask_formal_guards.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `embodied.tests.test_fg_s3_gate::test_threshold_boundaries_float32_bfloat16_and_safe_wait`
- PASS: `embodied.tests.test_fg_s3_gate::test_audited_ast_accepts_only_the_reviewed_transition`
- PASS: `embodied.tests.test_fg_s3_gate::test_gate_only_audit_rejects_market_minute_semantic_change`
- PASS: `embodied.tests.test_fg_s3_gate::test_direct_init_real_runner_uses_twenty_reports_and_regression_stop`
- PASS: `embodied.tests.test_fg_s3_gate::test_real_cross_threshold_ready_source_is_reset_without_other_changes[True-direct_full]`
- PASS: `embodied.tests.test_fg_s3_gate::test_real_cross_threshold_ready_source_is_reset_without_other_changes[True-after_bc_s3a]`
- PASS: `embodied.tests.test_fg_s3_gate::test_same_threshold_transfer_preserves_authenticated_window_and_streak`
- PASS: `embodied.tests.test_fg_s3_gate::test_real_availability_head_raw_mask_and_metrics_share_threshold`
- PASS: `embodied.tests.test_fg_s3_source_transfer::test_full_source_dependency_inventory_includes_stream_and_serializer`
- PASS: `embodied.tests.test_fg_s3_source_transfer::test_online_model_and_environment_counters_have_separate_domains`
- PASS: `embodied.tests.test_fg_s3_source_transfer::test_actual_three_modes_inherit_complete_state_and_reject_foreign`
- PASS: `embodied.tests.test_fg_s3_source_transfer::test_no_inference_source_and_frozen_snapshot`
- PASS: `embodied.tests.test_fg_s3_source_transfer::test_missing_baseline_requires_authenticated_features`
- PASS: `embodied.tests.test_action_mask_formal_guards::test_train_warmup_preserves_loaded_formal_actor_gate`
- PASS: `embodied.tests.test_action_mask_formal_guards::test_train_warmup_keeps_warmup_checkpoint_disabled`
- PASS: `embodied.tests.test_action_mask_formal_guards::test_require_formal_actor_rejects_missing_gate_interface`
- PASS: `embodied.tests.test_action_mask_formal_guards::test_require_formal_actor_rejects_warmup_checkpoint`
- PASS: `embodied.tests.test_action_mask_formal_guards::test_require_formal_actor_accepts_formal_checkpoint`
- PASS: `embodied.tests.test_action_mask_formal_guards::test_eval_only_rejects_warmup_checkpoint`
- PASS: `embodied.tests.test_action_mask_formal_guards::test_live_trading_rejects_warmup_checkpoint`

### C_native_01 (exit 1)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/C_native_01.xml tests/entry_v1/test_wp08_session.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- FAIL: `tests.entry_v1.test_wp08_session::test_native_actor_1024_multiday_trace_and_account`

### C_native_02 (exit 1)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/C_native_02.xml tests/entry_v1/test_wp08_session.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- FAIL: `tests.entry_v1.test_wp08_session::test_native_actor_1024_multiday_trace_and_account`

### C_native_03 (exit 1)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/C_native_03.xml tests/entry_v1/test_wp08_session.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- FAIL: `tests.entry_v1.test_wp08_session::test_native_actor_1024_multiday_trace_and_account`

### C_native_04 (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/C_native_04.xml tests/entry_v1/test_wp08_session.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_session::test_native_actor_1024_multiday_trace_and_account`

### D_daily_02 (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/D_daily_02.xml tests/entry_v1/test_wp08_metrics.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_metrics::test_native_trade_denominators_risk_fees_and_daily_accounts`

### D_daily_final (exit 1)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/D_daily_final.xml tests/entry_v1/test_wp08_metrics.py --override-ini addopts=
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- FAIL: `tests.entry_v1.test_wp08_metrics::test_native_trade_denominators_risk_fees_and_daily_accounts`

### D_metrics_01 (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/D_metrics_01.xml tests/entry_v1/test_wp08_metrics.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_metrics::test_native_trade_denominators_risk_fees_and_daily_accounts`

### E_fair_01 (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/E_fair_01.xml tests/entry_v1/test_wp08_compare.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_compare::test_baseline_rng_delay_visible_history_and_no_oracle`
- PASS: `tests.entry_v1.test_wp08_compare::test_same_source_full_interval_baselines_and_f3_f4`

### E_fair_shared_seed (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/E_fair_shared_seed.xml tests/entry_v1/test_wp08_compare.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_compare::test_baseline_rng_delay_visible_history_and_no_oracle`
- PASS: `tests.entry_v1.test_wp08_compare::test_same_source_full_interval_baselines_and_f3_f4`

### E_seed_identity (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/E_seed_identity.xml tests/entry_v1/test_wp08_compare.py::test_shared_eval_seed_preserves_original_model_seed_and_parameters
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_compare::test_shared_eval_seed_preserves_original_model_seed_and_parameters`

### F_evidence_01 (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/F_evidence_01.xml tests/entry_v1/test_wp08_evidence.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_evidence::test_all_wait_and_too_few_trades_never_become_quality_go`
- PASS: `tests.entry_v1.test_wp08_evidence::test_unregistered_and_synthetic_outcomes_not_dev_or_generalization`
- PASS: `tests.entry_v1.test_wp08_evidence::test_registered_joint_checks_and_market_support_are_both_required`

### F_evidence_final (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/F_evidence_final.xml tests/entry_v1/test_wp08_evidence.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_evidence::test_all_wait_and_too_few_trades_never_become_quality_go`
- PASS: `tests.entry_v1.test_wp08_evidence::test_unregistered_and_synthetic_outcomes_not_dev_or_generalization`
- PASS: `tests.entry_v1.test_wp08_evidence::test_registered_joint_checks_and_market_support_are_both_required`
- PASS: `tests.entry_v1.test_wp08_evidence::test_native_comparison_has_engineering_evidence_without_research_claim`

### G_cold_01 (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/G_cold_01.xml tests/entry_v1/test_wp08_checkpoint.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_checkpoint::test_1024_equals_333_new_process_691_exactly`
- PASS: `tests.entry_v1.test_wp08_checkpoint::test_strict_latest_corrupt_chunk_and_source_change_refuse`

### G_strict_boundary (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/G_strict_boundary.xml tests/entry_v1/test_wp08_checkpoint.py::test_strict_latest_corrupt_chunk_and_source_change_refuse tests/entry_v1/test_wp08_contract.py tests/entry_v1/test_wp08_actor.py::test_policy_rng_state_and_frozen_mutation_rejection
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_checkpoint::test_strict_latest_corrupt_chunk_and_source_change_refuse`
- PASS: `tests.entry_v1.test_wp08_contract::test_pure_contract_stable_hash_and_no_io`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[exam_mode-]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[exam_mode-SEALED_TEST_FINAL]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[contracts-value2]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[seed-True]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[random_seeds-value4]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[baselines-value5]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[chunk_steps-0]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[output_dir-/tmp/../run]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[end_day-2026-01-01]`
- PASS: `tests.entry_v1.test_wp08_contract::test_train_corpus_cannot_be_dev_and_empty_is_not_draft`
- PASS: `tests.entry_v1.test_wp08_contract::test_role_refused_before_selected_price_open`
- PASS: `tests.entry_v1.test_wp08_contract::test_symlink_and_forged_validation_receipt_refused`
- PASS: `tests.entry_v1.test_wp08_contract::test_json_cannot_mint_native_validation_admission`
- PASS: `tests.entry_v1.test_wp08_actor::test_policy_rng_state_and_frozen_mutation_rejection`

### H_cli_01 (exit 1)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/H_cli_01.xml tests/entry_v1/test_wp08_cli.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_cli::test_cli_refusal_no_output_or_training_side_effect[test]`
- PASS: `tests.entry_v1.test_wp08_cli::test_cli_refusal_no_output_or_training_side_effect[empty]`
- PASS: `tests.entry_v1.test_wp08_cli::test_cli_refusal_no_output_or_training_side_effect[dev_train]`
- PASS: `tests.entry_v1.test_wp08_cli::test_cli_refusal_no_output_or_training_side_effect[training_flag]`
- PASS: `tests.entry_v1.test_wp08_cli::test_cli_refusal_no_output_or_training_side_effect[gpu_flag]`
- FAIL: `tests.entry_v1.test_wp08_cli::test_synthetic_cli_plan_pause_resume_verify_compare_report`

### H_cli_02 (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/H_cli_02.xml tests/entry_v1/test_wp08_cli.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_cli::test_existing_eval_path_keeps_default_data_overwrite_guard`
- PASS: `tests.entry_v1.test_wp08_cli::test_cli_refusal_no_output_or_training_side_effect[test]`
- PASS: `tests.entry_v1.test_wp08_cli::test_cli_refusal_no_output_or_training_side_effect[empty]`
- PASS: `tests.entry_v1.test_wp08_cli::test_cli_refusal_no_output_or_training_side_effect[dev_train]`
- PASS: `tests.entry_v1.test_wp08_cli::test_cli_refusal_no_output_or_training_side_effect[training_flag]`
- PASS: `tests.entry_v1.test_wp08_cli::test_cli_refusal_no_output_or_training_side_effect[gpu_flag]`
- PASS: `tests.entry_v1.test_wp08_cli::test_synthetic_cli_plan_pause_resume_verify_compare_report`

### H_cli_03 (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/H_cli_03.xml tests/entry_v1/test_wp08_cli.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_cli::test_existing_eval_path_keeps_default_data_overwrite_guard`
- PASS: `tests.entry_v1.test_wp08_cli::test_formal_dev_without_trained_actor_nonzero_and_no_outputs`
- PASS: `tests.entry_v1.test_wp08_cli::test_cli_refusal_no_output_or_training_side_effect[test]`
- PASS: `tests.entry_v1.test_wp08_cli::test_cli_refusal_no_output_or_training_side_effect[empty]`
- PASS: `tests.entry_v1.test_wp08_cli::test_cli_refusal_no_output_or_training_side_effect[dev_train]`
- PASS: `tests.entry_v1.test_wp08_cli::test_cli_refusal_no_output_or_training_side_effect[training_flag]`
- PASS: `tests.entry_v1.test_wp08_cli::test_cli_refusal_no_output_or_training_side_effect[gpu_flag]`
- PASS: `tests.entry_v1.test_wp08_cli::test_synthetic_cli_plan_pause_resume_verify_compare_report`

### H_contract_risk_boundaries (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/H_contract_risk_boundaries.xml tests/entry_v1/test_wp08_contract.py tests/entry_v1/test_wp08_metrics.py tests/entry_v1/test_wp08_actor.py::test_corrupt_actual_source_checkpoint_rejected_before_model_construction
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_contract::test_pure_contract_stable_hash_and_no_io`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[exam_mode-]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[exam_mode-SEALED_TEST_FINAL]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[contracts-value2]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[seed-True]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[random_seeds-value4]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[baselines-value5]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[chunk_steps-0]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[output_dir-/tmp/../run]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[end_day-2026-01-01]`
- PASS: `tests.entry_v1.test_wp08_contract::test_train_corpus_cannot_be_dev_and_empty_is_not_draft`
- PASS: `tests.entry_v1.test_wp08_contract::test_role_refused_before_selected_price_open`
- PASS: `tests.entry_v1.test_wp08_contract::test_symlink_and_forged_validation_receipt_refused`
- PASS: `tests.entry_v1.test_wp08_contract::test_json_cannot_mint_native_validation_admission`
- PASS: `tests.entry_v1.test_wp08_contract::test_actual_training_corpus_manifest_is_not_a_dev_market_source`
- PASS: `tests.entry_v1.test_wp08_contract::test_disjoint_contract_windows_are_explicit_and_inside_study`
- PASS: `tests.entry_v1.test_wp08_metrics::test_native_trade_denominators_risk_fees_and_daily_accounts`
- PASS: `tests.entry_v1.test_wp08_actor::test_corrupt_actual_source_checkpoint_rejected_before_model_construction`

### H_daily_risk_before (exit 1)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/H_daily_risk_before.xml tests/entry_v1/test_wp08_metric_windows.py::test_censored_risk_does_not_invent_negative_sl_or_tp_label
```

```bash
env FG_WP_RECEIPTS=wp08 PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- FAIL: `tests.entry_v1.test_wp08_metric_windows::test_censored_risk_does_not_invent_negative_sl_or_tp_label`

### H_daily_risk_fixed (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/H_daily_risk_fixed.xml tests/entry_v1/test_wp08_metric_windows.py tests/entry_v1/test_wp08_metrics.py
```

```bash
env FG_WP_RECEIPTS=wp08 FG_WP08_D_ROOT=artifacts/fg_s3_v2/e_quality_wp08/Final_D_01 PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_metric_windows::test_native_tp_market_minute_window_boundaries[5-expected0]`
- PASS: `tests.entry_v1.test_wp08_metric_windows::test_native_tp_market_minute_window_boundaries[15-expected1]`
- PASS: `tests.entry_v1.test_wp08_metric_windows::test_native_tp_market_minute_window_boundaries[30-expected2]`
- PASS: `tests.entry_v1.test_wp08_metric_windows::test_native_tp_market_minute_window_boundaries[60-expected3]`
- PASS: `tests.entry_v1.test_wp08_metric_windows::test_native_tp_market_minute_window_boundaries[61-expected4]`
- PASS: `tests.entry_v1.test_wp08_metric_windows::test_adverse_tp_is_not_sl_and_exit_bar_not_underwater`
- PASS: `tests.entry_v1.test_wp08_metric_windows::test_censored_risk_does_not_invent_negative_sl_or_tp_label`
- PASS: `tests.entry_v1.test_wp08_metric_windows::test_tp_crosses_official_day_without_reset_or_forced_exit`
- PASS: `tests.entry_v1.test_wp08_metrics::test_native_trade_denominators_risk_fees_and_daily_accounts`

### H_final_fair (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/H_final_fair.xml tests/entry_v1/test_wp08_compare.py
```

```bash
env FG_WP_RECEIPTS=wp08 FG_WP08_E_ROOT=artifacts/fg_s3_v2/e_quality_wp08/Final_E_01 PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_compare::test_baseline_rng_delay_visible_history_and_no_oracle`
- PASS: `tests.entry_v1.test_wp08_compare::test_same_source_full_interval_baselines_and_f3_f4`
- PASS: `tests.entry_v1.test_wp08_compare::test_shared_eval_seed_preserves_original_model_seed_and_parameters`

### H_final_targeted (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/H_final_targeted.xml tests/entry_v1/test_wp08_contract.py tests/entry_v1/test_wp08_actor.py tests/entry_v1/test_wp08_metric_windows.py tests/entry_v1/test_wp08_evidence.py tests/entry_v1/test_wp08_cli.py tests/entry_v1/test_wp08_boundaries.py::test_contracts_start_separate_accounts_carries_and_rng
```

```bash
env FG_WP_RECEIPTS=wp08 FG_WP08_H_ROOT=artifacts/fg_s3_v2/e_quality_wp08/Final_H_01 FG_WP08_CONTRACT_ROOT=artifacts/fg_s3_v2/e_quality_wp08/Final_contracts_01 PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_contract::test_pure_contract_stable_hash_and_no_io`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[exam_mode-]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[exam_mode-SEALED_TEST_FINAL]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[contracts-value2]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[seed-True]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[random_seeds-value4]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[baselines-value5]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[chunk_steps-0]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[output_dir-/tmp/../run]`
- PASS: `tests.entry_v1.test_wp08_contract::test_invalid_contract_no_output[end_day-2026-01-01]`
- PASS: `tests.entry_v1.test_wp08_contract::test_train_corpus_cannot_be_dev_and_empty_is_not_draft`
- PASS: `tests.entry_v1.test_wp08_contract::test_role_refused_before_selected_price_open`
- PASS: `tests.entry_v1.test_wp08_contract::test_symlink_and_forged_validation_receipt_refused`
- PASS: `tests.entry_v1.test_wp08_contract::test_json_cannot_mint_native_validation_admission`
- PASS: `tests.entry_v1.test_wp08_contract::test_actual_training_corpus_manifest_is_not_a_dev_market_source`
- PASS: `tests.entry_v1.test_wp08_contract::test_disjoint_contract_windows_are_explicit_and_inside_study`
- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f3_readonly_loading_and_teacher_free_argmax`
- PASS: `tests.entry_v1.test_wp08_actor::test_policy_rng_state_and_frozen_mutation_rejection`
- PASS: `tests.entry_v1.test_wp08_actor::test_corrupt_actual_source_checkpoint_rejected_before_model_construction`
- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f4_complete_schema_and_value_preserved[SHADOW]`
- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f4_complete_schema_and_value_preserved[ONLINE_SUPERVISED]`
- PASS: `tests.entry_v1.test_wp08_actor::test_actual_f4_complete_schema_and_value_preserved[ONLINE_RL_CONTROLLED]`
- PASS: `tests.entry_v1.test_wp08_metric_windows::test_native_tp_market_minute_window_boundaries[5-expected0]`
- PASS: `tests.entry_v1.test_wp08_metric_windows::test_native_tp_market_minute_window_boundaries[15-expected1]`
- PASS: `tests.entry_v1.test_wp08_metric_windows::test_native_tp_market_minute_window_boundaries[30-expected2]`
- PASS: `tests.entry_v1.test_wp08_metric_windows::test_native_tp_market_minute_window_boundaries[60-expected3]`
- PASS: `tests.entry_v1.test_wp08_metric_windows::test_native_tp_market_minute_window_boundaries[61-expected4]`
- PASS: `tests.entry_v1.test_wp08_metric_windows::test_adverse_tp_is_not_sl_and_exit_bar_not_underwater`
- PASS: `tests.entry_v1.test_wp08_metric_windows::test_censored_risk_does_not_invent_negative_sl_or_tp_label`
- PASS: `tests.entry_v1.test_wp08_evidence::test_all_wait_and_too_few_trades_never_become_quality_go`
- PASS: `tests.entry_v1.test_wp08_evidence::test_unregistered_and_synthetic_outcomes_not_dev_or_generalization`
- PASS: `tests.entry_v1.test_wp08_evidence::test_registered_joint_checks_and_market_support_are_both_required`
- PASS: `tests.entry_v1.test_wp08_evidence::test_native_comparison_has_engineering_evidence_without_research_claim`
- PASS: `tests.entry_v1.test_wp08_cli::test_worker_refuses_undeclared_job_before_source_io`
- PASS: `tests.entry_v1.test_wp08_cli::test_existing_eval_path_keeps_default_data_overwrite_guard`
- PASS: `tests.entry_v1.test_wp08_cli::test_formal_dev_without_trained_actor_nonzero_and_no_outputs`
- PASS: `tests.entry_v1.test_wp08_cli::test_cli_refusal_no_output_or_training_side_effect[test]`
- PASS: `tests.entry_v1.test_wp08_cli::test_cli_refusal_no_output_or_training_side_effect[empty]`
- PASS: `tests.entry_v1.test_wp08_cli::test_cli_refusal_no_output_or_training_side_effect[dev_train]`
- PASS: `tests.entry_v1.test_wp08_cli::test_cli_refusal_no_output_or_training_side_effect[training_flag]`
- PASS: `tests.entry_v1.test_wp08_cli::test_cli_refusal_no_output_or_training_side_effect[gpu_flag]`
- PASS: `tests.entry_v1.test_wp08_cli::test_synthetic_cli_plan_pause_resume_verify_compare_report`
- PASS: `tests.entry_v1.test_wp08_boundaries::test_contracts_start_separate_accounts_carries_and_rng`

### H_finite_train_readonly (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp08_train_probe.py artifacts/fg_s3_v2/e_quality_wp08/train_probe_01
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

### H_native_boundaries (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/H_native_boundaries.xml tests/entry_v1/test_wp08_boundaries.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_boundaries::test_native_main_roll_and_unfinished_trade_are_distinct`
- PASS: `tests.entry_v1.test_wp08_boundaries::test_contracts_start_separate_accounts_carries_and_rng`
- PASS: `tests.entry_v1.test_wp08_boundaries::test_declared_actor_actions_reproduce_and_legal_action_tamper_is_rejected`

### H_parse_01 (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py --help
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

### H_previous_H_correct_path (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp08_regress.py H_previous_H_02 online_h
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

### H_previous_baseline_path_repro (exit 1)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py online-probe --output-dir /home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp08/baseline_wrong_probe_path --execution-mode ONLINE_SUPERVISED
```

### H_previous_regressions (exit 1)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp08_regress.py H_previous_01
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

### H_random_newprocess (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/H_random_newprocess.xml tests/entry_v1/test_wp08_checkpoint.py::test_random_baseline_rng_and_driver_restore_in_new_process
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_checkpoint::test_random_baseline_rng_and_driver_restore_in_new_process`

### H_risk_window_math (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/H_risk_window_math.xml tests/entry_v1/test_wp08_metric_windows.py
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `tests.entry_v1.test_wp08_metric_windows::test_native_tp_market_minute_window_boundaries[5-expected0]`
- PASS: `tests.entry_v1.test_wp08_metric_windows::test_native_tp_market_minute_window_boundaries[15-expected1]`
- PASS: `tests.entry_v1.test_wp08_metric_windows::test_native_tp_market_minute_window_boundaries[30-expected2]`
- PASS: `tests.entry_v1.test_wp08_metric_windows::test_native_tp_market_minute_window_boundaries[60-expected3]`
- PASS: `tests.entry_v1.test_wp08_metric_windows::test_native_tp_market_minute_window_boundaries[61-expected4]`
- PASS: `tests.entry_v1.test_wp08_metric_windows::test_adverse_tp_is_not_sl_and_exit_bar_not_underwater`
- PASS: `tests.entry_v1.test_wp08_metric_windows::test_censored_risk_does_not_invent_negative_sl_or_tp_label`

### H_shared (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/gym-trading-env/docs/receipts/wp08/H_shared.xml ../dreamerv3/embodied/tests/test_fg_s3_gate.py ../dreamerv3/embodied/tests/test_fg_s3_source_transfer.py ../dreamerv3/embodied/tests/test_action_mask_formal_guards.py ../dreamerv3/embodied/tests/test_fg_full_checkpoint.py ../dreamerv3/embodied/tests/test_driver_account_checkpoint.py tests/entry_v1/test_s3_market_minute.py tests/entry_v1/test_macd_dated_tail.py tests/entry_v1/test_wp02_macd.py tests/entry_v1/test_fg_s3_e_quality_course_preflight.py tests/entry_v1/test_wp03_reader.py tests/entry_v1/test_data_multi_f3.py tests/entry_v1/test_data_multi.py::test_incompatible_model_contract_and_changed_adapter_bytes_rejected
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

- PASS: `::test_threshold_boundaries_float32_bfloat16_and_safe_wait`
- PASS: `::test_audited_ast_accepts_only_the_reviewed_transition`
- PASS: `::test_gate_only_audit_rejects_market_minute_semantic_change`
- PASS: `::test_direct_init_real_runner_uses_twenty_reports_and_regression_stop`
- PASS: `::test_real_cross_threshold_ready_source_is_reset_without_other_changes[True-direct_full]`
- PASS: `::test_real_cross_threshold_ready_source_is_reset_without_other_changes[True-after_bc_s3a]`
- PASS: `::test_same_threshold_transfer_preserves_authenticated_window_and_streak`
- PASS: `::test_real_availability_head_raw_mask_and_metrics_share_threshold`
- PASS: `::test_full_source_dependency_inventory_includes_stream_and_serializer`
- PASS: `::test_online_model_and_environment_counters_have_separate_domains`
- PASS: `::test_actual_three_modes_inherit_complete_state_and_reject_foreign`
- PASS: `::test_no_inference_source_and_frozen_snapshot`
- PASS: `::test_missing_baseline_requires_authenticated_features`
- PASS: `::test_train_warmup_preserves_loaded_formal_actor_gate`
- PASS: `::test_train_warmup_keeps_warmup_checkpoint_disabled`
- PASS: `::test_require_formal_actor_rejects_missing_gate_interface`
- PASS: `::test_require_formal_actor_rejects_warmup_checkpoint`
- PASS: `::test_require_formal_actor_accepts_formal_checkpoint`
- PASS: `::test_eval_only_rejects_warmup_checkpoint`
- PASS: `::test_live_trading_rejects_warmup_checkpoint`
- PASS: `::test_replay_restores_partial_chunks_online_queue_rng_and_updates`
- PASS: `::test_prefetch_snapshot_restores_pending_batches_and_consecutive_context`
- PASS: `::test_runtime_restores_ratio_carry_aggregates_and_gate_samples`
- PASS: `::test_exact_schedule_resume_and_overshoot_rejection`
- PASS: `::test_atomic_full_bundle_retention_identity_and_corrupt_recovery`
- PASS: `::test_failed_save_preserves_valid_points_and_final_is_protected`
- PASS: `::test_train_loop_full_resume_matches_uninterrupted_without_neural_training[False]`
- PASS: `::test_train_loop_full_resume_matches_uninterrupted_without_neural_training[True]`
- PASS: `::test_deterministic_prefetch_propagates_worker_failure_without_waiting`
- PASS: `::test_fg_guard_rejects_full_checkpoint_optout_before_model_init`
- PASS: `::test_driver_explicitly_fetches_jax_carry_under_transfer_guard`
- PASS: `::test_driver_resumes_pending_actions_and_account[False]`
- PASS: `::test_driver_resumes_pending_actions_and_account[True]`
- PASS: `tests.entry_v1.test_s3_market_minute::test_inclusive_fill_reward_age_and_sparse_oracle[1]`
- PASS: `tests.entry_v1.test_s3_market_minute::test_inclusive_fill_reward_age_and_sparse_oracle[2]`
- PASS: `tests.entry_v1.test_s3_market_minute::test_inclusive_fill_reward_age_and_sparse_oracle[5]`
- PASS: `tests.entry_v1.test_s3_market_minute::test_inclusive_fill_reward_age_and_sparse_oracle[6]`
- PASS: `tests.entry_v1.test_s3_market_minute::test_inclusive_fill_reward_age_and_sparse_oracle[15]`
- PASS: `tests.entry_v1.test_s3_market_minute::test_inclusive_fill_reward_age_and_sparse_oracle[16]`
- PASS: `tests.entry_v1.test_s3_market_minute::test_inclusive_fill_reward_age_and_sparse_oracle[60]`
- PASS: `tests.entry_v1.test_s3_market_minute::test_inclusive_fill_reward_age_and_sparse_oracle[61]`
- PASS: `tests.entry_v1.test_s3_market_minute::test_closed_hours_metadata_zero_volume_and_chunk_restore[lunch]`
- PASS: `tests.entry_v1.test_s3_market_minute::test_closed_hours_metadata_zero_volume_and_chunk_restore[overnight]`
- PASS: `tests.entry_v1.test_s3_market_minute::test_zero_volume_real_bar_counts_and_censored_or_rejected_is_not_tp`
- PASS: `tests.entry_v1.test_s3_market_minute::test_first_adverse_market_bar_boundaries[5--0.5]`
- PASS: `tests.entry_v1.test_s3_market_minute::test_first_adverse_market_bar_boundaries[6--0.2]`
- PASS: `tests.entry_v1.test_s3_market_minute::test_first_adverse_market_bar_boundaries[15--0.2]`
- PASS: `tests.entry_v1.test_s3_market_minute::test_first_adverse_market_bar_boundaries[16-0.0]`
- PASS: `tests.entry_v1.test_s3_market_minute::test_metadata_discount_matches_independent_env_enumeration_and_s3b`
- PASS: `tests.entry_v1.test_s3_market_minute::test_cross_semantic_snapshot_rejected_before_model_use`
- PASS: `tests.entry_v1.test_s3_market_minute::test_model_return_uses_feedback_aligned_market_discounts`
- PASS: `tests.entry_v1.test_macd_dated_tail::test_certified_short_interval_does_not_commit_its_partial_close`
- PASS: `tests.entry_v1.test_macd_dated_tail::test_missing_last_minute_cannot_be_disguised_as_a_session_break`
- PASS: `tests.entry_v1.test_macd_dated_tail::test_previous_schema_snapshot_cannot_resume_with_new_tail_semantics`
- PASS: `tests.entry_v1.test_macd_dated_tail::test_zero_duration_interval_end_is_a_marker_not_a_fifth_close`
- PASS: `tests.entry_v1.test_wp02_macd::test_A01_partial_from_committed_ema_and_hist_without_double`
- PASS: `tests.entry_v1.test_wp02_macd::test_A02_warmup_and_price_scale`
- PASS: `tests.entry_v1.test_wp02_macd::test_A03_stream_restore_and_prefix_invariance`
- PASS: `tests.entry_v1.test_wp02_macd::test_A04_bad_minutes_fail_closed[duplicate]`
- PASS: `tests.entry_v1.test_wp02_macd::test_A04_bad_minutes_fail_closed[missing]`
- PASS: `tests.entry_v1.test_wp02_macd::test_A04_bad_minutes_fail_closed[order]`
- PASS: `tests.entry_v1.test_wp02_macd::test_A04_bad_minutes_fail_closed[late]`
- PASS: `tests.entry_v1.test_wp02_macd::test_A04_bad_minutes_fail_closed[nonfinite]`
- PASS: `tests.entry_v1.test_wp02_macd::test_A04_bad_minutes_fail_closed[negative]`
- PASS: `tests.entry_v1.test_wp02_macd::test_A05_session_break_carries_ema_but_cannot_join_incomplete_buckets`
- PASS: `tests.entry_v1.test_wp02_macd::test_A06_native_snapshot_repeated_queries_cold_restore_and_macd_hash`
- PASS: `tests.entry_v1.test_wp02_macd::test_A07_course_v1_and_sidecar_cannot_masquerade_as_v2`
- PASS: `tests.entry_v1.test_wp02_macd::test_A08_actual_calendar_snapshot_future_ohlc_and_arrivals_do_not_change_prefix`
- PASS: `tests.entry_v1.test_wp02_macd::test_A09_native_dated_calendar_midnight_weekend_holiday_and_lunch`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_01_12_native_draft_calls_reward_validator_and_never_loads_markets`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_13_cli_repeated_drafts_have_identical_non_authenticated_receipts`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_16_cli_disables_bytecode_without_external_python_flags`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_13_native_hash_changes_for_order_seed_or_source_hash`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_12_formal_complete_matching_metadata_still_fails_without_certification`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_08_declared_oracle_evidence_never_certifies_a_draft`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_02_to_12_cli_negative_status_exit_files_and_no_side_effects[composition]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_02_to_12_cli_negative_status_exit_files_and_no_side_effects[old_reward]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_02_to_12_cli_negative_status_exit_files_and_no_side_effects[candidate]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_02_to_12_cli_negative_status_exit_files_and_no_side_effects[minute_shape]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_02_to_12_cli_negative_status_exit_files_and_no_side_effects[context_shape]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_02_to_12_cli_negative_status_exit_files_and_no_side_effects[clock]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_02_to_12_cli_negative_status_exit_files_and_no_side_effects[lots]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_02_to_12_cli_negative_status_exit_files_and_no_side_effects[test_fold]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_02_to_12_cli_negative_status_exit_files_and_no_side_effects[leak]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_02_to_12_cli_negative_status_exit_files_and_no_side_effects[fake_oracle]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_02_to_12_cli_negative_status_exit_files_and_no_side_effects[unknown_behavior]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_02_to_12_cli_negative_status_exit_files_and_no_side_effects[blank_mode]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_02_to_12_cli_negative_status_exit_files_and_no_side_effects[blank_source]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_02_to_12_cli_negative_status_exit_files_and_no_side_effects[empty_sources]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_02_to_12_cli_negative_status_exit_files_and_no_side_effects[implicit_formal]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_02_to_12_cli_negative_status_exit_files_and_no_side_effects[forged_ready]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_02_to_12_cli_negative_status_exit_files_and_no_side_effects[forged_matrix]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_12_16_missing_or_invalid_json_never_becomes_draft[missing_mode]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_12_16_missing_or_invalid_json_never_becomes_draft[missing_source]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_12_16_missing_or_invalid_json_never_becomes_draft[duplicate_json]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_12_16_missing_or_invalid_json_never_becomes_draft[malformed_json]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_06_08_formal_source_errors_are_nonzero_and_write_nothing[missing]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_06_08_formal_source_errors_are_nonzero_and_write_nothing[hash]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_06_08_formal_source_errors_are_nonzero_and_write_nothing[profile]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_06_08_formal_source_errors_are_nonzero_and_write_nothing[test_source]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_06_08_formal_source_errors_are_nonzero_and_write_nothing[teacher_reward]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_06_08_formal_source_errors_are_nonzero_and_write_nothing[forged]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_06_08_formal_source_errors_are_nonzero_and_write_nothing[malformed_pool]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_06_08_formal_source_errors_are_nonzero_and_write_nothing[false_eligibility]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_16_report_paths_cannot_overwrite_or_touch_protected_outputs[existing]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_16_report_paths_cannot_overwrite_or_touch_protected_outputs[reserved]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_16_report_paths_cannot_overwrite_or_touch_protected_outputs[formal_root]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_16_report_paths_cannot_overwrite_or_touch_protected_outputs[symlink_parent]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_16_report_paths_cannot_overwrite_or_touch_protected_outputs[source_directory]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_native_validator_failures_are_not_suppressed[reward]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_native_validator_failures_are_not_suppressed[observation]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_native_validator_failures_are_not_suppressed[score]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_native_validator_failures_are_not_suppressed[public_guard]`
- PASS: `tests.entry_v1.test_fg_s3_e_quality_course_preflight::test_WP01_14_real_legacy_quality_and_e_configurations_keep_their_contracts`
- PASS: `tests.entry_v1.test_wp03_reader::test_C11_writer_and_reader_use_only_V3`
- PASS: `tests.entry_v1.test_wp03_reader::test_C12_old_or_forged_index_versions_refused_without_writes[receipt_v2]`
- PASS: `tests.entry_v1.test_wp03_reader::test_C12_old_or_forged_index_versions_refused_without_writes[missing]`
- PASS: `tests.entry_v1.test_wp03_reader::test_C12_old_or_forged_index_versions_refused_without_writes[database_v2]`
- PASS: `tests.entry_v1.test_wp03_reader::test_C01_native_contiguous_episode_safe_windows_all_stages[WM_NATURAL]`
- PASS: `tests.entry_v1.test_wp03_reader::test_C01_native_contiguous_episode_safe_windows_all_stages[HEADS_VALID]`
- PASS: `tests.entry_v1.test_wp03_reader::test_C01_native_contiguous_episode_safe_windows_all_stages[ACTOR_STRICT]`
- PASS: `tests.entry_v1.test_wp03_reader::test_C01_native_contiguous_episode_safe_windows_all_stages[ACTOR_QUALITY_PAIR]`
- PASS: `tests.entry_v1.test_wp03_reader::test_C02_seed_and_complete_rng_state_exact_next_batch[NATURAL]`
- PASS: `tests.entry_v1.test_wp03_reader::test_C02_seed_and_complete_rng_state_exact_next_batch[STRATIFIED]`
- PASS: `tests.entry_v1.test_wp03_reader::test_C02_seed_and_complete_rng_state_exact_next_batch[MIXED]`
- PASS: `tests.entry_v1.test_wp03_reader::test_C03_missing_class_is_support_failure_without_exposure_or_rng_change`
- PASS: `tests.entry_v1.test_wp03_reader::test_C04_restore_rejects_source_index_stage_or_sampling_change`
- PASS: `tests.entry_v1.test_wp03_reader::test_C05_same_price_different_account_and_reset_never_form_pair`
- PASS: `tests.entry_v1.test_wp03_reader::test_C06_original_macd_bytes_and_bounded_cache_no_parallel_replay`
- PASS: `tests.entry_v1.test_wp03_reader::test_C07_relocation_preserves_reader_restore_without_old_absolute_root`
- PASS: `tests.entry_v1.test_wp03_reader::test_C08_open_reader_detects_changed_sidecar_or_chunk`
- PASS: `tests.entry_v1.test_wp03_reader::test_C09_mixed_requires_explicit_weights_and_never_interprets_empty_mode`
- PASS: `tests.entry_v1.test_wp03_reader::test_C10_window_crosses_original_native_chunk_files_without_crossing_episode`
- PASS: `tests.entry_v1.test_data_multi_f3::test_f3_roles_native_bytes_provenance_and_exact_state[None]`
- PASS: `tests.entry_v1.test_data_multi_f3::test_f3_roles_native_bytes_provenance_and_exact_state[NATURAL]`
- PASS: `tests.entry_v1.test_data_multi_f3::test_f3_roles_native_bytes_provenance_and_exact_state[STRATIFIED]`
- PASS: `tests.entry_v1.test_data_multi_f3::test_f3_roles_native_bytes_provenance_and_exact_state[MIXED]`
- PASS: `tests.entry_v1.test_data_multi_f3::test_auxiliary_rng_and_objective_preserve_natural_stream`
- PASS: `tests.entry_v1.test_data_multi_f3::test_auxiliary_failure_and_bad_restore_roll_back_whole_group`
- PASS: `tests.entry_v1.test_data_multi_f3::test_f3_role_identity_changes_refuse_restore[auxiliary]`
- PASS: `tests.entry_v1.test_data_multi_f3::test_f3_role_identity_changes_refuse_restore[spec]`
- PASS: `tests.entry_v1.test_data_multi_f3::test_f3_role_identity_changes_refuse_restore[geometry]`
- PASS: `tests.entry_v1.test_data_multi_f3::test_f3_missing_support_and_canonical_label_attack`
- PASS: `tests.entry_v1.test_data_multi_f3::test_f3_changed_external_source_invalidates_group`
- PASS: `tests.entry_v1.test_data_multi_f3::test_f3_cli_and_original_checkpoint_cold_processes[None]`
- PASS: `tests.entry_v1.test_data_multi_f3::test_f3_cli_and_original_checkpoint_cold_processes[MIXED]`
- PASS: `tests.entry_v1.test_data_multi_f3::test_f3_cli_requires_explicit_geometry_and_natural_specs`
- PASS: `tests.entry_v1.test_data_multi_f3::test_f3_natural_statistics_preserve_source_and_global_denominators`
- PASS: `tests.entry_v1.test_data_multi_f3::test_f3_pair_owner_keeps_teacher_suffix_domain`
- PASS: `tests.entry_v1.test_data_multi::test_incompatible_model_contract_and_changed_adapter_bytes_rejected`

### H_source_exact_reopen (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp08_source_probe.py artifacts/fg_s3_v2/e_quality_wp08/source_probe_04 --reopen
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

### H_source_reopen_ready (exit 0)

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp08_source_probe.py artifacts/fg_s3_v2/e_quality_wp08/source_probe_04
```

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp08-20261019/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```
