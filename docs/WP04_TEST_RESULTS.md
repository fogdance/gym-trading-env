# WP04 实际测试与退出码

选定最终用例口径：**379 个不同用例 PASS，0 FAIL，0 SKIP**。不是全项目通过。`C_shared_model` 的 1 项失败以同一项 `C_shared_recheck` 修正后的实跑覆盖；其他最终集合均 exit 0。旧失败不计为通过，全部保留。

| 原始命令标签 | exit | 收集 / fail / skip | 秒 | 输出 |
|---|---:|---|---:|---|
| A_contract | 0 | 12 / 0 / 0 | 1.76 | [receipt](receipts/wp04/A_contract.json) / [log](receipts/wp04/A_contract.log) |
| A_guards | 0 | 43 / 0 / 0 | 6.70 | [receipt](receipts/wp04/A_guards.json) / [log](receipts/wp04/A_guards.log) |
| A_model | 0 | 1 / 0 / 0 | 24.32 | [receipt](receipts/wp04/A_model.json) / [log](receipts/wp04/A_model.log) |
| B_byte_review | 0 | 1 / 0 / 0 | 8.93 | [receipt](receipts/wp04/B_byte_review.json) / [log](receipts/wp04/B_byte_review.log) |
| B_final | 0 | 17 / 0 / 0 | 165.89 | [receipt](receipts/wp04/B_final.json) / [log](receipts/wp04/B_final.log) |
| B_inspect_layout | 0 | — | 13.60 | [receipt](receipts/wp04/B_inspect_layout.json) / [log](receipts/wp04/B_inspect_layout.log) |
| B_reader_regression | 0 | 56 / 0 / 0 | 28.28 | [receipt](receipts/wp04/B_reader_regression.json) / [log](receipts/wp04/B_reader_regression.log) |
| B_retry | 1 | 15 / 1 / 0 | 143.10 | [receipt](receipts/wp04/B_retry.json) / [log](receipts/wp04/B_retry.log) |
| B_schedule | 0 | — | 1.17 | [receipt](receipts/wp04/B_schedule.json) / [log](receipts/wp04/B_schedule.log) |
| B_shared | 0 | 20 / 0 / 0 | 122.70 | [receipt](receipts/wp04/B_shared.json) / [log](receipts/wp04/B_shared.log) |
| B_smoke | 1 | 1 / 1 / 0 | 94.64 | [receipt](receipts/wp04/B_smoke.json) / [log](receipts/wp04/B_smoke.log) |
| B_targeted | 1 | 9 / 1 / 0 | 108.62 | [receipt](receipts/wp04/B_targeted.json) / [log](receipts/wp04/B_targeted.log) |
| C_burnin | 1 | 1 / 1 / 0 | 379.52 | [receipt](receipts/wp04/C_burnin.json) / [log](receipts/wp04/C_burnin.log) |
| C_diagnostics | 0 | 7 / 0 / 0 | 213.65 | [receipt](receipts/wp04/C_diagnostics.json) / [log](receipts/wp04/C_diagnostics.log) |
| C_final | 1 | 9 / 1 / 0 | 420.38 | [receipt](receipts/wp04/C_final.json) / [log](receipts/wp04/C_final.log) |
| C_reference_recheck | 0 | 1 / 0 / 0 | 278.43 | [receipt](receipts/wp04/C_reference_recheck.json) / [log](receipts/wp04/C_reference_recheck.log) |
| C_shared_baseline | 1 | — | 47.00 | [receipt](receipts/wp04/C_shared_baseline.json) / [log](receipts/wp04/C_shared_baseline.log) |
| C_shared_model | 1 | 19 / 1 / 0 | 353.30 | [receipt](receipts/wp04/C_shared_model.json) / [log](receipts/wp04/C_shared_model.log) |
| C_shared_recheck | 0 | 1 / 0 / 0 | 44.98 | [receipt](receipts/wp04/C_shared_recheck.json) / [log](receipts/wp04/C_shared_recheck.log) |
| D_admission | 0 | 12 / 0 / 0 | 21.14 | [receipt](receipts/wp04/D_admission.json) / [log](receipts/wp04/D_admission.log) |
| D_checkpoint_shared | 0 | 15 / 0 / 0 | 4.47 | [receipt](receipts/wp04/D_checkpoint_shared.json) / [log](receipts/wp04/D_checkpoint_shared.log) |
| D_cli_probe | 0 | — | 89.76 | [receipt](receipts/wp04/D_cli_probe.json) / [log](receipts/wp04/D_cli_probe.log) |
| D_cold | 1 | 14 / 2 / 0 | 147.24 | [receipt](receipts/wp04/D_cold.json) / [log](receipts/wp04/D_cold.log) |
| D_cold_retry | 1 | 14 / 1 / 0 | 137.03 | [receipt](receipts/wp04/D_cold_retry.json) / [log](receipts/wp04/D_cold_retry.log) |
| D_final | 0 | 14 / 0 / 0 | 125.93 | [receipt](receipts/wp04/D_final.json) / [log](receipts/wp04/D_final.log) |
| D_identity_observation | 0 | — | 2.61 | [receipt](receipts/wp04/D_identity_observation.json) / [log](receipts/wp04/D_identity_observation.log) |
| D_worker_continuous_start_4 | 0 | — | — | [receipt](receipts/wp04/D_worker_continuous_start_4.json) / [log](receipts/wp04/D_worker_continuous_start_4.log) |
| D_worker_continuous_start_4_294e1c74446e4f38b2b0511a6ce1a0c7 | 0 | — | — | [receipt](receipts/wp04/D_worker_continuous_start_4_294e1c74446e4f38b2b0511a6ce1a0c7.json) / [log](receipts/wp04/D_worker_continuous_start_4_294e1c74446e4f38b2b0511a6ce1a0c7.log) |
| D_worker_continuous_start_4_9a6ef4adc2f543a6b262963d1306bcff | 0 | — | — | [receipt](receipts/wp04/D_worker_continuous_start_4_9a6ef4adc2f543a6b262963d1306bcff.json) / [log](receipts/wp04/D_worker_continuous_start_4_9a6ef4adc2f543a6b262963d1306bcff.log) |
| D_worker_continuous_start_4_f83ddf3ce6114dbe84ad60cd08a6562f | 0 | — | — | [receipt](receipts/wp04/D_worker_continuous_start_4_f83ddf3ce6114dbe84ad60cd08a6562f.json) / [log](receipts/wp04/D_worker_continuous_start_4_f83ddf3ce6114dbe84ad60cd08a6562f.log) |
| D_worker_split_resume_4 | 1 | — | — | [receipt](receipts/wp04/D_worker_split_resume_4.json) / [log](receipts/wp04/D_worker_split_resume_4.log) |
| D_worker_split_resume_4_1b4abcbf65a6479e93096bf545f4380f | 0 | — | — | [receipt](receipts/wp04/D_worker_split_resume_4_1b4abcbf65a6479e93096bf545f4380f.json) / [log](receipts/wp04/D_worker_split_resume_4_1b4abcbf65a6479e93096bf545f4380f.log) |
| D_worker_split_resume_4_500d344cca5d4d83b4de1584fb21d6ba | 0 | — | — | [receipt](receipts/wp04/D_worker_split_resume_4_500d344cca5d4d83b4de1584fb21d6ba.json) / [log](receipts/wp04/D_worker_split_resume_4_500d344cca5d4d83b4de1584fb21d6ba.log) |
| D_worker_split_resume_4_602cbea38fe841b4ac2d0229fd06c552 | 0 | — | — | [receipt](receipts/wp04/D_worker_split_resume_4_602cbea38fe841b4ac2d0229fd06c552.json) / [log](receipts/wp04/D_worker_split_resume_4_602cbea38fe841b4ac2d0229fd06c552.log) |
| D_worker_split_start_2 | 0 | — | — | [receipt](receipts/wp04/D_worker_split_start_2.json) / [log](receipts/wp04/D_worker_split_start_2.log) |
| D_worker_split_start_2_0bb093c0559343faa97cb51b9f54146e | 0 | — | — | [receipt](receipts/wp04/D_worker_split_start_2_0bb093c0559343faa97cb51b9f54146e.json) / [log](receipts/wp04/D_worker_split_start_2_0bb093c0559343faa97cb51b9f54146e.log) |
| D_worker_split_start_2_6b7f3c9d72754ec1a846c542482f48f3 | 0 | — | — | [receipt](receipts/wp04/D_worker_split_start_2_6b7f3c9d72754ec1a846c542482f48f3.json) / [log](receipts/wp04/D_worker_split_start_2_6b7f3c9d72754ec1a846c542482f48f3.log) |
| D_worker_split_start_2_a4e8cf0e133548cca68f969835caaf61 | 0 | — | — | [receipt](receipts/wp04/D_worker_split_start_2_a4e8cf0e133548cca68f969835caaf61.json) / [log](receipts/wp04/D_worker_split_start_2_a4e8cf0e133548cca68f969835caaf61.log) |
| E_contracts | 0 | 246 / 0 / 0 | 158.60 | [receipt](receipts/wp04/E_contracts.json) / [log](receipts/wp04/E_contracts.log) |
| E_current_evidence | 0 | — | 3.67 | [receipt](receipts/wp04/E_current_evidence.json) / [log](receipts/wp04/E_current_evidence.log) |
| E_docs_validate | 0 | — | 0.15 | [receipt](receipts/wp04/E_docs_validate.json) / [log](receipts/wp04/E_docs_validate.log) |
| E_document_facts | 0 | — | 1.17 | [receipt](receipts/wp04/E_document_facts.json) / [log](receipts/wp04/E_document_facts.log) |
| E_document_facts_final | 0 | — | 1.19 | [receipt](receipts/wp04/E_document_facts_final.json) / [log](receipts/wp04/E_document_facts_final.log) |
| E_final_evidence | 0 | — | 3.69 | [receipt](receipts/wp04/E_final_evidence.json) / [log](receipts/wp04/E_final_evidence.log) |
| E_gate_audit_fixed | 0 | 1 / 0 / 0 | 2.69 | [receipt](receipts/wp04/E_gate_audit_fixed.json) / [log](receipts/wp04/E_gate_audit_fixed.log) |
| E_gate_baseline | 0 | — | 2.71 | [receipt](receipts/wp04/E_gate_baseline.json) / [log](receipts/wp04/E_gate_baseline.log) |
| E_legacy_resume | 0 | 6 / 0 / 0 | 141.43 | [receipt](receipts/wp04/E_legacy_resume.json) / [log](receipts/wp04/E_legacy_resume.log) |
| E_reports | 0 | — | 0.07 | [receipt](receipts/wp04/E_reports.json) / [log](receipts/wp04/E_reports.log) |
| E_required_final | 0 | 20 / 0 / 0 | 119.94 | [receipt](receipts/wp04/E_required_final.json) / [log](receipts/wp04/E_required_final.log) |
| E_required_shared | 1 | 20 / 1 / 0 | 133.12 | [receipt](receipts/wp04/E_required_shared.json) / [log](receipts/wp04/E_required_shared.log) |
| E_restore_final | 0 | 22 / 0 / 0 | 138.30 | [receipt](receipts/wp04/E_restore_final.json) / [log](receipts/wp04/E_restore_final.log) |
| E_teacher_loss | 0 | 11 / 0 / 0 | 87.35 | [receipt](receipts/wp04/E_teacher_loss.json) / [log](receipts/wp04/E_teacher_loss.log) |
| E_valid_identity | 0 | 8 / 0 / 0 | 25.33 | [receipt](receipts/wp04/E_valid_identity.json) / [log](receipts/wp04/E_valid_identity.log) |
| E_wp04 | 0 | 40 / 0 / 0 | 353.57 | [receipt](receipts/wp04/E_wp04.json) / [log](receipts/wp04/E_wp04.log) |
| G1_native_reverify | 0 | — | 1288.11 | [receipt](receipts/wp04/G1_native_reverify.json) / [log](receipts/wp04/G1_native_reverify.log) |
| G1_read_only | 0 | — | 52.33 | [receipt](receipts/wp04/G1_read_only.json) / [log](receipts/wp04/G1_read_only.log) |

## 精确执行命令

以下来自实际收据，不是计划命令。均从本 WP04 Gym worktree 执行；原工作树基线命令的 cwd/PYTHONPATH 在对应脚本与收据中。临时 `--cmd` 脚本同时归档为 `receipts/wp04/<文件名>.py.txt`。反复执行应使用新 receipt 标签和新输出目录，避免覆盖旧记录。

公共环境：`PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`；`FG_REPLAY_DREAMER_ROOT` 指向本工作树兄弟 Dreamer。

### A_contract（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/A_contract.xml tests/entry_v1/test_wp04_admission.py -k 'not A05'
```

### A_guards（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/A_guards.xml tests/entry_v1/test_s3_composition.py tests/entry_v1/test_wp03_cli.py::test_D12_public_E_quality_lifecycle_stays_closed_without_outputs tests/entry_v1/test_fg_s3_quality_student_v2.py::test_pair_identity_soft_targets_and_empty_gradients
```

### A_model（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/A_model.xml tests/entry_v1/test_wp04_admission.py::test_A05_fresh_complete_E_quality_model_and_no_teacher_training_inputs
```

### B_byte_review（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/B_byte_review.xml tests/entry_v1/test_wp04_wm.py::test_B12_signed_zero_is_a_native_byte_change_even_when_values_equal
```

### B_final（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/B_final.xml tests/entry_v1/test_wp04_wm.py
```

### B_inspect_layout（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /tmp/wp04_inspect_layout.py
```

### B_reader_regression（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/B_reader_regression.xml tests/entry_v1/test_wp03_admission.py tests/entry_v1/test_wp03_join.py tests/entry_v1/test_wp03_reader.py
```

### B_retry（exit 1）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/B_retry.xml tests/entry_v1/test_wp04_wm.py -x
```

### B_schedule（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /tmp/wp04_schedule_probe.py
```

### B_shared（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/B_shared.xml ../dreamerv3/embodied/tests/test_fg_s3_gate.py ../dreamerv3/embodied/tests/test_fg_s3_source_transfer.py ../dreamerv3/embodied/tests/test_action_mask_formal_guards.py
```

### B_smoke（exit 1）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/B_smoke.xml tests/entry_v1/test_wp04_wm.py::test_B03_actual_original_optimizer_step_and_frozen_slots
```

### B_targeted（exit 1）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/B_targeted.xml tests/entry_v1/test_wp04_wm.py -x
```

### C_burnin（exit 1）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/C_burnin.xml tests/entry_v1/test_wp04_diagnostics.py::test_C06_reset_episode_full_forward_and_finite_K_comparison
```

### C_diagnostics（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/C_diagnostics.xml tests/entry_v1/test_wp04_diagnostics.py
```

### C_final（exit 1）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/C_final.xml tests/entry_v1/test_wp04_diagnostics.py
```

### C_reference_recheck（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/C_reference_recheck.xml tests/entry_v1/test_wp04_diagnostics.py::test_C06_reset_episode_full_forward_and_finite_K_comparison
```

### C_shared_baseline（exit 1）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /tmp/wp04_baseline_em04.py
```

### C_shared_model（exit 1）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/C_shared_model.xml ../dreamerv3/embodied/tests/test_fg_s3_e_model.py tests/entry_v1/test_wp02_macd.py
```

### C_shared_recheck（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/C_shared_recheck.xml ../dreamerv3/embodied/tests/test_fg_s3_e_model.py::test_EM04_actual_policy_context_changes_rssm_and_actor
```

### D_admission（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/D_admission.xml tests/entry_v1/test_wp04_admission.py -k 'not A05'
```

### D_checkpoint_shared（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/D_checkpoint_shared.xml ../dreamerv3/embodied/tests/test_fg_full_checkpoint.py ../dreamerv3/embodied/tests/test_checkpoint_retention.py
```

### D_cli_probe（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py f1-probe --output-dir artifacts/fg_s3_v2/e_quality_wp04/G0_cli_probe
```

### D_cold（exit 1）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/D_cold.xml tests/entry_v1/test_wp04_checkpoint.py
```

### D_cold_retry（exit 1）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/D_cold_retry.xml tests/entry_v1/test_wp04_checkpoint.py
```

### D_final（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/D_final.xml tests/entry_v1/test_wp04_checkpoint.py --basetemp=artifacts/fg_s3_v2/e_quality_wp04/G0_cold_final
```

### D_identity_observation（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /tmp/wp04_identity_inspect.py
```

### D_worker_continuous_start_4（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/tests/entry_v1/wp04_cold_worker.py /tmp/pytest-of-v/pytest-478/wp04_cold0/settings.json /tmp/pytest-of-v/pytest-478/wp04_cold0/continuous /tmp/pytest-of-v/pytest-478/wp04_cold0/join.sqlite start 4
```

### D_worker_continuous_start_4_294e1c74446e4f38b2b0511a6ce1a0c7（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/tests/entry_v1/wp04_cold_worker.py /tmp/pytest-of-v/pytest-481/wp04_cold0/settings.json /tmp/pytest-of-v/pytest-481/wp04_cold0/continuous /tmp/pytest-of-v/pytest-481/wp04_cold0/join.sqlite start 4
```

### D_worker_continuous_start_4_9a6ef4adc2f543a6b262963d1306bcff（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/tests/entry_v1/wp04_cold_worker.py /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_restore_review_final/wp04_cold0/settings.json /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_restore_review_final/wp04_cold0/continuous /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_restore_review_final/wp04_cold0/join.sqlite start 4
```

### D_worker_continuous_start_4_f83ddf3ce6114dbe84ad60cd08a6562f（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/tests/entry_v1/wp04_cold_worker.py /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_cold_final/wp04_cold0/settings.json /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_cold_final/wp04_cold0/continuous /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_cold_final/wp04_cold0/join.sqlite start 4
```

### D_worker_split_resume_4（exit 1）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/tests/entry_v1/wp04_cold_worker.py /tmp/pytest-of-v/pytest-478/wp04_cold0/settings.json /tmp/pytest-of-v/pytest-478/wp04_cold0/split /tmp/pytest-of-v/pytest-478/wp04_cold0/join.sqlite resume 4
```

### D_worker_split_resume_4_1b4abcbf65a6479e93096bf545f4380f（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/tests/entry_v1/wp04_cold_worker.py /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_restore_review_final/wp04_cold0/settings.json /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_restore_review_final/wp04_cold0/split /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_restore_review_final/wp04_cold0/join.sqlite resume 4
```

### D_worker_split_resume_4_500d344cca5d4d83b4de1584fb21d6ba（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/tests/entry_v1/wp04_cold_worker.py /tmp/pytest-of-v/pytest-481/wp04_cold0/settings.json /tmp/pytest-of-v/pytest-481/wp04_cold0/split /tmp/pytest-of-v/pytest-481/wp04_cold0/join.sqlite resume 4
```

### D_worker_split_resume_4_602cbea38fe841b4ac2d0229fd06c552（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/tests/entry_v1/wp04_cold_worker.py /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_cold_final/wp04_cold0/settings.json /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_cold_final/wp04_cold0/split /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_cold_final/wp04_cold0/join.sqlite resume 4
```

### D_worker_split_start_2（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/tests/entry_v1/wp04_cold_worker.py /tmp/pytest-of-v/pytest-478/wp04_cold0/settings.json /tmp/pytest-of-v/pytest-478/wp04_cold0/split /tmp/pytest-of-v/pytest-478/wp04_cold0/join.sqlite start 2
```

### D_worker_split_start_2_0bb093c0559343faa97cb51b9f54146e（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/tests/entry_v1/wp04_cold_worker.py /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_restore_review_final/wp04_cold0/settings.json /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_restore_review_final/wp04_cold0/split /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_restore_review_final/wp04_cold0/join.sqlite start 2
```

### D_worker_split_start_2_6b7f3c9d72754ec1a846c542482f48f3（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/tests/entry_v1/wp04_cold_worker.py /tmp/pytest-of-v/pytest-481/wp04_cold0/settings.json /tmp/pytest-of-v/pytest-481/wp04_cold0/split /tmp/pytest-of-v/pytest-481/wp04_cold0/join.sqlite start 2
```

### D_worker_split_start_2_a4e8cf0e133548cca68f969835caaf61（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/tests/entry_v1/wp04_cold_worker.py /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_cold_final/wp04_cold0/settings.json /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_cold_final/wp04_cold0/split /home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_cold_final/wp04_cold0/join.sqlite start 2
```

### E_contracts（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/E_contracts.xml tests/entry_v1/test_fg_s3_e_quality_course_preflight.py ../dreamerv3/dreamerv3/tests/test_fg_s3_e_quality_course_contract.py tests/entry_v1/test_wp02_strict.py tests/entry_v1/test_wp02_corpus_facts.py tests/entry_v1/test_wp02_replay.py tests/entry_v1/test_wp02_cli.py tests/entry_v1/test_wp02_snapshot_performance.py tests/entry_v1/test_wp03_admission.py tests/entry_v1/test_wp03_join.py tests/entry_v1/test_wp03_reader.py tests/entry_v1/test_wp03_cli.py tests/entry_v1/test_s3_market_minute.py tests/entry_v1/test_fg_single3_env.py tests/entry_v1/test_fg_single3_replay.py tests/unit/test_quality_tp_reward_v1.py
```

### E_current_evidence（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /tmp/wp04_final_evidence.py
```

### E_docs_validate（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /tmp/wp04_docs_validate.py
```

### E_document_facts（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /tmp/wp04_write_docs.py
```

### E_document_facts_final（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /tmp/wp04_write_docs.py
```

### E_final_evidence（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /tmp/wp04_final_evidence.py
```

### E_gate_audit_fixed（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/E_gate_audit_fixed.xml ../dreamerv3/embodied/tests/test_fg_s3_gate.py::test_audited_ast_accepts_only_the_reviewed_transition
```

### E_gate_baseline（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /tmp/wp04_baseline_gate.py
```

### E_legacy_resume（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/E_legacy_resume.xml ../dreamerv3/embodied/tests/test_fg_s3_e_checkpoint.py ../dreamerv3/embodied/tests/test_driver_account_checkpoint.py
```

### E_reports（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /tmp/wp04_final_reports.py
```

### E_required_final（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/E_required_final.xml ../dreamerv3/embodied/tests/test_fg_s3_gate.py ../dreamerv3/embodied/tests/test_fg_s3_source_transfer.py ../dreamerv3/embodied/tests/test_action_mask_formal_guards.py
```

### E_required_shared（exit 1）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/E_required_shared.xml ../dreamerv3/embodied/tests/test_fg_s3_gate.py ../dreamerv3/embodied/tests/test_fg_s3_source_transfer.py ../dreamerv3/embodied/tests/test_action_mask_formal_guards.py
```

### E_restore_final（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/E_restore_final.xml tests/entry_v1/test_wp04_checkpoint.py tests/entry_v1/test_wp04_resume_identity.py --basetemp=artifacts/fg_s3_v2/e_quality_wp04/G0_restore_review_final
```

### E_teacher_loss（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/E_teacher_loss.xml tests/entry_v1/test_wp03_cpu_loss.py
```

### E_valid_identity（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/E_valid_identity.xml tests/entry_v1/test_wp04_resume_identity.py
```

### E_wp04（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/docs/receipts/wp04/E_wp04.xml tests/entry_v1/test_wp04_admission.py tests/entry_v1/test_wp04_wm.py tests/entry_v1/test_wp04_diagnostics.py
```

### G1_native_reverify（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B /tmp/wp04_native_reverify.py
```

### G1_read_only（exit 0）

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py f1-read-check --config configs/experiments/fg_s3_v2/e_quality_f1_read_check_wp04_20261018.yaml --output-dir docs/receipts/wp04/G1_read_only
```
