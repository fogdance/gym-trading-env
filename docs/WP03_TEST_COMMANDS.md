# WP03 实际命令与退出码

工作目录：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。JSON 原收据包含双仓 HEAD/dirty 和实际执行源码字节；这些哈希只用于报告取证。

各轮重叠，最终选定集合为 D_release_final 的177项，不能累加成总通过数。

## A_final：exit=0

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 1.109 秒；用例计数 `{'PASS': 14, 'FAIL': 0, 'ERROR': 0, 'SKIP': 0}`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env/docs/receipts/wp03/A_final.xml tests/entry_v1/test_wp03_admission.py
```

## A_targeted：exit=0

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 1.150 秒；用例计数 `{'PASS': 14, 'FAIL': 0, 'ERROR': 0, 'SKIP': 0}`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env/docs/receipts/wp03/A_targeted.xml tests/entry_v1/test_wp03_admission.py
```

## B_final：exit=0

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 21.227 秒；用例计数 `{'PASS': 22, 'FAIL': 0, 'ERROR': 0, 'SKIP': 0}`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env/docs/receipts/wp03/B_final.xml tests/entry_v1/test_wp03_join.py tests/entry_v1/test_fg_s3_quality_student_v2.py::test_pair_identity_soft_targets_and_empty_gradients tests/entry_v1/test_fg_s3_quality_student_v2.py::test_q_weak_wait_and_independent_prediction_masks
```

## B_legacy_pairs：exit=0

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 8.843 秒；用例计数 `{'PASS': 16, 'FAIL': 0, 'ERROR': 0, 'SKIP': 0}`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env/docs/receipts/wp03/B_legacy_pairs.xml tests/entry_v1/test_fg_s3_quality_student_v2.py::test_pair_identity_soft_targets_and_empty_gradients tests/entry_v1/test_fg_s3_quality_student_v2.py::test_q_weak_wait_and_independent_prediction_masks tests/entry_v1/test_wp03_admission.py
```

## B_targeted：exit=0

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 15.539 秒；用例计数 `{'PASS': 18, 'FAIL': 0, 'ERROR': 0, 'SKIP': 0}`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env/docs/receipts/wp03/B_targeted.xml tests/entry_v1/test_wp03_join.py
```

## C_final：exit=0

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 31.200 秒；用例计数 `{'PASS': 49, 'FAIL': 0, 'ERROR': 0, 'SKIP': 0}`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env/docs/receipts/wp03/C_final.xml tests/entry_v1/test_wp03_admission.py tests/entry_v1/test_wp03_join.py tests/entry_v1/test_wp03_reader.py tests/entry_v1/test_fg_s3_quality_student_v2.py::test_pair_identity_soft_targets_and_empty_gradients
```

## C_targeted：exit=0

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 14.416 秒；用例计数 `{'PASS': 14, 'FAIL': 0, 'ERROR': 0, 'SKIP': 0}`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env/docs/receipts/wp03/C_targeted.xml tests/entry_v1/test_wp03_reader.py
```

## D_actual_coverage：exit=0

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 3.809 秒；用例计数 `None`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_e_quality_corpus.py coverage --corpus /home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp02/real_small_04
```

## D_cpu_forward_fix：exit=0

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 75.027 秒；用例计数 `{'PASS': 1, 'FAIL': 0, 'ERROR': 0, 'SKIP': 0}`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env/docs/receipts/wp03/D_cpu_forward_fix.xml tests/entry_v1/test_wp03_cpu_loss.py::test_D01_real_E_macd_quality_agent_loss_ready_without_update -x
```

## D_cpu_retry：exit=0

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 154.854 秒；用例计数 `{'PASS': 11, 'FAIL': 0, 'ERROR': 0, 'SKIP': 0}`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env/docs/receipts/wp03/D_cpu_retry.xml tests/entry_v1/test_wp03_cpu_loss.py -x
```

## D_cpu_targeted：exit=1

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 77.790 秒；用例计数 `{'PASS': 1, 'FAIL': 10, 'ERROR': 0, 'SKIP': 0}`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env/docs/receipts/wp03/D_cpu_targeted.xml tests/entry_v1/test_wp03_cpu_loss.py
```

## D_data_cli：exit=0

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 51.635 秒；用例计数 `{'PASS': 60, 'FAIL': 0, 'ERROR': 0, 'SKIP': 0}`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env/docs/receipts/wp03/D_data_cli.xml tests/entry_v1/test_wp03_admission.py tests/entry_v1/test_wp03_join.py tests/entry_v1/test_wp03_reader.py tests/entry_v1/test_wp03_cli.py -x
```

## D_final_review：exit=1

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 15.972 秒；用例计数 `{'PASS': 30, 'FAIL': 1, 'ERROR': 0, 'SKIP': 0}`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env/docs/receipts/wp03/D_final_review.xml tests/entry_v1/test_wp03_admission.py tests/entry_v1/test_wp03_cli.py tests/entry_v1/test_wp03_join.py::test_B13_invalid_strict_reasons_never_become_valid_wait tests/entry_v1/test_wp03_join.py::test_B14_real_native_quality_suffix_tie_certifies_wait_only_with_two_actions tests/entry_v1/test_wp02_strict.py::test_D03_forced_wait_and_unknown_boundary_never_action_labels -x
```

## D_native_G3：exit=0

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 1694.455 秒；用例计数 `None`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_e_quality_reader.py source --purpose native-verification --corpus /home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp02/real_small_04 --producer-gym /home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/gym-trading-env --out artifacts/fg_s3_v2/e_quality_wp03/native_g3_01
```

## D_native_tie_contract：exit=0

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 16.939 秒；用例计数 `{'PASS': 4, 'FAIL': 0, 'ERROR': 0, 'SKIP': 0}`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env/docs/receipts/wp03/D_native_tie_contract.xml tests/entry_v1/test_wp03_join.py::test_B14_real_native_quality_suffix_tie_certifies_wait_only_with_two_actions tests/entry_v1/test_wp03_join.py::test_B07_fake_strict_wait_and_rational_witness_rejected tests/entry_v1/test_wp02_cli.py::test_E04_resigned_teacher_forgery_is_rejected_by_native_facts -x
```

## D_native_tie_fix：exit=1

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 2.661 秒；用例计数 `{'PASS': 0, 'FAIL': 1, 'ERROR': 0, 'SKIP': 0}`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env/docs/receipts/wp03/D_native_tie_fix.xml tests/entry_v1/test_wp03_join.py::test_B14_real_native_quality_suffix_tie_certifies_wait_only_with_two_actions -x
```

## D_real_reader_01：exit=0

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 72.066 秒；用例计数 `None`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_e_quality_reader.py read --corpus /home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp02/real_small_04 --out artifacts/fg_s3_v2/e_quality_wp03/real_reader_01 --length 64 --seed 202
```

## D_real_reader_final：exit=0

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 65.053 秒；用例计数 `None`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_e_quality_reader.py read --corpus /home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp02/real_small_04 --out artifacts/fg_s3_v2/e_quality_wp03/real_reader_final --length 64 --seed 202
```

## D_release_final：exit=0

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 303.268 秒；用例计数 `{'PASS': 177, 'FAIL': 0, 'ERROR': 0, 'SKIP': 0}`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env/docs/receipts/wp03/D_release_final.xml tests/entry_v1/test_wp03_admission.py tests/entry_v1/test_wp03_join.py tests/entry_v1/test_wp03_reader.py tests/entry_v1/test_wp03_cpu_loss.py tests/entry_v1/test_wp03_cli.py tests/entry_v1/test_fg_s3_e_quality_course_preflight.py tests/entry_v1/test_fg_replay.py tests/entry_v1/test_fg_single3_replay.py tests/entry_v1/test_wp02_macd.py tests/entry_v1/test_wp02_replay.py tests/entry_v1/test_fg_s3_quality_student_v2.py::test_pair_identity_soft_targets_and_empty_gradients tests/entry_v1/test_fg_s3_quality_student_v2.py::test_q_weak_wait_and_independent_prediction_masks tests/entry_v1/test_wp02_strict.py::test_D03_forced_wait_and_unknown_boundary_never_action_labels tests/entry_v1/test_wp02_cli.py::test_E04_resigned_teacher_forgery_is_rejected_by_native_facts ../dreamerv3/embodied/tests/test_fg_s3_gate.py ../dreamerv3/embodied/tests/test_fg_s3_source_transfer.py ../dreamerv3/embodied/tests/test_action_mask_formal_guards.py
```

## D_release_suite：exit=0

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 320.218 秒；用例计数 `{'PASS': 168, 'FAIL': 0, 'ERROR': 0, 'SKIP': 0}`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env/docs/receipts/wp03/D_release_suite.xml tests/entry_v1/test_wp03_admission.py tests/entry_v1/test_wp03_join.py tests/entry_v1/test_wp03_reader.py tests/entry_v1/test_wp03_cpu_loss.py tests/entry_v1/test_wp03_cli.py tests/entry_v1/test_fg_s3_e_quality_course_preflight.py tests/entry_v1/test_fg_replay.py tests/entry_v1/test_fg_single3_replay.py tests/entry_v1/test_wp02_macd.py tests/entry_v1/test_wp02_replay.py tests/entry_v1/test_fg_s3_quality_student_v2.py::test_pair_identity_soft_targets_and_empty_gradients tests/entry_v1/test_fg_s3_quality_student_v2.py::test_q_weak_wait_and_independent_prediction_masks ../dreamerv3/embodied/tests/test_fg_s3_gate.py ../dreamerv3/embodied/tests/test_fg_s3_source_transfer.py ../dreamerv3/embodied/tests/test_action_mask_formal_guards.py
```

## D_review_producer_after：exit=0

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 3.792 秒；用例计数 `{'PASS': 26, 'FAIL': 0, 'ERROR': 0, 'SKIP': 0}`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env/docs/receipts/wp03/D_review_producer_after.xml tests/entry_v1/test_wp03_admission.py tests/entry_v1/test_wp03_cli.py
```

## D_review_producer_before：exit=1

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 0.945 秒；用例计数 `{'PASS': 0, 'FAIL': 1, 'ERROR': 0, 'SKIP': 0}`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env/docs/receipts/wp03/D_review_producer_before.xml tests/entry_v1/test_wp03_cli.py::test_D13_matching_head_but_modified_frozen_verifier_cannot_certify
```

## D_shared：exit=0

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env`。

实际耗时 226.618 秒；用例计数 `{'PASS': 20, 'FAIL': 0, 'ERROR': 0, 'SKIP': 0}`。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016/gym-trading-env/docs/receipts/wp03/D_shared.xml ../dreamerv3/embodied/tests/test_fg_s3_gate.py ../dreamerv3/embodied/tests/test_fg_s3_source_transfer.py ../dreamerv3/embodied/tests/test_action_mask_formal_guards.py
```

## native_verification：exit=0

实际cwd：`/home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/gym-trading-env`。

实际耗时 1693.454 秒；用例计数 `None`。

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_e_quality_corpus.py verify --corpus /home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp02/real_small_04
```
