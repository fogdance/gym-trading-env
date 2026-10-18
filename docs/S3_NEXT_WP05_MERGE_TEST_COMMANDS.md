# 实际测试命令与退出码

运行目录：`/home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env`。以下为原始收据记录的实际子进程 argv 与环境，不是拟执行命令。失败收据保留；最终选择规则见合并审查。重新执行应选择新的 JUnit/收据/Probe 输出路径。

## Integration_contract_v1

实际退出码：`1`。耗时：15.36 秒。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-next-integration/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env/docs/receipts/wp05/Integration_contract_v1.xml tests/entry_v1/test_wp05_contract.py tests/entry_v1/test_wp05_transfer.py -x
```

## Integration_data_reader_v1

实际退出码：`1`。耗时：48.27 秒。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-next-integration/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env/docs/receipts/wp05/Integration_data_reader_v1.xml tests/entry_v1/test_macd_dated_tail.py tests/entry_v1/test_wp02_macd.py tests/entry_v1/test_s3_market_minute.py tests/entry_v1/test_corpus_dev_mode.py tests/entry_v1/test_corpus_production_controls.py tests/entry_v1/test_wp03_join.py tests/entry_v1/test_wp03_reader.py tests/entry_v1/test_wp03_cli.py
```

## Integration_data_reader_v2

实际退出码：`0`。耗时：52.74 秒。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-next-integration/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env/docs/receipts/wp05/Integration_data_reader_v2.xml tests/entry_v1/test_macd_dated_tail.py tests/entry_v1/test_wp02_macd.py tests/entry_v1/test_s3_market_minute.py tests/entry_v1/test_corpus_dev_mode.py tests/entry_v1/test_corpus_production_controls.py tests/entry_v1/test_wp03_join.py tests/entry_v1/test_wp03_reader.py tests/entry_v1/test_wp03_cli.py
```

## Integration_shared_v1

实际退出码：`0`。耗时：279.84 秒。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-next-integration/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env/docs/receipts/wp05/Integration_shared_v1.xml ../dreamerv3/embodied/tests/test_fg_s3_gate.py ../dreamerv3/embodied/tests/test_fg_s3_source_transfer.py ../dreamerv3/embodied/tests/test_action_mask_formal_guards.py ../dreamerv3/embodied/tests/test_fg_full_checkpoint.py ../dreamerv3/embodied/tests/test_fg_s3c_resume.py
```

## Integration_f1_v1

实际退出码：`0`。耗时：373.75 秒。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-next-integration/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env/docs/receipts/wp05/Integration_f1_v1.xml tests/entry_v1/test_wp04_admission.py tests/entry_v1/test_wp04_wm.py tests/entry_v1/test_wp04_checkpoint.py tests/entry_v1/test_wp04_resume_identity.py
```

## Integration_wp05_v2

实际退出码：`1`。耗时：403.24 秒。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-next-integration/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env/docs/receipts/wp05/Integration_wp05_v2.xml tests/entry_v1/test_wp05_contract.py tests/entry_v1/test_wp05_transfer.py tests/entry_v1/test_wp05_updates.py tests/entry_v1/test_wp05_diagnostics.py tests/entry_v1/test_wp05_checkpoint.py tests/entry_v1/test_wp05_cli.py
```

## Integration_cli_v3

实际退出码：`0`。耗时：91.95 秒。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-next-integration/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env/docs/receipts/wp05/Integration_cli_v3.xml tests/entry_v1/test_wp05_cli.py
```

## Integration_legacy_quality_v1

实际退出码：`0`。耗时：91.14 秒。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-next-integration/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env/docs/receipts/wp05/Integration_legacy_quality_v1.xml tests/entry_v1/test_fg_s3_quality_student_v2.py tests/entry_v1/test_fg_s3_quality_prediction_v2.py
```

## Integration_probe_v1

实际退出码：`0`。耗时：115.6 秒。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-next-integration/dreamerv3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py f2-probe --output-dir artifacts/fg_s3_v2/e_quality_wp05/integration_probe_01 --eval-device cpu
```
