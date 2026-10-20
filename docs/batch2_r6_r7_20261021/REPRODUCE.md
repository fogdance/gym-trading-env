# 定向复验

原始命令是事实来源：`receipts/FINAL_V5_*.json`、`receipts/MERGE_VALIDATION_V5.json` 和 `receipts/performance_final_v3/*_command.json` 保留了 argv、cwd、环境、CPU affinity、实际 SHA 和退出码。原始 JUnit 列出每个收集用例。

从对应干净兄弟 worktree 的 Gym 根目录使用既有 Python 环境，例如最小数据层集合：

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 \
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider \
  tests/entry_v1/test_batch2_source_freeze.py \
  tests/entry_v1/test_batch2_reader_transactions.py \
  tests/entry_v1/test_batch2_reader_versions.py \
  tests/entry_v1/test_batch2_cold_checkpoint.py
```

其余 F1/F2/F3 模型冷恢复、阶段接续和共享回归按各分组 JSON 执行；有些现有测试在 cwd 下写 `docs/receipts/wp04` 等路径，需按原收据用独立 artifacts cwd，并预建 wp04/wp05/wp06/wp07 目录、复制已有 `B_initial_layout.json`。本轮精确运行器和 setup 行为已在收据中保留。不要把正式 Corpus 或 Run 用作 pytest basetemp。

同一性能工况的最小命令如下；`--config` 指向本轮只生成一次、两个版本共用的合成来源配置，`--out` 必须是全新开发输出目录：

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 \
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/batch2_mpl \
  taskset -c 0-1 /home/v/miniconda3/envs/dreamerv3/bin/python -B \
  tests/entry_v1/batch2_reader_benchmark.py \
  --config /home/v/Documents/work/research/fg-s3-batch2-r6-r7-20261021/gym-trading-env/artifacts/batch2/bench_v1/config.json \
  --out artifacts/batch2/reproduce_f1_100k_NEW \
  --mode f1 --seen 100000 --batches 2 --versioned
```

随后在完全新的 Python 进程使用相同命令追加 `--load-only`，核对 `result_load.json` 与 `result.json` 的 state_digest。旧版对照使用同一 benchmark 文件、同一 config，并去掉 `--versioned`；不可在两个版本中分别重建 Corpus 再假定身份相同。benchmark 在不存在 config 时可以构建少量合法合成输入，绝不读取 TEST 或启动正式训练。

`--seen` 是人工历史覆盖压力，不能作为真实市场覆盖。有效位分母由额外未计时重放核对逐批哈希；原实现和实验脚本均保存在 `receipts/reproduction_scripts/`。这些协调脚本按其原路径 `artifacts/batch2/` 计算工作区根目录，移机复现时先复制到该位置，并按原 argv 调整已固定的实验目录；不要直接从收据目录执行具有 merge 动作的脚本。

复验是 CPU 开发行为。正式 Corpus、Producer pin、独立 G3、训练授权和新 Run 冻结仍使用既有正式流程，本页不授予正式训练权限。

历史 `postmerge_reports_v4.py.NOT_RUN.txt` 是未执行的开发脚本存档；最终协调脚本为带 v5 后缀的源码，实际命令和通过结果以 FINAL_V5 / MERGE_VALIDATION_V5 收据为准。
