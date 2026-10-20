# Corpus SemVer / R1–R4 双仓合并验收

## Summary

已按用户本轮指令将 `feat/corpus-semver-r1-r4-20261020` 合入本地双仓 `integration/s3-next`，两仓均无冲突。包含 Corpus ABI **3.1.0**、F1/F2/F3 显式 Producer pin、Prepared/Run 来源绑定和严格最新 Checkpoint 恢复。

| 仓库 | 合并前主线 | 合入分支 | 合并提交 |
|---|---|---|---|
| Gym | `8619f04b6f996f4c4d7f169486e4d10c25a3b2b9` | `20cdedf4b1d01ab7c2f9dffd979a01d80c9f5067` | `12f2e567885733872ca348cf48876b407d60c3a7` |
| Dreamer | `b7a693b3c7b0b45eeecdec954818be42d40e67b6` | `4c4a36636bfa1a3f6a02c6df7f7d164ca8ec7036` | `b2287a9d892e932339857ebb36f75499163a3046` |

最终测试代码提交：Gym `5d752e471398d4ac65f60fa3636afc2e47cfe923`；Dreamer `b2287a9d892e932339857ebb36f75499163a3046`。Gym 随后的交付提交仅新增本报告及本轮收据，不修改测试或运行源码。

原 [IMPLEMENTATION_REVIEW.md](corpus_semver_r1_r4_20261020/IMPLEMENTATION_REVIEW.md) 是锁定的合并前验收快照，其中“未合并”保留其当时语义；本文件记录之后完成的合并。

## Evidence observed

- 两个合并节点的完整 Git tree 分别等于本轮来源分支，无冲突覆盖或额外运行代码修改。
- 原有 **183** 份锁定文档/收据 SHA256 全部吻合。
- 两个原 Corpus 共 **176** 个文件 SHA256 全部不变；没有修改、重签或重新生成。
- 受保护冻结 Producer 双仓维持原 HEAD，工作树干净。
- 原实现的 268 个独立测试 / 298 次执行见原验收包。本轮重新执行的集成检查为下方 **107** 项，两个口径分别保留。

证据：[合并与完整性审计](receipts/corpus_semver_r1_r4_merge_20261021/MERGE_AND_INTEGRITY_AUDIT.json)、[测试修正后的源码审计](receipts/corpus_semver_r1_r4_merge_20261021/POST_TEST_ISOLATION_SOURCE_AUDIT.json)。

## Root cause

合并后的组合执行新增发现测试顺序依赖：F2 运行模块在顶层 `from ... import inspect_f1, transfer_parameters`，旧测试只 patch 来源模块。先执行 verifier binding 测试后，缓存的函数别名带入后续 offline recovery 测试，简化 Agent 进入真实参数迁移函数，报 `TypeError: 'int' object is not subscriptable`。

该顺序在未修改来源分支（Gym `20cdedf4b1d01ab7c2f9dffd979a01d80c9f5067` / Dreamer `4c4a36636bfa1a3f6a02c6df7f7d164ca8ec7036`）独立复现：第一例通过、第二例同样失败，退出码 **1**。见 [ORDER_BASELINE.json](receipts/corpus_semver_r1_r4_merge_20261021/ORDER_BASELINE.json) 及原始日志。该缺陷属于测试隔离，运行源码没有合并回归。

## Changes made

双仓使用 `git merge --no-ff` 保存完整已验收提交链。额外测试提交 `5d752e471398d4ac65f60fa3636afc2e47cfe923` 只调整两个测试文件：先导入运行模块，再安装替身；F2 同时 patch 实际调用的函数别名。断言和金融/标签/模型/恢复逻辑保持原样。

## Verification performed

| 检查 | 通过 | 失败/错误/跳过 | 退出码 | 秒 |
|---|---:|---|---:|---:|
| AGENTS 指定的 Gate / Source Transfer / Action Mask 三文件 | 20 | 0 / 0 / 0 | 0 | 124.364 |
| ABI / Producer pin / verifier binding / F1–F3 offline recovery / 严格 CP / FullCheckpoint / WP07 CP 核心 | 87 | 0 / 0 / 0 | 0 | 80.828 |
| 合计（独立测试） | **107** | **0 / 0 / 0** | **0** | — |

首次组合执行的 **86 PASS / 1 FAIL**、未修改分支复现和修复后完整复测均保存，未覆盖失败收据。共享回归后仅变更两个测试文件，运行源码和共享测试文件哈希未变；没有重复运行相同共享检查。

所有测试使用 CPU 和合成数据，禁用 GPU。准确 argv、cwd、环境、测试双仓 SHA、耗时及退出码见 [FINAL_COMMANDS.json](receipts/corpus_semver_r1_r4_merge_20261021/FINAL_COMMANDS.json)；逐项清单见 [FINAL_TEST_INDEX.json](receipts/corpus_semver_r1_r4_merge_20261021/FINAL_TEST_INDEX.json)，汇总见 [FINAL_RESULTS.json](receipts/corpus_semver_r1_r4_merge_20261021/FINAL_RESULTS.json)。

实际命令：

```bash
cd /home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env
env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 taskset -c 0-15 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider ../dreamerv3/embodied/tests/test_fg_s3_gate.py ../dreamerv3/embodied/tests/test_fg_s3_source_transfer.py ../dreamerv3/embodied/tests/test_action_mask_formal_guards.py --junitxml=/home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env/docs/receipts/corpus_semver_r1_r4_merge_20261021/shared.xml
```

```bash
cd /home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env/artifacts/semver_merge_validation_20261021/contracts
env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=/home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env:/home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env/src:/home/v/Documents/work/research/fg-s3-next-integration/dreamerv3:/home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env/tests/entry_v1 taskset -c 0-15 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider /home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env/tests/entry_v1/test_corpus_compatibility.py /home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env/tests/entry_v1/test_corpus_producer_pin.py /home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env/tests/entry_v1/test_corpus_verifier_binding.py /home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env/tests/entry_v1/test_corpus_offline_recovery.py /home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env/tests/entry_v1/test_wp07_checkpoint_core.py /home/v/Documents/work/research/fg-s3-next-integration/dreamerv3/embodied/tests/test_fg_course_strict_checkpoint.py /home/v/Documents/work/research/fg-s3-next-integration/dreamerv3/embodied/tests/test_fg_full_checkpoint.py --junitxml=/home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env/docs/receipts/corpus_semver_r1_r4_merge_20261021/contracts_recheck.xml
```

## Risks / not verified

生产 Native G3、真实正式 prepare/start/resume、长期训练和主机断电验证：**NOT_RUN**。未读取 TEST 行情。代码合并及 CPU 回归通过不代表正式教材或正式训练已准入；既有 Run 继续使用各自冻结源码与恢复身份。

## Next recommended step

后续正式准备使用本轮合并后的冻结双仓，按 [INTERFACE.md](corpus_semver_r1_r4_20261020/INTERFACE.md) 显式指定 Producer pin 及外部 SHA256，沿用原严格 G3、Reader 和正式训练授权流程。
