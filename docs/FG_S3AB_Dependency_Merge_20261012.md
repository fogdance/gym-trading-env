# S3-A / S3-B 依赖合并与训练隔离核对

## Summary

根据用户指令，将两份本机依赖仓库已提交的更新合入各自当前
`feat/fg-s3b-certified-bc` 分支。没有切换或改写 S3-A 分支，没有修改生产训练仓库、
Checkpoint 或实验日志，没有启动正式 S3-A / S3-B GPU 训练。

| 仓库 | 合并前 S3-B HEAD | 合入依赖 HEAD | 合并提交 |
|---|---|---|---|
| gym-trading-env2 | `76ab73a926916f39d399b61e256bc894c9b61246` | `7216d90eb22bd28cdf1abe34e3585213e8aa9f18` | `ee82eed7f17ebf8b0ea22b0ae943bce5e00ff27d` |
| dreamerv3-dev | `b6aac90465dca384f0fd40ecbbbf21bdd831a523` | `09e2e5ce5b07433790e7f4c3e40be7ca4e084064` | `fd20e281f6e27a87958ad5edfce3848b2e508d6e` |

## Evidence observed

两份依赖均来自本机对应生产仓库的 `feat/fg-single3-tp5-sl30-oracle-replay`
分支；核对其实际 Git HEAD 后只读取 Git 对象到开发仓库。
Gym 新增/变更 7 个依赖文件，Dreamer 变更 12 个依赖文件。合并后逐文件与依赖
仓库比较，全部字节一致；两次 Git 合并均无冲突。

合并前后核对 S3-A / S3-B 命名空间中的 Python / YAML 文件：Gym 18 个、Dreamer
16 个，34 个文件哈希全部不变。两仓库原有未跟踪文件均保留。
本次合并没有重签旧数据、修改原训练身份或重新生成 Oracle。

以下已有缓存的全部**已绑定代码**哈希仍与本轮代码相符：

| Manifest | 已绑定代码文件数 | 代码哈希变化数 |
|---|---:|---:|
| `artifacts/fg_s3b_v1/values_full_v2/manifest.json` | 52 | 0 |
| `artifacts/fg_s3b_v1/weights_full_v1/manifest.json` | 53 | 0 |
| `artifacts/fg_s3a_v1/smoke_real_v3/features/manifest.json` | 11 | 0 |

这项检查只说明本次合并没有改变这些 manifest 已绑定的代码；不等同于对任意
源 Checkpoint、所有数据文件或未来训练环境重新完成准入。

## Root cause

本轮为依赖同步，不是已复现的 S3-A / S3-B 训练冲突修复。
训练是否互相覆盖取决于进程、模型实例、输出目录及恢复身份；GPU 资源竞争还
取决于所选设备及实际显存用量。

## Changes made

保留全部 S3-A / S3-B 提交，通过普通 merge 合入依赖分支的配置、在线接续、
离线 Gate 和 Mask 概率诊断修正。没有额外修改业务代码。
旧开发验收收据仍说明当时的版本；本报告和新增收据记录本次授权依赖更新，
不把更新后的依赖哈希回填到旧保护快照。

## Verification performed

两仓库的合并差异 `git diff --check HEAD^1..HEAD` 均通过。针对直接影响的
公共 Mask 接口及 Actor 训练隔离运行最小 CPU 检查：**3 passed in 17.02s**。

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/v/Documents/work/gym-trading-env2/src:/home/v/Documents/work/dreamerv3-dev \
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
MPLCONFIGDIR=/tmp/fg_s3b_mpl \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -q -p no:cacheprovider \
  /home/v/Documents/work/dreamerv3-dev/dreamerv3/tests/test_action_mask.py::test_masked_probability_reports_zero_for_unavailable_add \
  /home/v/Documents/work/dreamerv3-dev/embodied/tests/test_fg_s3b.py::test_ones_byte_identical_to_s3a_and_analytic_weighted_gradient \
  /home/v/Documents/work/dreamerv3-dev/embodied/tests/test_fg_s3b.py::test_three_interfaces_exact_resume_isolation_and_updated_policy \
  --junitxml=/tmp/fg_s3ab_dependency_merge_20261012_tests.xml
```

上述检查实际覆盖：非法 ADD 概率为 0；全 1 权重与 S3-A 更新字节一致；
加权梯度与解析目标相符；非 Actor 参数及非零原优化器/Norm/Gate 状态不变；
MATCHED / CERTIFIED 的批次索引一致；断点恢复精确一致；更新后的 Actor 推理
同步；跨模式或 S3-A / S3-B Checkpoint 协议混用拒绝。

本次专属收据位于 `artifacts/fg_s3b_v1/dependency_merge_20261012/`：
`before.json`、`git.json`、`verification.json`、`tests.xml`。
未重复运行完整 B0–B4 数据审计或原 S3-A 测试集。

## Risks / not verified

当前实现下，S3-A 和 S3-B 分别从同一只读源 Checkpoint 启动独立模型实例，
使用各自输出目录、优化器和 Checkpoint，可以隔离训练状态。
S3-A 的协议为 `FG_S3A_BC_V1`，S3-B 对照入口的协议为 `FG_S3B_BC_V1`。
S3-B 内部三种模式 `S3A_FULL` / `S3A_MATCHED` / `S3B_CERTIFIED` 也绑定不同
恢复身份，不能互相直接 resume。它们共享的标签和冻结特征是只读输入。

不要让两个训练器更新同一 Agent 对象，也不要给它们相同输出目录。
创建新实验会拒绝已有目录，恢复会严格比较身份；这不代表允许同时写同一个
实验目录。本轮没有测试两个训练进程并发写同一目录。

同一张 GPU 并行训练会共享显存和算力。本轮未启动 GPU 训练，未测量并发显存
需求，因此不宣称当前 GPU 可以承受并行运行。正式源 Checkpoint 的完整性、
profile / update budget 及绑定身份仍由既有准入检查执行，没有放宽条件。
S3-A / S3-B 代码完成与当前 Replay 训练进度分别记录。

## Next recommended step

正式训练使用两个独立进程、不同输出目录和各自 run ID；公平比较使用相同
源 Checkpoint、标签、冻结特征、seed 和训练预算。单 GPU 优先顺序运行；有
足够独立 GPU 时，按设备分别运行。每次启动先执行对应 admission 检查，再
使用该实验自己的 Checkpoint 恢复。
