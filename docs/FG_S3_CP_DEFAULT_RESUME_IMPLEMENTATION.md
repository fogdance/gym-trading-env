# FG-S3 F1/F3 Checkpoint 默认恢复实施记录

## Summary

F1/F3 Resume 使用原 Run 签名 `run_identity.json` 中的 identity，继续由原 FullCheckpoint 严格检查该 Run 的 Checkpoint。当前代码版本及无关配置不再作为恢复许可条件；模型及优化器等实际状态不能完整加载时仍明确失败。

基线：Gym `1ae672aa86ca5a3c8e8c92d4de55d1eec6330b62`；Dreamer `8a393f08aacd3844367a98633087db1436f0e1b4`。开发位于 `fg-s3-cp-default-resume-20261021/{gym-trading-env,dreamerv3}`，分支均为 `codex/fg-s3-cp-default-resume-20261021`。

实施提交：Gym `ffb79398c5e6879afd11c3a39b8e1b5b3735fd1d`；Dreamer `990f61f8b7ac6bad25ff9e41b65624ae13ccf497`。本报告单独提交在 Gym 实施提交之后；按用户追加要求，将两仓交付提交快进合入 `integration/s3-next`，不追加测试。

## Evidence observed / Root cause

- 原 F1/F3 `start_prepared()` 直接比较 `identity.code`，以及整份当前/保存的 Prepared identity。基线定向测试实际得到 `F1_PREPARED_CODE_CHANGED` 和 `F3_PREPARED_CODE_CHANGED`。
- 原 FoundationRun/ActorRun 重建包含当前 code/packages 的身份，与历史 Run 全字典比较，再把它传给 FullCheckpoint。因此当前执行代码被误当成历史 Checkpoint 的归属。
- 仅复用 Run identity 仍不足以处理无关配置：`VersionedStream.binding` 包含适配器完整 settings。实际修改报告周期后，原路径报 `READER_CHECKPOINT_REFERENCE_IDENTITY`。
- Agent 原有加载先修改计数，再校验参数形状；没有显式比较所有模型/优化器槽位 dtype。因此 F1/F3 恢复入口增加实际序列化状态布局预检查，仍由原加载器恢复全部状态。

## Changes made

| 仓库/文件 | 改动 |
| --- | --- |
| Gym `research/fg_s3_e_quality_foundation.py` | 删除 F1 Prepared 源码与训练配置整体比较；保留 Producer pin、来源内容、Index 和授权校验。 |
| Gym `research/fg_s3_e_quality_actor.py` | 对称调整 F3；仅恢复路径跳过新 Run 参数接续时的目标模型配置比较。 |
| Gym `research/fg_s3_e_quality_actor_checks.py` | Evaluation 不再比较当前源码身份；检查原 Run/CP、完成状态、真实来源和实际 Agent 加载。 |
| Dreamer `embodied/run/fg_s3_e_quality_foundation.py` | F1 Resume 读取原 identity；共享原 Run 归属检查、实际 Agent 状态预检查及 Reader 引用归属恢复。 |
| Dreamer `embodied/run/fg_s3_e_quality_actor.py` | F3 Resume 同样使用原身份与完整恢复路径，不重新继承参数或初始化优化器。 |
| Dreamer `embodied/run/fg_s3_e_quality_transfer.py` | 新 Run 继续执行原目标模型匹配；恢复/评价仍核验来源工件、完成度和真实训练证据，不再重新判定目标配置是否相同。 |

Reader 的局部处理只在已验证 CP 后沿用其原引用 owner，随后继续执行原 `validate_checkpoint_stream`、Source/Index、跨成员版本图、pending batch 哈希及 FullCheckpoint 校验。没有更改 Reader、Prefetch 或 CP 格式，也没有修改通用 `fg_checkpoint.py`。

保留模型/Optimizer、策略镜像、pending sync/输出、Python/NumPy RNG、Reader/Prefetch、更新计数、调度和日志恢复。新保存的 CP 仍绑定原 identity；不迁移、不覆盖历史 Run identity 或 manifest。

没有新增 EXPLORE、训练模式、版本矩阵、Semantic ABI、兼容授权文件或训练 CLI 开关。`_resuming` / `_check_target` 仅是已有内部调用的分支参数，不授予训练权限。

## Verification performed

全部验证使用原 Dreamer Python 环境和 CPU 工程 Fixture；没有 FG809 真实更新。

通用环境前缀：

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 \
  JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/fg_cp_default_resume_mpl \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -q
```

在 Gym 工作树执行，前缀后分别追加：

```bash
tests/entry_v1/test_cp_default_resume.py \
tests/entry_v1/test_wp04_checkpoint.py \
tests/entry_v1/test_wp06_checkpoint.py \
--maxfail=1 \
--junitxml=artifacts/cp_default_resume/default_resume.xml

tests/entry_v1/test_wp04_resume_identity.py \
tests/entry_v1/test_batch2_reader_transactions.py \
../dreamerv3/embodied/tests/test_fg_course_strict_checkpoint.py \
--junitxml=artifacts/cp_default_resume/restore_guards.xml

../dreamerv3/embodied/tests/test_fg_s3_gate.py \
../dreamerv3/embodied/tests/test_fg_s3_source_transfer.py \
../dreamerv3/embodied/tests/test_action_mask_formal_guards.py \
--junitxml=artifacts/cp_default_resume/shared.xml
```

实际结果：

| 集合 | 结果 | 收据 |
| --- | --- | --- |
| 默认恢复 / WP04 / WP06 | 24 passed、1 failed、0 skipped，566.751 秒 | `default_resume.log`、`default_resume.xml` |
| Reader 事务 / strict CP / 变更定义恢复 | 29 passed、0 failed、0 skipped，130.786 秒 | `restore_guards.log`、`restore_guards.xml` |
| AGENTS 要求的共享回归 | 20 passed、0 failed、0 skipped，118.040 秒 | `shared.log`、`shared.xml` |
| 修正断言后单独复验 F04 | PASS，退出码 0 | `wrong_stage_recheck.log` |

首组唯一失败是旧断言仍匹配 `IDENTITY_MISMATCH`，实际程序已拒绝错误阶段并报 `OFFLINE_RESUME_RUN_STAGE_OR_BACKEND_MISMATCH`。已将断言更新为新的具体错误名，复用刚生成的 CPU Fixture 直接执行同一个测试函数，确认拒绝且 Run 文件字节不变。未重跑整组；保留原失败 JUnit，不将其改写成全绿。总计 pytest 73 项通过，另 1 项修正后定向复验通过，没有跳过。

该单项复验的实际命令：

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 \
  JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/fg_cp_default_resume_mpl \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/v/miniconda3/envs/dreamerv3/bin/python -B -c 'import json,tempfile; from pathlib import Path; from test_wp06_checkpoint import test_F04_f1_checkpoint_is_not_f3_resume; root=Path("/tmp/pytest-of-v/pytest-790/wp06_cold0"); s=json.loads((root/"settings.json").read_text()); output=Path(tempfile.mkdtemp(prefix="wrong_stage_",dir="artifacts/cp_default_resume")); test_F04_f1_checkpoint_is_not_f3_resume((root,s,root/"settings.json",root/"join.sqlite"),output); message="PASS test_F04_f1_checkpoint_is_not_f3_resume (existing CPU fixture, unchanged Run files)"; (output/"result.txt").write_text(message+"\n"); print(message)' \
  > artifacts/cp_default_resume/wrong_stage_recheck.log 2>&1
```

| 要求 | 验证内容 |
| --- | --- |
| T1–T2 | 两阶段 Prepared 后改变 code 和模型配置元数据，Start 编排继续；来源/Index 攻击仍拒绝。Native 执行和 Runner 使用定向替身，不运行 G3。 |
| T3–T4 | 两阶段 CPU 连续两步，与一步→退出→新进程改变 HEAD/hash/dirty 元数据→恢复第二步，比较完整 Agent/Optimizer、RNG、Reader/Prefetch、下一批、计数和日志。 |
| T5 | 旧 identity 和旧 CP 文件保持字节不变；新 CP 仍用原 identity，再次新进程恢复，并允许报告频率变化。 |
| T6 | 仅在 Fixture 副本构造有效成员 SHA 的模型 dtype / Optimizer 槽位结构变化；加载明确失败，不发布新 CP，不改旧 CP。 |
| T7 | 最新节点损坏、成员缺失、latest 回退，以及原 strict-latest / Reader 事务回归。 |
| T8 | CPU Fixture 实际 F1→F3 参数接续；源码元数据变化后调用原 Evaluation 入口，加载真实保存的 Agent，并执行原生 Actor/Driver/Env/Replay/reconciler，Teacher 构造被禁止。 |
| T9 | 实际 Index、Corpus Replay 字节变动仍拒绝。 |

T8 的合成 Corpus 没有生产 source-pool/account 元数据，测试仅替换该外部元数据桥接；不将其当作真实 TRAIN 来源认证。已有真实 Fixture 来源准入在此前执行；模型加载与原生 Actor/金融环境实际运行。

实际 Evaluation 收据记录：`market_advance=8`、`teacher_used=false`、`all_parameter_bytes_equal=true`、`complete_account_replay.status=PASS`、`real_TRAIN_updates=0`。

原始日志、JUnit 和子进程命令位于 `artifacts/cp_default_resume/` 及本次 pytest Fixture 目录。旧测试产生的少量固定名称收据另存 `artifacts/cp_default_resume/fixture_receipts/`，工作树中原历史收据恢复原字节。

已审查两仓实际 diff：生产改动仅上述六个文件；历史收据未提交；通用 `fg_checkpoint.py`、训练数学、模型结构和生产规则未改。两仓 `git diff --check` 通过。

## Risks / not verified

- 未启动 FG809 真实训练、GPU 训练或完整 G3；没有复制/修改生产 Corpus、生产工作树或现存用户 Run/Checkpoint。
- 允许加载不代表修改前后的算法语义一致；本实现不作数学语义判断。
- 实际参数、槽位或 Reader/Prefetch 等必需状态不能完整加载时仍失败，不重置、不部分成功、不回退。
- 原 native 来源准入及启动/恢复时的 G3 调用保持原样；本轮没有实现认证缓存或减少 G3。
- 验证范围限上述定向测试，不声称全项目通过。

## Next recommended step / existing CLI

以下仍为原 CLI；示例未执行。使用原 Prepared、Run 和原已审批 Producer pin，训练授权要求不变：

```bash
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
"$PY" -B tools/fg_s3_v2_experiment.py f1-resume \
  --prepared "$F1_PREPARED" --producer-gym "$PRODUCER_GYM" \
  --producer-lock "$PRODUCER_LOCK" --producer-lock-sha256 "$PRODUCER_LOCK_SHA256" \
  --authorize-training

"$PY" -B tools/fg_s3_v2_experiment.py f3-resume \
  --prepared "$F3_PREPARED" --producer-gym "$PRODUCER_GYM" \
  --producer-lock "$PRODUCER_LOCK" --producer-lock-sha256 "$PRODUCER_LOCK_SHA256" \
  --authorize-training
```
