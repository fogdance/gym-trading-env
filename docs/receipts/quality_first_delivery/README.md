# Quality-first 实际验收收据索引

本目录保存执行输出的原始字节副本；原签名、绝对来源路径及执行时 Git/dirty 状态不改写。[归档清单](archive_manifest.json) 记录 72 个副本的来源和 SHA256。行情数据与完整模型 Checkpoint 保留在原路径，不重复放入 Git。

- [机器可读交付摘要](delivery_summary.json)：150 项测试、实际覆盖、低资金候选、梯度／参数变化、预测指标和未认证项。
- [210 个运行源码文件的实际哈希](executed_source_match.json)：最终 CPU Run 与真实 1M 来源 CPU prepare 的代码逐一一致。最终 Git 绑定另在原运行的 `final_git_binding.json` 生成，交付 SHA 见最终答复。
- [源码提交绑定](implementation_git_binding.json)：运行文件与 Gym `f71124a45d52b80b862cf7f813f1491d8a901de7` / Dreamer `29ac494b6e3cdbd405a262f66dfcc4a040219ef2` 逐字节相等；最终文档提交保留该源码。
- [实施记录](../../FG_S3_QUALITY_FIRST_IMPLEMENTATION.md)、[回归根因与清单](../../FG_S3_QUALITY_REGRESSION_FIX.md)、[研究启动提案](../../FG_S3_QUALITY_FIRST_RESEARCH_LAUNCH_PLAN.md)。

## 测试

| 实际命令所选测试 | 结果 | XML / 日志 |
| --- | ---: | --- |
| Quality 数学、教师／分层 Catalog、时钟／资金／右删失、原 Env／Oracle、组合及 Tracker | 107 / 107 | [XML](../quality_first_delivery_regression.xml) · [日志](../quality_first_delivery_regression.log) |
| 真实 JAX CPU Student 后端，含保留 Full 合成监督与默认 HEADS_ONLY | 8 / 8 | [XML](../quality_first_delivery_student.xml) · [日志](../quality_first_delivery_student.log) |
| 共享入口 prepare→100→resume 200，与连续 200 精确比较及 Catalog 前缀验证 | 1 / 1 | [XML](../quality_first_delivery_lifecycle.xml) · [日志](../quality_first_delivery_lifecycle.log) |
| 六组现有合同回归，修复旧测试夹具后 | 34 / 34 | [XML](../quality_first_compatibility_fixed.xml) · [日志](../quality_first_compatibility_fixed.log) |

共 150 项，不重复计入单独目标重跑。[目标回归](../quality_first_compatibility_fix_targeted.xml) 包括奖励不匹配仍须拒绝的断言。修复前的 [开发树失败](../quality_first_delivery_shared_compatibility.xml) 和 [干净旧基线失败](../preexisting_actor_metrics_c3a8084.xml) 原样保留。两套独立 Patch 基线各 41 项另见 [Patch 验收](../../quality_first_patch_review_20261015/ACCEPTANCE.md)。

解释器固定 `/home/v/miniconda3/envs/dreamerv3/bin/python -B`；`JAX_PLATFORMS=cpu`、`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`，pytest 使用 `-p no:cacheprovider`。六组现有合同回归实际命令如下，GPU 专项排除：

```bash
python -B -m pytest -q -p no:cacheprovider \
  tests/unit/test_fg_reward_candidates.py \
  tests/entry_v1/test_fg_reward_offline.py \
  tests/entry_v1/test_fg_s3_v2_pool.py \
  tests/entry_v1/test_fg_s3_v2.py \
  tests/entry_v1/test_fg_s3_v2_observation.py \
  tests/entry_v1/test_fg_s3_tutor_e2.py -k 'not gpu'
```

## 实际运行

| 范围 | 证据 | 有效边界 |
| --- | --- | --- |
| 原生 TRAIN Catalog，250 步 scripted ADD | [Catalog](native_catalog_250/quality_catalog.json) · [旁路日志](native_catalog_250/teacher_journal.jsonl) · [来源校验](native_train_100/independent_catalog_validation.log) | 12 个合法空仓候选，12 个可评分；5 个低于 30 万元均无严格动作认证；238 个持仓被迫 WAIT |
| 实际神经训练，100 Env / 2 次联合更新 | [运行收据](native_train_100/training_receipt.json) · [更新明细](native_train_100/updates.jsonl) · [Checkpoint manifest](native_train_100/checkpoint_manifest.json) | 四个辅助头各发生 1 次非零更新；89 组其他非优化器参数摘要完全不变；首次 warmup LR=0 |
| CPU 精确恢复 | [恢复收据](lifecycle/receipt.json) · [100 步 Catalog](lifecycle/quality_catalog_step_000000000100.json) · [200 步 Catalog](lifecycle/quality_catalog_step_000000000200.json) | 恢复与连续运行的全部更新指标及参数摘要一致；日志追加允许，前缀篡改拒绝 |
| Actor-only DEV | [推理](dev/inference/receipt.json) · [补评](dev/posthoc/receipt.json) | 100 步，8 个评分候选；MAE 24.20572，TRAIN 先验 15.31975；未达到预测门槛 |
| Actor-only FG401 质量监督留出 | [推理](holdout/inference/receipt.json) · [补评](holdout/posthoc/receipt.json) | 未加入质量训练；原 Backbone 之前见过原 TRAIN，不能称从未见过合约 |
| 同状态参考 | [参考推理](reference/inference/receipt.json) · [参考补评](reference/posthoc/receipt.json) · [比较](matched_comparison/receipt.json) | 101 条因果记录／11 个评分候选，独立模型 carry、相同执行历史及原生标签；参考回放不作为自主收益 |
| 真实 Tutor Full 1M 来源 CPU 准入 | [准备](source_1m_cpu_admission/manifest.json) · [SourceAdmission](source_1m_cpu_admission/source_admission.json) · [原 Checkpoint manifest](source_1m_cpu_admission/untouched_original_checkpoint_manifest.json) | 来源 40000 更新／1M Env，源 manifest d6b03a8f…；PREPARED_NO_TRAINING，未在全尺寸目标上执行参数迁移／GPU 更新 |
| 合成后端 Full / HEADS_ONLY | [Full](synthetic_backend/joint_cpu_receipt.json) · [HEADS_ONLY](synthetic_backend/heads_only_receipt.json) · [warm-start](synthetic_backend/warm_start_receipt.json) | 模块梯度、冻结、stop-gradient、恢复与参数合同；不是预测效果或研究准入 |

## D0–D7 对照

| 阶段 | 已实现与实测 | 仍未认证 |
| --- | --- | --- |
| D0 | 双仓真实调用链、Reward／Oracle／来源源码审计；旧数值 Q 为 null | 历史交易所费用真实性不新增认证 |
| D1 | 原 Patch 独立验收，原生账户 Teacher、B(t,d)、官方交易日和低资金事实 | E 明确拒绝；极低保证金真实样本未覆盖 |
| D2 | 四层 Catalog、真实 TRAIN、Replay 所有权、逐任务有效位、来源校验与覆盖 | 全量 18 合约吞吐／长期右删失资源成本 |
| D3 | 默认辅助四头；新 Actor、多窗口 Outcome、Score、双 Q 保留并后端测试 | Full 实际研究训练及真实 Q 有效覆盖 |
| D4 | Natural/Balanced/E2、Strict/Quality Pair、Soft、多任务与预测→logit，梯度边界实测 | 未通过预测／Q 门槛，生产配置不启用其 Actor 影响 |
| D5 | 共享 SourceAdmission、参数继承合同、FullCheckpoint 精确恢复 | 真实合格预测来源的 Full 迁移；全尺寸 GPU 迁移／恢复 |
| D6 | DEV／FG401 留出、先封存后补评、同状态参考和来源证明 | 多 seed、多时间 fold、校准／策略改善 |
| D7 | 150 项 CPU 测试及实际收据；初次 GPU／10K／1M 提案 | GPU 未启动；TEST 未解封 |

`PREDICTION_GATE_PENDING / NOT_CALIBRATED / FULL_Q_PENDING` 保持原状态，没有根据工程验收制造研究 PASS。
