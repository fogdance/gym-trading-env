# 2026-10-16 Quality-first 审查修复收据

[逐项修复记录](../../FG_S3_QUALITY_FIRST_REVIEW_FIXES.md) 说明 F1–F8 的根因、代码合同和研究边界。[机器摘要](delivery_summary.json) 记录 151 项不重复测试、实际有效位、梯度、冻结及未认证项目。

[归档清单](archive_manifest.json) 保存 86 个原始文件的来源、长度和 SHA256；副本字节不改写。行情与完整 Checkpoint 留在原路径，不重复提交。归档之外的原始运行目录仍保留，旧 150 项交付见 [历史索引](../quality_first_delivery/README.md)。

[源码绑定](source_git_binding.json)：Gym 代码基线 `dd0fe47a3bd43e33c10d6a71e16407efd6d4b50d`，Dreamer `d33985cb2989dcf2e3fe6dc06584ffb757011c3d`；211 个运行源码文件及 1 个独立诊断工具均逐字节匹配 Git。最终交付的后续文档提交不改变这些源码。三份最终 prepared 和真实 CPU Run 的运行源码映射也完全一致；未覆盖旧实验。

## 实际测试

| 范围 | 结果 | 原始 XML / 日志 |
| --- | ---: | --- |
| 数学、Native Teacher、Catalog、Gate、原 Env/Oracle、组合、共用 Tracker | 108 / 108 | [XML](../quality_review_regression_final.xml) · [日志](../quality_review_regression_final.log) |
| 真实 JAX CPU Student | 8 / 8 | [XML](../quality_review_student_final.xml) · [日志](../quality_review_student_final.log) |
| 初始 policy 冻结、实际头参数更新、最终预算限制、FullCheckpoint 精确恢复及 posthoc Q 预算 | 1 / 1 | [XML](../quality_review_lifecycle_final_v2.xml) · [日志](../quality_review_lifecycle_final_v2.log) |
| 当前共享合同回归（六组，排除 GPU） | 34 / 34 | [XML](../quality_review_contracts_v1.xml) · [日志](../quality_review_contracts_v1.log) |

合计 **151 项，无失败、无跳过**。不累计目标重跑。首次命令写错预测测试文件名的 exit 4 原始输出仍在 `quality_review_native_final.log`；后续真实路径均通过。没有增加旧测试兼容分支。

解释器 `/home/v/miniconda3/envs/dreamerv3/bin/python -B`，`PYTHONPATH=src:../dreamerv3:tests/entry_v1`，`JAX_PLATFORMS=cpu`，`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、`PYTHONDONTWRITEBYTECODE=1`；pytest 使用 `-p no:cacheprovider`。Student 最终重跑使用独立 `--basetemp=artifacts/fg_s3_v2/quality_checks/QUALITY_FIRST_REVIEW_STUDENT_v2`，保留各头的原始数值收据。

主验收文件为 `test_quality_tp_reward_v1.py`、`test_fg_s3_quality_teacher.py`、`test_fg_s3_quality_patch_acceptance.py`、`test_fg_s3_quality_teacher_v2.py`、`test_fg_s3_quality_prediction_v2.py`、`test_fg_s3_quality_admission_v2.py`、`test_fg_single3_env.py`、`test_fg_single3_oracle.py`、`test_s3_composition.py` 及 `research_tracking/tests/test_quality.py`。六组共享合同的实际命令与旧夹具根因见 [历史专项](../../FG_S3_QUALITY_REGRESSION_FIX.md)；本轮只有真实结果重跑，无兼容层改造。

## 严格候选与资源

| 合约 | 候选数 | 原生抽验 cursor / 风险分层 | 选中严格路径逐笔重放交易数 | 原生完整推进 |
| --- | ---: | ---: | ---: | ---: |
| FG201 | 31,259 | 254 / 66 | 3,255 | 31,259 |
| FG205 | 23,219 | 248 / 74 | 1,708 | 23,219 |

证据：[双合约完整严格见证](pool_prepare/review_native_strict_proofs.json)、[构建资源](pool_prepare/quality_build_performance.json)、[原始执行输出](../quality_review_final_native_proofs.log)。每个原生反馈、退出结算、ADD 序列、总现金及最终空仓均一致。抽验复用原 128 随机与既有边界规则，另加质量分层；**未逐一原生重放所有单独候选**。

双合约最终构建的候选阶段：FG201 约 163.54 秒、峰值 RSS 608,239,616 字节；FG205 约 123.46 秒、493,621,248 字节。持仓分钟 `[min,p50,p90,p99,max]` 分别为 `[1,13,69,153,316]` 和 `[1,21,114,262,483]`，独立候选持仓总和为 829,560 和 1,001,923 市场分钟。worker=1 是最大合约 pilot 后只剩一个独立任务；完整池的资源并行分配公式已实现，**18 合约实际并发峰值未验收**。

## Catalog、训练和模型变化

新真实 Run 为 `artifacts/fg_s3_v2/runs/QUALITY_FIRST_REVIEW_CPU_FG201_v2`，CPU 100 Env / 2 次更新；[收据](native_train/training_receipt.json)、[完整参数摘要比较](native_train/review_model_evidence.json)、[每次梯度](native_train/updates.jsonl)。

| 头 | 第二次更新梯度范数 | 参数变化范数 |
| --- | ---: | ---: |
| Event | 0.1687844992 | 1.1836698e-6 |
| Touch | 0.04275016487 | 9.8856981e-7 |
| Adverse | 0.06405421346 | 8.9388010e-7 |
| Score | 0.1021376550 | 8.9683624e-7 |

首次 warmup LR=0；第二次四头实际改变。其余 **89 组非优化器参数摘要不变**，Actor/QualityActor/Critic/Backbone/双 Q 梯度和参数变化均为 0。[Full 合成监督](synthetic_backend/joint_cpu_receipt.json) 则记录七个新模块、各 Actor Loss 项的实际梯度，以及 stop-gradient、无支持分支冻结与精确下一次更新恢复；不是实际预测或 Q 准入。

[250 步 Catalog](native_catalog/quality_catalog.json)：12 个合法空仓候选均可评分，7 个严格有效；5 个低资金候选有效评分、严格认证 0；238 条持仓被迫 WAIT，10 TP/2 SL 为独立反事实，不能相加作为实际收益。真实数值 Q 有效数 0，落在默认 12 根精确域的潜在候选也为 0 / 12。[资源收据](native_catalog/receipt.json) 同时记录日志字节、回放耗时和缓存计数；此单次路径 cache hit=0，不宣称吞吐提升。

[恢复收据](lifecycle/receipt.json)：100→resume200 与连续200 的全部更新指标和最终参数摘要相等；初始 FullCheckpoint 绑定不可变合成 policy；中间保存点不能用于准入；Q 评价预算来自原训练准备。合成阈值不能用于研究。

## 只读评价与决策边界

| 范围 | 结果与证据 |
| --- | --- |
| DEV Actor-only 100 步 | 8 个可评分候选，Score MAE 24.2057153，TRAIN 中位数 MAE 15.3197464；[补评](dev/posthoc/receipt.json) |
| FG401 质量监督留出 100 步 | 11 个候选，MAE 49.5793882；[补评](holdout/posthoc/receipt.json) |
| 同状态初始参考 | 独立 carry、101 条因果记录和相同原生标签，参考 MAE 49.5793855；未证明改善；[对照](matched_comparison/receipt.json) |
| Outcome+Score 私有干预 | 100 步全 WAIT，0 成交；100 个独立可评分反事实不是实际交易成绩；[补评](diagnostic/posthoc/receipt.json) 明确 `autonomous_actor_performance=false` |
| 独立原生动作审计 | 基线 8 / 8、干预轨迹 100 / 100 个空仓状态严格可认证；两条完整账户最终状态都与封存推理一致；[基线](native_action_audits/dev/receipt.json) / [干预](native_action_audits/diagnostic/receipt.json) |
| 原生安全支持拒绝 | 基线 1 ENTRY、0 EXIT，存在右删失；干预 0 ENTRY。二者连最小 1 笔已结束交易的支持条件都不满足，因此拒绝安全准入；[边界收据](native_action_audits/decision_boundary_receipt.json) |

同基线历史上唯一动作改变位于 cursor 7，ADD→WAIT，与该状态原生严格 WAIT 一致；这个单点不构成多合约、时间 fold、seed 或成交闭环的研究 PASS。实际未来标签均在预测封存后生成。FG401 留出只针对新质量监督，原 Backbone 见过原 TRAIN，不能称从未见过。

## 来源与未验证

首个 GPU 技术提案 `QUALITY_FIRST_REVIEW_HEADS_GPU_1K_v2` 已 [CPU prepare](source_1m_prepare/manifest.json)，原 Tutor Full 来源 1M Env / 40,000 更新；[原始 Checkpoint manifest](source_1m_prepare/untouched_original_checkpoint_manifest.json) SHA256 仍为 `d6b03a8f2e3604885a7550c9bd217471e290c7840f91dd1ecc93bf93658e9098`。没有在全尺寸目标上执行 GPU 参数迁移或更新。

状态保持 `PREDICTION_GATE_PENDING / DECISION_GATE_PENDING / FULL_Q_PENDING / NOT_CALIBRATED`。生产阈值和未来未观察窗口未冻结，实际多合约/fold/seed 预测/决策认证、真实数值 Q 覆盖、合格来源的 Full/Q 正向迁移链路、18 合约/1M 资源、GPU 更新及恢复均未验收。GPU/10K/1M 未启动，TEST 未解封。启动与完整对照见 [提案](../../FG_S3_QUALITY_FIRST_RESEARCH_LAUNCH_PLAN.md)。
