# Quality-first 后续复审：P0／P1 修复收据

[逐项根因、修改、风险口径和 TODO](../../FG_S3_QUALITY_FIRST_FOLLOWUP_FIXES.md)。工程修复通过；Quality Full **NOT GO**。预测、Q 和决策安全均 PENDING，无 GPU／10K／1M 启动。

- [机器摘要与实际命令](delivery_summary.json)：最终不重复计数 115 项，114 项 Quality 回归＋1 项独立 CPU 生命周期，无失败／跳过；18 项定向回归不重复累计。完整回归与生命周期并行运行。
- [源码绑定](source_git_binding.json)：211 个运行源码文件逐字节匹配 Gym `3e62851483f07ea9a61ad188e911d0fbb0202429`、Dreamer `d33985cb2989dcf2e3fe6dc06584ffb757011c3d`；新生命周期 Run 与 prepared 的运行源码映射完全一致。
- [原始字节归档清单](archive_manifest.json)：复制文件的原路径、长度和 SHA256。之前 86 个归档副本及原始文件再次逐一核对，均未改变。

## 测试证据

| 范围 | 结果 | 原始文件 |
| --- | --- | --- |
| 修复前目标复现 | 6 失败，0 error | [XML](tests/quality_review_followup_before.xml) · [日志](tests/quality_review_followup_before.log) |
| 第一次实现 | 18 项，2 失败；起点时间误作终点游标，被原生回归发现 | [XML](tests/quality_review_followup_targeted.xml) · [日志](tests/quality_review_followup_targeted.log) |
| 定向修复验证 | 18 / 18 | [XML](tests/quality_review_followup_targeted_v2.xml) · [日志](tests/quality_review_followup_targeted_v2.log) |
| 最终 Quality 回归 | 114 / 114 | [XML](tests/quality_review_followup_regression.xml) · [日志](tests/quality_review_followup_regression.log) |
| 最终 CPU FullCheckpoint 生命周期 | 1 / 1 | [XML](tests/quality_review_followup_lifecycle.xml) · [日志](tests/quality_review_followup_lifecycle.log) |

P0 回归显式隔离上游研究门禁，只验证消费者能接受不含 `trained_contracts` 的收据并从绑定 Policy 读取。未制造真实预测准入 PASS。源码仅修改既有两个模块和一个 admission 测试文件，没有增加旧测试兼容代码。

## 原生交易与风险

[完整原生复现](native_cohort/receipt.json) 记录合成行情、真实原生 Env 的动作、逐根权益、原生成交、平仓 attribution 与输入文件哈希；[执行日志](logs/quality_review_followup_native_cohort.log)。输入 [行情](native_cohort/market.csv)、[环境配置](native_cohort/env.yaml)、[合约配置](native_cohort/contract.yaml)、[方向](native_cohort/directions.yaml) 原样归档。

同一自然日跨两个官方交易日执行 SL→TP→SL，净损益依次为 −1,818／282／−1,818 元。Fold A 只拥有第一笔：SL 率 1、完整回撤 1,818；Fold B 拥有后两笔：SL 率 0.5、回撤 1,818。其前置权益锚点已扣除带入 A 的损失，不把该损失算入 B。Fold A 跨 Fold 已平仓结果与截止时未平仓、最终右删失分别单列。[回归风险收据](native_cohort/entry_fold_risk_receipt.json)、[右删失收据](native_cohort/censored_fold_risk_receipt.json)。

[既有真实账户重算](real_accounts/receipt.json) 先验证封存文件和先前独立原生回放的最终状态哈希，再计算新风险口径：DEV 为 RIGHT_CENSORED，零成交诊断为 INSUFFICIENT_CLOSED_SUPPORT。此处没有新推理／训练，最低闭环样本 1 仅为诊断下界，不是事前生产 Policy。[正确调用日志](logs/quality_review_followup_native_accounts_v2.log)；首个调用未规范化内部官方日格式的[失败日志](logs/quality_review_followup_native_accounts.log)保留。

## CPU 模型更新与恢复

[生命周期收据](lifecycle/receipt.json)、[训练收据](lifecycle/training_receipt.json)、[实际每次更新](lifecycle/updates.jsonl)、[Run 身份](lifecycle/run_identity.json)。新 Run `QUALITY_CPU_ACCEPTANCE_7c0ca2e2c5` 为合成原生监督，200 Env／4 更新；Score 有效 106，Event 742、Touch 848、Adverse 424，双 Q 有效 0。

[逐参数初始／最终摘要与梯度](model_update_evidence.json)和[独立摘要审计](model_update_validation.json)确认四个预测头的 20 个非优化器参数键改变，其他 89 个非优化器参数键不变。Actor、QualityActor、Critic、Backbone 和双 Q 均冻结。100→resume200 与连续 200 的全部更新指标和最终参数摘要相等。

零更新 Checkpoint 拥有不可变事前 Policy；[初始](lifecycle/initial_checkpoint_manifest.json)／[最终](lifecycle/final_checkpoint_manifest.json) manifest 原样保留。[合成 Policy](lifecycle/synthetic_policy_fixture.json)仅供测试，禁止作为生产研究默认门槛。prepare／start100／resume200／uninterrupted200 的原始日志均在 `lifecycle/`。

之前真实 FG201／FG205 Catalog、亏损账户、真实市场 CPU 头更新和 Full 合成梯度证据见[前次完整交付](../quality_review_delivery/README.md)，本轮不把历史测试重复计入 115。
