# WP05_IMPLEMENTATION_REVIEW

## Summary

**WP05_CODE_GO（CPU 工程）/ REAL_DATA_READ_GO（LIMITED）；TRAIN_GO=NO_GO，MODEL_RESEARCH_GO=NO_GO。GPU 更新验收未通过。**

A–F 已实现：四头 F2、非优化器参数接续、逐头有效损失与精确冻结、唯一 Reader 诊断、原 FullCheckpoint 精确恢复及统一 CLI。本次未提供正式受训 F1 权重；真实 TRAIN 更新为 0；没有读取 TEST、长 GPU 训练或 18 合约生产。

独立成对工作树为 `fg-s3-e-quality-wp05-20261018/{gym-trading-env,dreamerv3}`，双仓分支 `feat/s3-e-quality-wp05-20261018`。父提交 Gym `b7d62113f1e8d4d7ddbdce175acd83568422420e` / Dreamer `9715352d4e548b82ba24d3b31070b9fa20d345a6`；创建前 WP04 均 clean。没有合并 integration 或 DATA 分支。

六阶段双仓真实 SHA 见 [提交表](WP05_COMMITS.md) 和提交后生成的 [交付锁](WP05_DELIVERY_COMMIT_LOCK.json)。最终 HEAD 的冷恢复原件为 `receipts/wp05/F_final_recovery_v2.*`，提交后生成并保留本机 docs；不以旧收据代替新执行。

## Evidence observed

实际选定 CPU 用例、准确命令、退出码、HEAD/dirty、原始失败及最终清单见 [测试结果](WP05_TEST_RESULTS.md) / [机器清单](receipts/wp05/F_selected_test_inventory.json)。用真实文件和 collected test name 去重，避免 JUnit 空 classname 或 pytest root 差异重复计数；选定集合不是全项目。

CPU CLI 内置合成源完成 4 次原优化器更新：四头各 5 参数叶变化；349 状态叶中，285 冻结叶逐字节相同。允许集合为 20 头参数、40 对应 slots、4 共享时钟；无效或关闭的头继续冻结。见 `C_*_delta.json`、`E_inherited_foundation_freeze.json` 及 [CPU Probe](../artifacts/fg_s3_v2/e_quality_wp05/CPU_probe_01/probe.json)。

全部非 `opt/` 参数及正常状态逐键继承，优化器、Reader、计数 fresh，Policy mirror 同步且 pending 清空。[迁移收据](receipts/wp05/B_transfer_receipt.json) 明确 `ENGINEERING_ONLY / FOUNDATION_NOT_TRAINED`。正式来源额外要求完整训练记录、初始完整节点、全状态 schema/finite、对应计数及实际 WM 参数变化，不能仅凭 JSON 声明已完成；本轮没有真实正式 F1 正例。

连续 4 与 2+冷进程+2 的 Agent、优化器、Reader RNG、原 Prefetch queue/seed、Python/NumPy RNG、计数、日程、指标及下一批字节一致。无效 batch 消费后 CP 位置为 1、优化器仍 0，恢复不重复 handoff。见 [冷恢复](receipts/wp05/E_cold_evidence.json) / [跳过恢复](receipts/wp05/E_skip_evidence.json)。Checkpoint 不复制整个 Corpus；Runtime 包含恢复所需的待消费批次输入和标签。

## Root cause / Review findings

课程矩阵参与 Corpus 哈希，直接改四头会破坏身份，因此保留 V2 上界，增加独立执行子集。原预测目标耦合 Q，F2 提取共享纯预测项；零梯度不能冻结 Adam/衰减，采用逐头状态权限和原值恢复。自审补上动态规则与正白名单交集，注入 WM 规则负例已拒绝。

无效 batch 不能推进优化器，但须保存 Reader 消费，因此 Checkpoint 的 batch 位置与优化器计数分离。完成审查另复现 P1/K90/T8 的有效标签全在 burn-in、所有目标段均无效；新增原索引上的启用损失支持检查，prepare/Run 在模型创建前拒绝。检查不改变原 Reader 分布、不重采样、不推进 RNG；单个无效批次仍按原规则记录跳过。正式来源自审发现仅有节点/计数不足，增加只在 FORMAL 要求的完整训练记录及 WM 真实变化验证，保留受控工程假来源。

早期构造器 mode 白名单遗漏已修正。NaN 负例的 event 夹具误用浮点类别，改用整型非法索引，风险/评分 NaN 保留。短 K4/T8 的 15m 零消融曾无预测变化；没有据此修改 Encoder。原生 K32/T64 下 Encoder、posterior、四头输出均响应；短窗口具体原因未认定为生产缺陷。

GPU：沙箱缺少设备节点；真实主机可见 RTX 4060 Ti/595.84，但主机重跑仍 `cuInit CUDA_ERROR_UNKNOWN / No visible GPU devices`。没有确认驱动层根因，没有改系统；GPU 模型更新未完成，原始 exit 1 见 `F_gpu_probe`、`F_gpu_host`。CPU 结果不替代 GPU 验收。

## Changes made

A：F2 协议、配置、四头子集、完整模型初始化和 HEADS_VALID；B：参数 handoff、来源/镜像/计数；C：共享原 CE/BCE/Huber/≥80/合法 mono，WM detached，逐头状态权限、无效跳过及非有限整次回滚；D：三种采样、分母/支持度/评分与多尺度诊断；E：薄阶段协调器复用原 Agent、Reader、Optimizer、Prefetch、RuntimeState、FullCheckpoint；F：统一 f2-*、权限负例、集成验收、正式 F1 训练证据校验及文档。

职责按 AGENTS 放在 Gym research、Dreamer dreamerv3 和 embodied/run，没有第二套 Env、Replay、World Model、Optimizer 或 Trainer。课程 V2、金融 Env、Reward/Score、Observation/MACD、原 Encoder/RSSM/WM、FullCheckpoint 实现字节未改，见 `F_protected_source_audit.json`。

## Verification performed

[54 项验收矩阵](WP05_ACCEPTANCE_MATRIX.md) 映射真实测试；[源码反向 Review](WP05_SOURCE_REVIEW.md) 说明边界。共享 Gate/SourceTransfer/formal guards、原 FullCheckpoint/retention/S3C resume 38 PASS；Reader/Join/teacher loss/MACD/市场分钟 83 PASS；原 F1 WM 与 legacy Quality 26 PASS。CPU 选定集合最终通过；不表示 GPU 或全项目通过。

原 FG201 主力资格末两官方交易日（2021-12-08/09）的 TRAIN Corpus：4146 物理行、690 独立市场 cursor、6 账户。本轮重新验证来源/Join，并做 12 个 NATURAL HEADS_VALID 窗口前向：768 target 暴露、165 独立 target cursor、764 不同 target 账户物理行，全部 349 叶前后相同、更新 0。见 [真实只读报告](receipts/wp05/F_real_read_02/read_check.json) / [覆盖报告](WP05_COVERAGE_REPORT.md)。

旧 WP02/03/04 双仓仍为原 HEAD 且 clean，见 `F_repository_protection.json`。失败产物、日志、Checkpoint 均保留；没有导出 patch。

## Risks / not verified

真实受训 F1、真实 F2 参数更新、FORMAL start/resume 正例、多合约独立校准、正式 G3 本轮重跑、成功 GPU 更新、长训练、TEST、18 合约、Actor 自主交易和外部人工 Review 均 **NOT_RUN**；GPU 尝试本身为 **ATTEMPTED_FAILED_ENV**。

真实只读使用 fresh 小模型诊断权重，不消费课程不匹配的合成 F1；该配对 prepare 负例已拒绝。独立训练/验证后缀切分与研究校准为 **UNSUPPORTED**。NATURAL 是 HEADS_VALID 合法窗口条件分布，不等于无条件市场入场概率；不把两日 smoke 指标称为泛化。

工程测试使用本机保留的 WP04 合成节点；内置 CLI probe 在其缺失时可用原 fixture/Run 重建合成源。K 是有限历史近似。没有把 WP04 旧 G3 报告转换成本轮训练授权。

## Next recommended step

真实 F1 来源、足量可信课程、预算和授权齐备后，再评审正式 F2；本轮不启动。GPU 环境修复另行处理。WP06 工程可独立推进，不由 F2 Prediction Admission 阻塞。
