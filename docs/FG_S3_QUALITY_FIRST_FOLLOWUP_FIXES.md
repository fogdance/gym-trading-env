# Quality-first 复审三项修复与验收（2026-10-16）

本次处理 gy16/d316 后续复审提出的 P0、两项 P1。三项均先复现、再修复；工程回归通过不代表 Quality Full 研究准入。预测、Q、决策安全仍为 **PENDING**，没有启动 GPU、10K 或 1M 训练。

源码提交：Gym `3e62851483f07ea9a61ad188e911d0fbb0202429`；Dreamer `d33985cb2989dcf2e3fe6dc06584ffb757011c3d`（本轮未改）。[收据索引](receipts/quality_review_followup_delivery/README.md) 保存本轮结果、来源绑定和原始文件校验和；之前的 [151 项交付](receipts/quality_review_delivery/README.md) 原样保留。

## 观察与确认的根因

| 问题 | 修复前实际证据 | 最小根因 |
| --- | --- | --- |
| P0：CONTROLLED_FULL 准备接口 | 合法形状的准入收据不含 `trained_contracts`，准备抛出 `KeyError: 'trained_contracts'` | 消费者误把事前 Policy 字段当成准入收据必填字段；验证器返回的确是原始收据 |
| P1：二元预测的对照门槛 | 最后一个 Touch 或 Adverse 窗口较初始参考差，但仍满足绝对样本／校准门槛；旧 `_improvements()` 未拒绝 | 仅检查七个事件窗口和 Score，缺少 Touch／Adverse 的相对 NLL／Brier 比较 |
| P1：跨 Fold 风险 | 原生账户在 Fold A 入场、Fold B 止损，净损失 1,818 元；旧 Fold A 回撤仅 9 元开仓费 | SL 使用完整交易结局，回撤只截取 Fold 内权益，二者暴露范围不同；且未包含首次入场前的权益锚点 |

[修复前 XML](receipts/quality_review_followup_before.xml) 中六个针对性用例全部以目标行为失败，无 collection／setup error。P0 用例隔离的是消费者合同：上游研究准入被显式 stub，不伪造真实合格预测模型。

## 已实施修复

### P0：训练合约只从经过认证的 Policy 读取

`prepare_quality()` 先调用 `require_prediction_admission()`，再通过收据的 Checkpoint 调用 `checkpoint_policy()`，读取其不可变事前 Policy 的 `trained_contracts`。同时核对收据文件哈希、Policy 哈希和本次 TRAIN 合约。

准入收据不需要重复保存该字段；验证器继续返回原始收据，没有添加兼容字段、别名或回退路径。回归证明不含该字段的收据能到达后续原生构建边界，而 TRAIN 合约不匹配仍被拒绝。真正研究 PASS 后的完整 CONTROLLED_FULL prepare 尚无可用合格模型验证。

### P1：所有参与预测的窗口都经过同状态对照

`_improvements()` 现在检查七个事件窗口、八个 Touch 窗口、四个 Adverse 窗口，以及 Score。每个事件／二元窗口都要求有限指标，且同时满足：

```text
NLL   < min(TRAIN prior NLL,   实际初始 Checkpoint 同状态参考 NLL)
Brier < min(TRAIN prior Brier, 实际初始 Checkpoint 同状态参考 Brier)
```

Score 的 MAE 条件、既有样本数／ECE／SL recall 条件保持原义。缺失指标或无效监督不能跳过检查；错误消息指出具体窗口。此检查沿用每个合约、事前时间 Fold、replay seed 的同状态比较，不把整体均值当作逐窗口通过。

预测准入与原生决策安全准入仍独立。网络公式、损失权重、梯度边界和融合公式均未更改。

### P1：完整交易归入入场 Fold

风险统计使用 `ENTRY_FOLD_COMPLETE_TRADE_COHORT_V1`：

1. 依据实际成交 K 线的**官方交易日**确定入场 Fold。原生成交 `time` 是 K 线起点，权益／退出是终点；从未平仓条目或已平仓 attribution 的 `entry_cursor` 读取物理成交游标，并核对原生时钟。不能把起点时间戳直接映射成终点游标。
2. 该组交易的 SL 率使用全部已观察到的完整平仓结果，允许平仓发生在 Fold 之后。回撤从首次入场前一根权益锚点，计算到该组最后一笔实际平仓，包括跨 Fold 后续；延续原有封存 tape 的逐根收盘权益口径，不改为盘中极值口径。
3. 排除前一 Fold 带入交易及之后 Fold 新入场交易。单持仓协议下，该组交易的完整跨度不得夹入其他入场组；缺失原生权益锚点／平仓端点直接拒绝。
4. `post_fold_complete_entry_ids` 明确标注跨 Fold 后已平仓交易；`right_censored_entry_ids` 标注封存终点仍未平仓交易；`open_at_fold_end_entry_ids` 记录 Fold 截止时持仓。尚未观察到 Fold 截止时，后者为 `null` 且 `fold_cutoff_observed=false`，不把未知写成空仓。
5. `RIGHT_CENSORED` 或 `INSUFFICIENT_CLOSED_SUPPORT` 不能通过决策准入。空交易组和右删失组的 SL／回撤为 `null`，不报告成零风险；门禁在比较风险增量前检查 `COMPLETE_COHORT`。

原生回归使用同一自然日的 14:59／15:00 和 21:01–21:06，官方交易日分别为 `20260105`／`20260106`，真实执行 SL→TP→SL。Fold A 的一笔跨日 SL 归入 A，SL 率 1、回撤 1,818；Fold B 的两笔 TP／SL 归入 B，SL 率 0.5、回撤 1,818，排除 A 带入的损失。这是合成行情通过真实原生 Env 的执行验收，不是 FG201 实际交易收益。

第一次实现误用起点时间查询游标，被上述原生回归发现（18 项中两项失败）；改用原生物理游标后通过。[中间失败输出](receipts/quality_review_followup_targeted.log) 与最终通过输出均保留。

## 实际验证

| 验证 | 结果 | 证据 |
| --- | ---: | --- |
| 修复前六项复现 | 6 个目标失败，0 error | [XML](receipts/quality_review_followup_before.xml) |
| 定向合同／预测回归 | 18 / 18 | [XML](receipts/quality_review_followup_targeted_v2.xml) |
| 数学、Teacher、Catalog、Gate、Env／Oracle、组合、共用 Tracker | 114 / 114，41.148 秒 | [XML](receipts/quality_review_followup_regression.xml) |
| 共享入口 CPU prepare→start100→resume200，与连续 200 比较 | 1 / 1，105.184 秒 | [XML](receipts/quality_review_followup_lifecycle.xml) |

最终不重复计数 **115 项，无失败、无跳过**；18 项定向测试是完整回归的子集。完整回归与 CPU 生命周期独立并行运行。没有为了旧测试新增兼容分支；本次仅修改两个源码模块和最近的既有 admission 测试模块。

实际 CPU Run：`artifacts/fg_s3_v2/runs/QUALITY_CPU_ACCEPTANCE_7c0ca2e2c5`。200 个 Env 推进、4 次联合更新；合成原生监督中 Score 有效 106，Event 742、Touch 848、Adverse 424，双 Q 有效数均为 0。最后一次更新证据：

| 头 | 梯度范数 | 参数变化范数 |
| --- | ---: | ---: |
| Event | 0.2614396811 | 3.796151077e-6 |
| Touch | 0.0548961982 | 3.145014034e-6 |
| Adverse | 0.1083301678 | 2.953647936e-6 |
| Score | 0.1796996295 | 2.941790171e-6 |

Actor、QualityActor、Critic、Backbone 和双 Q 的更新范数均为 0。初始 Policy 由零更新 FullCheckpoint 持有；100→resume200 与连续 200 的全部更新指标及最终参数摘要相等；非最终预算 Checkpoint 被拒绝用于准入。合成 fixture 的 Policy 阈值不是生产研究默认值。

对先前已封存、已独立原生回放验证的真实 DEV／只读诊断账户，重新认证封存文件与最终账户哈希，再以新口径计算：DEV 有一笔入场但零平仓，状态为 `RIGHT_CENSORED`；干预账户零入场，状态为 `INSUFFICIENT_CLOSED_SUPPORT`。两者 SL／回撤均为 `null`，研究状态继续 PENDING。此处仅重算既有账户风险，没有新推理或训练。[实际账户收据](receipts/quality_review_followup_delivery/real_accounts/receipt.json)。首个诊断命令未将 ISO 官方日转为门禁内部 `YYYYMMDD`，断言失败；修正诊断调用后通过，原始失败日志保留。

## 尚未验证与下一步

- [x] P0 的原始接口错误、Touch／Adverse 漏门槛、跨 Fold 风险口径及权益锚点已通过回归；本轮源码已自审。
- [ ] 事前冻结生产研究 Policy、独立 replay seeds 和尚未观察的时间窗口；测试 fixture 不能替代该步骤。
- [ ] 真实预测准入 PASS，包含全部 Touch／Adverse 窗口对 TRAIN prior 和初始参考的改善。
- [ ] 同模型独立原生决策安全 PASS，以及有足够真实数值监督的双 Q 准入。
- [ ] 以真实合格准入模型完成 CONTROLLED_FULL 的准备与运行验收；当前测试仅证明消费者合同。
- [ ] 18 合约实际吞吐、首次 GPU、10K／1M 研究训练与 Full 效果仍待审批及执行。现有 [对照与启动方案](FG_S3_QUALITY_FIRST_RESEARCH_LAUNCH_PLAN.md) 继续适用，并增加本次逐窗口预测、完整交易组风险口径。

本次源码变化使旧 prepared 与当前执行源码不一致；旧快照、Run、Checkpoint 和历史收据保留。后续研究须在冻结新提交上创建独立 prepared／Run，不改绑旧工件。已完成 Tutor-CD、Tutor-CD Full、B2 正式 1M 工件均未修改。
