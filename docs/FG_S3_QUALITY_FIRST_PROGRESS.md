# Quality-first 实施进度与 TODO

本页状态依据真实源码、测试 XML 和运行收据更新。完整实现与边界见 [实施记录](FG_S3_QUALITY_FIRST_IMPLEMENTATION.md)，预算及对照见 [启动方案](FG_S3_QUALITY_FIRST_RESEARCH_LAUNCH_PLAN.md)。GPU 未启动；预测准入保持 PENDING。

## 2026-10-16 后续复审

[三项修复与验收](FG_S3_QUALITY_FIRST_FOLLOWUP_FIXES.md)：Full 准备从已认证 Policy 读取 TRAIN 合约；全部 Touch／Adverse 窗口新增同状态 NLL／Brier 对照；SL／回撤统一为完整交易的入场 Fold 归属，区分跨 Fold 完整结果、右删失和未观察到的截止。

- [x] 修复前六项真实目标失败；修复后 114 项回归与独立 CPU 生命周期 1 项通过（定向 18 项不重复计数）。
- [x] 新 CPU Run 200 Env／4 次更新、四预测头实际梯度与参数变化、完整恢复一致性；旧账户按新口径仍不满足风险支持。
- [ ] 真实预测／Q／决策准入和真实合格来源的 CONTROLLED_FULL 仍未通过，不能宣布全链路 GO。

## 2026-10-16 Review 修复

逐项根因、源码修改、证据字段及当前收据见 [审查修复记录](FG_S3_QUALITY_FIRST_REVIEW_FIXES.md)。下文 150 项为 10 月 15 日历史基线。

- [x] F1：事前冻结 policy，完整 7H、SL/低资金/官方日/校准/多合约/时间 fold/replay seed，固定初始参考及最终预算保存点。
- [x] F2：Q 原生完整快照/有理数/转移见证独立复验，绑定实际模型更新与评价权重，训练/留出双 Q 错误门槛。
- [x] F3：128 随机＋原风险边界＋质量风险分层；选中严格路径逐笔原 Env 回放。
- [x] F4：独立决策门禁；相对 WAIT 方向证明、严格 WAIT 平局误 ADD、闭环 SL/回撤；只读诊断及 Q 权重私有副本。
- [x] F6：删除旧 MAE 扣分函数；B 公式统一，边界与固定随机一致性测试。
- [x] F7：完整账户状态缓存、实测 RSS 后分配构建 worker、候选/回放/日志资源记录。
- [x] F8：保留 L3 原 30 万元证书边界，低资金 L1/L2 仍可有效。
- [ ] F5 真实数值 Q 监督仍不足；精确域默认 12 根不变，不能称已解决覆盖。
- [ ] 生产准入 policy 的研究阈值、未观察 fold 和完整多 seed 计划尚未冻结；不能使用测试 fixture 门槛。
- [ ] 预测、Q、决策三类真实研究 PASS、18 合约吞吐、GPU/10K/1M 仍未验证。

最终本轮源码 SHA、151 项不重复测试及实际运行证据见 [收据索引](receipts/quality_review_delivery/README.md)。

## 已完成

- [x] D0 双仓源码和真实调用链审计，独立开发 worktree；保留旧实验。
- [x] 新 ZIP 原始 Patch 在指定旧基线与 integration 基线验收：各 41 项通过。
- [x] FG201 原生 0／1／2 次 SL 后完整账户反事实重放及同 cursor WAIT 对照，五组均与独立原 Env 一致。
- [x] 四层 L0–L3 Catalog：合法性、单笔后果、B(t,d) 质量分、独立严格动作认证；30 万元边界及旧数值 Q 缺失不制造标签。
- [x] 多时间范围、跨官方日、有效市场分钟、拒单／OTHER／右删失及逐任务有效位。
- [x] HEADS_ONLY 的 Outcome／touch／adverse／Score 训练与梯度隔离；实际 CPU 参数更新；原 Actor／Q／Backbone 冻结。
- [x] 保留 Full 网络、各 Loss、预测到 logit 的接线及合成监督梯度验证，受预测准入门槛限制。
- [x] 共享 prepare／start／resume／evaluate、SourceAdmission、FullCheckpoint 与共用 Tracker 接线。
- [x] CPU 100→200 步 resume 与连续 200 步参数、更新指标逐项相等。
- [x] 实际 1M Tutor Full 来源只读 CPU 准入；无 GPU 更新。
- [x] 最终数学、教师／预测边界、原 Env／Oracle 等 107 项通过，见 `receipts/quality_first_delivery_regression.xml`。

## 当前收尾 TODO

- [x] 在最新源码上完成 `QUALITY_FIRST_HEADS_CPU_FINAL_v5` 的实际 100 步神经运行、250 步 Catalog 及强来源校验。
- [x] DEV 与互斥 TRAIN 合约 FG401 的 Actor-only 评价及封存后补评。
- [x] 使用初始 Checkpoint、独立模型 carry 和相同实际动作历史完成参考预测比较，记录真实样本数和预测指标。
- [x] 最终专项回归、实际 Student 后端和 FullCheckpoint 恢复测试：116 项通过；旧测试夹具更新后 34 项通过，总计 150 项，保留修复前失败证据。
- [x] 最终源码自审、两仓源码提交与实际 SHA；210 个运行源码文件与提交逐一核对，见 [源码提交绑定](receipts/quality_first_delivery/implementation_git_binding.json)。
- [x] 更新启动方案与对照矩阵，首次 GPU／10K／1M 的授权审阅材料已在 docs；未启动训练。

现有合同回归六组清单、提交级根因和修复后 34／34 结果见 [专项记录](FG_S3_QUALITY_REGRESSION_FIX.md)。完整收据与 D0–D7 状态见 [交付索引](receipts/quality_first_delivery/README.md)。

## 未通过研究认证

预测能力、校准、Actor 改进、真实数值 Q 覆盖、多 seed／多时间 fold、全量 Catalog 吞吐及 GPU／10K／1M。CPU 接线与恢复验收不代表上述项目已通过。
