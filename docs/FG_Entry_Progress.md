# FG Entry 开发进度

当前约 85%。

- [x] 两个仓库独立分支，保留原工作树修改。
- [x] 交易环境、逐笔止盈、固定 Reward、亏损持仓恢复。
- [x] FG 主力数据导出至 2025-12-31，按用户要求删除 FG805。
- [x] 人工配置统一为 configs/fg_entry_v1/fg.yaml。
- [x] 主力资格结束退出协议、训练数据冻结和分集防泄漏。
- [x] Dreamer 两动作接入、WAIT 回退与账户/Driver 检查点恢复。
- [x] 68 项环境/兼容回归，13 项 Dreamer 相关测试；合成数据实际训练 2 次，49 个有限指标。
- [x] FG501 的 dataset_v3 测试集三种基准运行完成，账本总额均为 300000 元。
- [x] 数据及配置阶段 fe7607b；Dreamer 接入阶段 5a84068。
- [ ] 正式训练与样本外对照：用户明确禁止当前启动，等待后续授权。
- [ ] 最终择时能力验收报告。

阶段提交：0120424（业务协议文档），f65f7c2（交易环境）。后续提交及运行证据继续补充。

## 正式训练禁令与启动审计

用户于当前会话明确要求“不能开始正式训练”。此前启动尝试在 JAX 后端初始化时 AssertionError 退出（退出码 1），尚未进入训练循环。进程查询没有该训练进程。保留日志 `/tmp/fg-entry-v1/train-fg-seed0.log` 和启动配置，不删除审计材料；不得重试正式训练。合成数据两次更新仅为既有接入冒烟测试。

## GPU 小步 smoke 已验收

用户仅授权少量 GPU smoke，禁止 CPU 与正式训练。重启后恢复持久工作树和数据，主程序完成 20＋10 步、7 次更新；损失有限，账户与账本连续。修复 GPU carry 的显式保存/恢复传输和短任务最终金融检查点落盘。GPU 相关回归 14 项通过；详见 Dreamer 工作树 docs/FG_Entry_GPU_Smoke_Acceptance.md。正式批量显存和长期训练尚未验收，不能据此声称择时能力。


## 2026-10-10 原目录迁移

两个原仓库已直接切到 `feat/fg-entry-v1.1`，旧独立工作树移除。当前数据集使用 `artifacts/fg_entry_v1/dataset_v1`。目录、备份及验证详见 [FG_Entry_Migration.md](FG_Entry_Migration.md)。迁移验证完成，未启动训练。


## 按主力合约边界切分

当前数据集入口改为 `artifacts/fg_entry_v1/dataset_v2_main_split`；训练截止 2022-12-09，验证截止 2024-08-19，测试截止 2025-12-31。FG301 完整进入训练，FG409 完整进入验证；构建器拒绝合约中途切分。旧冻结数据集保留为历史记录，未启动训练。


## 当前 345 分钟数据入口

当前数据集使用 `artifacts/fg_entry_v1/dataset_v3_345_audited`，基于 `artifacts/fg_entry_v1/main_export_345_audited`。按用户要求保留早期交易日，仅在导出阶段截掉 23:01～23:30，23:00 保留。此前 dataset_v2_main_split 等产物仅作历史审计；统一配置已更新，其旧哈希不能用于当前训练准入。切分日期保持不变，未启动训练。详细修复及日志见 [FG_Entry_Minute_Clock_Fix.md](FG_Entry_Minute_Clock_Fix.md)。


## 2026-10-10 当前分组：完整合约均衡划分

用户停止旧日期切分训练，确认跨年份、交割月份及方向均衡分组，并补充 FG809=SHORT。唯一配置为 `configs/fg_entry_v1/fg.yaml`，当前数据入口为 `artifacts/fg_entry_v1/dataset_v4_contract_groups`：14 个完整训练、4 个完整验证、5 个完整测试，另加 FG605 删失测试。训练每轮无放回打散合约，chunk 保留账户与当前合约，检查点保存并恢复抽样器状态。三组均覆盖 LONG/SHORT 和 01/05/09。

环境 38 项及 Dreamer/Driver 7 项回归通过；全部真实输入哈希与训练准入通过。名单、数量、复现与限制见 [FG_Entry_Dataset_Split.md](FG_Entry_Dataset_Split.md)。评审工具实施方案已同步新分组，工具本身仍待实现。生成数据不提交；旧运行与检查点保留。本次未启动训练，新实验不能恢复旧 logdir/replay/模型权重。
