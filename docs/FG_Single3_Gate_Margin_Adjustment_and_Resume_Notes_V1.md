# FG Single3 Gate 余量调整与续训说明 V1

状态：已完成代码审查和 CPU 规则验证；新续训 Checkpoint 与收据已准备；本次操作未启动 GPU 训练。

## 实测依据

使用 C1_M1_1M 的真实 `train_updates.jsonl`，仅选择最后一次连续 1～9,900 更新的 attempt。
开启前与开启后的统计分别分析，避免将 Actor/Critic 开启后的训练轨迹当成纯预热证据。

6,000～9,000 更新的 31 个统计窗口：

| 指标 | 实测 |
| --- | ---: |
| prior 准确率均值 | 0.98998976 |
| prior 准确率窗口标准差 | 0.00048788 |
| prior 准确率最高值 | 0.99088379 |
| prior 误放行率均值 | 0.00374642 |
| prior 误放行率窗口标准差 | 0.00039775 |

原规则在更新 9,000 开启 Gate，更新 9,500～9,900 连续五个窗口失败后停训。
0.994 并非已达到的模型水平；单纯延长五个贴近底线的通过窗口也没有提供安全余量。

## 本次固定规则

只对已准入的 C1_M1_1M 启用，不新增 YAML/CLI 配置项。

| 条件 | 开启前 | 开启后 |
| --- | --- | --- |
| prior 准确率 | ≥0.9915 | ≥0.99 |
| prior 误放行率 | ≤0.004 | ≤0.005 |
| 其他指标及 ADD 正类支持 | 原要求 | 原要求 |
| 连续通过窗口 | 20 个；每窗口 100 更新，共 2,000 更新 | 原规则 |
| 连续退化 | 尚未开启时重置通过计数 | 五个窗口失败仍停训 |

准确率余量 0.0015 约为上述实测标准差的三倍，误放行率保留 0.001 的余量。
这是基于现有日志选择的候选规则；窗口不是独立同分布样本，不能解释成统计置信保证，也不是已验证最优值。
现有窗口未达到新开启门槛；不能保证继续预热一定达到，更不能保证联合训练后持续通过。

## 代码及验证

- `embodied/run/action_mask_warmup.py`：开启与运行阈值分开；固定 Single3 1M 开启余量与 20 个窗口；日志记录实际开启参数。
- `embodied/run/offline_pretrain.py`：在已准入的审计模式、62,500 更新预算下选用新规则；原 pilot 和通用 Gate 保持原判断。
- `dreamerv3/fg_single3_offline.py`：实验 identity 记录实际开启规则，并加入 Gate 源码哈希。
- 在现有 Gate 测试中增加两个主路径/边界测试；扩展已有准入测试覆盖 C1 身份记录。
- 相关 CPU 测试：22 passed；包括余量不足拒绝开启、连续窗口中断重置、状态保存恢复、开启后使用原底线、五次退化停训及原 pilot 接口。
- 实验启动器相关测试：7 passed。提交沿用这些已通过的结果，不重复运行测试。
- 重放同一份实际日志：原规则复现 9,000 开启与 9,900 停训；新规则不会开启。日志重放仅验证判断逻辑，不预测新训练轨迹。
- 完整验证 8,000 更新 Checkpoint 的 2,442 个产物文件、原身份与更新计数；该 Checkpoint 的 Actor Gate 关闭、Actor 有效更新为零。

证据与哈希：[cpu_gate_verification.json](../artifacts/fg_single3_v1/experiments/C1_M1_1M_gate_margin_v2_20261011_01/cpu_gate_verification.json)。

## 续训状态和边界

1. 从真实有效的更新 8,000 完整 Checkpoint 保留模型、优化器、采样器、待消费流、训练 carry 和审计计数；不从零初始化。更新 9,001～9,900 的 Actor 权重未保存，不能从 9,900 恢复。
2. 新身份为 `1e88d9b5903bac73e80e70de5bf92bd7be8187b17bc74120d6772be350899faf`，独立输出目录为 `/data/logdir/fg_single3/gate_margin_v2/C1_M1_1M`。原收据、日志、Replay 和 Checkpoint 不覆盖；原 `resume.sh` 绑定旧身份，不能直接用于新规则。
3. 已重新开始 Gate 连续达标观察，不继承旧门槛产生的一个通过窗口；保留历史更新计数，剩余预算为 54,500 更新。
4. 已校验学习配置、数据与完整状态；迁移副本仅允许改变 `config.yaml` 与 `action_mask_warmup.pkl`，并记录新身份和对应哈希。8,000 步 Checkpoint 的其余 2,440 个载荷文件哈希完全一致，未跳过原身份检查或修改旧 Checkpoint 标签。
5. 达到新门槛的能力和后续退化原因仍需真实训练验证；本次没有启动 GPU，工程通过不代表择时有效。

## 审查发现及修正

最终报告要求保留 0、2,000、10,000、25,000、50,000、62,500 步的受保护 Checkpoint。
最初迁移只保留了续训起点 8,000 步，会导致最终报告在查找 0/2,000 步 Checkpoint 时失败。
已在新目录补齐 0/2,000 步的完整历史副本，两者均通过原身份校验和新身份下的完整包校验。
这两个历史时点的 Actor/Critic 有效更新数均为零；各有 2,440 个载荷文件哈希与原文件一致，变更范围与 8,000 步迁移相同。
旧检查点未改动，新目录的 `latest` 仍为 8,000 步；后续里程碑按原训练保存计划生成。

- [8,000 步续训迁移收据](../artifacts/fg_single3_v1/experiments/C1_M1_1M_gate_margin_v2_20261011_02/checkpoint_resume_receipt.json)
- [0/2,000 步历史 Checkpoint 收据](../artifacts/fg_single3_v1/experiments/C1_M1_1M_gate_margin_v2_20261011_02/historical_checkpoint_continuation_receipt.json)

## 续训命令

```bash
bash /home/v/Documents/work/gym-trading-env/artifacts/fg_single3_v1/experiments/C1_M1_1M_gate_margin_v2_20261011_02/resume.sh
```
