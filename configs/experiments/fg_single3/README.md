# FG Single3 训练配置

每个实验一份 YAML，共用 `tools/fg_single3_experiment.py` 启动入口。公共网络定义复用 DreamerV3 presets，运行时保存完整的解析后配置和哈希。

本次使用 `C1_M1_10K_then_online_1M.yaml`：

- 离线：从随机初始化开始，10,000 次梯度更新，使用已验收 C1 Natural + P4 Replay；Actor/Critic Gate 全程关闭。
- 在线：离线完成后恢复整个 Agent 与优化器，开始 1M 环境步；使用相同 TRAIN 行情、Reward 和网络。在线 Replay、账户及 Gate 观察窗口重新开始。
- 在线 Gate 使用原有运行阈值、0.9915 / 0.004 开启余量以及连续 20 次观察；未达标期间 Actor/Critic 损失保持关闭。
- 离线 `step` 表示梯度更新，在线 `step` 表示环境步；模型累计更新计数保留。

当前 10,000 次离线更新已完成；配置已指向诊断接口修复后的在线目录，原失败目录保留。恢复准入通过后，`resume` 跳过已完成的离线阶段，保留完整模型和优化器。

2026-10-12 参数同步修复已提交至 DreamerV3 `0b3a641`。源码哈希已改变，旧准备收据与 Checkpoint 身份不能直接用于新源码恢复；需先生成并验证新的继续训练收据，保留全部旧产物。下面是统一入口形式，目前不能作为新源码已经通过恢复准入的命令。现有进程不会自动加载源码修复。见 [参数同步修复记录](../../../docs/FG_Single3_Policy_Parameter_Sync_Fix_20261012.md)。

```bash
python tools/fg_single3_experiment.py resume --config configs/experiments/fg_single3/C1_M1_10K_then_online_1M.yaml
```

新实验复制 YAML 并修改结果目录，完成 `prepare` 后使用 `start`；同一实验中断后使用 `resume`。各实验共用入口；已生成的配置快照和实验收据不修改。

`prepare` 只执行 CPU 准入。训练由用户执行 `start` 或 `resume` 后启动。当前两阶段不自动评估冻结的离线 Actor；独立 DEV 评估需使用在线训练后的模型，封存 TEST 不用于训练。

本次诊断错误及真实 GPU 验证见 [启动修复记录](../../../docs/FG_Single3_Online_Startup_Diagnostics_Fix_20261012.md)。

prior ADD 的只读 CPU 审计共用 `tools/fg_single3_prior_audit.py --config <实验YAML> --output <新目录>`；通过 `--offline-checkpoint` 和 `--online-checkpoint` 可固定已有阶段的 Checkpoint。工具复用原生 Replay 标注与模型的因果上下文、availability head，保存逐位置概率、曲线及输入哈希，不启动训练。2026-10-12 的实际结果见 [prior ADD 概率审计](../../../docs/FG_Single3_Prior_ADD_Probability_Audit_20261012.md)。
