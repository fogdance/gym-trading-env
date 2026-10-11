# FG Single3 训练配置

每个实验一份 YAML，共用 `tools/fg_single3_experiment.py` 启动入口。公共网络定义复用 DreamerV3 presets，运行时保存完整的解析后配置和哈希。

本次使用 `C1_M1_10K_then_online_1M.yaml`：

- 离线：从随机初始化开始，10,000 次梯度更新，使用已验收 C1 Natural + P4 Replay；Actor/Critic Gate 全程关闭。
- 在线：离线完成后恢复整个 Agent 与优化器，开始 1M 环境步；使用相同 TRAIN 行情、Reward 和网络。在线 Replay、账户及 Gate 观察窗口重新开始。
- 在线 Gate 使用原有运行阈值、0.9915 / 0.004 开启余量以及连续 20 次观察；未达标期间 Actor/Critic 损失保持关闭。
- 离线 `step` 表示梯度更新，在线 `step` 表示环境步；模型累计更新计数保留。

在项目根目录、dreamerv3 环境中运行：

```bash
python tools/fg_single3_experiment.py start --config configs/experiments/fg_single3/C1_M1_10K_then_online_1M.yaml
```

同一实验中断后将 `start` 改成 `resume`。新实验复制 YAML 并修改结果目录，用同一个入口选择配置；已生成的配置快照和实验收据不修改。

`prepare` 只执行 CPU 准入。训练由用户执行 `start` 后启动。当前两阶段不自动评估冻结的离线 Actor；独立 DEV 评估需使用在线训练后的模型，封存 TEST 不用于训练。
