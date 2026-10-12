# FG Single3 在线启动诊断接口修复（2026-10-12）

## 观察与根因

离线阶段已实际完成 10,000 次更新，世界模型与 Reward Head 有效更新各 9,999 次，Actor/Critic 有效更新和 Gate 开启更新均为 0。在线阶段保存第 0 步完整 Checkpoint 后，首次策略调用退出。

在线子进程日志显示 `jax.debug.callback failed to find a local CPU device`。仅补 CPU 设备不能解决根因：在 CPU 上复现 WAIT-only 掩码时，真正触发的是 `Action must be valid under its mask`。

新增的 `p_add` 诊断有两处接口错误：

1. `dreamerv3/agent.py` 查询 ADD 概率时调用 `MaskedCategorical.prob()`；此前该方法间接调用严格的 `logp()`。ADD 不可用时，查询概率误触发训练动作合法性断言。
2. 策略输出中的裸 `p_add` 随 Driver 进入 Replay。真实 GPU 驱动 1,152 步后，首个训练批次多出 `p_add`，触发 `embodied/jax/agent.py` 的输入字段断言。

Git 证据：严格掩码和损失检查来自 2026-06-12 的 `6facac8` / `4019200`；调用上述检查的 `p_add` 诊断来自 2026-10-11 的 `926bed0`。此次故障来自新增诊断接线，并非六月的离线训练协议发生了要求变化。

## 最小修复

- `MaskedCategorical.prob()` 直接读取已掩码分布的概率；禁止动作返回 0。`logp()` 的严格动作检查保留。
- 普通在线训练在策略输出边界把 `p_add` 改为 `log/p_add`。Driver 的已有日志通道保留诊断统计；Replay 的已有 `log/` 过滤排除该字段。
- 准备收据增加 `embodied/jax/outs.py` 的源码哈希。
- 使用新准备目录和在线目录保留旧失败日志及第 0 步 Checkpoint，复用原离线最终完整 Agent/优化器。

未修改 JAX 初始化和设备选择、训练损失、Reward、Gate 阈值、Replay 比例、网络、horizon、撮合或账户系统。最初尝试的启动环境变量调整已回退。

## 实测验证

- 新增一个概率查询回归用例：修复前失败；修复后通过。扩充现有在线接续用例，检查诊断未进入训练批次。
- 运行 `dreamerv3/tests/test_action_mask.py` 和 `embodied/tests/test_fg_add_gate_training.py`：**8 passed in 1.86s**。
- 真实 GPU、原完整网络、原在线第 0 步 Checkpoint、原 TRAIN 环境、现有 Driver/Replay/Stream：连续 **1,152** 个环境步；其中 **1,151** 步仅 WAIT 可用，此时 ADD 概率均为 0，动作均为 WAIT。
- 首批训练数据字段检查通过；独立冒烟模型成功完成 **1** 次梯度更新，`opt/loss=19.905277252197266`，Actor/Critic 参数变化均为 False，Gate 保持关闭。模型计数由 10,000 到 10,001；该更新仅属于冒烟副本，不计入正式训练预算。
- GPU 验证耗时 **102.59 秒**；原 Checkpoint 完整字节校验通过且未修改。
- 修复后准备准入及官方在线准入通过；离线身份仍为 `847e50f6270678c9192da3ee2bea79e0ee77005369133553e049728416d17e47`。新旧在线学习配置完全相同。
- 封存 TEST 价格未访问；正式在线训练未代用户启动。

证据：

- [原失败日志](../artifacts/fg_single3_v1/experiments/C1_M1_OFFLINE_10K_from_zero_20261011_01/online_stdout_40c54399451f469f8d2c5eb524f81d4f.log)
- [GPU 验证日志](../artifacts/fg_single3_v1/experiments/C1_M1_ONLINE_startup_fix_smoke_20261012_03.log)
- [GPU 验证收据](../artifacts/fg_single3_v1/experiments/C1_M1_ONLINE_startup_fix_smoke_20261012_03/manifest.json)
- [修复后准备收据](../artifacts/fg_single3_v1/experiments/C1_M1_OFFLINE_10K_online_callback_fix_20261012_02/manifest.json)
- [恢复准入收据](../artifacts/fg_single3_v1/experiments/C1_M1_OFFLINE_10K_online_callback_fix_20261012_02/startup_fix_resume_receipt.json)

## 恢复命令与限制

在 dreamerv3 环境运行：

```bash
python /home/v/Documents/work/gym-trading-env/tools/fg_single3_experiment.py resume \
  --config /home/v/Documents/work/gym-trading-env/configs/experiments/fg_single3/C1_M1_10K_then_online_1M.yaml
```

该命令检查已完成的离线最终 Checkpoint，跳过预训练，以其完整模型和优化器进入修复后的新在线目录：`/data/logdir/fg_single3/online_after_10k_callback_fix/C1_M1_ONLINE_1M`。旧在线未完成任何环境步，所以这里重新开始在线账户与 Replay，不重置预训练模型。

验证只证明这次启动及首批更新接口正常；尚未完成正式在线 1M 环境步训练，不证明 Actor 已有效训练、已学会择时或在 DEV 中优于简单策略。
