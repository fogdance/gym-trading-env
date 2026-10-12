# FG Single3 参数同步修复（2026-10-12）

本次只修复训练权重到策略推理缓存的接线；Gate 门槛、观察窗口、停训协议、Reward、Replay、horizon 和网络配置保持不变。

## 已确认的问题

在线 180,000 环境步 Checkpoint 的全部 60 个策略缓存参数与在线初始 Checkpoint 相同，但训练中的 Encoder 9 个、Dynamics 27 个和 Decoder 13 个参数均已改变；`sync_train_metrics=true` 且没有待发布参数。完整输入哈希和比较结果见 [原始审计证据](../artifacts/fg_single3_v1/diagnostics/gate_analysis_20261012_01/policy_sync_audit.json)。

`embodied/jax/agent.py` 的 `Agent.train()` 将参数发布条件写成 `enable_policy and not sync_train_metrics`，同步返回训练指标时因此完全跳过参数发布。这与 Gate 是否达标是两个独立问题；Gate 计算仍使用训练权重。

## 最小修复

DreamerV3 提交：`0b3a641`。参数发布条件恢复为 `enable_policy`，指标返回逻辑保持不变。继续使用既有 `pending_sync` 机制、非捐赠的上一更新参数快照以及 `policy()` 调用后的缓存切换，保留既有异步延迟与缓冲区所有权，不新增每步完整权重复制。

仅修改生产代码的一处条件及说明，并新增一个覆盖同步／异步两种模式的回归测试。Actor/Critic 的更新仍由原 Gate 控制，离线阶段仍冻结 Actor/Critic。

## 验证结果

修复前同一回归测试：异步模式通过；同步模式失败于 `pending_sync is not None`，实际为 `None`。

修复后执行：

```bash
cd /home/v/Documents/work/dreamerv3
JAX_PLATFORMS=cpu PYTHONPATH=. /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -q embodied/tests/test_jax_policy_sync.py dreamerv3/tests/test_action_mask_warmup.py
```

结果：**15 passed**。回归测试使用真实 `Agent.train()`、`Agent.policy()`、JAX JIT 梯度更新、参数缓冲区与完整 Agent 保存／恢复，验证连续参数发布、慢消费者、恢复后继续发布，以及同步指标仍对应当前更新。推理输出按发布快照从 `0.25` 更新至 `0.52`、`0.616`。

这是小型梯度核对共享 JAX 包装器的验证，不是完整 FG 神经网络 GPU 训练验收，也不代表 Actor 已学会择时。唯一测试告警为沙箱无法写入 DreamerV3 的 pytest 缓存；测试本身全部通过。已自审补丁并通过 `git diff --check`。

## 代码哈希与运行限制

| 文件 | SHA256 | 状态 |
| --- | --- | --- |
| `embodied/jax/agent.py` | `1d0059ede3be4d347e2c46fb9c4f0cc2bae4bfe007478dde95ca94a9de5abb15` | 已修复 |
| `embodied/tests/test_jax_policy_sync.py` | `c40d5c74ba12be7ff6384862f6239097c2db684d9d74bfb902655b4239e3f0cc` | 新增回归测试 |
| `embodied/run/action_mask_warmup.py` | `a280a4f67566d1d95d753b537d015e389df30bce0d0295da28b1e03f5ad78c2b` | 与修复前逐字节相同 |
| `dreamerv3/configs.yaml` | `4e5661456a77e25fc06af5c4bfbdcbc85e8e2043790e9092d6eec070b45778ce` | 与修复前逐字节相同 |

本次没有停止或重新启动正式训练，也没有修改运行目录、冻结收据或旧 Checkpoint。现有 Python 进程不会自动加载源码修改。源码哈希变化后，恢复准入必须通过新的合法继续训练收据；旧恢复命令尚未通过新源码的身份校验，不可直接当作已可用命令。后续恢复保留训练参数、优化器和进度，不能以修改旧哈希或绕过检查代替准入。
