# 已完成 Tutor-CD 与当前 Full E2 配置对比

核对时间：2026-10-15T05:24:52.758030+08:00。来源为实际 prepared 快照、Run 收据及更新日志；没有修改训练配置。

| 项目 | 已完成 Tutor-CD V0 | 当前 Tutor-CD Full E2 |
| --- | --- | --- |
| 实验 | `TUTOR_CD_V0_MARKET_MINUTE_FORMAL_1M_v1` | `TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014` |
| 预算/进度 | `FORMAL_COMPLETE`，1,000,000 环境步 / 40,000 更新 | 预算 1,000,000 步；核对时 57,400 步 / 2,296 更新 |
| Natural / Balanced BC | 开；lambda=1，alpha=0.8 | 开；参数相同 |
| Pair Ranking | 开；lambda=0.1，tau=1 | 开；参数相同 |
| E2 难样本加权 | **关**，kappa=0；wmax=2 未激活 | **开**，kappa=1，wmax=2 |
| 分项梯度诊断 | 未配置 | 每 100 次更新采样 |
| beta / 衰减 | 固定 1；无 schedule | 固定 1；无 schedule |
| Critic / RL | 冻结 | 冻结 |
| 数据及采样 | 同一市场分钟 E/O/Catalog，18 TRAIN 合约，片段 1000 分钟 | 相同 |
| 网络与初始化 | 原 1m 网络；Online180K 参数 warm-start | 冻结网络配置逐项相等；相同源 checkpoint，独立新 Run |
| 收集/更新/保存 | 100 步 / 4 次更新 / 100K 步完整保存 | 相同 |
| batch / Replay / seed | 1 / 32768 / 202 | 相同 |
| 组合配置 | 旧版 mode=tutor | F0 schema；legacy_1m + tutor_cd；E 多尺度及 B2 均未开启 |

Full E2 的 Tutor 教学项均已开启，但并非把框架支持的其他算法全部打开。beta 衰减、C2 RL、E 多尺度编码和 B2 Outcome/Adapter 不属于这次配置。E2 表示样本难度加权，并不表示 E 多尺度编码器。

实际更新日志的计数：Actor 教学有效 2,176 次；Pair 与 E2 加权分别 1,603、1,603 次；梯度采样 22 次；Critic 参数变化 0 次。Pair/E2 需要 batch 内存在有效正负样本，并非每次更新都触发。

旧版完成收据记录有效 Tutor 更新 39,257 次、Pair 更新 28,401 次、RL Actor/Critic 更新均为零。当前 FG701 在线模拟使用旧版的 **800K checkpoint**；它属于 V0、kappa=0，不是正在训练的 Full E2。当前 Full E2 使用原 Online180K 参数新建训练，并非从旧版 800K 继续训练。

证据位置：

- 已完成配置：[TUTOR_CD_V0_MARKET_MINUTE_FORMAL_1M.yaml](../configs/experiments/fg_s3_v2/TUTOR_CD_V0_MARKET_MINUTE_FORMAL_1M.yaml)；归档真实快照 `/data/logdir/fg_single3/archives/tutor_cd_800k/prepared/settings.json`。
- 当前配置：[TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M.yaml](../configs/experiments/fg_s3_v2/TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M.yaml)；冻结真实快照 `/home/v/Documents/work/research/fg-s3-tutor-full-train/gym-trading-env/artifacts/fg_s3_v2/prepared/TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014/settings.json`。
- 已完成运行：`/data/logdir/fg_single3/s3_v2/TUTOR_CD_V0_MARKET_MINUTE_FORMAL_1M_v1/training_receipt.json`。
- 当前运行：`/data/logdir/fg_single3/s3_v2/TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014/updates.jsonl` 与 `cycles.jsonl`；上述计数为核对时的运行快照。
- 样本难度权重及条件：兄弟 Dreamer 的 `dreamerv3/fg_s3_tutor_loss.py:tutor_objective`；Critic 冻结：`dreamerv3/fg_s3_v2_agent.py:train`。

这里比较配置和执行状态，没有运行新的 Actor-only 对照评价；不能据此断言 Full E2 的交易效果更好。后续按相同 checkpoint 步数、相同 DEV 数据和账户规则比较 Actor-only 指标。
