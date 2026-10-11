# FG S3-A 实施与验证进度

任务：只实现 C1_P4 Oracle Actor-only Behavior Cloning，完成 D0–D4 CPU Gate。
本报告随开发更新；未通过的项目不会写为完成。

## Summary

两个独立开发仓库均已建立 `feat/fg-s3a-oracle-bc` 分支。
开发基线：gym `79aa9a9664a648fc961277986290874a6eb1c9d1`；Dreamer `446e6fc9dbc92a1be24d0d44b840e85d4f96010e`。
原生产仓库 `/home/v/Documents/work/gym-trading-env`、`/home/v/Documents/work/dreamerv3` 只读。

| Gate | 状态 | 收据 / 下一步 |
| --- | --- | --- |
| D0 | 进行中 | 分支、来源和生产保护审计 |
| D1 | 待实施 | 原生 Replay / Sidecar 标签校验及全 TRAIN 索引 |
| D2 | 待实施 | 原模型教师历史 policy 前向及特征缓存 |
| D3 | 待实施 | Actor-only 更新、冻结核查、Checkpoint / resume |
| D4 | 待实施 | CPU 端到端、CLI、自审和提交 |
| D5 | NOT_RUN | 正式 GPU 训练和独立 DEV 回测不在本次执行范围 |

## Evidence observed

- 规划阶段只读扫描：488,629 行；402,096 合法空仓标签；ADD 16,171；WAIT 385,925。
- 当前 Observation 对应 Sidecar `lots_after`；使用 `lots_before` 会漏掉 14,087 个标签。
- 5 条当时合法、已实际执行的 WAIT 下一反馈分钟无效，按原定义保留并单列审计。
- 原 Mask / Replay context 基线 CPU 测试：7 passed（不等于 S3-A 测试通过）。
- 生产仍在运行；正式最终 Checkpoint 未用于本次开发。

## Root cause / 设计依据

这是新功能。监督必须使用本行 outgoing action；Reward / incoming action 是此前执行反馈。
特征必须用原 policy carry 和教师过去动作生成，不能把打乱后的窗口作为独立 RSSM 序列。

## Changes made

新增隔离命名空间，不改原 Agent / JAX wrapper / 配置 / Env / Oracle / Replay 模块。

## Verification performed

各 Gate 的命令、结果、哈希和 commit 将在这里补齐。

## Risks / not verified

未启动正式 GPU BC，未加载生产最终模型进行 BC，未运行正式 DEV，未访问 sealed Test。
Oracle 标签带后验信息；教师历史 hidden state 不等同于部署策略状态；CPU 工程通过不证明择时有效。

## Next recommended step

完成 D0–D4 的实现和 CPU 证据，再确认源 Checkpoint 就绪后的实际兼容性与后续 D5。
