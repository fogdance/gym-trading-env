# E+Quality Master Roadmap V8：WP05 与 DATA/MACD V3 集成

日期：2026-10-18。本版记录用户授权的 `integration/s3-next` 双仓合并。
WP05 工程历史见 [V7](FG_S3_E_Quality_Master_Roadmap_20261018_V7.md)，DATA/MACD V3 历史见 [DATA V3](FG_S3_E_Quality_Master_Roadmap_20261018_V3.md)。同名的 WP04 设计基线另保留为 [WP04 baseline V3](FG_S3_E_Quality_Master_Roadmap_20261018_V3_WP04_Baseline.md)。这些历史报告的 PASS 不自动成为当前合并版本的 PASS。

| 双仓 | 合并前 integration | WP05 功能分支 |
| --- | --- | --- |
| Gym | `0e4af493bcbbfddad1dfcf5eb8b30b5b81a7a65b` | `61dd64b5428588868f374389d4f3d79e0845cdac` |
| Dreamer | `fd144113ce706e2d8f1ce8e7828ece0799c8bf7a` | `be7ce01ebbe63144e4d335b1603ca73da4fa1c79` |

最终合并 SHA、当前范围的测试命令、原始结果和限制见 [集成报告](S3_NEXT_WP05_MERGE_REVIEW.md) 及其结果收据。

| 工作包 | 当前集成状态 | 后续验收 |
| --- | --- | --- |
| WP01 | 课程 V2；MACD 数学子合同 V3；草案/正式机制保留 | 旧 V2 MACD 身份不能用于 V3 准入或同 Run 恢复 |
| WP02 | 原生 P0–P4、L1/L2、真实账户轨迹生产器；保留 DATA 范围与 worker 控制 | 本次没有启动生产、重签 Corpus 或重新执行真实 Native G3 |
| WP03 | 统一 Join/Reader、Owner/Stepid、episode-safe 采样和恢复 | 教材发布及正式训练仍需对应版本的真实来源证据 |
| WP04 / F1 | 原生六个 WM 组件的工程更新、冻结与 FullCheckpoint 路径已合入 | 当前版本的真实 TRAIN F1 训练及研究验收未运行 |
| WP05 / F2 | 原有 WM/Reader/Optimizer；仅四预测头训练；参数接续与精确冷恢复已合入 | 不能用合成 F1 冒充真实训练底座；FORMAL/RESEARCH 仍 NO_GO |
| WP06 / F3 | 本次未实施 | 独立 Quality Actor、严格 L2/Pair、冻结边界及恢复 |
| WP07 / F4 | 本次未实施 | 受控在线适配、阶段交接及单独授权 |
| WP08 | 本次未实施 | 隔离 Actor-only 考试；教师成绩不能代替学生成绩 |
| WP09 | 保留并行 DATA 线 | 生产进度本次未核验；不自动扩至全部 18 合约 |
| WP10 | 本次未实施 | 完整研究冻结、复现与最终 GO/NO_GO |

本次集成只修正当前合同下的合成测试来源和内置工程 Probe，不迁移旧 Run/Checkpoint/Corpus。新工程来源通过当前原生 Env、Teacher、Reader 和 F1 Run 生成；仍显式为 ENGINEERING_ONLY / FOUNDATION_NOT_TRAINED。

DEV 使用定向 CPU 测试和合成更新；DATA 发布与 FORMAL 训练保持各自的验证及授权要求。金融撮合、原生 Quality TP Reward、MACD 数学、真实来源验证和公开 E+Quality Guard 不因合并放宽。
