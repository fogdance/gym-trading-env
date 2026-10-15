# 最新1M Checkpoint 的 Actor-only DEV 评估

- 当前训练已完成：`E_TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261015`，1000000市场分钟/40000模型更新；latest为`step_000001000000`，manifest SHA256为`17c28209a087c7c1b7f77ea24fa3fccbc2d2227f703b8a91eb502e539f034041`。
- 用户允许在较新的`fg-s3-e-review-fix-20261015`工作树完成评估接线。原训练Gym `871ca09`、Dreamer `730939d`及Checkpoint、prepared、旧收据保持不变。
- 冻结评估循环没有向DEV Env传E profile。新的只读启动器直接复用原冻结模型、ActorPolicy、评估循环、Source/Composition/完整Checkpoint校验，只对DEV Env构造补上现有原生E观察接口，并对实际解析的同份CSV字节核对SHA256。
- 使用FG409 DEV、5000真实市场分钟、eval argmax、beta0、无Teacher/Oracle指导。TEST价格与Oracle数据打开操作在评估进程内拒绝。模型/Gate/Reward/Oracle规则不修改。
- 独立run_view只链接选定Checkpoint与原身份，评估输出位于research的新目录；不写正式Run或W&B。启动时直接请求GPU提权，不重复GPU存在性探测。
- 只执行这次真实评估，完成后记录动作、收益、TP5/SL30、截尾仓位、净PnL、回撤、E输入身份和模型/数据来源到本docs目录。
