# Gate 修复执行记录

实施依据：`FG_Single3_Availability_Gate_Repair_Before_Resume_Codex_V1.md`。

顺序：D0 冻结来源与日志 → D1 真实交易/Replay 时序对账 → D2 共用概率阈值与最少接线修改 → D3 CPU 只读概率、窗口和15步想象预检 → D4 新身份完整状态迁移与恢复验证 → D5 自审、证据报告和提交。

来源固定为 Online `step_000000180000`。不重训、不读取 DEV/TEST，不改变 Reward、网络、H333/Imag15、混合比例、80% Recall、FPR或20窗规则。旧实验和审计产物只读。

必须分别判断：接线修复、候选阈值全指标达标、连续窗口达标、状态迁移安全。任何缺少证据的项标记 NOT_RUN/NOT_VERIFIED，不以全量 Recall 代替 Gate 验收。

本次产物目录：`artifacts/fg_single3_v1/diagnostics/gate_repair_20261012_01/`。

完成状态：D0–D5全部完成。Dreamer `4659d32`、Gym代码 `cba9080`；0.30冻结20窗连续通过，实际双路径完整恢复131.8秒通过，正式CLI无授权拒绝通过。恢复包已发布到新目录并生成CPU证书。GPU/DEV/TEST未运行或读取，正式Gate仍关闭/streak0；最终结论和手动命令见实施与验收报告。
