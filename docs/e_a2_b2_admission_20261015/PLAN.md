# E+A2 / E+B2 正式准入验收计划

基线：Gym `6a176392becf7464213c48d05ddbf29399603954`、Dreamer `131e7ccd4ef0648c8230e088d95e2c6da991557c`，均为已合并的 `integration/s3-next`。独立验收双仓分支 `verify/s3-e-a2-b2-20261015`；运行前提交四份新命名配置并冻结，不改旧 Run / prepared / Checkpoint。

1. 复用旧 CPU 已通过的协议和完整 E TRAIN collection，不重建 15m 数学或 Provider；因最终 Env SHA256 修复和源码身份不同，补发当前版本两模式 CPU 收据。
2. CPU 每模式真实 FG201/FG301 共 2200 分钟、6 次联合更新，在 1100 分钟保存并退出，独立冷进程核对下一批次、参数、优化器、RNG、指标完全一致，再完成后半程。B2 必须包含 OutcomeHead 和 Adapter 更新。
3. CPU 完成后，每模式原 size50m GPU 1000 分钟 / 1000 更新，在 500 保存并退出，独立冷恢复到 1000。保留三分支非零梯度、参数变化、策略同步、完整 Checkpoint、真实来源、显存及速度收据。使用真实 GPU 命令直接提权；GPU 监测只用于运行时资源审计。
4. 两模式均 PASS 后，审查最小准入状态变更；保持 E 与 teaching 正交，保留真实数据认证、direct_init、原 Gate / Reward / Oracle / TP5-SL30 / Mask 和旧 B2 warm-start 严格认证。Tutor-CD+B2 保持 NOT_IMPLEMENTED。只执行必要配置/身份回归，不启动正式 1M。
5. 报告、所有失败及原始收据写入 docs。正式 1M 仍须另外授权并使用冻结新代码和新 prepared。

当前证据：旧 A2 / B2 CPU 收据均为 PASS_CPU_REAL_TRAIN、2200 分钟 / 6 更新 / cold_resume_exact=true，但旧代码与当前不同，不能充当当前 GPU 准入收据。原模型功能不重做、不扩展测试矩阵。

资源只读观测：可用内存 12892389376 字节；项目 /dev/sdd2 可用 22206763008 字节；/data /dev/sda1 可用 104071180288 字节。既有完整 size50m Checkpoint 约 947772259 字节。CPU 两模式按可用资源并行，GPU 单卡串行，所有新增产物使用独立路径。
