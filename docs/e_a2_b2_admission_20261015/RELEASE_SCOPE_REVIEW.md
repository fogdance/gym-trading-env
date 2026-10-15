# 准入变更范围与冻结边界

此文为验收期间的变更审查范围；GPU 全部通过前不修改生产能力状态。

- 验收双仓工作树保持现有 HEAD 不变；Run、prepared、完整 Checkpoint 与来源绑定持续引用这套冻结代码。报告后续发布至 integration 工作树，避免文档提交改变冻结 Run 的 Git 身份。
- 用户后续允许 GPU 并行。两模式各分配 12 个逻辑 CPU，原 size50m、非预分配显存，直接提权执行原 `admit-e-start/resume`。通用资源审计包装仅启动既有命令并每 5 秒读显存/RSS；不提供另一套训练流程。
- 仅在 E+A2、E+B2 各自当前 CPU 与 GPU 1K、独立冷恢复、三分支梯度和完整来源证据全部 PASS 后，将这两种能力状态设为 ready。E+Tutor 保留已有 ready；Tutor-CD+B2 仍 NOT_IMPLEMENTED。
- 正式 E 来源绑定增加 A2、B2 的明确 scope；Tutor 的既有 scope 字符串保持一致。继续拒绝合成数据用于 formal，继续验证原始完整 TRAIN pool、schema、文件 SHA256、代码、配置、seed 和 direct_init。
- 不改变旧 B2 warm-start 认证，不改 Encoder、RSSM、Loss、Reward、Oracle、TP5/SL30、Mask、Gate、Replay 或 FullCheckpoint 算法。
- 解锁后的正式配置仅做 CPU prepare / 来源审计。新配置使用独立 prepared / Run 名称和新 collection，完整 size50m，B2 采用既有 Full recipe。不会启动正式 1M，也不覆盖已完成的 E+Tutor Run。
- 必要回归限定于 Composition、E SourceAdmission 及受状态变化直接影响的断言；不重新执行全部模型测试或历史训练。
