# WP04 C：可观测性核对

数据与模型：使用原 Env、原 E 特征计算器、原 Quality TP Reward 和原六策略 Corpus/三方 Join。新增受控源保留 320 步完整前史，发布后段 100 行/账户；收盘价按整数 tick 变化。合成 fixture 默认 LONG，用于工程梯度证明；真实 FG201 SHORT 数据不接受参数更新。没有改写 Feature/valid，没有将 teacher 后验值输入 F1。

原 WM 可选只读诊断输出 reward prediction、post/prior availability、continue target 和 Encoder tokens；默认关闭，训练路径不增加诊断前向。诊断使用 committed native reward_events 的准确 reason，未使用 L1 反事实奖励替换 Replay reward。仅有 1m 序列与 position 两种 Decoder 目标。

`C_multiscale_learning*.json` 保存每键参数哈希、原 Loss/权重、梯度、TP/SL/zero/nonzero 条件分组及 ADD recall/FPR。无支持标 NO_SUPPORT，诊断始终没有 research/train 准入。Reader 覆盖与 optimizer 重复曝光分开计算。额外 32 步预热是有限历史近似。

已观察失败与处理：

- `C_shared_model`: 18 PASS、1 FAIL。EM04 要求单次硬分类采样必变；在固定 WP03 Gym 7366886 / Dreamer 446d1bc 原样复现相同失败 (`C_shared_baseline`)。生产代码不因此改变。测试先验证 posterior logits，再验证固定 8 步真实 Actor 概率响应；`C_shared_recheck` 1 PASS。
- `C_burnin`、`C_final`: 不同批量形状的 Encoder token 最大绝对差 6.7055225e-8，原逐字节断言失败。此对照的 float32 GEMM 容差明确为绝对 1e-7；仍检验因果输入选择、K 预热作用和状态有限性。FullCheckpoint 同形状 CPU 恢复继续要求 bitwise exact。

完整 reset 前向与分段前向使用原 RSSM 各自 PRNG 分区，报告实际状态差异，不将单帧/有限 K 冒充全历史或声称两种随机扫描 bitwise 相等。

精确命令、退出码、收集用例与原失败输出见 `docs/receipts/wp04/C_*.{json,xml,log}`。后续 D/E 的未完成状态不能由 C 的 PASS 代替。

共享 E/MACD 回归的其余 18 项在原命令通过；修正 EM04 不是降低准入条件：离散 posterior 的 logits 必须实际变化，固定多步 Actor 概率必须实际变化，未增加重试随机种子或运行兼容别名。
