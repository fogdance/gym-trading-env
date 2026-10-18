# WP04 反向源码 Review

本轮在实现后重新按任务合同审查调用链、实际文件和运行产物；由同一开发代理执行，不冒称外部人工签发。独立证据检查脚本重新读取持久 FullCheckpoint，而非只信 pytest PASS 或旧报告。

| 检查 | 源码证据与结论 |
|---|---|
| 是否另造 WM/优化器/Replay | F1 loss 最终调用 `_S3V2Model.wm_only -> Agent.world_model`；仅一次原 `self.opt`。`FoundationRun` 只协调现有 Reader/Agent/Prefetch/FullCheckpoint，无 Env collector。 |
| 是否走旧 HEADS_ONLY / 旧权重 | 显式 FOUNDATION_WM marker；完整 quality_init_loss 初始化；非初始化分支直接 F1。没有 source-checkpoint 或旧 1M load 路由。 |
| 是否 teacher 泄漏 | Adapter 正向枚举 native replay_fields 与 stepid/consec，删除所有 teacher/quality 后验；WM_NATURAL 不按后验标签选择窗口；C03 loss/全部梯度重写对照相同。 |
| 时序及账户 | byte-bound 原生 stepid + 连续 ordinal + policy/segment 同一性，原 incoming/action 对齐；K+1 真实可见前史，stop_gradient，T 个目标，每窗新 carry。 |
| 是否假学习 | 有效原生多尺度样本上 enc 的 15/15 叶均变化，context15/macd5 都有梯度与变化。恒价不足 warmup 的旧合成样本明确不作为此证明。 |
| 是否冻结漂移 | 144 个非白名单叶逐键 bitwise 不变；真实非零动量、weight decay 和 signed zero 回归。初始化及运行时未知 schema 拒绝。 |
| 数值失败 | finite candidate 回滚完整 Ninjax state，host 计数/政策发布前拒绝；非有限输入或原生字节不匹配在 Adapter 拒绝。 |
| 外部恢复 | 原 FullCheckpoint 原子 manifest/checksum/inventory；独立 course_reader saveable；save/load 验证外部字节；原 Prefetch pending/source state 与 Agent batch seeds 同时保存。 |
| 断点是否表面一致 | E_restore_final 后再从 artifacts 读取 continuous/split 两份原件，重新比较全部字节与下一 batch，exit0。不是同进程对象复用。 |
| 历史迁移是否被放宽 | source audit 仅识别默认 False 的新 WM 诊断签名，保留完整 AST。固定旧 Gate 两端仍通过；旧 WP03→F1、默认 True、数学/helper/factory 改动拒绝。 |
| 真实金融/时钟/特征 | 对 WP03 祖先做全差异检查：Gym envs/utils/rewards/e_data_v1、Dreamer RSSM/E Encoder/Gate 未修改；真实 6 账户新 G3，共 4146 行。 |
| 正式训练边界 | loaded JSON/伪造 native dataclass/无授权不能启动；public non-F1 Gate 保持关闭。DEV/DATA/FORMAL 按用户新规则，日常开发不重跑全 G3、不强加正式冻结。 |

恢复负例另补强 8 项定向验证：先用当前代码正向 cold restore，再通过 preflight 合法解析真正改变 K、seed、size preset、policy spec、原 optimizer lr、reward loss scale 和双仓代码身份；全部在模型创建/目录写入前拒绝。它们不依赖向旧收据增加未知字段来制造表面拒绝。

发现并修正：Ninjax 初始化重入测试相位、原优化器首步 lr=0 的错误断言、signed-zero 原生比较、15m/MACD 支持不足、单次 categorical 采样错误断言、配置 tuple/list 序列身份误拒绝、负例漏 index 收据、WM extraction 审计新签名冲突。旧失败与修复后正/负证据在 C/D Review 和测试总表。未改金融规则或降低 Guard。

局限：K32 是有限历史近似。reset 全量扫描与分段扫描的原随机分区不同，状态差异不是历史等价证明；同 PRNG 分区改变过去输入确有状态影响。不同 batch shape 的 Encoder 对照采用明确 float32 绝对 1e-7 容差；同形状 cold restore 始终 bitwise。

真实 F1 优化器更新、GPU、完整正式 start/resume、F1 基础权重、模型获利/泛化、18 合约、TEST、W&B 发布、外部人工 Review 均 NOT_RUN。代码与真实只读工程 GO 不代表这些验收通过。

E 最后审查将协调层 FORMAL checkpoint reserve 恢复为原正式路径的 10 GiB，合成诊断 1 GiB。仅重新执行受影响的 22 项恢复/合法配置拒绝测试（E_restore_final，exit0），没有为该开发修改重跑 G3。
