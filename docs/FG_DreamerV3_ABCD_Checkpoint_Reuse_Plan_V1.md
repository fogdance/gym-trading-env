# FG ABC 固定步数与完整 Checkpoint 复用实施计划 V1

日期：2026-10-10。状态：代码与非模型验收已完成；真实 GPU smoke 待授权，所有模型训练保持停止。代码直接修改两个仓库现有 feat/fg-abcd-comparison-v1 分支。

## 已确认规则

- 顺序训练，完整主力合约生命周期，四个 fold 与三个正式 seed 不变；盲测封存。
- Gate/报告每 5000 Driver 环境交互步检查；常规完整恢复点每 10000 步保存。reset 观察计入交互，实际推进 K 线另报。
- Gate 的指标、阈值、最低 25000 训练样本和连续 5 次通过规则不变。样本不足保留累计值，记录跳过，不计一次通过或失败。
- 常规恢复点保留最近两个；70K、100K、700K、1M 分支/最终节点永久保存。只清理本协议新生成、发布完成的旧常规点，旧实验不动。
- 不改 Obs、Action、Reward、撮合、账户及风险规则。不训练 JM，不允许 CPU 模型训练。

## 执行图

每个 fold/seed 使用同一个全历史训练目录：A 0→700K→1M，C 从固定 700K 节点分支、近期再训练 300K；B 独立近期 0→1M。正式唯一计算量每组 230 万步，12 组 2760 万步，较重复训练节省 840 万步。

FG409 seed0 顺序先导：A 保存 70K，继续到 100K；B 到 100K；C 从 70K 近期训练 30K。新先导 A/B 可在一致性校验后续训至正式目标，不能额外再训练 1M。先导 C 不作为正式 C 近期阶段。旧轮转 checkpoint 不复用。

新版本 v3_shared_prefix，训练产物在 /data/logdir/fg_abcd_v1/v3_shared_prefix；生成清单在 artifacts/fg_abcd_v1/v3_shared_prefix。生成数据、日志、checkpoint 不提交。

## 完整恢复协议

保存 Agent/优化器/随机计数、Replay 内存中的全部有效 chunk（包括部分 chunk 和后续更新）、索引/FIFO/在线队列/Uniform RNG、Driver/账户/待执行动作、训练与报告 carry/stream、预取批次、Ratio、Gate 状态/待评估统计及报告下一触发位置。

统一安全点暂停预取并等待在途批次结束，保存独立不可变文件；恢复所有内容后启动预取。完整快照不调用旧 Replay.save()，不能通过关闭部分 chunk 改变学习状态。恢复不扫描运行目录中未属于节点的后续 Replay 文件。

C 跨阶段只加载完整 Agent（含优化器与随机计数）及 Gate 学习状态，近期 Replay、账户、Ratio、流和待评估统计新建。C 阶段中断恢复自身全状态，不重复迁移。源账户完整保留在源节点。

校验 fold/seed、有序数据及文件 hash、方向/交易配置、模型/训练/调度配置、代码与依赖运行环境。普通 from_checkpoint 仍不允许替代完整恢复。日志路径、方法显示名及终止预算不作为学习身份；源节点检查实际 step，不要求等于源 A 最终预算。C 数据变化必须为同 fold 的近期子集。

快照先写临时目录、完成文件校验、最后发布 done 与目录；不覆盖有效节点。启动前估计新增快照空间并至少保留 10 GiB，不足停止，禁止删除其他数据。常规点只有新节点发布成功后才轮换。

## 固定调度

每步学习更新完成→固定报告/Gate→快照。保存下一触发步；恢复不重复同一检查。Driver 每次最多推进 10 步，但在报告、保存、关键及预算边界缩短推进，精确停在目标。正常阶段最终点固定保存；意外中断只从完整恢复点继续，不能标记阶段完成。

验证使用独立进程，不消耗训练 Replay 或影响训练随机计数。同 Seed 的 A/C 比较是共享前缀的配对比较，不能当作独立样本。

## 验收与提交

1. 文档及协议定义提交。
2. 完整恢复与固定调度：最小非模型训练测试，覆盖恢复样本/队列/计数/carry/Gate、精确边界、坏节点拒绝、保留策略；提交代码和结果。
3. 编排：共享前缀去重、先导预算续算、C 固定源节点、全部命令 dry-run 和 guard 校验；提交代码和说明。
4. 本轮不启动任何模型训练。真实 GPU 恢复与 C 分支 smoke 必须后续授权；通过非训练回归不能宣称 GPU 学习已验收。

## 实施证据与限制

- 两仓分支均为 feat/fg-abcd-comparison-v1。文档提交 1724520；Dreamer 第一阶段核心提交 736a192，后续修正见 git log。
- Dreamer 新增 embodied/run/fg_checkpoint.py，FG_FULL_V1 opt-in 协议；旧训练默认 fg_full_checkpoint=False，原保存及墙钟报告规则保留。
- 完整 Agent 额外保存实际 batch 随机计数、policy_params、pending_sync、pending_outs、pending_mets。普通恢复保留；C 新阶段丢弃历史 Replay 写回与待返回指标，新 carry 在载入参数后初始化。
- 非神经网络端到端回归发现原异步预取在相同交互位置取到 [2,3] 与 [3,4]。确认后台 Replay.sample 与前台 Replay.add/update 交错导致取样人口不同。FG 完整恢复模式仍使用预取，但等待预取下一批完成后才继续，固定训练/报告流取样顺序；旧模式不启用这个屏障。
- embodied/tests/test_fg_full_checkpoint.py：10 项通过。包含真实 Prefetch、在线 Replay、部分 chunk、数据更新、RNG/Ratio/carry、Gate 首次通过后恢复、C 新账户/Replay、损坏恢复点回退、保存失败、最近两点保留、精确边界及 worker 异常。
- dreamerv3/tests/test_fg_c_full_node.py：验证受保护源节点实际 70，而源 A 配置目标 100，允许迁移；实际步数或报告配置错误时拒绝。没有初始化任何神经网络。
- 新清单 hash：6890b1da2a9ef1797496308db60593baf0ee76a88816ca99d92ca2ac730252bd。39 个逻辑实例、52 个逻辑阶段、37 条唯一训练线路；全部 52 阶段通过 FG 准入检查，同一线路先导/正式及 A/C 身份一致。
- readiness_audit.json、生成配置、命令及报告在 artifacts/fg_abcd_v1/v3_shared_prefix，仅本地保存，不提交。
- 正式 runner 支持用户显式 --execute；默认只打印模板。本轮 GPU smoke 与先导已获授权，正式训练由用户启动。
- 尚未验证：真实 GPU 上的完整恢复、迁移、传输 guard，以及大 Replay 保存耗时与磁盘增长。不能把 mock Agent 回归或旧 smoke 结果当作新模型运行验收。
- 严格文件 Hash 在保存发布与恢复时检查。常规点轮换仅查看本协议完成节点的身份/保护标记，避免每 10000 步反复读取所有旧 Replay；损坏或不认识的元数据不进入清理。

2026-10-11 补齐：C 的快照身份绑定 `fg_source_steps` 与迁移凭据哈希，70K/700K 不互相恢复。新增父节点回归，相关测试合计 62 项通过。准备检查现由 `tools/audit_fg_abcd_ready.py` 重复执行；C 实际父节点在启动时校验。新版 GPU smoke 正在运行，尚未报告 PASS。
