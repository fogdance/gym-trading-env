# R6 / R7 实施 Review

最终验收状态及计数由 `receipts/FINAL_ACCEPTANCE.json` 给出；本文件描述实际代码、验证范围和不可混用的证据等级。双仓集成与最终交付 HEAD 见 [基线与集成](BATCH2_BASELINE_AND_INTEGRATION.md) 及工作区 `baseline-lock.json`。

## Summary

R6 将 F1 冻结参数与 optimizer slots 的字节审计集中到一个 Inspector。R7 已实现完整方案：Reader 嵌套增量事务、跨角色/来源回滚、Prefetch 多版本、覆盖图自包含 Checkpoint，以及 F1/F2/F3 Run 接线；无需退回仅局部事务方案。

`FINAL_V5_*` 只是第 5 轮冻结代码验收的收据前缀，教材仍是原生 V3、ABI 3.1.0，不代表创建了 V5 教材。

最终运行代码固定在以下干净、互为兄弟的测试 worktree：

| 仓 | 最终代码 SHA | 测试目录 |
|---|---|---|
| Gym | `4c881d39d9a482542534ee678e53787547afae89` | `rounds/final-v5/gym-trading-env` |
| Dreamer | `a6aa3e649ca36022b767e2d89d0392868953f5e9` | `rounds/final-v5/dreamerv3` |

目录相对本次工作区 `/home/v/Documents/work/research/fg-s3-batch2-r6-r7-20261021/`。文档交付提交与普通 merge 会有不同 SHA；不把这些 SHA 混称为原始测试 HEAD。

## Evidence observed / Root cause

R6 原代码的 `inspect_f1` 未调用完整冻结 Inspector，直接 F1→F3 与另外两条路径不对称；slot 名只按尾部 `count` 豁免也会接受未知所有者。新合成回归在原代码上记录 19 PASS / 15 FAIL，修复后 34 PASS，原始收据为 `R6_before` / `R6_after`。

R7 原 Reader、Adapter、ActorStreams、MultiCorpus 和 Prefetch 都可能在每批复制、排序整个累计 seen 集合。实际采样 SQL 与抽样分布无需改变：将撤销状态改为本批新增键，将 Prefetch 状态改为具有明确 owner 的不可变版本引用，完整物化移到 Checkpoint 边界即可。

额外故障测试发现：只让 `course_reader.pkl` 绑定最新节点摘要，不能独立绑定压缩后的覆盖基础。`coverage_basis` 在旧实现上真实失败，现由跨成员完整图摘要拒绝。另一个回归证明，对已恢复 pending 的未启动 Prefetch 再次 load 会接受混合状态；现在 load / load_full 均先拒绝。失败收据完整保留。

最后的回退/回收测试复现：废弃分支收集后，原冷恢复将 next_id 从 3 重建为 1。现在完整图精确保存分配计数器并校验其严格高于已保留 ID；正常线性采样和金融结果不变，显式回退后也不复用历史版本号。原始失败见 `R7_version_counter_before`，最终复验包含该正例和恶意计数器拒绝负例。

来源重排负例也已真实复现：canonical 摘要忽略字典插入顺序，但覆盖向量按 Reader 遍历位置解释。旧解析器接受了重排的 Reader 映射；现在每个版本都核对有序静态 Reader 身份，在任何原 Reader 状态改变前拒绝。正常采样与保存格式不变，原始失败保留于 `R7_reader_order_before`。

## Changes made

- Dreamer `embodied/run/fg_s3_e_quality_transfer.py:39`：`inspect_f1` 唯一调用完整冻结证明；`inspect_stage(F1)` 复用，`inspect_stage(F2)` 重验上游并核对 V2 lineage。`source_freeze_evidence:105` 复用既有 `state_owners` / schema 和实际保存的 optimizer 配置，精确识别可变时钟、WM slots 与受保护 slots。`transfer_parameters` 未改。
- Gym `fg_s3_reader_transactions.py`：mark 记录 RNG、计数及本批新 market/account 键；内层 commit 合并 undo，外层失败仍可精确撤销。Group 覆盖 scheduler、角色/来源统计、audit，来源 Guard 失败不会被回滚“洗白”。Reader 原完整公开状态接口保留。
- Gym `fg_s3_reader_versions.py`：一层生命周期 wrapper 复用原 WP03 Reader 和现有 Adapter；不是新 Replay。每个 Reader 只允许一个图 owner。版本包含 session、来源/角色/索引绑定、parent、small state、精确 delta 和 native batch 摘要。
- Prefetch：区分 source 最新位置、已消费位置和每个 pending 的位置；正常推进回收不再使用的 lease 并压缩共同历史。完整 CP 只保存一份覆盖基础及所需版本图，不为每个 pending 复制整个历史。
- `course_reader.pkl` 的 `S3_READER_CHECKPOINT_REF_V2` 同时绑定当前 ref 和整个 `graph_sha256`；runtime 持有唯一完整图。FullCheckpoint 的 pause 内导出一次，由 runtime 消费同一份图。冷恢复先校验图、跨成员绑定、来源及 Journal，再构建/恢复 Agent。
- 正式 F1/F2/F3 Run identity 绑定 `READER_STREAM_STATE_V2` / `PREFETCH_PENDING_STATE_V2`。单源 Run 已接线；MultiCorpus / MultiCorpusF3 使用相同 wrapper 与 Runtime/Checkpoint 接口，无阶段专属数据系统。原多源诊断 CLI 保留完整旧快照模式，其回归亦执行；这不等于新增正式多源训练入口。

完整生命周期、回收上界、格式和接入示例见 [状态合同](R7_STATE_VERSION_CONTRACT.md)。

## Verification performed

以 `FINAL_V5_*` 为最终冻结代码复验，`R6_*` 为已独立合入的来源/参数接续证据；之前轮次单独列出，不能当成当前自动 PASS。准确测试列表、源 SHA、dirty 状态、argv、cwd、环境、退出码、JUnit 及去重计数由 [测试收据索引](TEST_EVIDENCE_INDEX.md) 和 `receipts/FINAL_ACCEPTANCE.json` 提供。

验证覆盖：六种来源拓扑逐批数组和完整原 Reader 状态等价；嵌套 undo 与 audit/第二角色/第二来源/外层 merge 故障；0/1/2/4 pending；新 Python 进程 N+M 对照；模型、optimizer、carry、RNG、schedule、覆盖集合及队列；大于 100k 历史覆盖；失链/重复/跨角色/跨来源 ref/乱序 pending/教师字节/基础覆盖篡改；缺 done、latest 回退、Journal 和 CP 成员损坏；真实 F1/F2/F3 类在无授权/身份错误时拒绝；既有非 S3 Prefetch 与 Native Replay 共享恢复回归。

保留原始冷恢复目录、完整 CP 成员清单及哈希、重新加载后完整图相等证明；索引见 `receipts/FINAL_V5_COLD_ARTIFACT_INVENTORY.json`。不是只比较下一批，也没有用热进程注册表冒充冷恢复。

性能使用相同合法合成输入、seed、权重、B=4、T=8、K=4、pending=2 与固定 CPU affinity，每工况三次。1k/10k/100k/300k 集合属于用户批准的 **DEV_ONLY 覆盖状态压力**，不计为实际市场覆盖。全部输出批次和最终原 Reader 状态的强哈希相同，且每次新进程 CP load 状态相同。表、分母、RSS、快照/回滚与 CP 保存/加载成本见 [性能迭代](R7_PERFORMANCE_ITERATIONS.md)。不把 Reader 吞吐提升表述为 Producer 或完整训练加速。

## 逐项反向 Review

| 项 | 核对结果 |
|---|---|
| 1 第一批实际集成 | 起始双 HEAD 已包含第一批；以 ancestry、实际源码与本次回归核实，未重复 merge 归档版本。 |
| 2 共同 F1 Inspector | 三条路径均取得同一 source_freeze V2；直接 F1 分支未重复做第二套冻结审计。 |
| 3 slots / normalizer | 用真实 owner/schema、精确时钟和 slot 映射；未知 slot、受保护值、键/shape/dtype/finite 漂移拒绝。 |
| 4 R6 语义边界 | 至少一组 WM 参数改变即可，逐组计数不等于六组已学会；Loss、参数结构与冻结白名单未改。 |
| 5 增量回滚 | 只记录新增 key；支持 LIFO 嵌套及外层 undo，来源损坏保持拒绝。 |
| 6 三个采样前沿 | source / consumed / pending 各自引用对应 small state 和精确覆盖版本；不能统一为最新 RNG。 |
| 7 自包含 CP | 图、基础、delta、pending 数组和 ref 均在 CP 字节内；不依赖旧节点或进程外注册表。 |
| 8 严格最新与 Journal | `fg_checkpoint.py` 字节未改；新图验证和既有 Journal 检查均在 Agent 恢复之前。 |
| 9 ABI / pin / Run | ABI 3.1.0、来源/验证器 pin 及 prepared 验证未改；新状态协议仅供新 Run，旧 Run 沿用冻结代码。 |
| 10 金融与采样等价 | Env/Reward/Teacher/L1/L2/Loss 文件字节未改；完整原生数组、Owner/来源及精确 uniques 等价，非近似计数。 |
| 11 非 S3 | Prefetch 新分支显式 opt-in，legacy / Native Replay 继续原协议并有共享恢复测试。 |
| 12 真实性能 | 测量 Reader+Adapter+Prefetch、RSS 和完整 CP；另列微基准，未用其冒充总体吞吐。 |
| 13 证据等级 | 合成 CPU 工程验证、DEV_ONLY 压力与真实训练/G3/GPU/TEST 分开；后四者未运行。 |
| 14 可回滚边界 | R6、局部事务、版本图分独立提交；图未降低 strict_latest 或覆盖要求。本轮图已通过所列冷恢复，无需发布局部回退。 |
| 15 实际交付 | 双仓源码/测试提交、原始失败与修复收据、性能重复、普通 merge 和版本锁均提供实际身份。 |

## Risks / not verified

真实已发布 TRAIN Corpus 的新版本训练集成、正式 Native G3、GPU、长期正式训练与模型研究效果：**NOT_RUN**。未读取 TEST 行情。CPU 小模型测试含受控合成更新，不能称为正式训练准入。数据压力历史键为人工种入，不能称作真实分钟/账户曝光。

精确集合仍需 O(累计覆盖) 内存，完整 CP 仍需相应物化、校验及 I/O；本轮解决的是每批重复复制与 pending 重复持久化。新的 CP 格式不升级任何既有 Run。pending 的 native 数组由版本摘要绑定；Agent 添加的 seed 保留于 runtime 的完整成员字节及其 CP 文件哈希，恢复不重新生成。

## Next recommended step

后续以新冻结双仓准备新 Run，沿用正式 Corpus 准入/Producer pin/G3 和用户训练授权。真实 TRAIN/GPU 验收另行执行；本轮只给出 `B2_CODE_GO` 工程结论，不给模型质量或策略收益结论。

## 最终验收状态

**B2_CODE_GO_INTEGRATED**：最终冻结代码 330 个不重复用例 PASS；集成目录再执行 70 项 PASS，合计 400 次验收执行，0 FAIL/ERROR/SKIP。两仓代码合并 tree 与冻结测试 tree 相同。12 个工况各三次，旧/新批次与全 Reader 状态哈希一致，所有新版均有新进程 CP load 比对。

| 仓 | 源码普通合并 HEAD |
|---|---|
| dreamerv3 | `a6a6ad9e18196f799c4517f370dd26a4ca68aaf5` |
| gym-trading-env | `1073542163849540ea4ac9a626310cee2f3a8932` |

真实 TRAIN/G3/GPU/长期训练保持 NOT_RUN；TEST 保持 NOT_READ。最终文档交付 HEAD 和运行源码一致性由工作区版本锁记录。
