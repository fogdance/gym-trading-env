# WP04 D：Reader 与完整恢复

复用 `EQualityCourseReader`、`Agent.stream` 的原 Prefetch/种子协议、`StepSchedule`、`RuntimeState` 及原 FullCheckpoint 原子 bundle。新增 `external_immutable_course_reader` 后端，与 native Replay 分支显式区分，禁止 `cp.replay=reader`。原 Replay 后端行为由既有 15 项非神经回归与六项 E/账户恢复回归验证。

完整保存 Agent 参数/优化器/策略缓冲/计数；Reader RNG、来源与 index、曝光状态；原 Prefetch 已排队的下一批及其 seed；Python/NumPy RNG、空 carry 规则、有效目标/预热/更新计数和 schedule。快照引用外部 Corpus/index 并重验；只保存必要的待消费 batch，不能把“未复制整份 Corpus”说成没有任何行情数组。最新 F1 bundle 损坏必须拒绝，保留旧点但不自动回滚到 step 0。

热路径只做 Reader 缓存/文件状态检查；完整哈希在 checkpoint save/load，G3 在显式数据准入，不每步执行。

首轮真实失败：

1. D_cold：JSON 将 config tuple 序列保存为 list，内存对象直接比较误拒绝；D_identity_observation 逐字段记录。使用既有 `plain()` 统一序列表示，未忽略任何身份字段。
2. D_cold_retry：跨进程连续4步与2+2已逐字节通过；负例只复制被篡改 SQLite、漏复制原合法收据，先因 missing JSON 失败。补全合法夹具后再篡改，保持生产拒绝标准。
3. E_required_shared：新 WM 只读诊断 keyword 与历史 extraction 审计结构不一致。WP03固定基线该项 PASS，C 引入新签名后失败。审计显式识别默认 False 的新签名，仍将完整 WM/诊断分支纳入 AST 比较。旧 WP03→F1、默认 True、数学/Helper/factory 变异全部必须拒绝。没有扩大历史全状态迁移资格。

最终 D_final 在持久的隔离 artifacts 路径运行，避免 pytest 临时目录清理掉恢复原件。每次试验命令、退出码与旧失败保留；最终结论以原始 JSON/XML 和现存 bundle 为准。DEV/DATA/FORMAL 分级遵从用户最新规则，见同目录分级文档。
