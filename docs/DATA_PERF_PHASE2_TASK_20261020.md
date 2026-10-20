# FG-S3 DATA-PERF — 第一阶段逻辑风险审查与第二轮全局性能优化

**日期**：2026-10-20
**审查范围**：用户上传 `gy25.tar.bz2` / `d325.tar.bz2` 的实际 Git 对象；对比先前已审查的 `gy24/d324` 公共主线。
**Gym HEAD**：`integration/s3-next = e328cf73e0e284e506e549a35743458029e353a6`；比较基线 `7d6e662`。
**Dreamer HEAD**：`integration/s3-next = b7a693b3c7b0b45eeecdec954818be42d40e67b6`，本轮无新增公共主线提交。
**证据边界**：Git 源码、差异和归档中已有测试文档已阅读；用户报告的107次真实账户查询完整标签相符属用户执行证据。当前审查环境缺 Gymnasium/正式金融原始数据，**未在本环境重新运行完整真实 Env、G3或训练**。因此本报告是源码审查+待回归风险清单，不是正式金融等价认证。

## 0. 结论

第一阶段优化的方向合理：消除重复严格 Oracle 计算、共享精确数学证明、改进标签进程通信、优化 MACD/报价/市场行缓存、加速账户快照和哈希。没有从静态差异中发现已经确证的 TP5/SL30 或 Quality TP Reward 算法改坏；但**三类保护性质或行为确有变化**，需要作为下一次实证回归的明确检查项：

1. **来源监视频率降低**：正式生成默认 `source_guard_interval=1024`，替代原先每次查询检查；收尾检查能够检测持续变化，却不等同于每次查询的即时 fail-closed。
2. **本地生成环节不再重复全部 Teacher 标签核验**：`_behavior_job` 传 `verify_labels=False`；全量标签检查下移至独立 Native G3。生产产物只能标记 `COLLECTED_UNVERIFIED`，通过 G3 前不得作为正式可训练教材。
3. **协议身份已升级**：Strict Suffix 与 Teacher/Corpus 由 V2 转 V3，严格证明编码改变。新产物与旧版不可默默互认，已有 Run/Checkpoint 不得重签。

第一轮107次完整标签比较通过，是非常有价值的回归证据，但不能覆盖全部日期、长历史、跨夜盘/结算、数据异常和所有账户路径。

## 1. 第一阶段源码实际变化及验证建议

| 领域 | 实际实现 | 本次判断 / 必须保持的不变量 |
|---|---|---|
| L2 精确后缀 | `fg_s3_e_quality_strict.py` 构建共享精确后缀、预计算配对数值；`fg_s3_suffix_store.py` 用 mmap 存储精确分子分母、并提供不可变Pair | 不应修改严格 `ADD > WAIT`、tie→WAIT、GAMMA=332/333、资金证书、完整 TRAIN 后缀；逐状态对旧慢版比对**精确有理数**与行为标签，不能只比浮点。 |
| Teacher 后验 | `fg_s3_quality_teacher_v2.py` 中 clone 使用原生 Env，报价和市场行由 private cache 加速 | 每次合法 ADD 仍执行原生金融过渡；缓存不能包括资金、持仓、成交、Reward 或账户决策。修改条款、源对象、更换行情、跨交易日与尾部应重新校验。 |
| MACD | `e_data_v1/core.py` 改为 session 索引、numpy列、有限容量 LRU + 稀疏checkpoint，去掉多次 Series构造 | 要验证全量前缀顺序与跳跃访问、checkpoint前后、跨 session、最后不足5m的桶、延期可见价格、负例篡改；不能只用连续访问的短样本。 |
| 标签传输 | `fg_s3_quality_label_pool.py` 每 worker 自有有序 history replica，进程间只传增量 | 同账户顺序保持、worker 交错、reset/新stream、回退拒绝、历史篡改拒绝、account_snapshot_hash 与 Driver/G3 逐字一致。 |
| Snapshot/哈希 | `snapshot_json.py` 对标准 `equity_history` 行直接复制；`account_fingerprint.py` 缓存历史前缀 JSON 字节 | `export_state` 必须返回真正独立的完整 JSON 快照，`import_state` 复原金融状态；带 `-0.0`、非标准行、复杂meta、变更旧行与历史缩短时 Hash 不可错。 |
| Producer/G3 | Generate 不再在本阶段全部重算标签，但仍进行原生账户 Replay 对账；G3单独重算每条标签 | 如果 G3未运行或失败，禁止Reader admission/正式训练；G3必须有**独立重建**的精确后缀缓存、原始金融账户、全部标签数量对账。 |
| 契约/导入 | E-Data MACD纯计算通过已审计的私有加载隔离 Gym/数据库依赖 | 原`macd_5m_causal.py`数学不变、源SHA认证；其加载路径仍有历史 AST/exec债务，独立架构整理，不应在本轮性能优化中顺便无证重写。 |

### 1.1 实质性风险：来源检查降低频率

`fg_s3_e_quality_corpus_cli.py::generate_small` 将 `source_guard_interval` 设置为1024。`NativeQualityTeacher.query_state()` 第一次及每1024次查询检查 `_source_stats`，`close()` 再检查。该策略能捕捉大多数持久文件变化，并在关闭时拒绝，但旧代码是每次查询都检查；因此**异常源文件在查询中暂时改变又恢复**的检测性质变弱。这是安全不变量变化，不是纯性能重构。建议单独衡量每次 `os.stat`/文件身份检查的成本；优先恢复低成本的逐查询元数据防篡改，保留批次级完整 SHA 和 close guard，而不是直接取消即时检查。必要时在文档中明确不可恢复的时间窗口，并在正式数据认证 Gate 阶段拒绝这一折衷。

### 1.2 实质性风险：验证推迟

`generate_behavior(... verify_labels=True)` API 仍支持完整本地检查，但正式 `_behavior_job` 设置 `verify_labels=False`；`reconcile_trace()` 此时仅验证金融/Observation回放而不重算 Teacher。代码将状态写为 `COLLECTED_UNVERIFIED` 并在 `verify_corpus` 中要求每个行为 `teacher_labels_verified == records`；正式Reader准入路径另要求独立G3。这是**门槛转移**而不是算法等价的自动证明。必须做负例：缺G3、G3只校验部分标签、G3失败、G3结果属于其他Code SHA、训练器试图强行打开均 fail-closed。

### 1.3 目前没有独立验证的风险

- 不能从107个查询推广到六账户×全区间的所有候选。
- 未在本轮审查环境重复运行用户测试。归档报告记录的是定向测试而非所有真实长程金融路径。
- `NativeMarketExecutionCache` 将 DataFrame行/有效条款保留在内存：必须额外验证**同一对象内发生原地更改**不会使用过时缓存。正式来源冻结是一道屏障，但应有负例而不是依赖注释。
- MACD 有限 LRU/稀疏checkpoint 性能与正确性应在长行情、乱序访问、暂停与恢复时复核。

## 2. 第二轮性能证据——先做对的总量分解

用户提供的107个真实账户查询 Profile：

| 测量项 | 累计时间 | 约占97.5秒 |
|---|---:|---:|
| `copy_account_snapshot` | 62.1 s | 63.7% |
| 历史 Pickle 比较 | 8.7 s | 8.9% |
| 2265次原生金融步骤 | 10.9 s | 11.2% |
| 其他开销（仅在前三者互斥时） | 15.8 s | 16.2% |

367次完整复制 = 107次worker恢复 + 107次Teacher账户导出 + 51个合法L1×3次（clone恢复、最终状态导出、源账户未变校验）。平均每次约0.169秒。**复制的语义理由是成立的，但“必须独立所有权”不推出“每次都必须物理复制全部历史”**。

若62.1秒被完全消除且其他部分不变，则97.5秒理论最低35.4秒，约2.75x；这只是本轮查询Profiler 的 Amdahl 上限，不是完整Generate/G3的已达成加速倍数。Pickle计时可能包含于其他计时段，不能在不知道调用树前直接相加作为净收益。

## 3. 全局核心建议：从复制函数转到“账户状态所有权/表示”

### 3.1 明确三类状态，不要再把它们当一个对象

**A. 公共完整快照 `export_state()/import_state()`**：保持已有 JSON Schema、字段、状态哈希、Source/Resume身份和独立可变所有权。正式Archive、FullCheckpoint、Reader、G3外部边界仍用这套合约。

**B. Worker/Teacher 的内部只读历史前缀**：同一账户的已完成 `equity_history` 历史一旦验证、承诺不可变，可用持久化分段、不可变tuple块或版本化append-only视图共享。来源Driver与多个私有clone持有同一冻结前缀，**无共享可变列表**。每条新样本只追加到所属账户的私有尾段。

**C. 当前时刻金融热状态**：资金、Ledger、Entries、fills、reward_state、RNG、当前Step、E snapshot等必须拥有独立实例；clone的模拟事务不得回写真实账户。小状态与历史分离，恢复时只复制热状态并切换到指定历史版本。

这样既保持旧保护边界，又不再每次递归复制2万多条历史行。

### 3.2 第一优先：内部“冻结历史前缀 + 可写尾段”实验

新增**仅供原生Teacher的内部协议**，不更改公共 `export_state` 返回值。建议命名 `NativeAccountSnapshotView` / `ForkAccountState`（实际名字按原owner确定），字段包含：来源身份、真实账户owner、市场游标、已认证历史的版本/长度/哈希、只读历史chunks、独立可写尾段、热状态。`clone.restore_private()` 的结果必须与 `clone.import_state(full_public_state)` 在相同输入上完全等价。

可考虑每256/512条历史行封存一个不可变chunk，新查询复制少量尾段；但**chunk大小只是可调实现参数**，不是正式数据协议。正确方案必须显式处理：APPEND、新账户起始、rewind、reset、历史行篡改、不同账户同市场时刻、跨worker/进程、跨合约、主力切换，以及 `equity_history` 列表兼容读者。

目标是**降低历史复制复杂度**而不是仅加快每行拷贝：从每次 O(H) 复制降低到近似 O(尾段+热状态)；不能通过取消真实来源校验达成表面速度。

### 3.3 第二优先：历史Pickle比较转成版本/内容承诺

当前 `AccountFingerprint.digest()` 每次执行 `pickle.dumps(history[:count])`，需要重新切出并序列化完整历史前缀，尽管后面的SHA已经缓存。这个8.7秒是**按旧历史重新证明旧历史没有变化**的成本。

更好的结构是在历史封存时计算每段内容 hash（或累积 SHA 前缀）；当历史仅通过受控追加API演进，旧块不可变，验证版本ID、chunk根哈希、行数、尾段哈希即可。公共 `account_snapshot_hash` 仍须按**原本的排序JSON SHA256字节序列**输出，不能用新的Merkle哈希代替公共身份。如果外部/测试直接修改历史行，要么封存后禁止变更并明确抛错，要么强制回退到完整慢路径重新认证。

### 3.4 第三优先：减少重复公共快照构造，但保留对账

最值得检查的结构如下：

```text
Driver 当前账户（只读）
  ├─ 一次取得带所有权证明的账户版本 + 当前热状态
  ├─ 107次 worker 恢复：传增量 + 私有历史版本引用
  ├─ 107次 Teacher根账户证明：复用已认证版本的同等JSON SHA，不重物化历史
  └─ 51个合法L1
       ├─ clone仅复制热状态，历史前缀冻结共享
       ├─ clone原生金融Env执行 ADD→TP/SL（仍实际执行2265步）
       ├─ clone结束时对其**新追加尾段**构建完整公共状态/Hash证明
       └─ 源账户前后同版本/热状态一致性证明，必要时周期性完整导出验证
```

**重要**：不允许只用 `id(env)`、长度或同一对象引用就声称源账户没变；必须结合不可变历史内容承诺、热状态审计与原始公共哈希。不能依靠测试期抽样代替正式独立G3的账户一致性。

### 3.5 不要忽略复制优化后的第二瓶颈

源码还存在 `EntryEnv.import_state()` 重建Ledger/PositionManager/Rewards及 `_get_obs()`，`EntryEnv._prefix_hash()` 通过 `MarketPrefix.matches()` 做 DataFrame 全量相等验证；`NativeQualityTeacher.query_state()` 使用 `env._raw.equals(self.clone._raw)` 每查询验证源市场相等。这些检查理论上可能成为删除快照复制后的新热点。需要**在上一轮修改后立即重新测端到端结果**，再定下一优化项。不能先凭静态判断把它们全部关闭；可以把可信源版本化、固定不可变数据并加入单独变更检测，保留真实审计能力。

## 4. 推荐迭代顺序（每一步都要修改代码并测速度）

| 顺序 | 代码任务 | 验证 | 失败处理 |
|---|---|---|---|
| 01 | 基于现有107次真实查询固定基线，记录调用树和源版本 | 同一来源、账户状态、动作、严格标签字节/语义对照 | 不能建立相同工作量的基线则先修基准 |
| 02 | 将L1 clone还原的长历史从全复制改为内部冻结prefix+可写tail；先走最小可控路径 | 相同107条所有标签与来源Hash、clone末状态、旧账户不变完全一致，并验证异常/rewind | 如恢复失配，快速撤回且保留独立隔离 |
| 03 | 将Teacher根状态/worker导入接入同一内部所有权模型，去掉两次导出造成的重复物化 | 107条完整等价、不同workers与乱序批次、连续stream和reset | 继续保留公共慢路径直到新旧等价 |
| 04 | `AccountFingerprint` 使用已认证不可变chunk承诺替代每次全历史Pickle，仍输出旧JSON SHA | 所有新旧Hash逐字一致、负例篡改 fail-closed | 只对证明充分的内部记录启用fast path |
| 05 | 对改完后的真实多交易日、六账户小样进行 Generate → G3 → Reader/Coverage完整比较 | 相同金融状态和L1/L2标签，独立G3完整PASS，总 wall time/吞吐上升 | 不以单函数提升代替端到端效果 |
| 06 | 继续针对新 Top-1热点优化，直到主要可行方案进入平台期 | 每轮留下旧/新提交SHA、耗时、吞吐、真实性差异报告 | 退化回滚，保留最快正确版本 |

**不要中止当前冻结生产；新实现、Schema或快照协议变化需要新代码身份。** 优先基于当前主线另建独立 Data-Perf Worktree。不要为了性能改动正在运行的正式Producer及旧教材的收据。

## 5. 关键等价性测试矩阵

1. 完全相同的当前真实账户：旧 `export/import` 与新private restore，余额、Ledger、Reward、有效mask、Observation、RNG、e_snapshot以及后续逐步行动相同。
2. 源账户独立：克隆后修改/追加历史、做ADD、TP5/SL30、异常退出，Driver所有字段与公共HASH不变。
3. 旧历史篡改：直接修改早期 `equity_history` 行、删除、替换、更改 -0.0/0.0 或 `True/1`，不能静默复用旧证明。
4. 公共合同一致：公共 `export_state` 保留完整JSON和独立所有权、标准`import_state`兼容历史数据与Checkpoint。
5. 金融极端：跨官方交易日/夜盘、日尾不强平、资金不足、非法ADD、TP/SL同bar歧义、强制WAIT、到数据边界RIGHT_CENSORED、reset/rewind。
6. 老师：L1所有事实和风险窗口、L2精确大小关系、tie→WAIT、严格资金证书、全未来仅在TRAIN后验。
7. 缓存：MACD会话与偏序访问、同一DataFrame原地变更、source文件变化、cache eviction、冷恢复和fork/spawn跨进程，必须拒绝污染。
8. 原生G3：六种P0–P4全部账户重播及标签重新计算，与生产缓存隔离；未通过G3则训练Admission拒绝。
9. 持续性能：107-query micro、足够覆盖正常TP/SL的中等长区间、真实两官方交易日（允许离线独立试验，不需要整份28,264分钟重跑多次）。

## 6. Codex持续优化指令

> 直接开发第二轮DATA-PERF。以双仓实际最新`integration/s3-next`为基线，在新的独立Worktree工作。审查本文件第1节列出的第一阶段逻辑风险并补足针对性负例；不要因此停在审计报告。
>
> 107查询基准定位到367次完整历史快照复制共62.1秒。按“公共完整快照 / 内部不可变历史前缀 / 私有可写热状态”拆分生命周期，**先把 L1 clone 恢复真正改成不复制全历史，再改 worker/Teacher根状态，随后消除每次pickle整个前缀**。不能简单删除恢复、最后状态导出或源账户隔离校验，它们的保护语义必须由新的不可变版本和完整原始SHA证明替代。
>
> 每次代码改动后立刻在同一真实 TRAIN 基准比较速度与完整107条标签，变慢撤回，变快继续下一个瓶颈。关键金融、因果、所有权和协议更改完成后，再执行真实多日六账户 Generate→Native G3→Reader 等价验证。避免重复跑整份28,264分钟合约。保留冻结生产批次，不读取DEV/TEST价格，不自动启动正式F1训练。
>
> 提交每轮SHA、Profiler前后结果、实际端到端wall时间、总吞吐、测试PASS/FAIL/NOT_RUN、剩余热点。**只有真实端到端提速并保证等价，才算优化交付。**

## 7. 独立反向Review

- [x] 未把107查询的标签相同当成所有市场场景已验证。
- [x] 未把“有保护用途的复制”误判成可以直接删除；提出不变所有权语义下的结构替换。
- [x] 未把公共SHA替换成私有指纹；公共身份仍需逐字保持。
- [x] 保留全部原生金融执行、TP/SL与账户对账，不牺牲G3。
- [x] 明确指出来源检查频率降低是实际行为变更，要求复核。
- [x] 隔离生产冻结身份、DEV/TEST和正式训练。
- [x] 真实加速倍数尚未测得，所有理论值均标注为上限或推断。
- [ ] 当前机器独立长程性能与金融回归：NOT_RUN，需在项目环境运行。

**最终判断**：第一阶段可以作为第二轮优化的代码基线，但仍应补做保护边界回归。第二轮的最高投资回报方向是**去掉历史“重复物化”，而不是去掉金融审计或继续优化单条dict的复制循环**。稳定接口可保持不变，内部状态表示应允许为快照所有权做一次有证据的大幅重构。

## 8. 追加代码复核：最容易遗漏的两条逻辑路径

### 8.1 L2动态规划的时间折现分解

`S3MarketClock.reward_exponent(decision,feedback) = prefix[feedback] - prefix[decision+1]`，`advance(before,after) = prefix[after+1] - prefix[before+1]`。因此对合法的 `cursor <= entry < feedback`：

`reward_exponent(cursor, feedback) = advance(cursor, entry) + reward_exponent(entry, feedback)`。

这是新 `_value` 将完整后缀拆为“当前entry局部收益+下一entry续值”的必要数学等价条件。**已按实际Clock公式核对恒等关系**；但交易实际排序、release边界、无效分钟及下一入场指针是否覆盖各种时间边界，仍必须以旧标量Oracle作为独立参考实现进行实测。不应仅让生产者和G3都调用相同的新动态规划函数，再把两者一致当作旧算法完全等价。

### 8.2 E原生反事实执行跳过Observed Wrapper

`fg_s3_quality_teacher_v2.py::_trace_native_trade()` 在E分支将 `execute_observed(clone,action)` 改成直接 `clone.step(action)`，再将奖励转为 `float32`。`execute_observed`本身只是围绕同一个Env reset/step收集审计Replay字段，理论上不会改变原生金融执行；新分支减少了大量临时bar、fills、events复制。仍需增加**逐步原生Trace与旧Wrapper**的对照（包括chunk结束/重启、entry、exit、reward feedback、cursor、Reward Event、以及right-censored），不能只比较汇总标签。非E `shadow` 分支仍走原来Observed路径。

### 8.3 保护机制分层决策

本轮确认的正式保护强度变化，不能以“同一个代码版本的生产者和校验者互相一致”替代风险评估。推荐同时保留三种证据：**旧算法逐标签Golden Reference；当前新版独立Native G3；源版本与账户不可变/内容承诺负例**。这三者负责不同性质，不可相互替代。
