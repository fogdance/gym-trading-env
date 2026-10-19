# FG-S3 E+Quality TP Master Roadmap V12 — WP07交付 / WP08正式开发

**更新：2026-10-19。** 以用户最新提交的`gy23.tar.bz2` / `d323.tar.bz2`归档内双仓 `integration/s3-next`为本次唯一已审查源码。保留V11及更早历史于本文末尾，不把历史未更新状态当作当前事实。目标不变：**学生掌握市场基础→理解入场质量→独立决策→受控在线练习→真实独立考试；合法空仓优先WAIT，优质机会ADD，追求快速TP5、小逆行、控制SL30与净风险。**

## V12.1 双仓冻结源及工程/训练区分

| 项目 | Gym | Dreamer | 意义 |
|---|---|---|---|
| 最新公共 `integration/s3-next` | `0a42553` | `745b7e0` | WP07已集成与报告提交 |
| WP07行为复验冻结代码 | `1a28f0a20866ea2f6c2d38f31a963b42ca92155a` | `232e57f70b2751adfc9733d7e695d834b73ce439` | 两HEAD之后的差异仅docs和收据 |
| 工程验收 | 247项选定CPU PASS（归档实施报告），WP07 A–H完整 | 共享Agent / RL / Checkpoint | 并非全项目GPU/真实研究PASS |
| 正式 F1/F2/F3/F4受训学生 | **本次仍无已验证完成的正式权重** | — | 不可宣称学会市场或入场 |
| 真实DEV/TEST考试 | NOT_RUN | — | WP08做工具，WP10最终封存TEST |

源码报告已证实WP07 CODE_GO：SHADOW、onlineBC、同一 `quality_actor` 想象式RL受控代码、来源接续与冷恢复。受训正式来源、RL Gate独立Research/WM calibration、真实多Corpus训练与DEV评价仍需另行证明。

## V12.2 WP01–WP10 大功能 / 小任务 / 当前状态

| WP | 大功能 | 小任务/交付 | 2026-10-19状态 | 接续 |
|---|---|---|---|---|
| WP01 | 新课程合同及身份 | E/Quality TP/MACD V3课程、阶段权限、来源身份 | 工程已交付，课程由后续版本迭代 | WP02–08 |
| WP02 | 原生教材与老师 | 5m部分MACD、P0–P4、L1质量、L2严格动作、TP/SL/风险账本 | 工程已交付；完整V3新真实生产数量按生产收据另核实，不能凭计划算已生成 | WP03/DATA |
| WP03 | Trainer统一监督Reader | Trace×Replay×L1/L2 Join、Natural/Aux、Reader恢复 | 工程已交付，公共多Corpus接口后续已扩展 | WP04–07 |
| WP04 | F1学生市场基础 | E Encoder/RSSM/WM heads的Quality TP监督、K-step burnin、WM-only更新与CP | CODE完成；正式真实F1权重未提供 | F2/F3 |
| WP05 | F2机会质量理解 | Event/Touch/Adverse/Score四头、F1参数接续、冻结/CP | CODE完成；正式真实F2权重未提供 | F3 |
| WP06 | F3独立开仓决策 | strict L2 BC、Pair、辅助L1、唯一quality_actor更新、原生短诊断 | CODE完成、主线已集成；正式真实F3权重未提供 | F4/WP08 |
| **WP07** | **F4 SHADOW、在线监督、受控想象式RL** | A来源/运行合同；B Teacher-free原生采集；C延迟后验与混合；D在线BC；E quality_actor RL；F探索/Gates；G原子CP；H统一CLI与回归 | **CODE_GO，实际发布在当前s3-next**，报告247 PASS；真实长期训练/RL研究NO-GO | WP08 |
| **WP08** | **Actor-only独立DEV考试** | **A 考试源/身份；B F3/F4只读模型；C 多合约长区间native Session；D TP5速度/MAE/SL/成本；E公平基线；F支持量与统计；G Eval Session恢复；H CLI/报告/验收** | **本次源码接口Review完成、正式Codex执行任务书已交付；未开发** | WP09/WP10 |
| WP09 / DATA | 正式教材扩量 + 多Corpus + 消融 | V3 Corpus生产及覆盖，MultiCorpus Natural/Aux/RNG精确恢复，多合约模型消融及评估 | **多Corpus公共Reader已出现在WP07源基线**；V3真实生产完结、真实训练使用证据仍需新收据；WP09消融尚未验收 | F1–F4/WP08/10 |
| WP10 | 候选冻结与最终TEST | 双仓/数据/模型/指标/seed/baseline/授权冻结；单次最终TEST，研究GO/NO-GO | 待实施，不代表实盘交易授权 | — |

## V12.3 依赖关系（大功能 / 小任务）

```text
WP01课程 → WP02教材教师 → WP03统一Reader → ┌─ WP04 F1（市场） → WP05 F2（机会风险）
                                             │                          │
DATA V3生产/Native/G3 → DATA-MULTI调度 ─────┴──────────────────────────┤
                                                                        ↓
                                                                WP06 F3 quality_actor
                                                                        ↓
                                             WP07 F4 Shadow → onlineBC → controlled RL
                                                                        ↓
    EVAL-BASE / WP08 A–H：独立认证DEV → Teacher-free长区间 → 风险/财务 → 基线
                                                                        ↓
                                                 WP09 分层泛化/消融/比较
                                                                        ↓
                                               WP10 冻结候选 → TEST
```

**代码完成与真实训练两条轴分开看：** WP04–07代码已实现，但真实连续已受训F1→F2→F3→F4尚缺完整受信源与权重；应持续推进DATA真实生产、实际F1基础训练启动准备和GPU/数值诊断。WP08 CODE开发可用合成数据，不必等受训学生存在。

## V12.4 WP07→WP08 真实接口变化

- F4 `online-evaluate` (`research/fg_s3_e_quality_online.py::evaluate_checkpoint`)：只在TRAIN来源+≤512步读Checkpoint，不能当WP08考试。
- F3 `evaluate_native` (`research/fg_s3_e_quality_actor_checks.py`)：Teacher-free Argmax与native对账的可靠底层，但步数≤512。
- 原始 `ActorPolicy` + `fg_replay.collect_trace/reconcile_trace`：WP08的共享policy/金融真源。
- WP07 `OnlineCollector`要求TRAIN fold、在线持Replay/账号：**WP08不可把DEV伪装为TRAIN**。应设计只读ExamSession，复用底层Env/Driver/trace而非在线更新协调器。
- 旧固定FG409/C1 `prepare_dev` 不适用于新的E/MACD V3 Quality TP独立DEV考试；需建立真实validation来源/E Profile准入。
- F4 `FullCheckpoint`是只读**模型来源**，WP08独立EvalSession保存账号与carry，不恢复online训练。

源码审查和最终任务详情以两份本版文件为准：`FG_S3_WP07_to_WP08_Interface_Code_Review_gy23_d323_20261019.md`、`FG_S3_E_Quality_WP08_Actor_Only_Exam_Final_Codex_gy23_d323_20261019.md`。

## V12.5 开发集成与严格度

- WP08在独立worktree按A–H逐步提交；先定向CPU合成，至少一条>512步Teacher-free合法轨迹、一组公平基线及全新进程冷恢复；之后源码Review通过再双仓成对合并 `integration/s3-next`。
- **DEV**：必要的是金融真实账户/时钟/TP/SL、因果E+partial MACD、Teacher隔离、冻结模型、准确分母，开发期间不必每次完整G3或资源巡检；假定CPU/GPU/RAM空闲可高利用率。**DATA发布**：真实来源/Reader/标签身份与样本分母。**正式考试**：真实DEV合法验证源与已训练Checkpoint，不能拿TRAIN替代。**TEST**：只由WP10授权。
- 每次新上传代码先做接口Diff与变更SHA审查、更新当期WP验收和DATA/正式训练状态，保持本路线图为唯一最新总表。绝不为旧模型兼容修改当前算法/课程，也不以代码测试通过代替策略收益证据。

---

## 历史附录：Master Roadmap V11（保留原文以供版本追溯）

# FG-S3 E+Quality TP — Master Roadmap V11（WP07开发中 / WP08预设计）

**更新：2026-10-19。持续更新的唯一项目总路线图。** 本次仍使用最新已上传的 `gy22/d322` 为源码审查基线；用户报告 **WP07正在开发**，未收到WP07最新源码与实施收据。**WP08本轮只完成预设计与反向Review；未开发、未评估真实DEV/TEST。** 双仓 `integration/s3-next` 在上一归档的已验证版本：Gym `d5153268bcd33d5a4e54721767c8782f895283f9`，Dreamer `e8c2688bc817662e5e6fa7e8cee62825bcf80b00`；用户本机后续主线是否已变更需下一次查证。

## 0. 一条主线、三个角色、四级工程/研究状态

学生从零学市场（F1 Encoder/RSSM/WM）、学入场后果（F2 Quality四预测头）、学L2严格ADD/WAIT（F3 `quality_actor`）、在真实TRAIN模拟交易中受控继续学习（F4/WP07）；老师L1/L2只在TRAIN教学阶段利用未来，**正式Actor-only决策无老师接管**；教材是E+MACD V3/P0–P4/真实Quality TP多行为课程。

金融目标不变：做空方向受外部条件约束，单批3手，TP5、SL30；重点找**低MAE、快速TP、少短时逆行10点**的机会；同一官方日内完成是质量偏好，非新增强制平仓规则。

持续区分：`CODE_IMPLEMENTED`（代码存在）/ `CODE_TESTED`（实际测试及退出码）/ `DATA_ADMITTED`（真实训练数据与来源准入）/ `RESEARCH_EFFECTIVE`（独立考试证明有效）。前者绝不能自动替代后者。

## 1. 全局大功能、小任务、状态、依赖

| 功能包 | 核心目标 | 小任务/验收口径 | 当前状态（V11） | 依赖/下游 |
|---|---|---|---|---|
| WP01 | 统一课程/来源和关键保护 | 奖励、E Observation、阶段矩阵、身份与准入 | 工程已完成；后续MACD V3身份需用实际最新版 | 全部 |
| WP02 | 原生V3教材与双教师生成 | MACD partial、P0–P4、L1/L2、真实TP/SL/MAE、Producer | 代码已完成；V3正式多合约Corpus仍在生产（用户报告）；完成量待真实收据 | WP03/DATA |
| WP03 | 教材读取与合法监督 | 原生Replay/Trace/Sidecar Join，Natural/Stratified/Mixed，精确恢复 | 工程已完成；基本Reader原先以单Corpus为主 | F1–F4 |
| WP04 | F1六组World Model更新 | Encoder/RSSM/Decoder/Reward/Continue/Avail、冻结、Reader+CP | **代码完成**；截至上次证据无正式真实F1受训权重 | F2/F3/F4 |
| WP05 | F2预测Head教育 | Event/Touch/Adverse/Score、阶段接续、四头冻结/损失 | **代码完成**；未证实已有正式F2训练权重 | F3/F4 |
| WP06 | F3入场决策训练 | L2自然/辅助/Pair、quality_actor唯一更新、Actor-only工程检查、CP | **代码完成，双仓s3-next上一归档已集成**；没有正式已受训F3与策略效果证据 | F4/WP08 |
| **WP07** | **F4受控在线** | 07.A F3来源，B SHADOW，C在线标签/混合，D online BC，E同`quality_actor` imag RL，F Gate，G精确恢复，H集成 | **用户报告正在开发，尚未提交新源码**。工程可先用合成数据，正式RL/在线学习仍需独立准入 | WP08/WP10 |
| **WP08** | **Teacher-free独立考试** | **08.A冻结DEV/TEST考试合同；B F3/F4只读Actor；C多合约原生长区间；D TP时效/MAE/逆行/成本/风险；E同源对照；F统计与GO；G评价session恢复；H CLI+审计** | **本次预设计 + 源码静态 Review 完成；WP08尚未开发**。先合成/有限TRAIN调试，待WP07源码后升级正式任务书 | WP09/WP10 |
| WP09/DATA | 教材扩充、跨Corpus统一采样与消融 | DATA-V3生产/质量、DATA-MULTI F1/F2/F3 natural+aux+来源RNG/CP、合约/官方日/市场环境分层、消融 | 用户报告V3教材正在生成，DATA-MULTI已开始支持F3；尚未收到最新源码/真实V3完整产物的验收证据 | F1–F4及WP08/10 |
| WP10 | 研究身份冻结与最终验收 | 冻结checkpoint/策略/Eval规则/数据/split/baselines，封存TEST的最终一次独立考试、研究GO/NO-GO | 规划，不能冒充实盘交易授权 | 研究终局 |

## 2. WP08如何和WP07/WP09/WP10避免重复开发

```text
已完成WP01–WP06工程代码 ──────┬────────────> WP07 CODE（在开发）
                                │               ├─ SHADOW
DATA V3教材 + DATA-MULTI ───────┼────> 真正F1→F2→F3受训权重 ──┤─ onlineBC/RL
                                │               │
                                └── WP08考试框架先行开发 ──────────┘
                                                 │
                                          冻结的 DEV Actor-only考试
                                                 │
                                WP09 跨合约分层 + 合法消融 + 统计支持
                                                 │
                                WP10 冻结候选/协议 → 封存TEST最终考试
```

- **WP07** 产生/更新学生：TRAIN在线学习与探索可控，训练参数会变化。
- **WP08** 评价学生：0教师接管/0探索/0参数或归一化更新，真正`quality_actor` Argmax；原生金融账户对账，输出`ADD/WAIT, TP5时效, MAE, early adverse10, SL30, reward, netPnL, drawdown, missing/uncertain support`。零入场是`NO_ENTRY_SKILL_EVIDENCE`。
- **WP09** 使用WP08同一个Evaluator，对更多合约/年份/市场环境、F3 vs F4、RL off/on、教材抽样方式等消融。**不另造评价金融内核**。
- **WP10** 锁定最佳候选及固定统计/风控门槛，再准入TEST，不能让TEST反向指导同一最终选择。

## 3. WP08首轮8个小开发提交（预设计）

| 提交 | 具体交付 | 当前可并行? | 依赖核对 |
|---|---|---|---|
| 08.A | `EXAM_PLAN_V1`/来源/模型/分片/评估身份、DEV/TEST权限 | 是：纯合同/合成案例 | 实际DEV白名单/TEST封存由用户来源另行提供 |
| 08.B | F3/F4只读Actor、固定Argmax、无teacher或fusions | F3适配器可先做 | F4真实模型/CP接口等WP07落地 |
| 08.C | Native长区间多合约回放与会话分块 | 是：合成环境与复用旧Env原语 | WP07 SHADOW若复用同Runner须最终接口去重 |
| 08.D | 真实trade质量、风险与账户资金对账 | 是：合成/有限TRAIN数据 | 原金融Env/Clock/成交属性保持不变 |
| 08.E | AlwaysWAIT/ADD/Random/Delay，F3/F4 checkpoint对照 | 初版基线可先做 | F4等WP07真实权重；不可用Oracle P4作部署基线 |
| 08.F | 非退化、样本量、跨日/合约统计与研究判定 | 是：合成多簇数据 | 阈值在TEST前预注册，不假设DEV已经有效 |
| 08.G | Eval Session自己的确定性冷恢复、只读身份 | 是：合成环境 | 不用训练CP代替评价carry/账户恢复 |
| 08.H | CLI、结构化收据、集成/定向测试与反向Review | 部分可先做 | WP07最终源码与DEV来源确认后正式验收 |

接口静态核对：Gym `fg_s3_e_quality_actor_checks.evaluate_native()`只允许**≤512步**，`evaluate_checkpoint()`当前装的是**F3训练Corpus**，不是独立DEV考试；旧 `fg_single3_evaluation.analyze_trace()`是旧C1统计与一份固定DEV合同，不能直接给新E+Quality开放正式考试。应复用其policy/原生Driver/trace reconciler与指标原语，构建新的独立考试来源身份和长区间Session。

## 4. 并行计划与必要/可选分层

**建议并行：** A) Codex继续WP07；B) V3 Corpus生成/真实发布；C) DATA-MULTI对F1/F2/F3统一Reader；D) EVAL-BASE/WP08的合同、真实账户长跑、金融质量统计与评价冷恢复；E) 真实F1/F2/F3训练实验准备（正式长跑另行明确授权）。**不重复开发同一个模块**，共享基础设施提交核对后及时合并成对`integration/s3-next`。

**所有DEV都必须保留：**真实金融Env/TP/SL、数据因果、5m partial MACD不看未来、Actor-only无教师、合法mask、账户完整对账、源身份基本可追踪。**DEV可选/按需：**每次修改都全库测试、完整G3、完整机况盘点、科研显著性结论、GPU长跑。默认CPU/GPU/RAM空闲，需要计算则优先合理利用，不重复`free -h`、`nvidia-smi`或CPU盘点。**DATA发布：**真正的Corpus/Reader/Teacher身份及质量数据、分母收据；**FORMAL考试：**完整模型和DEV输入身份、0更新、双仓代码冻结、统一账务；**TEST：**WP10单独冻结协议和权限。

## 5. 本次交付 / 后续同步更新规则

- `FG_S3_E_Quality_WP08_Actor_Only_Exam_PreDesign_Reviewed_V1_20261019.md`（WP08 A–H预设计、45项验收、18项反向Review）。
- 本文件 `FG_S3_E_Quality_Master_Roadmap_20261019_V11.md`（唯一总路线图最新版）。

WP07完成并上传`gy23/d323`或其他实际命名的新双仓归档及实施报告后，先做WP07→WP08接口Diff，确认SHADOW/RL/F4Checkpoint/Teacherfree Collector/EVAL-BASE实情，再将WP08升级成正式Codex执行版，同时刷新实际成对SHA与Master Roadmap。**不预设新归档的文件名或其是否已集成。**

---

## 附录：Master Roadmap V10及更早历史（只读，不再作为当前状态）

# FG-S3 E+Quality TP — Master Roadmap V10（WP06已完成 / WP07正式开发）

**更新：2026-10-19。当前真实源码：gy22/d322；双仓 `integration/s3-next` 已集成 WP06。** V9及更早版本保留在本文后部作为历史记录；发生状态冲突时以V10为准。

## 0. 大方向不变

学生 F1（E Encoder/RSSM/WM+因果5m MACD）→ F2（Quality Event/Touch/Adverse/Score）→ F3（strict L2 监督的唯一 `quality_actor`）→ F4（TRAIN历史真实Env中SHADOW、online BC、受控RL）→ WP08 Actor-only考试 → WP09多合约泛化/消融 → WP10研究冻结。Quality TP Reward、3手空单WAIT/ADD、TP+5/SL−30、日内优先但不强平、低MAE和快速TP的目标不变。L1/L2后验绝不进入Actor执行时观察。

## 1. 当前大功能包 → 小任务 → 状态

| 工作包 | 大功能与已完成小任务 | 当前真实状态 | 下游 |
|---|---|---|---|
| WP01 | 合同/课程/基础Guard | 已完成、已由后续MACD V3课程演进；受信Corpus hash不可静默改 | 全部 |
| WP02 | E/MACD/多行为P0–P4、L1/L2教师、原生数据生产 | 已实现；完整当前 V3多合约训练教材**尚需实际发布证据** | WP03/DATA |
| WP03 | Reader/Join/Sidecar/Natural/Stratified/Mixed、精确Reader恢复 | 已实现；单Corpus Reader被F1/F2/F3复用 | WP04–07 |
| WP04 | F1六组WM优化器、K burn-in、FullCheckpoint | **CODE完成**，合成更新验证；**无正式真实受训F1权重** | F2/F3/F4 |
| WP05 | F2 Event/Touch/Adverse/Score +F1接续与冻结 | **CODE完成并已集成s3-next**，无正式已受训F2 | WP06/07 |
| **WP06** | **F3 Quality Actor：06.A阶段、B来源、C仅Actor更新、D L2+Pair、E Actor-only、F冷恢复** | **CODE完成且双仓s3-next已集成**：Gym `d5153268bcd33d5a4e54721767c8782f895283f9`（运行 `53ffea2`），Dreamer `e8c2688bc817662e5e6fa7e8cee62825bcf80b00`（运行 `ffe6a83`）；WP06报告96+20项CPU PASS；正式F3训练=0，GPU update未成功 | WP07/08 |
| **WP07** | **F4：07.A来源、B SHADOW、C在线标签与数据混合、D online BC、E同头imag RL、F Gate/探索、G原子CP、H集成** | **正式Codex设计完成、代码未实施**。不得直接启旧pol RL；以当前双仓集成主线新建feature | WP08/10 |
| WP08 | Actor-only真实独立考试、成本/MAE/TP时间/风险 | EVAL-BASE工具可并行研发；真实F3未训练，不可宣称真实胜率 | WP09/10 |
| WP09 / DATA | 18合约P0–P4教材扩量，DATA-MULTI跨Corpus reader/可恢复调度、覆盖和消融 | **并行进行**，当前未收到已发布完整V3多合约Corpus收据；DATA-MULTI在本归档以独立feature存在，未成为main的公共执行代码 | F1–F4/研究 |
| WP10 | 代码/模型/教师/数据/seed/场景冻结与Go-NoGo | 规划，非自动实盘部署 | — |

## 2. 不可忽略的真实代码与数据证据

- WP06完整A–F运行提交与双仓配对SHA见最新归档 `docs/WP06_COMMIT_LOCK.json`。两个公开integration/s3-next分支已指向 WP06 发布；**不必再合并WP06自身**。用户本机当前worktree是否dirty需按实际为准。
- WP06报告：96个WP06+20共享Guard选定CPU测试PASS，合成F3只更新quality_actor/；FullCheckpoint冷恢复一致；真实GPU update因cuInit失败未通过；正式F1/F2/F3权重均无。
- 完整V3训练Corpus在WP06实施报告指定路径未见；实际`f3-read-check`只是开发用极少WAIT样本。**WP07代码开发不必等待，正式训练与研究必须等真实数据和受训来源。**
- `dreamerv3/agent.py` 中 imagined action与imag policy loss仍取 `pol/`；WP07需要单一路由转到同一个quality_actor/，并验证val/slowval、WM与mask。原 `LiveRun` 对Quality V2有旧来源语义，不可直接宣称F4已就绪。
- DATA-MULTI若在Codex实际机上已更新主线，WP07应使用最新成对HEAD，避免复制公共Reader。

## 3. 项目依赖 / 并行工作（保持大功能、小任务双层）

```text
双仓 integration/s3-next [WP01..WP06 工程代码]
    ├── DATA: Corpus生成/发布 ── DATA-MULTI（跨Corpus Reader）────┐
    ├── TRAIN：正式F1 WM ── F2 Heads(可选用于F3) ── F3 Actor ─┐│
    │                                                      ├┴─> WP07
    ├── EVAL-BASE：Teacher-free独立交易考试工具 ────────────────┘     │
    └── WP07 CODE：SHADOW→online BC→quality_actor imag RL(单独Gate)  │
                                    ↓                                 │
                            WP08 独立DEV评估/冻结TEST规则             │
                                    ↓                                 │
                            WP09 消融、扩量及稳健性 ── WP10正式研究冻结
```

**并行优先级：** DATA真实V3教材路径/G3/Reader/覆盖证据，DATA-MULTI公共reader/CP集成，F1正式训练实验准备（按用户明确授权），EVAL-BASE，WP07工程开发。避免全WP编码完成却无真正受训学生。

## 4. 合并、开发、资源政策

WP内部在独立worktree按小任务提交；完成源码Review后双仓成对、定向CPU验收通过及时合并 `integration/s3-next`；DATA/EVAL独立合并共享模块，注意不覆盖WP07并行修改。训练Run引用独立不可变成对代码/Corpus与来源签名，主线推进不能改变已冻结Run。**DEV**快速定向测试，不反复跑free/gpu/cpu巡检，默认机器资源空闲、尽量利用CPU/GPU；**DATA**发布需要真实原生身份/分母/质量；**FORMAL**长期真实训练、TEST评估需明确许可。任何阶段都必须保证因果性、金融执行、Teacher隔离和来源可追溯。

## 5. 本版交付 / 下一次更新触发

- `FG_S3_WP06_to_WP07_Interface_Code_Review_gy22_d322_20261019.md`（本轮源码级差异 Review）
- `FG_S3_E_Quality_WP07_Online_Adapt_Final_Codex_gy22_d322_20261019.md`（可开发A–H）
- 本文 `FG_S3_E_Quality_Master_Roadmap_20261019_V10.md`。

WP07最新源码到手后再做真实接口验收、修订WP08，并更新V11；若 DATA-MULTI、Corpus、正式F1训练的收据先完成，单独更新对应小任务、成对SHA与训练/研究状态，不改总体方向。

---

## 附录：V9及之前的历史总路线图（未经重写）

# FG-S3 E+Quality TP — Master Roadmap V9（WP07预设计、F4在线适配与质量Actor RL）

**日期：2026-10-18。** 这是V8的增量更新；V8全文在后面保留为历史基线。**最新当前状态：WP05代码完成；WP06正在开发；WP07预设计完成，等待WP06最终源码审查；DATA生产状态以新收据为准。** 此更新不代表WP06/07实际训练已运行。

## V9：WP07大功能与小任务（2026-10-18）

**最终目标不变：** Encoder/RSSM等先学市场(F1)→Quality四头学结果(F2)→`quality_actor`学入场(F3)→真实TRAIN在线采集、监督适配和有门槛RL(F4)→Actor-only考试(WP08)→多合约泛化/消融(WP09)→研究冻结(WP10)。任何一步不得混淆学生、L1/L2老师、P0–P4教材和后验参考收益。

| 功能包 | 大功能/当前状态 | 本版新增的小任务与依赖 | 当前工程/正式结论 |
|---|---|---|---|
| WP01–03 | 课程/教材/Reader 已实现；与新MACD课程版本统一身份 | 提供F4来源合同、P0–P4后验教师和Reader | 已有小样，但不能代表训练充分 |
| WP04 | F1 WM工程完成 | 等多Corpus规模与用户实际训练授权 | 无正式受训F1权重证据 |
| WP05 | F2四Heads工程完成 | F1参数来源可用后开展真实F2训练 | CODE完成，正式F2尚未训练 |
| **WP06** | **F3 Quality Actor开发中** | 源码完成后独立接口Review；F3受训策略是WP07的最终输入 | 尚无WP06新源码与模型成绩，不预填PASS |
| **WP07** | **本轮预设计完成** | **07.A**执行合同+F3来源；**B**无教师接管采集；**C**后验L1/L2+离线/在线混合；**D**在线监督更新quality_actor；**E**真实Quality Actor想象RL与Critic；**F**Gate/合法探索；**G**多流精确CP；**H**集成消融 | 设计Review完成，执行代码`NOT_IMPLEMENTED`；正式训练`NO_GO` |
| WP08 | Actor-only独立考试 | EVAL-BASE工具可独立提前做；考试必须teacher-free且无探索 | 未有F3受训模型，不评价收益 |
| WP09/DATA | 跨合约教材+DATA-MULTI统一Reader | 在线采集与历史不可变Corpus混合必须防跨账户Pair，Reader RNG可恢复；真实新批次收据决定训练规模 | 继续并行，不能凭计划认定已生成 |
| WP10 | 研究结果冻结 | 双仓+Corpus+老师+参数+seed+实验/风控版本+GoNoGo | 后续 |

### V9关键依赖与更新规则

```text
公共 s3-next (双仓成对集成，MACD V3)
  ├─ WP06 F3受监督 Actor ──> F3实际受训权重 ──> WP07 F4
  │                                  │             ├─ SHADOW
  │                                  │             ├─ ONLINE_SUPERVISED
  │                                  │             └─ ONLINE_RL_CONTROLLED(独立门槛)
  ├─ DATA(P0–P4真实TRAIN) ─> DATA-MULTI Reader ────┘
  └─ EVAL-BASE Actor-only工具 ──────────────────> WP08
                                                ↓
                                        WP09 → WP10
```

**关键架构冲突需WP07.A解决：** 原Dreamer想象RL从`pol`取动作并更新原Actor/Critic，不能直接训练实际执行头`quality_actor`；Quality V2的旧`LiveRun`关闭RL Gate。WP07必须真实修改策略/损失路由并验收。现有 `ONLINE_ADAPT`课程声明把 Prediction/Q/Decision/RL全列为门槛，因此`ONLINE_SUPERVISED`优先作为F3有身份的online continuation，或正式版本升级F4合同；不允许无版本绕过、也不为了不使用的Q/决策预测拖慢DEV工程。

**DEV/DATA/FORMAL仍分级：** 默认资源空闲，不例行盘点free/GPU/CPU；DEV合成测试与有限TRAIN诊断直接进行；DATA最终发布有真实来源和标签正确性；FORMAL需已受训来源、完整身份、明确授权。未来信息、金融账户/Reward、教师标签、Native action mask的正确性始终为强制项。

**开发完成及时集成：** WP内小提交在独立worktree，WP审查通过后双仓成对合并`integration/s3-next`；DATA-MULTI/EVAL-BASE按真实变化单独Review合入；正式训练引用不可变双仓源码+数据快照。更新每一WP时继续更新此单一总表，不追加平行分叉的“新总计划”。

**本轮新增文件：** `FG_S3_E_Quality_WP07_Online_Adapt_PreDesign_Reviewed_V1_20261018.md`。WP06交付后重新核对F3接口、生成最终Codex任务书与V10路线图，不以预设计猜测最新接口。

---

# FG-S3 E+Quality TP — Master Roadmap V8（WP05已交付 · WP06正式开发 · 双仓集成策略）

**日期**：2026-10-18。**统一项目路线图**，继承 V1–V6 的 WP01–WP10 及并行 DATA 线；最新源码基线为 `gy21/d321` 的 WP05功能分支（Gym `61dd64b`，Dreamer `be7ce01`）；当前双仓 `integration/s3-next` 分别为 Gym `0e4af49`、Dreamer `fd14411`，已含DATA MACD V3修复，但尚未集成WP04/WP05。

## 0. 大方向不变

**学生**：从零训练 Encoder(1m/15m/当前可见partial5m MACD)、RSSM、WM Heads → Quality Event/Touch/Adverse/Score → Quality Actor；不继承旧1M模型，不维护旧Checkpoint ABI。  
**老师**：Quality V2 L1入场结果/质量/逆势风险；L2在完整TRAIN后缀中按Quality TP折扣价值比较立即ADD与WAIT。后验只能产生标签、不得污染学生当前Observation。  
**教材**：WP02生成真实P0–P4、多行为正/负/中性账户轨迹；WP03进行真实三方Join、分窗口有效遮罩和不可变Reader；DATA并行扩充跨合约跨官方日课程。  
**金融目标**：按既定空单方向，WAIT或ADD 3手，TP+5、SL−30，强调快速止盈、低MAE和规避短时逆势10点；跨官方日TP仅降低质量分，金融退出条件不变。  
**最终衡量**：Actor-only在真正隔离的考试集自主择时，优质ADD/合理WAIT、风险/收益/时间/合约稳健性；不把老师预测或P4盈利当学生成绩。

## 1. 依赖图（大功能→小任务，不在WP06结束）

```text
WP01 课程+交易/观察身份 ────────────┐
          ↓                         │
WP02 Producer、E+MACD、L1/L2/P0–P4 │
          ↓                         │
WP03 认证、Join、唯一Reader          │
          ↓                         │
WP04 F1 WM基础（工程已完成）         │
          ├─────────────┐           │
          ↓             ↓           │
WP05 F2 Quality四头    WP06 F3 Quality Actor
          └───────┬─────┘
                  ↓
WP07 F4 阶段接续、受控在线学习
                  ↓
WP08 Actor-only考试/风险/稳健性
                  ↕
WP09 多合约教材/消融/效率（DATA并行）
                  ↓
WP10 研究版冻结、证据及Go/No-Go
```

F2 Prediction Admission不是F3严格动作监督的前置条件。模型学习路径不等于机器准备路径：DATA教材生产可以与WP04–WP06工程开发并行。

## 2. 截至本版真实状态（四轴与来源区分）

| WP | 大功能 | 小任务/实际交付 | 最新状态 | 下游 |
|---|---|---|---|---|
| **01** | 课程合同 | Reward/Observation/权限矩阵、Draft/正式、Guard | 完成。Gym`2904f3e`/Dreamer`347724e`；后续WP02升课程V2。**WP05已实现独立F2四头执行白名单，旧Corpus数学身份不因权限收窄而改变** | 全部WP |
| **02** | 学生输入+教材+教师 | E+partialMACD、P0–P4、L1/L2/数据生产器、原生质量结果 | 源码完成：Gym`e57189b`/Dreamer`44a6cc6`；有限FG201真实小样。跨合约扩量另见DATA线 | WP03/09 |
| **03** | 课程认证与Reader | 可信TRAIN来源、3-way Join、NATURAL/MIXED/STRATIFIED、精确Reader RNG、教师损失ready | 完成：Gym`7366886`/Dreamer`446d1bc`；真实小样4146行、690独立市场cursor，**数据规模LIMITED** | WP04/05/06 |
| **04** | F1 完整World Model工程 | 04.A新F1；B六WM真实更新与冻结；C诊断；D Reader+FullCheckpoint；E合成/真实只读集成 | **CODE_GO/REAL_DATA_READ_GO，TRAIN_GO NO-GO**；新Gym`b7d62113f1e8d4d7ddbdce175acd83568422420e`、Dreamer`9715352d4e548b82ba24d3b31070b9fa20d345a6`；报告379个选定CPU PASS；真实TRAIN正式权重**尚无** | WP05/06 |
| **05** | F2 四Quality预测头 | A合同与阶段；B F1→F2；C四头更新/冻结；D采样/校准；E精确恢复；F集成验收 | **WP05 A–F源码已实施：Gym `61dd64b`、Dreamer `be7ce01`；选定CPU工程CODE_GO，GPU更新尝试失败；无真实F1/F2受训权重，真实TRAIN更新0**；四头Event/Touch/Adverse/Score；两个Q头、Actor、WM冻结。F2真实训练需正式F1权重 | WP07，F3可并行 |
| **06** | F3 Quality Actor | 06.A阶段与Guard、B阶段参数handoff、C仅Actor真实更新、D L2优先监督及L1一致质量排序、E Actor-only基础考试、F精确Resume/验收 | **WP05→WP06实际接口Review完成，正式Codex任务书就绪；WP06代码未实施**；F3只训练`quality_actor/`；L2优先，L1冲突Pair必须过滤。实际DEV/FORMAL训练未发生 | WP08/07 |
| **07** | 阶段衔接与受控在线 | F1/F2/F3→F4正式来源链、原RL/Gate、在线真实账户恢复、受控Trainer | 规划；F1→F2最小正确接续已划归WP05，不应重复开发 | WP08 |
| **08** | Actor-only独立考试 | **EVAL-BASE工具可提前开发**：纯Actor原生Action、Teacher隔离、真实金融Env、TP时间/SL/MAE/风险、分块汇总；正式DEV/TEST留待协议准入 | 工具可先在CPU合成/有限TRAIN实现；无已训练F3，不得声称有效结果 | WP09/10 |
| **09** | 教材扩充和消融 | DATA-A-E真实批次生产；**DATA-MULTI多Corpus统一Reader/完整恢复**；稀缺题型；性能与消融 | **用户曾报告DATA正在生成；新的成功批次收据未在本次上传附件中，不能当作PASS**。此前MACD incomplete bucket问题有V3代码修复，且当前integration已合并相应提交；DATA-MULTI可单独开发并合成验收 | 可并行WP05–08 |
| **10** | 完整研究版冻结 | 双仓/Corpus/教师/参数/seed/实验/风险与Go-NoGo | 规划，不表示实盘部署 | — |

### 已实现的WP04→WP05接口及WP06继续复用的部分

- `EQualityCourseReader.open(...stage='HEADS_VALID', sampling_spec=..., index_path=...)` + `sample(B,K+T) → [B,K+T+1]`。
- F1 `dreamerv3/fg_s3_e_quality_foundation.py::context/train/model_extensions`：因果K步预热、原World Model/Opt六前缀。
- F1 `embodied/run/fg_s3_e_quality_foundation.py::FoundationRun/CourseStream`：原FullCheckpoint外部课程Reader精确恢复。
- F2已经新增独立stage及F1→F2参数handoff；`expected_stage_contract('QUALITY_HEADS')`允许六头**有有效标签时**训练，但WP03 Q无认证目标，WP05实际使用**四头运行时白名单+Q冻结**，**没有修改参与Corpus hash的派生阶段矩阵**。WP06继续按这一机制实现仅Actor更新。不同身份、不同运行不能伪装Resume。

## 3. 学生参数学习权限（修订目标）

| 网络组 | F1/WP04 | F2/WP05 | F3/WP06 | F4/WP07 |
|---|---|---|---|---|
| E Encoder、RSSM、dec/rew/con/avail | TRAIN | 精确冻结 | 原则冻结 | 经独立准入可能放开 |
| Quality event/touch/adverse/score | 冻结 | **逐头合法mask下TRAIN** | 原则冻结 | 经独立准入可能放开 |
| Quality Q c1/v2、Soft/Q动作修正 | 冻结 | **冻结** | 冻结 | 需数值目标认证与决策准入 |
| `quality_actor/` | 冻结 | 冻结 | **TRAIN** | TRAIN |
| 原`pol/val/slowval`、Gate与RL | 冻结 | 冻结 | 冻结 | RL/Gate单独证据决定 |

WP05完成不意味着Prediction/Decision Gate开放；用户最终要的是Actor入场能力，不是漂亮的预测Loss。

## 4. 统一开发与资源使用原则（用户新增）

**默认开发机器CPU、GPU、内存完全空闲。** Codex不应重复执行`free -h`、`nvidia-smi`、`lscpu`、`top`、`ps`等例行巡检，也不要为“资源确认”占用开发回合；按任务选择CPU/GPU，允许合成批处理/并发和适度调大batch，目标尽量充分利用算力。**只有真正OOM、设备竞争、吞吐低或用户主动要求检查时才诊断**，并据实际错误针对性调整。不要在默认配置中无故将并发降到极低。对于DATA线用户已经明确的`--workers 20`预算，按该具体任务约束执行。

开发门槛 `DEV`：真实金融/因果/标签隔离必须正确；可以快迭代、直接修补并重跑定向测试、有限TRAIN只读、小合成优化器更新，不必每次全G3和重新锁定所有SHA。  
教材发布 `DATA`：真实TRAIN分批生产，最终G3/Reader/覆盖与产物身份；不能混DEV/TEST未来或把失败改标签。  
正式训练 `FORMAL`：新的完整F1来源、Corpus、模型身份和阶段授权；真实TRAIN较长优化器训练和独立研究结果另请用户授权；未经允许不自动启动长跑。  
任何阶段：**不虚构性能/测试/来源/训练成果**。

## 5. 近期执行次序与交付维护

1. **WP05已完成开发，不重复执行**。先把已完成的WP04/WP05从功能分支集成到双仓公共 `integration/s3-next`；以当前主线MACD V3为统一数据/课程语义。真实受训F1/F2参数依然不存在。
2. **DATA此前的MACD尾部故障修复代码已在当前integration**，包含MACD V3；新正式B01–B04批次是否完成须以其真实收据确认，不能以Git提交或DEV合成烟测代替。避免两个分支重复改Observation/Reader接口。
3. 如果用户希望启动正式F1训练，必须先单独确定真实课程规模、目标批次、Checkpoint位置和授权，再冻结完成后启动F2真实训练。不要因为WP05代码GO就声称学生已经受训。
4. **WP05→WP06接口Review已完成**。WP06正式任务书基于gy21/d321和当前主线真实接口；从已认证的双仓成对集成HEAD启动06.A–F，不从旧WP05分叉继续叠加；不依赖F2 Prediction Admission。
5. **EVAL-BASE并行**：在独立worktree预先开发Actor-only测试工具，CPU合成/有限TRAIN对恒WAIT、合法ADD、数据隔离、完整Env与恢复做工程验收；真实DEV/TEST研究考试仍属于WP08，不提前打开测试集。
6. 每次源码交付，在当前Master Roadmap更新**双仓HEAD、真实完工小任务、选定测试事实、未运行项、四轴GO/NO-GO、下游依赖**。旧V1–V5保留历史，不用旧状态覆盖新状态。

## 6. 版本日志

| 版本 | 变化 |
|---|---|
| V1–V3 | 确立WP01–WP10、WP03真实接口、WP04正式开发范围。 |
| V4 | WP05预设计、DATA跨合约教材提前生产计划。 |
| V5 | 独立DATA生产者与MACD故障、DEV/DATA/FORMAL规则分层。 |
| V6 | 从gy20/d320实际确认WP04双仓完成和F1未正式训练；完成WP05 A–F正式开发计划，明确F2四头执行白名单是现有课程条件许可的严格子集、避免更改`course_identity_hash`导致教材失效；禁止F1→F2冒充Resume；明确默认资源空闲、尽量跑满、避免重复硬件探针。 |
| **V7（历史版）** | WP06基于gy20/d320实际`quality_actor`/`teaching_objective`、WP03 `ACTOR_STRICT`/`ACTOR_QUALITY_PAIR`完成预设计；新增L1与L2标签冲突约束、F3仅Actor训练、真实策略Argmax门槛；并行新增DATA-MULTI多Corpus Reader、EVAL-BASE Actor-only工具、TRAIN-PLAN正式F1研究准备；DATA状态以新收据为准，不凭“正在生成”虚构成功。 |

## 7. 本版独立并行任务的开发边界

| 工作线 | 建议负责人/工作树 | 当前可执行范围 | 与主线的交付点 |
|---|---|---|---|
| WP05 | 既有WP05 worktree | **已交付F2源码和报告，工程完成，真实F1/F2训练未发生** | 双仓并入统一集成基线，保留原功能分支与运行证据 |
| DATA-QUALITY | 既有DATA生产worktree | 实际TRAIN批次生成/Reader验证/正负覆盖 | 新批次G3/Reader/coverage/身份锁，失败保留日志 |
| **DATA-MULTI** | **新增独立worktree** | 多Corpus混合批次、来源命名空间防Pair串联、跨source调度RNG与FullCheckpoint状态；先合成后真实 | F1/F2/F3统一读取API，正式跨Corpus训练准入 |
| **EVAL-BASE** | **新增独立worktree** | Actor-only合成/限定TRAIN金融回放、指标、教师隔离 | WP06/F3更新后即时独立考试；DEV/TEST由WP08控制 |
| TRAIN-PLAN | 设计/配置为主 | 真实F1训练Pilot/多合约课程计划，不自动运行 | 用户明确授权后的F1真实基础模型训练 |

资料：`FG_S3_E_Quality_WP06_Actor_Supervised_PreDesign_Review_V1_20261018.md` 与 `FG_S3_E_Quality_Parallel_Workstreams_20261018_V1.md`；它们是计划，不是新源码提交。**整个研究目标始终是高质量入场择时，不是完成多少功能包。**


## 8. V8：双仓持续集成与独立研究冻结（2026-10-18）

### 集成规则（从当前版本开始统一执行）

- 单个WP A–F小任务在同一feature branch连续提交；**WP工程完成且定向Review通过后，应及时合并到`integration/s3-next`**，不能等WP10再一次性合并。
- 涉及Gym/Dreamer的WP按**成对SHA**验收，先建一次性隔离集成worktrees，自动合并后核对语义及最少必要CPU回归；失败保持feature分支、原integration、已冻结Run不变。真实主线分支改动需由用户/Codex正式执行；本次只审阅归档。
- WP05本次归档两仓Git `merge-tree` 结果：Gym只有 `docs/FG_S3_E_Quality_Master_Roadmap_20261018_V3.md` 的add/add，Dreamer无Git冲突；**但课程Observation是MACD V2/V3语义差异**，文本合并无冲突不等于可直接正式训练。应先融合WP04+WP05与最新MACD V3，并运行课程/Reader/金融因果/优化器/CP的定向验收。
- DATA-MULTI/EVAL-BASE等并行支线各有独立PR/worktree和所有者；验收后合入统一integration。涉及Agent/Reader的共享修改须与WP06合并基线协调，防止覆盖。数据生产工件必须带自己的Producer/Corpus哈希，不能由代码合并自动重签。
- 正式训练另建不可变`run/<experiment>`源码与数据快照；**integration代码推进不自动改变正在跑的实验**。WP10负责冻结最终研究结果，不代表之前九个WP都不能合并。
- 三层规则：DEV定向合成/小数据快速迭代，DATA真实TRAIN生产后验证发布，FORMAL真实训练完整身份/Checkpoint/研究门槛。默认机器资源空闲，不重复查询CPU/GPU/RAM；出现真实运行问题再诊断。

### V8最新状态和下游关系

| 工作包/支线 | 代码 | 数据/训练 | 下一任务 |
|---|---|---|---|
| WP01–WP03 | 已实现，需以公共MACD V3重新统一身份 | 旧FG201小样仅限工程开发 | DATA规模扩充 |
| WP04 | 完成，Gym `b7d6211`/Dreamer `9715352` | 正式F1训练未发生 | 合入公共主线并准备F1 Pilot |
| **WP05** | **完成，Gym `61dd64b`/Dreamer `be7ce01`** | 归档报告281项选定CPU PASS；GPU更新失败，真实F2训练=0 | **双仓集成/消除V2与V3合同差别** |
| **WP06** | **正式Codex任务书就绪；尚未开发** | 合成学习与Actor-only工程验证先做 | 06.A–F，仅quality_actor更新 |
| WP07–08 | 规划/工具可并行 | 尚无自主受训F3策略 | 在线受控学习及真实独立考试 |
| WP09.DATA | 新Producer/MACD V3已进主线；数据量以新收据为准 | 跨Corpus调度/正式规模仍待完成 | DATA-MULTI/EVAL-BASE并行 |
| WP10 | 规划 | 无冻结研究成果 | 最终结果身份与GO/NO-GO |

### 本版新增交付文档

- `FG_S3_WP05_to_WP06_Interface_Code_Review_gy21_d321_20261018.md`
- `FG_S3_E_Quality_WP06_Actor_Supervised_Final_Codex_gy21_d321_20261018.md`

**V8版本结论：** 进入WP06工程开发前，优先完成双仓主线融合；真正受训F1模型和足量多合约教材仍是后续研究训练的独立依赖。
