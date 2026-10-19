# WP08 正式 Codex 工程任务书 — E+Quality Actor-only 独立考试 V2

**日期**：2026-10-19  
**基线**：`gy23/d323` 双仓 `integration/s3-next`：Gym `0a42553`、Dreamer `745b7e0`。WP07真实模型/金融/CP接口依据对应双仓源码与《FG_S3_WP07_to_WP08_Interface_Code_Review_gy23_d323_20261019.md》。  
**状态**：正式 CODE 开发任务；不代表已完成真实DEV模型评估，不授权TEST、GPU长训练或修改金融规则。  
**总体目标**：在当前同一个原生 Single3 Env 上，不依赖老师/Oracle/探索，加载冻结的F3或F4 `quality_actor`，连续、多合约地执行合法WAIT/ADD，并以真实TP5速度、MAE/早期逆行、SL30、Reward、净PnL和回撤为核心指标，公平比较冻结策略。

## 0. 开发前必读与不可更改的方向

- 阅读双仓 `AGENTS.md`、WP07实施报告、WP08接口Review、WP08预设计以及最新Master Roadmap；先只读核对两个 `integration/s3-next` 的HEAD/dirty/worktrees，选择**当前真实最新成对集成基线**。有比本文更新的 DATA-MULTI/EVAL-BASE成果时先做差异核对、复用，不重写已有接口。
- WP01–WP07 = 课程/教材/Reader/F1 WM/F2 Quality Heads/F3 `quality_actor`/F4在线适配；**WP08只考试，不负责训练**。WP09扩展考试范围与消融，WP10只为最终冻结TEST负责。
- 新模型无需向旧C1训练Checkpoint兼容，但继续使用原 Quality TP Reward，给定合法方向SHORT、3手、TP +5/SL -30、官方交易日、原native可见市场与 partial5m MACD，不得改变金融成交、开仓权限或教师标签。
- 教师L1/L2可以在严格隔离的TRAIN课本侧生成答案；WP08考试执行和比较不访问Teacher对象或后验结果，Oracle P4不能作为实时可部署基线。
- **开发等级：** CPU合成及有限TRAIN只读验证可直接迭代；CODE_GO定向集成测试一次到位；实际DEV需真实已认证的DEV价格/模型来源；TEST只在WP10显式封存后执行。资源默认空闲，不反复调用free/gpu/cpu检查，发生OOM才调查。

## 1. 实际源码调用链和必须复用的能力

### 1.1 策略与模型

- F3：Gym `research/fg_s3_e_quality_actor.py::create_agent()`、`fg_s3_e_quality_actor_checks.py::evaluate_native()` / `evaluate_checkpoint()`；源阶段的完整核验与参数Schema归原F3实现。
- F4：Gym `research/fg_s3_e_quality_online.py::evaluate_checkpoint()` 已经提供 `run_identity.json`、`validate_bundle(node,identity)`、`create_agent(settings,...,execution_mode=...)`、`load_exact(agent.pkl)`，但它**仅用 `_live_inputs(proof,steps)` 读取TRAIN**、最多512步，返回诊断而不是DEV考试。抽离成只读的 `load_exam_actor(kind,node,...)`，为F3/F4分辨来源，不改变原入口。
- 实际执行一律 Gym `fg_single3_evaluation.ActorPolicy` 调用 `agent.policy(...,mode='eval')`，验证动作来自 `quality_actor`；禁止 `online.OnlineRun`、旧`pol`、Teacher/Oracle、online探索与Prediction/Q fuse。需要内部Sync参数时仅在考试初始阶段执行，考试全程禁止改权重、slots/normalizer/RNG。
- 当前WP07已完成在线SHADOW/BC/RL代码与真实合成冷恢复。若存在新EVAL-BASE工作树成果，优先复用其eval session并审计，不建立第二个考试Runner。

### 1.2 原生金融执行

- Gym `research/fg_replay.py::collect_trace(...)` / `reconcile_trace(...)`、`ActorPolicy`、`E_QUALITY_PROTOCOL`、原`Observed`/Driver/Env接口为金融真源。
- WP07 `OnlineCollector`只允许`source.fold=='TRAIN'`；WP08 **不可直接将这个类用于DEV**、不可把DEV标成TRAIN。复用其中可共享的Driver和观察行编码，新增独立只读ExamSession/Collector；保持Observer schema、真实逐分钟mask及交易账户规则相同。
- 现有 `fg_s3_v2_evaluation.prepare_dev` 是固定FG409 +旧C1训练合同；**不能直接用于E/MACD V3+Quality TP**。新考试源是合法 validation row/官方Clock/E Profile 的新只读协议；训练Corpus/Sidecar不构成考试源。

### 1.3 WP07恢复与来源

- F4训练 FullCheckpoint 用于原子保存在线训练状态，Eval加载后不可对原Run写入；WP08 EvalSession独立存 `eval_plan_id / actor_identity / cursor / account / RSSM carry / baselines_rng / committed_chunks / metrics_state`，精确冷恢复但没有Optimizer动作。
- 现有 DATA-MULTI F1/F2/F3 Reader可用于训练，**考试绝不复用其训练抽样**；评估基线读取相同官方日区间的独立不可变验证市场源。

## 2. 固定考试数学与数据原则

**动作：** `a_t=argmax masked(quality_actor(obs_t,carry_t))`；执行的是原Env允许的WAIT/ADD。记录每个legal add状态的`p_add`、提出ADD、真实entry fill、主动WAIT与mask强制WAIT，各分母独立。

**金融：** SHORT每次3手；官方真实TP5/SL30由Env判定；Quality TP Reward与实际净盈亏（手续费、未实现权益）分别报告。TP发生于官方交易日内是额外质量特征，不强平。负面包括5/15分钟内逆行10点最终SL30、MAE高但最终TP，皆仅从已观察合法市场路径及真实成交推导，右删失必须显式。

**因果：**部分5m MACD使用到决策前当前分钟可见部分、不能重复提交历史EMA；1m/15m同样使用原Obs；不能因考试而读未来行情。指标计算可正常遍历已公开的完整历史离线文件，但每步特征只能调用当时可见前缀，使用未来价格的地方仅在事后评价/合法会计，不进入策略状态。

**样本：** 每合约合法连续区间从原生真实金融初态起，跨夜盘/白盘保持官方日归属；跨新合约/roll依实际native reset。一个MarketDay不能因为多基线或多分块而计成多个独立样本。同一策略持仓穿过分块需要保留账户，不得独立开新账户再拼接PnL。

**三种权限：**
- `SYNTHETIC_DEV`：合成/有限TRAIN，允许CODE验证与快速调试；不可宣称真实DEV或盈利。
- `DEV_EXAM`：独立认证的非训练validation市场、不可变来源/模型，可重复考核；严禁执行Teacher/optimizer。
- `SEALED_TEST_FINAL`：首版WP08只提供显式拒绝与接口占位，WP10冻结模型、策略、基线及一次最终TEST权限后才能开启；**不要为了首版CODE_GO强制提供真实TEST**。

## 3. 大功能/小开发提交（A–H）

### 08.A 考试合同、E+Quality DEV来源及基本准入

**目标**：新 Gym `research/fg_s3_e_quality_exam_contract.py` 与 `..._exam_source.py`（实际路径遵守AGENTS），配置协议建议 `FG_S3_E_QUALITY_EXAM_PLAN_V1`；纯合同隔离文件系统访问，底层来源适配可I/O。

**输入**：`exam_mode`、官方日/合约列表、原验证白名单+市场/Env/合约来源、E/MACD/Reward合同、选定模型Checkpoint、基线与seed、采样时间边界、成本和结果根路径。单一配置指向所需的实际E+Quality合法验证源，不附Teacher标签。

**输出**：`exam_plan.json`、`exam_plan_hash`、来源SHA矩阵、分片/账户初始化合同、`PRECHECK.json`，显式标记 `SYNTHETIC_DEV` / `DEV_EXAM`，正式DEV需要验证 `source.role=validation`、`eligible_for_training=false`、TRAIN/DEV日期不重叠、E Observation/MACD V3可用于该来源；TEST默认拒绝。

**异常**：拒绝混fold、错模型/Env/交易参数、未经支持的E Profile schema或warmup、未来价格进入Obs、伪造评估许可、非法绝对路径/软链源覆盖。训练Corpus不可作为DEV SourceAdmission替代品。

**必须回答的代码问题**：既有validation源是否有与新E/MACD V3兼容的完整可见K线、官方Clock及初始化文件？若缺则 `DEV_SOURCE_NOT_AVAILABLE`，可继续合成开发，不能把旧C1 `prepare_dev()`或训练Corpus改名冒充。

### 08.B F3/F4只读加载与统一策略接口

实现 `ExamFrozenActor.from_f3_checkpoint(...)` 与 `.from_f4_checkpoint(...)`，内部复用现有 `validate_bundle`, `create_agent`, `load_exact`, `ActorPolicy`, `execution_identity`；把模型、运行身份、来源、课程、Observation Schema、Reward、策略模式和参数摘要保存到只读 `actor_receipt`。

必须证明：同一输入下action由 `quality_actor` 的实际argmax+mask决定；修改旧`pol`参数不能改变动作；关闭teacher/prediction/q修正和探索；同一checkpoint推理重复值相同；考试从始至终模型参数、优化器/slots/normalizer不改变，且不写回原始训练Run。

对于F4 RL受训权重，正确保留 Value/slowval 作为加载身份的一部分，但它们只能被动存在，不在推理期间更新。F4不同于F3的网络配置或actor optimizer状态不可被静默忽略。

### 08.C 多合约、跨官方日的Native长区间ExamSession

必须复用已有Env/Driver/Observed与 `E_QUALITY_PROTOCOL` trace，不创建并行金融模拟器。新增 `ExamSession` 接收 `ExamPlan × FrozenActor × contract`，以原生起始账户持续执行到预先确定的区间结束。按可配置chunk逐次写入完整stepid、action/mask、fill、price/fee、Reward、account快照及SHA；账号、RSSM carry和完整市场时序在chunk间不断裂。单合约EvalSession可以恢复，多合约由外层调度，每个合约独立初始化模型carry与账户。

**必测**：超512步（例如1024/2048分钟）的真实合成连续单账户；夜盘/白盘、周末/休市与partial MACD、有仓位跨chunk、末端未平仓/roll；连续运行与split+新进程冷恢复逐行与账务相同；严禁用分段反复`collect_trace()`新初态后拼收益。

### 08.D 交易质量、风险和资金对账

从真实native account ledger / trace计算并导出：

- `Decisions`：all_market_advances、eligible_add_states、主动WAIT、forcedWAIT、ADD proposal、实际entry fill、非法ADD、mask冲突、pADD分位；分清提议/执行/成交。
- `Trades`：Entry ID/价格/时间、原生TP5/SL30/MAIN_ROLL/RIGHT_CENSORED等、持有有效市场分钟、同官方日/跨日TP、TP≤5/15/30/60、每笔netPnL/fee/reward、逐笔完整账户归属。
- `Risk`：全程MAE、前5/15分钟MAE、early adverse10首次时间、5/15分钟逆行10且SL30、最大浮亏/underwater时间及valid/censor，不因收益结果忽略长时间逆势。
- `Accounting`：每官方日账户权益、总Reward、净已实现/未实现盈亏、手续费、最大权益回撤，和原生account ledger精确对账。Reward与PnL不同量纲，不允许混算或重复减fee。
- `Coverage`：按contract×official_day×night/day统计独立时间、有效ADD、实际成交与训练无关状态；多账户/基线不当作重复独立行情。

无交易 -> trade TP率N/A，明确`NO_ENTRY_SKILL_EVIDENCE`，不能因为0回撤而PASS；样本太少 -> `INSUFFICIENT_TRADE_SUPPORT`/`UNDERPOWERED`。

### 08.E 公平的同源基线与F3/F4比较

支持：`ALWAYS_WAIT`、`ALWAYS_ADD`（遵守native mask）、`RANDOM_P10`（多个预注册seed）、`DELAY_15`（只用可见历史和有效市场分钟）、`F3_FROZEN`、`F4_SHADOW/BC/RL`有效来源模型。

每个策略独立模拟账户但使用相同市场、官方日、成本、起始资金、方向和权属；随机种子在生成前固定，不允许择优展示。对照按合约×日分块报告paired deltas，交易密度差异、权益回撤、TP速度/MAE和样本支持一起报告。不允许调用后验P4作为实时账户策略。

### 08.F 稳健性与独立研究判定

工程通过必须：合法mask无违规、Teacher/探索/训练0使用、参数不变、native账户全对账、恢复一致、指标分母正确。

研究证据单独分：`VALID_DEV_EVIDENCE`（DEV身份与回放可信）、`ENTRY_SKILL_EVIDENCE`（主动有效ADD，不是AllWAIT）、`QUALITY_SIGNAL`（TP速度、MAE/逆行、SL、净收益/回撤联合）、`GENERALIZATION_SUPPORT`（多个独立合约/官方日/时间段）。无足够样本标 `INCONCLUSIVE/UNDERPOWERED`，不要人为预设>51%就是合格。统计可采用按合约与官方日分组的block bootstrap；首版至少输出block指标与支持数量，复杂生存分析、可视化可选。

阈值在首次正式封存TEST前单独预注册；不得为了当前DEV结果调整同一TEST判定。

### 08.G 评价Session冷恢复与不可变发布

提供独立EvalSession state，包含exam plan hash、冻结actor checkpoint hash、账户/Env/Driver、RSSM carry、可见MACD构建态、随机基线状态、last committed cursor、chunk/hash列表、指标聚合、日志已提交位置。它不复用训练optimizer FullCheckpoint，也不修改任何TRAIN Run。先写chunk后原子提交状态，重启时严格验证latest、原始输入与模型未变；损坏块拒绝/留证，不静默回退；N vs N1+全新进程N2绝对一致。

### 08.H CLI / 报告 / 回归 / 反向Review

在 Gym `tools/fg_s3_v2_experiment.py` 增加 `exam-plan`, `exam-run`, `exam-resume`, `exam-verify`, `exam-compare`, `exam-report`（最终CLI名称在开工审查确认，避免与已有命令冲突），薄入口调用已有研究和运行模块。目录新建且不可覆盖，输出：`EXAM_RECEIPT.json`, `PLAN.json`, `MODEL_READONLY_RECEIPT.json`, `DECISIONS`, `TRADES`, `DAILY`, `CONTRACT`, `RISK`, `EQUITY`, `COMPARISONS`, `EXAM_REPORT.md`, `SOURCE_CODE_IDENTITY.json`, `WP08_IMPLEMENTATION_REVIEW.md`。

提交源/模型/Env/课程/时间/费用/分母/失效原因/真实命令和退出码；不能声称TEST或真实盈利。按本任务书45项行为做可追溯的测试映射，而不是形式化必须45个函数；复用WP07已有测试，不要复制一遍。

## 4. 45项测试行为验收矩阵（按功能分组）

| 组 | 必须验证的行为 |
|---|---|
| A:01–08 | 合法SYNTH、非法fold/TEST、训练Corpus不能冒DEV、E/MACD V3/Reward/Contract身份、合法官方日、费用与账户一致、文件伪造拒绝、相同plan哈希稳定 |
| B:09–15 | quality_actor真实执行、Teacher访问失败探针、无探索/Fusion、mask/argmax对账、hidden carry正确、旧pol变动不影响决策、模型及normalizer/slots全部不可变 |
| C:16–23 | 原生TP、SL、跨夜、真实roll/持仓、右删失、partial5m因果、跨合约隔离、超过512步连续=分块/恢复 |
| D:24–32 | 真正entry分母、官方日TP、TP5/15/30/60、MAE5/15/全程、早期10逆行后SL、underwater、净PnL/fee与Reward对账、RIGHT_CENSORED/MAIN_ROLL、主动/强制WAIT分离 |
| E/F:33–41 | P0=0交易非GO、P1真实mask、Random多seed复现、Delay无未来、同源F3/F4公平、P4不为实时baseline、少样本UNDERPOWERED、合法可入时AllWAIT NO_ENTRY_SKILL_EVIDENCE、市场非独立重复数校正 |
| G/H:42–45 | 全新进程EvalSession精确恢复、损坏chunk/来源身份拒绝、WP06短诊断+WP07 A-H无回归、CLI/收据0训练副作用+不可覆盖 |

可以把多个行为映射到一个真实测试，但每项都要有结果/argv/退出码。失败必须修复根因，不准忽略失败或改小断言让所有测试绿。

## 5. 开发阶段、DATA、DEV研究、TEST的必要/可选区分

| 功能 | 必要 | 可选/延期 |
|---|---|---|
| DEV迭代 | 原生金融账户、Teacher-free、无未来、5m MACD、mask、读只读模型、对应定向回归 | 每个小改动完整G3/全项目测试、GPU长跑、例行资源盘点 |
| WP08 CODE_GO | 上述A–H的完整合成/有限TRAIN工程链、>512步、原生重播、可复核报告、一次真冷恢复 | 真实已受训模型尚不可得时的盈利结论、大规模统计显著性、图表仪表盘 |
| DEV研究考试 | 真实合法DEV与真实受训冻结F3/F4模型、同源公平baseline、真实金融归因 | 可按数据规模补block CI，不足则UNDERPOWERED |
| TEST最终封存 | WP10明示批准并冻结单模型/阈值/seed/基线与不可变数据、一次封存考试 | 在WP08第一版跑TEST或围绕TEST选择参数，明确禁止 |

开发机默认CPU/GPU/RAM空闲，测试和多策略回放合理并发，不重复`free -h`或`nvidia-smi`；当出错、OOM或并发损害账户确定性才诊断。并行只跨独立账户/策略/合约，单账户时间推进保持顺序。

## 6. 反向 Review 与GO/NO-GO

**必须修正或明确拒绝：**
1. 以WP07 `online-evaluate`替代WP08真实DEV，因TRAIN与512步上限 NO-GO。
2. DEV环境冒充TRAIN以绕过OnlineCollector检查，NO-GO。
3. 借用OnlineRun或在线Teacher发布路径做考试，NO-GO。
4. F3/F4模型来源身份混淆，NO-GO。
5. 拆chunk时重建账户/重置RSSM却拼接收益，NO-GO。
6. MACD部分5m用未来完整K线，NO-GO。
7. 把Oracle L1/L2候选盈利当真实成交，NO-GO。
8. TP成功率删掉STOP/roll/右删失，NO-GO。
9. 全WAIT或仅几笔高收益就宣布策略有效，NO-GO。
10. 开发测试工具没有真实已训练模型时停工，错误；应输出正式DEV NOT_RUN。
11. 每次开发都强制formal TRAIN/G3/资源巡检，错误；分级处理。
12. 正式TEST参与DEV反馈迭代，NO-GO。

**验收分级：**
- `WP08_CODE_GO`：所有A–H工程能力合成/有限TRAIN有效、定向回归+恢复、输出收据，策略结果`NOT_RUN`即可。
- `WP08_DEV_EVIDENCE_GO`：受训模型+真实独立DEV源+源与账户对账+预先声明分母/对照，成绩可以差。
- `STRATEGY_QUALITY_GO`：独立考核显示持续非退化入场、可接受TP速度/MAE/SL/成本收益/多合约稳定性（规则需预注册）；不足写 `INCONCLUSIVE`。
- `SEALED_TEST_READY`：WP10单独审核，WP08不自动授予。

## 7. Codex 施工顺序与交付物

- 独立 WP08 双仓worktrees，WP内提交 `08.A`…`08.H`；可按实际改动归属其中一仓或两仓，不强制制造空提交。完成后经独立Code Review + 定向CPU回归，双仓成对合并 `integration/s3-next`，不要自行改写正在运行的训练Run。
- 优先顺序 **A → B → C（先真实合法Session）→ D → G（恢复）→ E/F → H**；可单独并行指标开发，但不引入另一套训练/金融系统。
- 交付8个小任务各自提交SHA/路径、真实测试命令+退出码、合成与真实数据状态、账户reconcile/模型不可变、>512连续性+冷恢复、负例、`WP08_IMPLEMENTATION_REVIEW.md`、更新 Master Roadmap。
- 若WP07真实F3/F4训练权重、多个正式V3 Corpus、真实DEV源尚不可得，标 `NOT_RUN` 而不是放宽准入；不阻塞 A–H CODE 实现。
