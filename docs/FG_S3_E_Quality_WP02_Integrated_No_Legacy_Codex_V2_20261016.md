# FG-S3 E+Quality — WP02 完整教材与原生 MACD 直接接入：Codex 任务书 V2

**日期：2026-10-16**  
**状态：设计完成，待用户确认后交 Codex；未执行源码变更、训练或数据生产。**  
**基线：** `gy17.tar.bz2` Gym `integration/s3-next`=`713a70f37a6dc1d0800cc9c9cf239752848d5b75`，`d317.tar.bz2` Dreamer=`6114b557e5cc5a134a23ae5b446b351e4f95cc0a`。WP01 未合并提交：Gym `2904f3e58cd84fb383f9e5a0d2d2aa643fcb10d2`，Dreamer `347724e3207fca16907facab021302217f47cf6b`。Codex 必须以用户当前真实工作树 HEAD 为准重新核查，旧归档仅作为本次设计源代码证据。

## 0. 用户最新决定——高优先级、覆盖 WP02 V1 的相关条款

1. **这是新架构的正常开发，不需要旧模型/旧 Observation/旧 Replay/旧 Trainer 的向后兼容。** 所有学生参数将从零重新训练，不读取、不迁移旧 E+Tutor 1M 权重。不得为了兼容旧 Checkpoint 阻碍新输入的原生实现。旧实验产物只读留档；不以兼容为验收条件。
2. **一步到位指 WP02 最终交付物必须可直接给新 E 学生使用，不是先做 Sidecar 等 WP04 再接。** 5m MACD 在 WP02 就进入 Gym Native Env 的 Observation、Replay 和 Dreamer E Encoder；包括模式、shape、mask、source identity、恢复、因果测试。
3. **WP02 要形成完整且可用的教材，而不是永远 P4_PENDING。** 本任务书把 Quality TP 条件下的严格教师 ADD/WAIT 后缀评价与 P4 轨迹列入 WP02 的最终交付；原路线图 WP03 不再重复编写它，只在 WP03 做教师标签准入、训练 Join、Soft/Q 教学与效果验证。开发仍可拆提交，但 WP02 总验收不接受空壳 P4。
4. **统一 Quality TP Reward** `FG_SINGLE3_REWARD_MARKET_MINUTE_TP_QUALITY_V1`，C1，WAIT=0/ADD=1，单批3手，TP+5 / SL−30。官方同交易日（日夜盘连续）完成盈利质量高；隔日完成质量低；**不强制日内平仓**，不改撮合、手续费、仓位、止损、Clock、Mask 和 Reward 数学公式。
5. 入场质量老师与最终动作老师分离：前者在合法状态给出候选 ADD 的真实后验路径、Quality Score/时间/风险事实，可出现多个好入场点；后者依据 Quality TP 的完整未来后缀、等待的机会成本和**真实账户合法性**判断当前执行 ADD 还是 WAIT。二者后验信息只在 TRAIN 教师域；永不进入学生的因果 Observation。
6. MACD(12,26,9) 为辅助特征，**老师不受 MACD 交易规则约束**；暂不开发箱体识别/突破分类器和 MACD 策略规则。5m 未完成桶必须用决策时刻可见的 1m 部分计算临时 MACD；不得重复提交 EMA 或读取未来 K 线。
7. 首版教材不强制正负比例、不人为造 SL、不为样本漂亮篡改后验、不追求完美指标；但是**正确性、完整输出契约、全链真实复现、性能边界和严格教师能力一次设计完整**。

**解释“无需兼容”的边界：** 可以主动替换当前新项目的 E Observation ABI/课程合同/Replay Schema 和模型初始化；但新模型内部仍必须支持同版本完整 Checkpoint 冷恢复。金融规则、源数据可信性、未来不可见、旧训练资产保护与真实 TEST 隔离仍为强制条件。这些是正确性和可复现性，不是旧模型兼容负担。

## 1. 已有源码与必须修改的真实入口

| 源代码 | 当前真实实现 | 必做改动（只做一条正式路径） |
|---|---|---|
| WP01 `dreamerv3/fg_s3_e_quality_course_contract.py` | 固定 `FG_S3_E_QUALITY_COURSE_V1`、旧 E V1 四个 Policy 字段、P0–P4 声明、四阶段矩阵，全状态 CONTRACT_ONLY | 升级为 **COURSE_V2**，规定新5字段 Observation、MACD算法/scale/partial语义、完整P4 Teacher来源、身份；移除旧 V1 兼容承诺，更新对应测试与示例。不得沿用旧 V1 hash。 |
| Gym `research/fg_s3_e_observation.py` | E 原生 `[60,18]` + `[24,7]`，`ENCODER_KEYS` 只有4项，`e_array_hash` 只有2个市场数组 | 直接发布 **E_OBSERVATION_V2 / E_REPLAY_V2**，加入 `market_macd_5m:[4] float32`，同时将新数组纳入验证、哈希、Policy 白名单与 Provenance；整套 Snapshot 一次生产，禁止 MACD 外挂造成对齐漂移。 |
| Gym `envs/entry_env.py::_get_obs` | `e_profile` 原生分支构造 E 两数组，ObsSpace 严格匹配 | 将 `market_macd_5m` 真正加入 ObsSpace 与 `_get_obs()`，与同一当前 cutoff 构造和验证；通过 `export_state/import_state` 或确定性的市场时间重新计算方式保证恢复等价。 |
| Dreamer `dreamerv3/fg_s3_e_encoder.py::S3EMultiscaleEncoder` | 严格要求四字段，1m 子Encoder+15m分支+Fusion | 新版严格要求5字段；MACD 有效位 gating、独立小投影分支，再与1m、15m融合同步训练。训练使用全新网络；不编写旧参数 shape 迁移。Decoder仍只重建必要原生目标，变化需显式记录。 |
| Gym `research/fg_single3_replay.py`, `fg_replay.py` | P0–P3 策略/collector，公开 `specs()` 不接收 Quality P4；collector 限旧协议 | 复用原 Driver/Env/Replay/Trace/reconcile；适配新 E+Quality 正式 Corpus 协议；P0–P4 五类全部真实可生成，并逐决策对账。不得再造平行金融模拟器或独立 Replay 格式。 |
| Gym `research/fg_s3_quality_score.py` / `fg_s3_quality_teacher_v2.py` | 原生0–100 Quality Score、TP/SL/MAE/窗口事实；Quality Teacher 当前对 E Profile 的安全认证未建立 | 经隔离的 Native E Snapshot/Fork/Reconcile 路径生成事实；验证通过后开放 E Quality 事实采集。不通过就 fail closed，不能粗暴删除 NOT_VERIFIED。 |
| Gym `research/fg_single3_oracle.py` | `CandidateIndex/optimize` 按旧 C1 Candidate Reward；不能冒充 Quality TP | 重用动态规划/候选索引的结构，在同一金融合同下实现 **Quality TP strict suffix** 的候选价值计算、未来 WAIT 价值比较和 P4 Schedule；必须由真实 Native Env 慢路径抽样证明结果等价。 |
| Gym `fg_s3_quality_catalog_v2.py` 及相关 Admission | 有 Soft/Q/Label 有效性与准入体系 | 让新教材暴露合法 sidecar 事实与 action-valid 标签；不能把未认证 Q/Prediction/Decision 权限绕过。WP02 不启动这些模块训练。 |

**版本规则：** 由于当前 V1 的源码检查是 exact equality，旧课程与 Obs 字段不可能“默默加 MACD”；必须更换课程+Obs+Replay身份，并更新原WP01匹配的测试，不保留对旧模型可执行的双schema适配。保留旧证据记录供历史复核即可。

## 2. 新 E-native 学生输入：从现在开始就可消费 5m MACD

### 2.1 Policy Observation（目标）

```text
market_seq_e_shared      float32 [60,18]   现有共享1m特征
market_context_15m       float32 [24,7]    现有15m上下文
market_macd_5m           float32 [4]       (normalized_dif, normalized_signal, normalized_hist, valid)
position_state           float32 [9]
action_mask              float32 [2]
```

`market_macd_5m` 是**真实 Policy Observation 字段**，而不是 Teacher/Host/Sidecar。`macd_raw_dif/signal/hist`、EMA提交状态、原始桶/决策时间/available_at放在 HOST 诊断元数据，不提供给 Actor。字段仅允许当前可见值，不允许未来结果、Oracle、Quality Score、TP/SL 标签混入。

### 2.2 指标数学定义与标准化

- 完成桶以 5m 可见 Close 更新 `EMA12`, `EMA26`，`DIF=EMA12-EMA26`，`Signal=EMA9(DIF)`，`Histogram=DIF-Signal`；**不乘2**。EMA alpha=2/(period+1)。EMA的首值、最短预热长度、缺bar行为、交易会话5m桶锚点为固定算法参数并进入哈希。
- 首版采用固定、可直接跨合约解释的**可见价格锚标准化**，避免旧计划的原始价格幅度直接混入不同期货合约：`student_value = 1000 * raw_macd_value / anchor_close`，`anchor_close` 为当前决策已可见的1m Close，必须有限且正值；valid=0时三个数值全0。原始raw值仍保留HOST审计。不得用 TRAIN/DEV/TEST 全局均值或者未来分布拟合scale；此缩放公式作为确定性 V2 特征合同，通过有限 TRAIN 小样的分布审计确认范围，超界不得悄悄 clip。
- `valid` 冷启动仅在 **≥35个此前完整、已可见的5m桶**后取1；没有足够历史时为0。有效位只证明输入可计算且预热足够，不声称指标可预测盈利。
- 部分桶：第1–4根已可见完整1mK线，以该前缀最新收盘价从**上一已提交5m EMA状态**临时演算当前MACD，不更新内部EMA；第5根完整可见才**提交一次**。重复查询输出相同位模式/Hash，无双重递推。若真实市场时钟规定当前时刻不同，按截止可见数据而非人为用未来“第5根”判断。
- 5m桶边界按冻结的交易会话及 MarketClock 定义；夜盘、跨午夜、午休、停盘或缺失分钟不得从自然墙时钟简单连续拼桶。`available_at>visible_cutoff` 的源行不可读；缺失/乱序/重复分钟按已签的失败/无效规则处理，不能前填后补造行情。
- 由新正式 E Encoder 单独把 `[4]` 投影为小向量，再与原1m+15m融合；禁止把MACD当成未经处理的未来标签导入。`valid=0`必须遮蔽全部MACD分支效果，不能因bias产生虚假市场信号。

**必须证明：** 新构建 Agent 能训练新参数，Runtime Env、Replay、Encoder三端 shape/dtype/字段完全一致，真实Actor-only和模型保存/恢复能读取同一MACD字节。WP02不要求更新训练参数或完成GPU训练，但必须用小型CPU/synthetic Agent forward 验证通路真实可用，而不是只在ObsSpace出现一个字段。

## 3. 教材：入场质量、最终动作、真实多行为轨迹

### 3.1 两个老师（同时产生可追溯但不同的标签）

**L1 EntryQualityTeacher**：对每个符合边界的合法空仓候选ADD，依据同一可信TRAIN真实金融路径输出：实际TP5/SL30/OTHER/RIGHT_CENSORED、TP/SL耗时（市场分钟）、原生Quality Score、官方交易日跨越、同日TP、最大逆势MAE、MAE(5/15/60/240)、首次≥10点逆势分钟、浮亏有效分钟总数与最长连续段、最终手续费及净值影响、label有效性与删失原因。一个片段内可能有多个合格正例；最终TP但早期逆行>10属于高风险盈利教材，不改写为SL。

**L2 FinalActionTeacher**：对于当前**真实账户空仓且ActionMask同时允许WAIT和ADD**的位置，完整比较现在ADD的Quality TP价值与随后最佳WAIT后再ADD的价值；保留 `action_value_add`, `action_value_wait`, `margin`, `strict_action`, `tie`, `teacher_valid`, `suffix_boundary_identity`。遇到成本/账户差异必须以真实金融状态为准。持仓强制WAIT、终止状态、未知suffix或源结尾右删失不能成为主动WAIT标签。

- 明确区分 Quality Score（时间/跨日）与 MAE 风险画像，**不重写正式 Quality TP Reward**，也不要未经定义便用 MAE 改变 strict 优化目标。
- 质量老师允许多个好ADD事实；严格最终动作老师考虑放弃未来机会的成本。两种标签不可互相覆盖。
- 后验标签只在 TRAIN sidecar；不允许进入学生 Obs/Encoder。保持 stepid/cursor/source/account identity 关联可复查。

### 3.2 五种策略一个不少

| ID | 行为 | 结果要求 |
|---|---|---|
| P0 | Always WAIT | 区分合法自愿WAIT、持仓forced WAIT及终止/invalid |
| P1 | Always ADD | 每个合法空仓机会ADD，按原生Env持有到实际退出；保留SL和浮亏 |
| P2 | Random 0.1 / 0.5（示例，可配置） | 同seed复现、合法Mask驱动，不看未来 |
| P3 | Delay（例如15有效市场分钟） | 按真实可交易分钟计时，重置/结算后规则明确 |
| P4 | **Quality TP Strict Oracle Schedule** | 完整的原生Quality目标后缀优化、认证动作序列、真实账户轨迹，不是旧C1或“见到TP就ADD” |

原P0–P3使用同样的本期 E+Quality Obs/Reward；P4后验策略只准TRAIN教材。五策略彼此完整隔离账户和来源，不能把同一市场时刻不同模拟账户合成一次真实交易。

**P4 的实施原则：** 优先复用已有 C1 Index/optimize 的思想与 Native金融Env，不要手工写一个第二套费用/撮合器。将 Quality Reward 的候选评价单独实现、版本化、索引加速，但必须在小样上以真实慢速Env全路径验证：ADD成交/退出、Reward累计、费用、仓位、日终、等待、账户状态；递推边界不能把人为截取的第5000步伪装成数据结束。对无法在TRAIN边界内完整观察到的结局标记删失，不能越界进入DEV/TEST补标签。遇不可认证状态fail closed，未通过验证不得签发P4合格教材。

### 3.3 教材负例与不确定性

- 正：TP5，速度越快、官方同日完成越优；保留MAE作独立风险特征。
- 负：真实SL30，重点5/15有效市场分钟内首次逆势≥10。不能因为触到≥10就冒充SL30。
- 困难中间例：大幅逆势后仍TP、慢TP、跨官方交易日、OTHER、CENSORED、持续WAIT、强制WAIT；正确输出label valid/censor reason。
- 不设虚假的正负比例/固定阈值“必须出现N个SL”才通过；如自然语料SL稀少，保留完整分布并在报告提示。数据采样权重与真实自然频率分开，不能影响金融结果。

## 4. 完整教材生产与来源合同

**唯一真实原生路径：** 签名TRAIN行情/官方交易日/E Profile → Native E Env (Quality TP) → 统一MACD因果Snapshot → 五策略驱动Env → 原生Replay + 账户/行为Trace → 真实金融重播 → L1候选事实 + L2后缀教师标签 → 覆盖审计与不可变Corpus收据。

必须分别保存：

- `student_replay/`：真实可见Obs（包括MACD）、Position、Mask、action、当前Reward、MarketClock、有效位和stepid，**没有未来**。
- `teacher_sidecar/`：后验outcome/Quality Score/MAE/risk、L2的严格ADD/WAIT、有效位、来源hash及suffix证明。任何模型Reader导入教师字段到policy需要fail closed。
- `source_manifest` 与 `corpus_manifest`：原行情内容SHA、官方交易日来源、E profile/obs/reward/MACD算法和源码身份、行为seed、市场时钟、teacher目标、版本、金融账户起始资金、训练折叠边界、各Trace哈希与操作收据。沿用现有fingerprint/signed工具，不新造签名体系。

三种口径必须分开：**唯一市场cursor/合同交易日**、**不同账户实际决策状态数**、**重复优化器曝光（未来训练产生）**。P4标签和反事实候选不得把同一真实行情多账户路径当作独立统计显著性样本。

### 样本审计（有分母、有效位及原始单位）

- P0–P4分别：合法ADD机会、实际成交ADD、主动/强制WAIT、TP5、SL30、OTHER、右删失、唯一有效市场分钟、实际账户轨迹、完整退出率。
- TP分钟P50/P90、同官方日TP/跨日TP、MAE全程和指定前缀、逆势≥10的首次发生时间与5/15分钟比例、真实浮亏累计与最长连续段、Quality Score分布。
- 每品种/官方交易日/行为/关键行情段分布，MACD有效槽比、数值范围、源文件字节、Replay体积、CPU候选路径吞吐与峰值RSS；先用有限认证TRAIN小样估算扩展成本。

**真实资源限制：** 只运行CPU合成测试、有限经认证TRAIN真实小样。全18合约Corpus生产、任何GPU训练、W&B和TEST行情读取都需要后续单独授权。没有真实本机TRAIN源时，`REAL_TRAIN=NOT_RUN`并阻断“真实Corpus验收”，不可用合成行情冒充。

## 5. Codex开发提交与验收计划

拆提交是便于开发和定位，不是留功能半成品：最后一个提交必须使 WP02 完整达到下述目标。

1. **Commit A：课程 V2 / E Observation V2 + Gym Native partial MACD5m**。升级WP01合同、训练折叠身份和E原生Env；因果流、官方会话、重复查询、EMA提交、错误数据和未来可见性负例。不要建另一个平行市场聚合系统。
2. **Commit B：Dreamer E Encoder 真接入 MACD + Replay schema/cold restore**。更新严格输入键/shape，MACD投影及valid-gating，native→Replay→Encoder CPU forward链和冷恢复确定性证明。参数从零训练；不实现旧WM迁移或旧Checkpoint适配器。
3. **Commit C：E-native P0–P3 多行为轨迹 + L1质量候选事实**。原始金融Env、交易账户重播、E Snapshot、Source身份和风险/日内事实完整一致；不自造结果；样本覆盖报告。
4. **Commit D：Quality TP 严格后缀 / L2最终动作老师 / P4**。旧C1算法只复用经过改造的结构，原生Quality目标重新计算，重播若干完整轨迹和边界，严格正负动作标签通过独立CPU核对。
5. **Commit E：Corpus CLI、认证小样、质量报告、独立Review**。一个只读/受限写入工具提供`validate`, `generate-small`, `verify`, `coverage`；输入/输出分离，不覆盖旧Run、Prepared、Checkpoint。生成可复现的Manifest/证据，运行所有新测试及必要的Shared金融/Clock/Quality回归。

## 6. 必须通过的自动化测试（最少25组）

- **课程与输入**：课程V1身份不得冒充新V2；V2五字段、types、Reward、Clock、TP/SL/Mask/品种/正式来源校验；MACD数组字节进入真实`e_array_hash`；拒绝sidecar字段混入policy。
- **5m时钟**：未完成桶第1/2/3/4根前缀，完整桶只提交一次；更改任何未来OHLC/到达时间不改变较早输出；重复查询稳定；batch/stream一致；夜盘、跨午夜、午休、跨官方日、节假日、缺/重bar；冷启动valid=0；无非有限输出和未来桶引用。
- **模型**：真实E Agent构造+CPU forward，MACD有效与无效都正常；valid=0时改变数值通道无影响；合法MACD扰动能影响模型输出，证明没有被静默丢弃；Replay实际Shape与模型完全一致；模型参数/优化器完整Checkpoint保存、加载、精确resume的合成测试。
- **行为**：P0/P1/P2(2 seeds)/P3/P4分别真实账户轨迹；Mask非法ADD绝不成交；forced WAIT不能当主动负例；价格撮合、费用、Reward、TP/SL、日终一致；chunk reset后持仓/EMA状态连续一致。
- **事实/老师**：真实TP且同日/跨日，真实SL，早期逆势10但最后TP；MAE与underwater定义独立；右删失/OTHER有效位；L1允许多优质点；L2区分当前ADD和等待价值；P4完整后缀以slow Native Env核验（不能只验证生成了schedule）；数据末端不穿越TRAIN边界或把不完整未来标WAIT。
- **来源/攻击负例**：伪造来源证书、损坏source字节、非法DEV/TEST fold、路径穿越、未认证时间、未来label进入Replay、账本改写、不同账户混状态、课程/hash不匹配、旧C1误标Quality、旧V1 obs伪装新版全部应fail-closed。
- **Corpus审计**：同市场多轨迹三分母独立；coverage可复算；哈希确定性；重放与生成一致；小样真实性；无GPU训练和正式入口被意外放行。

**别再要求跑“旧E/Tutor/B2模型能读取新输入”之类兼容测试。** 但是现有共享金融Env、Quality Reward及真实Source/Time合同的回归仍是必须；**新增 E+Quality 公共训练入口仍保持关闭**，直到WP04–WP07分别建立课程训练与认证。

## 7. WP02 的 GO / NO-GO

**G1 功能GO**：V2课程、E Env、Replay、Dreamer Encoder全链可消费MACD；P0–P4及L1/L2均完成真实轨迹与事实/标签生成；完整Native重播和后缀等价有证据。任何一段只是sidecar、shape占位、P4 pending或伪造L2，均NO-GO。

**G2 教材审计GO**：正负中间/删失结果与风险画像完整，能够量化教材偏向与弱项；不因自然SL少而编造数据。随机性、身份、指标时钟、内存成本有真实记录。

**G3 有限真实TRAIN可信GO**：已认证有限TRAIN市场数据、官方交易日、Quality Reward、MACD时序、账户轨迹、Origin/Target身份、严格P4后缀与真实市场回放、Corpus Manifest全部通过。如果本机真实数据或必要认证缺失，G3=`NOT_RUN/NO-GO`，不能以G1/G2推断真实生产可靠。

**训练权限仍 NO-GO**：WP02的成功是教材与新学生输入的工程能力，不是World Model/Quality Actor已经训练成功。下一阶段才执行 FOUNDATION_WM、HEADS、ACTOR 的训练器与训练准入；保持现有 E+Quality公开训练Guard，除非另有明确阶段任务授权。

## 8. 反向 Review（执行前必须接受的限制）

| 反面问题 | 处理结果 |
|---|---|
| “一步到位”会不会变成无休止算法项目？ | 不做箱体识别、形态分类、MACD规则交易、指标组合搜索；只做真实Quality TP事实+完整P4，与研究目标直接相关。 |
| MACD只生成却不进学生？ | 否。WP02必须把MACD接进真实Gym Obs/Replay/Encoder并通过CPU forward，取消WP04才接的设计。 |
| 改Observation为何不做旧ABI适配？ | 全新模型从零训练。直接升版本、更新课程与测试，不为历史Checkpoint兼容投入额外工作。 |
| WP01已完成会不会白做？ | 不会，复用其纯合同、只读来源审核和课程矩阵；把V1输入声明升级为V2，不复制第二套课程框架。 |
| 是否可以把原C1 Oracle P4直接用来省时？ | 不行。旧C1与Quality TP目标不同；必须在真实金融执行和同一Quality函数下重新核对。 |
| Quality Score是否要加MAE惩罚？ | 不改BaseScore/Reward。把MAE/早期逆势独立保留，后续训练损失可显式组合。 |
| 严格后缀动态规划是否一定能覆盖所有账户情形？ | 不一定。只能在已认证的Single3合法空仓状态和真实账户/资金条件下输出strict标签；无法认证就invalid，不能猜。 |
| 若当前真实1M模型训练仍在运行怎么办？ | 开发工作树隔离，旧Run绝对只读；不能动运行中Worktree或旧数据。 |
| 不兼容意味着可以不测金融代码吗？ | 不可以。金融时钟、撮合、Reward及同版本恢复是正确性，与向后兼容无关。 |
| 研发阶段没有真实TRAIN源怎么办？ | 合成测试工程结果单列；真实 G3 必须NOT_RUN，不能冒充。 |

## 9. 可直接交给 Codex 的执行指令

> 阅读本 V2 任务书、WP01实际提交及其报告、`gy17/d317`最新双仓 AGENTS.md。在真实 `integration/s3-next` 基线上创建独立WP02 worktrees，先记录当前HEAD/dirty/worktree及WP01是否已集成，再按Commit A→E实施。此次是新研究模型的主动改版，**不需要兼容旧E Obs ABI、旧Replay ABI或旧Checkpoint，也不使用旧E+Tutor 1M权重**。一开始就完成MACD partial 5m的原生 Gym Observation → Replay → Dreamer Encoder；不要留到WP04。采用原生Quality TP Reward，五策略P0–P4全部生成，两个老师L1质量事实和L2严格动作标签完整分工；不得把旧C1 Oracle冒充Quality P4，不得未来泄漏。保留金融撮合/时间/账户来源验证、同版本Resume和WP01的草案/正式校验机制（更新为课程V2）。按每提交运行CPU测试并记录退出码，最后在经过认证的有限TRAIN源上生成并完整重播小样、输出coverage与证据。**不改真实旧Run，不读取TEST行情，不自动开放训练Gate，不做GPU训练或18合约全量生产。** 交付五次提交SHA、各阶段实际测试、G1/G2/G3结论、数据/模型合同哈希、源码审查和`WP02_IMPLEMENTATION_REVIEW.md`。遇到来源不可验证或后缀证明失败时如实NO-GO，不能降级定义来表面通过。
