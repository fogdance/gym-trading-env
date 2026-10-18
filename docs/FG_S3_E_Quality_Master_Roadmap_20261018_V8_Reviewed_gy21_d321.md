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
