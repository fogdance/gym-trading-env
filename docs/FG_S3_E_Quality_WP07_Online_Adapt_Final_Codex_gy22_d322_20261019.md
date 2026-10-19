# FG-S3 E+Quality TP — WP07 正式 Codex 任务书 V2

**日期：2026-10-19。状态：可正式交付 Codex 开发；不代表训练授权。** 基线 `gy22/d322`，双仓均已把 WP06 集成到 `integration/s3-next`。必读《FG_S3_WP06_to_WP07_Interface_Code_Review_gy22_d322_20261019.md》和 Master Roadmap V10，以及两个仓库的 `AGENTS.md`。

## 0. 唯一研究目标与工程结果

将真正受监督训练后的 `quality_actor` 接入历史 TRAIN 行情的模拟在线练习：优质做空机会才 ADD，没机会 WAIT；三手空单，TP+5、SL−30，优先快速TP、低逆行、回避快速逆行10点。`SHADOW`可独立执行；`ONLINE_SUPERVISED`可事后使用 TRAIN L1/L2 更新相同 `quality_actor`；`ONLINE_RL_CONTROLLED`实现真正面向同一 Actor 的想象式RL与Critic，并被技术/研究 Gate 有效约束。

**不是实时实盘下单，不引入第二套金融Env或Reward，不重训旧1M、不做旧模型兼容、不把预测/Q/教师后验塞入学生观察、不先验宣称RL改善绩效。**

### 版本与前置

- Gym `integration/s3-next`: `d5153268bcd33d5a4e54721767c8782f895283f9`；运行源码冻结节点 `53ffea2353618be99f35673ab8f4c6c544b7273d`。
- Dreamer `integration/s3-next`: `e8c2688bc817662e5e6fa7e8cee62825bcf80b00`；运行源码冻结节点 `ffe6a831877ee91769f611dd16c374fb96941ebd`。
- Codex在真实本机打印完整 `rev-parse`、父提交/dirty、worktree，确认实际主线是否比归档更新；若 DATA-MULTI、EVAL-BASE 新增完成，直接用新版接口，避免重复开发。不要 reset/clean 用户正在训练的树。
- **真实受训 F1/F2/F3、完成的 V3 多合约教材、GPU更新、正式研究授权目前均没有完成的证据**。这些不阻断 DEV 合成代码开发，但正式TRAIN训练不得由默认执行自动触发。

## 1. 现有真实接口必须复用

| 责任 | 实际已有接口 | WP07要求 |
|---|---|---|
| F3 模型与损失 | `dreamerv3/fg_s3_e_quality_actor.py::{distribution,actor_objective,train,features,register_state_schema}` | 在线监督复用本 Actor owner、L2优先损失和因果 burn-in，别另造BC算法 |
| F3阶段与运行 | `embodied/run/fg_s3_e_quality_actor.py::{ActorStreams,ActorRun}`，继承 `HeadsRun` | 参数接续/状态/日志/CP复用；F4只写必要薄协调器 |
| F3来源 | `embodied/run/fg_s3_e_quality_transfer.py::{inspect_stage,transfer_parameters}` | F3→新Run参数级拷贝，不复制旧更新步数、优化器与随机状态；Source证明完整 |
| 原实时Env采集 | `gym_trading_env/research/fg_s3_v2_collector.py::V2Collector` 与 `fg_s3_dagger_d0.py::D0Collector` | 复用Driver/Obs/Replay/ledger，真正证明 teacher-free；禁止旧 teacher query 参与动作 |
| Evaluation | `gym_trading_env/research/fg_s3_e_quality_actor_checks.py::{evaluate_checkpoint,...}`，原 native `reconcile_trace` | 扩展为持久、可恢复、未修改 action 的 SHADOW & Actor-only 评价 |
| 训练与Checkpoint | `embodied/run/fg_s3_e_quality_heads.py::HeadsRun`，`embodied/run/fg_checkpoint.py::FullCheckpoint` | 双来源 + Env/Collector 的原子快照，复用既有生命周期原语 |
| 旧想象RL | `dreamerv3/agent.py::{loss,_sample_imag_action}` / `fg_s3_v2_agent.py::RL_MODES` | 增加仅F4生效的 `quality_actor` 想象路由，不能训练`pol`冒充 |
| 教材标签 | WP02 P0–P4、WP03原生V3 Join/Reader | 延迟发布训练后验，合法有效位、Owner、stepid，禁DEV/TEST教师 |

## 2. 执行模式与版本规则

| 模式 | 学生策略 | 后验教师 | 更新 owner | Gate |
|---|---|---|---|---|
| `SHADOW` | 无教师接管、Argmax固定或显式固定探索 | 仅独立诊断，不入策略 | 无；模型/slots逐键不变 | Collector/金融/时序基本正确 |
| `ONLINE_SUPERVISED` | `quality_actor` 真实动作 | 仅TRAIN标注成功后用于loss | **仅 `quality_actor/`** | 数据与L2身份有效；不要求未使用的Q/Prediction/Decision准入 |
| `ONLINE_RL_CONTROLLED` | 同一个`quality_actor` | 只可作辅助诊断/BC防遗忘 | `quality_actor/` + 经授权的 `val/slowval/`，WM默认冻结 | 数值、WorldModel、mask、value、正式RL独立认证 |

**解决课程身份冲突：**既有 `expected_stage_contract('ONLINE_ADAPT')` 把 Prediction/Q/Decision/RL 捆绑为前置。对 SHADOW/ONLINE_SUPERVISED，首先采用明确 `source_stage='ACTOR_SUPERVISED'`、`execution_mode` 的 **F3 online continuation**，带独立 Run/Execution ID，不冒称正式 F4 `ONLINE_ADAPT` 获准。RL采用新的版本化 `FG_S3_E_QUALITY_F4_EXECUTION_V1` 与现有合同上限一致的认证协议。若在不改变课程语义的情况下确实无法组合，则提出明确的**一次性新版本**，经Review后更改；禁止悄悄篡改现有 Corpus `course_identity_hash` 或移除正式Guard。

## 3. 八个大任务 / 小提交

### 07.A — 基线审计、Execution模式、F3→在线新Run来源

**文件归属：**优先 `dreamerv3/fg_s3_e_quality_online.py`、`embodied/run/fg_s3_e_quality_online.py`，Gym 的薄 `fg_s3_e_quality_online.py` 和原统一CLI；仅必要共享代码小补丁。

实现可独立检验的 execution spec：来源 F3 checkpoint / 完整非优化器参数、Actor-only provenance、课程/Obs/MACD V3/Reward身份；`mode`、model seed、Env seed、采集/更新预算、前后验教师使用权限、混合比例、探索策略、Checkpoint间隔、Run路径。F3→新Run复制所有兼容 non-opt 网络与所需模型状态，新的在线优化器、步骤、随机种子、在线Replay、账户初态；同Run resume只能Exact Resume。支持工程 synthetic F3 source 和正式F3 source 两类，不能互冒。

**A验收：**正确分配 owner 与新run身份；原F3模型参数100%核对；optimizer fresh；非法源、课程错误、TEST来源、冒充F4准入在构造真实Run前拒绝。无完整真实F3不阻断DEV演示。

### 07.B — 无教师接管的真实金融账户与 SHADOW

复用 `V2Collector/D0Collector`、Driver/Env和 ActorPolicy，设置 `teacher_enabled=False, beta=0`、无quality_catalog/outcome_catalog/beta_schedule，仍须实际证明旧 D0 不会在动作路径调用Oracle；必要时抽取单一 teacher-free adapter，不新建金融撮合器。每次决策保存原生 mask、当前可见E+5m partial MACD、原生Actor logits与动作、执行后 reward/position/TP/SL、账户/owner、真实市场分钟和日志提交点。支持 `SHADOW` 0更新、指定预算退出、完整账户 ledger、分段checkpoint与 resume。

**B验收：**至少两段合成行情含持仓、TP、SL、合法WAIT/ADD、交易日切换；原生 Replay与 `reconcile_trace` 对账一致； teacher 函数注入 `raise` 也完全不影响 Actor真实动作与Reward；模型与优化器全部逐键不变。

### 07.C — 新旧课程融合与延迟合法标签发布

在线 Replay 是可变账户历史，旧 WP03 Corpus 是不可变只读。用现有/已完成的 DATA-MULTI API 作为历史统一Reader；新增最薄 在线 Reader 与**按来源身份区分的 sampler**。四条防串联：单条窗口不可跨账户/合约/segment；strict Pair不能跨owner；Teacher只读已证明同TRAIN后缀；混合 batch 的 Natural/aux/online 每项独立实曝和源身份。

在线L1/L2不能凭当前 `actor_action` 生成：必须用官方TRAIN的合法完整suffix和已审Oracle计算，关联 *post-step* cursor、账户状态、teacher有效位；数据暂不能合法标注则 pending/invalid，绝不补 WAIT 或凭已执行 TP 伪造teacher_action。新增 sidecar publication ledger，记录等待、认证、发布、消费游标，恢复不重复或漏发。

**C验收：**两份合成history Corpus + 在线 native Replay；含合法L2两类、无效、晚到、跨账户拒绝；确定性种子与消耗状态；真实2份已发布Corpus的终验可按实际来源状态标NOT_RUN。

### 07.D — Online BC：更新原 `quality_actor`

直接复用 `fg_s3_e_quality_actor.py::{features,distribution,actor_objective,train}` + Frozen WM；最小扩展 adapter 适应在线流，保留原 `K`步 burn-in，不把单前缀冒充完整历史。在线监督数据和历史 Natural/aux 共用相同L2 action标签合同；L1 Pair仅严格L2一致、默认权重0；不存在合法L2时只记录消费，不更新。WM/Quality Heads/Q/旧 pol/Val全部参数和slots冻结（在线BC阶段）。可观察自然/平衡/严格Pair损失的独立梯度、有效样本、实际ADD曝光与PADD分布。

**D验收：**真实合成在线流至少若干次有效ADD/WAIT更新，只有quality_actor owner和其optimizer slots变化；不存在标签、无效mask、异常数值全部安全处理；早期特征变化可驱动输出变化；teacher/未来字段变更不影响Actor前向。

### 07.E — 同一个 Quality Actor 的想象式 RL + Critic

**这是新开发主体。** 现有 `Agent.loss()` 和 `_sample_imag_action()`从`self.pol()`采样，不能直接开启Gate。将策略头选择封装成仅F4显式启用的**单一路由**：真实Env `quality_actor` 与想象动作采样及 imag policy loss 使用同一模型参数、同一 masked categorical/unimix语义；其他模式继续保持原路由。允许独立`val/slowval`作为 value owner，不保留旧pol为实际交易策略。使用原WM `enc/dyn/rew/con/avail`，正确市场分钟折扣、Quality TP Reward，以及关于 TP/SL 的金融语义。WM默认冻结；想象动作mask由 `avail`推理时必须实测合法性、false-open、空mask fallback，并在有风险时拒绝 RL，而不是默认为合法。

**E验收：**CPU合成真实 `ninjax` 梯度与optimizer试验：RL打开时仅quality_actor +获准val/slowval owner变；旧pol绝不变；RL关闭时val/slowval逐键不变；真实与想象头参数身份/ logits可追踪。拒绝无法校准的 Imag Mask/Reward/Continue。不要仅以loss finite、entropy变大或 Actor出现ADD作效果证据。

### 07.F — Exploration / numeric /研究Gate与消融

实现 `COLLECTION_VALID`,`SUPERVISED_STABLE`,`WM_VALID_FOR_RL`,`RL_NUMERIC_READY` 四道独立Gate；如果启用其他预测/Q/Decision修正，再分别受本身Admission控制（默认全部关闭）。探索只作用于显式开发或TRAIN合法环境、仅在真实账户合法动作mask内；Actor-only考试完全确定性、无Teacher、无探索。报告预设自然/分层样本基准、L2一致Pair、RL_OFF / RL_ON、探索率、合法成交、SL、TP时间与MAE。支持RL技术模块开发通过但 formal Gate 拒绝状态，**不为凑GO而改阈值或欺骗资金风险**。

### 07.G — 原子 FullCheckpoint / source transfer / 完整冷恢复

在原 `FullCheckpoint` 能力上组合：Agent所有参数/slots/正常化器与Critic、主/辅助Reader及其RNG、live Replay/chunk索引、Sampler RNG和Prefetch、Env/Driver/实际持仓/账户、政策与探索RNG、L1/L2 sidecar发布/消费账本、阶段Budget/Gate、日志提交点。数据不复制不可变历史Corpus，只存 identity + reader cursor/RNG；可变Replay保留自身chunk WAL/commit identity。冻结双仓/课程/来源身份后才运行，不做“最新模型+任意Replay”拼接恢复。

**G验收：**至少四类新进程复原：SHADOW、online BC、在线标签刚好发布中断、带RL optimizer slots 的合成RL；连续N vs N1+resume+N2，在动作/成交、loss、权重、slots、全部读者RNG、日志和下一batch一致；破坏最新node必须显式失败，不静默回退。

### 07.H — 集成、测试、报告、路线图

双仓独立feature worktrees开发，A-H每阶段最少一组定向测试及对应提交，完成后在一次性隔离integration树按配对SHA审查/合入。原E+MACD V3金融因果、WP03 Join、WP04 F1、WP05 F2、WP06 F3 原有功能仍可工作；既有运行代码不得被隐藏改成新F4。最少产生完整可复核的 `WP07_IMPLEMENTATION_REVIEW.md` 和更新 Master Roadmap。

## 4. 主验收矩阵（开发层）

| 类别 | 测试/必须结果 |
|---|---|
| A：身份/来源 | 01 明确F3 source；02跨阶段fresh optimizer；03同run exact-resume；04非法来源/TEST拒绝；05 F3 online续课不得冒称F4正式准入 |
| B：Teacher-free | 06 beta=0原生动作；07注入teacher失败仍可交易；08real mask合法；09 partial 5m MACD 不看未来；10 TP/SL/roll与ledger重播 |
| C：数据 | 11 stepid Owner+post-step准确；12等待label不是WAIT；13自然/aux/onlive三流身份；14单窗口不跨账户；15跨source Pair拒绝；16两份Reader随机/游标恢复；17 MIXED exposure非独立市场N |
| D：在线BC | 18 only quality_actor参数与slot更新；19 WM/quality heads/Q/旧pol/value冻结；20无合法标签0更新；21 K预热实际响应；22异常数值原子回滚；23 L2优先、冲突L1 Pair被过滤 |
| E：RL | 24同一Actor在真实/想象路径；25 RL更新仅quality_actor/Value owner；26原pol未更新；27 mask false-open审查；28 reward/continue/discount真实语义；29Critic有限梯度；30RL关闭归零且冻结 |
| F：探索/Gate | 31探索在真实mask内；32shadow不更新；33技术不够时Gate阻止RL；34预测/Q融合off；35teacherfree eval deterministic |
| G：恢复 | 36连续vs恢复SHADOW；37在线BC；38教师标签发布中断；39 RL slots；40不一致代码/数据/CP身份非零拒绝 |
| H：回归 | 41 F1/F2/F3和课程原guard；42原行情MACD V3因果；43在线native账户跨夜与资金语义；44无GPU亦可通过CPU工程测试；45真实TRAIN未启动必须标NOT_RUN |

上面45项是**验收行为**，不是承诺45个刚好命名的 pytest 函数。Codex不得跳过失败或降低金融规则来凑数。

## 5. 开发与正式运行严格度区分

**DEV：**默认CPU/GPU/内存空闲，停止重复 `free -h`、`nvidia-smi`、CPU/内存探针；CPU合成更新和必要的小型GPU功能烟测可直接做，定向测试优先，出现实际OOM/CUDA设备异常再定位。因果性、action mask、金融Env/Reward与Teacher隔离在任何阶段均必需。

**DATA：**真实TRAIN新Corpus/在线Sidecar发布需保留实际来源/Replay/身份/质量记录；当前开发调试不要求每次代码提交都重新G3。

**FORMAL：**只有真实F1/F2/F3受训权重、完整 V3 TRAIN Corpus、跨来源调度和对应正式准入通过，用户明确授权后才能长期训练。DEV技术性RL实现≠真实可用RL；正式 TEST只在WP08冻结后使用。当前GPU初始化失败不准标GPU PASS，开发实现不因此停工。

## 6. Codex 提交与交付格式

A-H按功能小提交，重大共享Agent改动单独 Review。每组报告准确 commit SHA、文件清单、定向CPU/小GPU执行命令、原始退出码、各网络leaf/slots变化、Teacher隔离证据、Env ledger与CP冷恢复对照。真实多合约数据、正式F3来源、GPU更新、RL研究价值未测则用NOT_RUN/NO_GO。

最终《WP07_IMPLEMENTATION_REVIEW.md》包含：代码身份（双仓配对）、功能完成与残留、SHADOW/BC/RL三模式测试、L1/L2合法标签曝光、无教师决策统计、真实TP/SL、关键场景失败与修复、实际资源耗时（运行自然产生的日志即可，不做例行占用采样）、对WP08的接口，以及 Master Roadmap 更新记录。不得删除旧运行或擅改训练主线。

## 7. 反向设计 Review：必须避免的陷阱

1. RL只训练旧`pol`，执行却用quality_actor：拒绝。
2. 以 Quality V2旧`LiveRun`冒充新F4：拒绝。
3. 在真实决策时让Oracle/Sidecar改变动作：拒绝。
4. 将无法认证的在线L2标签默认转WAIT：拒绝。
5. 用同一个账户不同行为后验加总为多份独立金融结果：拒绝。
6. 在线监督仅有WAIT样本却宣称已掌握ADD：拒绝。
7. 想象 mask把仓位内部状态预测错后疯狂ADD：未完成校准的RL Gate关闭。
8. `F3`在线续课误当F4 `ONLINE_ADAPT`全准入：拒绝。
9. 修改课程Reward/金融Env/Observation不更新身份：拒绝。
10. 冻结参数被动量/权重衰减改变：逐leaf与optimizer slots证据。
11. Checkpoint缺Env/Reader/标签发表游标，无法可重复在线继续：拒绝。
12. 直接复制 WP03 Reader 实现新Online Reader/跨Corpus采样：优先共用必要原语和DATA-MULTI。
13. 因GPU环境错误或真实Corpus不足停止全部开发：可以完成CPU工程，但如实报告NOT_RUN。
14. 把训练场景合成TP、P4教师结果当成Actor盈利：拒绝。

## 8. 完成定义及下一步

- **WP07 CODE_GO**：SHADOW、ONLINE_SUPERVISED、受控Quality Actor想象RL三个技术模式有真正代码、CPU原生时序/梯度/恢复证据；不可用的正式Gates仍关闭。
- **WP07 FORMAL_GO**：真实已训练F3、完整可认证教材、来源、授权、GPU/模型数值与策略基线齐备后独立申请，不随CODE_GO自动开放。
- 下一步WP08继续 Actor-only冻结考试；WP09扩多合约教材和消融；WP10研究冻结。**整个项目目标始终是优质空单入场，而非代码模块数量。**
