# FG-S3 E + Quality TP：长期主路线图（Master Roadmap V1）

**日期**：2026-10-16　**当前核对版本**：Gym `e57189b` / Dreamer `44a6cc6`（WP02功能分支，尚未合并 integration）。

**文档地位**：后续每轮开发、Code Review、设计变更及验收均应更新这份主路线图的**当前状态表、依赖、风险、版本/变更记录**，避免 WP 编号漂移。为防止只更新讨论文本而遗漏长期状态，每次实现 Review 都应交付新的 `Master_Roadmap_vN.md`，保留历史版本；不自动宣称用户仓库被同步修改。

## 0. 不变的项目目标和约束

- **学生**：从零训练新 E Encoder（1m/15m + 当前可见的 partial 5m MACD）、RSSM、World Model heads、Quality Heads、`quality_actor`。旧 E/Tutor 1M 参数不复用；新旧 ABI 不需要兼容。
- **老师**：Quality V2；L1“入场质量老师”评价立即 ADD 的 TP5/SL30、止盈速度、官方日、MAE、早期逆势；L2“最终动作老师”对当下真实账户合法 ADD/WAIT 做 Quality TP 严格完整后缀比较。老师可看 TRAIN 未来；学生实时观测不能看未来。
- **教材**：真实 TRAIN 金融 Env 的 P0–P4 六条原生行为轨迹、正负及中间质量教材、教师 Sidecar 与因果 MACD；**先有可用数据，再凭 Actor 考试反馈提高覆盖**。
- **金融目标**：固定 Single3 做空入场，3手、TP+5、SL−30；高质量ADD优先快速TP、低MAE，低质量/无机会 WAIT；跨官方日完成只降低质量，**不强制日内平仓**；按同一原生 Quality TP Reward，不暗改结算/费用/Mask/时间。
- **最终成功口径**：独立 Actor-only 在冻结、隔离的合约/交易日样本上真正选中优质ADD，且快速止盈、早期逆势、SL、成本及风险指标可被客观评估。训练 loss/更新/Oracle P4收益都不替代这个标准。
- **实施原则**：新架构直接一步到位，不为旧模型造兼容适配；功能按小任务和独立提交推进。保留因果性/来源/身份/精确恢复/账户正确性等与兼容无关的必要保障。任何未授权 GPU 训练、全量数据生成、读取 TEST 或解除正式 Guard 都必须先获得明确许可。

## 1. 唯一主路线（不是到 WP06 结束）

```text
WP01 课程合同（已完成）
   ↓
WP02 新 E 输入 + 多行为教材 + L1/L2（已报告完成，代码已审）
   ↓
WP03 Teacher 数据认证 + Replay/Sidecar Join + 统一抽样Reader（当前待开发）
   ↓
WP04 FOUNDATION_WM：从零训练 E Encoder/RSSM/WM  ⟶ ┐
   ↓                                                │
WP05 QUALITY_HEADS：训练 Outcome/Risk/Score Heads ───┼→ WP07 阶段交接/在线适配
   ↘                                                │       ↓
WP06 ACTOR_SUPERVISED：新 quality_actor 真正学ADD/WAIT ┘  WP08 Actor-only独立考试
   ↑                                                      ↓
   └────────── 依据考试反馈做受控迭代 ← WP09 多合约扩展/课程消融/性能
                                                          ↓
                                                  WP10 冻结实验与研究交付
```

**依赖澄清**：WP04 的WM模型与WP03 Corpus是 WP05/WP06 的共同前置。WP05预测训练是计划中的教学环节，**不要求 Prediction/Decision Admission 通过才允许 WP06 使用真实L2标签训练Actor**；F3 的独立授权由 WP06 实现。WP08可先评价离线WP06 Actor、随后评价WP07在线结果；WP09可以与阶段训练并行开展扩展设计，但正式多合约数据生产另需授权。

## 2. 大功能包 → 小任务 → 输出/验收总表

| WP | 大功能 | 小任务编号 / 关键工作 | 输出与验收 | 当前状态 |
|---|---|---|---|---|
| **WP01** | 统一课程协议 | 01.1 Quality TP Reward/Obs/Teacher合同；01.2 F1–F4参数矩阵；01.3 Source preflight/DRAFT-FORMAL；01.4 负例与Guard | 课程合同与预检；据实施报告241选定CPU测试通过；训练NO-GO | **报告级完成**；其后由WP02升级至课程V2 |
| **WP02** | 原生教材、老师与新Observation | 02.1 V2 E+partial5m MACD/Encoder；02.2 P0–P3多账户；02.3 L1结果/风险；02.4 Quality TP严格L2/P4；02.5 Replay发布/真实账户复核/来源/覆盖 | `FG_S3_E_QUALITY_CORPUS_V2`，真实小样G1/G2/G3；312选定CPU测试按WP02报告；独立静态源码审查 | **源码已审、报告完成**；未合并integration；真实原始数据未随Git归档上传 |
| **WP03** | 数据监督与课程Reader | **03.1** Corpus G3/账户/身份核验；**03.2** V2 L1/L2→loss字段投影；**03.3** `stepid+account`完整Join；**03.4** episode-safe自然/分层/混合抽样；**03.5** Agent CPU对齐及收据 | 唯一训练批次入口；真实TRAIN小样Reader GO；正式Train仍NO-GO | **任务书完成，待Codex实现** |
| **WP04** | F1新学生世界模型基础训练 | 04.1 Fresh E Agent/Quality TP初始化；04.2 WM-only优化器/冻结规则；04.3 Decoder/Reward/Continue/Avail目标；04.4真实Replay离线小样→阶段checkpoint；04.5loss/梯度/因果/恢复和初步DEV表征测试 | Encoder/RSSM/WM Heads 参数确实更新，Actor不更新；独立F1准入 | 未开始 |
| **WP05** | F2 Quality Heads | 05.1 L1各窗口有效位与Head目标；05.2严格冻结WM/Actor；05.3 Outcome/Risk/Score参数更新；05.4预测可靠性和消融；05.5可选Q独立目标合同 | Heads的来源、梯度及预测证据；PredictionAdmission仍单独申请，不作为F3硬前提 | 未开始 |
| **WP06** | F3 Quality Actor基础训练 | 06.1新Actor与freeze matrix；06.2 L2 strict BC/平衡/合法Pair；06.3 L1质量排序（与strict动作分离）；06.4自定义采样与真实曝光计数；06.5Actor参数变化、排序与Argmax ADD、no-future证明 | Actor-only 能在DEV产生可审计的自主WAIT/ADD；排除全WAIT和“仅教师操作”假成功 | 未开始 |
| **WP07** | F4接续与受控在线训练 | 07.1 F1→F2/F3 checkpoint语义/优化器重建；07.2 Replay/Reader RNG恢复；07.3 LIVE Env/SourceAdmission/Commit identity；07.4 Gate条件下Actor/Critic/RL；07.5在线恢复、teacher仅TRAIN训练用 | 冻结来源到在线训练完全打通；阶段切换与冷恢复正确 | 未开始 |
| **WP08** | 独立考试及策略指标 | 08.1 Actor-only Eval适配；08.2 固定DEV与后续严格TEST一次性协议；08.3 TP速度、MAE、快速逆势10、SL/手续费、成本与亏损；08.4合法机会覆盖、等待机会成本、分合约稳健性；08.5 对照Oracle/简单规则/随机且不拿Oracle控制策略 | 可解释的真实交易质量证据，明确GO/NO-GO；不过拟合研究数据 | 未开始 |
| **WP09** | 题库扩展与实验改进 | 09.1 经授权扩大TRAIN合约/官方日；09.2保留快速SL与艰难TP覆盖；09.3自然vs分层/指标消融/老师权重；09.4吞吐/内存优化并验证不改语义；09.5统计不确定性/多随机种子 | 不牺牲因果与身份地改善模型，版本化课程实验 | 未开始；与WP04–08可受控交织 |
| **WP10** | 研究版本冻结与后续使用 | 10.1精确代码/数据/参数/种子/训练收据；10.2可复核评价报告；10.3结果不足的NO-GO后续计划；10.4若研究达标，才讨论下一步仿真/部署 | 不需要用户猜测哪个Run好；保证可恢复、可比较、可回滚 | 未开始 |

## 3. WP阶段合同：到底训练哪个网络

| 课程阶段 | E Encoder / RSSM / 常规WM头 | Quality预测Heads | `quality_actor` | `pol/val`及预测融合 |
|---|---|---|---|---|
| WP03 Reader | 全冻结，无参数更新 | 全冻结 | 全冻结 | 全冻结 |
| WP04 F1 FOUNDATION_WM | **训练** Enc/Dyn/Dec/Rew/Con/Avail（有效目标按原模型协议） | 冻结 | 冻结 | 冻结 |
| WP05 F2 QUALITY_HEADS | 冻结（如将来解冻须独立实验） | **训练**各有有效证据的Head | 冻结 | 冻结 |
| WP06 F3 ACTOR_SUPERVISED | 默认冻结（可在后续消融对照） | 冻结 | **训练** `quality_actor` | `pol/val`冻结；Q/Soft/Prediction-Fusion默认OFF |
| WP07 F4 ONLINE_ADAPT | 根据独立门槛选择WM微调 | 仅认证分支 | 持续训练 | RL Actor/Critic/Gate由单独准入控制 |

**规则**：某阶段声明训练≠实际参数已更新；所有阶段必须报告有效梯度、参数前后差异、优化器状态、有效样本和失活分支。Quality Head预测好≠允许修正Actor动作；双教师答案好≠学生已学会。

## 4. 交付与 Review 循环（今后每一轮遵循）

1. 冻结**当前实际双仓SHA**、分支、dirty、已完成WP；不把旧 `integration/s3-next` 自动当最新。
2. 每个大WP先给子任务/输入输出/合同/负例和Review，然后Codex在独立功能worktree分小提交实现。
3. 用户上传最新源码与实施报告→我们独立核对源码及报告→明确 `CODE_GO`/`REAL_DATA_GO`/`TRAIN_GO` 三个状态；不混用。
4. 随后更新本文状态表、实际SHA、已发现风险、下一步分包、变更记录，再发布最新 WP 的Codex任务书。
5. 变更核心方向需用户讨论；具体实现名可以随真实源码调整但不得无声改变金融规则与教学目标。

## 5. 当前阻断/交叉风险

- **R01** WP01/WP02 还在功能分支；后续必须从 WP02 真正提交继续，不允许双仓SHA交叉错配。
- **R02** WP02 Corpus 真正 `real_small_04` 未在这次Git归档中；WP03必须在用户机器核对真实数据，不能沿用报告成训练准入。
- **R03** 新 teacher envelope 与旧 `quality_catalog.FIELDS`不相同；WP03必须映射，不可原样旧attach。
- **R04** 旧 `quality_q_v2.UTILITY` 不等于本轮严格Quality TP后缀，默认Q/Soft Loss无效且F3融合关闭。
- **R05** 真实教材只有FG201两官方日，不能据此调优训练容量或解释策略泛化。
- **R06** F1/F2/F3真实优化器和SourceAdmission尚未完成；不解除公开 E+Quality Guard。
- **R07** 质量分数不包含完整MAE风险；风险独立保留，不改正式Quality TP Reward。
- **R08** 现有 legacy Quality `HEADS_ONLY` 仍有旧C1 Reward约束、`CONTROLLED_FULL` 对 PredictionAdmission有额外要求；WP04–07须实现本轮新课程的正式阶段生命周期，不能假定 WP03 Reader 接好就能运行训练。

## 6. 版本变更记录

| 版本 | 时间 | 更新 |
|---|---|---|
| V1 | 2026-10-16 | 基于 `gy18/d318` 将原WP01–WP08扩展为WP01–WP10持续研究闭环，更新WP02实际身份和WP03任务边界；确定每次WP Review同步更新本路线图。 |

**下一里程碑**：WP03 Reader 工程 GO + 有限真实TRAIN来源与teacher的 DATA GO（均不等于训练授权）；之后准备 WP04 F1 的完整训练工程设计。
