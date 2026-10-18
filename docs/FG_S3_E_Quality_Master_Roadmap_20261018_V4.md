# FG-S3 E+Quality TP：Master Roadmap V4 — 后续唯一持续跟踪基线

**更新日期：2026-10-18。** 对应源码 Gym WP03 `7366886f3b40bf74aa4485978ffcc97e39d0d53e`，Dreamer WP03 `446d1bc99debd2d05c313eafd7d0c414ebd2815c`。这是本轮共同祖先；当前 WP04 A–E 成对分支与最终 SHA 见 WP04_IMPLEMENTATION_REVIEW / WP04_DELIVERY_COMMIT_LOCK.json，不能从归档默认 HEAD 或 integration 代替此祖先。本文件继承 V2 的目标与功能边界，**更新 WP04 实际 A–E 交付、只读验收和 DEV/DATA/FORMAL 分级；后续范围不缩减**。

## 0. 不变的方向

- **学生**：新 E 1m + 15m + 可见partial 5m MACD、RSSM、原生WM Heads、Quality Outcome/Risk/Score Heads、Quality Actor；**从零训练**，不使用旧E1M权重、不维护旧模型ABI。基础学生包含WM和Actor，实际分别在WP04、WP06完成。
- **老师**：Quality V2；L1 给每个候选合法入场的真实TP5/SL30、官方交易日完成、时间、MAE、快速逆势等后验质量事实；L2对当前合法账户的ADD/WAIT按Quality TP折扣后缀严格判定。老师能用TRAIN的未来给答案，学生 Observation 绝不含未来。
- **教材**：E-native P0、P1、P2-low、P2-mid、P3、P4六种真实账户轨迹（对应五类策略）、正/负/中间质量、Sidecar三方Join、统一Reader；先实现真实有效版本，靠后续独立考试指导扩题。
- **交易规则**：Single3，方向由既定项目环境提供，实际ADD三手做空、TP+5/SL−30，优先低MAE和快速TP；跨官方日TP**降质量而不强制日内平仓**；正式 `FG_SINGLE3_REWARD_MARKET_MINUTE_TP_QUALITY_V1`、原金融Env撮合/Mask/时钟不改。
- **最终验收**：Actor-only（不能使用老师代下单）在冻结且隔离的数据上自主优质ADD、没有机会WAIT；风险、时间、成本、止损、不同合约可靠性和统计不确定性均有真实证据。
- **研发原则**：大WP+小任务、真实源码SHA/证据/GO分层/下一步可开发合同；不为旧模型兼容牺牲新架构，但不取消金融正确性、因果、数据/代码来源认证、断点精确恢复。GPU训练、真实大规模跑数、读取TEST及正式部署另请授权。

## 1. 总功能流（不会在WP06结束）

```text
 WP01 Course/Reward/Obs身份     [已完成]
    ↓
 WP02 新E+partialMACD / 正负教材 / L1-L2 / P0-P4   [已交付源码/真实小样报告]
    ↓
 WP03 Source + 三方Join + Reader + teacher loss-ready  [源码交付、本机重新核验 GO]
    ↓
 WP04 F1 FOUNDATION_WM：新 E 世界模型真实优化器训练工程 [CODE_GO + REAL_DATA_READ_GO；真实权重 NOT_RUN]
    ├──────────────────────────────────────────────┐
    ↓                                              ↓
 WP05 F2 QUALITY_HEADS：L1有效头监督         WP06 F3 ACTOR_SUPERVISED：L2严格动作+L1排序
    └──────────────────────┬───────────────────────┘
                           ↓
 WP07 F4 阶段接续与在线适配 / source+optimizer+replay  [计划]
                           ↓
 WP08 Actor-only 冻结独立考试及策略风险审计         [计划]
                           ↕
 WP09 多日/多合约数据拓展+课程/指标/学习消融与性能优化 [计划，可与04–08受控交织]
                           ↓
 WP10 冻结代码、数据、模型、报告和GO/NO-GO研究版本     [计划]
```

依赖不是线性阻断：WP05预测Head能力不是WP06 Strict BC的先决准入，Quality Prediction/Decision Gate不得因此放松。WP08可在WP06后先做离线Actor-only考试，再对WP07在线结果做同口径对照；WP09按真实瓶颈插入研究回路。

## 2. 状态与大功能—小任务表（每次交付更新）

| WP | 大功能 | 小任务 / 对应产物 | 最新实际状态与边界 | 下游依赖 |
|---|---|---|---|---|
| **01** | 课程合同与安全准入 | 01.1交易/Quality Reward/Obs；01.2 F1–F4训练矩阵；01.3 Draft/Formal；01.4 Guard与负例 | **CODE：报告GO**。Gym `2904f3e`、Dreamer `347724e`；历史241 CPU selected PASS报告。WP02升级为课程V2。训练NO-GO | WP02、后续所有阶段身份 |
| **02** | 原生教材 + E+MACD + L1/L2 | 02.1Observation V2与Encoder；02.2P0–P3；02.3真实Outcome/风险；02.4Quality TP严格L2与P4；02.5Corpus+G3与审计 | **CODE：已交付**。Gym `e57189b`、Dreamer `44a6cc6`，WP03继续沿其分支；真实FG201小样仅开发机可访问；旧E模型不兼容是预期 | WP03、WP04–06 |
| **03** | 教材认证与唯一统一Reader | 03.A数据内容校验与native G3；03.B Replay/Trace/Sidecar同账户Join与Q/Soft禁用；03.C自然/分层/混合Reader、RNG恢复；03.D fresh E+Quality纯CPU Loss/冷Agent恢复 | **代码已交付、报告级 CODE_GO + REAL_TRAIN_DATA_GO**。Gym `7366886` / Dreamer `446d1bc`；177 CPU PASS+真实6账户G3+Reader报告；4146物理行、690独立市场时刻、119成交、171 strict ADD/1279 WAIT。本轮 WP04 已在本机重新执行原生 G3 和有限 F1 只读，未进行真实参数更新。**CURRICULUM_SUPPORT LIMITED；TRAIN NO-GO** | WP04实际代码入口；WP05/06抽样 |
| **04** | F1 世界模型基础学生 | 04.A Fresh整模与F1专用准入；04.B原生WM-only optimizer白名单与K步burn-in；04.C Reward/Avail/多尺度Probe；04.D Immutable Corpus Reader+FullCheckpoint状态；04.E CPU/真实只读核验、Review | **A–E 工程已交付，379 selected PASS；本轮新 G3/只读 GO**。Gym A–D 至 `22fbb52` / Dreamer 至 `c6c35b0`，E 完整 SHA 见 `WP04_DELIVERY_COMMIT_LOCK.json`。原 WM/单优化器、K burn-in、冻结和精确冷恢复均实跑；只有合成参数更新，真实 F1 权重/训练/研究 NO-GO | WP05/06训练所需WM基础 |
| **05** | F2 Quality Heads | 05.A L1有效性目标+各窗口；05.B冻结WM/Actor及对应optimizer；05.C Event/Touch/Adverse/Score头真实更新；05.D校准、长尾、Q单独目标准入；05.E可恢复交付 | 未设计正式执行版。F2只学预测，不自动允许决策融合；旧Q/Soft目标在当前WP03禁用 | WP06可独立进行；WP07权重融合研究 |
| **06** | F3 Quality Actor基础入场 | 06.A fresh训练阶段接续；06.B严格ADD/WAIT BC+Balanced；06.C同Owner合法Pair和质量排序；06.D正负曝光、Argmax自主ADD；06.E Actor-only smoke | 未设计正式执行版；**F2 Prediction Admission不阻止F3按合法L2 teacher训练**；学生先训练Actor才算完成“有基础”定义 | WP08首轮Actor-only考试 |
| **07** | F4在线适配与阶段handoff | 07.A WP04→05/06模型/optimizer来源；07.B native Reader/RNG/Live Replay完整恢复；07.C RL Actor/Critic/Gate和在线Teacher权限；07.D受控继续训练与失败恢复 | 已规划，需WP04/05/06真实模型后开发 | WP08在线测试 |
| **08** | 独立Actor-only测试 | 08.A固定DEV、TEST一次性隔离；08.B与P4/规则/随机对照；08.C TP速度/SL/早期逆势10/MAE/资金曲线/成本；08.D合约和官方日稳健性、置信区间 | 已规划，无Actor研究成绩；**不把严格教师的赢家表现写成模型获利** | WP09迭代/最终WP10 |
| **09** | 题库扩展、消融及效率 | 09.A 合约/官方日扩充；09.B快速SL/艰难TP覆盖；09.C Nature vs Mixed/指标MACD/Quality权重；09.D内存/磁盘/训练性能；09.E多随机种子 | 已规划，不能用单合约两日训练集谈泛化；完整18合约扩展单独授权 | 反馈WP02–08，最终WP10 |
| **10** | 冻结研究版与交付 | 10.A双仓代码+Corpus+Teacher+checkpoint/seed；10.B独立考试报告；10.C优化失败/继续研究的明确NO-GO；10.D达到目标再讨论后续应用 | 计划中，非部署许可 | — |

## 3. 阶段参数矩阵（固化）

| 学习阶段 | `enc,dyn,dec,rew,con,avail` | `quality_event,touch,adverse,score` | `quality_actor` | 原`pol/val/slowval` + Q/Soft/融合 |
|---|---|---|---|---|
| WP03数据准备 | 无优化器 | 无更新 | 无更新 | 全禁 |
| WP04 F1 | **TRAIN** | 精确冻结 | 精确冻结 | 精确冻结 |
| WP05 F2 | 默认精确冻结 | **TRAIN有效目标** | 精确冻结 | 默认关闭，独立认证才开 |
| WP06 F3 | 默认精确冻结 | 默认精确冻结 | **TRAIN** | 原参考Actor/critic不冒认Quality Actor；Q/Soft/Fusion保持OFF |
| WP07 F4 | 经审计的可选更新 | 经认证按需更新 | 持续训练 | RL/Gate完整资格与实验门槛另设 |

额外约束：WP04必须先完整fresh创建`quality_*`及actor网络布局并冻结；Reward/Con/Avail学习的原生目标不随L1/L2后验更改。六个WM参数前缀≠六个loss键。所有阶段需逐键记录params/optimizer state/有效样本与梯度，不能用loss下降冒称参数已学。

## 4. 当前真实证据和阻断表

| 状态维度 | 当前判断 | 直接证据/限制 |
|---|---|---|
| WP03源码接口 | **CODE_GO（静态核对）** | 两仓WP03分支ref中的Source/Reader/Catalog/Agent Forward实际代码齐全 |
| WP03真实小样Reader | **本轮重新核验 GO** | WP04 G1_native_reverify / G1_read_only，实际 Corpus 字节与 native G3/source/index 匹配，真实更新0 |
| 教材规模 | **LIMITED** | FG201单合约两交易日/六账户；高质量正负分层与泛化不能保证 |
| WP04 WM optimizer | **CODE_GO** | 显式 FOUNDATION_WM 路由与单原优化器，合成四步有逐键变化；真实更新0 |
| WP04随机窗口时序 | **ENGINEERING_PASS** | 原首帧+K32 detached warmup，T64；同窗原生动作/账户校验、每窗新 carry；有限历史近似，不冒称全历史 |
| WP04完整Checkpoint | **ENGINEERING_PASS** | 原 FullCheckpoint external immutable Reader backend；连续4 vs2+冷恢复+2全部状态/下一批/指标 bitwise |
| WP04参数冻结 | **ENGINEERING_PASS** | 六组正向白名单，349实际叶，144精确冻结；非零动量/weight decay/unknown schema 负例 |
| E+Quality公开正式Guard | **CLOSED（正确）** | 正式prepare/start/resume/eval继续阻断，WP04只允许独立认证的F1阶段开发 |
| GPU/真实F1优化器/TEST | **NOT_RUN/NO-GO** | 本轮无启动授权；真实F1训练效果和Actor获利均未知 |

## 5. 接下来实际行动和版本维护约定

- **当前行动**：核对 WP04_IMPLEMENTATION_REVIEW、42项验收表、原始测试与交付SHA；可继续 WP05/WP06 工程。真实 F1 训练另行授权，不能用合成 Checkpoint 充当真实 WM 底座。
- **新代码交付**：用户上传新的双仓Git归档、`WP04_IMPLEMENTATION_REVIEW.md`和测试/收据；本 V4 已按实际原件更新，后续 Review 独立比对源码与可运行证据，更新每个WP的父子SHA、状态和阻断，不改变目标。
- **下一包**：WP05 F2 Quality预测Heads，WP06 F3独立Actor，WP07 F4接续，WP08考试，WP09有针对性扩数据和消融，WP10研究冻结。WP05与WP06都使用同一WP03 Reader + WP04产生的学生底座，避免另建数据系统。
- **状态统一四轴**：`CODE_GO`、`REAL_DATA_GO`、`TRAIN_AUTHORIZED`、`RESEARCH_GO`；任何缺项必须明示`NOT_RUN/NO_GO/LIMITED`。不以自签hash、CPU loss或教师胜率代替真正研究成绩。

## 6. 变更历史

| 版本 | 更新 |
|---|---|
| V1，2026-10-16 | 确定WP01–WP10方向、WP02→WP03真实接口静态Review。 |
| V2，2026-10-16 | WP04 F1预设计五提交与冻结/Checkpoint/Probe路线，等WP03真实源码。 |
| **V3，2026-10-18** | **实审gy19/d319 WP03 ref**，锁Gym`7366886`/Dreamer`446d1bc`，核对WP03报告与收据，明确F1 Reader K步warmup、WM-only optimizer及FullCheckpoint外部Reader三处实际缺口，正式发布WP04 Codex任务V2并保留训练NO-GO。 |

## V4 本轮新增证据与开发规则

WP04 新原生 G3：六账户各691行、4146总行，1288.11秒，exit0；新只读 F1 12窗/768目标、128独立市场时刻，真实参数更新0。测试379个最终不同选定用例通过，不表述全项目通过；历史失败与修复见 WP04_TEST_RESULTS。

采用用户新 DEV/DATA/FORMAL 三级规则：DEV 不要求每次源码冻结或完整 G3，允许临时目录、CPU合成、有限TRAIN与最小定向测试；DATA批次发布与FORMAL保留认证。三级均要求金融、MACD因果与Teacher隔离。正式GPU/全量18合约/TEST/研究Go仍无授权。WP05–10功能与研究验收范围保持。
