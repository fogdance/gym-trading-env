# S3 下一代：Quality-first Teacher + Strict Oracle-as-Auditor 详细设计 V1

**日期** 2026-10-15 | **依据** gy14 / d314 Git 归档实际 `feat/s3-e-model` 树：Gym `871ca09`，Dreamer `730939d`。本文件分清 **本次已实现的 Stage 1** 和 **未来待开发的 Stage 2–4**。不声称新一代模型已经完成训练。

## 0. 最终研究目标和必须保持的策略合同

- FG Single3 只做行情入场择时：动作 WAIT=0、ADD=1；一次3手、仅一批、TP+5点、SL−30点，原真实 Env 的成交、费用、官方交易日、强平与单仓释放语义完全保留。
- **第一目标**：选择风险合理、具有真实退出价值的高质量 ADD；无好机会 WAIT；时间是质量因素，但不是唯一因素。
- **第二目标**：比较立即 ADD 与保留空仓的后续机会，必要时由 Strict Oracle 证明最优性；不要求每个训练样本都先取得“资金>30万元”的旧 Oracle 证明。
- Strict Oracle 认证失败意味着“该账户下最优动作未认证”，**不代表 ADD 不合法、不代表单笔后果不可测、更不代表 WAIT 正例**。
- 新版本与既有 `fg_single3_oracle.py` / `fg_s3_oracle_state_query.py`、`Single3Reward` / `FGRewardScorer`、历史 P4 TeacherCatalog、Tutor-CD、B2、已完成的训练与 Checkpoint 完全隔离。

## 1. 研究标签分成四层（不要混用）

| 层 | 真值/预测 | 如何取得 | 允许用途 | 禁止用法 |
|---|---|---|---|---|
| L0 交易合法性 | `legal_wait,legal_add` | **当前真实 Env** 的 mask、账户保证金和仓位规则 | 决策动作筛选、实验入场资格 | 不从Oracle标签反推合法性 |
| L1 单笔真实后果 | fill/reject、TP/SL/OTHER/RIGHT_CENSORED、净PnL、MAE、持仓时长 | 冻结当前**完整账户快照**后 Fork 原Env，ADD一次、后续WAIT至退出/边界 | 事实监督、风险/退出模型、质量分布 | 单笔盈利≠当前一定应ADD；未结案≠亏损 |
| L2 入场质量 | `B(t,d)∈[0,100]`、TP事件Reward、SL失败、风险属性 | L1结案后独立计算；不加入额外MAE扣分 | 回归/排序/质量表征教学、事后案例分析 | 90分≠Oracle ADD标签；不能单凭高分修改策略动作 |
| L3 相对行动价值 | `ADD vs WAIT`；可选`Q_ADD-Q_WAIT` | 旧 Strict Oracle 在旧认证域，或后续**同账户**因果参考WAIT策略双轨反事实 | 严格标签（经证明）、机会成本排序（注明参考策略） | 未认证值伪装成“精确Q”；按历史cursor粗联结 |

**接口要求**：四层各自有独立的 `valid/reason/protocol/source`。下游训练不能用单一 `teacher_valid` 混淆“合法、真实后果、质量、最优动作”。

## 2. TP5质量基础分与事件Reward（本次已实现纯计算）

给定：`t` = 从真实入场执行bar开始的**有效市场分钟**（不能用自然分钟或零时长元数据）；`d` = 退出相对入场跨越了多少**官方交易日**（同官方交易日 d=0；夜盘服从数据库 `trading_date`）。

```
B(t,d) = max(20,45-5*(d-1))                   , d >= 1
         100                                   , d=0,t<=10
         100-10*(t-10)/64                      , d=0,10<t<=74
         90-5*(t-74)/46                        , d=0,74<t<=120
         85-10*(t-120)/120                     , d=0,120<t<=240
         60+15*exp(-(t-240)/120)               , d=0,t>240
R_TP_new = B/100
```

边界值：4min=1、74min=.90、120min=.85、240min=.75、480min≈.6203（仅数学同日假设）、下一交易日=.45、再下一日=.40、最低=.20。配置参数**不新增**。

- TP5完整成交 => L2 score = `B`、新TP事件Reward=`B/100`。
- SL30完整成交 => L2失败score=0，**原SL惩罚保持原样**。不再额外扣一次MAE。
- OTHER退出/拒单/未退出/右删失 => 不制造0分，不制造WAIT负例；按事件事实和单独净PnL保留，`quality_score=null`。
- MAE、早期10点逆行、净PnL、持仓占用 **单列**；原C1的早期逆行/SL/OTHER Reward项不变。
- 本阶段的`new_reward_undiscounted = actual_C1_un-discounted - old_TP_event + new_TP_event`，只是对**已观察单笔路径**的离线替换。不存在已实现的新正式 Env Reward / 新 Oracle / 新训练来源身份；C1历史不能被重签成新Reward。
- γ=332/333的时间折扣与机会价值仍属于另一层目标。本阶段**没有**用未折扣新Reward取代动态规划中的折扣回报。

## 3. 离线后果教师：工作流（Stage 1 本次补丁）

```
现有真实Env当前观察/账户/执行动作（只读，TRAIN或DEV）
         │ 保存完整 export_state/hash + 当前真实mask
         ├────(可选) 严格旧Oracle.query() ──> strict_valid/action/reason
         └──── copy+import_state 到独立原Env
                   │ 执行 ADD 一次（再由原Env确认成交/拒单）
                   │ 持仓期间 WAIT；跨chunk按原Env恢复
                   │ 原生TP、SL、非标准退出、DATA_END/截断
                   └──── 合法成交事件 / 市场分钟 / official trading_date
                                    │
                                B(t,d), 质量有效性, MAE, 原Reward/新TP替换
                                    │
                                只读 post-hoc record
```

不经过 `CandidateIndex.empty_at()` 重建“初始30万”账户，不经过 `funds_certificate()`；直接使用**当前实际账户**持有的资金和保证金。原 Env 对真实开仓失败、追保和MARGIN_CALL的处理自然保留。只有真实开仓并完整TP/SL，才打定量质量标签。

本次 `quality_record(env,fold='TRAIN'|'DEV',strict_query=None,max_market_steps=None)` 为独立调用接口：
- 必须 ACTIVE、FLAT、真实mask=[1,1]、C1、原旧Encoder配置；对尚未验证的E mutable observation cache显式拒绝。
- 先收集当前账号快照；Fork 不写入原账户、不调用有潜在磁盘写入的 `Env.close()`。运行完成检查原账户哈希一致。
- 拒单/其他退出/未结案/不足观察期分别保留失败或右删失事实；新 `quality_valid` 不会被无故置真。
- 可选严格Oracle结果与事实质量并排输出。严格Oracle不认证时，仍可以形成 `quality_valid=true` 的真实已结案交易质量记录。
- `training_action_label_allowed=false`：**第一阶段只做事实教师、模型外分析，不允许下游偷用质量分数直接监督ADD动作。**
- 因新阶段尚未接入 TRAIN 身份认证与原Replay Sidecar join，第一阶段不能宣称已生成可用于正式训练的数据集。

## 4. Stage 2：质量Catalog与真实TRAIN来源准入（尚未实现）

生成独立 `FG_S3_QUALITY_TEACHER_V1` catalog，必须含：

1. `fold=TRAIN`、contract、direction、cursor、source Env/Market SHA、current account snapshot hash、原始executed_action/stepid（有则存，无则null）、score protocol、算法Git SHA。
2. `actual_legal_{add,wait}`；`outcome_valid` 与 reason；`quality_valid`、`quality_score`、`risk`、`tp_event_reward_new`；`strict_valid`/`strict_action`/证书来源；右删失、完整原Env settlement/hash。
3. 历史回放中的 OBS 是当时真实因果输入。未来ADD反事实及结果只写旁路 catalog，**不进入** Actor obs、RSSM carry、native Replay 物理chunk或 policy cache。
4. TRAIN Catalog 单独认证和提交，不能通过实验输出文件名猜身份；DEV用于只读评价，不准作为TRAIN标签、调参目标；封存TEST不读。
5. 用全量有效空仓决策作分母统计：真实候选数、合法数、填单数、TP/SL/OTHER/右删失、质量可评分数、严格可认证数、亏损后覆盖比例。拒单不自动作为负监督。
6. 批量生成要避免每个cursor都完整重播至结尾的二次复杂度；可复用按合约价格索引和批量fork快照，但须在原Env的独立抽样中逐笔对账且仍支持低资金状态。

## 5. Stage 3：Quality Student的职责与Loss（设计，尚未接线）

**不能简单把`B/100`当成`P(ADD)`或二分类Oracle标签。** 单笔交易很好，也可能不如WAIT。正确顺序：

- 分支 A：原因果Encoder/RSSM表示 `h_t`，只使用当前及历史真实观测和上一已执行动作；E由独立经审计的encoder轴提供。
- 分支 B：辅助Outcome/QualityHead，在合法空仓状态学 `P(TP/SL/OTHER/CENSORED | ADD,h_t)`、条件TP质量、风险和不确定性。监督只来自已认证L1/L2数据；将未知/右删失单列或使用生存分析，不当作坏入场。
- 分支 C：Actor继续 WAIT/ADD 两动作。**动作教师**优先是经认证L3 strict Oracle；也允许以后使用同账户、固定因果 WAIT baseline 的相对价值差（必须显式标作 reference-policy-relative、不是全局最优）。
- 质量回归不会直接驱动 Actor 开仓：可通过对合法 P/N 或相对价值排序的**额外约束**研究Actor；在固定来源下单变量消融。先验证质量预测的NLL/Brier/校准/风险分层，之后才开启可控的决策影响。
- 旧Tutor Natural/Balanced/Pair/E2是单独轴，保留原行为对照。先小预算TRAIN、固定源、多个seed，在TRAIN留出合约和独立DEV评价，确认是否改善真正 ADD recall 且不扩大SL。
- 严格Oracle资金认证失败的样本，**仍可训练结果质量辅助头**；但不能作为严格Oracle WAIT/ADD教师正负例。

## 6. Stage 4：新TP Reward正式化 + 新Oracle（设计，尚未实现）

- 新独立身份，例如 `FG_SINGLE3_REWARD_QUALITY_TP_V1`，将 `B/100` 替换完整TP事件；保留早期逆行/SL/OTHER、原Env执行、单仓规则。原C1、所有原Checkpoint、原Oracle收据继续可精确恢复。
- 新Reward改变优化目标，必须重新生成 TRAIN CandidateIndex、TeacherCatalog/Oracle；原 `C1_H333.json` / `first[t]` / `value_intervals` / 签名全部不复用为新目标的认证。
- 新 Oracle 使用新奖励重新计算 `ADD vs WAIT`；再研究原资金证明怎样在新 Reward 中成立。若跳变的跨官方交易日奖励导致某些非TP区间性质发生变化，必须重新核对 dominance 与非正收益的前提。
- 不在同一DEV多次挑最优评分参数；锚点由用户已冻结，不因报告盈利多少而临时改曲线。

## 7. 对照与门槛

| 实验 | 范围 | 应先看到什么 | 否则怎么办 |
|---|---|---|---|
| P0公式/来源校验 | 单元/静态 | 锚点、边界、日期口径、C1原Reward不变 | NO-GO |
| P1实际账户Fork | TRAIN真实状态（含亏损后） | 和原Env ADD→WAIT闭环一致、不可写原State | NO-GO |
| P2跨日/异常退出/追保 | TRAIN真实与可控夹具 | TP/SL/OTHER/CENSORED正确分流；资金低不影响事实标签 | NO-GO |
| P3首批质量Catalog | 新的隔离路径 | 身份、批次、风险、右删失、漏标统计、无未来输入泄漏 | 冻结不训练 |
| P4学生只学质量头 | 小预算，多seed | 结果预测优于训练先验及可重现的BASELINE | 不接Actor |
| P5新增相对价值教学 | 与旧Tutor同源消融 | Teacher ADD recall与错误ADD率均改善；真实交易闭环有进步 | 不扩大1M |
| P6跨合约独立验证 | 至少多个时间fold与合约 | 摆脱FG409/单种子偶然结论 | 不解封TEST |

## 8. 本次补丁与外部依赖限制

`FG_S3_Quality_Teacher_V1_Gym.patch`：只新增两个research模块和一组测试，不改动既有任何生产入口/Reward/Oracle/Agent。支持用真实 **旧 C1 Env** 对单个当前账户状态评分；DEV只可离线诊断。若项目上有 `state_path`，禁止在fork上调用 `close()` 造成文件写入。E-profile fork未认证，本阶段拒绝。

本评审环境未安装 `gymnasium`，项目完整测试在pytest collection阶段失败，**没有**实际运行 FG Env；纯TP公式测试和补丁应用通过。此设计不声称 Stage2–4完成。
